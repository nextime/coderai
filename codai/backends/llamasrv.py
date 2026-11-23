# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""OpenAI proxy to a managed ``llama-server`` -- llama.cpp's own HTTP server.

The wire protocol is the same OpenAI shape :class:`VllmBackend` already speaks,
including the parts that are easy to lose: structured ``tool_calls``, the
separated ``reasoning_content``, per-token logprobs, the remote's own
``finish_reason``, and a mid-stream error arriving as an SSE event under HTTP
200. Each of those cost a bug to get right once, so this subclasses that proxy
instead of growing a second copy of it, and overrides only what actually
differs: the lifecycle (which worker builds and launches the server), and the
bearer the server requires.

What this backend is *for* is in :class:`codai.config.LlamaSrvConfig` -- briefly,
the in-process llama-cpp-python path cannot reach llama.cpp's speculative
decoding, because the draft-and-verify loop lives in ``common/``, not in
``libllama``. It is an alternative to the bindings, never a replacement: a GGUF
reaches it only by pinning ``backend: "llamasrv"``.
"""

from typing import Dict, Optional

from codai.backends.vllm import VllmBackend


class LlamaSrvBackend(VllmBackend):
    """Thin OpenAI proxy to a managed ``llama-server``."""

    #: llama.cpp serves vision models (mtmd) through the same chat route, so
    #: multipart content goes to the server as it arrived. A proxy that
    #: flattened it would silently drop the image.
    supports_vision = True

    def __init__(self, cfg=None):
        if cfg is None:
            from codai.config import LlamaSrvConfig
            cfg = LlamaSrvConfig()
        super().__init__(cfg)
        self._model_id = getattr(cfg, "model_id", "") or "llamasrv"
        self._bearer: str = ""

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def load_model(self, model_name: str, **kwargs) -> None:
        from codai.api import llamasrv_worker
        from codai.backends.overrides import apply, model_block

        if model_name:
            self._model_id = model_name

        _ctx = kwargs.get("n_ctx", kwargs.get("ctx"))
        if isinstance(_ctx, (list, tuple)):
            _ctx = _ctx[0] if _ctx else None
        try:
            _ctx = int(_ctx) if _ctx else 0
        except (TypeError, ValueError):
            _ctx = 0

        # This model's own `llamasrv` block over the global template, the same way
        # a model's `vllm` block refines the vLLM template: the tree to build, the
        # slot count, the KV cache types, the drafting flags.
        # Under the ENGINE's own key: a model pinned to the fork refines it with
        # a `llamampere` block. One shared key would mean a per-model override
        # written for the fork also rewrote the stock engine's launch.
        # A block written under the engine's FORMER key is still that model's
        # override. Dropping it would not fail: the launch would fall back to
        # the engine-wide template and serve the model with another model's
        # context size and cache types, which looks like a working server.
        _key = llamasrv_worker.engine_key(self._cfg)
        blk = model_block(model_name, _key) or {}
        for _old in getattr(self._cfg, "LEGACY_ENGINE_KEYS", ()) or ():
            if blk:
                break
            blk = model_block(model_name, _old) or {}
            if blk:
                print(f"[{_key}] {model_name!r}: per-model overrides read from "
                      f"its former {_old!r} block", flush=True)
        if blk:
            self._cfg = apply(self._cfg, blk, _LLAMASRV_FIELDS, _key)

        # The window the model is SERVED with. A per-model n_ctx has to reach the
        # launch, or the number reported to clients and the one the server will
        # enforce disagree by design -- and the client plans against the wrong one.
        blk_ctx = 0
        try:
            blk_ctx = int((blk or {}).get("ctx") or 0)
        except (TypeError, ValueError):
            blk_ctx = 0
        if not blk_ctx and _ctx > 0:
            try:
                self._cfg = apply(self._cfg, {"ctx": _ctx}, _LLAMASRV_FIELDS, _key)
            except Exception:
                pass
        try:
            self._ctx = int(getattr(self._cfg, "ctx", 0) or 0) or self._ctx
        except (TypeError, ValueError):
            pass

        model_path = self._resolve_gguf(model_name)
        self._served_name = (model_name
                             or getattr(self._cfg, "model_id", "") or "llamasrv")
        _resolved, self._svc_key = llamasrv_worker.resolve_service_key(
            self._cfg, model_path)
        self._url, self._bearer = llamasrv_worker.ensure_service(
            self._cfg, model_file=model_path, ctx=(blk_ctx or _ctx or None),
            model_name=model_name, mmproj=self._resolve_mmproj(model_name))

    def _resolve_gguf(self, model_name: str) -> Optional[str]:
        """The local ``.gguf`` for this model, from the model list.

        A model pinned to this engine never ALSO runs the in-process backend, so
        the old "no download, it would race the in-process downloader" worry does
        not apply to it: this engine is its only loader. That matters on a rented
        pod, where the weights live on HuggingFace, not on local disk — without a
        fetch here the fork would answer "none resolved" for every HF/URL model
        and never serve. So: find a local file first, and only if none resolves,
        fetch the URL/HF source through the in-process backend's OWN cache (not a
        second cache) and hand the fork the resolved local file.
        """
        import os

        def _usable(p: str) -> Optional[str]:
            p = os.path.expanduser((p or "").strip())
            if not p:
                return None
            if p.lower().endswith(".gguf") and os.path.isfile(p):
                return os.path.abspath(p)
            # A directory of shards: llama-server takes the first and finds the rest.
            if os.path.isdir(p):
                shards = sorted(f for f in os.listdir(p)
                                if f.lower().endswith(".gguf"))
                if shards:
                    return os.path.join(os.path.abspath(p), shards[0])
            return None

        direct = _usable(model_name)
        if direct:
            return direct
        entry = self._entry_for(model_name)
        cand = model_name
        if entry:
            path = str(entry.get("path") or "").strip()
            if path:
                u = _usable(path)
                if u:
                    return u
                cand = path
        # Nothing local yet. If the model names a URL or an HF repo, fetch it
        # through the in-process backend's resolver, which downloads once into the
        # shared GGUF cache and returns the local path — exactly what a rented
        # fork pod needs, since the fork server takes a file, not a repo.
        if cand and (cand.startswith(("http://", "https://"))
                     or (not os.path.isabs(cand) and "/" in cand and " " not in cand)):
            try:
                from codai.models.cache import load_model as _fetch_gguf
                fetched = _fetch_gguf(cand)
                u = _usable(fetched)
                if u:
                    return u
            except Exception as exc:
                print(f"[llamampere] could not fetch weights for {cand!r}: {exc}",
                      flush=True)
        return _usable(cand) or (cand if entry else None)

    def _entry_for(self, model_name: str) -> Optional[dict]:
        """This model's models.json entry, matched by path/basename/alias/id.

        Exactly the match the vLLM proxy does. Separated from the path lookup
        because more than the path comes from the entry: the vision projector
        does too, and deriving them from one matched entry is the only way they
        cannot disagree about which model they describe.
        """
        import os
        try:
            from codai.admin.routes import config_manager
            md = getattr(config_manager, "models_data", {}) or {}
            name_l = (model_name or "").strip().lower()
            for lst in md.values():
                if not isinstance(lst, list):
                    continue
                for m in lst:
                    if not isinstance(m, dict):
                        continue
                    path = str(m.get("path") or "").strip()
                    if not path:
                        continue
                    names = {path.lower(),
                             os.path.basename(path.rstrip("/")).lower(),
                             str(m.get("alias") or "").lower(),
                             str(m.get("id") or "").lower()}
                    if name_l in names:
                        return m
        except Exception:
            pass
        return None

    def _resolve_mmproj(self, model_name: str) -> Optional[str]:
        """The vision projector for this model, if it has one that exists.

        A multimodal GGUF whose projector is never loaded is a text model that
        accepts an image and describes nothing -- it answers, plausibly, about
        nothing it was shown. So this is read from the same entry the in-process
        path reads, and a model keeps its sight when it is pinned here.

        A declared-but-missing file is skipped with a line in the log rather than
        passed through: llama-server refuses to start without it, which would
        turn 'no vision' into 'no model'.
        """
        import os
        entry = self._entry_for(model_name)
        if not entry:
            return None
        mm = str(entry.get("mmproj") or "").strip()
        if not mm:
            return None
        mm = os.path.expanduser(mm)
        if os.path.isfile(mm):
            return os.path.abspath(mm)
        print(f"[llamasrv] {model_name}: mmproj {mm!r} is declared but not on disk; "
              "serving without vision rather than refusing to start", flush=True)
        return None

    def cleanup(self) -> None:
        """Stop this model's server. THIS is how llamasrv frees VRAM.

        The manager evicts a llamasrv-served model the same way it evicts any
        other -- it drops the backend and calls cleanup() -- so everything that
        makes eviction work for an in-process model has to work here, and the
        key has to be the one the service was registered under. The fallback is
        engine-prefixed for that reason: an unprefixed key matches no service,
        stop_service does nothing, and the manager goes on believing it freed
        the whole model's VRAM while the server still holds it.
        """
        from codai.api import llamasrv_worker
        key = getattr(self, "_svc_key", None)
        if not key:
            base = getattr(self._cfg, "model_id", "") or self._model_id
            key = f"{llamasrv_worker.engine_key(self._cfg)}::{base}"
        llamasrv_worker.stop_service(key)
        self._url = None
        self._bearer = ""

    # ------------------------------------------------------------------ #
    # the two things that differ from the vLLM proxy
    # ------------------------------------------------------------------ #
    def _base(self) -> str:
        if not self._url:
            raise RuntimeError("llama-server not started")
        return self._url

    def _headers(self) -> Dict[str, str]:
        """``llama-server`` is launched with ``--api-key``, so every call carries it.

        The worker mints the bearer per launch when none is configured. See
        VllmBackend._headers for why this is a hook rather than a constant.
        """
        return {"Authorization": f"Bearer {self._bearer}"} if self._bearer else {}

    def context_length(self):
        """The window the SERVER reports, from ``/props``, or None.

        ``/v1/models`` on llama.cpp does not carry the context length the way
        vLLM's does, so the parent's probe would always answer None and the
        reported window would silently fall back to config -- the exact
        disagreement that probe exists to prevent. ``/props`` carries it as
        ``default_generation_settings.n_ctx``.
        """
        if getattr(self, "_ctx_len", "unset") != "unset":
            return self._ctx_len
        self._ctx_len = None
        try:
            import requests
            r = requests.get(self._base() + "/props", headers=self._headers(),
                             timeout=5)
            if r.ok:
                body = r.json() or {}
                gen = body.get("default_generation_settings") or {}
                for src in (gen, body):
                    for k in ("n_ctx", "n_ctx_per_seq", "context_length"):
                        if src.get(k):
                            self._ctx_len = int(src[k])
                            break
                    if self._ctx_len:
                        break
        except Exception:
            pass
        return self._ctx_len


#: Which ``LlamaSrvConfig`` fields a per-model ``llamasrv`` block may override.
#: Explicitly NOT here: install_dir, auto_build, service_url and api_key, which
#: are install-level, and require_matching_arch -- a per-model block must not be
#: able to switch off the architecture gate for everyone sharing the build.
_LLAMASRV_FIELDS = (
    "tree", "repo_url", "git_ref", "cuda_architectures", "extra_cmake_args",
    "ctx", "n_gpu_layers", "parallel", "flash_attn", "cache_type_k",
    "cache_type_v", "spec_type", "spec_draft_n_max", "extra_args", "extra_env",
    "ready_timeout_s",
)
