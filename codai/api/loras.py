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

"""
LoRA training endpoints.

Train a small per-character (or per-style) LoRA adapter from a handful of
reference images, then apply it to BOTH image and video diffusion pipelines for
consistent identity across models.

POST   /v1/loras/train      – train a LoRA from a saved character profile or images
GET    /v1/loras            – list trained LoRAs
GET    /v1/loras/progress   – training progress (for the active job)
GET    /v1/loras/{name}     – info about one trained LoRA
DELETE /v1/loras/{name}     – delete a trained LoRA

All model execution stays server-side.  Training runs in-process so it can share
the model manager's VRAM (it evicts resident models first) and honour the global
thermal-protection checkpoints.
"""

import base64
import io
import json
import os
import secrets
import dataclasses
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from codai.platform_paths import default_loras_dir
from codai.queue.manager import queue_manager
from codai.cluster import ddp as _ddp
from codai.tasks import task_registry, TaskCancelled

router = APIRouter()

_LORAS_DIR: Optional[str] = None

# Single-job training progress (training is VRAM-heavy; we run one at a time).
_progress_lock = threading.Lock()
_progress = {
    "active": False,
    "name": None,
    "step": 0,
    "total": 0,
    "status": "idle",      # idle | preparing | training | saving | done | error
    "message": "",
    "started_at": 0.0,
    "path": None,
}
_train_lock = threading.Lock()

# ── Multi-client job registry ─────────────────────────────────────────────────
# Training is serialized server-wide (one GPU job at a time), but several clients
# may submit concurrently. Each submission gets a unique job_id and is recorded
# here so a client only ever polls *its own* job (no cross-user spillover). The
# registry is mirrored to disk so a recovering client (after a server OR client
# restart) can re-attach by job_id / session token. `_active_job_id` names the
# job currently executing, so _set_progress can mirror live progress into it.
_jobs_lock = threading.Lock()
_jobs: dict = {}                  # job_id -> record
_active_job_id: Optional[str] = None
# Base model of the job currently executing on the GPU. The trainer loads its base
# pipeline OUTSIDE the model manager (and unloads all manager models first), so the
# engine's loaded_models would otherwise read 0 mid-training even though the GPU is
# busy. Surfaced via active_training_model() so the engine status reflects reality.
_active_train_model: Optional[str] = None
_bg_tasks: set = set()            # strong refs to detached train tasks
# Job ids with a pending cancel. Survives the window before a queued job's task
# is registered (the worker checks this set right after acquiring the GPU lock).
_cancel_requested: set = set()
# Job ids that must resume from their checkpoint even when global recovery is off
# (an explicit manual Restart from the Tasks page).
_force_resume_jobs: set = set()
_JOB_ACTIVE_STATES = ("queued", "preparing", "training", "saving")
# Cap on retained *terminal* (done/cancelled/error/interrupted) job records so the
# persisted registry can't grow without bound. Active jobs are always kept; the
# oldest terminal records beyond this many are pruned (with their sidecar request
# files). A removed job is deleted outright (remove_job); this just bounds the
# slow accumulation of finished ones nobody removed by hand.
_JOB_HISTORY_MAX = 200
_MIRROR_FIELDS = ("active", "name", "step", "total", "status", "message",
                  "started_at", "path")
_last_job_persist = 0.0


def _jobs_file() -> str:
    return os.path.join(_loras_dir(), "_train_jobs.json")


def _prune_jobs_locked() -> None:
    """Drop the oldest terminal job records beyond _JOB_HISTORY_MAX. Caller holds
    _jobs_lock. Also removes each pruned job's persisted request sidecar."""
    terminal = [(jid, r) for jid, r in _jobs.items()
                if r.get("status") not in _JOB_ACTIVE_STATES]
    excess = len(terminal) - _JOB_HISTORY_MAX
    if excess <= 0:
        return
    terminal.sort(key=lambda kv: kv[1].get("updated_at") or kv[1].get("started_at") or 0)
    for jid, _ in terminal[:excess]:
        _jobs.pop(jid, None)
        try:
            p = _job_req_path(jid)
            if os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass


def _persist_jobs_locked(force: bool = False) -> None:
    """Write the registry to disk. Throttled to ~5s unless forced (status change)
    so per-step progress updates don't hammer the disk."""
    global _last_job_persist
    now = time.time()
    if not force and (now - _last_job_persist) < 5.0:
        return
    _last_job_persist = now
    _prune_jobs_locked()
    try:
        p = _jobs_file()
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_jobs, f)
        os.replace(tmp, p)
    except Exception:
        pass


def _new_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        rec = {"job_id": job_id, "updated_at": time.time()}
        rec.update(fields)
        _jobs[job_id] = rec
        _persist_jobs_locked(force=True)


def _update_job(job_id: Optional[str], force: bool = False, **fields) -> None:
    if not job_id:
        return
    with _jobs_lock:
        rec = _jobs.get(job_id)
        if rec is None:
            rec = {"job_id": job_id}
            _jobs[job_id] = rec
        # A status change is significant — persist immediately.
        if "status" in fields and fields["status"] != rec.get("status"):
            force = True
        rec.update(fields)
        rec["updated_at"] = time.time()
        _persist_jobs_locked(force=force)


# When False (set via --no-resume-jobs or config.jobs.resume_on_restart=false),
# interrupted training is NOT recovered on restart: mid-flight jobs are marked
# 'cancelled' (not 'interrupted') and the per-job checkpoint auto-resume is
# disabled. Checkpoints are kept on disk, so a job can still be restarted
# manually (Tasks page) — that path passes resume=True explicitly.
_RESUME_ENABLED: bool = True


def set_resume_enabled(enabled: bool) -> None:
    """Enable/disable interrupted-training recovery on restart. Call before
    set_global_args() (which runs _load_jobs_on_start)."""
    global _RESUME_ENABLED
    _RESUME_ENABLED = bool(enabled)


def _load_jobs_on_start() -> None:
    """Load the persisted registry and reconcile jobs that were mid-flight when
    the process died (their in-memory GPU training is gone).

    With recovery enabled they become 'interrupted' — a polling client resubmits
    and the server resumes from the last on-disk checkpoint. With recovery
    disabled (--no-resume-jobs / config) they become 'cancelled' and are not
    auto-resumed; checkpoints are kept so they can be restarted manually."""
    global _jobs
    try:
        with open(_jobs_file()) as f:
            data = json.load(f)
    except Exception:
        return
    if not isinstance(data, dict):
        return
    changed = False
    for rec in data.values():
        if rec.get("status") in _JOB_ACTIVE_STATES:
            if _RESUME_ENABLED:
                rec["status"] = "interrupted"
                rec["message"] = "interrupted by server restart — resubmit to resume"
            else:
                rec["status"] = "cancelled"
                rec["message"] = "cancelled — recovery disabled on restart (checkpoint kept)"
            rec["active"] = False
            changed = True
    with _jobs_lock:
        _jobs = data
        if changed:
            _persist_jobs_locked(force=True)


def set_global_args(args):
    global _LORAS_DIR
    base = getattr(args, 'file_path', None)
    if base and os.path.isdir(base):
        root = base
    elif base:
        root = os.path.dirname(base)
    else:
        root = None
    _LORAS_DIR = os.path.join(root, 'loras') if root else str(default_loras_dir())
    os.makedirs(_LORAS_DIR, exist_ok=True)
    _load_jobs_on_start()


def _loras_dir() -> str:
    if _LORAS_DIR:
        os.makedirs(_LORAS_DIR, exist_ok=True)
        return _LORAS_DIR
    d = str(default_loras_dir())
    os.makedirs(d, exist_ok=True)
    return d


def _lora_dir(name: str) -> str:
    # A peer rank of a multi-node training must not leave a LoRA behind: what
    # the trainers save goes to scratch that the ddp context removes.
    _d = _ddp.current()
    if _d is not None and not _d.is_main:
        return os.path.join(_d.scratch_dir(), name)
    return os.path.join(_loras_dir(), name)


def _lora_weight_file(name: str) -> Optional[str]:
    """Return the path to the trained weights file for a LoRA, if present."""
    d = _lora_dir(name)
    for fn in ("pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin"):
        p = os.path.join(d, fn)
        if os.path.isfile(p):
            return p
    return None


# ── Content-addressed LoRA blob store ─────────────────────────────────────────
# Lets clients on a *different* machine supply a LoRA without sharing a
# filesystem: upload the file once (POST /v1/loras/upload), then reference it by
# its sha256 in generation requests. Identical files dedupe automatically, so a
# match's env/fighter LoRAs are transmitted at most once.

def _lora_blob_dir() -> str:
    d = os.path.join(_loras_dir(), "_blobs")
    os.makedirs(d, exist_ok=True)
    return d


def _lora_blob_hex(h: str) -> str:
    """The bare lowercase hex of a blob id (accepts an optional 'sha256:')."""
    h = (h or "").strip()
    if h.lower().startswith("sha256:"):
        h = h[7:]
    return h.lower()


def _lora_blob_path(h: str) -> str:
    """Local path for a blob given its hex sha256 (accepts an optional
    'sha256:' prefix). Returns the path whether or not the file exists yet.

    A blob is normally one file. A text LoRA for vLLM is a DIRECTORY, so it is
    stored unpacked under the same hash; when that directory is present it is
    what the caller wants, because that is the path vLLM can load.
    """
    h = _lora_blob_hex(h)
    unpacked = os.path.join(_lora_blob_dir(), h)
    if os.path.isdir(unpacked):
        return unpacked
    return os.path.join(_lora_blob_dir(), f"{h}.safetensors")


def lora_blob_exists(h: str) -> bool:
    p = _lora_blob_path(h)
    return os.path.isfile(p) or os.path.isdir(p)


def _unpack_lora_bundle(data: bytes, dest: str) -> None:
    """Extract an adapter bundle into `dest`, refusing anything that escapes it.

    The bytes arrive over the network, so a member named ``../../etc/something``
    would write outside the blob store. Names are checked here AND the extraction
    uses tar's data filter, because either alone has been bypassed before.
    """
    import io
    import tarfile
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
        for info in tar.getmembers():
            if not (info.isfile() or info.isdir()):
                raise HTTPException(status_code=400,
                                    detail=f"adapter bundle member {info.name!r} "
                                           "is not a regular file")
            target = os.path.realpath(os.path.join(dest, info.name))
            if target != os.path.realpath(dest) and not target.startswith(
                    os.path.realpath(dest) + os.sep):
                raise HTTPException(status_code=400,
                                    detail=f"adapter bundle member {info.name!r} "
                                           "escapes the blob store")
        os.makedirs(dest, exist_ok=True)
        try:
            tar.extractall(dest, filter="data")
        except TypeError:          # filter= is newer than some runtimes
            tar.extractall(dest)


def save_lora_blob(data: bytes) -> tuple:
    """Write `data` to the content-addressed store. Returns (hex_sha256, existed).

    Tar bytes are a bundled adapter DIRECTORY (a PEFT adapter: config plus
    weights) and are unpacked, because vLLM loads a text LoRA from a directory
    and cannot do anything with the tar. Detected from the tar magic rather than
    from a flag, so the sender does not have to declare the shape.
    """
    import hashlib
    import shutil
    from codai.models.text_loras import blob_id, is_bundle
    h = hashlib.sha256(data).hexdigest()
    if is_bundle(data):
        # The id must name the CONTENT, not the container it arrived in. A tar
        # of the same adapter made by GNU tar, by this bundler, or by anything
        # else hashes differently -- mtimes, ownership, member order -- so an
        # id taken from the uploaded bytes is one that nobody else can
        # reproduce. Measured live: a client uploaded a tar hashing to
        # b8eac366…, the catalogue recorded that, every subsequent push
        # re-bundled the unpacked directory and stored 2e820c…, and the pod
        # looked up b8eac366… and got a 404. It HAD the bytes, under another
        # name, and so served the base model under the adapter's name.
        #
        # So an uploaded bundle is unpacked first and stored under the id of
        # THIS bundler's packing of it, which is what every side computes.
        tmp = os.path.join(_lora_blob_dir(), f"incoming-{h}.tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        _unpack_lora_bundle(data, tmp)
        canonical = (blob_id(tmp) or f"sha256:{h}").split(":", 1)[-1]
        dest = os.path.join(_lora_blob_dir(), canonical)
        if os.path.isdir(dest):
            shutil.rmtree(tmp, ignore_errors=True)
            return canonical, True
        os.replace(tmp, dest)
        if canonical != h:
            print(f"[lora] stored an uploaded bundle as {canonical[:12]}… "
                  f"(its own bytes hashed to {h[:12]}…): the id names the "
                  f"content, so any sender computes the same one", flush=True)
        return canonical, False
    p = os.path.join(_lora_blob_dir(), f"{h}.safetensors")
    if os.path.isfile(p):
        return h, True
    tmp = p + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, p)
    return h, False


def _spec_get(spec, key):
    """Read a field from a LoRA spec that may be a dict OR a pydantic object."""
    if isinstance(spec, dict):
        return spec.get(key)
    return getattr(spec, key, None)


def _strip_data_uri(s: str) -> str:
    """Drop a leading 'data:...;base64,' prefix if present."""
    if isinstance(s, str) and s.startswith("data:"):
        comma = s.find(",")
        if comma != -1:
            return s[comma + 1:]
    return s


def resolve_lora_ref(spec) -> Optional[str]:
    """Resolve a LoRA request spec to a concrete local path (or HF id) usable by
    diffusers' load_lora_weights().

    Accepts, in priority order, on a dict or LoraConfig/VideoLoraConfig object:
      * id   — "name:<registered>" (a trained LoRA on this server),
               "sha256:<hex>" or a bare 64-char hex (an uploaded blob)
      * file / data — base64 (optionally a data: URI) of the .safetensors bytes;
               stored in the blob cache and used from there
      * url  — http(s) URL the server downloads (and caches by content hash)
      * model / path — a local path or HuggingFace repo id (legacy behaviour)

    Raises HTTPException(400) when a ref is given but cannot be resolved.
    """
    if spec is None:
        return None

    # 1. id reference
    ref_id = _spec_get(spec, "id")
    if ref_id:
        ref_id = str(ref_id).strip()
        if ref_id.startswith("name:"):
            name = ref_id[5:].strip()
            wf = _lora_weight_file(name) or (
                _lora_dir(name) if os.path.isdir(_lora_dir(name)) else None)
            if not wf:
                raise HTTPException(status_code=400,
                                    detail=f"LoRA '{name}' not found on server")
            return wf
        # sha256:<hex> or bare hex
        h = ref_id[7:] if ref_id.lower().startswith("sha256:") else ref_id
        if lora_blob_exists(h):
            return _lora_blob_path(h)
        raise HTTPException(
            status_code=400,
            detail=f"LoRA blob '{ref_id}' not on server — upload it via "
                   f"POST /v1/loras/upload first")

    # 2. inline base64 file
    blob = _spec_get(spec, "file") or _spec_get(spec, "data")
    if blob:
        import base64
        try:
            raw = base64.b64decode(_strip_data_uri(str(blob)))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid base64 LoRA file: {e}")
        h, _ = save_lora_blob(raw)
        return _lora_blob_path(h)

    # 3. url
    url = _spec_get(spec, "url")
    if url:
        import urllib.request
        try:
            with urllib.request.urlopen(str(url)) as resp:
                raw = resp.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"could not download LoRA from {url}: {e}")
        h, _ = save_lora_blob(raw)
        return _lora_blob_path(h)

    # 4. legacy path / HF id
    return _spec_get(spec, "model") or _spec_get(spec, "path")


def resolve_request_loras(loras) -> None:
    """Resolve every LoRA spec in a request's `loras` list to a concrete local
    path IN PLACE, writing it back onto each spec's `model` field. Call this in
    the async request handler (before the heavy generation work) so a missing
    blob / unknown name surfaces as a clean HTTP 400. After this, downstream code
    that reads `lora.model` (signature dedup, VRAM estimate, load_lora_weights)
    works unchanged regardless of how the client supplied the weights."""
    if not loras:
        return
    for lora in loras:
        path = resolve_lora_ref(lora)
        if not path:
            continue
        if isinstance(lora, dict):
            lora["model"] = path
        else:
            try:
                lora.model = path
            except Exception:
                pass


def _require_api_auth(request: Request) -> None:
    """Raise 401 if auth is enabled and the request carries no valid credential."""
    try:
        from codai.admin import routes as _admin_routes
        sm = _admin_routes.session_manager
    except Exception:
        return
    if sm is None:
        return
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        if sm.verify_token(token):
            return
    cookie = request.cookies.get("session", "")
    if cookie.endswith(".MUST_CHANGE"):
        cookie = cookie[:-12]
    if cookie and sm.validate_session(cookie):
        return
    raise HTTPException(
        status_code=401,
        detail={"message": "Invalid API key. Provide a valid Bearer token.",
                "type": "invalid_request_error", "code": "invalid_api_key"},
    )


# ── Pydantic models ───────────────────────────────────────────────────────────

class LoraTrainRequest(BaseModel):
    name: str                              # output LoRA name (folder)
    base_model: str                        # image model key (models.json) or HF id / path
    # Optional separate UNet-based SD1.x/SDXL model to train the LoRA against,
    # when `base_model` (the generation model) is a transformer/DiT (Z-Image,
    # Flux, SD3) this trainer can't target.  Falls back to base_model if unset.
    train_base_model: Optional[str] = None
    # Target pipeline for the LoRA: "image" (default, SD1.x/SDXL UNet) or "video"
    # (Wan video DiT). For "video", `base_model` is the VIDEO model id/path and the
    # LoRA is trained against that exact model so it loads on the video pipeline.
    # image | video | wan | ltx2 | h3 | krea. "video" auto-detects the architecture
    # from the model id (Wan unless it looks like LTX-2 / H3); the explicit names
    # force one. See codai/api/lora_archs.py.
    target: Optional[str] = "image"
    # Quantize the (large) video transformer to 4-bit for training (QLoRA). Lets a
    # 14B video model's LoRA fit on a consumer GPU. Ignored for image targets.
    quantize_4bit: Optional[bool] = True
    # LongCat-Video (target "longcat") trains through SimpleTuner, which wants a
    # captioned VIDEO dataset described by its own data-backend config. coderai will not
    # synthesise one from `images`: a still-image dataset would train a different thing
    # than a video LoRA, and silently so.
    dataset_config: Optional[str] = None   # path to a SimpleTuner data-backend config
    train_resolution: Optional[str] = None  # "480x832"; sides divisible by 16
    num_frames: Optional[int] = 1          # training video length (1 = stills-only)
    character: Optional[str] = None        # saved character profile to pull images from
    environment: Optional[str] = None      # OR saved environment profile to pull images from
    images: Optional[List[str]] = None     # OR explicit base64/data-uri images
    instance_prompt: Optional[str] = None  # e.g. "a photo of sks man"; auto from name if None
    steps: Optional[int] = 800             # training steps (balanced default)
    rank: Optional[int] = 16               # LoRA rank
    learning_rate: Optional[float] = 1e-4
    resolution: Optional[int] = 512
    seed: Optional[int] = 42
    # Run the job asynchronously: the POST returns immediately with a job_id and
    # the client polls /v1/loras/progress?job=<id>. Avoids long read-timeouts on
    # multi-hour video trainings. Default True keeps blocking behaviour for old
    # clients that expect the result inline.
    wait: Optional[bool] = True
    # Caller-supplied session token. Recorded with the job so the owning client —
    # and only it — can recover its job(s) after a restart. Auto-generated if
    # omitted.
    session: Optional[str] = None
    # Several machines: cluster node names (or {url, api_key} blocks) that train
    # the same job on their own card, gradients of the adapter averaged every
    # step (codai/cluster/ddp.py). This engine is rank 0 and saves the result.
    nodes: Optional[List] = None
    dist_backend: Optional[str] = None     # gloo (default) | nccl
    # Set by the coordinator on the request it sends to each peer; never by a
    # client: {rank, world_size, master_addr, master_port, backend, timeout_s}.
    distributed: Optional[dict] = None
    model_config = ConfigDict(extra="allow")


# ── Base-model resolution ─────────────────────────────────────────────────────

def _configured_train_nodes(base_model: str):
    """Per-model `lora_train_nodes` (models.json): the cluster nodes every
    LoRA trained on this model spreads over, unless the request names its own."""
    try:
        from codai.models.manager import _model_entry_for
        entry = _model_entry_for(base_model) or {}
        v = entry.get("lora_train_nodes")
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, list) and v:
            return v
    except Exception:
        pass
    return None


def _configured_train_base(base_model: str) -> Optional[str]:
    """Read a per-model `lora_train_base_model` override from the image model's
    config (models.json entry), if any.  Lets a deployment declare once that e.g.
    a Z-Image generation model should train LoRAs against an SDXL model, so no
    client has to know.  Returns the configured model key/path or None."""
    try:
        from codai.models.manager import multi_model_manager
        for key in (f"image:{base_model}", base_model):
            cfg = multi_model_manager.config.get(key)
            if not cfg:
                continue
            for src in (cfg, cfg.get('_raw_cfg') or {}):
                v = src.get('lora_train_base_model') if isinstance(src, dict) else None
                if v and isinstance(v, str) and v.strip():
                    return v.strip()
    except Exception:
        pass
    return None


def _resolve_base_model_path(base_model: str, category: str = "image") -> str:
    """Resolve a model key (or path/HF id) to a diffusers model directory.

    `category` selects the models.json key prefix to try first ("image" or
    "video"), so a video LoRA resolves against the video model entry."""
    try:
        from codai.models.manager import multi_model_manager
        keys = ([f"{category}:{base_model}", f"image:{base_model}",
                 f"video:{base_model}", base_model])
        seen = set()
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            cfg = multi_model_manager.config.get(key)
            if cfg:
                for k in ('path', 'model_path', 'model', 'diffusers_path'):
                    v = cfg.get(k)
                    if v and isinstance(v, str):
                        return v
    except Exception:
        pass
    # Treat as a direct path or HF repo id.
    return base_model


def _decode_image(ref: str):
    from PIL import Image as PILImage
    if ref.startswith('data:'):
        ref = ref.split(',', 1)[1]
    raw = base64.b64decode(ref)
    return PILImage.open(io.BytesIO(raw)).convert('RGB')


def _gather_images(req: LoraTrainRequest):
    """Return a list of PIL images from the character profile and/or inline images."""
    imgs = []
    if req.character:
        try:
            from codai.api.characters import resolve_character_profiles
            for b64 in resolve_character_profiles([req.character]):
                try:
                    imgs.append(_decode_image(b64))
                except Exception:
                    pass
        except Exception:
            pass
    if req.environment:
        try:
            from codai.api.environments import resolve_environment_profiles
            for b64 in resolve_environment_profiles([req.environment]):
                try:
                    imgs.append(_decode_image(b64))
                except Exception:
                    pass
        except Exception:
            pass
    for ref in (req.images or []):
        try:
            imgs.append(_decode_image(ref))
        except Exception:
            pass
    return imgs


def _set_progress(**kw):
    with _progress_lock:
        _progress.update(kw)
        job_id = _active_job_id
    # Mirror live progress into the executing job's record so its owner (and only
    # its owner) can poll it by job_id.
    if job_id:
        mirror = {k: kw[k] for k in kw if k in _MIRROR_FIELDS}
        if mirror:
            _update_job(job_id, **mirror)
        # Mirror step/total into the live task registry (task id == job id).
        if "step" in kw or "total" in kw:
            task_registry.step(job_id, kw.get("step", 0), kw.get("total"))
        if "message" in kw:
            task_registry.update(job_id, message=str(kw["message"])[:200])


def _check_train_cancel() -> None:
    """Raise TaskCancelled if the running training job was cancelled, or block
    while it is paused. Called each training step; the task id is the active job
    id. The pause wait stays responsive to cancellation."""
    jid = _active_job_id
    if jid and (jid in _cancel_requested or task_registry.is_cancelled(jid)):
        raise TaskCancelled(jid)
    # Cooperative pause: block here (in the training thread) while paused.
    if jid:
        was_paused = task_registry.is_paused(jid)
        if was_paused:
            _update_job(jid, force=True, message="paused")
        task_registry.wait_if_paused(jid)
        if was_paused:
            _update_job(jid, force=True, message="training")


def _job_req_path(job_id: str) -> str:
    """Per-job sidecar file holding the full training request, so a job can be
    restarted (resumed from its checkpoint) after the original client is gone.
    Kept separate from _train_jobs.json so inline base64 images don't bloat the
    registry that's rewritten on every progress tick."""
    return os.path.join(_loras_dir(), "_jobs", f"{job_id}.json")


def _save_job_request(job_id: str, req) -> None:
    try:
        os.makedirs(os.path.dirname(_job_req_path(job_id)), exist_ok=True)
        data = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        with open(_job_req_path(job_id), "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"  [lora] could not persist request for {job_id}: {e}")


def _load_job_request(job_id: str):
    try:
        with open(_job_req_path(job_id)) as f:
            return LoraTrainRequest(**json.load(f))
    except Exception:
        return None


def list_jobs() -> list:
    """Snapshot of all training jobs (newest first) for the Tasks view."""
    with _jobs_lock:
        recs = [dict(r) for r in _jobs.values()]
    recs.sort(key=lambda r: r.get("started_at") or r.get("updated_at") or 0, reverse=True)
    return recs


def remove_job(job_id: str) -> bool:
    """Dismiss a finished/cancelled/errored training job from the view. Refuses a
    job that is still active (cancel it first → returns False). Drops the job
    record + its saved request sidecar; the trained LoRA weights and any training
    checkpoint on disk are kept."""
    with _jobs_lock:
        rec = _jobs.get(job_id)
        if rec is None:
            return False
        if rec.get("status") in _JOB_ACTIVE_STATES:
            return False
        _jobs.pop(job_id, None)
        _persist_jobs_locked(force=True)
    try:
        p = _job_req_path(job_id)
        if os.path.isfile(p):
            os.remove(p)
    except Exception:
        pass
    _cancel_requested.discard(job_id)
    _force_resume_jobs.discard(job_id)
    task_registry.remove(job_id)
    return True


def cancel_job(job_id: str) -> bool:
    """Cancel a queued or running training job. Running jobs stop at the next
    step; queued jobs are marked cancelled immediately (and aborted before any
    GPU work if they later acquire the lock). Checkpoints are kept."""
    with _jobs_lock:
        rec = _jobs.get(job_id)
        status = rec.get("status") if rec else None
    if rec is None:
        return False
    _cancel_requested.add(job_id)
    task_registry.cancel(job_id)  # no-op if the task isn't registered yet
    if status in ("queued", "interrupted", "preparing"):
        _update_job(job_id, status="cancelled", active=False, force=True,
                    message="cancelled")
    return True


def restart_job(job_id: str) -> Optional[str]:
    """Restart a finished/cancelled/interrupted training job, resuming from its
    last on-disk checkpoint. Returns the (same) job_id on success, None if the
    request payload could not be recovered."""
    import asyncio
    req = _load_job_request(job_id)
    if req is None:
        return None
    _force_resume_jobs.add(job_id)
    _cancel_requested.discard(job_id)
    _update_job(job_id, status="queued", active=True, force=True,
                message="restarting (resume from checkpoint)", step=0)
    task = asyncio.create_task(_detached_train(req, job_id))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return job_id


def _resume_allowed(req) -> bool:
    """Whether to auto-resume this training from its on-disk checkpoint. True when
    recovery is enabled globally, or when this is an explicit manual restart
    (restart_job adds the job id to _force_resume_jobs)."""
    return _RESUME_ENABLED or (_active_job_id in _force_resume_jobs)


def _lora_debug_enabled() -> bool:
    """LoRA training step logging to the terminal is gated on --debug-lora."""
    try:
        from codai.api.state import get_global_args
        return bool(getattr(get_global_args(), "debug_lora", False))
    except Exception:
        return False


def _dbg_lora(msg: str) -> None:
    if _lora_debug_enabled():
        print(f"  [lora][debug] {msg}", flush=True)


def _free_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.mem_get_info()[0] / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _free_train_vram() -> None:
    """Fully release training-base-model VRAM after a job.

    The trainer loads the SD1.x/SDXL base directly (not via the model manager),
    so eviction can't see it. peft adapter hooks create reference cycles that
    keep the modules alive until a gc pass runs, so empty_cache() alone leaves
    the memory pinned. Collect twice, then drop the allocator cache.
    """
    import gc
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _dbg_lora(f"freed training VRAM — {_free_vram_gb():.1f} GB now free")
    except Exception:
        pass


# ── Mid-training checkpoints (resume across restarts) ─────────────────────────
# Every _CKPT_EVERY steps we snapshot the raw PEFT adapter state(s) + a
# train_state.json into the LoRA's own folder (name-scoped, so two clients
# training different LoRAs never collide). If the process dies, the next
# submission for the same name/base/target/rank reloads the snapshot and
# continues from the saved step instead of restarting from scratch. Optimizer
# momentum is not restored (re-warms in a few steps); the trained weights — the
# expensive part — are preserved. The snapshot records the owning session so a
# resume can verify ownership.
_CKPT_EVERY = 100


def _train_state_path(name: str) -> str:
    return os.path.join(_lora_dir(name), "train_state.json")


def _save_train_checkpoint(name: str, state: dict, peft_states: dict) -> None:
    if not _ddp.is_main():
        return                      # rank 0's checkpoint is the job's
    from safetensors.torch import save_file
    d = _lora_dir(name)
    os.makedirs(d, exist_ok=True)
    try:
        for key, sd in peft_states.items():
            cpu_sd = {k: v.detach().to("cpu") for k, v in sd.items()}
            save_file(cpu_sd, os.path.join(d, f"_ckpt_{key}.safetensors"))
        tmp = _train_state_path(name) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, _train_state_path(name))
    except Exception as e:
        _dbg_lora(f"checkpoint save failed for '{name}': {e}")


def _load_train_state(name: str, *, base_path: str, target: str,
                      rank: int, session: Optional[str] = None) -> Optional[dict]:
    """Return a valid resumable checkpoint state for `name`, or None. Validates
    that base/target/rank match (and session, when provided) so we never resume
    one config's weights into another's run."""
    p = _train_state_path(name)
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            st = json.load(f)
    except Exception:
        return None
    if (st.get("base_path") != base_path or st.get("target") != target
            or int(st.get("rank", -1)) != int(rank)):
        return None
    if session is not None and st.get("session") not in (None, session):
        return None
    step = int(st.get("step", 0))
    if step <= 0 or step >= int(st.get("total", 0)):
        return None
    return st


def _apply_peft_checkpoint(name: str, key: str, model) -> None:
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    f = os.path.join(_lora_dir(name), f"_ckpt_{key}.safetensors")
    if not os.path.isfile(f):
        raise FileNotFoundError(f)
    set_peft_model_state_dict(model, load_file(f), adapter_name="default")


def _clear_train_checkpoint(name: str) -> None:
    if not _ddp.is_main():
        return
    d = _lora_dir(name)
    if not os.path.isdir(d):
        return
    for fn in os.listdir(d):
        if fn.startswith("_ckpt_") or fn == "train_state.json":
            try:
                os.remove(os.path.join(d, fn))
            except Exception:
                pass


# ── Training base-model cache ────────────────────────────────────────────────
# The SD1.x/SDXL base used for LoRA training is expensive to load from disk.
# We keep its components cached *on CPU* between consecutive training jobs so a
# back-to-back job against the same base skips the reload. Holding it on CPU
# (not GPU) means it consumes no VRAM between jobs, so image/video generation
# can use the GPU freely — components are moved to the GPU only while a job is
# actually running, then moved back to CPU when it finishes.
_base_lock = threading.RLock()
_base_cache = {"path": None, "arch": None, "components": None}


def _load_base_components(base_path, arch, dtype):
    """Load all base components on CPU at the given dtype (no GPU placement)."""
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    if arch == "sdxl":
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
        return {
            "tokenizer_1": CLIPTokenizer.from_pretrained(base_path, subfolder="tokenizer"),
            "tokenizer_2": CLIPTokenizer.from_pretrained(base_path, subfolder="tokenizer_2"),
            "text_encoder_1": CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder").to(dtype=dtype),
            "text_encoder_2": CLIPTextModelWithProjection.from_pretrained(base_path, subfolder="text_encoder_2").to(dtype=dtype),
            "vae": AutoencoderKL.from_pretrained(base_path, subfolder="vae").to(dtype=dtype),
            "unet": UNet2DConditionModel.from_pretrained(base_path, subfolder="unet").to(dtype=dtype),
            "noise_scheduler": DDPMScheduler.from_pretrained(base_path, subfolder="scheduler"),
        }
    from transformers import CLIPTextModel, CLIPTokenizer
    return {
        "tokenizer": CLIPTokenizer.from_pretrained(base_path, subfolder="tokenizer"),
        "text_encoder": CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder").to(dtype=dtype),
        "vae": AutoencoderKL.from_pretrained(base_path, subfolder="vae").to(dtype=dtype),
        "unet": UNet2DConditionModel.from_pretrained(base_path, subfolder="unet").to(dtype=dtype),
        "noise_scheduler": DDPMScheduler.from_pretrained(base_path, subfolder="scheduler"),
    }


def _acquire_base(base_path, arch, dtype):
    """Return cached CPU-resident base components, loading from disk if needed.

    A change of base_path/arch drops the previous cache first. Must be called
    inside a training job (holds the train lock for the job's duration)."""
    with _base_lock:
        c = _base_cache
        if c["components"] is not None and c["path"] == base_path and c["arch"] == arch:
            _dbg_lora(f"reusing cached base model (CPU): {base_path}")
            return c["components"]
        if c["components"] is not None:
            _dbg_lora(f"base model changed ({c['path']} → {base_path}); dropping cache")
            _drop_base_cache_locked()
        _dbg_lora(f"loading base model from disk: {base_path} ({arch})")
        comps = _load_base_components(base_path, arch, dtype)
        c.update(path=base_path, arch=arch, components=comps)
        return comps


def _drop_base_cache_locked() -> None:
    _base_cache["components"] = None
    _base_cache["path"] = None
    _base_cache["arch"] = None


def _drop_base_cache() -> None:
    with _base_lock:
        had = _base_cache["components"] is not None
        _drop_base_cache_locked()
    if had:
        _free_train_vram()


def _release_base_cache(needed_gb: float = 0.0) -> float:
    """External VRAM releaser (registered with the model manager).

    Drops the cached training base on demand. Skips while a training job is
    running (the base is in use). Between jobs the base lives on CPU, so this
    mainly reclaims host RAM; it returns the VRAM delta it observed."""
    if not _train_lock.acquire(blocking=False):
        return 0.0
    try:
        had = False
        with _base_lock:
            had = _base_cache["components"] is not None
        with _wan_lock:
            had = had or _wan_cache["experts"] is not None
        if not had:
            return 0.0
        before = _free_vram_gb()
        _drop_base_cache()
        _drop_wan_cache()   # the Wan transformer(s) sit on GPU between jobs
        return max(0.0, _free_vram_gb() - before)
    finally:
        _train_lock.release()


# ── Wan video-DiT base cache ─────────────────────────────────────────────────
# The Wan transformer expert(s) are very expensive to load (a 14B A14B model can
# take tens of minutes). Keep them resident between consecutive trainings against
# the same base so a back-to-back job skips the reload. Unlike the SD base (held
# on CPU), 4-bit Wan transformers can't move off the GPU, so they hold VRAM
# between jobs — the external releaser above drops them the moment a generation
# needs the GPU.
_wan_lock = threading.RLock()
# Caches the whole Wan stack between same-base jobs: transformer expert(s) on the
# GPU, plus the VAE / tokenizer / text-encoder held on CPU (moved to the GPU only
# while encoding). So a back-to-back training reloads nothing from disk.
_wan_cache = {"path": None, "quantize": None, "experts": None, "boundary": None,
              "vae": None, "tokenizer": None, "text_encoder": None}


def _drop_wan_cache_locked() -> None:
    exp = _wan_cache.get("experts")
    if exp:
        for _, tr in exp:
            try:
                tr.delete_adapters("default")
            except Exception:
                pass
    _wan_cache.update(path=None, quantize=None, experts=None, boundary=None,
                      vae=None, tokenizer=None, text_encoder=None)


def _drop_wan_cache() -> None:
    with _wan_lock:
        had = _wan_cache.get("experts") is not None
        _drop_wan_cache_locked()
    if had:
        _free_train_vram()


def _acquire_wan_components(base_path, quantize, load_fn):
    """Return the cached Wan stack {vae, tokenizer, text_encoder, experts,
    boundary} for base_path, building it via load_fn() on a miss. A change of
    base_path/quantize drops the previous cache first."""
    with _wan_lock:
        c = _wan_cache
        if (c["experts"] is not None and c["path"] == base_path
                and c["quantize"] == quantize):
            _dbg_lora(f"reusing cached Wan stack (no reload): {base_path}")
            return c
        if c["experts"] is not None or c["vae"] is not None:
            _dbg_lora(f"Wan base changed ({c['path']} → {base_path}); dropping cache")
            _drop_wan_cache_locked()
        comps = load_fn()  # {vae, tokenizer, text_encoder, experts, boundary}
        c.update(path=base_path, quantize=quantize, **comps)
        return c


# Let the model manager reclaim these caches when a generation needs VRAM.
try:
    from codai.models.manager import multi_model_manager as _mmm
    _mmm.register_external_vram_releaser(_release_base_cache)
except Exception:
    pass


# ── Training core ─────────────────────────────────────────────────────────────

def _train_lora_sync(req: LoraTrainRequest) -> dict:
    """Run a DreamBooth-style LoRA training in-process. Returns {name, path}."""
    import torch
    import torch.nn.functional as F
    from diffusers import (
        AutoencoderKL, DDPMScheduler, UNet2DConditionModel,
    )
    from diffusers.optimization import get_scheduler
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig as PeftLoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    name = req.name

    # ── Video (Wan DiT) LoRA target ───────────────────────────────────────────
    # Train directly against the configured VIDEO model so the resulting LoRA
    # loads on the video pipeline. No SD UNet fallback here — the base IS the DiT.
    from codai.api.lora_archs import detect_arch as _detect_arch
    _target = (req.target or "image").lower()
    if _target in ("video", "wan", "ltx2", "h3", "krea"):
        _cat = "image" if _target == "krea" else "video"
        video_path = _resolve_base_model_path(req.base_model, category=_cat)
        steps = max(50, min(5000, int(req.steps or 800)))
        rank = max(2, min(128, int(req.rank or 16)))
        resolution = int(req.resolution or 512)
        lr = float(req.learning_rate or 1e-4)
        seed = int(req.seed if req.seed is not None else 42)
        _set_progress(active=True, name=name, step=0, total=steps,
                      status="preparing", message="loading reference images",
                      started_at=time.time(), path=None)
        images = _gather_images(req)
        if not images:
            raise HTTPException(status_code=400,
                                detail="No training images (provide `character` or `images`)")
        if req.instance_prompt:
            instance_prompt = req.instance_prompt
        elif req.environment and not req.character:
            instance_prompt = f"a video of {name} place"
        else:
            instance_prompt = f"a video of {name} person"
        try:
            from codai.models.manager import multi_model_manager
            multi_model_manager.unload_all_models()
            # Also release any co-located sibling engine's model (shared-GPU split):
            # training needs the whole card and the swap-gate doesn't cover it.
            _freed = multi_model_manager.evict_cosited_siblings()
            if _freed:
                print(f"  [lora] freed {_freed:.1f} GB from co-located engine(s) for training")
        except Exception as e:
            print(f"  [lora] could not unload models before training: {e}")
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        # Which flow-matching architecture is this? An explicit target wins,
        # otherwise it is read off the model id — so `target: video` keeps meaning
        # Wan for existing configs, while an LTX-2 or H3 path routes itself.
        _arch = _detect_arch(video_path, _target) or "wan"
        if _arch != "wan":
            print(f"  [lora] training a {_arch} adapter for {video_path}")
            # Some architectures do not train in this process at all. LongCat-Video is
            # the case: its venv is standalone (Python 3.10 / torch 2.6) so it cannot
            # reuse this module the way the overlay trainer does, AND its repo has no way
            # to create trainable LoRA layers — only load_lora() from a file, in its own
            # key layout. SimpleTuner implements it properly, so coderai drives that.
            from codai.api.lora_archs import external_trainer as _external
            if _external(_arch):
                return _train_externally(_arch, req, video_path, images,
                                         instance_prompt, steps, rank, resolution,
                                         lr, seed)
            # H3 and Krea 2 need diffusers >= 0.40 and this process has 0.38 — and
            # diffusers is imported once per process, so they train in the overlay
            # venv instead. LTX-2 needs nothing newer and stays in-process.
            if _arch in ("h3", "krea") and not _diffusers_has(_arch):
                return _train_via_overlay(_arch, req, video_path, images,
                                          instance_prompt, steps, rank, resolution,
                                          lr, seed, device)
            return _train_flow_dit(_arch, req, video_path, images, instance_prompt,
                                   steps, rank, resolution, lr, seed, device)
        return _train_wan(req, video_path, images, instance_prompt,
                          steps, rank, resolution, lr, seed, device)

    # Resolve the model to TRAIN against (the generation model may be a DiT this
    # trainer can't target). Precedence: explicit request override > per-model
    # `lora_train_base_model` from models.json config > the base_model itself.
    train_base = (req.train_base_model
                  or _configured_train_base(req.base_model)
                  or req.base_model)
    if train_base != req.base_model:
        print(f"  [lora] training '{name}' against base '{train_base}' "
              f"(generation model: '{req.base_model}')")
    base_path = _resolve_base_model_path(train_base)
    steps = max(50, min(5000, int(req.steps or 800)))
    rank = max(2, min(128, int(req.rank or 16)))
    resolution = int(req.resolution or 512)
    lr = float(req.learning_rate or 1e-4)
    seed = int(req.seed if req.seed is not None else 42)

    _set_progress(active=True, name=name, step=0, total=steps,
                  status="preparing", message="loading reference images",
                  started_at=time.time(), path=None)

    images = _gather_images(req)
    if not images:
        raise HTTPException(status_code=400,
                            detail="No training images (provide `character` or `images`)")

    if req.instance_prompt:
        instance_prompt = req.instance_prompt
    elif req.environment and not req.character:
        instance_prompt = f"a photo of {name} place"
    else:
        instance_prompt = f"a photo of {name} person"

    # Free VRAM: evict every resident model so training has the whole GPU.
    try:
        from codai.models.manager import multi_model_manager
        multi_model_manager.unload_all_models()
        # Also release any co-located sibling engine's model (shared-GPU split):
        # unload_all_models() only frees THIS engine; a sibling's resident model
        # (e.g. the gguf text engine) would otherwise share the card and OOM
        # training. The swap-gate doesn't cover training, so evict explicitly.
        _freed = multi_model_manager.evict_cosited_siblings()
        if _freed:
            print(f"  [lora] freed {_freed:.1f} GB from co-located engine(s) for training")
    except Exception as e:
        print(f"  [lora] could not unload models before training: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float32  # train in fp32 for stability

    _set_progress(status="preparing", message=f"loading base model: {base_path}")

    # This DreamBooth-LoRA trainer only supports UNet-based SD1.x / SDXL
    # pipelines.  Transformer/DiT models (Z-Image, Flux, SD3, …) have a
    # `transformer/` subfolder and no `unet/`; loading their non-CLIP tokenizer
    # as a CLIPTokenizer crashes deep inside transformers.  Detect that up front
    # and fail with an actionable message instead.
    import os as _os, json as _json2
    _has_unet = _os.path.isdir(_os.path.join(base_path, "unet"))
    _has_transformer = _os.path.isdir(_os.path.join(base_path, "transformer"))
    # Z-Image is a DiT and must train via _train_dit (its LoRA then loads on the
    # Z-Image pipeline). Detect it by NAME first: _resolve_base_model_path returns an
    # HF id verbatim (e.g. 'unsloth/Z-Image-Turbo-...') for models with no local
    # `path`, so the isdir(transformer/) check below is False and it would otherwise
    # misroute to the SDXL/SD15 trainer (which crashes loading a CLIP tokenizer
    # Z-Image doesn't have). Fall back to the diffusers-config class name for local
    # dirs / other DiTs.
    _bm_name = f"{base_path} {getattr(req, 'base_model', '') or ''}".lower()
    _is_zimage = ("z-image" in _bm_name) or ("zimage" in _bm_name)
    if not _is_zimage and _has_transformer and not _has_unet:
        try:
            for _p in (_os.path.join(base_path, "model_index.json"),
                       _os.path.join(base_path, "transformer", "config.json")):
                if _os.path.isfile(_p):
                    if "zimage" in (_json2.load(open(_p)).get("_class_name") or "").lower():
                        _is_zimage = True
                        break
        except Exception:
            pass
    if _is_zimage:
        return _train_dit(req, base_path, images, instance_prompt,
                          steps, rank, resolution, lr, seed, device)
    if _has_transformer and not _has_unet:
        raise ValueError(
            f"LoRA training base model '{base_path}' is a transformer/DiT "
            f"architecture (has 'transformer/', no 'unet/'). Native DiT training is "
            f"currently implemented for Z-Image only. For Flux/SD3, configure an SDXL "
            f"`lora_train_base_model` override, or use an external DiT trainer."
        )

    # Detect SDXL by attempting to load a second tokenizer.
    is_sdxl = False
    try:
        from transformers import CLIPTokenizer as _CT
        _CT.from_pretrained(base_path, subfolder="tokenizer_2")
        is_sdxl = True
    except Exception:
        is_sdxl = False

    if is_sdxl:
        result = _train_sdxl(req, base_path, images, instance_prompt,
                             steps, rank, resolution, lr, seed, device)
    else:
        result = _train_sd15(req, base_path, images, instance_prompt,
                             steps, rank, resolution, lr, seed, device)
    return result


def _make_dataset(images, tokenizers, text_encoders, instance_prompt,
                  resolution, vae, device, weight_dtype, is_sdxl):
    """Pre-encode latents + text embeddings for every reference image once."""
    import torch
    from torchvision import transforms

    tfm = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    latents_list = []
    with torch.no_grad():
        for img in images:
            px = tfm(img).unsqueeze(0).to(device, dtype=vae.dtype)
            lat = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
            latents_list.append(lat.to(weight_dtype).cpu())
    return latents_list


def _train_sd15(req, base_path, images, instance_prompt,
                steps, rank, resolution, lr, seed, device):
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from diffusers import StableDiffusionPipeline
    from peft import LoraConfig as PeftLoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    name = req.name
    g = torch.Generator(device=device).manual_seed(seed)

    # Consistent fp32 precision (see _train_sdxl) to avoid mixed-dtype crashes.
    weight_dtype = torch.float32
    # Components come from the cross-job CPU cache; move to GPU for this job.
    comps = _acquire_base(base_path, "sd15", weight_dtype)
    tokenizer = comps["tokenizer"]
    text_encoder = comps["text_encoder"].to(device, dtype=weight_dtype)
    vae = comps["vae"].to(device, dtype=weight_dtype)
    unet = comps["unet"].to(device, dtype=weight_dtype)
    noise_scheduler = comps["noise_scheduler"]

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_cfg = PeftLoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    _ensure_peft_awq_compat()
    unet.add_adapter(lora_cfg, adapter_name="default")
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    _ddp.broadcast_params(lora_params)      # all ranks start from rank 0's init
    optimizer = torch.optim.AdamW(lora_params, lr=lr)

    # Resume from a mid-training checkpoint if one survives a prior restart.
    start_step = 0
    _ck = _load_train_state(name, base_path=base_path, target="image", rank=rank,
                            session=getattr(req, "session", None)) if _resume_allowed(req) else None
    if _ck:
        try:
            _apply_peft_checkpoint(name, "default", unet)
            start_step = int(_ck["step"])
            _set_progress(step=start_step,
                          message=f"resuming from step {start_step}/{steps}")
            _dbg_lora(f"resumed SD1.5 '{name}' from checkpoint step {start_step}")
        except Exception as e:
            print(f"  [lora] could not resume '{name}': {e}; starting fresh")
            start_step = 0

    # Pre-encode latents and the (single) instance-prompt embedding.
    latents_list = _make_dataset(images, [tokenizer], [text_encoder], instance_prompt,
                                 resolution, vae, device, torch.float32, is_sdxl=False)
    with torch.no_grad():
        tok = tokenizer(instance_prompt, padding="max_length",
                        max_length=tokenizer.model_max_length, truncation=True,
                        return_tensors="pt").input_ids.to(device)
        encoder_hidden_states = text_encoder(tok)[0]

    # VAE + text encoder are done; move them back to CPU (keeps them cached for
    # the next job) so only the UNet stays resident during training.
    vae.to("cpu")
    text_encoder.to("cpu")
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    _set_progress(status="training", message="training (SD1.5)")
    unet.train()
    n = len(latents_list)
    for step in range(start_step, steps):
        _check_train_cancel()
        latents = latents_list[_ddp.index(step, n)].to(device)
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (bsz,), device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)
        model_pred = unet(noisy, timesteps, encoder_hidden_states).sample

        if noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss.backward()
        _ddp.sync_grads(lora_params)      # average over the ranks
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0 or step == steps - 1:
            _set_progress(step=step + 1, message=f"step {step+1}/{steps} loss={loss.item():.4f}")
            _dbg_lora(f"SD1.5 step {step+1}/{steps} loss={loss.item():.4f}")
        if (step + 1) % _CKPT_EVERY == 0 and step + 1 < steps:
            _save_train_checkpoint(name,
                {"name": name, "base_path": base_path, "target": "image",
                 "rank": rank, "step": step + 1, "total": steps, "seed": seed,
                 "session": getattr(req, "session", None)},
                {"default": get_peft_model_state_dict(unet)})
        # Mid-training thermal checkpoint (pauses if CPU/GPU too hot).
        try:
            from codai.models.thermal import checkpoint as _thermal_checkpoint
            _thermal_checkpoint(context="lora-train", throttle_seconds=2.0)
        except Exception:
            pass

    _set_progress(status="saving", message="saving LoRA weights")
    save_dir = _lora_dir(name)
    os.makedirs(save_dir, exist_ok=True)
    unet_lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(save_directory=save_dir,
                                              unet_lora_layers=unet_lora,
                                              safe_serialization=True)
    _write_meta(name, req, base_path, len(images), "sd15", instance_prompt)
    _clear_train_checkpoint(name)

    # Job done: drop this job's adapter + transients and move the UNet back to
    # CPU. The base stays cached on CPU (reused by the next job); no VRAM is held
    # afterwards, so the next image/video request gets the full GPU.
    try:
        unet.delete_adapters("default")
    except Exception:
        pass
    unet.to("cpu")
    try:
        del optimizer, latents_list, lora_params, encoder_hidden_states
    except Exception:
        pass
    _free_train_vram()

    path = _lora_weight_file(name) or save_dir
    _set_progress(active=False, status="done", message="done", path=path)
    return {"name": name, "path": path}


def _train_sdxl(req, base_path, images, instance_prompt,
                steps, rank, resolution, lr, seed, device):
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig as PeftLoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from torchvision import transforms

    name = req.name
    g = torch.Generator(device=device).manual_seed(seed)

    # Train in a single consistent precision. Checkpoints can store each
    # component in a different native dtype (e.g. fp16 text encoders + fp32
    # UNet), which otherwise crashes with "mat1 and mat2 have the same dtype"
    # (Half != float) in cross-attention. fp32 is the stable reference for
    # LoRA fine-tuning.
    weight_dtype = torch.float32

    # Components come from the cross-job CPU cache; move to GPU for this job.
    comps = _acquire_base(base_path, "sdxl", weight_dtype)
    tokenizer_1 = comps["tokenizer_1"]
    tokenizer_2 = comps["tokenizer_2"]
    text_encoder_1 = comps["text_encoder_1"].to(device, dtype=weight_dtype)
    text_encoder_2 = comps["text_encoder_2"].to(device, dtype=weight_dtype)
    vae = comps["vae"].to(device, dtype=weight_dtype)
    unet = comps["unet"].to(device, dtype=weight_dtype)
    noise_scheduler = comps["noise_scheduler"]

    for m in (vae, text_encoder_1, text_encoder_2, unet):
        m.requires_grad_(False)

    lora_cfg = PeftLoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    _ensure_peft_awq_compat()
    unet.add_adapter(lora_cfg, adapter_name="default")
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    _ddp.broadcast_params(lora_params)      # all ranks start from rank 0's init
    optimizer = torch.optim.AdamW(lora_params, lr=lr)

    # Resume from a mid-training checkpoint if one survives a prior restart.
    start_step = 0
    _ck = _load_train_state(name, base_path=base_path, target="image", rank=rank,
                            session=getattr(req, "session", None)) if _resume_allowed(req) else None
    if _ck:
        try:
            _apply_peft_checkpoint(name, "default", unet)
            start_step = int(_ck["step"])
            _set_progress(step=start_step,
                          message=f"resuming from step {start_step}/{steps}")
            _dbg_lora(f"resumed SDXL '{name}' from checkpoint step {start_step}")
        except Exception as e:
            print(f"  [lora] could not resume '{name}': {e}; starting fresh")
            start_step = 0

    tfm = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Pre-encode latents.
    latents_list = []
    with torch.no_grad():
        for img in images:
            px = tfm(img).unsqueeze(0).to(device, dtype=vae.dtype)
            lat = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
            latents_list.append(lat.float().cpu())

    # SDXL text conditioning: concat hidden states + pooled embeds from encoder 2.
    with torch.no_grad():
        ids_1 = tokenizer_1(instance_prompt, padding="max_length",
                            max_length=tokenizer_1.model_max_length, truncation=True,
                            return_tensors="pt").input_ids.to(device)
        ids_2 = tokenizer_2(instance_prompt, padding="max_length",
                            max_length=tokenizer_2.model_max_length, truncation=True,
                            return_tensors="pt").input_ids.to(device)
        enc1 = text_encoder_1(ids_1, output_hidden_states=True)
        enc2 = text_encoder_2(ids_2, output_hidden_states=True)
        # penultimate hidden states
        prompt_embeds = torch.cat([enc1.hidden_states[-2], enc2.hidden_states[-2]], dim=-1)
        pooled = enc2[0]  # text_embeds (pooled) from projection encoder

    add_time_ids = torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]],
        device=device, dtype=prompt_embeds.dtype,
    )

    # VAE + text encoders are no longer needed during the training loop; move
    # them back to CPU (keeps them cached for the next job) so only the UNet
    # stays resident — keeps SDXL fp32 training in VRAM budget.
    vae.to("cpu")
    text_encoder_1.to("cpu")
    text_encoder_2.to("cpu")
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    _set_progress(status="training", message="training (SDXL)")
    unet.train()
    n = len(latents_list)
    for step in range(start_step, steps):
        _check_train_cancel()
        latents = latents_list[_ddp.index(step, n)].to(device)
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (bsz,), device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)
        added = {"text_embeds": pooled, "time_ids": add_time_ids}
        model_pred = unet(noisy, timesteps, prompt_embeds, added_cond_kwargs=added).sample

        if noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss.backward()
        _ddp.sync_grads(lora_params)      # average over the ranks
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0 or step == steps - 1:
            _set_progress(step=step + 1, message=f"step {step+1}/{steps} loss={loss.item():.4f}")
            _dbg_lora(f"SDXL step {step+1}/{steps} loss={loss.item():.4f}")
        if (step + 1) % _CKPT_EVERY == 0 and step + 1 < steps:
            _save_train_checkpoint(name,
                {"name": name, "base_path": base_path, "target": "image",
                 "rank": rank, "step": step + 1, "total": steps, "seed": seed,
                 "session": getattr(req, "session", None)},
                {"default": get_peft_model_state_dict(unet)})
        try:
            from codai.models.thermal import checkpoint as _thermal_checkpoint
            _thermal_checkpoint(context="lora-train", throttle_seconds=2.0)
        except Exception:
            pass

    _set_progress(status="saving", message="saving LoRA weights")
    save_dir = _lora_dir(name)
    os.makedirs(save_dir, exist_ok=True)
    unet_lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(save_directory=save_dir,
                                                unet_lora_layers=unet_lora,
                                                safe_serialization=True)
    _write_meta(name, req, base_path, len(images), "sdxl", instance_prompt)
    _clear_train_checkpoint(name)

    # Job done: drop this job's adapter + transients and move the UNet back to
    # CPU. The base stays cached on CPU for the next job; no VRAM held afterwards
    # so the next image/video request gets the full GPU (see _train_sd15 note).
    try:
        unet.delete_adapters("default")
    except Exception:
        pass
    unet.to("cpu")
    try:
        del optimizer, latents_list, lora_params
        del prompt_embeds, pooled, add_time_ids
    except Exception:
        pass
    _free_train_vram()

    path = _lora_weight_file(name) or save_dir
    _set_progress(active=False, status="done", message="done", path=path)
    return {"name": name, "path": path}


def _ensure_peft_awq_compat():
    """peft's LoRA AWQ dispatcher does `from gptqmodel.nn_modules.qlinear.gemm_awq
    import AwqGEMMQuantLinear`, but gptqmodel 7.1.0 renamed that class to
    AwqGEMMLinear. Since peft calls dispatch_awq for ANY non-bnb target whenever
    gptqmodel is installed, the failed import crashes EVERY add_adapter() (SDXL, Wan
    and Z-Image LoRA training alike). Delegates to the shared shim (also used by
    the inference LoRA-apply path)."""
    try:
        from codai.models.peft_compat import ensure_peft_awq_compat
        ensure_peft_awq_compat()
    except Exception:
        pass


def _train_dit(req, base_path, images, instance_prompt,
               steps, rank, resolution, lr, seed, device):
    """Train a LoRA for an image diffusion-TRANSFORMER (DiT) — currently Z-Image
    (ZImageTransformer2DModel) — so it loads on the Z-Image pipeline. The SDXL-UNet
    trainer produces `unet.`-prefixed keys that silently no-op on a DiT; this path
    produces `transformer.`-prefixed keys diffusers' ZImagePipeline.load_lora_weights
    applies correctly.

    Reverse-engineered from diffusers ZImagePipeline so training matches inference:
      * prompt: tokenizer.apply_chat_template → Qwen3 text_encoder → hidden_states[-2],
        masked to a variable-length per-sample embedding (the transformer takes a LIST).
      * latents: AutoencoderKL, scaled (lat - shift_factor) * scaling_factor.
      * transformer I/O is list-based: a LIST of [C,1,H,W] latents, a normalized
        timestep (1000 - t)/1000, and a prompt-embed LIST; output is a list, stacked.
      * Z-Image NEGATES the model output before the flow-match step, so the
        transformer's RAW target is (x0 - noise) = -velocity.
      * timesteps are sampled from the TURBO discrete schedule (set_timesteps with the
        resolution shift mu), not a uniform sigma — staying on the ~9 turbo nodes keeps
        the distilled model sharp (uniform sampling blurs it).

    v1: validate with one short train (rank 8, ~800 steps) + a generation. If the
    LoRA inverts the subject, flip the velocity sign (target = noise - x0); if results
    are soft/blurry, the turbo-node sampling is the knob.
    """
    import os as _os, json as _json, random as _random
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig as PeftLoraConfig
    from peft.utils import get_peft_model_state_dict
    from torchvision import transforms
    try:
        from diffusers import (AutoencoderKL, ZImageTransformer2DModel,
                               ZImagePipeline, FlowMatchEulerDiscreteScheduler)
        try:
            from diffusers import BitsAndBytesConfig as _DiffBnb
        except Exception:
            _DiffBnb = None
        from diffusers.utils import convert_state_dict_to_diffusers
        from transformers import AutoTokenizer, AutoModel
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=("Z-Image LoRA training needs a diffusers build with Z-Image "
                    f"support (ZImageTransformer2DModel/ZImagePipeline): {e}"))

    name = req.name
    torch.manual_seed(seed)
    compute_dtype = torch.bfloat16

    def _calc_shift(seq_len, base=256, mx=4096, bs=0.5, ms=1.15):
        m = (ms - bs) / (mx - base)
        return seq_len * m + (bs - m * base)

    # ── Load via the ZImagePipeline (the proven inference loader) ──────────────
    # The unsloth build quantizes the TEXT ENCODER too; loading components piecemeal
    # (AutoModel/AutoencoderKL by subfolder) leaves bitsandbytes' 4-bit quant-state
    # unreconstructed -> AssertionError in Linear4bit.forward. The pipeline loads
    # every component exactly as inference does (which works), 4-bit where the
    # checkpoint specifies, so the frozen base stays ~4 GB on-GPU (no offload, fast).
    # We LoRA-train only the transformer.
    _set_progress(status="preparing", message=f"loading Z-Image pipeline: {base_path}")
    pipe = ZImagePipeline.from_pretrained(base_path, torch_dtype=compute_dtype)
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    transformer = pipe.transformer

    def _to_dev(m):
        """Move a component to the GPU unless it's bnb-4-bit (already device-mapped;
        .to() on a 4-bit module raises)."""
        try:
            return m.to(device)
        except Exception:
            return m

    # ── VAE encode reference images → scaled latents ───────────────────────────
    _set_progress(status="preparing", message="encoding reference images (VAE)")
    vae = _to_dev(vae); vae.requires_grad_(False); vae.eval()
    # The pipeline loads the VAE in the pipeline dtype (bf16); the input image tensor
    # must match the VAE's weight dtype or conv2d raises "Input type (float) and bias
    # type (BFloat16) should be the same".
    try:
        _vdt = next(vae.parameters()).dtype
    except Exception:
        _vdt = compute_dtype
    sf = float(getattr(vae.config, "scaling_factor", 1.0) or 1.0)
    shf = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    spatial = (int(resolution) // 16) * 16
    tfm = transforms.Compose([
        transforms.Resize(spatial, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(spatial),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    latents_list = []
    with torch.no_grad():
        for img in images:
            px = tfm(img.convert("RGB")).unsqueeze(0).to(device, dtype=_vdt)
            lat = vae.encode(px).latent_dist.sample()
            lat = (lat - shf) * sf
            latents_list.append(lat.to(compute_dtype).cpu())
    try:
        vae.to("cpu")
    except Exception:
        pass
    pipe.vae = None; vae = None; _free_train_vram()

    # ── Qwen3 text encode of the instance prompt (chat template, hidden[-2]) ────
    _set_progress(status="preparing", message="encoding prompt (Qwen3)")
    text_encoder = _to_dev(text_encoder); text_encoder.requires_grad_(False); text_encoder.eval()
    with torch.no_grad():
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": instance_prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        ti = tokenizer([templated], padding="max_length", max_length=512,
                       truncation=True, return_tensors="pt")
        ids = ti.input_ids.to(device); msk = ti.attention_mask.to(device).bool()
        hs = text_encoder(input_ids=ids, attention_mask=msk,
                          output_hidden_states=True).hidden_states[-2]
        prompt_embed = hs[0][msk[0]].to(compute_dtype).cpu()   # [seq, dim]
    try:
        text_encoder.to("cpu")
    except Exception:
        pass
    pipe.text_encoder = None; text_encoder = None; _free_train_vram()

    # ── LoRA on the transformer ────────────────────────────────────────────────
    lora_cfg = PeftLoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                              target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    transformer.requires_grad_(False)
    _ensure_peft_awq_compat()
    transformer.add_adapter(lora_cfg, adapter_name="default")
    _hooks = []
    # Gradient checkpointing is intentionally OFF: with a bnb-4-bit frozen base the
    # checkpoint recompute can desync bitsandbytes' quant-state (the same failure
    # class as the text-encoder load). LoRA grads still reach the adapters without it
    # (each LoRA layer's own params keep its output in the autograd graph), and at
    # 512 res / batch 1 the ~4 GB 4-bit base + activations fit. Re-enable it (plus an
    # all_x_embedder requires-grad hook) only if higher-res training OOMs.
    transformer.train()
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    if not lora_params:
        raise HTTPException(status_code=500,
                            detail="Z-Image LoRA: no trainable adapter params were created")
    _ddp.broadcast_params(lora_params)      # all ranks start from rank 0's init
    optimizer = torch.optim.AdamW(lora_params, lr=lr)

    start_step = 0
    _ck = _load_train_state(name, base_path=base_path, target="image", rank=rank,
                            session=getattr(req, "session", None)) if _resume_allowed(req) else None
    if _ck:
        try:
            _apply_peft_checkpoint(name, "default", transformer)
            start_step = int(_ck["step"])
            _set_progress(step=start_step, message=f"resuming from step {start_step}/{steps}")
        except Exception as e:
            print(f"  [lora] could not resume Z-Image '{name}': {e}; starting fresh")
            start_step = 0

    # ── Turbo discrete timestep schedule (match inference) ─────────────────────
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(base_path, subfolder="scheduler")
    num_train_t = float(getattr(sched.config, "num_train_timesteps", 1000) or 1000)
    _lh, _lw = latents_list[0].shape[-2], latents_list[0].shape[-1]
    image_seq_len = (_lh // 2) * (_lw // 2)
    mu = _calc_shift(image_seq_len,
                     getattr(sched.config, "base_image_seq_len", 256),
                     getattr(sched.config, "max_image_seq_len", 4096),
                     getattr(sched.config, "base_shift", 0.5),
                     getattr(sched.config, "max_shift", 1.15))
    try:
        sched.sigma_min = 0.0
    except Exception:
        pass
    try:
        sched.set_timesteps(num_inference_steps=9, device=device, mu=mu)
    except TypeError:
        sched.set_timesteps(num_inference_steps=9, device=device)
    node_t = sched.timesteps.to(device).float()
    node_sigma = sched.sigmas.to(device).float()[:len(node_t)]

    _set_progress(status="training", message="training (Z-Image LoRA)")
    n = len(latents_list)
    embed_dev = prompt_embed.to(device)
    for step in range(start_step, steps):
        _check_train_cancel()
        x0 = latents_list[_ddp.index(step, n)].to(device, dtype=compute_dtype)   # [1,C,h,w]
        noise = torch.randn_like(x0)
        j = _random.randint(0, len(node_t) - 1)
        t = node_t[j]; sigma = node_sigma[j]
        s = sigma.view(1, 1, 1, 1)
        x0f = x0.float(); nf = noise.float()
        x_t = ((1.0 - s) * x0f + s * nf).to(compute_dtype)
        target = (x0f - nf).to(compute_dtype)                          # -velocity (Z-Image negates output)
        timestep_norm = ((num_train_t - t) / num_train_t).view(1).to(compute_dtype)  # (1000-t)/1000

        in_list = list(x_t.unsqueeze(2).unbind(dim=0))                 # [ [C,1,h,w] ]
        out = transformer(in_list, timestep_norm, [embed_dev], return_dict=False)[0]
        pred = torch.stack([o.float() for o in out], dim=0).squeeze(2)  # [1,C,h,w]
        loss = F.mse_loss(pred, target.float(), reduction="mean")
        loss.backward()
        _ddp.sync_grads(lora_params)      # average over the ranks
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step(); optimizer.zero_grad()

        _set_progress(step=step + 1, message=f"step {step+1}/{steps} loss={loss.item():.4f}")
        if step % 10 == 0 or step == steps - 1:
            _dbg_lora(f"Z-Image step {step+1}/{steps} loss={loss.item():.4f}")
        if (step + 1) % _CKPT_EVERY == 0 and step + 1 < steps:
            _save_train_checkpoint(name,
                {"name": name, "base_path": base_path, "target": "image",
                 "rank": rank, "step": step + 1, "total": steps, "seed": seed,
                 "session": getattr(req, "session", None)},
                {"default": get_peft_model_state_dict(transformer)})
        try:
            from codai.models.thermal import checkpoint as _thermal_checkpoint
            _thermal_checkpoint(context="lora-train", throttle_seconds=2.0)
        except Exception:
            pass

    # ── Save (transformer LoRA layers, diffusers format) ───────────────────────
    _set_progress(status="saving", message="saving Z-Image LoRA weights")
    save_dir = _lora_dir(name)
    os.makedirs(save_dir, exist_ok=True)
    tlayers = convert_state_dict_to_diffusers(get_peft_model_state_dict(transformer))
    ZImagePipeline.save_lora_weights(save_directory=save_dir,
                                     transformer_lora_layers=tlayers,
                                     safe_serialization=True)
    _write_meta(name, req, base_path, len(images), "zimage", instance_prompt)
    _clear_train_checkpoint(name)
    for _h in _hooks:
        try:
            _h.remove()
        except Exception:
            pass
    try:
        transformer.delete_adapters("default")
    except Exception:
        pass
    try:
        pipe.transformer = None
    except Exception:
        pass
    del transformer
    try:
        del pipe
    except Exception:
        pass
    _free_train_vram()
    path = _lora_weight_file(name) or save_dir
    _set_progress(active=False, status="done", message="done", path=path)
    return {"name": name, "path": path}


def _diffusers_has(arch: str) -> bool:
    """Can THIS process train that architecture, or does it need the overlay?"""
    try:
        from codai.api.lora_archs import load_classes
        load_classes(arch)
        return True
    except Exception:
        return False


def _train_via_overlay(arch, req, base_path, images, instance_prompt,
                       steps, rank, resolution, lr, seed, device):
    """Train in the overlay venv (diffusers >= 0.40) as a subprocess.

    The reference images are written out as PNGs and the request is flattened to
    JSON — nothing is shared but the filesystem, because the child runs a different
    diffusers than we do. Progress is tailed back from the job file so the usual
    /v1/loras/progress polling keeps working.
    """
    import json as _json
    import subprocess
    import tempfile
    import threading

    from codai.api import h3_worker
    try:
        py = h3_worker.overlay_python({})
    except Exception as exc:
        raise HTTPException(status_code=501, detail=(
            f"{arch} LoRA training needs diffusers >= 0.40, which this process does "
            f"not have, and the overlay venv could not be built: {exc}"))

    work = tempfile.mkdtemp(prefix=f"lora_{arch}_")
    img_paths = []
    for i, im in enumerate(images):
        pth = os.path.join(work, f"ref_{i:03d}.png")
        im.save(pth)
        img_paths.append(pth)
    job_path = os.path.join(work, "job.json")
    _json.dump({
        "arch": arch, "base_path": base_path, "images": img_paths,
        "instance_prompt": instance_prompt, "steps": steps, "rank": rank,
        "resolution": resolution, "lr": lr, "seed": seed, "device": device,
        "request": {k: getattr(req, k) for k in
                    ("name", "num_frames", "quantize_4bit", "session", "target")
                    if hasattr(req, k)},
    }, open(job_path, "w"))

    script = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "tools", "lora_train_worker.py")
    print(f"  [lora][{arch}] training out-of-process in the overlay venv", flush=True)
    proc = subprocess.Popen([py, script, "--job", job_path],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    # The most expendable thing on the box, and one of the fattest: a training run
    # holds tens of GB for hours, and losing it costs a restart of one job rather
    # than every request in flight. Tell the OOM killer to come here first. Set
    # from the parent (raising another process's score needs no privilege) rather
    # than in a preexec_fn, which runs Python between fork and exec in a threaded
    # server. Best-effort: a training job that cannot be marked still trains.
    try:
        from codai.util import oom
        oom.mark("training", proc.pid)
    except Exception:
        pass

    def _pump():
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"  [lora][{arch}] {line}", flush=True)
    threading.Thread(target=_pump, daemon=True).start()

    # Mirror the child's progress onto ours so the normal polling endpoint works.
    prog_path = job_path + ".progress"
    seen = 0
    while proc.poll() is None:
        time.sleep(2.0)
        try:
            _check_train_cancel()
        except Exception:
            proc.terminate()
            raise
        try:
            lines = open(prog_path).read().splitlines() if os.path.exists(prog_path) else []
            for ln in lines[seen:]:
                try:
                    _set_progress(**{k: v for k, v in _json.loads(ln).items() if k != "at"})
                except Exception:
                    pass
            seen = len(lines)
        except Exception:
            pass

    res_path = job_path + ".result"
    if not os.path.exists(res_path):
        raise HTTPException(status_code=500, detail=(
            f"{arch} LoRA trainer exited (code {proc.returncode}) without a result"))
    out = _json.load(open(res_path))
    if not out.get("ok"):
        raise HTTPException(status_code=500,
                            detail=f"{arch} LoRA training failed: {out.get('error')}")
    return out["result"]


def _train_externally(arch: str, req, base_path, images, instance_prompt,
                      steps, rank, resolution, lr, seed):
    """Hand a job to an external trainer that owns this architecture.

    Only LongCat-Video today. Progress comes back on the same JSON-lines protocol the
    overlay trainer uses, so _set_progress and the job records need no special case.

    Note what is NOT done here: no images are re-encoded and no dataset is built in this
    process. SimpleTuner wants a captioned video dataset with its own backend config, and
    translating a list of stills into one silently would train something other than what
    was asked for. The dataset path is the caller's, and its absence is an error."""
    import tempfile
    from codai.api import longcat_worker

    dataset = (getattr(req, "dataset_config", None) or "").strip() \
        if hasattr(req, "dataset_config") else ""
    if not dataset:
        raise RuntimeError(
            f"{arch} LoRA training needs `dataset_config` — a path to a SimpleTuner "
            f"data-backend config describing your captioned video clips (its quickstart "
            f"suggests 50-100 clips of 10-30s). coderai will not invent one from the "
            f"reference images: a still-image dataset would train a different thing "
            f"than a video LoRA.")
    if not os.path.isfile(os.path.expanduser(dataset)):
        raise RuntimeError(f"dataset_config does not exist: {dataset}")

    # _lora_dir(name) already resolves the per-LoRA directory — and routes a
    # non-main DDP rank to scratch. Joining the name onto it again would have
    # written <loras>/<name>/<name> even once the missing argument was supplied.
    out_dir = _lora_dir(getattr(req, "name", "longcat-lora"))
    work = tempfile.mkdtemp(prefix="longcat-train-")
    job = {
        "arch": arch,
        "request": {k: v for k, v in vars(req).items()
                    if isinstance(v, (str, int, float, bool, type(None)))},
        "base_path": base_path,
        "output_dir": out_dir,
        "dataset_config": os.path.expanduser(dataset),
        "instance_prompt": instance_prompt,
        "steps": int(steps), "rank": int(rank), "lr": float(lr), "seed": int(seed),
        "resolution": getattr(req, "train_resolution", None) or "480x832",
        "num_frames": int(getattr(req, "num_frames", 93) or 93),
    }

    def _progress(**kw):
        _set_progress(step=kw.get("step"), total=kw.get("total"),
                      message=kw.get("message") or kw.get("status") or "")

    result = longcat_worker.train_lora(job, work, on_progress=_progress)
    return {"path": result.get("path") or out_dir, "steps": int(steps),
            "rank": int(rank), "arch": arch, "trainer": "simpletuner"}


def _train_flow_dit(arch: str, req, base_path, images, instance_prompt,
                    steps, rank, resolution, lr, seed, device):
    """LoRA for a flow-matching DiT — LTX-2, MiniMax-H3, Krea 2.

    All three are the same training problem as Wan: encode the references to VAE
    latents, encode the prompt once, add a rectified-flow adapter to the
    transformer's attention projections, and regress the velocity. Only the
    classes, the latent rank and the timestep shift differ, so those live in
    codai/api/lora_archs.py and this loop is shared rather than copied.

    NOT yet run against real weights for any of the three — the classes and the
    contracts are right, the arithmetic is the same as the proven Wan path, but
    treat the first real run as a debugging session.
    """
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig as PeftLoraConfig
    from peft.utils import get_peft_model_state_dict
    from torchvision import transforms
    from diffusers.utils import convert_state_dict_to_diffusers

    from codai.api.lora_archs import load_classes
    tr_cls, vae_cls, te_cls, tok_cls, spec = load_classes(arch)

    name = req.name
    torch.manual_seed(seed)
    compute_dtype = torch.bfloat16
    quantize = bool(getattr(req, "quantize_4bit", True))
    is_video = spec["kind"] == "video"
    num_frames = max(1, int(getattr(req, "num_frames", 1) or 1)) if is_video else 1

    q_cfg = None
    if quantize:
        try:
            from diffusers import BitsAndBytesConfig as _DiffBnb
            q_cfg = _DiffBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=compute_dtype)
        except Exception as e:
            print(f"  [lora][{arch}] 4-bit unavailable ({e}); loading in bf16")

    _set_progress(status="preparing", message=f"loading {spec['label']}: {base_path}")
    vae = vae_cls.from_pretrained(base_path, subfolder="vae", torch_dtype=torch.float32)
    vae.requires_grad_(False)
    vae.eval()
    tokenizer = tok_cls.from_pretrained(base_path, subfolder="tokenizer")
    text_encoder = te_cls.from_pretrained(base_path, subfolder="text_encoder",
                                          torch_dtype=compute_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    tr_kwargs = dict(subfolder="transformer", torch_dtype=compute_dtype)
    if q_cfg is not None:
        tr_kwargs["quantization_config"] = q_cfg
    transformer = tr_cls.from_pretrained(base_path, **tr_kwargs)
    if q_cfg is None:
        transformer = transformer.to(device)

    # ── 1. references -> latents ──────────────────────────────────────────────
    _set_progress(status="preparing", message="encoding reference images (VAE)")
    vae.to(device)
    spatial = (resolution // 32) * 32          # every one of these VAEs likes /32
    tfm = transforms.Compose([
        transforms.Resize(spatial, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(spatial),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    scale = float(getattr(getattr(vae, "config", None), "scaling_factor", 1.0) or 1.0)
    latents_list = []
    with torch.no_grad():
        for img in images:
            px = tfm(img)
            if is_video:
                vid = px.unsqueeze(0).unsqueeze(2)              # [1,3,1,H,W]
                if num_frames > 1:
                    vid = vid.repeat(1, 1, num_frames, 1, 1)
                sample = vid
            else:
                sample = px.unsqueeze(0)                        # [1,3,H,W]
            enc = vae.encode(sample.to(device, dtype=torch.float32))
            lat = enc.latent_dist.sample() if hasattr(enc, "latent_dist") else enc.latents
            latents_list.append((lat * scale).to(compute_dtype).cpu())
    vae.to("cpu")
    _free_train_vram()
    if not latents_list:
        raise HTTPException(status_code=400, detail="No reference images could be encoded")

    # ── 2. prompt -> one cached embedding ─────────────────────────────────────
    text_encoder.to(device)
    with torch.no_grad():
        tok = tokenizer(instance_prompt, padding="max_length",
                        max_length=int(spec.get("max_text_len", 256)),
                        truncation=True, return_tensors="pt")
        ids = tok.input_ids.to(device)
        mask = tok.attention_mask.to(device)
        out = text_encoder(ids, attention_mask=mask)
        enc = getattr(out, "last_hidden_state", None)
        if enc is None:
            enc = out[0]
        encoder_hidden_states = (enc * mask.unsqueeze(-1)).to(compute_dtype).cpu()
    text_encoder.to("cpu")
    _free_train_vram()

    # ── 3. adapter ────────────────────────────────────────────────────────────
    lora_cfg = PeftLoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                              target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    transformer.requires_grad_(False)
    _ensure_peft_awq_compat()
    transformer.add_adapter(lora_cfg, adapter_name="default")
    try:
        transformer.enable_gradient_checkpointing()
    except Exception:
        pass
    transformer.train()
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    if not lora_params:
        raise HTTPException(status_code=500,
                            detail=f"{arch} LoRA: no trainable adapter params were created")
    _ddp.broadcast_params(lora_params)      # all ranks start from rank 0's init
    optimizer = torch.optim.AdamW(lora_params, lr=lr)

    start_step = 0
    _ck = (_load_train_state(name, base_path=base_path, target=arch, rank=rank,
                             session=getattr(req, "session", None))
           if _resume_allowed(req) else None)
    if _ck:
        try:
            _apply_peft_checkpoint(name, "t", transformer)
            start_step = int(_ck["step"])
            _set_progress(step=start_step, message=f"resuming from step {start_step}/{steps}")
        except Exception as exc:
            print(f"  [lora][{arch}] checkpoint ignored: {exc}")
            start_step = 0

    # ── 4. rectified-flow training ────────────────────────────────────────────
    shift = float(spec.get("shift", 3.0))
    num_train_t = 1000.0
    _set_progress(status="training", message=f"training ({spec['label']} LoRA)")
    n = len(latents_list)
    for step in range(start_step, steps):
        _check_train_cancel()
        x0 = latents_list[_ddp.index(step, n)].to(device, dtype=compute_dtype)
        noise = torch.randn_like(x0)
        u = torch.rand(1, device=device, dtype=torch.float32)
        sigma = (shift * u) / (1.0 + (shift - 1.0) * u)
        s = sigma.view(*([-1] + [1] * (x0.dim() - 1)))
        x0f = x0.float()
        x_t = ((1.0 - s) * x0f + s * noise.float()).to(compute_dtype)
        target = (noise.float() - x0f).to(compute_dtype)
        timestep = (sigma * num_train_t).to(torch.float32)
        pred = transformer(hidden_states=x_t, timestep=timestep,
                           encoder_hidden_states=encoder_hidden_states.to(device),
                           return_dict=False)[0]
        loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
        loss.backward()
        _ddp.sync_grads(lora_params)      # average over the ranks
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()
        if (step + 1) % 10 == 0 or step + 1 == steps:
            _set_progress(step=step + 1, total=steps,
                          message=f"step {step + 1}/{steps} loss={loss.item():.4f}")
        if (step + 1) % 100 == 0:
            _save_train_checkpoint(name, {"step": step + 1, "base_path": base_path,
                                          "target": arch, "rank": rank,
                                          "session": getattr(req, "session", None)},
                                   {"t": get_peft_model_state_dict(transformer)})

    # ── 5. save ───────────────────────────────────────────────────────────────
    _set_progress(status="saving", message="saving adapter")
    save_dir = _lora_dir(name)
    os.makedirs(save_dir, exist_ok=True)
    layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(transformer))
    try:
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in layers.items()},
                  os.path.join(save_dir, "pytorch_lora_weights.safetensors"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Saving the adapter failed: {exc}")
    _write_meta(name, req, base_path, len(images), arch, instance_prompt)
    _clear_train_checkpoint(name)
    try:
        transformer.delete_adapters(["default"])
    except Exception:
        pass
    del transformer
    _free_train_vram()
    return {"name": name, "path": save_dir, "arch": arch, "steps": steps}


def _train_wan(req, base_path, images, instance_prompt,
               steps, rank, resolution, lr, seed, device):
    """Train a LoRA for a Wan video DiT directly, so it loads on the video
    pipeline. Stills are treated as 1-frame videos: VAE-encoded to latents, then a
    rectified-flow (flow-matching) loss trains PEFT LoRA adapters on the
    transformer attention layers. Wan2.2 A14B has two experts (transformer +
    transformer_2 routed by a noise boundary); both get adapters and are trained
    on the timestep range they own. The base is quantized to 4-bit (QLoRA) when
    requested so a 14B model's LoRA fits on a consumer GPU.
    """
    import os as _os
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig as PeftLoraConfig
    from peft.utils import get_peft_model_state_dict
    from torchvision import transforms
    try:
        from diffusers import (AutoencoderKLWan, WanTransformer3DModel,
                               FlowMatchEulerDiscreteScheduler, WanPipeline)
        from diffusers.utils import convert_state_dict_to_diffusers
        from transformers import UMT5EncoderModel, AutoTokenizer
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=("Wan video LoRA training needs a diffusers build with Wan "
                    f"support (AutoencoderKLWan/WanTransformer3DModel): {e}"))

    name = req.name
    torch.manual_seed(seed)
    compute_dtype = torch.bfloat16
    quantize = bool(getattr(req, "quantize_4bit", True))
    num_frames = max(1, int(getattr(req, "num_frames", 1) or 1))

    q_cfg = None
    if quantize:
        try:
            from diffusers import BitsAndBytesConfig as _DiffBnb
            q_cfg = _DiffBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=compute_dtype)
        except Exception as e:
            print(f"  [lora][wan] 4-bit unavailable ({e}); loading in bf16")
            q_cfg = None

    def _load_transformer(subfolder):
        kw = dict(subfolder=subfolder, torch_dtype=compute_dtype)
        if q_cfg is not None:
            kw["quantization_config"] = q_cfg
        tr = WanTransformer3DModel.from_pretrained(base_path, **kw)
        if q_cfg is None:
            tr = tr.to(device)
        return tr

    def _build_components():
        # Built once per base and cached: VAE/tokenizer/text-encoder live on CPU
        # between jobs (moved to GPU only while encoding); experts stay on GPU.
        _set_progress(status="preparing", message=f"loading Wan VAE: {base_path}")
        _vae = AutoencoderKLWan.from_pretrained(base_path, subfolder="vae",
                                                torch_dtype=torch.float32)
        _vae.requires_grad_(False)
        _vae.eval()
        _set_progress(status="preparing", message="loading UMT5 text encoder")
        _tok = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer")
        _te = UMT5EncoderModel.from_pretrained(
            base_path, subfolder="text_encoder", torch_dtype=compute_dtype)
        _te.requires_grad_(False)
        _te.eval()
        _set_progress(
            status="preparing",
            message=f"loading Wan transformer{' (4-bit)' if quantize else ''}")
        exp = [("transformer", _load_transformer("transformer"))]
        if _os.path.isdir(_os.path.join(base_path, "transformer_2")):
            exp.append(("transformer_2", _load_transformer("transformer_2")))
        b = getattr(exp[0][1].config, "boundary_ratio", None)
        return {"vae": _vae, "tokenizer": _tok, "text_encoder": _te,
                "experts": exp, "boundary": b}

    # Reuse the whole Wan stack across consecutive trainings against the same base.
    comps = _acquire_wan_components(base_path, quantize, _build_components)
    vae = comps["vae"]
    tokenizer = comps["tokenizer"]
    text_encoder = comps["text_encoder"]
    experts, boundary = comps["experts"], comps["boundary"]

    # ── 1. VAE (3D): encode each still as a 1-frame video latent ───────────────
    _set_progress(status="preparing", message="encoding reference images (VAE)")
    vae.to(device)
    z_dim = int(vae.config.z_dim)
    lat_mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(device)
    lat_std = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(device)
    spatial = (resolution // 16) * 16  # Wan VAE prefers multiples of 16
    tfm = transforms.Compose([
        transforms.Resize(spatial, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(spatial),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    latents_list = []
    with torch.no_grad():
        for img in images:
            px = tfm(img)                              # [3,H,W] in [-1,1]
            vid = px.unsqueeze(0).unsqueeze(2)         # [1,3,1,H,W]
            if num_frames > 1:
                vid = vid.repeat(1, 1, num_frames, 1, 1)
            vid = vid.to(device, dtype=torch.float32)
            lat = vae.encode(vid).latent_dist.sample()  # [1,z,t,h,w]
            lat = (lat - lat_mean) / lat_std
            latents_list.append(lat.to(compute_dtype).cpu())
    vae.to("cpu")                                       # keep cached, free its VRAM
    _free_train_vram()

    # ── 2. Text encoder (UMT5): encode the instance prompt once ────────────────
    _set_progress(status="preparing", message="encoding prompt (UMT5)")
    text_encoder.to(device)
    with torch.no_grad():
        tok = tokenizer(instance_prompt, padding="max_length", max_length=512,
                        truncation=True, return_tensors="pt")
        ids = tok.input_ids.to(device)
        mask = tok.attention_mask.to(device)
        enc = text_encoder(ids, attention_mask=mask).last_hidden_state
        encoder_hidden_states = (enc * mask.unsqueeze(-1)).to(compute_dtype).cpu()
    text_encoder.to("cpu")                              # keep cached, free its VRAM
    _free_train_vram()

    # ── 3. Transformer expert(s) + LoRA ───────────────────────────────────────
    lora_cfg = PeftLoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    lora_params = []
    hook_handles = []
    for _, tr in experts:
        tr.requires_grad_(False)
        _ensure_peft_awq_compat()
        tr.add_adapter(lora_cfg, adapter_name="default")
        try:
            tr.enable_gradient_checkpointing()
            # Make checkpointing track grads through a frozen/quantized base.
            # Keep the handle so the hook is removed at job end — the transformer
            # is cached and reused, so hooks must not accumulate across jobs.
            hook_handles.append(tr.patch_embedding.register_forward_hook(
                lambda m, i, o: o.requires_grad_(True)))
        except Exception:
            pass
        tr.train()
        lora_params += [p for p in tr.parameters() if p.requires_grad]
    if not lora_params:
        raise HTTPException(status_code=500,
                            detail="Wan LoRA: no trainable adapter params were created")
    _ddp.broadcast_params(lora_params)      # all ranks start from rank 0's init
    optimizer = torch.optim.AdamW(lora_params, lr=lr)

    # Resume from a mid-training checkpoint if one survives a prior restart. This
    # is the big win for Wan: the A14B base reload alone is ~27 min, and training
    # runs for hours — a reboot otherwise throws all of it away.
    start_step = 0
    _ck = _load_train_state(name, base_path=base_path, target="video", rank=rank,
                            session=getattr(req, "session", None)) if _resume_allowed(req) else None
    if _ck:
        try:
            _apply_peft_checkpoint(name, "t", experts[0][1])
            if len(experts) > 1:
                _apply_peft_checkpoint(name, "t2", experts[1][1])
            start_step = int(_ck["step"])
            _set_progress(step=start_step,
                          message=f"resuming from step {start_step}/{steps}")
            _dbg_lora(f"resumed Wan '{name}' from checkpoint step {start_step}")
        except Exception as e:
            print(f"  [lora] could not resume Wan '{name}': {e}; starting fresh")
            start_step = 0

    try:
        sched = FlowMatchEulerDiscreteScheduler.from_pretrained(base_path, subfolder="scheduler")
        shift = float(getattr(sched.config, "shift", 3.0) or 3.0)
        num_train_t = float(getattr(sched.config, "num_train_timesteps", 1000) or 1000)
    except Exception:
        shift, num_train_t = 3.0, 1000.0

    def _pick_expert(t_val):
        if len(experts) > 1 and boundary is not None:
            # High-noise expert (index 0) owns t >= boundary; low-noise is index 1.
            return experts[0][1] if t_val >= float(boundary) else experts[1][1]
        return experts[0][1]

    def _tr_in_channels(tr):
        """Channels the transformer's patch_embedding expects."""
        try:
            c = getattr(getattr(tr, "config", None), "in_channels", None)
            if c:
                return int(c)
        except Exception:
            pass
        try:
            return int(tr.patch_embedding.weight.shape[1])
        except Exception:
            return None

    # I2V transformers (in_channels=36 = 16 noise + 16 image-cond + 4 mask) expect a
    # wider input than the bare VAE latent (z_dim, typically 16). We have no init-image
    # conditioning during identity-LoRA training, so feed the well-defined "no condition"
    # case: zero-pad the noise latent up to in_channels (zero image latents + zero mask).
    # The transformer still outputs z_dim channels, so the flow-matching target/loss is
    # unchanged, and the attention-projection LoRA it trains applies during real i2v too.
    _want_in_ch = _tr_in_channels(experts[0][1])
    if _want_in_ch and _want_in_ch > z_dim:
        _dbg_lora(f"Wan I2V transformer wants {_want_in_ch} in-channels; zero-padding "
                  f"the {z_dim}-channel latent (no-init-image training regime)")

    _set_progress(status="training", message="training (Wan video LoRA)")
    n = len(latents_list)
    for step in range(start_step, steps):
        _check_train_cancel()
        x0 = latents_list[_ddp.index(step, n)].to(device, dtype=compute_dtype)
        noise = torch.randn_like(x0)
        # Rectified-flow timestep with Wan resolution shift applied to sigma.
        # Compute the interpolation in fp32 for stability, then cast x_t back to
        # the model's compute dtype — the patch-embedding Conv3d is NOT quantized
        # (bnb only touches Linear), so it needs inputs in compute_dtype.
        u = torch.rand(1, device=device, dtype=torch.float32)
        sigma = (shift * u) / (1.0 + (shift - 1.0) * u)
        s = sigma.view(-1, 1, 1, 1, 1)
        x0f = x0.float()
        x_t = ((1.0 - s) * x0f + s * noise.float()).to(compute_dtype)
        target = (noise.float() - x0f).to(compute_dtype)  # flow-matching velocity
        timestep = (sigma * num_train_t).to(torch.float32)  # cast internally by Wan
        tr = _pick_expert(float(sigma.item()))
        # Pad to the transformer's expected in_channels for I2V models (see note above).
        x_in = x_t
        want_ch = _tr_in_channels(tr) or x_t.shape[1]
        if want_ch > x_t.shape[1]:
            pad = torch.zeros((x_t.shape[0], want_ch - x_t.shape[1], *x_t.shape[2:]),
                              device=x_t.device, dtype=x_t.dtype)
            x_in = torch.cat([x_t, pad], dim=1)
        pred = tr(hidden_states=x_in, timestep=timestep,
                  encoder_hidden_states=encoder_hidden_states.to(device),
                  return_dict=False)[0]
        loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
        loss.backward()
        _ddp.sync_grads(lora_params)      # average over the ranks
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Report progress every step so the web UI bar advances smoothly (cheap
        # dict update); throttle the terminal debug print to every 10 steps.
        _set_progress(step=step + 1,
                      message=f"step {step+1}/{steps} loss={loss.item():.4f}")
        if step % 10 == 0 or step == steps - 1:
            _dbg_lora(f"Wan step {step+1}/{steps} loss={loss.item():.4f}")
        if (step + 1) % _CKPT_EVERY == 0 and step + 1 < steps:
            _ckpt_states = {"t": get_peft_model_state_dict(experts[0][1])}
            if len(experts) > 1:
                _ckpt_states["t2"] = get_peft_model_state_dict(experts[1][1])
            _save_train_checkpoint(name,
                {"name": name, "base_path": base_path, "target": "video",
                 "rank": rank, "step": step + 1, "total": steps, "seed": seed,
                 "session": getattr(req, "session", None)},
                _ckpt_states)
        try:
            from codai.models.thermal import checkpoint as _thermal_checkpoint
            _thermal_checkpoint(context="lora-train", throttle_seconds=2.0)
        except Exception:
            pass

    # ── 4. Save (transformer + optional transformer_2 LoRA layers) ─────────────
    _set_progress(status="saving", message="saving Wan LoRA weights")
    save_dir = _lora_dir(name)
    os.makedirs(save_dir, exist_ok=True)
    save_kwargs = {"transformer_lora_layers":
                   convert_state_dict_to_diffusers(get_peft_model_state_dict(experts[0][1]))}
    if len(experts) > 1:
        t2 = convert_state_dict_to_diffusers(get_peft_model_state_dict(experts[1][1]))
        try:
            WanPipeline.save_lora_weights(save_directory=save_dir,
                                          transformer_2_lora_layers=t2,
                                          safe_serialization=True, **save_kwargs)
        except TypeError:
            print("  [lora][wan] this diffusers WanPipeline.save_lora_weights has no "
                  "transformer_2 arg — saving high-noise expert LoRA only")
            WanPipeline.save_lora_weights(save_directory=save_dir,
                                          safe_serialization=True, **save_kwargs)
    else:
        WanPipeline.save_lora_weights(save_directory=save_dir,
                                      safe_serialization=True, **save_kwargs)
    _write_meta(name, req, base_path, len(images), "wan", instance_prompt)
    _clear_train_checkpoint(name)

    # ── 5. Tear down THIS job's adapter, but KEEP the transformer(s) cached so a
    # next training against the same base skips the (very slow) reload. Remove the
    # gradient-checkpoint hooks and the adapter; the experts stay resident on the
    # GPU (the external releaser drops them if a generation needs the VRAM).
    for h in hook_handles:
        try:
            h.remove()
        except Exception:
            pass
    for _, tr in experts:
        try:
            tr.delete_adapters("default")
        except Exception:
            pass
        try:
            tr.eval()
        except Exception:
            pass
    try:
        del optimizer, lora_params, latents_list, encoder_hidden_states
    except Exception:
        pass
    _free_train_vram()

    path = _lora_weight_file(name) or save_dir
    _set_progress(active=False, status="done", message="done", path=path)
    return {"name": name, "path": path}


def _write_meta(name, req, base_path, n_images, arch, instance_prompt):
    meta = {
        "name": name,
        "base_model": req.base_model,
        "train_base_model": (req.train_base_model
                             or _configured_train_base(req.base_model)
                             or req.base_model),
        "base_path": base_path,
        "arch": arch,
        "instance_prompt": instance_prompt,
        "steps": req.steps,
        "rank": req.rank,
        "resolution": req.resolution,
        "num_images": n_images,
        "created_at": int(time.time()),
    }
    try:
        with open(os.path.join(_lora_dir(name), "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass


# ── Endpoints ─────────────────────────────────────────────────────────────────

# Scheduler model key for training. A constant key makes the central queue
# serialize trainings (one at a time, protecting the shared base cache) while
# still admitting them through the same scheduler as every other request.
_TRAIN_MODEL_KEY = "lora-train"


def active_training_model() -> Optional[str]:
    """The base model of the training job currently running on the GPU, or None.

    The trainer holds its base pipeline outside the model manager, so this is the
    only way the engine status can report that the GPU is busy with training rather
    than showing 0 loaded models."""
    if _train_lock.locked():
        return _active_train_model
    return None


def _ddp_is_main_block(req) -> bool:
    d = getattr(req, "distributed", None)
    return not d or int(d.get("rank") or 0) == 0


def _dispatch_peers(req: LoraTrainRequest, job_id: Optional[str]) -> list:
    """Send this job to each node in ``req.nodes`` as a peer rank, with the
    images inlined (a character or environment profile exists only here),
    and mark this request as rank 0. Returns the peers with their job ids —
    what a cancel or a failure here must reach."""
    import base64
    import io
    from codai.api import pod_http
    from codai.cluster.rpc import advertise_host
    nodes = _ddp.parse_nodes(req.nodes)
    peers = _ddp.resolve_peers(nodes)
    if not peers:
        return []
    adv = ""
    try:
        from codai.admin.routes import config_manager
        adv = getattr(getattr(config_manager.config, "cluster", None), "advertise_host", "") or ""
    except Exception:
        pass
    master = advertise_host(adv)
    port = _ddp.free_port()
    world = 1 + len(peers)
    backend = (req.dist_backend or "").strip().lower() or "gloo"
    # The peers receive the images themselves: a profile name means nothing
    # on another machine.
    images = _gather_images(req)
    if not images:
        raise HTTPException(status_code=400,
                            detail="No training images (provide `character` or `images`)")
    inline = []
    for im in images:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        inline.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    payload = req.model_dump(exclude_none=True)
    for k in ("nodes", "character", "environment", "session", "wait"):
        payload.pop(k, None)
    payload["images"] = inline
    payload["wait"] = False
    req.distributed = {"rank": 0, "world_size": world, "master_addr": master,
                       "master_port": port, "backend": backend, "timeout_s": 600}
    out = []
    for rank, peer in enumerate(peers, 1):
        body = dict(payload)
        body["distributed"] = {**req.distributed, "rank": rank}
        headers = {"Content-Type": "application/json"}
        if peer.get("api_key"):
            headers["Authorization"] = f"Bearer {peer['api_key']}"
        try:
            r = pod_http.post(peer["url"] + "/v1/loras/train", json=body, headers=headers,
                              timeout=120, verify=(False if peer.get("verify") == "off" else True))
            if r.status_code != 200:
                raise RuntimeError(f"{r.status_code} {(r.text or '')[:200]}")
            pj = (r.json() or {}).get("job_id")
        except Exception as exc:
            _cancel_peers(out)
            raise HTTPException(status_code=502,
                                detail=f"training node {peer['name']} refused the job: {exc}")
        out.append({**peer, "job_id": pj, "rank": rank})
        print(f"  [lora][ddp] rank {rank} → {peer['name']} ({peer['url']}) job {pj}", flush=True)
    print(f"  [lora][ddp] rank 0 here, rendezvous {master}:{port}, world {world}, {backend}",
          flush=True)
    _set_progress(message=f"training on {world} machines: " +
                  ", ".join(["local"] + [p["name"] for p in peers]))
    return out


def _cancel_peers(peers) -> None:
    """Best effort: a peer left waiting on an all-reduce holds its card until
    the group times out; tell it to stop now."""
    from codai.api import pod_http
    for p in peers or []:
        if not p.get("job_id"):
            continue
        headers = {"Authorization": f"Bearer {p['api_key']}"} if p.get("api_key") else {}
        try:
            pod_http.post(f"{p['url']}/v1/loras/jobs/{p['job_id']}/cancel", headers=headers,
                          timeout=15, verify=(False if p.get("verify") == "off" else True))
        except Exception:
            pass


def _train_lora_blocking(req: LoraTrainRequest, job_id: Optional[str] = None) -> dict:
    """Run one training job to completion (called inside a worker thread).

    Holds _train_lock for the job's duration so a second training never overlaps
    (also the signal _release_base_cache uses to know a job is in flight). The
    central queue already serializes us, so this acquire returns immediately.
    `_active_job_id` is set so live progress mirrors into this job's record (and
    only this job's) for its owner to poll.
    """
    global _active_job_id, _active_train_model
    _train_lock.acquire()
    _active_job_id = job_id
    _active_train_model = getattr(req, "base_model", "") or None
    # Live cancellable task (id == job id). Registered here, when the job actually
    # starts on the GPU, so its progress mirrors via _set_progress.
    if job_id:
        task_registry.register("training", title=getattr(req, "name", "") or "",
                               model=getattr(req, "base_model", "") or "",
                               total=getattr(req, "steps", 0) or 0,
                               job_id=job_id, task_id=job_id, restartable=True,
                               status="running")
        task_registry.start(job_id)
        # A job cancelled while still queued aborts before any GPU work.
        try:
            _check_train_cancel()
        except TaskCancelled:
            _update_job(job_id, status="cancelled", active=False, force=True,
                        message="cancelled before start")
            _active_job_id = None
            _train_lock.release()
            raise
    # Reserve the GPU for the whole training: block ordinary model loads on this
    # engine (image/video) AND on co-located siblings (gguf text), so nothing
    # reloads mid-training and OOMs the trainer. Released in the finally below.
    try:
        from codai.models import gpu_lock
        gpu_lock.reserve(f"lora-training:{getattr(req, 'name', '') or '?'}")
    except Exception as _gle:
        print(f"  [lora] could not reserve GPU lock: {_gle}")
    peers = None
    try:
        # Several machines: this engine is rank 0; the peers get the same job.
        if not getattr(req, "nodes", None) and not getattr(req, "distributed", None):
            _cfg_nodes = _configured_train_nodes(getattr(req, "base_model", "") or "")
            if _cfg_nodes:
                req.nodes = _cfg_nodes
        if getattr(req, "nodes", None) and not getattr(req, "distributed", None):
            peers = _dispatch_peers(req, job_id)
        with _ddp.context(getattr(req, "distributed", None)):
            result = _train_lora_sync(req)
        if not _ddp_is_main_block(req):
            # A peer rank: rank 0 holds the LoRA; nothing of ours outlives the job.
            result = {"name": req.name, "path": None,
                      "rank": int((req.distributed or {}).get("rank") or 0)}
        if job_id:
            _update_job(job_id, status="done", active=False, force=True,
                        message="done", path=result.get("path"))
            task_registry.finish(job_id, "done")
        return result
    except TaskCancelled:
        _cancel_peers(peers)
        # Cooperative cancel from the training loop — not an error.
        try:
            _set_progress(active=False, status="cancelled", message="cancelled")
        except Exception:
            pass
        if job_id:
            _update_job(job_id, status="cancelled", active=False, force=True,
                        message="cancelled by user")
            task_registry.finish(job_id, "cancelled")
        # The base/wan caches may hold a half-applied adapter — drop them.
        _drop_base_cache()
        _drop_wan_cache()
        raise
    except Exception as e:
        _cancel_peers(peers)
        import traceback
        traceback.print_exc()
        # On error the base may be in a half-moved / inconsistent state — drop the
        # caches entirely (and reclaim their VRAM) rather than reuse them.
        try:
            _set_progress(active=False, status="error", message="training failed")
        except Exception:
            pass
        if job_id:
            _update_job(job_id, status="error", active=False, force=True,
                        message=f"training failed: {e}"[:300])
            task_registry.finish(job_id, "error", f"{e}"[:200])
        _drop_base_cache()
        _drop_wan_cache()
        raise
    finally:
        try:
            from codai.models import gpu_lock
            gpu_lock.release()
        except Exception:
            pass
        _active_job_id = None
        _active_train_model = None
        if job_id:
            _cancel_requested.discard(job_id)
            _force_resume_jobs.discard(job_id)
        _train_lock.release()


async def _run_train_job(req: LoraTrainRequest, job_id: str) -> dict:
    """Acquire a scheduler slot, then run the blocking job in a worker thread.
    Used for both the inline (wait=True) and detached (wait=False) paths so the
    queue serialization and job bookkeeping are identical."""
    import asyncio
    request_id = f"lora-train-{job_id}"
    lease = await queue_manager.acquire(request_id, _TRAIN_MODEL_KEY)
    try:
        return await asyncio.to_thread(_train_lora_blocking, req, job_id)
    finally:
        await queue_manager.release(lease)


@router.post("/v1/loras/train", summary="Train a new LoRA")
async def train_lora(req: LoraTrainRequest, _auth=Depends(_require_api_auth)):
    """Train a LoRA. Admitted through the central request scheduler so concurrent
    trainings queue and run one-at-a-time alongside all other model requests.

    With `wait=false` the call returns immediately with a `job_id`; the client
    polls /v1/loras/progress?job=<job_id>. The job keeps running on the server
    regardless of the client — so a client that dies/restarts can re-attach by
    job_id (or list its session's jobs with ?session=<token>) rather than
    restarting the training."""
    import asyncio
    import uuid
    if not req.name or '/' in req.name or '..' in req.name:
        raise HTTPException(status_code=400, detail="Invalid LoRA name")
    if not req.base_model:
        raise HTTPException(status_code=400, detail="base_model is required")

    session = req.session or f"sess-{uuid.uuid4().hex[:12]}"
    req.session = session
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    _new_job(job_id, session=session, name=req.name, base_model=req.base_model,
             target=(req.target or "image"), status="queued", active=True,
             step=0, total=req.steps or 0, message="queued",
             started_at=time.time(), path=None)
    # Persist the full request so the job can be restarted later (Tasks page)
    # without the original client.
    _save_job_request(job_id, req)

    if not req.wait:
        # Detach: run independently of this request's lifetime. Keep a strong
        # reference so the loop doesn't garbage-collect (cancel) the task.
        task = asyncio.create_task(_detached_train(req, job_id))
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)
        return {"ok": True, "async": True, "job_id": job_id, "session": session,
                "status": "queued"}

    try:
        result = await _run_train_job(req, job_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LoRA training failed: {e}")
    return {"ok": True, "job_id": job_id, "session": session, **result}


async def _detached_train(req: LoraTrainRequest, job_id: str) -> None:
    """Background runner for wait=false jobs. Errors are recorded on the job
    record (the client polls for them); nothing is raised to a waiting request."""
    try:
        await _run_train_job(req, job_id)
    except TaskCancelled:
        # Already recorded as 'cancelled' by _train_lora_blocking — don't clobber.
        pass
    except Exception as e:
        _update_job(job_id, status="error", active=False, force=True,
                    message=f"training failed: {e}"[:300])


@router.post("/v1/loras/jobs/{job_id}/cancel", summary="Cancel a training job")
async def cancel_train_job(job_id: str, _auth=Depends(_require_api_auth)):
    """Stop a queued or running training job — also what the rank-0 machine
    of a multi-node training calls on its peers when it aborts."""
    if not cancel_job(job_id):
        raise HTTPException(status_code=404, detail="unknown job")
    return {"ok": True, "job_id": job_id}


@router.get("/v1/loras/progress", summary="LoRA training progress")
async def lora_progress(job: Optional[str] = None, session: Optional[str] = None):
    """Training progress.

    - `?job=<job_id>`  → that job's record (a client polls only its own job).
    - `?session=<tok>` → the most recent job for that session (recovery after a
      client restart that lost the job_id).
    - neither          → the global active-job snapshot (back-compat).
    """
    if job:
        with _jobs_lock:
            rec = _jobs.get(job)
            rec = dict(rec) if rec else None
        if rec is None:
            return {"active": False, "status": "unknown", "job_id": job,
                    "name": None, "step": 0, "total": 0,
                    "message": "no such job"}
        return rec
    if session:
        with _jobs_lock:
            mine = [dict(r) for r in _jobs.values() if r.get("session") == session]
        if not mine:
            return {"active": False, "status": "unknown", "session": session,
                    "name": None, "step": 0, "total": 0,
                    "message": "no jobs for session"}
        # Prefer an active job; otherwise the most recently updated.
        active = [r for r in mine if r.get("status") in _JOB_ACTIVE_STATES]
        pick = (active or mine)
        pick.sort(key=lambda r: r.get("updated_at", 0))
        return pick[-1]
    with _progress_lock:
        return dict(_progress)


@router.post("/v1/loras/upload", summary="Upload LoRA weights (content-addressed)")
async def upload_lora(request: Request, _auth=Depends(_require_api_auth)):
    """Upload a LoRA file into the content-addressed store so a remote client can
    use it without sharing a filesystem.

    Accepts the file as multipart/form-data (field `file`), as JSON
    {"file": "<base64>"} (optionally a data: URI), or as a raw request body.
    Returns {id: "sha256:<hex>", bytes, existed}. Reference the returned id as
    {"id": "<id>", "weight": ...} in image/video generation requests."""
    ctype = request.headers.get("content-type", "")
    data = b""
    if "multipart/form-data" in ctype:
        form = await request.form()
        up = form.get("file")
        if up is None:
            raise HTTPException(status_code=400, detail="multipart upload missing 'file' field")
        data = await up.read() if hasattr(up, "read") else bytes(up)
    elif "application/json" in ctype:
        body = await request.json()
        blob = body.get("file") or body.get("data")
        if not blob:
            raise HTTPException(status_code=400, detail="JSON upload missing 'file'/'data' (base64)")
        import base64
        try:
            data = base64.b64decode(_strip_data_uri(str(blob)))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid base64: {e}")
    else:
        data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    h, existed = save_lora_blob(data)
    return {"id": f"sha256:{h}", "bytes": len(data), "existed": existed}


@router.get("/v1/loras/blob/{hash}", summary="Check an uploaded LoRA blob exists")
async def lora_blob_info(hash: str, _auth=Depends(_require_api_auth)):
    """Existence check for an uploaded LoRA blob. 200 with metadata when present,
    404 when absent — lets a client skip re-uploading a file the server already
    has. `hash` may be a hex sha256 or 'sha256:<hex>'."""
    if not lora_blob_exists(hash):
        raise HTTPException(status_code=404, detail="blob not found")
    p = _lora_blob_path(hash)
    if os.path.isdir(p):
        size = sum(os.path.getsize(os.path.join(r, f))
                   for r, _d, fs in os.walk(p) for f in fs)
        return {"id": f"sha256:{os.path.basename(p)}", "bytes": size,
                "exists": True, "unpacked": True}
    return {"id": f"sha256:{os.path.basename(p)[:-len('.safetensors')]}",
            "bytes": os.path.getsize(p), "exists": True}


class LoraRegisterRequest(BaseModel):
    name: str = Field(..., description="The name requests will ask for.")
    base: str = Field(..., description="An existing model this adapter is trained on.")
    adapter: str = Field(..., description="A 'sha256:…' id from /v1/loras/upload, a "
                                          "HuggingFace repo id, or a path on the server.")
    weight: float = Field(1.0, gt=0, description="Scaling applied to the adapter.")
    pool: Optional[str] = Field(None, description="Pod pool to join. Defaults to the "
                                                  "base's, which is what puts the "
                                                  "adapter on the card already holding "
                                                  "the base weights.")
    model_config = ConfigDict(extra="allow", protected_namespaces=())


@router.post("/v1/loras/register", summary="Serve an uploaded adapter under its own name")
async def register_lora(body: LoraRegisterRequest, _auth=Depends(_require_api_auth)):
    """Turn an uploaded adapter into a model name this install will serve.

    `/v1/loras/upload` puts the bytes somewhere; this is what makes them
    answerable. Without it the last step was editing models.json by hand, which
    is not a step a team retraining adapters can automate.

    The adapter JOINS THE BASE'S POOL by default, which is the whole economy of
    it: the base weights are already loaded on a rented card, an adapter is tens
    of megabytes of deltas, and vLLM applies them per request. One GPU serves the
    base and every adapter of it, each under its own name, with no reload —
    see docs/runpod.md.

    Identity is (weights, name). An adapter and its base share a `path` by
    definition, so nothing here may key on the path alone; that mistake is why a
    pod once answered under an adapter's name with base weights.
    """
    name = (body.name or "").strip()
    base = (body.base or "").strip()
    adapter = (body.adapter or "").strip()
    if not name or not base or not adapter:
        raise HTTPException(status_code=400,
                            detail="name, base and adapter are all required")

    from codai.admin.routes import config_manager
    md = getattr(config_manager, "models_data", None)
    if not isinstance(md, dict):
        raise HTTPException(status_code=503, detail="no model catalogue is loaded")

    # The base, by any name it answers to.
    base_entry, section = None, "text_models"
    for sec, lst in md.items():
        if not isinstance(lst, list):
            continue
        for m in lst:
            if not isinstance(m, dict):
                continue
            if base in (str(m.get("id") or ""), str(m.get("alias") or ""),
                        str(m.get("path") or "")):
                base_entry, section = m, sec
                break
        if base_entry is not None:
            break
    if base_entry is None:
        raise HTTPException(status_code=404,
                            detail=f"no model named {base!r} — register the base first")

    # The bytes have to be HERE before a pod can be told the id: the id is all
    # that travels, and a pod sent an id nobody holds serves the base instead.
    if adapter.startswith("sha256:") and not lora_blob_exists(adapter):
        raise HTTPException(
            status_code=404,
            detail=f"{adapter} is not in the blob store — POST it to "
                   f"/v1/loras/upload first (a PEFT adapter directory travels "
                   f"as a tar; it is unpacked here)")

    block = base_entry.get("runpod")
    pool = (body.pool or "").strip()
    if not pool and isinstance(block, dict):
        pool = str(block.get("pool") or "").strip()

    entry = {"id": name, "enabled": True,
             "loras": [{"name": name, "path": adapter, "weight": float(body.weight)}]}
    if isinstance(block, dict):
        # The base's own placement, budgets and card choice: an adapter is
        # served by the same engine on the same pod, so inheriting is the only
        # answer that does not quietly rent a second card.
        rp = {k: v for k, v in block.items()
              if k not in ("served_model", "pool", "min_pods", "keep_warm")}
        rp["served_model"] = name
        if pool:
            rp["pool"] = pool
        entry["runpod"] = rp
        if str(base_entry.get("backend") or "").strip():
            entry["backend"] = base_entry["backend"]

    lst = md.setdefault(section, [])
    existing = next((m for m in lst if isinstance(m, dict)
                     and name in (str(m.get("id") or ""), str(m.get("alias") or ""))),
                    None)
    if existing is not None:
        if not existing.get("loras"):
            raise HTTPException(
                status_code=409,
                detail=f"{name!r} is already a model here and is not an adapter — "
                       f"pick another name rather than replace it")
        existing.clear()
        existing.update(entry)
        state = "updated"
    else:
        lst.append(entry)
        state = "registered"
    config_manager.save_models()

    # Usable now, not after a restart. A pod already holding the base is handed
    # the adapter by the engine that serves it (vllm_worker), so this does not
    # rent anything and does not reload the base model.
    try:
        from codai.api.models_transfer import register_model_runtime
        register_model_runtime(entry, section)
    except Exception as exc:
        print(f"[lora] {name} saved but not live until restart: {exc}", flush=True)

    return {"ok": True, "state": state, "name": name, "base": base,
            "adapter": adapter, "pool": pool or None, "section": section}


@router.get("/v1/loras", summary="List registered LoRAs")
async def list_loras(_auth=Depends(_require_api_auth)):
    """List every trained/registered LoRA in the registry.

    Returns one entry per LoRA with its `name`, on-disk weight `path` and any saved
    metadata (base model, target, training params). These names can be referenced from
    image/video requests via `loras: [{"id": "name:<name>"}]`.
    """
    out = []
    d = _loras_dir()
    if os.path.isdir(d):
        for name in sorted(os.listdir(d)):
            wf = _lora_weight_file(name)
            if not wf:
                continue
            meta = {}
            mp = os.path.join(_lora_dir(name), "meta.json")
            if os.path.isfile(mp):
                try:
                    with open(mp) as f:
                        meta = json.load(f)
                except Exception:
                    pass
            out.append({"name": name, "path": wf, "source": "trained", **meta})
    # …and the adapters REGISTERED as servable names (/v1/loras/register).
    # Those live in the model catalogue, not in the training registry, so this
    # listed nothing after a successful registration — which reads exactly like
    # the registration having failed.
    seen = {e["name"] for e in out}
    try:
        from codai.admin.routes import config_manager
        md = getattr(config_manager, "models_data", None) or {}
    except Exception:
        md = {}
    for lst in md.values():
        if not isinstance(lst, list):
            continue
        for entry in lst:
            if not isinstance(entry, dict) or not entry.get("loras"):
                continue
            nm = (str(entry.get("id") or "").strip()
                  or str(entry.get("alias") or "").strip())
            if not nm or nm in seen:
                continue
            seen.add(nm)
            specs = entry["loras"] if isinstance(entry["loras"], list) else []
            first = specs[0] if specs and isinstance(specs[0], dict) else {}
            out.append({
                "name": nm, "source": "registered",
                "adapter": str(first.get("path") or first.get("source") or ""),
                "weight": float(first.get("weight", 1.0) or 1.0),
                "base": str(entry.get("path") or ""),
                "pool": str((entry.get("runpod") or {}).get("pool") or "") or None,
                "served": True,
            })
    return {"loras": out}


@router.get("/v1/loras/{name}", summary="Get a registered LoRA")
async def get_lora(name: str, _auth=Depends(_require_api_auth)):
    """Fetch one registered LoRA by name, including its weight path and metadata."""
    wf = _lora_weight_file(name)
    if not wf:
        raise HTTPException(status_code=404, detail=f"LoRA '{name}' not found")
    meta = {}
    mp = os.path.join(_lora_dir(name), "meta.json")
    if os.path.isfile(mp):
        try:
            with open(mp) as f:
                meta = json.load(f)
        except Exception:
            pass
    return {"name": name, "path": wf, **meta}


@router.delete("/v1/loras/{name}", summary="Delete a registered LoRA")
async def delete_lora(name: str, _auth=Depends(_require_api_auth)):
    """Delete a registered LoRA and all its files from the registry."""
    d = _lora_dir(name)
    if not os.path.isdir(d):
        raise HTTPException(status_code=404, detail=f"LoRA '{name}' not found")
    import shutil
    shutil.rmtree(d)
    return {"ok": True, "name": name}


# ═════════════════════════════════════════════════════════════════════════════
# Text (causal-LM) QLoRA
# ═════════════════════════════════════════════════════════════════════════════
#
# Its own route rather than a `target: "text"` on the one above. The bodies have
# almost nothing in common -- that one takes images, a resolution and a frame
# count; this one takes a JSONL of chat examples -- and one model carrying both
# would make every field conditional on the target, which is a schema that can
# no longer tell a client what to send.
#
# What IS shared is everything underneath: the job record, the session that
# survives a client restart, checkpoint resume, the Tasks page, cancellation and
# the adapter registry. Those are above the diffusion/text split, so a second
# subsystem would have been a second copy of them.

class TextLoraTrainRequest(BaseModel):
    """Train a LoRA on a causal LM from a JSONL of chat examples."""

    name: str = Field(..., description="Output adapter name")
    base_model: str = Field(..., description="Base model: a models.json key, an "
                                             "HF repo id, or a local path")
    # One of the two. A path for an install that has the file; an id for a
    # client that uploaded it from another machine.
    dataset: Optional[str] = Field(None, description="Server-side path to a JSONL")
    dataset_id: Optional[str] = Field(None, description="'sha256:…' from "
                                                        "POST /v1/loras/datasets")
    #: Field renaming, so a dataset is not regenerated to be trainable. Maps
    #: canonical name -> the name this file uses.
    field_map: Optional[Dict[str, str]] = Field(
        None, description='e.g. {"answer": "response", "weight": "w"}')

    rank: Optional[int] = 16
    alpha: Optional[int] = None            # None = 2x rank
    dropout: Optional[float] = 0.05
    learning_rate: Optional[float] = 1e-4
    max_len: Optional[int] = 8192
    accumulation_steps: Optional[int] = 8
    epochs: Optional[float] = 1.0
    enable_thinking: Optional[bool] = False
    seed: Optional[int] = 42

    #: Refuse rather than train on a dataset that turned out much smaller than
    #: the caller thinks. Skips are legitimate (an example over max_len cannot
    #: be trained) but a large fraction means the length limit, the template or
    #: the field mapping is wrong, and "it ran" would hide it.
    max_skipped_fraction: Optional[float] = 0.05
    #: Stop after the dataset pass and report what WOULD be trained. The cheap
    #: half of a run, and the one that catches a bad mask before a pod is rented.
    dry_run: Optional[bool] = False

    # --- distributed ------------------------------------------------------
    # The card, the image, the pod count and the money are the MODEL's
    # configuration (`lora_train` in models.json, edited on the Models page),
    # not this request's. Same rule /v1/runpod/scale states for itself: a
    # request cannot change a model's configuration, which is what makes this
    # route safe to expose. The three fields below may only NARROW it.
    #: Run distributed at all. Requires the model's `lora_train.enabled`.
    distributed: Optional[bool] = None
    #: How many pods this job wants. Clamped by the model's `max_pods`, the
    #: same way a scale request may only narrow a model's pool.
    pods: Optional[int] = None
    #: A ceiling this job accepts, narrowing the model's own.
    max_pods: Optional[int] = None
    #: Refuse to START below this many. Training a job that asked for sixteen
    #: pods on two is usually worse than not training: it bills for a result
    #: nobody wanted, and the round length was chosen for the fleet size.
    min_pods: Optional[int] = None
    #: More or fewer averaging rounds: duration, not fleet size or price.
    rounds: Optional[int] = None
    #: A LOWER $/hr ceiling than the model's, never a higher one.
    max_pod_hourly_usd: Optional[float] = None
    max_fleet_hourly_usd: Optional[float] = None

    #: Prove the whole path for the price of a few minutes of one pod: rent,
    #: boot, push, train a handful of examples, pull the adapter, average,
    #: release -- reporting each stage with its timing, so a failure says
    #: WHERE. `dry_run` proves the dataset and needs no GPU; this proves
    #: everything a real run does except its duration.
    test_run: Optional[bool] = False
    #: How many examples the test trains on. Enough to exercise a step and an
    #: optimizer update, few enough to cost nothing.
    test_examples: Optional[int] = 8

    wait: Optional[bool] = False
    session: Optional[str] = None
    model_config = ConfigDict(extra="allow")


@router.post("/v1/loras/datasets", summary="Upload a training dataset (content-addressed)")
async def upload_text_dataset(request: Request, _auth=Depends(_require_api_auth)):
    """Stream a JSONL training set into the content-addressed store.

    The body is the file; it is hashed while being written, never held in
    memory, because these run to hundreds of megabytes. Returns
    ``{id, bytes, existed}``. The id is the plain sha256 of the file, so a
    client can compute it locally and ask :func:`text_dataset_info` first
    rather than send the bytes twice.
    """
    from codai.api import text_lora_dataset as tld

    async def _chunks():
        async for chunk in request.stream():
            yield chunk

    # Collected into the sync store writer; the stream is consumed once.
    buf: List[bytes] = []
    async for chunk in _chunks():
        buf.append(chunk)
    if not any(buf):
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        hexd, existed, total = tld.save_dataset_stream(buf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"could not store dataset: {e}")
    return {"id": f"sha256:{hexd}", "bytes": total, "existed": existed}


@router.get("/v1/loras/datasets/{ident}", summary="Check an uploaded dataset exists")
async def text_dataset_info(ident: str, _auth=Depends(_require_api_auth)):
    """Existence check, so a client can skip re-sending a file already here."""
    from codai.api import text_lora_dataset as tld
    if not tld.dataset_exists(ident):
        raise HTTPException(status_code=404, detail="dataset not found")
    path = tld.dataset_path(ident)
    return {"id": f"sha256:{tld.dataset_hex(ident)}", "exists": True,
            "bytes": os.path.getsize(path)}


@router.post("/v1/loras/train/text", summary="Train a text (causal-LM) LoRA")
async def train_text_lora(req: TextLoraTrainRequest,
                          _auth=Depends(_require_api_auth)):
    """Train a QLoRA adapter on a causal LM from a JSONL of chat examples.

    Each record carries the messages, the answer to learn, optional character
    ranges of the answer NOT to learn, and an optional weight that scales the
    example's loss. See :mod:`codai.api.text_lora_dataset`.

    Returns a ``job_id`` immediately unless ``wait`` is set; poll
    ``/v1/loras/progress?job=<id>``, or recover it with ``?session=<token>``
    after a client restart. ``dry_run`` does the dataset pass and reports what
    would be trained without touching a GPU.
    """
    import asyncio
    import uuid
    if not req.name or "/" in req.name or ".." in req.name:
        raise HTTPException(status_code=400, detail="Invalid LoRA name")
    if not req.base_model:
        raise HTTPException(status_code=400, detail="base_model is required")

    from codai.api import text_lora_dataset as tld
    try:
        dataset_path = tld.resolve_dataset(req.dataset or "", req.dataset_id or "")
    except tld.DatasetError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session = req.session or f"sess-{uuid.uuid4().hex[:12]}"
    req.session = session
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    _new_job(job_id, session=session, name=req.name, base_model=req.base_model,
             target="text", status="queued", active=True, step=0, total=0,
             message="queued", started_at=time.time(), path=None)
    _save_job_request(job_id, req)

    if not req.wait:
        task = asyncio.create_task(_detached_text_train(req, job_id, dataset_path))
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)
        return {"ok": True, "async": True, "job_id": job_id, "session": session,
                "status": "queued"}
    try:
        result = await _run_text_train_job(req, job_id, dataset_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LoRA training failed: {e}")
    return {"ok": True, "job_id": job_id, "session": session, **result}


async def _detached_text_train(req: TextLoraTrainRequest, job_id: str,
                               dataset_path: str) -> None:
    """Background runner: errors land on the job record, not on a caller."""
    try:
        await _run_text_train_job(req, job_id, dataset_path)
    except TaskCancelled:
        print(f"[text-lora] {job_id}: cancelled", flush=True)
    except Exception as e:
        # PRINTED as well as recorded. A job that dies before it rents anything
        # wrote nothing to the log, so the only trace was a field on a record
        # a client has to ask for -- which is how a silent fall-through took
        # four queries to diagnose instead of one grep.
        print(f"[text-lora] {job_id}: FAILED: {type(e).__name__}: {e}",
              flush=True)
        _update_job(job_id, status="error", active=False, force=True,
                    message=f"training failed: {e}"[:300])


def _text_dataset_pass(req: TextLoraTrainRequest, dataset_path: str, tokenizer):
    """Encode the dataset and decide whether it is fit to train on.

    Split out from the training so it can run without a GPU (``dry_run``) and
    be tested: this is where a bad mask, a wrong field name or a too-small
    length limit shows up, and all three are cheaper to find here than after a
    pod boots.
    """
    from codai.api import text_lora_dataset as tld
    examples = tld.load_jsonl(dataset_path, field_map=req.field_map or None)
    encoded, stats = tld.prepare(
        tokenizer, examples, max_len=int(req.max_len or 8192),
        enable_thinking=bool(req.enable_thinking))
    total = stats.examples + len(stats.skipped)
    if total == 0:
        raise ValueError(f"{dataset_path} produced no usable examples")
    frac = len(stats.skipped) / float(total)
    limit = req.max_skipped_fraction
    limit = 0.05 if limit is None else float(limit)
    if frac > limit:
        # The refusal exists because the alternative is a successful-looking run
        # on a fraction of the data, which nothing downstream can detect.
        worst = "; ".join(f"line {s.line}: {s.reason}" for s in stats.skipped[:5])
        raise ValueError(
            f"{len(stats.skipped)} of {total} examples were skipped "
            f"({frac:.1%} > the {limit:.1%} limit). Raise max_skipped_fraction "
            f"to train anyway, or fix the cause: {worst}")
    return encoded, stats


def _looks_like_error(msg: object) -> bool:
    """Whether a fleet event is a failure worth keeping in full.

    Deliberately broad: the cost of keeping a long line that turns out to be
    progress is a few hundred bytes, and the cost of truncating a real one is
    somebody guessing a field name from half a message.
    """
    low = str(msg or "").lower()
    return any(w in low for w in (
        "http 4", "http 5", "error", "failed", "refus", "reject", "invalid",
        "could not", "cannot", "no candidate", "unavailable", "validation",
        "timeout", "timed out", "unreachable"))


def _resolve_text_base(base_model: str) -> str:
    """A loadable path or HF id for the base.

    A models.json key is resolved to its path so the catalogue's copy is used
    rather than a second download of the same weights into a different cache.
    """
    try:
        from codai.models.manager import _model_entry_for
        entry = _model_entry_for(base_model) or {}
        path = str(entry.get("path") or "").strip()
        if path:
            return os.path.expanduser(path)
    except Exception:
        pass
    # Same fallback as the config lookup, and for the same reason: that helper
    # reads the catalogue a PROCESS holds, and an engine where the admin module
    # was never initialised holds none -- so an alias came back unresolved and
    # went to from_pretrained verbatim. Caught by a probe that set
    # config_manager to None deliberately; it would otherwise have reappeared
    # the moment the training entry's own path became the base.
    try:
        from codai.api import text_lora_fleet as _flt
        for entry in _flt._entry_from_disk(base_model):
            path = str(entry.get("path") or "").strip()
            if path:
                return os.path.expanduser(path)
    except Exception:
        pass
    return base_model


async def _run_text_train_job(req: TextLoraTrainRequest, job_id: str,
                              dataset_path: str) -> dict:
    """Admitted through the scheduler like any other GPU work, then run in a
    thread: the training loop is blocking and must not sit on the event loop."""
    import asyncio
    _update_job(job_id, status="preparing", message="reading the dataset")
    return await asyncio.to_thread(_text_train_blocking, req, job_id, dataset_path)


def _text_train_blocking(req: TextLoraTrainRequest, job_id: str,
                         dataset_path: str) -> dict:
    """The dataset pass, then (unless dry_run) the QLoRA run itself.

    ``_active_job_id`` is set for the duration so the shared
    :func:`_check_train_cancel` applies here too -- which is what makes the
    Tasks page's cancel AND pause work on a text job without a second
    mechanism that would drift from the first.
    """
    global _active_job_id
    from codai.api import text_lora_trainer as tlt
    from transformers import AutoTokenizer

    _active_job_id = job_id

    from codai.api import text_lora_fleet as _flt
    _train_cfg = _flt.narrow(
        _flt.config_for_model(req.base_model),
        pods=req.pods, max_pods=req.max_pods, rounds=req.rounds,
        max_pod_hourly_usd=req.max_pod_hourly_usd,
        max_fleet_hourly_usd=req.max_fleet_hourly_usd)
    # Anything that ASKS for pods must refuse when the model has no block --
    # not fall through to training here. That fall-through is what turned a
    # test run into tens of gigabytes of download on a GPU-less orchestrator
    # and a CUDA error twenty minutes later, and the request looked fine.
    _asked_for_pods = any((req.distributed, req.test_run, req.pods,
                           req.min_pods, req.max_pods))
    if _asked_for_pods and not _train_cfg.enabled:
        # Name WHERE it looked. "There is no block" was technically true of the
        # lookup and false of the disk, which sent someone to check a file that
        # was already correct.
        where = _flt.catalogue_source()
        msg = (f"{req.base_model!r} has no `lora_train` block with "
               f"enabled=true — read from {where}. Set it on the Models page "
               "(card floor, pod ceiling, $/hr) for the model you want to "
               "train, and name THAT entry as base_model: the block's own "
               "base_model field is what gets trained, so an alias like "
               "'qwen38' is right even when the weights are "
               "'Qwen/Qwen3.8-27B'.")
        if req.dry_run:
            # A dry run rents nothing, so this must not stop it: the dataset
            # pass is free and is exactly what someone is trying to get an
            # answer from. Reported, not fatal.
            print(f"[text-lora] {job_id}: {msg}", flush=True)
            _update_job(job_id, message="dry run: no lora_train block"[:200])
        else:
            raise RuntimeError(msg)

    # What to TRAIN comes from the BLOCK, not from the request string. The
    # request names the catalogue entry so the configuration can be found; that
    # entry may serve something no QLoRA can start from (an AWQ checkpoint) or
    # have no local path at all, and then the request string is not a loadable
    # model at all. Resolved here rather than at the pod launch because the
    # tokenizer and the dataset pass need it too -- reaching from_pretrained
    # with a catalogue alias fails identically under dry_run, which is how this
    # was found, on a run that cost nothing and told us everything.
    base = _resolve_text_base(_train_cfg.base_model or req.base_model)
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)

    encoded, stats = _text_dataset_pass(req, dataset_path, tokenizer)
    summary = {"examples": stats.examples, "skipped": len(stats.skipped),
               "trained_tokens": stats.trained_tokens,
               "masked_tokens": stats.masked_tokens}
    _update_job(job_id, total=stats.examples,
                message=f"prepared: {stats.describe()}")
    print(f"[text-lora] {job_id}: {stats.describe()}", flush=True)
    for s in stats.skipped[:20]:
        print(f"[text-lora] skipped line {s.line}: {s.reason}", flush=True)

    if req.dry_run:
        _update_job(job_id, status="done", active=False, force=True,
                    message=f"dry run: {stats.describe()}")
        return {"dry_run": True, **summary}

    st = tlt.QLoraSettings(
        rank=int(req.rank or 16),
        alpha=int(req.alpha or (2 * int(req.rank or 16))),
        dropout=float(req.dropout if req.dropout is not None else 0.05),
        learning_rate=float(req.learning_rate or 1e-4),
        max_len=int(req.max_len or 8192),
        accumulation_steps=int(req.accumulation_steps or 8),
        enable_thinking=bool(req.enable_thinking),
        seed=int(req.seed or 42))

    if req.test_run:
        # One pod, one round, a handful of examples, and a short leash. The
        # point is to exercise every stage -- not to train anything -- so a
        # failure is attributable in minutes rather than discovered hours into
        # a real run.
        _train_cfg = dataclasses.replace(
            _train_cfg, max_pods=1, rounds=1,
            pod_max_lifetime_s=min(float(_train_cfg.pod_max_lifetime_s or 0)
                                   or 3600.0, 3600.0))
        n = max(1, int(req.test_examples or 8))
        encoded = encoded[:n]
        print(f"[text-lora] {job_id}: TEST RUN — 1 pod, 1 round, "
              f"{len(encoded)} example(s)", flush=True)
        # IN THE RECORD, not only the log. A reader seeing "step 0/1" on a
        # job that asked for 16 rounds concludes the request was ignored --
        # which is exactly what happened: a test run was read as a broken
        # real run, and the rounds path was investigated instead. The record
        # has to say which kind of run it is, next to the number it explains.
        _update_job(job_id, test_run=True, rounds=1,
                    rounds_note=(f"test run: rounds forced to 1 and the set "
                                 f"truncated to {len(encoded)} example(s), "
                                 f"whatever the request asked for"))
    # ASKING FOR A POD IS ASKING FOR A POD. This used to require
    # `max_pods > 1 or test_run`, so a request for exactly ONE pod -- `pods: 1,
    # min_pods: 1` -- fell through to the local path and tried to train on an
    # orchestrator with no card. Seven submissions died that way, and the local
    # fallback is the same failure a comment two screens up calls "a test run
    # turned into tens of gigabytes of download on an orchestrator with no
    # cards". `min_pods` was not even consulted, though it is the clearest
    # signal a caller wants a fleet.
    _want_dist = (_train_cfg.enabled
                  and (_train_cfg.max_pods > 1 or req.test_run
                       or (req.pods or 0) >= 1 or (req.min_pods or 0) >= 1)
                  and req.distributed is not False)
    if _asked_for_pods and not _want_dist:
        # Belt and braces: whatever combination gets here asked for pods and
        # would still train locally. Refusing is right even when the local host
        # HAS a card -- someone who asks for a fleet and silently gets one
        # machine has no way to tell, and finds out from the bill or the clock.
        raise RuntimeError(
            f"this request asks for pods (pods={req.pods!r}, "
            f"min_pods={req.min_pods!r}, max_pods={req.max_pods!r}, "
            f"distributed={req.distributed!r}, test_run={req.test_run!r}) but "
            f"the job would train LOCALLY: "
            + ("`distributed: false` was passed explicitly"
               if req.distributed is False else
               f"the model's lora_train block is "
               + ("disabled" if not _train_cfg.enabled else
                  f"max_pods={_train_cfg.max_pods}"))
            + ". Refusing rather than training somewhere you did not ask for.")
    if _want_dist:
        # Rent a fleet and drive the rounds. Everything above this point was
        # free; everything below it spends.
        try:
            return {**summary, **_text_train_distributed(
                req, job_id, dataset_path, encoded, stats, st, _train_cfg)}
        except TaskCancelled:
            _update_job(job_id, status="cancelled", active=False, force=True,
                        message="cancelled; pods released")
            raise
        finally:
            _active_job_id = None

    # Local training needs a local card. Checked BEFORE the base is fetched,
    # because the failure otherwise arrives after tens of gigabytes of download
    # as a CUDA error that names neither the cause nor the fix.
    try:
        import torch as _torch
        _has_gpu = bool(_torch.cuda.is_available())
    except Exception:
        _has_gpu = False
    if not _has_gpu:
        raise RuntimeError(
            "this install has no CUDA device, so a text LoRA cannot be trained "
            "here. Configure `lora_train` on the model (Models page) and ask "
            "for pods, or run the job on a machine with a card.")

    out_dir = os.path.join(_loras_dir(), req.name)
    try:
        result = tlt.run_training(
            base=base, encoded=encoded, settings=st, out_dir=out_dir,
            on_progress=lambda **kw: _update_job(job_id, **kw),
            # The shared per-step hook: raises TaskCancelled on a cancel and
            # blocks while paused, exactly as the diffusion trainer does.
            check=_check_train_cancel)
    except TaskCancelled:
        _update_job(job_id, status="cancelled", active=False, force=True,
                    message="cancelled; the checkpoint is kept")
        raise
    finally:
        _active_job_id = None
    _update_job(job_id, status="done", active=False, force=True,
                path=out_dir, message=f"saved {req.name}")
    return {**summary, **result, "path": out_dir, "name": req.name}


# ─────────────────────────────────────────────────────────────────────────────
# Distributed text QLoRA
# ─────────────────────────────────────────────────────────────────────────────
#
# The orchestrator DRIVES. There are no pod-facing endpoints here and there is
# no callback URL: coderai already addresses pods -- that is how it proxies
# inference to them -- and a direct TCP mapping bypasses the RunPod proxy, so
# the orchestrator pushes the dataset and each round's work, polls the pod's
# own /status, and pulls the finished adapter.
#
# An earlier version of this had the pods calling in, on the belief that a pod
# could not be addressed. It can, and this is both smaller and free of the one
# configuration that could strand a whole fleet: a wrong callback URL meant
# pods that booted, could not phone home, and billed until reaped.

_train_coordinators: Dict[str, Any] = {}
_coord_lock = threading.Lock()


class _TextTrainCoordinator:
    """The orchestrator's side of a distributed run.

    Holds the shard plan, the round tracker and the fleet, and is what the three
    pod-facing endpoints reach. It initiates nothing over the network: pods ask
    for work, hand in adapters and fetch averages, and this decides when a round
    is done.
    """

    def __init__(self, job_id: str, *, plan, tracker, fleet, encoded,
                 out_dir: str, settings, dataset_id: str = "",
                 base: str = "", field_map: Optional[dict] = None,
                 total_rounds: int = 1):
        from codai.api import text_lora_rounds as tlr
        self.job_id = job_id
        self.plan = plan
        self.tracker = tracker
        self.fleet = fleet
        self.encoded = encoded
        self.out_dir = out_dir
        self.settings = settings
        self.dataset_id = dataset_id
        self.base = base
        self.field_map = dict(field_map or {})
        self.total_rounds = max(1, int(total_rounds))
        self.secret = secrets.token_urlsafe(32)
        self._averages: Dict[int, str] = {}
        self._lock = threading.Lock()
        os.makedirs(out_dir, exist_ok=True)

    # -- what a pod is told ---------------------------------------------- #

    def finished(self) -> bool:
        return self.tracker.round >= self.total_rounds

    def round_payload(self, shard: int) -> dict:
        """What to POST to a pod's /round for the open round."""
        a = self.assignment(int(shard))
        return {"round": a["round"], "indices": a["indices"],
                "settings": a["settings"], "base": self.base,
                "field_map": self.field_map or None}

    def assignment(self, shard: int) -> dict:
        rnd = self.tracker.round
        if self.finished():
            # The pod's exit signal. Without it a pod polls forever on a job
            # that is over, which is a pod billing for nothing.
            return {"job": self.job_id, "shard": int(shard), "round": rnd,
                    "finished": True, "rounds": self.total_rounds}
        return {
            "job": self.job_id, "shard": int(shard), "round": rnd,
            "finished": False, "rounds": self.total_rounds,
            "n_shards": self.plan.n_shards,
            "base": self.base, "field_map": self.field_map or None,
            "indices": self.plan.indices_for(int(shard), len(self.encoded),
                                             round=rnd,
                                             rounds=self.total_rounds),
            "dataset_id": self.dataset_id,
            # Where to start: the previous round's average, or the bare base on
            # round 0. Named as a round rather than a blob id so a pod that
            # restarts mid-round fetches the same thing its siblings did.
            "resume_round": (rnd - 1) if rnd > 0 else None,
            "settings": {"rank": self.settings.rank, "alpha": self.settings.alpha,
                         "dropout": self.settings.dropout,
                         "learning_rate": self.settings.learning_rate,
                         "max_len": self.settings.max_len,
                         "accumulation_steps": self.settings.accumulation_steps,
                         "enable_thinking": self.settings.enable_thinking,
                         "seed": self.settings.seed},
        }

    def submit(self, sub) -> None:
        with self._lock:
            self.tracker.submit(sub)

    def average_path(self, rnd: int) -> str:
        with self._lock:
            return self._averages.get(int(rnd), "")

    # -- the round boundary ----------------------------------------------- #

    def close_round_if_ready(self) -> Optional[int]:
        """Average and advance when the round is done. Returns the round closed.

        The average is written BEFORE the tracker advances, so a pod that asks
        for round k's adapter the instant k+1 opens gets the file rather than a
        404 -- the two are not atomic from the pod's side and the ordering is
        what makes the gap harmless.
        """
        from codai.api import text_lora_rounds as tlr
        with self._lock:
            if not self.tracker.ready():
                return None
            rnd = self.tracker.round
            subs = self.tracker.have()
            weights = self.tracker.weights_for(subs)
            out = os.path.join(self.out_dir, f"round{rnd}.safetensors")
            tlr.average_adapter_files([s.path for s in subs], weights, out)
            self._averages[rnd] = out
            closed, absent = self.tracker.close_round()
        print(f"[text-lora] {self.job_id}: round {rnd} averaged from "
              f"{len(subs)} shard(s) ({int(sum(weights))} examples)"
              + (f"; absent: {absent}" if absent else ""), flush=True)
        return rnd

    def reap_the_dead(self) -> List[int]:
        """Release pods that stopped checking in, and say which shards are free.

        Their shards are not reassigned here: the round protocol averages what
        arrives, so a lost shard costs its share of one round rather than the
        run. Re-renting is a decision about money and belongs to the caller.
        """
        gone = []
        for pod in self.fleet.to_reap():
            if self.fleet.release(pod.shard, "stopped checking in"):
                gone.append(pod.shard)
        return gone


# ─────────────────────────────────────────────────────────────────────────────
# Distributed text QLoRA: the coordinator loop
# ─────────────────────────────────────────────────────────────────────────────

def _price_for(client, gpu_type_id: str, cloud_type: str) -> Optional[float]:
    """The catalogue's $/hr for this card, or None if it cannot be read.

    The per-pod ceiling has to bind on the REAL price, not on a number the
    caller guessed: a ceiling checked against an estimate is a ceiling that
    agrees with whatever the request claimed.
    """
    want = (gpu_type_id or "").strip().lower()
    try:
        for g in client.list_gpu_types() or []:
            if str(g.get("id") or "").strip().lower() != want:
                continue
            if (cloud_type or "").upper() == "COMMUNITY":
                return g.get("community_price") or g.get("on_demand_price")
            return g.get("secure_price") or g.get("on_demand_price")
    except Exception as exc:
        print(f"[text-lora] could not read the GPU catalogue: {exc}", flush=True)
    return None


def _text_train_distributed(req: "TextLoraTrainRequest", job_id: str,
                            dataset_path: str, encoded, stats, settings,
                            cfg) -> dict:
    """Rent a fleet, drive the rounds, release everything. The spending path.

    ``cfg`` is the MODEL's ``lora_train`` block, already narrowed by whatever
    the request asked for. Nothing here reads a card, an image or a price out
    of the request: those are the operator's, set on the Models page, and
    that is what makes this route safe for a client to call.

    Structured so the release is in a ``finally`` that cannot be skipped: every
    other failure mode here -- no capacity, a cancel, every pod dying, an
    exception in the averaging -- ends with the pods terminated, because the one
    outcome that is not acceptable is a fleet left running.
    """
    from codai.api import runpod_client as rpc
    from codai.api import text_lora_fleet as flt
    from codai.api import text_lora_rounds as tlr
    from codai.models.manager import get_active_runpod_config

    account = get_active_runpod_config()
    if account is None or not getattr(account, "api_key", ""):
        raise RuntimeError("distributed training needs the RunPod account "
                           "configured (runpod.api_key)")

    n_shards = max(1, int(cfg.max_pods or 1))
    rounds = max(1, int(cfg.rounds or 1))
    # Recorded so "step x/y" has a stated source. A request may RAISE rounds
    # (narrow() passes it through: rounds change duration, not fleet size or
    # price), so when y is not what was asked for, the record should say what
    # it is and why rather than leaving it to be inferred.
    _update_job(job_id, test_run=False, rounds=rounds,
                rounds_note=(f"{rounds} averaging round(s); each one is a "
                             f"checkpoint, so an interruption costs at most "
                             f"1/{rounds} of the work"))
    _clouds = flt.clouds_for(cfg)
    # The FIRST is what a single-cloud create falls back to; the candidates
    # carry their own. "SECURE+COMMUNITY" in a log line is the honest
    # description of a ranking that spans both.
    cloud = _clouds[0]
    cloud_label = "+".join(c.lower() for c in _clouds)
    ceiling = float(cfg.max_pod_hourly_usd or 0.0)

    client = rpc.RunpodClient(account)
    # Any card that meets the VRAM floor and the ceiling, cheapest first. A
    # ranked list rather than one choice because capacity moves: the cheapest
    # eligible card is often sold out by the time the pod is requested.
    cards = flt.eligible_cards(client, cfg, ceiling=ceiling)
    if not cards:
        raise RuntimeError(
            f"no RunPod card has >= {cfg.min_vram_gb:.0f} GB at or under "
            f"${ceiling:.2f}/hr {cloud_label}"
            + (f" within {cfg.gpu_types}" if cfg.gpu_types else "")
            + " — raise the ceiling, lower the floor, or allow COMMUNITY")
    price = cards[0]["hourly_usd"]
    print(f"[text-lora] {job_id}: {n_shards} pod(s), {rounds} round(s), "
          f"{cloud_label}, >= {cfg.min_vram_gb:.0f}GB at <= ${ceiling:.2f}/hr; "
          f"candidates: {flt.describe_cards(cards)}", flush=True)

    out_dir = os.path.join(_loras_dir(), req.name)
    plan = tlr.RoundPlan(n_shards=n_shards, seed=settings.seed)
    tracker = tlr.RoundTracker(
        plan,
        min_fraction=0.75,
        deadline_s=max(60.0, float(cfg.poll_interval_s or 20.0) * 60))
    # The base is pulled BY THE POD, so the token has to travel with it. A pod
    # used to get three CODERAI_TRAIN_* variables and nothing else, which made
    # every pull anonymous: a 404 on a gated repo, a rate-limit on the rest,
    # discovered minutes into a boot that is already billing.
    _hf = flt.hf_token_for_pods(cfg)
    _pod_env = {}
    if _hf:
        # Both names: the hub library reads either, and which one an image
        # honours is not worth finding out one failed rental at a time.
        _pod_env = {"HF_TOKEN": _hf, "HUGGING_FACE_HUB_TOKEN": _hf}
    else:
        print(f"[text-lora] {job_id}: no Hugging Face token configured "
              f"(lora_train.hf_token, or HF_TOKEN in the orchestrator's "
              f"environment); pods will pull "
              f"{_resolve_text_base(cfg.base_model or req.base_model)!r} "
              f"anonymously, which fails "
              f"outright for a gated repo", flush=True)
    spec = flt.TrainerPodSpec(
        image=cfg.image, gpu_types=[c["id"] for c in cards], env=_pod_env,
        # Same order as gpu_types: each offer keeps the cloud it was priced in.
        gpu_clouds=[str(c.get("cloud_type") or cloud) for c in cards],
        # Each offer's real price, so a rented pod reports what it costs.
        gpu_prices=[float(c.get("hourly_usd") or 0.0) for c in cards],
        cloud_type=cloud, hourly_usd=price,
        max_pod_hourly_usd=ceiling,
        container_disk_gb=int(cfg.container_disk_gb or 120),
        network_volume_id=(cfg.network_volume_id or ""),
        max_silence_s=float(cfg.pod_max_silence_s or 1800.0),
        max_lifetime_s=float(cfg.pod_max_lifetime_s or 86400.0))
    fleet = flt.TrainerFleet(
        client, job_id, spec, ledger_key=f"text-lora:{req.name}",
        max_fleet_hourly_usd=float(cfg.max_fleet_hourly_usd or 0.0),
        # 200 characters fits a progress line and HALVES a validation error.
        # A reader then has to guess which field an API rejected -- which
        # happened: "additional properties 's…" was all that survived, and the
        # guess was wrong. Progress stays short; an error keeps enough to name
        # the cause, in a field of its own so it is not overwritten by the
        # next heartbeat.
        on_event=lambda m: _update_job(
            job_id, message=str(m)[:200],
            **({"last_error": str(m)[:2000]} if _looks_like_error(m) else {})))
    coord = _TextTrainCoordinator(
        job_id, plan=plan, tracker=tracker, fleet=fleet, encoded=encoded,
        out_dir=out_dir, settings=settings,
        # What to TRAIN: the block's own base when it names one, because the
        # entry may SERVE something that cannot be a QLoRA base (an AWQ
        # checkpoint, or a pod-served model with no local path at all).
        dataset_id=req.dataset_id or "",
        base=_resolve_text_base(cfg.base_model or req.base_model),
        field_map=req.field_map, total_rounds=rounds)

    # The image runs the baked runner by default, so there is no start command
    # and no URL of ours anywhere in the pod's environment. It is given only
    # the bearer its own API requires.
    # No URL and no credential here: the fleet mints a bearer per pod, and the
    # image runs its baked runner by default.
    spec.env = {
        "CODERAI_TRAIN_WORKDIR": "/workspace/train",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    }

    with _coord_lock:
        _train_coordinators[job_id] = coord
    poll = max(5.0, float(cfg.poll_interval_s or 20.0))
    stages = flt.Stages()
    try:
        started = fleet.provision(list(range(n_shards)))
        if not started and not fleet.pods:
            stages.failed("rent", "no pods could be rented")
            raise RuntimeError("no pods could be rented — nothing to train on")
        stages.done("rent", pods=len(fleet.pods), hourly_usd=price)

        # Wait for every address AT ONCE. This used to be a `for` loop, so a
        # four-pod job needed four CONSECUTIVE boots inside a budget sized for
        # one: shard 0 was waited out to completion before shard 1 was even
        # looked at, while shards 1..n sat booted and billing. Pods boot in
        # parallel, so the wall clock is one boot, not n -- and the per-pod
        # timeout now means what it says instead of being the nth share of a
        # race nobody could win.
        #
        # A pod that never publishes a port is still released as soon as ITS
        # own wait gives up, rather than at the end: it is billing and it will
        # never be useful.
        ready = []
        _boot_timeout = float(cfg.pod_boot_timeout_s or 1800.0)
        _shards = sorted(fleet.pods)

        def _bring_up(shard):
            try:
                url = fleet.address(shard, timeout=_boot_timeout)
            except Exception as exc:
                print(f"[text-lora] {job_id}: shard {shard} never came up: "
                      f"{exc}", flush=True)
                fleet.release(shard, "never published a port")
                return None
            if not url:
                fleet.release(shard, "no address")
                return None
            return shard

        if len(_shards) == 1:
            got = _bring_up(_shards[0])
            if got is not None:
                ready.append(got)
        else:
            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(
                    max_workers=min(len(_shards), 32),
                    thread_name_prefix=f"boot-{job_id}") as pool:
                futures = {pool.submit(_bring_up, sh): sh for sh in _shards}
                for fut in _cf.as_completed(futures):
                    try:
                        got = fut.result()
                    except Exception as exc:
                        print(f"[text-lora] {job_id}: shard "
                              f"{futures[fut]} boot raised: {exc}", flush=True)
                        continue
                    if got is not None:
                        ready.append(got)
        # Cancel AFTER the waits, not inside them: a cancel mid-boot would
        # leave a rented pod with nobody to release it, and release_all in the
        # caller's finally is what guarantees the fleet goes away.
        _check_train_cancel()
        # The dataset goes out once the fleet is known, in shard order, so a
        # failure names a shard rather than whichever thread got there first.
        for shard in sorted(ready):
            flt.push_dataset(fleet.pods[shard], dataset_path)
            fleet.note_checkin(shard)
        if not ready:
            stages.failed("reachable", "no pod published a port")
            raise RuntimeError("no pod became reachable — nothing to train on")
        floor = max(0, int(req.min_pods or 0))
        if floor and len(ready) < floor:
            # Refused rather than quietly scaled down: the round length was
            # chosen for a fleet of n, and training a 16-pod job on 2 bills for
            # a result nobody asked for. The finally below releases them.
            stages.failed("min_pods",
                          f"{len(ready)} of at least {floor} pods came up")
            raise RuntimeError(
                f"only {len(ready)} pod(s) became ready and this job requires "
                f"at least {floor} — released rather than training on "
                f"{len(ready)}")
        stages.done("reachable", pods=len(ready))
        stages.done("dataset", bytes=os.path.getsize(dataset_path))
        _update_job(job_id, status="training", total=rounds,
                    message=f"{len(ready)} pod(s) ready")

        while not coord.finished():
            _check_train_cancel()
            rnd = coord.tracker.round
            # Start the round on every pod that is idle and has not submitted.
            for shard in sorted(fleet.pods):
                pod = fleet.pods[shard]
                if not pod.url or any(s.shard == shard
                                      for s in coord.tracker.have(rnd)):
                    continue
                status = flt.poll_status(pod)
                state = str(status.get("state") or "")
                if state in ("training", "encoding", "loading-tokenizer"):
                    fleet.note_checkin(shard)
                    continue
                if state == "done" and int(status.get("round", -1)) == rnd:
                    # Finished: pull it and record the submission.
                    dest = os.path.join(out_dir, f"r{rnd}-s{shard}.safetensors")
                    flt.fetch_adapter(pod, dest)
                    from codai.api import text_lora_rounds as _tlr
                    coord.submit(_tlr.Submission(
                        shard=shard, round=rnd, path=dest,
                        examples=int(status.get("examples_trained") or 0)))
                    fleet.note_checkin(shard, rounds_done=rnd + 1)
                    continue
                if state == "error":
                    stages.failed(f"train(shard {shard})",
                                  str(status.get("error") or "")[:200])
                    fleet.release(shard, f"round {rnd} failed")
                    continue
                if state == "unreachable":
                    continue        # the silence window decides, not one poll
                # idle, or done with an older round: give it this one.
                if rnd > 0:
                    prev = coord.average_path(rnd - 1)
                    if prev:
                        flt.push_start_adapter(pod, prev)
                try:
                    flt.start_round(pod, coord.round_payload(shard))
                    fleet.note_checkin(shard)
                except flt.PodBusy:
                    pass

            closed = coord.close_round_if_ready()
            if closed is not None:
                stages.done(f"round {closed}",
                            shards=len(coord.tracker.have(closed)))
                _update_job(job_id, step=closed + 1,
                            message=f"round {closed} averaged; "
                                    f"{len(fleet.pods)} pod(s) running")
                continue
            gone = coord.reap_the_dead()
            if gone:
                print(f"[text-lora] {job_id}: released silent shard(s) {gone}",
                      flush=True)
            if not fleet.pods:
                raise RuntimeError(
                    f"every pod is gone at round {coord.tracker.round} of "
                    f"{rounds}; the last average is kept in {out_dir}")
            time.sleep(poll)

        final = coord.average_path(rounds - 1)
        if not final or not os.path.isfile(final):
            raise RuntimeError(f"the last round produced no average in {out_dir}")
        import shutil
        adapter = os.path.join(out_dir, "adapter_model.safetensors")
        shutil.copyfile(final, adapter)
        print(f"[text-lora] {job_id}: finished {rounds} round(s); "
              f"adapter at {adapter}", flush=True)
        stages.done("adapter", path=adapter)
        return {"rounds": rounds, "shards": n_shards,
                "cards": [c["id"] for c in cards[:6]], "hourly_usd": price,
                "examples": stats.examples, "path": out_dir,
                "adapter": adapter, "test_run": bool(req.test_run),
                # What the run cost in time, stage by stage. On a first run the
                # answer to "why did that take twenty minutes" is one stage,
                # and rarely the one anyone guessed.
                **stages.as_dict(),
                "spend_usd": round(sum(p.cost_so_far()
                                       for p in fleet.pods.values()), 4)}
    finally:
        # The guarantee. Not conditional on how we got here: a cancel, an
        # exception, a fleet that died -- the pods stop billing.
        released = fleet.release_all("training job ended")
        leftover = [o for o in flt.orphans() if o.get("job") == job_id]
        print(f"[text-lora] {job_id}: released {released} pod(s)"
              + (f"; STILL RUNNING: {[o['pod_id'] for o in leftover]}"
                 if leftover else ""), flush=True)
        with _coord_lock:
            _train_coordinators.pop(job_id, None)
