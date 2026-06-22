# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Spawn and supervise engine subprocesses for the front proxy.

One engine per GPU (or a configured count). Each engine is this same codebase
relaunched with ``--engine-only --internal-port P`` and ``CUDA_VISIBLE_DEVICES``
pinned to its GPU, so inside the engine its GPU is always ``cuda:0`` and the
existing per-process VRAM/eviction logic is untouched.

The supervisor polls each engine's auth-free ``/internal/engine-state`` to keep the
:class:`EngineRegistry` current (health, resident models, VRAM) and respawns an
engine that dies or stops answering — which is also how a CUDA-poisoned engine
recovers (the front and sibling engines survive).
"""

import atexit
import collections
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Optional

import httpx

from codai.frontproxy.registry import Engine, EngineRegistry


def _engine_preexec():
    """Run in the child just before exec: put the engine in its OWN process group
    (so the terminal's Ctrl-C reaches only the front, which then stops engines
    deterministically) and ask the kernel to SIGKILL the engine if the front dies
    unexpectedly — even by SIGKILL, where our atexit/handlers can't run. Linux-only;
    best-effort elsewhere."""
    try:
        os.setsid()
    except Exception:
        pass
    try:
        import ctypes
        # prctl(PR_SET_PDEATHSIG, SIGKILL) — parent-death signal.
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, 9, 0, 0, 0)
    except Exception:
        pass


def _port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """True if ``port`` can be bound right now on ``host``."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def detect_gpus() -> list:
    """Return CUDA GPU indices via nvidia-smi (no torch). Empty when none found."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run(
            [smi, "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return []
        return [int(line.strip()) for line in out.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _gpu_selectors(spec: dict, env: dict) -> list:
    """Which physical cards an engine owns, as selectors thermal can match against:
    NVIDIA UUIDs (precise) and/or vendor keywords ("nvidia"/"amd"/"intel").

    Derived from the engine's CUDA_VISIBLE_DEVICES (UUIDs), its ``gpus`` vendor
    keyword, its Vulkan ICD, and its backend."""
    sels = []
    for tok in (env.get("CUDA_VISIBLE_DEVICES") or "").split(","):
        tok = tok.strip()
        if tok.startswith("GPU-"):
            sels.append(tok)          # precise NVIDIA UUID
        elif tok.isdigit():
            sels.append("nvidia")     # index → vendor fallback
    vmap = {"radeon": "amd", "amd": "amd", "intel": "intel", "nvidia": "nvidia"}
    gpus_kw = (spec.get("gpus") or "").strip().lower()
    if gpus_kw in vmap:
        sels.append(vmap[gpus_kw])
    icd = (env.get("VK_ICD_FILENAMES") or "").lower()
    if "radeon" in icd or "amd" in icd:
        sels.append("amd")
    elif "intel" in icd:
        sels.append("intel")
    elif "nvidia" in icd:
        sels.append("nvidia")
    if (spec.get("backend") or "").lower() == "nvidia" and not sels:
        sels.append("nvidia")
    sels = list(dict.fromkeys(sels))
    # When we have precise NVIDIA UUIDs, drop the broad "nvidia" vendor so two
    # separate NVIDIA engines don't each match every NVIDIA card.
    if any(s.startswith("GPU-") for s in sels):
        sels = [s for s in sels if s != "nvidia"]
    return sels


class EngineSupervisor:
    def __init__(self, config, args, registry: EngineRegistry, models_path=None,
                 internal_token=None, debug=False):
        self.config = config
        self.args = args
        self.registry = registry
        self.models_path = models_path   # for computing per-engine model assignment
        self.internal_token = internal_token  # shared secret stamped on engine calls
        self.debug = debug               # --debug-engine: verbose engine lifecycle
        self._health = {}                # engine_id -> last healthy bool (for debug)
        self._assign_mtime = 0.0         # models.json mtime → live re-assignment
        self._stopped = threading.Event()
        self._poll_thread = None
        self._logs = {}   # engine_id -> deque tail
        self._restart_lock = threading.RLock()
        # Serialise terminal writes across engine pump threads + track whether the
        # last thing we printed was an in-place tqdm progress line (so the next
        # normal line finalises it with a newline).
        self._log_lock = threading.Lock()
        self._log_progress_tag = None    # tag currently owning the \r line, or None
        self._log_last_pct = {}          # tag -> last printed % (non-TTY throttle)

    def _assign_models(self, engines) -> None:
        """Give each engine the set of models it owns (via CODERAI_ENGINE_MODELS), so
        a model is registered on exactly one engine. With a single engine there's
        nothing to partition — it owns everything."""
        if not self.models_path or len(engines) < 2:
            return
        try:
            from codai.frontproxy.assignment import compute_assignment
            default_engine = getattr(self.config.server, "default_engine", None)
            ds4 = getattr(self.config, "ds4", None)
            assignment = compute_assignment(engines, self.models_path,
                                            default_engine, ds4)
            for e in engines:
                owned = assignment.get(e.name, [])
                e.assigned_models = set(owned)   # the front's router enforces this
                # Also hand the set to the engine so it only registers/pre-loads its
                # assigned models (avoids e.g. whisper-server starting on every
                # engine). models.json itself stays full for the admin view.
                e.env["CODERAI_ENGINE_MODELS"] = json.dumps(owned)
                print(f"[front] engine '{e.name}' assigned {len(owned)} model(s): "
                      f"{', '.join(owned) if owned else '(none)'}", flush=True)
        except Exception as exc:
            print(f"[front] model assignment skipped: {exc}", flush=True)

    def _alloc_port(self) -> int:
        """Next free internal port at/above internal_port_base, skipping the front's
        own port and any port already in use, so engines never collide with the
        front or each other (or a stale process on the base port)."""
        p = self._port_cursor
        front_port = int(getattr(self.config.server, "port", 0) or 0)
        while p == front_port or not _port_is_free(p):
            p += 1
        self._port_cursor = p + 1
        return p

    # ----------------------------------------------------------------- planning
    def _build_engines(self) -> list:
        """Return the list of Engine objects to launch.

        Explicit ``engine_specs`` (heterogeneous: per-engine backend + env, e.g. an
        NVIDIA card and a Radeon card) take precedence. Otherwise auto-detect the
        LOCAL hardware and create one engine per GPU vendor actually present —
        NVIDIA (CUDA), AMD/Radeon (Vulkan), Intel (Vulkan) — so e.g. a box with an
        NVIDIA + a Radeon gets both engines without any config. A machine with no
        GPU gets a single CPU engine.
        """
        srv = self.config.server
        self._port_cursor = srv.internal_port_base
        specs = getattr(srv, "engine_specs", None)
        engines = []

        # Expose every GPU to the engines (allow_cross) when cross-backend pooling is
        # wanted: either the global default (offload.gpu_split) OR any individual model
        # opts into a split ("Split — <engine> first" → per-model gpu_split). Without
        # this the lead engine wouldn't even see the foreign card, so a per-model split
        # couldn't use it. Non-split models are kept on their own backend by the GGUF
        # loader (it zeroes the foreign devices) so this doesn't resurrect the
        # "everything spreads onto the Radeon" bug.
        _cross = bool(getattr(getattr(self.config, "offload", None), "gpu_split", False))
        if not _cross and self.models_path:
            try:
                import json as _json
                with open(self.models_path) as _mf:
                    _md = _json.load(_mf)
                for _lst in _md.values():
                    if isinstance(_lst, list) and any(
                            isinstance(_e, dict) and _e.get("gpu_split") for _e in _lst):
                        _cross = True
                        break
            except Exception:
                pass
        if specs:
            from codai.frontproxy.gpu_detect import vendor_env
            for idx, spec in enumerate(specs):
                backend = (spec.get("backend") or "auto").strip()
                # Vendor keyword → all of that vendor's cards on this machine. A
                # plain nvidia backend defaults to "nvidia" (unambiguous); Vulkan
                # vendors must be named ("radeon"/"amd"/"intel"). Explicit env wins.
                gpus_kw = (spec.get("gpus") or "").strip().lower()
                if not gpus_kw and not spec.get("env") and backend == "nvidia":
                    gpus_kw = "nvidia"
                detected = vendor_env(gpus_kw, allow_cross=_cross) if gpus_kw else {}
                explicit = {str(k): str(v) for k, v in (spec.get("env") or {}).items()}
                env = {**detected, **explicit}     # explicit overrides detected
                # Tell the engine which physical cards it owns, so thermal
                # protection scopes GPU cooldowns to this engine (CPU stays global).
                sels = _gpu_selectors(spec, env)
                if sels and "CODERAI_ENGINE_GPUS" not in env:
                    env["CODERAI_ENGINE_GPUS"] = ",".join(sels)
                caps = set(spec.get("capabilities") or [])
                engines.append(Engine(
                    id=idx, gpu=None, port=self._alloc_port(), primary=(idx == 0),
                    name=spec.get("name") or f"engine#{idx}",
                    backend=backend, env=env, capabilities=caps,
                ))
            return engines

        # Auto: one engine per GPU vendor actually present on this machine. Vendors
        # come from Vulkan enumeration AND the sysfs PCI-vendor fallback, so AMD/Intel
        # are detected even without vulkaninfo installed.
        from codai.frontproxy.gpu_detect import nvidia_gpus, gpu_vendors, vendor_env
        vendors = gpu_vendors()
        # (engine name, vendor keyword, backend). NVIDIA first so it's the primary
        # (it owns admin/sessions and has the broadest capabilities). NVIDIA needs
        # CUDA, so it's gated on nvidia-smi rather than the Vulkan/sysfs presence.
        plan = []
        if nvidia_gpus():
            plan.append(("nvidia", "nvidia", "nvidia"))
        if "amd" in vendors:
            plan.append(("radeon", "amd", "vulkan"))
        if "intel" in vendors:
            plan.append(("intel", "intel", "vulkan"))

        if not plan:
            engines.append(Engine(id=0, gpu=None, port=self._alloc_port(),
                                  primary=True, name="cpu", backend="auto", env={}))
            return engines

        for idx, (name, vkw, backend) in enumerate(plan):
            env = vendor_env(vkw, allow_cross=_cross)
            sels = _gpu_selectors({"backend": backend, "gpus": vkw}, env)
            if sels:
                env["CODERAI_ENGINE_GPUS"] = ",".join(sels)
            engines.append(Engine(id=idx, gpu=None, port=self._alloc_port(),
                                  primary=(idx == 0), name=name,
                                  backend=backend, env=env))
        return engines

    # ------------------------------------------------------------------ spawning
    def _engine_cmd(self, port: int) -> list:
        """Build the command to relaunch this codebase as an engine."""
        # sys.argv[0] is the launcher script (``coderai``); preserve all original
        # args (config dir, model selection, …) and append the engine flags. Strip
        # any flag that would re-trigger front mode or fix a different port.
        passthrough = []
        skip_next = False
        for a in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if a in ("--single-process", "--engine-only"):
                continue
            if a == "--internal-port":
                skip_next = True
                continue
            passthrough.append(a)
        return [sys.executable, sys.argv[0], *passthrough,
                "--engine-only", "--internal-port", str(port)]

    def _spawn(self, engine: Engine) -> None:
        env = dict(os.environ)
        # Engine stdout is a pipe (not a TTY), so CPython block-buffers print()
        # output — debug lines (e.g. --debug-requests) would stall in the buffer
        # until it fills or the process exits, unlike tqdm which flushes stderr
        # itself. Force unbuffered so engine logs reach the front terminal live.
        env["PYTHONUNBUFFERED"] = "1"
        # Per-engine env block (device pinning, Vulkan ICD, etc.). Empty-string
        # values are honoured (e.g. CUDA_VISIBLE_DEVICES="" hides all CUDA cards).
        for k, v in (engine.env or {}).items():
            env[str(k)] = str(v)
        # The global host-RAM cap (offload.max_ram_gb) is SHARED across all engines,
        # not split: tell each engine the front's PID so it measures the whole
        # fleet's RAM (front + every engine + workers) against the one cap.
        env["CODERAI_FRONT_PID"] = str(os.getpid())
        # Only the primary engine talks to the AISBF broker, so N engines don't
        # register N times under the same provider id.
        if engine.primary:
            env["CODERAI_ENGINE_PRIMARY"] = "1"
        # Shared secret: the engine rejects any HTTP request that doesn't carry it,
        # so only the front (which has it) can reach the engine on localhost.
        if self.internal_token:
            env["CODERAI_INTERNAL_TOKEN"] = self.internal_token
        # Resolve this engine's concurrency limits (global default, or a per-engine
        # override keyed by engine name) and hand them down so a bigger card can run
        # more in parallel than a smaller one.
        srv = self.config.server
        mdl = self.config.models
        par = (srv.max_parallel_requests_overrides or {}).get(engine.name,
                                                              srv.max_parallel_requests)
        inst = (getattr(mdl, "max_model_instances_overrides", None) or {}).get(
            engine.name, getattr(mdl, "max_model_instances", 1))
        if par is not None:
            env["CODERAI_MAX_PARALLEL"] = str(int(par))
        if inst is not None:
            env["CODERAI_MAX_MODEL_INSTANCES"] = str(int(inst))
        # Force this engine's backend (the engine reads this in --engine-only mode
        # and overrides config.backend.type) so a Vulkan/Radeon engine doesn't
        # auto-pick CUDA, and vice-versa.
        if engine.backend and engine.backend != "auto":
            env["CODERAI_ENGINE_BACKEND"] = engine.backend
        cmd = self._engine_cmd(engine.port)
        tag = engine.name + (f"(gpu{engine.gpu})" if engine.gpu is not None else "")
        print(f"[front] launching {tag} on port {engine.port}: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            preexec_fn=_engine_preexec if os.name == "posix" else None,
        )
        engine.proc = proc
        tail = self._logs.setdefault(engine.id, collections.deque(maxlen=30))
        threading.Thread(target=self._pump_logs, args=(tag, proc, tail, engine),
                         daemon=True).start()

    # tqdm progress line, e.g. "Loading weights:  12%|██ | 50/427 [..]" → desc, n, total.
    _PROGRESS_RE = re.compile(r'^(.*?):\s*\d+%\|.*?\|\s*(\d+)/(\d+)')
    # Phase markers that name the model being loaded / signal completion.
    _LOAD_START_RE = re.compile(
        r'(?:Loading model on demand|Loading HuggingFace model|Loading model):\s*(\S.*)')
    _LOAD_DONE_RE = re.compile(r'Model loaded successfully|failed to load|Error loading')

    def _pump_logs(self, tag, proc, tail, engine):
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            tail.append(line)
            self._note_load_progress(engine, line)
            self._emit_log(tag, line)

    def _emit_log(self, tag, line):
        """Print an engine log line, rendering tqdm progress bars as a single
        in-place updating line (carriage return) on a TTY instead of one line per
        update. On a non-TTY (redirected to a file) we keep newlines but throttle
        to whole-percent changes so the log isn't flooded."""
        pm = self._PROGRESS_RE.match(line)
        is_progress = bool(pm)
        ts = time.strftime('%H:%M:%S')
        with self._log_lock:
            tty = False
            try:
                tty = sys.stdout.isatty()
            except Exception:
                pass
            if is_progress and tty:
                # Overwrite the current line; pad to clear any longer previous one.
                print(f"\r[{ts}][{tag}] {line}\033[K", end="", flush=True)
                self._log_progress_tag = tag
                return
            if is_progress:
                # Non-TTY: only emit when the integer percent advanced.
                try:
                    pct = int(round(int(pm.group(2)) / max(1, int(pm.group(3))) * 100))
                except Exception:
                    pct = -1
                if pct == self._log_last_pct.get(tag):
                    return
                self._log_last_pct[tag] = pct
            # A normal line: finalise any in-place progress line first.
            if self._log_progress_tag is not None:
                print(flush=True)
                self._log_progress_tag = None
            print(f"[{ts}][{tag}] {line}", flush=True)

    def _note_load_progress(self, engine, line):
        """Track model-load progress from the engine's log stream so the front can
        show a 'loading' task even while the engine's event loop is GIL-blocked and
        unpollable. Best-effort string matching; never raises."""
        try:
            m = self._LOAD_START_RE.search(line)
            if m:
                engine.loading = {"model": m.group(1).strip().split("  ")[0],
                                  "message": "Loading", "step": 0, "total": 0}
                return
            if self._LOAD_DONE_RE.search(line):
                engine.loading = None
                return
            pm = self._PROGRESS_RE.match(line)
            if pm:
                desc = pm.group(1).strip() or "Loading"
                # Only treat known model-load bars as a load (ignore unrelated tqdm,
                # e.g. LoRA training steps which have their own task).
                if engine.loading is None:
                    if not re.search(r'Loading (weights|checkpoint shards|'
                                     r'pipeline components|shards)|Fetching', desc,
                                     re.IGNORECASE):
                        return
                    engine.loading = {"model": "", "message": "", "step": 0, "total": 0}
                step, total = int(pm.group(2)), int(pm.group(3))
                engine.loading.update(
                    message=f"{desc}: {step}/{total}", step=step, total=total)
        except Exception:
            pass

    # -------------------------------------------------------------------- lifecycle
    def _set_primary(self, engines) -> None:
        """The primary engine owns admin/sessions/config. Honour the configured
        engine (server.default_engine) as the primary when it's present; otherwise
        keep the first engine (the build order) as primary."""
        de = (getattr(self.config.server, "default_engine", None) or "").strip().lower()
        if not de or len(engines) < 2:
            return
        match = next((e for e in engines
                      if e.name.lower() == de or (e.backend or "").lower() == de), None)
        if match is None:
            return   # configured engine isn't present — leave the default primary
        for e in engines:
            e.primary = (e is match)
        print(f"[front] primary engine: '{match.name}' (from settings.default_engine)",
              flush=True)

    def start(self) -> None:
        engines = self._build_engines()
        self._set_primary(engines)       # configured engine owns admin/sessions
        self._assign_models(engines)     # set CODERAI_ENGINE_MODELS before spawning
        for engine in engines:
            self.registry.add(engine)
            self._spawn(engine)
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        atexit.register(self.stop_all)

    def _push_assignment_if_changed(self, client) -> None:
        """When models.json changes (admin add/remove model), re-compute the
        per-engine assignment and push it live to every engine + the front's
        router, so /v1/models and routing reflect the change without a restart."""
        if not self.models_path or len(self.registry.all()) < 2:
            return
        try:
            mtime = os.path.getmtime(self.models_path)
        except OSError:
            return
        if mtime == self._assign_mtime:
            return
        self._assign_mtime = mtime
        try:
            from codai.frontproxy.assignment import compute_assignment
            default_engine = getattr(self.config.server, "default_engine", None)
            ds4 = getattr(self.config, "ds4", None)
            assignment = compute_assignment(self.registry.all(), self.models_path,
                                            default_engine, ds4)
        except Exception as exc:
            print(f"[front] live reassignment skipped: {exc}", flush=True)
            return
        for e in self.registry.all():
            owned = assignment.get(e.name, [])
            e.assigned_models = set(owned)   # update the front's router
            try:
                client.post(e.url + "/internal/reload-config", json={"assigned": owned})
            except Exception as exc:
                print(f"[front] reload-config push to '{e.name}' failed: {exc}",
                      flush=True)
        print(f"[front] models.json changed — re-pushed assignment "
              f"({', '.join(f'{e.name}:{len(assignment.get(e.name, []))}' for e in self.registry.all())})",
              flush=True)

    def _poll_loop(self) -> None:
        _auth = ({"x-coderai-internal": self.internal_token}
                 if self.internal_token else {})
        client = httpx.Client(timeout=self.config.server.proxy_status_timeout,
                              headers=_auth)
        try:
            self._assign_mtime = (os.path.getmtime(self.models_path)
                                  if self.models_path else 0.0)
        except OSError:
            self._assign_mtime = 0.0
        while not self._stopped.is_set():
            self._push_assignment_if_changed(client)
            for engine in self.registry.all():
                # Respawn engines whose process has exited.
                if engine.proc is not None and engine.proc.poll() is not None:
                    self._maybe_restart(engine)
                    continue
                healthy = False
                try:
                    r = client.get(engine.url + "/internal/engine-state")
                    if r.status_code == 200:
                        d = r.json()
                        healthy = True
                        loaded = d.get("loaded_models") or []
                        self.registry.update_state(
                            engine.id, healthy=True,
                            loaded_models=loaded,
                            vram=d.get("vram"),
                            tasks=d.get("tasks") or [],
                            cooling=d.get("cooling"),
                        )
                        # Backstop: the engine answered, so it's no longer GIL-blocked
                        # loading. Clear the log-parsed loading state (the real task,
                        # if any, now comes through the poll).
                        if engine.loading is not None:
                            engine.loading = None
                    else:
                        self.registry.update_state(engine.id, healthy=False)
                except Exception:
                    # Connection refused / timeout: still-loading or dead. Mark
                    # unhealthy; the process-exit check above handles true death.
                    self.registry.update_state(engine.id, healthy=False)
                # --debug-engine: report health transitions (ready / lost).
                if self.debug and self._health.get(engine.id) != healthy:
                    self._health[engine.id] = healthy
                    print(f"[front] engine '{engine.name}' "
                          f"{'ready' if healthy else 'not responding'}", flush=True)
            self._stopped.wait(self.config.server.proxy_status_timeout)
        client.close()

    def _maybe_restart(self, engine: Engine) -> None:
        with self._restart_lock:
            if self._stopped.is_set():
                return
            code = engine.proc.poll() if engine.proc else None
            tail = " | ".join(list(self._logs.get(engine.id, []))[-3:])
            print(f"[front] engine#{engine.id} exited (code {code}); respawning. {tail}",
                  flush=True)
            self.registry.update_state(engine.id, healthy=False)
            time.sleep(1.0)   # avoid a tight crash loop
            self._spawn(engine)

    def restart_engine(self, engine_id: int, drain_grace: Optional[float] = None) -> bool:
        """Forcibly kill and respawn one engine (e.g. it's stuck in a loop).

        Before killing, mark the engine ``draining`` so the router stops sending it
        NEW requests, and wait up to ``drain_grace`` seconds for in-flight (streaming)
        requests to finish — so a config-triggered bounce doesn't sever active SSE
        streams mid-response. After the grace window any stragglers are dropped.

        Holds the restart lock so the poll loop's own respawn can't double-spawn."""
        engine = self.registry.get(engine_id)
        if engine is None:
            return False
        if drain_grace is None:
            drain_grace = float(getattr(self.config.server,
                                        "engine_restart_drain_grace", 30.0) or 0.0)
        with self._restart_lock:
            proc = engine.proc
            if proc is not None and proc.poll() is None and drain_grace > 0:
                engine.draining = True
                self.registry.update_state(engine_id, healthy=False)
                deadline = time.time() + drain_grace
                waited = False
                while engine.inflight > 0 and time.time() < deadline \
                        and not self._stopped.is_set():
                    if not waited:
                        print(f"[front] draining engine#{engine_id} ({engine.name}): "
                              f"waiting for {engine.inflight} in-flight request(s) "
                              f"(up to {drain_grace:.0f}s)", flush=True)
                        waited = True
                    time.sleep(0.25)
                if engine.inflight > 0:
                    print(f"[front] drain grace elapsed; bouncing engine#{engine_id} "
                          f"with {engine.inflight} request(s) still in flight",
                          flush=True)
                elif waited:
                    print(f"[front] engine#{engine_id} drained cleanly", flush=True)
            try:
                proc = engine.proc
                if proc is not None and proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=8)
                    except Exception:
                        pass
                    if proc.poll() is None:
                        try:
                            proc.kill()
                            proc.wait(timeout=3)
                        except Exception:
                            pass
                self.registry.update_state(engine_id, healthy=False)
                print(f"[front] restarting engine#{engine_id} ({engine.name}) on request",
                      flush=True)
                self._spawn(engine)
            finally:
                engine.draining = False
        return True

    def wait_ready(self, timeout: float = 1800.0) -> bool:
        """Block until at least the primary engine answers (best effort)."""
        deadline = time.time() + timeout
        while time.time() < deadline and not self._stopped.is_set():
            prim = self.registry.primary()
            if prim and prim.healthy:
                return True
            time.sleep(1.0)
        return bool(self.registry.primary())

    def stop_all(self, grace: float = 8.0) -> None:
        """Stop every engine, escalating to SIGKILL of the engine's whole process
        group if it doesn't exit within ``grace`` seconds — so a stuck (e.g.
        mid-CUDA) engine, and any children it spawned (whisper-server, ds4), are
        guaranteed dead. Idempotent and safe to call from a signal handler."""
        self._stopped.set()

        def _signal_group(proc, sig):
            # Engines are started in their own session (setsid), so killing the
            # process group reaps the engine + its grandchildren in one shot.
            try:
                os.killpg(os.getpgid(proc.pid), sig)
            except Exception:
                try:
                    proc.send_signal(sig)
                except Exception:
                    pass

        procs = [(e, e.proc) for e in self.registry.all()
                 if e.proc is not None and e.proc.poll() is None]
        # Phase 1: polite SIGTERM to each group.
        for _engine, proc in procs:
            _signal_group(proc, signal.SIGTERM)
        # Phase 2: wait up to `grace`, then SIGKILL whatever is still alive.
        deadline = time.time() + grace
        for _engine, proc in procs:
            remaining = max(0.0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except Exception:
                pass
        for _engine, proc in procs:
            if proc.poll() is None:
                _signal_group(proc, signal.SIGKILL)
                try:
                    proc.wait(timeout=3)
                except Exception:
                    pass
        self._reap_orphan_workers()
        print("[front] all engines stopped", flush=True)

    def _reap_orphan_workers(self) -> None:
        """SIGKILL any lingering download_worker processes — including orphans
        (``ppid=1``) left by an earlier instance that crashed or was killed before
        the parent-death signal could reap them. They hold huggingface_hub's
        per-blob file lock, so leaving them alive makes the next re-download deadlock
        at 0%. Scans /proc by cmdline so it catches workers this front never
        spawned. POSIX-only; a no-op (and harmless) elsewhere."""
        proc_root = "/proc"
        if not os.path.isdir(proc_root):
            return
        my_pid = os.getpid()
        killed = 0
        for entry in os.listdir(proc_root):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == my_pid:
                continue
            try:
                with open(os.path.join(proc_root, entry, "cmdline"), "rb") as fh:
                    cmdline = fh.read().replace(b"\x00", b" ").decode("utf-8", "replace")
            except Exception:
                continue
            if "codai.admin.download_worker" not in cmdline:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
            except Exception:
                pass
        if killed:
            print(f"[front] reaped {killed} orphaned download worker(s)", flush=True)
