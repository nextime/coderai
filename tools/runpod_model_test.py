#!/usr/bin/env python3
"""Test one model on RunPod without moving anything else.

Two rules, both learned the hard way on a live box.

ONE MODEL, NOT A CAPABILITY. The obvious way to send an image model to a pod is
``remotes.endpoints["images"] = "runpod"`` — and that is a production routing
switch: it moves EVERY request of that kind, not the one under test. An
embeddings remote left on from an earlier test quietly sent real traffic to a
rented pod while the local GPU sat idle. So this writes a ``runpod`` block on the
single model entry instead; per-model placement beats the capability map, which
is what lets two models of the same kind sit in different places.

CONFIRM THE ENGINE HAS IT. Writing models.json does not change where a request
goes: the front pushes a reload and the engine takes it only when idle. A test
that writes and immediately runs is testing the previous configuration — which
produced both a full pass of false "not configured" failures and, worse, a pass
that would have reported success for a request served in the wrong place. So it
polls /v1/models/test/state until the serving process reports the placement it
just wrote, and refuses to test if that never happens.

Usage:
    python3 tools/runpod_model_test.py <model> [<model> ...]
    python3 tools/runpod_model_test.py --engine vllm Qwen/Qwen2.5-0.5B-Instruct

models.json is restored on the way out, including on Ctrl-C or a crash.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

CONFIG_DIR = os.environ.get("CODERAI_CONFIG_DIR",
                            os.path.expanduser("~/.coderai"))
MODELS = os.path.join(CONFIG_DIR, "models.json")
BASE_URL = os.environ.get("CODERAI_URL", "http://127.0.0.1:8000")

#: How long to wait for the engine to take a config push. Deferred while the
#: engine is busy, so this is generous — but finite: silently testing a stale
#: config is the failure this exists to prevent.
CONFIG_TIMEOUT_S = 300


def _token() -> str:
    tok = os.environ.get("CODERAI_TOKEN", "").strip()
    if tok:
        return tok
    with open(os.path.join(CONFIG_DIR, "auth.json")) as fh:
        tokens = json.load(fh).get("tokens") or []
    for t in tokens:
        val = t.get("token") if isinstance(t, dict) else t
        if val:
            return val
    raise SystemExit("no API token found — set CODERAI_TOKEN")


def _curl(path: str, payload=None, timeout=2400):
    cmd = ["curl", "-s", "-m", str(timeout),
           "-H", f"Authorization: Bearer {_token()}"]
    if payload is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(payload)]
    cmd.append(BASE_URL.rstrip("/") + path)
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    try:
        return json.loads(out)
    except Exception:
        return {"error": f"unparseable response: {out[:300]}"}


def _load_models() -> dict:
    with open(MODELS) as fh:
        return json.load(fh)


def _save_models(data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(MODELS), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, MODELS)          # atomic: never leave a half-written catalogue


def _find(data: dict, model: str):
    """(section, index, entry) for a model named by path, alias or basename."""
    name = model.strip().lower()
    for section, entries in data.items():
        if not isinstance(entries, list):
            continue
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path") or "")
            cands = {path.lower(), os.path.basename(path.rstrip("/")).lower(),
                     str(entry.get("alias") or "").lower()}
            if name in cands:
                return section, i, entry
    return None, None, None


#: Picked by name rather than registered like a model, so there is nothing in the
#: catalogue to pin. A temporary entry gives them the same per-model placement —
#: the alternative was moving the whole OCR capability, which is a production
#: routing switch.
_OCR_ENGINES = ("paddle", "doctr", "surya")


def _pin_to_runpod(model: str, engine: str, vram: float, usd: float,
                   image: str = "", add_as: str = "",
                   capability: str = "") -> bool:
    """Give this ONE model a runpod block. True when the catalogue changed."""
    data = _load_models()
    section, index, entry = _find(data, model)
    if entry is None and model.strip().lower() in _OCR_ENGINES:
        add_as = add_as or "ocr_models"
    if entry is None and add_as:
        # Testing a capability whose images nobody has a model configured for
        # (music generation, a small video model) should not require editing the
        # real catalogue by hand first. The entry is temporary like every other
        # change this makes, and restored on the way out.
        section = add_as
        data.setdefault(section, [])
        data[section].append({"path": model, "model_type": add_as})
        index = len(data[section]) - 1
        entry = data[section][index]
        print(f"   {model}: added a temporary {add_as} entry", flush=True)
    if entry is None:
        print(f"   {model}: not in models.json — nothing to place", flush=True)
        return False
    entry = dict(entry)
    if capability:
        # Section alone maps XTTS to 'tts', but voice cloning is a different
        # endpoint served by a different pod image. An explicit capability is
        # the only way to place it as what it actually is.
        entry["capability"] = capability
    entry["backend"] = "runpod"
    entry["runpod"] = {"mode": "pods", "engine": engine, "min_vram_gb": vram,
                       "max_hourly_usd": usd, "max_pods": 1, "idle_timeout_s": 120}
    if image:
        # Test against an IMMUTABLE tag. A machine that already pulled `:latest`
        # may serve its cached copy, so a test against a floating tag cannot say
        # which build it exercised — an images pod reported a dependency missing
        # that had already been published under that very tag. Production still
        # runs `:latest`; this override exists so a result means something.
        entry["runpod"]["image"] = image
    data[section][index] = entry
    _save_models(data)
    os.utime(MODELS, None)           # the front watches this mtime to push a reload
    return True


def _wait_for_engine(model: str) -> str:
    """Block until the SERVING process reports this model pinned to a pod.

    Returns '' on success, or why it never happened.
    """
    deadline = time.time() + CONFIG_TIMEOUT_S
    last = {}
    while time.time() < deadline:
        state = _curl(f"/v1/models/test/state?model={model}", timeout=30)
        last = state
        if state.get("placement") == "pod":
            return ""
        time.sleep(5)
    return (f"the engine never picked up the config for {model!r} within "
            f"{CONFIG_TIMEOUT_S}s (it is pushed only when the engine is idle). "
            f"Last seen: placement={last.get('placement')!r} "
            f"backend={last.get('backend')!r} "
            f"has_runpod_block={last.get('has_runpod_block')!r}")


def _reap() -> None:
    """Terminate every pod this account has. A capability pod idles for 20
    minutes before the reaper takes it, which is real money between tests."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from codai.config import ConfigManager
        from codai.api.runpod_client import RunpodClient
        import codai.api.runpod_worker as rw
        cm = ConfigManager(CONFIG_DIR)
        cm.load()
        client = RunpodClient(cm.config.runpod)
        for pod in client.list_pods() or []:
            client.terminate_pod(pod["id"])
            rw.unregister_pod(pod["id"])
            print(f"   reaped {pod['id']} {pod.get('name')}", flush=True)
    except Exception as exc:
        print(f"   could not reap pods: {exc}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--engine", default="auto",
                    help="auto | vllm | llamacpp | coderai (default: auto)")
    ap.add_argument("--vram", type=float, default=16.0, help="min VRAM GB")
    ap.add_argument("--usd", type=float, default=1.0, help="max $/hr per pod")
    ap.add_argument("--keep-pods", action="store_true",
                    help="do not terminate pods after each test")
    ap.add_argument("--add-as", dest="add_as", default="",
                    help="models.json section to create a TEMPORARY entry in "
                         "when the model is not configured, e.g. "
                         "audio_gen_models. Restored afterwards like every "
                         "other change this makes.")
    ap.add_argument("--capability", default="",
                    help="place the model as this capability instead of the one "
                         "its section implies (voice cloning runs on a TTS "
                         "model but is a different image).")
    ap.add_argument("--image", default="",
                    help="exact pod image to run, e.g. "
                         "ghcr.io/nextime/coderai-embeddings:0.2.12. Use a "
                         "VERSIONED tag when testing: a machine may serve a "
                         "cached ':latest' and the result would not say which "
                         "build it exercised.")
    args = ap.parse_args()

    backup = MODELS + ".testrun-backup"
    shutil.copy(MODELS, backup)
    restored = False

    def restore(*_):
        nonlocal restored
        if not restored:
            shutil.copy(backup, MODELS)
            os.utime(MODELS, None)
            os.remove(backup)
            restored = True
            print("== models.json restored", flush=True)

    signal.signal(signal.SIGTERM, lambda *a: (restore(), sys.exit(1)))

    results = []
    try:
        for model in args.models:
            print(f"================ {model} ================", flush=True)
            if not _pin_to_runpod(model, args.engine, args.vram, args.usd,
                                  args.image, args.add_as, args.capability):
                results.append((model, {"error": "not in models.json"}))
                continue
            problem = _wait_for_engine(model)
            if problem:
                print(f"   {problem}", flush=True)
                results.append((model, {"error": problem}))
                continue
            result = _curl("/v1/models/test", {"model": model, "where": "runpod"})
            for key in ("capability", "where", "target", "ran", "status", "ok",
                        "seconds", "sample", "detail", "note", "error"):
                if result.get(key) not in (None, ""):
                    print(f"   {key:10} {result[key]}", flush=True)
            for pod in result.get("pods") or []:
                print(f"   pod        {pod.get('pod')} {pod.get('gpu')} "
                      f"{pod.get('state')} ${pod.get('usd_so_far')} "
                      f"serves={len(pod.get('serves') or [])}", flush=True)
            results.append((model, result))
            # Put the catalogue back BEFORE the next model, so a crash between
            # tests cannot leave a model pinned to a pod that no longer exists.
            shutil.copy(backup, MODELS)
            os.utime(MODELS, None)
            if not args.keep_pods:
                _reap()
    finally:
        restore()

    print("\n================ SUMMARY ================", flush=True)
    for model, result in results:
        mark = "PASS" if result.get("ok") else "FAIL"
        note = (result.get("sample") or result.get("detail")
                or result.get("error") or "")
        print(f"{mark:5} {model:52.52} {str(result.get('ran', '-')):24.24} "
              f"{note}"[:190], flush=True)
    return 0 if all(r.get("ok") for _, r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
