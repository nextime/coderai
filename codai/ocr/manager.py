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

"""OCR engine manager — per-engine instance pools for concurrent GPU OCR.

An "instance" is one loaded engine; for the VLM engine (olmOCR) it is one in-flight
request against the serving model, so the pool size still bounds that engine's
concurrency.

Each engine is loaded as N resident instances (``*_instances`` in
:class:`~codai.config.OcrConfig`); pages are fanned across the pool so many documents
OCR concurrently on one GPU. An engine instance handles one page at a time (OCR runtimes
are not reentrant), so pool size bounds that engine's concurrency; ``max_concurrency``
bounds the whole subsystem.
"""

import asyncio
import os
import threading
import time
from typing import Dict, List, Optional

from codai.ocr.base import (OcrEngine, OcrPage, OcrError, ink_coverage,
                            PAGE_OPTIONS, page_estimate, render_pages,
                            iter_pdf_page_pdfs, pdf_page_count, split_pdf_page,
                            unreadable_report)
from codai.ocr.budget import get_budget


# Engine registry: name → factory(cfg) → OcrEngine. Imports are deferred inside each
# factory so a missing OCR dependency never breaks manager import.
def _paddle_factory(cfg):
    from codai.ocr.paddle import PaddleEngine
    return PaddleEngine(cfg)


def _doctr_factory(cfg):
    from codai.ocr.doctr import DoctrEngine
    return DoctrEngine(cfg)


def _surya_factory(cfg):
    from codai.ocr.surya import SuryaEngine
    return SuryaEngine(cfg)


def _olmocr_factory(cfg):
    from codai.ocr.olmocr import OlmOcrEngine
    return OlmOcrEngine(cfg)


def _chandra_factory(cfg):
    from codai.ocr.chandra import ChandraEngine
    return ChandraEngine(cfg)


_ENGINE_FACTORIES = {
    "paddle": _paddle_factory,
    "doctr": _doctr_factory,
    "surya": _surya_factory,
    "olmocr": _olmocr_factory,
    "chandra": _chandra_factory,
}


#: What a request means when it names no language: let the engine recognise
#: whatever is in front of it. Not a guess at a particular language — a server
#: that has not been told what it is reading should not assume one.
#:
#: For surya-2, olmOCR and docTR this is simply the normal mode: they are
#: multilingual and have no language input to constrain (surya-2's whole prompt
#: is ``"OCR this block image to HTML."``). Only PaddleOCR must be handed a
#: concrete language, and ``AUTO_PADDLE_LANG`` is what it gets.
DEFAULT_LANG = "auto"

#: Values a caller may use to mean DEFAULT_LANG explicitly.
_AUTO_ALIASES = {"", "auto", "any", "multi", "multilingual", "detect"}

#: What ``auto`` becomes for PaddleOCR, which has no auto mode and must be
#: constructed for one language.
#:
#: This is a SCRIPT family, not detection: PaddleOCR's ``latin`` model covers
#: Italian, French, Spanish, German, Portuguese and the rest of the Latin
#: alphabet, and most ISO codes it accepts (``it``, ``es``, ...) resolve to it
#: anyway. It is the honest broad default — PaddleOCR's own is ``ch``, which
#: would be a worse guess for most documents than no guess at all. A document in
#: another script needs its language named (``lang=arabic``, ``lang=japan``,
#: ``lang=cyrillic``), or one of the multilingual engines, which need nothing.
#: Override with ``ocr.paddle_auto_lang``.
AUTO_PADDLE_LANG = "latin"

#: What this engine's ``conf`` actually is. ``scope`` says what one number
#: covers, which is not cosmetic: a per-line score can single out the line to
#: re-read, while a page-level one can only say the page went badly. The VLMs
#: report the mean of ``exp(logprob)`` over one generation's tokens (the same
#: computation surya's own client does), and surya calls per block where chandra
#: and olmOCR generate the whole page at once. ``measured`` false would mean the
#: number is invented; nothing here does that, but the per-page
#: ``confidence_measured`` flag still reports when a server declined logprobs and
#: left the figure at 0.0.
_CONFIDENCE = {
    "paddle":  {"measured": True, "scope": "line", "source": "recogniser score"},
    "doctr":   {"measured": True, "scope": "line", "source": "recogniser score"},
    "surya":   {"measured": True, "scope": "block", "source": "mean token probability"},
    "chandra": {"measured": True, "scope": "page", "source": "mean token probability"},
    "olmocr":  {"measured": True, "scope": "page", "source": "mean token probability"},
}


#: Engines whose LOADED instance is built for ONE language, so serving another
#: needs a separate pool.
#:
#: Only PaddleOCR: it loads a language-specific recognition model, and the
#: options are sent once at load (``{"cmd": "load", "opts": ...}``) — an instance
#: cannot be re-languaged afterwards. docTR's recognition is Latin-script-wide,
#: and surya-2 and olmOCR are VLMs whose prompt has no language slot at all
#: (surya-2's is the seven words ``"OCR this block image to HTML."``, and its own
#: module warns the wording is the training-time contract).
#:
#: This set must stay as small as the truth allows. A language-keyed pool is a
#: SECOND set of loaded instances holding VRAM — a production install runs 24
#: surya instances, and multiplying those per language would be ruinous for a
#: parameter those engines cannot use.
_LANG_SENSITIVE = {"paddle"}


def normalize_lang(value) -> str:
    """A language tag reduced to the form engines expect, with the ways of saying
    "work it out yourself" all collapsed onto ``auto``."""
    text = str(value or "").strip().lower().replace("_", "-")
    return DEFAULT_LANG if text in _AUTO_ALIASES else text


def resolve_lang(cfg, name: str, requested=None) -> str:
    """The language this request is read in, or ``auto``.

    Precedence is request > engine-specific config > global config > ``auto``. The
    REQUEST wins over ``<engine>_lang`` deliberately: a per-engine setting is an
    install's default for that engine, while a caller who names a language is
    telling us something about the document in hand, which is better information.

    A request that explicitly says ``auto`` means auto — it is not treated as
    "unset" and does not fall back to the configured language. Asking for
    auto-recognition on an install configured for Italian has to be possible.
    """
    want = normalize_lang(requested)
    if str(requested or "").strip():
        return want
    per_engine = normalize_lang(getattr(cfg, f"{name}_lang", ""))
    if per_engine != DEFAULT_LANG:
        return per_engine
    return normalize_lang(getattr(cfg, "lang", ""))


def engine_lang(cfg, name: str, lang: str) -> str:
    """``lang`` in the form the engine itself can be built with.

    Only PaddleOCR needs the translation: it has no auto mode, so ``auto`` becomes
    a broad script model. The multilingual engines never reach here.
    """
    if lang != DEFAULT_LANG:
        return lang
    if name == "paddle":
        # Not `normalize_lang(...) or AUTO_PADDLE_LANG`: normalize_lang("") is
        # "auto", which is truthy, so an unset override would have returned the
        # very value this exists to translate.
        configured = normalize_lang(getattr(cfg, "paddle_auto_lang", ""))
        return AUTO_PADDLE_LANG if configured == DEFAULT_LANG else configured
    return lang


def engine_uses_language(name: str) -> bool:
    """Whether ``name`` actually reads differently when told a language.

    Reported to clients so a caller can tell a language that was applied from one
    that was accepted and could not be: silently ignoring ``lang`` would be the
    worse answer.
    """
    return name in _LANG_SENSITIVE


def _engine_enabled(cfg, name: str) -> bool:
    if name == "paddle":
        return bool(cfg.paddle_enabled)
    if name == "doctr":
        return bool(cfg.doctr_enabled)
    if name == "surya":
        return bool(cfg.surya_enabled and cfg.surya_accept_license)
    if name == "olmocr":
        return bool(getattr(cfg, "olmocr_enabled", False))
    if name == "chandra":
        # License-gated like surya, and on narrower terms: the weights are a
        # modified OpenRAIL-M, not GPL (see OcrConfig.chandra_accept_license).
        return bool(getattr(cfg, "chandra_enabled", False)
                    and getattr(cfg, "chandra_accept_license", False))
    return False


#: Engines that spawn ONE OS PROCESS per instance, in their own venv. For these
#: ``*_instances`` is not a concurrency slot count — it is a process count, and
#: each process costs ~600 MB of RSS and a core's worth of page rendering. The
#: in-process engines (chandra, olmOCR) cost a slot and nothing more.
_PROCESS_PER_INSTANCE = ("surya", "paddle", "doctr")

#: Measured RSS of one idle surya worker on Aruba, 2026-10-12: 586-677 MB across
#: 64 of them. Rounded up for headroom, because the figure that matters is the
#: one *after* a page has been rendered in it.
_WORKER_RSS_GB = 0.8


def _host_instance_ceiling(name: str) -> int:
    """How many worker PROCESSES this host can actually run for one engine.

    `surya_instances: 64` was set on Aruba as though it were the GPU-side
    concurrency (it reads like it), and on 2026-10-12 it did exactly what it
    says: 64 worker processes, 37 GB of RSS, swap exhausted, load 25 on 16
    cores — on a box that also runs ~30 other containers. A batch then behaved
    precisely as the operator reported: one perfect 19-second burst, then no
    completions at all for nine minutes while the rented A40 sat at 2% KV-cache
    usage with `Waiting: 0`. The GPU was never the bottleneck; the host was.

    GPU-side concurrency is `runpod.scale_up_inflight_per_pod` and
    `runpod.max_pods`. This number is a local process pool and has to be bounded
    by local resources, so it is: one worker per core, and never more than half
    of free RAM. `CODERAI_OCR_INSTANCE_CEILING` overrides it for an operator who
    has measured their own host.
    """
    forced = (os.environ.get("CODERAI_OCR_INSTANCE_CEILING") or "").strip()
    if forced:
        try:
            return max(1, int(forced))
        except ValueError:
            pass
    if name not in _PROCESS_PER_INSTANCE:
        return 0                      # in-process: a slot, not a process
    try:
        by_cpu = max(1, os.cpu_count() or 4)
    except Exception:
        by_cpu = 4
    by_ram = by_cpu
    try:
        # Free RAM, not total: the rest of the host is someone else's working set.
        with open("/proc/meminfo") as fh:
            avail_kb = next(int(l.split()[1]) for l in fh
                            if l.startswith("MemAvailable:"))
        by_ram = max(1, int((avail_kb / 1048576.0) * 0.5 / _WORKER_RSS_GB))
    except Exception:
        pass
    return max(1, min(by_cpu, by_ram))


def _engine_instances(cfg, name: str) -> int:
    n = getattr(cfg, f"{name}_instances", 1)
    try:
        want = max(1, int(n))
    except Exception:
        want = 1
    ceiling = _host_instance_ceiling(name)
    if ceiling and want > ceiling:
        print(f"[ocr] engine {name!r}: {want} instances asked for, running "
              f"{ceiling} — each one is a worker PROCESS (~{_WORKER_RSS_GB} GB) "
              f"and this host has {os.cpu_count()} cores. GPU-side concurrency "
              f"is runpod.scale_up_inflight_per_pod / max_pods; raise "
              f"CODERAI_OCR_INSTANCE_CEILING to override.", flush=True)
        return ceiling
    return want


def _fleet_shape(name: str):
    """``(inflight_per_pod, max_pods)`` for the model this engine reads from.

    Taken from the model's own ``runpod`` block, which is where per-pod
    concurrency lives — one knob per model, not an OCR setting.
    """
    try:
        from codai.models.manager import _model_entry_for
        from codai.api.runpod_worker import parse_model_runpod
    except Exception:
        return None
    for served in _SERVED_NAMES.get(name, ()):
        try:
            entry = _model_entry_for(served)
            block = (entry or {}).get("runpod")
            if not isinstance(block, dict) or not block:
                continue
            mcfg = parse_model_runpod(block)
            if (mcfg.mode or "").lower() != "pods":
                continue
            per = int(getattr(mcfg, "scale_up_inflight_per_pod", 0) or 0)
            if per <= 0:
                per = int(getattr(mcfg, "max_inflight_per_pod", 0) or 0)
            if per > 0:
                return per, max(1, int(getattr(mcfg, "max_pods", 1) or 1))
        except Exception:
            continue
    return None


#: The served model names an engine may be reading from, for the fleet estimate.
_SERVED_NAMES = {"surya": ("datalab-to/surya-ocr-2", "surya-ocr-2"),
                 "chandra": ("datalab-to/chandra-ocr-2", "chandra-ocr-2"),
                 "olmocr": ("allenai/olmOCR-2-7B-1025-FP8", "olmocr")}

#: Measured on the Digesta orchestrator: ~7 min image pull + ~5 min warm-up.
_POD_READY_MINUTES = 12


def _warn_if_oversubscribed(name: str, pages: int) -> None:
    """Say how many pods this many pages implies, at the moment they are set up.

    Not a limit — the pool queues the surplus perfectly correctly. But pages
    beyond what the READY pods can hold do not go faster, they only wait, and
    because a pod takes ~12 minutes to appear the fleet cannot be scaled into a
    burst: the first burst is served by whatever is already warm, however high
    ``max_pods`` is. That matters most to a client counting finished DOCUMENTS,
    which completes none of them until one document's LAST page lands — so 64
    pages spread over sixteen documents reads from outside as a stall while
    every pod is busy. Exactly how the 2026-10-13 batch was reported, with the
    pod at 18 concurrent requests, ``Waiting: 0`` and 5-7% KV cache throughout.
    """
    shape = _fleet_shape(name)
    if not shape:
        return
    per, max_pods = shape
    need = (pages + per - 1) // per
    tail = (f"a pod takes ~{_POD_READY_MINUTES} min to become ready, so the "
            f"first burst is served by whatever is already warm")
    if need > max_pods:
        print(f"[ocr] engine {name!r}: {pages} pages at once needs {need} pods at "
              f"{per} in flight each, but max_pods is {max_pods} — "
              f"{max_pods * per} will be served and the rest will queue. "
              f"Lower instances/lanes, or raise max_pods.", flush=True)
    elif need > 1:
        print(f"[ocr] engine {name!r}: {pages} pages at once spreads over {need} "
              f"pods ({per} in flight each); {tail}.", flush=True)


def _post_page_to_pod(base: str, api_key: str, blob: bytes, name: str,
                      dpi: int, opts: dict, detect, engine: str = "surya") -> dict:
    """POST one page to a pod's own /v1/ocr and return its response dict.

    Through `pod_http`, not `requests`: a pod on the direct path serves HTTPS
    with a certificate signed by THIS install's CA, addressed by an IP that was
    not known when it was issued. Plain requests fails that twice over, which is
    the same trap olmOCR fell into (see tests/test_olmocr_pod_tls.py).
    """
    from codai.api import pod_http
    url = str(base or "").rstrip("/")
    if url.endswith("/v1"):
        url = url + "/ocr"
    else:
        url = url + "/v1/ocr"
    # The engine the POD must use. Hardcoded "surya" sent a chandra page to
    # surya, which the pod would have served with the wrong model -- and the
    # text would have come back looking plausible.
    fields = {"engine": str(engine or "surya")}
    if dpi:
        fields["dpi"] = str(int(dpi))
    if detect:
        fields["detect"] = str(detect)
    for k_src, k_dst in (("max_tokens", "max_tokens"),
                         ("max_continuations", "max_continuations")):
        v = (opts or {}).get(k_src)
        if v:
            fields[k_dst] = str(int(v))
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        r = pod_http.post(url, files={"file": (name, blob, "application/pdf")},
                          data=fields, headers=headers, timeout=600)
    except Exception as e:
        raise OcrError(f"pod OCR at {url} is unreachable: {e}", status=503)
    if r.status_code >= 400:
        body = ""
        try:
            body = (r.text or "")[:200]
        except Exception:
            pass
        raise OcrError(f"pod OCR at {url} answered {r.status_code}: {body}",
                       status=int(r.status_code))
    try:
        return r.json()
    except Exception as e:
        raise OcrError(f"pod OCR at {url} returned no JSON: {e}", status=502)


def _page_from_pod(d: dict, index: int) -> OcrPage:
    """One page of a pod's /v1/ocr response as an OcrPage.

    Carries the DIAGNOSTICS across, not just the text: `repeat_detected` and the
    rest are the whole reason a caller can tell a looping page from a blank one,
    and dropping them at this boundary would make a pod-read page less
    explicable than a locally-read one for no reason.
    """
    page = OcrPage(index=index, text=str(d.get("text") or ""),
                   width=int(d.get("width") or 0),
                   height=int(d.get("height") or 0))
    meta = dict(d.get("meta") or {})
    for k in ("repeat_detected", "repeat_chars", "model_attempts", "confidence",
              "confidence_measured", "unreadable", "blank", "ink",
              "finish_reason", "continuations"):
        if k in d:
            meta[k] = d[k]
    if meta:
        page.meta = meta
    return page


#: How long a pool stays marked as unable to run the pipeline on its own pods
#: before it is tried again. Long enough that a document does not pay a failed
#: round trip per page, short enough that fixing the pods (a rebuilt image, a
#: corrected seed) takes effect without restarting the orchestrator.
POD_PIPELINE_COOLDOWN_S = 300.0

#: model id -> (until, reason). See _read_on_pods().
_POD_PIPELINE_UNUSABLE: dict = {}
_POD_PIPELINE_LOCK = threading.Lock()


def _pod_pipeline_refusal(mid: str) -> str:
    """Why this pool's pods were set aside, or "" if they are to be tried."""
    with _POD_PIPELINE_LOCK:
        hit = _POD_PIPELINE_UNUSABLE.get(mid)
        if not hit:
            return ""
        until, reason = hit
        if until <= time.time():
            _POD_PIPELINE_UNUSABLE.pop(mid, None)
            return ""
        return reason


def _set_aside_pod_pipeline(mid: str, reason: str) -> bool:
    """Stop sending pages to this pool's own pipeline for a while.

    Returns True the first time, so the caller says it once per cooldown
    instead of once per page.
    """
    now = time.time()
    with _POD_PIPELINE_LOCK:
        hit = _POD_PIPELINE_UNUSABLE.get(mid)
        fresh = not hit or hit[0] <= now
        _POD_PIPELINE_UNUSABLE[mid] = (now + POD_PIPELINE_COOLDOWN_S, reason)
    return fresh


#: Answers that mean "this pod cannot run the pipeline", as opposed to "this
#: page went wrong". A pod that does not serve /v1/ocr, cannot attach its OCR
#: engine to the model it is serving, or has no model to attach to will say the
#: same thing for every page of every document -- so the pages belong on the
#: local pipeline, not in a retry loop.
_POD_CANNOT_RUN_PIPELINE = (404, 501, 503)


#: (engine, model, image) combinations already explained. The gate is asked
#: once per document, and the answer only changes when the configuration does,
#: so saying it every time would bury it; saying it never is what let a whole
#: deployment rasterise locally for two days without a word.
_PIPELINE_GATE_SAID: set = set()


def _say_pipeline_gate(name: str, mid: str, image: str, runs: bool) -> None:
    """Report, once, WHERE this engine's pipeline runs and what decided it."""
    key = (name, mid, image, runs)
    if key in _PIPELINE_GATE_SAID:
        return
    _PIPELINE_GATE_SAID.add(key)
    if runs:
        print(f"[ocr] {name!r}: pods for {mid!r} run {image!r}, which carries "
              f"the rasteriser — pages go there as one-page PDFs and this box "
              f"never rasterises", flush=True)
    elif image:
        print(f"[ocr] {name!r}: pods for {mid!r} run {image!r}, which serves "
              f"the model only — THIS BOX rasterises and runs the pipeline. "
              f"To move it to the pods, name an image carrying it (set "
              f"`engine: coderai-vllm-ocr` on the model's runpod block, or "
              f"`image` explicitly)", flush=True)
    else:
        print(f"[ocr] {name!r}: could not work out which image the pods for "
              f"{mid!r} run, so the pipeline stays on this box — check the "
              f"model's runpod block has an `engine` or an `image`", flush=True)


def pod_pipeline_model(cfg, engine: str = "") -> str:
    """The model id whose pods run the OCR pipeline themselves, or "".

    Answers for whichever VLM engine is being used -- surya, chandra or
    olmOCR -- each read from its own `<engine>_serve` / `<engine>_model_id`.

    THE IMAGE IS THE SWITCH. A model served by `coderai-vllm-ocr` carries surya
    and the rasteriser, so the page goes to it as a one-page PDF; one served by
    `coderai-vllm` serves only the model and the pipeline stays here. There is
    no second setting, because two switches that can disagree is the kind of
    invisible mismatch that gets blamed on a pod image.

    One switch, but it has to be READ the way a launch reads it: the image is
    usually implied by `runpod.engine` rather than written out, so this asks
    `pod_image_for_entry()` for the image a pod would actually be given.

    Returns "" for every other case -- no model mode, no entry, a different
    image -- so the caller simply does what it has always done.
    """
    name = (engine or "").strip().lower() or str(
        getattr(cfg, "default_engine", "") or "").strip().lower()
    # All three VLM engines can run on the pod, not just surya. The image is
    # what differs: surya needs its client venv baked in, while chandra and
    # olmOCR in served mode speak only HTTP (PIL + requests, already present),
    # so for them the pod side costs nothing beyond the rasteriser. Leaving
    # them on the orchestrator while surya moved was an asymmetry, not a
    # design.
    if name not in ("surya", "chandra", "olmocr"):
        return ""
    if str(getattr(cfg, f"{name}_serve", "") or "").strip().lower() != "model":
        return ""
    mid = str(getattr(cfg, f"{name}_model_id", "") or "").strip()
    if not mid:
        return ""
    try:
        from codai.api.runpod_worker import (_model_entry,
                                             image_runs_ocr_pipeline,
                                             pod_image_for_entry)
        entry = _model_entry(mid) or {}
        # The image a LAUNCH would use, not the raw `image` field. `image` is
        # optional and normally absent -- `engine: coderai-vllm-ocr` is how the
        # image is chosen -- so reading the field answered "" for the very
        # configuration that asks for the pod-side pipeline, and the whole
        # thing quietly ran here instead. See pod_image_for_entry().
        image = pod_image_for_entry(entry, model_key=mid)
        runs = image_runs_ocr_pipeline(image)
        _say_pipeline_gate(name, mid, image, runs)
        return mid if runs else ""
    except Exception as exc:
        print(f"[ocr] {name!r}: could not decide whether the pods for {mid!r} "
              f"run the pipeline ({exc}); reading on this box", flush=True)
        return ""


def _drain(stream, n: int) -> list:
    """Pull up to ``n`` pages off a page generator. Empty list = document finished.

    Separate from the loop that uses it so the whole pull can be handed to
    `asyncio.to_thread` in one call rather than per page.
    """
    out = []
    for _ in range(max(1, n)):
        try:
            out.append(next(stream))
        except StopIteration:
            break
    return out


def _unread_pages(pages) -> list:
    """Pages with ink on them and no text: read attempts that did not happen.

    Deliberately NOT the same list as `blank_pages`. A blank page is a fact
    about the document; an unread page is a fact about this run, and only one
    of the two is worth retrying.
    """
    out = []
    for page in pages or []:
        meta = getattr(page, "meta", None) or {}
        if (getattr(page, "text", "") or "").strip():
            continue
        why = ""
        if meta.get("engine_empty"):
            why = "the engine returned no tokens for this page"
        else:
            ink = meta.get("page_ink")
            if isinstance(ink, dict) and ink.get("blank") is False:
                why = str(ink.get("why") or "there is ink on this page")
        if why:
            out.append({"page": getattr(page, "index", len(out)), "why": why})
    return out


def _mark_blank_pages(cfg, images, pages) -> list:
    """Attach an ink measurement to any page that came back with little text.

    Returns the indices judged blank. Only short pages are measured: a page that
    read normally needs no explanation, and the measurement is only interesting
    where someone is about to ask why there is no text.
    """
    try:
        max_chars = int(getattr(cfg.ocr, "blank_page_max_chars", 200) or 0)
        floor = float(getattr(cfg.ocr, "blank_page_ink", 0.005) or 0.0)
    except Exception:
        max_chars, floor = 200, 0.005
    if max_chars <= 0:
        return []
    found = []
    for i, page in enumerate(pages):
        try:
            if len((page.text or "").strip()) > max_chars:
                continue
            img = images[i] if i < len(images) else None
            if img is None:
                continue
            ink = ink_coverage(img)
            if ink < 0:
                continue
            info = {"ink": round(ink, 5), "threshold": floor}
            if ink < floor:
                info["blank"] = True
                info["why"] = (
                    f"{ink * 100:.3f}% of this page's pixels are not background, "
                    f"below {floor * 100:.3f}% -- there is essentially nothing on "
                    f"it to read. A short result here is the page being empty, "
                    f"NOT the engine failing to read it; look at the source "
                    f"document.")
                found.append(i)
            else:
                info["blank"] = False
                info["why"] = (
                    f"{ink * 100:.3f}% of this page's pixels are not background, "
                    f"so there IS something on it -- a short result here is the "
                    f"engine failing to read it, not an empty page.")
            page.meta["page_ink"] = info
        except Exception:
            continue
    return found


def _engine_lanes(eng) -> int:
    """How many pages one instance of ``eng`` serves at once, asked politely.

    `OcrEngine` has `lanes()`, but an engine is anything with `recognize_image` —
    the fallback chain and several tests drive duck-typed ones — and a pool must
    not refuse to build because an engine does not know about a capability it
    simply does not have. Unknown means one.
    """
    try:
        return max(1, int(eng.lanes() or 1))
    except Exception:
        return 1


def _evict_for_ocr(needed_gb: float) -> None:
    """Ask the model manager to free ``needed_gb`` of VRAM before an OCR pool loads,
    so OCR instances contend for VRAM on equal footing with LLM/diffusion models
    (evict LRU models rather than OOM). Best-effort: never breaks OCR if the manager
    is unavailable."""
    try:
        if needed_gb <= 0:
            return
        from codai.models.manager import multi_model_manager
        multi_model_manager._evict_models_for_vram(float(needed_gb))
    except Exception as e:
        print(f"[ocr] VRAM evict-before-build skipped: {e}")


def _wait_thermal_safe() -> None:
    """Block until GPU temps are within safe limits (shared thermal governor).
    Runs OCR through the same cooldown gate as every other GPU workload."""
    try:
        from codai.models import thermal
        thermal.wait_until_safe(context="ocr")
    except Exception as e:
        print(f"[ocr] thermal wait skipped: {e}")


class _Pool:
    """A fixed-size pool of loaded engine instances, served via an asyncio.Queue."""

    def __init__(self, name: str, cfg, size: int):
        self.name = name
        self.cfg = cfg
        self.size = size
        self._q: asyncio.Queue = asyncio.Queue()
        self._instances = []          # every created instance (for release_sync)
        self._built = False
        self._build_lock = asyncio.Lock()
        self._fail_at = 0.0           # when the last build attempt failed
        self._fail_err = None         # and why (re-raised during the cooldown)
        # In-flight page accounting, for the clean swap in drain_and_release(). Touched
        # from the event loop (recognize) and READ from the model manager's eviction
        # thread, so it is guarded by a threading primitive rather than an asyncio one.
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._draining = False

    def _enter_page(self) -> None:
        with self._inflight_lock:
            self._inflight += 1
            self._idle.clear()

    def _leave_page(self) -> None:
        with self._inflight_lock:
            self._inflight = max(0, self._inflight - 1)
            if self._inflight == 0:
                self._idle.set()

    def _drain_timeout_s(self) -> float:
        try:
            return max(0.0, float(getattr(self.cfg, "evict_drain_timeout_s", 60.0)))
        except Exception:
            return 60.0

    def drain_and_release(self) -> float:
        """Let in-flight pages finish, THEN tear the pool down. Returns GB freed.

        This is the OCR side of the clean swap the model manager already does for a busy
        model (``release_idle_vram`` waits for a request boundary rather than evicting
        mid-request). Tearing the pool down under a page that is being recognised loses
        that page — and the caller's whole document — which is exactly what must not
        happen when OCR and a model take turns on one card. Bounded by
        ``ocr.evict_drain_timeout_s`` so a wedged engine cannot block the load forever."""
        self._draining = True
        try:
            budget = self._drain_timeout_s()
            with self._inflight_lock:
                busy = self._inflight
            if busy and budget > 0:
                print(f"[ocr] engine '{self.name}': {busy} page(s) in flight — waiting up "
                      f"to {budget:.0f}s for a clean swap before releasing VRAM")
                if not self._idle.wait(budget):
                    with self._inflight_lock:
                        stuck = self._inflight
                    print(f"[ocr] engine '{self.name}': {stuck} page(s) still in flight "
                          f"after {budget:.0f}s — releasing anyway (those pages will be "
                          f"re-read by the fallback engine)")
            return self.release_sync()
        finally:
            self._draining = False

    def _cooldown_s(self) -> float:
        try:
            return max(0.0, float(getattr(self.cfg, "build_retry_cooldown_s", 60.0)))
        except Exception:
            return 60.0

    async def _ensure_built(self):
        if self._built:
            return
        async with self._build_lock:
            if self._built:
                return
            # A release is draining right now: let it finish before rebuilding, or we
            # would re-evict the model that just took its turn and load into VRAM the
            # drain is still about to free. Taking turns means waiting for the handover.
            _waited = 0.0
            while self._draining and _waited < self._drain_timeout_s() + 5.0:
                await asyncio.sleep(0.05)
                _waited += 0.05
            # A build that just failed is very unlikely to succeed on the next request a
            # second later, and retrying is not free: a vLLM-backed engine spends ~35 s
            # booting before it dies. Surya-2 did that 338 times in a row against a card
            # that was 11 GB short — every one of those requests hung for half a minute
            # and then failed anyway. Inside the cooldown, fail FAST with the same reason.
            if self._fail_err is not None and (time.time() - self._fail_at) < self._cooldown_s():
                left = int(self._cooldown_s() - (time.time() - self._fail_at))
                raise OcrError(
                    f"{self._fail_err} (retrying in {left}s — last attempt failed; "
                    f"OCR engine '{self.name}' is in its build cooldown)",
                    status=getattr(self._fail_err, "status", 503))
            factory = _ENGINE_FACTORIES[self.name]
            self._instances = []
            try:
                for _ in range(self.size):
                    eng = factory(self.cfg)
                    first = not self._instances
                    need = 0.0
                    if first:
                        # Engines that boot a server claiming a fixed share of the card
                        # (the VLM engines on vLLM) must have the room BEFORE they load —
                        # the load is what fails otherwise. The figure covers the whole
                        # pool, because those instances share one server.
                        try:
                            need = float(eng.prelaunch_vram_gb() or 0.0)
                        except Exception:
                            need = 0.0
                        if need > 0:
                            await asyncio.to_thread(_evict_for_ocr, need)
                    # Load the first instance synchronously-in-thread so a missing
                    # dependency surfaces as OcrError(503) before we spawn the rest.
                    await asyncio.to_thread(eng.ensure_loaded)
                    # First instance of an engine that did NOT reserve up front: evict for
                    # its real footprint now that we know it (paddle/docTR), per instance.
                    # Reserving again here would evict live models for memory the engine's
                    # server has already taken.
                    if first and need <= 0:
                        await asyncio.to_thread(_evict_for_ocr, self.size * eng.vram_gb())
                    self._instances.append(eng)
                    # A lane is a page this instance can serve CONCURRENTLY, so
                    # the instance is offered once per lane. The queue is what
                    # bounds concurrency, and handing out the same engine twice
                    # is exactly what a multi-lane worker is for.
                    for _ in range(_engine_lanes(eng)):
                        await self._q.put(eng)
            except Exception as e:
                self.release_sync()
                self._fail_at = time.time()
                self._fail_err = e if isinstance(e, OcrError) else OcrError(
                    f"OCR engine '{self.name}' failed to load: {e}", status=503)
                raise self._fail_err
            self._fail_err = None
            self._built = True
            lanes = _engine_lanes(self._instances[0]) if self._instances else 1
            if lanes > 1:
                print(f"[ocr] engine '{self.name}': {self.size} instance(s) ready, "
                      f"{lanes} lane(s) each — {self.size * lanes} pages at once")
            else:
                print(f"[ocr] engine '{self.name}': {self.size} instance(s) ready")
            _warn_if_oversubscribed(self.name, self.size * lanes)

    def release_sync(self) -> float:
        """Tear down all instances (free VRAM). SYNC — safe to call from the model
        manager's eviction thread. Returns estimated GB freed."""
        freed = 0.0
        for eng in list(self._instances):
            try:
                freed += eng.vram_gb()
                eng.cleanup()
            except Exception:
                pass
        self._instances = []
        # Drain the queue so a rebuild starts clean.
        try:
            while not self._q.empty():
                self._q.get_nowait()
        except Exception:
            pass
        self._built = False
        return freed

    async def recognize(self, image) -> OcrPage:
        await self._ensure_built()
        eng: OcrEngine = await self._q.get()
        self._enter_page()
        try:
            # Engines that call back into coderai's async API (olmOCR/Chandra 'model'
            # mode) run in a worker thread and need a live loop to hand the coroutine to;
            # give them the app's own rather than letting them spin up a second one.
            try:
                eng._host_loop = asyncio.get_running_loop()
            except Exception:
                pass
            return await asyncio.to_thread(eng.recognize_image, image)
        finally:
            self._leave_page()
            await self._q.put(eng)


class OcrManager:
    """Holds per-engine pools; (re)configured from OcrConfig on demand."""

    def __init__(self):
        self._cfg = None
        self._pools: Dict[str, _Pool] = {}
        self._sem: Optional[asyncio.Semaphore] = None
        self._lock = asyncio.Lock()
        self._releaser_registered = False

    def configure(self, cfg) -> None:
        """Point the manager at the current OcrConfig. Rebuilds pools if params changed."""
        prev = self._cfg
        self._cfg = cfg
        if prev is None or self._pool_params(prev) != self._pool_params(cfg):
            # Drop stale pools. Tear down live instances first so their VRAM is freed
            # promptly (rather than waiting on GC of the subprocess workers / torch models).
            for pool in self._pools.values():
                try:
                    pool.release_sync()
                except Exception:
                    pass
            self._pools = {}
            self._sem = asyncio.Semaphore(max(1, int(getattr(cfg, "max_concurrency", 4))))
        self._register_releaser()

    def _register_releaser(self) -> None:
        """Register OCR VRAM as reclaimable by the model manager, so loading an
        LLM/diffusion model can evict OCR pools (not just the reverse)."""
        if self._releaser_registered:
            return
        try:
            from codai.models.manager import multi_model_manager
            multi_model_manager.register_external_vram_releaser(self._release_vram)
            self._releaser_registered = True
        except Exception as e:
            print(f"[ocr] could not register VRAM releaser: {e}")

    def _release_vram(self, needed_gb: float = 999.0) -> float:
        """External VRAM releaser (called as ``fn(needed_gb)`` from the model manager's
        eviction path, SYNC). Tears down every built OCR pool and returns the estimated
        GB freed. Pools rebuild lazily on the next OCR request, so OCR takes its turn on
        the card again like any other model rather than staying dead for the rest of the
        process' life. ``needed_gb`` is advisory — OCR pools are all-or-nothing per
        engine, so we release everything held.

        In-flight pages are allowed to FINISH first (see ``_Pool.drain_and_release``):
        the handover is a clean swap at a page boundary, not a kill."""
        freed = 0.0
        for pool in list(self._pools.values()):
            try:
                freed += pool.drain_and_release()
            except Exception:
                pass
        # The managed Surya-2 vLLM subprocess isn't a manager-tracked model and its VRAM
        # (gpu_memory_utilization × card) isn't reclaimed by tearing down the worker pool
        # above — it must be stopped explicitly. Do it here so on-request eviction can
        # reclaim it; the next OCR request re-boots it via ensure_service.
        freed += self._stop_surya_vllm()
        freed += self._stop_olmocr_vllm()
        freed += self._stop_chandra_vllm()
        if freed:
            print(f"[ocr] released ~{freed:.1f} GB (pools + Surya vLLM torn down for VRAM eviction)")
        return freed

    def _stop_surya_vllm(self) -> float:
        """Stop the managed Surya-2 vLLM subprocess (if this build serves Surya via vLLM).
        Returns estimated GB freed."""
        # NOT gated on surya_serve being "vllm" right now: if the setting was changed
        # while a service was up, the old subprocess is still holding its share of the
        # card and skipping the stop would leak it until a restart. stop_service_for() is
        # a no-op returning 0.0 when nothing is running, so asking is free.
        cfg = self._cfg
        if cfg is None:
            return 0.0
        try:
            from codai.api import vllm_worker
            from codai.models.manager import get_active_vllm_config
            vcfg = get_active_vllm_config()
            if vcfg is None:
                return 0.0
            model = (getattr(cfg, "surya_model", "") or "datalab-to/surya-ocr-2").strip()
            return vllm_worker.stop_service_for(
                vcfg, model_path=model, served_name=model,
                gpu_memory_utilization=float(getattr(cfg, "vlm_gpu_memory_utilization", 0.0) or 0.0) or None)
        except Exception as e:
            print(f"[ocr] Surya vLLM stop skipped: {e}")
            return 0.0

    def _stop_olmocr_vllm(self) -> float:
        """Stop the managed olmOCR vLLM instance (olmocr_serve = "vllm"), for the same
        reason Surya's needs stopping: its VRAM is the subprocess's, not a pool's."""
        cfg = self._cfg          # unconditional, for the reason in _stop_surya_vllm
        if cfg is None:
            return 0.0
        try:
            from codai.api import vllm_worker
            from codai.models.manager import get_active_vllm_config
            from codai.ocr.olmocr import DEFAULT_MODEL
            vcfg = get_active_vllm_config()
            if vcfg is None:
                return 0.0
            model = (getattr(cfg, "olmocr_model", "") or DEFAULT_MODEL).strip()
            return vllm_worker.stop_service_for(
                vcfg, model_path=model, served_name=model,
                gpu_memory_utilization=float(getattr(cfg, "vlm_gpu_memory_utilization", 0.0) or 0.0) or None)
        except Exception as e:
            print(f"[ocr] olmOCR vLLM stop skipped: {e}")
            return 0.0

    def _stop_chandra_vllm(self) -> float:
        """Stop the managed Chandra vLLM instance (chandra_serve = "vllm"), for the same
        reason Surya's and olmOCR's need stopping: its VRAM belongs to the subprocess,
        not to a pool, so tearing the pool down does not reclaim it."""
        cfg = self._cfg          # unconditional, for the reason in _stop_surya_vllm
        if cfg is None:
            return 0.0
        try:
            from codai.api import vllm_worker
            from codai.models.manager import get_active_vllm_config
            from codai.ocr.chandra import DEFAULT_MODEL
            vcfg = get_active_vllm_config()
            if vcfg is None:
                return 0.0
            model = (getattr(cfg, "chandra_model", "") or DEFAULT_MODEL).strip()
            return vllm_worker.stop_service_for(
                vcfg, model_path=model, served_name=model,
                gpu_memory_utilization=float(getattr(cfg, "vlm_gpu_memory_utilization", 0.0) or 0.0) or None)
        except Exception as e:
            print(f"[ocr] Chandra vLLM stop skipped: {e}")
            return 0.0

    @staticmethod
    def _pool_params(cfg) -> tuple:
        return (
            cfg.max_concurrency,
            # cfg.lang and paddle_auto_lang decide which model a paddle pool is
            # built with, so a change to either has to invalidate the pools.
            getattr(cfg, "lang", ""), getattr(cfg, "paddle_auto_lang", ""),
            cfg.paddle_enabled, cfg.paddle_instances, cfg.paddle_use_gpu,
            cfg.paddle_structure, cfg.paddle_lang,
            cfg.paddle_det_model_dir, cfg.paddle_rec_model_dir,
            cfg.paddle_venv, cfg.paddle_auto_build,
            cfg.doctr_enabled, cfg.doctr_instances, cfg.doctr_use_gpu,
            cfg.doctr_det_arch, cfg.doctr_reco_arch,
            cfg.surya_enabled, cfg.surya_accept_license, cfg.surya_instances,
            cfg.surya_langs, cfg.surya_venv, cfg.surya_auto_build,
            # The token trace patches surya's backend at load, so toggling it has
            # to rebuild the pool rather than quietly taking effect never.
            getattr(cfg, "surya_token_trace", False),
            getattr(cfg, "surya_repair_accents", True),
            getattr(cfg, "olmocr_enabled", False), getattr(cfg, "olmocr_instances", 1),
            getattr(cfg, "olmocr_serve", "model"), getattr(cfg, "olmocr_model", ""),
            getattr(cfg, "olmocr_model_id", ""), getattr(cfg, "olmocr_server_url", ""),
            getattr(cfg, "olmocr_longest_side", 1288),
            getattr(cfg, "chandra_enabled", False),
            getattr(cfg, "chandra_accept_license", False),
            getattr(cfg, "chandra_instances", 1),
            getattr(cfg, "chandra_serve", "model"), getattr(cfg, "chandra_model", ""),
            getattr(cfg, "chandra_model_id", ""), getattr(cfg, "chandra_server_url", ""),
            getattr(cfg, "chandra_layout", True),
            getattr(cfg, "chandra_logprobs", True),
            getattr(cfg, "olmocr_logprobs", True),
            getattr(cfg, "vlm_gpu_memory_utilization", 0.0),
        )

    def engine_catalogue(self) -> dict:
        """What a client may ask for, and what it would get.

        Until now a caller had to KNOW which engines an install had enabled; a
        wrong guess was a 400 and there was nothing to ask. Everything here is
        read from this install's own config, so it answers the two questions a
        client actually has: which names work, and which of them returns the
        boxes or the layout I need.
        """
        cfg = self._require_cfg()
        # What each engine can put in a page, so a client can choose on capability
        # rather than on folklore. olmOCR is a document VLM: it transcribes a whole
        # page in reading order and has no notion of where a word was.
        shapes = {
            "paddle":  {"boxes": True,  "regions": True,  "tables": True,  "meta": False},
            "doctr":   {"boxes": True,  "regions": True,  "tables": False, "meta": False},
            "surya":   {"boxes": True,  "regions": True,  "tables": True,  "meta": False},
            "olmocr":  {"boxes": False, "regions": False, "tables": True,  "meta": True},
            # Chandra is a VLM like olmOCR but it is ASKED for boxes: its ocr_layout
            # prompt returns one div per block with data-bbox and data-label, so it is
            # the only engine here that gives boxes, labelled regions, tables AND page
            # metadata in a single pass. (With ocr.chandra_layout off it has none of
            # the three, which is why the row is read from the setting.)
            "chandra": {"boxes": bool(getattr(cfg, "chandra_layout", True)),
                        "regions": bool(getattr(cfg, "chandra_layout", True)),
                        "tables": bool(getattr(cfg, "chandra_layout", True)),
                        "meta": True},
        }
        # A pipeline drives a model somebody else serves; an engine owns its own
        # weights and runs in process. The distinction decides whether "where does
        # it run" is even a question, so it is worth saying out loud.
        kinds = {"paddle": "engine", "doctr": "engine",
                 "surya": "pipeline", "olmocr": "pipeline", "chandra": "pipeline"}
        models = {"surya": ("surya_model_id", "surya_model"),
                  "olmocr": ("olmocr_model_id", "olmocr_model"),
                  "chandra": ("chandra_model_id", "chandra_model")}

        data = []
        for name in sorted(_ENGINE_FACTORIES):
            enabled = _engine_enabled(cfg, name)
            row = {
                "id": name,
                "kind": kinds.get(name, "engine"),
                "enabled": bool(enabled),
                "instances": _engine_instances(cfg, name),
                "returns": shapes.get(name, {}),
                # Whether `lang=` steers this engine. The multilingual ones accept
                # it and recognise what they find regardless; saying so lets a
                # client stop guessing which of the two it got.
                "language_aware": engine_uses_language(name),
                "lang": resolve_lang(cfg, name, None),
                # Whether `conf` on this engine is a measurement or a placeholder,
                # and over what. The detector engines score each line; the VLMs
                # average exp(logprob) over the tokens of one generation, which for
                # chandra and olmOCR covers a whole page and for surya a block. A
                # client that gates on confidence has to know which it is getting.
                "confidence": _CONFIDENCE[name],
            }
            if not enabled:
                if name == "surya" and cfg.surya_enabled and not cfg.surya_accept_license:
                    row["reason"] = ("Surya's licence has not been accepted "
                                     "(ocr.surya_accept_license)")
                elif (name == "chandra" and getattr(cfg, "chandra_enabled", False)
                        and not getattr(cfg, "chandra_accept_license", False)):
                    row["reason"] = ("Chandra's licence has not been accepted "
                                     "(ocr.chandra_accept_license) — its weights are a "
                                     "modified OpenRAIL-M, free for research, personal "
                                     "use and companies under $2M, and not for use "
                                     "competing with Datalab's API")
                else:
                    row["reason"] = f"not enabled (ocr.{name}_enabled)"
            # The ids that ALSO resolve to this engine, so a client can name the
            # model it knows about instead of the pipeline that drives it.
            aliases = []
            for field in models.get(name, ()):
                got = str(getattr(cfg, field, "") or "").strip()
                if got and got not in aliases:
                    aliases.append(got)
            if aliases:
                row["aliases"] = aliases
                row["model"] = aliases[0]
                row["serve"] = str(getattr(cfg, f"{name}_serve", "") or "").strip()
            data.append(row)

        default = str(getattr(cfg, "default_engine", "") or "").strip()
        try:
            resolved = self.resolve_engine(None)
        except OcrError:
            resolved = ""
        return {"object": "list", "default": default, "default_engine": resolved,
                "data": data}

    def _engine_serving_model(self, wanted: str) -> str:
        """The engine configured to read pages with this MODEL, or "".

        A client asks for OCR with a model it knows about — the id it sees in
        /v1/models — not with the name of the pipeline we happen to drive it with.
        Both are accepted: "surya" names the engine, and the Surya-2 checkpoint
        names the same thing from the caller's side. Where that model then runs
        (here, a pinned cluster node, a rented pod) is ours to decide and nothing
        the caller should have to know.
        """
        cfg = self._require_cfg()
        for engine, fields in (("surya", ("surya_model_id", "surya_model")),
                               ("olmocr", ("olmocr_model_id", "olmocr_model")),
                               ("chandra", ("chandra_model_id", "chandra_model"))):
            for field in fields:
                got = str(getattr(cfg, field, "") or "").strip().lower()
                if got and got == wanted:
                    return engine
        return ""

    def resolve_engine(self, requested: Optional[str]) -> str:
        cfg = self._require_cfg()
        name = (requested or cfg.default_engine or "paddle").strip().lower()
        if name not in _ENGINE_FACTORIES:
            by_model = self._engine_serving_model(name)
            if by_model:
                name = by_model
        if name not in _ENGINE_FACTORIES:
            raise OcrError(
                f"unknown OCR engine or model '{name}' (engines: "
                f"{', '.join(sorted(_ENGINE_FACTORIES))})", status=400)
        if not _engine_enabled(cfg, name):
            raise OcrError(
                f"OCR engine '{name}' is not enabled (or license not accepted). "
                f"Enable it in Settings → OCR.",
                status=400,
            )
        return name

    def engine_dpi(self, requested: Optional[str]) -> int:
        """PDF rasterisation DPI for the engine that will serve this request.

        The DPI has to be chosen BEFORE rasterising, and the engine decides it:
        surya-2 accepts a far larger image than the shared default was feeding it
        (see ``OcrConfig.surya_dpi``), while paddle and docTR have their own optima
        and are left alone. An explicit per-request ``dpi`` wins over both.

        Resolving the engine here also means an unknown or disabled engine is
        refused before a large PDF is rasterised rather than after it. If the engine
        later fails over to another one (``_with_fallback``), the pages keep the DPI
        chosen for the engine that was asked for — re-rasterising mid-request would
        cost more than the slightly-off resolution does.
        """
        cfg = self._require_cfg()
        try:
            name = self.resolve_engine(requested)
        except OcrError:
            # Let the real failure come from the engine path, with its own message.
            return int(cfg.dpi)
        per_engine = int(getattr(cfg, f"{name}_dpi", 0) or 0)
        return per_engine if per_engine > 0 else int(cfg.dpi)

    async def _get_pool(self, name: str, lang: str = "") -> _Pool:
        """The pool serving ``name``, in ``lang`` for the engines that read by it.

        A language-sensitive engine gets one pool PER language, because its
        instances are built for a language at load time and cannot be re-languaged
        (see ``_LANG_SENSITIVE``). Every other engine keeps a single pool whatever
        the caller asks for — keying those by language would duplicate loaded
        instances, and VRAM, for a parameter they cannot act on.
        """
        cfg = self._require_cfg()
        if engine_uses_language(name):
            # Key on what the engine is BUILT with, not what was asked: `auto` and
            # the language it maps to are the same loaded model, and must not be
            # two pools holding two copies of it.
            lang = engine_lang(cfg, name, resolve_lang(cfg, name, lang))
            key = f"{name}:{lang}"
        else:
            lang, key = "", name
        async with self._lock:
            pool = self._pools.get(key)
            if pool is None:
                pool = _Pool(name, self._cfg_for(cfg, name, lang),
                             _engine_instances(cfg, name))
                self._pools[key] = pool
            return pool

    @staticmethod
    def _cfg_for(cfg, name: str, lang: str):
        """``cfg`` with this request's language already resolved into it.

        The pool builds engines with ``factory(self.cfg)`` and each engine reads its
        own language out of the config, so resolving it here means no engine needs
        to know a per-request language exists. Both the global and the
        engine-specific field are set, so the request beats ``<engine>_lang`` too.
        """
        if not lang:
            return cfg
        try:
            import dataclasses
            fields = {f.name for f in dataclasses.fields(cfg)}
            changes = {"lang": lang}
            if f"{name}_lang" in fields:
                changes[f"{name}_lang"] = lang
            return dataclasses.replace(cfg, **changes)
        except Exception:
            # Not a dataclass, or a field that will not round-trip: the shared
            # config still serves the document, in the configured language.
            return cfg

    def _fallback_order(self, failed: str) -> List[str]:
        """Enabled engines to try after ``failed`` could not serve the document.

        An OCR request that comes back 503 is a document nobody read — and the whole
        point of having four engines is that a page can still be read when one of them
        will not come up (a VLM engine whose vLLM cannot fit on the card, a venv that
        needs rebuilding, a license flag). ``auto`` prefers the engines with the fewest
        ways to fail over the ones with the best output, because the alternative to a
        worse transcription here is no transcription."""
        cfg = self._require_cfg()
        raw = str(getattr(cfg, "fallback_engines", "auto") or "auto").strip().lower()
        if raw in ("off", "none", "no", "false"):
            return []
        if raw in ("auto", "", "on", "true"):
            names = ["paddle", "doctr", "olmocr", "chandra", "surya"]
        else:
            names = [n.strip() for n in raw.replace(",", " ").split() if n.strip()]
        return [n for n in names
                if n != failed and n in _ENGINE_FACTORIES and _engine_enabled(cfg, n)]

    async def _pages_with(self, name: str, images: List, lang: str = "",
                          base: int = 0) -> List[OcrPage]:
        pool = await self._get_pool(name, lang)
        sem = self._sem or asyncio.Semaphore(4)

        async def _one(idx, img):
            async with sem:
                page = await pool.recognize(img)
                # `base` is the document-wide index of this window's first page, so
                # a windowed read numbers pages the same way a whole-document read
                # always did -- callers index into `pages` by page number.
                page.index = base + idx
                return page

        pages = await asyncio.gather(*[_one(i, im) for i, im in enumerate(images)])
        return list(pages)

    async def ocr_pages(self, images: List, engine: Optional[str] = None,
                        lang: Optional[str] = None,
                        base: int = 0) -> (str, List[OcrPage]):
        """OCR a list of PIL page images. Returns (engine_name, [OcrPage]).

        The returned name is the engine that ACTUALLY read the pages, which is not
        necessarily the one asked for: when an engine cannot serve, the remaining
        enabled engines are tried in turn (see :meth:`_fallback_order`) rather than
        failing the document. Callers surface the name, so a caller that cares can see
        it fell back."""
        name = self.resolve_engine(engine)
        try:
            return name, await self._pages_with(name, images, lang, base)
        except Exception as first:
            chain = self._fallback_order(name)
            if not chain:
                raise
            print(f"[ocr] engine '{name}' could not serve ({first}); "
                  f"falling back to {' -> '.join(chain)}")
            for alt in chain:
                try:
                    pages = await self._pages_with(alt, images, lang, base)
                except Exception as e:
                    print(f"[ocr] fallback engine '{alt}' also failed: {e}")
                    continue
                print(f"[ocr] served by fallback engine '{alt}' instead of '{name}'")
                return alt, pages
            # Everything enabled is down: the original failure is the useful one.
            raise first

    async def ocr_document(self, data: bytes, filename: str = "", content_type: str = "",
                           engine: Optional[str] = None, dpi: Optional[int] = None,
                           detect: Optional[str] = None, structured: bool = False,
                           schema: Optional[str] = None,
                           lang: Optional[str] = None,
                           page_opts: Optional[dict] = None) -> dict:
        """OCR raw bytes (image/PDF) end-to-end → response dict.

        ``detect`` overrides the configured stamp/signature detection mode
        (off|layout|detector|both). ``structured`` triggers field extraction via the
        configured text model; ``schema`` overrides the extraction schema for this call.

        ``lang`` is the document's language; omitted or ``auto`` means let the engine
        recognise whatever it finds, which is what the multilingual engines do
        natively. The response reports the language that was applied and whether the
        engine that served the page can actually read by language, so a caller can
        tell a language that took effect from one that was merely accepted.
        """
        cfg = self._require_cfg()
        use_dpi = int(dpi) if dpi else self.engine_dpi(engine)

        # Price the document BEFORE rendering it: once the pages exist the memory is
        # already spent, so admission control has to work from the page tree.
        n_pages, est_bytes = page_estimate(data, filename=filename,
                                           content_type=content_type, dpi=use_dpi)
        window = self._page_window(cfg)
        # Charge for the WINDOW, not the document -- windowing is what makes that
        # honest, and charging whole documents would queue work that never coexists.
        # Plus two pages of headroom: `page_estimate` reports what a page retains,
        # and rendering the next one briefly holds pdfium's bitmap and its RGB copy
        # on top of the window already in hand.
        per_page = max(1, est_bytes // max(1, n_pages))
        reserve = per_page * (max(1, min(n_pages, window)) + 2)
        budget = get_budget(int(getattr(cfg, "max_resident_mb", 0) or 0))

        # Share the GPU thermal governor with every other workload: block here until
        # temps are safe (wait_until_safe blocks, so run it off the event loop).
        await asyncio.to_thread(_wait_thermal_safe)

        # THE POD-SIDE PIPELINE. When the model's pods run it themselves (the
        # image is the switch -- see pod_pipeline_model), the page goes to them
        # as a one-page PDF and this box never rasterises: no window to hold, no
        # resident budget to charge, no bitmap crossing the network. Falls
        # through to the local path whenever a pod cannot serve, because an
        # optimisation that cannot run must cost latency and not an error.
        _remote_model = pod_pipeline_model(cfg, engine)
        _out = None
        if _remote_model:
            try:
                _out = await self._read_on_pods(
                    data, filename, content_type, _remote_model,
                    dpi=use_dpi, page_opts=page_opts, detect=detect, cfg=cfg,
                    engine_name=(engine or "").strip().lower()
                    or str(getattr(cfg, "default_engine", "") or "").lower())
            except OcrError:
                raise
            except Exception as exc:
                print(f"[ocr] the pod-side pipeline for {_remote_model!r} could "
                      f"not run ({exc}); reading locally instead", flush=True)
                _out = None
        if _out is not None:
            name, pages, blank, stamps, signatures = _out
        else:
            async with budget.reserve(reserve):
                name, pages, blank, stamps, signatures = await self._read_windowed(
                    data, filename, content_type, use_dpi, window,
                    engine=engine, lang=lang, detect=detect,
                    page_opts=page_opts, cfg=cfg)
        # A page can be ruined without anything failing -- see `_retry_degraded`.
        # Done here rather than inside either pipeline, because the two produce
        # the same degraded pages for the same reasons and a reader of the
        # result cannot tell which path it came from.
        recovered = await self._retry_degraded(
            data, filename, content_type, name, pages, dpi=use_dpi,
            page_opts=page_opts, detect=detect, lang=lang, cfg=cfg,
            window=window)
        full_text = "\n\n".join(p.text for p in pages)

        structured_out = None
        if structured:
            from codai.ocr.extract import extract_fields
            structured_out = await extract_fields(full_text, cfg, schema=schema)

        applied = resolve_lang(cfg, name, lang)
        out = {
            "engine": name,
            "lang": applied,
            # Whether the engine that actually served can read by language at all.
            # surya-2, olmOCR and docTR are multilingual with no language input, so
            # a `lang` they were given was accepted and had no effect; saying so is
            # better than letting a caller believe it steered anything.
            "lang_applied": engine_uses_language(name),
            "num_pages": len(pages),
            "text": full_text,
            "pages": [p.to_dict() for p in pages],
            "stamps": stamps,
            "signatures": signatures,
            "structured": structured_out,
        }
        # Pages with (almost) nothing on them. Reported at the top level as well
        # as per page, because a caller deciding whether to accept a document
        # should not have to walk the pages to discover that it is empty -- and
        # because "the OCR failed" and "the document is blank" lead to entirely
        # different places.
        if blank:
            out["blank_pages"] = blank
        report = unreadable_report(pages)
        if report:
            out["unreadable"] = report
        # Pages this install itself judged readable and then produced no text
        # for. A caller cannot tell that from a blank page by looking at the
        # text, and the two lead to opposite places: one is a document to skip,
        # the other is a read to repeat. We already computed the judgement --
        # `page_ink` says there is ink on the page -- it simply never reached
        # the response, so ~1,000 documents were consumed from a work queue and
        # written off as unreadable scans after a lost engine lease.
        unread = _unread_pages(pages)
        if unread:
            out["unread_pages"] = unread
        if recovered:
            # Top level, because a caller deciding whether to trust this
            # document should not have to walk the pages to find out that part
            # of it was read by a different engine than the one it asked for.
            out["recovered_pages"] = recovered
        return out

    def _raster_workers(self, cfg) -> int:
        """How many processes may rasterise at once. 0 = auto, 1 = in-process."""
        try:
            return max(0, int(getattr(cfg, "raster_workers", 0) or 0))
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Degraded pages
    # ------------------------------------------------------------------
    @staticmethod
    def _page_degraded(page) -> str:
        """Why this page is not usable, or "" when it is.

        A degraded page is NOT an error. The engine answered 200 with
        plausible-looking text, and the only evidence is in the diagnostics it
        attached. Two causes, both of which make the page incomplete:

        * ``repeat_detected`` -- the decode fell into a loop and ran to the cap
          emitting the same fragment. Retrying the SAME engine at a higher
          temperature is what produced this flag; it does not clear it.
        * ``truncated`` -- it stopped at the cap and could not be continued.

        Deliberately NOT degraded: a short page, a blank page, or a page the
        engine reported it could not read. Those have their own handling and
        their own meaning, and re-reading them on a second engine would spend a
        card to learn what the first engine already said correctly.
        """
        meta = getattr(page, "meta", None) or {}
        if meta.get("repeat_detected"):
            return "looping"
        # Nothing came back from the model at all. Not the same as a page the
        # model read and found empty: this page was never read, so another
        # engine is exactly the right thing to try. A lost engine lease answers
        # like this, and so does a pod taken away mid-batch.
        if meta.get("engine_empty"):
            return "unread"
        ink = meta.get("page_ink")
        if (isinstance(ink, dict) and ink.get("blank") is False
                and not (getattr(page, "text", "") or "").strip()):
            # The orchestrator already measured ink on this page and said there
            # IS something on it. Answering 200 with no text for a page we
            # ourselves judged readable is the silent-wrong-answer case.
            return "unread"
        info = meta.get("model_attempts") or {}
        if isinstance(info, dict) and info.get("truncated"):
            return "truncated"
        return ""

    @staticmethod
    def _short_pages(pages: List, cfg) -> dict:
        """Pages far shorter than the document around them → {index: reason}.

        Reported from the same run: olmOCR returned 4,046 characters where
        chandra read 30,983 on the SAME page, and its fastest calls returned
        the least text. Nothing was flagged -- the reply ended cleanly, well
        short of the cap, so there is no `truncated` and no loop. The only
        evidence an early stop leaves is the company it keeps.

        So the measure is the document's OWN median, not a character count:
        a threshold in characters would have to be re-tuned for every language,
        DPI, layout and font, and would be wrong for all of them. A page at an
        eighth of its neighbours is an early stop whatever the corpus.

        OFF by default (`retry_short_pages_below = 0`) and deliberately so.
        This signal is INFERRED, where looping and truncation are attested by
        the engine itself -- and a genuine title page, index or signature page
        looks exactly the same from here. Only the operator knows whether their
        documents have those.
        """
        try:
            frac = float(getattr(cfg, "retry_short_pages_below", 0.0) or 0.0)
            min_doc = int(getattr(cfg, "retry_short_pages_min_doc", 4) or 4)
        except Exception:
            return {}
        if frac <= 0 or len(pages) < max(3, min_doc):
            return {}
        lens = sorted(len((getattr(p, "text", "") or "").strip()) for p in pages)
        mid = len(lens) // 2
        median = lens[mid] if len(lens) % 2 else (lens[mid - 1] + lens[mid]) / 2
        if median <= 0:
            return {}
        floor = median * frac
        out = {}
        for i, page in enumerate(pages):
            n = len((getattr(page, "text", "") or "").strip())
            # A page the blank check already explained is not an early stop:
            # there was nothing on it. Re-reading it buys the same answer from
            # a second card.
            if (getattr(page, "meta", None) or {}).get("blank"):
                continue
            if n < floor:
                out[i] = (f"short ({n} chars against a document median of "
                          f"{int(median)})")
        return out

    def _retry_degraded_order(self, failed: str) -> List[str]:
        """Engines to re-read a degraded page with, in order."""
        cfg = self._require_cfg()
        raw = str(getattr(cfg, "retry_degraded", "auto") or "auto").strip().lower()
        if raw in ("off", "none", "no", "false", "0"):
            return []
        if raw in ("auto", "", "on", "true", "1"):
            return self._fallback_order(failed)
        names = [n.strip() for n in raw.replace(",", " ").split() if n.strip()]
        return [n for n in names
                if n != failed and n in _ENGINE_FACTORIES and _engine_enabled(cfg, n)]

    async def _read_page_blobs(self, blobs, engine: str, *, filename: str,
                               content_type: str, dpi: int, page_opts, detect,
                               lang, cfg, window: int):
        """Read a handful of one-page documents with ONE named engine.

        Takes whichever of the two pipelines that engine's configuration
        selects, exactly as a whole document would -- a retry that quietly used
        the other path would be measuring the path, not the engine.
        """
        out = []
        remote = pod_pipeline_model(cfg, engine)
        for blob in blobs:
            got = None
            if remote:
                try:
                    got = await self._read_on_pods(
                        blob, filename, content_type, remote, dpi=dpi,
                        page_opts=page_opts, detect=detect, cfg=cfg,
                        engine_name=engine)
                except Exception as exc:
                    print(f"[ocr] re-reading on '{engine}' pods failed: {exc}",
                          flush=True)
                    got = None
            if got is not None:
                pages = got[1]
            else:
                _n, pages, _b, _s, _g = await self._read_windowed(
                    blob, filename, content_type, dpi, window, engine=engine,
                    lang=lang, detect=detect, page_opts=page_opts, cfg=cfg)
            out.append(pages[0] if pages else None)
        return out

    async def _retry_degraded(self, data: bytes, filename: str, content_type: str,
                              name: str, pages: List, *, dpi: int, page_opts,
                              detect, lang, cfg, window: int) -> str:
        """Re-read every degraded page on another engine. Returns a note, or "".

        Mutates `pages` in place, replacing only those pages that come back
        clean. This exists because an OCR run can fail almost completely with
        no error anywhere: a repeating decode answers 200, `fallback_engines`
        never fires, and a thousand documents land in the index truncated.

        Bounded by construction -- only pages already known to be unusable are
        re-read, so a clean document costs nothing and a ruined one costs at
        most a second pass over its ruined pages.
        """
        why = {}
        for i, page in enumerate(pages):
            reason = self._page_degraded(page)
            if reason:
                why[i] = reason
        for i, reason in self._short_pages(pages, cfg).items():
            why.setdefault(i, reason)
        bad = sorted(why)
        if not bad:
            return ""
        chain = self._retry_degraded_order(name)
        if not chain:
            return ""
        try:
            cap = max(0, int(getattr(cfg, "retry_degraded_max_pages", 0) or 0))
        except Exception:
            cap = 0
        dropped = 0
        if cap and len(bad) > cap:
            dropped = len(bad) - cap
            bad = bad[:cap]
        is_pdf = ((content_type or "").lower() == "application/pdf"
                  or (filename or "").lower().endswith(".pdf")
                  or data[:5] == b"%PDF-")
        if not is_pdf:
            if len(pages) != 1:
                return ""
            blobs = {0: data}
        else:
            blobs = {}
            for i in bad:
                try:
                    blobs[i] = split_pdf_page(data, i)
                except Exception as exc:
                    print(f"[ocr] cannot isolate page {i} to re-read it: {exc}",
                          flush=True)
        bad = [i for i in bad if i in blobs]
        if not bad:
            return ""
        print(f"[ocr] {len(bad)} of {len(pages)} page(s) came back DEGRADED from "
              f"'{name}' ("
              + ", ".join(f"p{i}:{why[i]}" for i in bad[:8])
              + (" ..." if len(bad) > 8 else "")
              + f") — the engine answered 200, so this is not a fallback: "
              f"re-reading those pages on {' -> '.join(chain)}", flush=True)
        if dropped:
            print(f"[ocr] and leaving {dropped} more as they are "
                  f"(ocr.retry_degraded_max_pages)", flush=True)
        fixed: dict = {}
        for alt in chain:
            todo = [i for i in bad if i not in fixed]
            if not todo:
                break
            try:
                got = await self._read_page_blobs(
                    [blobs[i] for i in todo], alt, filename=filename,
                    content_type=content_type, dpi=dpi, page_opts=page_opts,
                    detect=detect, lang=lang, cfg=cfg, window=window)
            except Exception as exc:
                print(f"[ocr] re-read on '{alt}' failed outright: {exc}", flush=True)
                continue
            for i, page in zip(todo, got):
                if page is None or self._page_degraded(page):
                    continue
                if why[i].startswith("short"):
                    # The complaint was length, so the answer has to be longer.
                    # A second engine that stops just as early has told us the
                    # page really is short, and taking its text would hide that
                    # behind a different engine's name.
                    if len((page.text or "").strip()) <= \
                            len((pages[i].text or "").strip()):
                        continue
                # Keep the ORIGINAL page index: the caller indexes by it, and a
                # page re-read on its own always comes back as page 0.
                page.index = pages[i].index
                meta = dict(getattr(page, "meta", None) or {})
                meta["degraded_retry"] = {
                    "from": name, "to": alt, "reason": why[i],
                    "why": (f"'{name}' returned this page {why[i]}, which is a "
                            f"200 and not a failure, so it was re-read on "
                            f"'{alt}' and that answer is the one here"),
                }
                page.meta = meta
                pages[i] = page
                fixed[i] = alt
        if not fixed:
            print(f"[ocr] no engine in {' -> '.join(chain)} read those pages any "
                  f"better — keeping what '{name}' produced, still flagged",
                  flush=True)
            return ""
        by_engine: dict = {}
        for alt in fixed.values():
            by_engine[alt] = by_engine.get(alt, 0) + 1
        note = ", ".join(f"{n} page(s) by {e}" for e, n in sorted(by_engine.items()))
        print(f"[ocr] recovered {len(fixed)} of {len(bad)} degraded page(s): "
              f"{note}", flush=True)
        return note

    async def _read_on_pods(self, data: bytes, filename: str, content_type: str,
                            model_id: str, *, dpi: int, page_opts, detect, cfg,
                            engine_name: str = "surya"):
        """Send ONE PAGE PER REQUEST to pods that run the pipeline themselves.

        The page travels as a small PDF and the pod renders and reads it on the
        card, with its own vLLM over loopback instead of four internet round
        trips for a page resumed four times. This box never rasterises.

        Page-level fan-out is unchanged: each page is an independent request and
        `placement.choose` picks an upstream for it, so every pod in the pool
        shares the work and none of them owns a document.

        Returns (engine_name, pages, blank, stamps, signatures) like
        `_read_windowed`, or None when no pod could serve -- in which case the
        caller does what it always did. A pod-side pipeline is an optimisation,
        and an optimisation that cannot run must cost latency, never an error.
        """
        from codai.models import placement

        is_pdf = ((content_type or "").lower() == "application/pdf"
                  or (filename or "").lower().endswith(".pdf")
                  or data[:5] == b"%PDF-")
        # A single image has nothing to split; it goes as it arrived.
        # Set aside recently? Then this document does not pay a round trip per
        # page to be told the same thing again. Said once, when it was set
        # aside; repeating it per document would bury it.
        standing = _pod_pipeline_refusal(model_id)
        if standing:
            return None
        parts = (list(iter_pdf_page_pdfs(data)) if is_pdf else [data])
        if not parts:
            return None
        sem = self._sem or asyncio.Semaphore(4)
        opts = dict(page_opts or {})
        refused: list = []

        async def _one(idx: int, blob: bytes):
            async with sem:
                if refused:
                    # Another page already found the pods cannot run it. Do not
                    # send the rest of the document one failure at a time.
                    return None
                try:
                    lease = placement.choose(
                        model_id, spill_after=placement.spill_threshold(model_id))
                except Exception as exc:
                    raise OcrError(f"no upstream for {model_id}: {exc}", status=503)
                try:
                    if lease.kind == "local":
                        # Nothing rented: the local pipeline is the right answer
                        # and the caller already has it.
                        return None
                    return await asyncio.to_thread(
                        _post_page_to_pod, lease.base, lease.api_key, blob,
                        f"{idx}.pdf" if is_pdf else (filename or "page"),
                        dpi, opts, detect, engine_name)
                except OcrError as exc:
                    # A POD THAT CANNOT RUN THE PIPELINE IS NOT A FAILED PAGE.
                    # It answers the same way for every page of every document,
                    # so raising turned "this optimisation is unavailable" into
                    # "this document cannot be read" -- while the local pipeline
                    # sat there able to read it. The comment on this method
                    # already said an optimisation that cannot run must cost
                    # latency and never an error; this is that, implemented.
                    if int(getattr(exc, "status", 0) or 0) not in _POD_CANNOT_RUN_PIPELINE:
                        raise
                    refused.append(str(exc))
                    return None
                finally:
                    lease.release()

        got = await asyncio.gather(*[_one(i, b) for i, b in enumerate(parts)],
                                   return_exceptions=True)
        if any(r is None for r in got):
            # At least one page had no pod. Do the WHOLE document locally rather
            # than stitch two pipelines together: a document read half one way
            # and half the other is the hardest kind of result to explain.
            #
            # SAY SO. This used to return None in silence, which meant a
            # deployment could be configured for the pod-side pipeline, fall
            # back to the local one on every request, and look exactly like a
            # success -- the operator would conclude the new path worked when it
            # had never run. One line is the difference between a test that
            # proves something and a test that proves nothing.
            _n = sum(1 for r in got if r is None)
            if refused:
                # The pods answered, and what they said was "not me". That is a
                # different fault from "nothing is rented", needs a different
                # fix, and must carry the pod's OWN words: this is the only
                # place they are visible.
                if _set_aside_pod_pipeline(model_id, refused[0]):
                    print(f"[ocr] the pods for {model_id!r} run an image that "
                          f"carries the OCR pipeline, but they REFUSE to run "
                          f"it: {refused[0]} — falling back to the pipeline on "
                          f"this box and not asking them again for "
                          f"{POD_PIPELINE_COOLDOWN_S:.0f}s. This box will "
                          f"rasterise in the meantime, so watch its memory.",
                          flush=True)
            else:
                print(f"[ocr] the pod-side pipeline was configured for "
                      f"{model_id!r} but {_n} of {len(parts)} page(s) had no "
                      f"pod to run it (placement answered 'local' — nothing "
                      f"rented, or the pool could not serve) — reading the "
                      f"WHOLE document here instead", flush=True)
            return None
        bad = [r for r in got if isinstance(r, BaseException)]
        if bad:
            raise bad[0]

        print(f"[ocr] read {len(parts)} page(s) ON THE PODS for {model_id!r}: "
              f"each page went as a one-page PDF and was rasterised and read "
              f"there, not here", flush=True)
        pages, blank, stamps, sigs = [], [], [], []
        served = ""
        for idx, res in enumerate(got):
            served = served or str(res.get("engine") or "")
            for p in (res.get("pages") or []):
                page = _page_from_pod(p, idx)
                pages.append(page)
            for i in (res.get("blank_pages") or []):
                blank.append(idx)
            for d in (res.get("stamps") or []):
                d = dict(d)
                d["page"] = idx
                stamps.append(d)
            for d in (res.get("signatures") or []):
                d = dict(d)
                d["page"] = idx
                sigs.append(d)
        return (served or "surya", pages, sorted(set(blank)), stamps, sigs)

    def _page_window(self, cfg) -> int:
        """How many pages of one document to hold rasterised at a time.

        0 means follow `max_concurrency`, capped at 32: the window should be at
        least as large as the work the engines will take concurrently (a smaller
        one would idle them between windows) without being large enough to bring
        back the unbounded residency it exists to prevent.
        """
        try:
            w = int(getattr(cfg, "page_window", 0) or 0)
        except Exception:
            w = 0
        if w > 0:
            return w
        try:
            mc = int(getattr(cfg, "max_concurrency", 64) or 64)
        except Exception:
            mc = 64
        return max(1, min(32, mc))

    async def _read_windowed(self, data: bytes, filename: str, content_type: str,
                             dpi: int, window: int, *, engine, lang, detect,
                             page_opts, cfg):
        """Read a document a window of pages at a time.

        Residency is O(window) rather than O(document): each window's bitmaps are
        rendered, read, measured, detected on, and dropped before the next window
        is rendered. A 400-page filing now costs what an 8-page one does.

        The engine is STICKY across windows. `ocr_pages` falls back per call, so
        without stickiness a document could be read by a different engine in every
        window and report only the last one. Pages already read keep their results
        when a later window has to fall back -- the whole-document path re-read
        them from page one.
        """
        # `chunk` is the page window: a pooled read borrows a worker per chunk,
        # so matching the two means the document is re-opened once per window
        # rather than once per few pages.
        stream = render_pages(data, filename=filename, content_type=content_type,
                              dpi=dpi, raster_workers=self._raster_workers(cfg),
                              chunk=window)
        pages_all: List[OcrPage] = []
        blank_all: list = []
        stamps_all: list = []
        sigs_all: list = []
        served = None
        base = 0
        token = PAGE_OPTIONS.set(dict(page_opts)) if page_opts else None
        try:
            while True:
                # Rendering is CPU-bound C (pypdfium2) and used to run on the event
                # loop, where a long PDF stalled every other request in the process.
                imgs = await asyncio.to_thread(_drain, stream, window)
                if not imgs:
                    break
                name, wpages = await self.ocr_pages(
                    imgs, engine=served or engine, lang=lang, base=base)
                if served is None:
                    served = name
                elif name != served:
                    print(f"[ocr] page {base}: engine switched '{served}' -> "
                          f"'{name}' mid-document", flush=True)
                    served = name
                blank_all.extend(base + i
                                 for i in _mark_blank_pages(cfg, imgs, wpages))
                st, sg = await self._detect_all(imgs, wpages, detect, base=base)
                stamps_all.extend(st)
                sigs_all.extend(sg)
                pages_all.extend(wpages)
                base += len(imgs)
                # Drop this window's bitmaps BEFORE the next is rendered -- holding
                # both is the whole thing being fixed.
                imgs = None
        finally:
            if token is not None:
                PAGE_OPTIONS.reset(token)
            stream.close()
        return (served or self.resolve_engine(engine), pages_all, blank_all,
                stamps_all, sigs_all)

    async def _detect_all(self, images, pages, detect: Optional[str], base: int = 0):
        """Run stamp/signature detection across pages; returns (stamps, signatures)
        with each detection tagged by page index. Empty when mode is 'off'."""
        from codai.ocr.detect import resolve_detect_mode, detect_page

        cfg = self._require_cfg()
        mode = resolve_detect_mode(cfg, detect)
        if mode == "off":
            return [], []

        sem = self._sem or asyncio.Semaphore(4)

        async def _one(idx, img, page):
            async with sem:
                res = await asyncio.to_thread(detect_page, img, page, cfg, mode)
                for d in res["stamps"]:
                    d["page"] = base + idx
                for d in res["signatures"]:
                    d["page"] = base + idx
                return res

        results = await asyncio.gather(*[_one(i, im, pg) for i, (im, pg) in enumerate(zip(images, pages))])
        stamps, signatures = [], []
        for r in results:
            stamps.extend(r["stamps"])
            signatures.extend(r["signatures"])
        return stamps, signatures

    def _require_cfg(self):
        if self._cfg is None:
            raise OcrError("OCR subsystem is not configured", status=503)
        if not getattr(self._cfg, "enabled", False):
            raise OcrError("OCR subsystem is disabled (enable it in Settings → OCR)", status=400)
        return self._cfg


# Module-level singleton (mirrors multi_model_manager).
ocr_manager = OcrManager()
