# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Fully-managed vLLM worker — the vLLM OpenAI server, driven as a subprocess.

vLLM (https://github.com/vllm-project/vllm) exposes an OpenAI-compatible HTTP server
(``python -m vllm.entrypoints.openai.api_server``) with continuous batching + paged KV,
so — like :mod:`codai.api.kt_worker` — coderai launches it as a managed subprocess,
health-checks ``/v1/models``, and :mod:`codai.backends.vllm` proxies to it.

vLLM pins its own torch/CUDA (e.g. torch 2.13 / cu13), which conflicts with the main
coderai venv, so it runs in an ISOLATED venv (``vllm.venv``) — same pattern as the OCR
Paddle/Surya engines. This module owns the venv + process lifecycle only. Also reused by
the OCR subsystem to serve Surya2 (a VLM) for :mod:`codai.ocr.surya`.
"""

import collections
import json
import os
import re
import shlex
import socket
import subprocess
import threading
import time
from typing import Optional

_lock = threading.RLock()
_services: dict[str, dict] = {}   # svc_key -> {"proc","port","url"}

# Repo-root requirements for the isolated vLLM venv (auto-build).
_REQ = os.path.join(os.path.dirname(__file__), "..", "..", "requirements-vllm.txt")


def resolve_venv_dir(cfg) -> str:
    """Isolated vLLM venv: config > baked /opt/coderai/vllm_venv > /cache mount > ~/.coderai."""
    configured = (getattr(cfg, "venv", "") or "").strip() \
        or (os.environ.get("CODERAI_VLLM_VENV") or "").strip()
    if configured:
        return configured
    baked = "/opt/coderai/vllm_venv"
    if os.path.isdir(baked):
        return baked
    cache = os.environ.get("CODERAI_CACHE_DIR") or ("/cache" if os.path.isdir("/cache") else "")
    if cache and os.path.isdir(cache):
        return os.path.join(cache, "vllm_venv")
    return os.path.expanduser("~/.coderai/vllm_venv")


def _venv_python(cfg) -> str:
    return os.path.join(os.path.expanduser(resolve_venv_dir(cfg)), "bin", "python")


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _pump_logs(proc, tail, meta: dict = None):
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            tail.append(line)
            # vLLM states how much of the budget was left for the cache. That figure is
            # the only direct measurement of an engine's real footprint we ever get, so
            # it is captured HERE as it streams — by the time the server is ready it has
            # scrolled out of the 80-line tail behind vLLM's route listing.
            if meta is not None and "kv_gib" not in meta:
                m = _KV_REPORT.search(line)
                if m:
                    try:
                        meta["kv_gib"] = float(m.group(1))
                    except Exception:
                        pass
            if meta is not None and "kv_tokens" not in meta:
                m = _KV_TOKENS.search(line)
                if m:
                    try:
                        meta["kv_tokens"] = int(m.group(1).replace(",", ""))
                    except Exception:
                        pass
            print(f"[vllm] {line}", flush=True)


def running_service_urls() -> list:
    """Base URLs of the vLLM services this process has running.

    Used to read serving load (KV cache, queue depth, preemptions) from the
    engine that actually holds the model, rather than inferring it from outside.
    """
    try:
        return [svc["url"] for svc in list(_services.values())
                if isinstance(svc, dict) and svc.get("url")]
    except Exception:
        return []


def _health_ok(url: str, timeout: float = 3) -> bool:
    import requests
    try:
        r = requests.get(url + "/v1/models", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def ensure_built(cfg) -> str:
    """Ensure the isolated vLLM venv exists; return its python. Builds it if auto_build."""
    py = _venv_python(cfg)
    if os.path.isfile(py):
        return py
    if not getattr(cfg, "auto_build", False):
        raise RuntimeError(
            f"vLLM isolated venv not found at {os.path.dirname(os.path.dirname(py))}. "
            f"Build it (python3 -m venv <dir> && <dir>/bin/pip install -r "
            f"requirements-vllm.txt) or enable vllm.auto_build.")
    venv_dir = os.path.expanduser(resolve_venv_dir(cfg))
    req = os.path.abspath(_REQ)
    print(f"[vllm] creating isolated venv at {venv_dir} …", flush=True)
    import sys as _sys
    try:
        os.makedirs(os.path.dirname(venv_dir) or ".", exist_ok=True)
        subprocess.run([_sys.executable, "-m", "venv", venv_dir], check=True)
        subprocess.run([py, "-m", "pip", "install", "-U", "pip"], check=True)
        if os.path.isfile(req):
            subprocess.run([py, "-m", "pip", "install", "-r", req], check=True)
        else:
            subprocess.run([py, "-m", "pip", "install", "vllm"], check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"vLLM: venv build failed: {exc}")
    if not os.path.isfile(py):
        raise RuntimeError("vLLM: venv python missing after build")
    return py


def resolve_service_key(cfg, model_path: Optional[str] = None):
    mp = os.path.expanduser((model_path or getattr(cfg, "model_path", "") or "").strip())
    key = mp or (getattr(cfg, "model_id", "vllm") or "vllm")
    return mp, key


def service_key_for(cfg, model_path=None, served_name=None):
    """The key for the vLLM instance that serves ``served_name``.

    ONE engine per set of weights, not one per name. vLLM registers adapters at
    launch and serves each under its own name, so a base and its LoRAs are one
    instance — keying on the name as well meant a second request for the same
    weights under a different name started a SECOND copy of the model on the
    same card. On a 48 GB card holding a 27B that is not a slowdown, it is an
    out-of-memory crash, and the first engine is the one that dies.

    ``served_name`` is still what the instance must ANSWER to; see
    :func:`_ensure_service_once`, which relaunches an engine that does not.
    """
    _resolved, key = resolve_service_key(cfg, model_path)
    return key


#: Model family -> the vLLM parser that reads its tool-call syntax. A tool call
#: is TEXT the model emits in its own trained format, so the parser belongs to
#: the family, not to the deployment. Longest match wins, so `qwen3-coder`
#: beats `qwen`.
_TOOL_PARSERS = (
    ("qwen3-coder", "qwen3_coder"), ("qwen3_coder", "qwen3_coder"),
    ("deepseek-v3", "deepseek_v3"), ("deepseek_v3", "deepseek_v3"),
    ("llama-3", "llama3_json"), ("llama3", "llama3_json"),
    ("mistral", "mistral"), ("ministral", "mistral"),
    ("granite", "granite"), ("internlm", "internlm"), ("jamba", "jamba"),
    ("glm-4.5", "glm45"), ("glm-4.6", "glm45"), ("glm45", "glm45"),
    # Qwen 2.5/3 and the many Hermes-style finetunes emit <tool_call> JSON.
    ("qwen", "hermes"), ("hermes", "hermes"),
)


def _chat_template_text(model_path: str) -> str:
    """The model's chat template, or '' if it cannot be read.

    Thin wrapper over codai.models.chat_style, which both parser choices share:
    vLLM's here, and coderai's own in ModelParserDispatcher. Imported lazily so
    a pod that never launches vLLM does not pay for it.
    """
    try:
        from codai.models.chat_style import chat_template_text
    except Exception:
        return ""
    return chat_template_text(model_path)


def _emits_qwen_xml_calls(model_path: str) -> bool:
    """True when the model's template writes tool calls as Qwen XML.

    ``<tool_call><function=NAME><parameter=KEY>value</parameter></function>``
    — what vLLM's ``qwen3_xml`` parser reads, and what ``hermes`` (JSON inside
    ``<tool_call>``) cannot: it raises JSONDecodeError on every single call.
    """
    try:
        from codai.models.chat_style import tool_call_style
    except Exception:
        return False
    return tool_call_style(model_path) == "qwen_xml"


def _tool_parser_for(cfg, model_config: dict, model_path: str) -> str:
    """Which vLLM tool-call parser to launch with, or '' for none.

    An explicit setting always wins — including "off", because a model whose
    family is guessed wrongly parses tool calls into nonsense, and silently
    producing wrong tool arguments is worse than refusing the request.

    A name alone cannot separate the two Qwen tool-call syntaxes. Qwen2.5 and
    the Hermes-style finetunes put JSON inside ``<tool_call>``; current Qwen3
    releases emit XML ``<function=``/``<parameter=`` tags, and ``hermes`` fails
    on every one of those with a JSONDecodeError — measured 2026-10-21 on
    Qwen3.8-27B, where the pod logged that traceback per request and the call
    was lost entirely when streaming. So the TEMPLATE decides between them,
    since it is the model's own statement of what it emits.
    """
    want = str(getattr(cfg, "tool_call_parser", "") or "").strip()
    if want and want.lower() not in ("auto", "detect"):
        return "" if want.lower() in ("off", "none", "false", "0") else want
    hay = " ".join(str(x or "").lower() for x in (
        model_path, (model_config or {}).get("path"),
        (model_config or {}).get("id"), (model_config or {}).get("alias")))
    best = ""
    for needle, parser in _TOOL_PARSERS:
        if needle in hay and len(needle) > len(best):
            best, found = needle, parser
    parser = dict(_TOOL_PARSERS).get(best, "") if best else ""
    if parser == "hermes" and _emits_qwen_xml_calls(model_path):
        print("[vllm] template emits XML tool calls — using 'qwen3_xml', "
              "not 'hermes'", flush=True)
        return "qwen3_xml"
    return parser


def _tool_args(cfg, model_config: dict, model_path: str) -> list:
    """`--enable-auto-tool-choice --tool-call-parser X`, when we know X.

    Both or neither: vLLM refuses to START with --enable-auto-tool-choice and
    no parser, and a pod that will not boot is worse than one that cannot do
    tool calls. When the family is unrecognised this says so, because the
    symptom otherwise lands on the CLIENT as a 400 at request time.
    """
    parser = _tool_parser_for(cfg, model_config, model_path)
    if not parser:
        want = str(getattr(cfg, "tool_call_parser", "") or "").strip().lower()
        if want not in ("off", "none", "false", "0"):
            print(f"[vllm] no tool-call parser known for {model_path!r} — "
                  f"requests carrying `tools` will be refused by vLLM. Set "
                  f"vllm.tool_call_parser (or the model's vllm block) to one "
                  f"of: {', '.join(sorted(set(p for _, p in _TOOL_PARSERS)))}",
                  flush=True)
        return []
    print(f"[vllm] tool calls parsed as {parser!r}", flush=True)
    return ["--enable-auto-tool-choice", "--tool-call-parser", parser]


def _reasoning_args(cfg, model_config: dict, model_path: str) -> list:
    """`--reasoning-parser X`, when this model thinks and we know how it marks it.

    Without it the thought and the answer come back as ONE string. The closing
    </think> is a special token and vLLM decodes with skip_special_tokens, so
    the boundary is deleted before the text exists -- there is nothing left
    downstream to split on, and the whole thought is delivered to the client as
    the answer (measured 2026-10-22, Qwen3.8 on an A40).

    An explicit `reasoning_parser` wins, "off" included: a parser aimed at the
    wrong family hides the answer inside a reasoning field nobody reads, which
    is worse than not separating it.
    """
    want = str(getattr(cfg, "reasoning_parser", "") or "").strip()
    if want and want.lower() not in ("auto", "detect"):
        if want.lower() in ("off", "none", "false", "0"):
            return []
        print(f"[vllm] reasoning separated by {want!r} (configured)", flush=True)
        return ["--reasoning-parser", want]
    try:
        from codai.models.chat_style import vllm_reasoning_parser
        parser = vllm_reasoning_parser(model_path)
    except Exception:
        parser = ""
    if not parser:
        return []
    print(f"[vllm] reasoning separated by {parser!r}", flush=True)
    return ["--reasoning-parser", parser]


def _launch_cmd(py, cfg, host: str, port: int, model_path: str,
                served_name: Optional[str] = None, model_config: dict = None,
                gpu_memory_utilization: Optional[float] = None,
                max_model_len: Optional[int] = None,
                max_num_batched_tokens: Optional[int] = None,
                max_num_seqs: Optional[int] = None) -> list:
    mids = _served_names(model_config,
                         served_name or (getattr(cfg, "model_id", "") or ""),
                         model_path)
    cmd = [py, "-m", "vllm.entrypoints.openai.api_server",
           "--host", host, "--port", str(port),
           "--model", model_path,
           "--served-model-name"] + mids
    # A side job serving a small model (the OCR VLM engines) overrides the context the
    # LLM backend is configured with: it inherited ctx=18432, and the activation peak
    # that size profiles for was 9.4 GiB of the 11.1 GiB that made surya-ocr-2 unable to
    # start. A page is not an 18k-token conversation.
    ctx = int(max_model_len or 0) or int(getattr(cfg, "ctx", 0) or 0)
    if ctx > 0:
        cmd += ["--max-model-len", str(ctx)]
    # Already absolute (resolve_gmu translated a side job's own-share if there was one).
    gmu = float(gpu_memory_utilization or 0) or float(getattr(cfg, "gpu_memory_utilization", 0) or 0)
    if gmu > 0:
        cmd += ["--gpu-memory-utilization", str(gmu)]
    # The profiling peak that decides whether the engine can start is driven by how many
    # tokens may be in one batch, NOT by max_model_len (chunked prefill is on), so this
    # is the lever that shrinks the footprint without shrinking the usable context.
    mnbt = int(max_num_batched_tokens or 0)
    if mnbt > 0:
        cmd += ["--max-num-batched-tokens", str(mnbt)]
    tp = int(getattr(cfg, "tensor_parallel_size", 0) or 0)
    if tp > 0:
        cmd += ["--tensor-parallel-size", str(tp)]
    pp = int(getattr(cfg, "pipeline_parallel_size", 0) or 0)
    if pp > 1:
        cmd += ["--pipeline-parallel-size", str(pp)]
    executor = (getattr(cfg, "distributed_executor_backend", "") or "").strip()
    if not executor and needs_ray(cfg):
        executor = "ray"
    if executor:
        cmd += ["--distributed-executor-backend", executor]
    mns = int(max_num_seqs or 0) or int(getattr(cfg, "max_num_seqs", 0) or 0)
    if mns > 0:
        cmd += ["--max-num-seqs", str(mns)]
    dtype = (getattr(cfg, "dtype", "") or "").strip()
    if dtype:
        cmd += ["--dtype", dtype]
    quant = (getattr(cfg, "quantization", "") or "").strip()
    if quant:
        cmd += ["--quantization", quant]
    # LoRA adapters configured on the MODEL (not on the vLLM backend): vLLM takes
    # them at launch as name=source pairs, where source is a path on this host or
    # a HuggingFace repo id. A QLoRA adapter is an ordinary LoRA here — the
    # quantisation is how the base model loads, which --quantization covers.
    cmd += _lora_args(model_config)
    # Tool calling. vLLM answers 400 to any request carrying `tools` unless it
    # was launched for it, and the message names flags the caller cannot set.
    cmd += _tool_args(cfg, model_config, model_path)
    cmd += _reasoning_args(cfg, model_config, model_path)

    # TOKEN IDS IN LOGPROBS. `ocr.surya_repair_accents` rebuilds a reply whose
    # accents were lost in detokenisation from the model's own token ids, and
    # vLLM only reports ids (`token_id:2131`) when launched with this flag --
    # `bytes` is empty for exactly the tokens that lost their characters, so
    # without it there is nothing to rebuild from and the repair silently does
    # nothing.
    #
    # Why it is forced rather than configured: the operator DID configure it
    # (`vllm.extra_args` on the renting install) and it still regressed, because
    # a rented pod launches its own vLLM from its own seeded config and that
    # setting never travelled. A page then came back with 25 replacement
    # characters, zero 0xC3 bytes and a surviving 3-byte em-dash -- the repair
    # unable to run, on a fix that had been shipped and verified months
    # earlier. A guard that depends on remembering to set it on every install
    # is not a guard.
    #
    # Safe to force: it changes only the `token` STRING inside logprobs, which
    # nothing in coderai reads except that repair.
    if _wants_token_ids(cfg, served_name, model_path):
        if "--return-tokens-as-token-ids" not in cmd:
            cmd += ["--return-tokens-as-token-ids"]

    extra = (getattr(cfg, "extra_args", "") or "").strip()
    if extra:
        cmd += shlex.split(extra)
    # Deduplicate: an operator who also put the flag in `extra_args` (as the
    # original fix did) must not launch vLLM with it twice.
    out, seen = [], set()
    for tok in cmd:
        if tok == "--return-tokens-as-token-ids":
            if tok in seen:
                continue
            seen.add(tok)
        out.append(tok)
    return out


#: Served names (and path fragments) that NEED token ids: a byte-level
#: vocabulary whose lost characters can only be recovered from them, AND a
#: repair that actually consumes them.
#:
#: surya only, deliberately. chandra and olmOCR are served by coderai's own
#: client, read their logprobs for confidence (`mean_token_prob`, a VALUE) and
#: have no id-based repair, so for them the flag would change the `token`
#: string that nothing reads -- and chandra demonstrably does not have the
#: fault: on the page where surya returned 25 U+FFFD and zero 0xC3 bytes,
#: chandra returned 40 0xC3 bytes with every accent intact.
_BYTE_LEVEL_OCR = ("surya",)


def _wants_token_ids(cfg, served_name: str = "", model_path: str = "") -> bool:
    """Whether this engine must report token ids in its logprobs.

    True when the operator asked (`vllm.return_token_ids`), and ALWAYS for a
    byte-level OCR model, whose accent repair cannot work without them.
    """
    if bool(getattr(cfg, "return_token_ids", False)):
        return True
    hay = f"{served_name or ''} {model_path or ''}".lower()
    return any(name in hay for name in _BYTE_LEVEL_OCR)


def _base_of(entry: dict) -> str:
    """What weights a model entry loads, for deciding which models share a base."""
    block = entry.get("runpod") if isinstance(entry.get("runpod"), dict) else {}
    return str(entry.get("path") or block.get("hf_repo") or entry.get("id") or "").strip().lower()


def _entry_name(entry: dict) -> str:
    """What a model entry is ASKED FOR by.

    `alias` first: a seeded entry (the only kind a pod has) carries one and no
    `id`, while a models.json entry carries both.
    """
    if not isinstance(entry, dict):
        return ""
    return (str(entry.get("alias") or "").strip()
            or str(entry.get("id") or "").strip())


def _catalogue_entries() -> list:
    """Every model entry in the catalogue, whatever SECTION lists it.

    A model's type is its section (main.py's _CATEGORY_TYPE_PREFIX), so one set
    of weights served as text and as vision is two entries in two lists. Both
    callers below ask a question about WEIGHTS -- which names one engine must
    answer to, and which adapters it must register at launch -- and the answer
    never depended on which list a name was written in. Reading only
    `text_models` made a vision entry on the same base invisible to both: its
    name was left off the launch (a relaunch per alternating request, which on
    a 27B sharing a card is an OOM), and its adapter was never registered, so
    the engine came up with no --enable-lora and answered under the adapter's
    name with BASE weights, failing nothing.

    One reader, so a third caller cannot drift from these two.
    """
    try:
        from codai.admin.routes import config_manager
        md = getattr(config_manager, "models_data", None) or {}
    except Exception:
        return []
    out = []
    for lst in md.values():
        if isinstance(lst, list):
            out.extend(e for e in lst if isinstance(e, dict))
    return out


def _sibling_lora_specs(model_config: dict) -> list:
    """Adapters of this model AND of every other model loading the same base.

    vLLM registers adapters at LAUNCH, and one vLLM serves one base model — so a
    pod launched for the base alone could never serve an adapter of it, however
    the request was routed. Observed as the whole point of a shared pool not
    working: ``qwen38-awq`` creates the pool, its vLLM starts with no
    ``--enable-lora``, and a later ``diogene-alpha`` request has nowhere to go.

    Collecting siblings here makes one engine serve the base and each adapter
    under its own served name, which is what lets different requests select a
    different adapter with no restart.
    """
    from codai.models.text_loras import configured_specs

    specs = list(configured_specs(model_config or {}))
    base = _base_of(model_config or {})
    if not base:
        return specs
    # Not `id`: a SEEDED entry has none. The kind of entry a pod holds is the
    # kind that arrives in CODERAI_SEED_MODELS, and that carries `alias`, not
    # `id` — so "is this me?" compared "" with "" and matched every sibling,
    # excluding all of them. The pod then launched vLLM with no --enable-lora
    # and answered under the adapter's name with BASE weights. Nothing failed.
    mine = _entry_name(model_config)
    for entry in _catalogue_entries():
        if not isinstance(entry, dict) or entry is model_config:
            continue
        name = _entry_name(entry)
        if name and mine and name == mine:
            continue
        if _base_of(entry) != base:
            continue
        for spec in configured_specs(entry):
            # A sibling's adapter is served under the sibling's OWN name: that is
            # the name the request will ask for.
            spec = dict(spec)
            spec["name"] = name or str(spec.get("name") or "")
            specs.append(spec)
    seen, unique = set(), []
    for spec in specs:
        if spec.get("name") in seen:
            continue
        seen.add(spec.get("name"))
        unique.append(spec)
    return unique


def _served_names(model_config: dict, served_name: str, fallback: str) -> list:
    """Every name this ONE engine must answer to.

    One engine serves a base and each adapter of it, so the launch cannot name
    only the model whose request happened to start it: whichever of the two
    arrived second would be unreachable, which is the failure this whole path
    exists to stop. Adapter names come from ``--lora-modules`` and must NOT be
    repeated here — vLLM would then have a model and an adapter under one name.
    """
    adapters = {str(s.get("name") or "") for s in _sibling_lora_specs(model_config or {})}
    names = [str(served_name or "")]
    base = _base_of(model_config or {})
    if adapters and base:
        # Several names over ONE engine. Which of them asked for it must not
        # decide what the engine answers to — order-dependence is the bug this
        # is being changed for — so the full set is collected every time: the
        # weights' own name, and every name the catalogue gives this base.
        #
        # Only when there ARE adapters. A model served alone is named exactly as
        # it was before, because nothing about it is ambiguous.
        names.append(str(fallback or ""))
        for entry in _catalogue_entries():
            if _base_of(entry) != base:
                continue
            names.append(str(entry.get("alias") or "").strip())
            names.append(str(entry.get("id") or "").strip())
    out, seen = [], set()
    for n in names:
        if n and n not in seen and n not in adapters:
            seen.add(n)
            out.append(n)
    return out or [fallback]


#: This engine serves a shared POOL — several names over one copy of the
#: weights — so the LoRA machinery is allocated whether or not an adapter is
#: known yet. Set by the orchestrator when the model's runpod block names a
#: pool (see runpod_worker._plan); a model entry's own `runpod.pool` works too,
#: for an install driving vLLM directly.
_ENABLE_LORA_ENV = "CODERAI_VLLM_ENABLE_LORA"


def _serves_a_pool(model_config: dict) -> bool:
    raw = (os.environ.get(_ENABLE_LORA_ENV) or "").strip().lower()
    if raw in ("1", "true", "on", "yes"):
        return True
    block = (model_config or {}).get("runpod")
    return bool(isinstance(block, dict) and str(block.get("pool") or "").strip())


def _lora_capacity_args(n_mods: int) -> list:
    """The flags that must be decided at LAUNCH, separate from the adapter list.

    vLLM takes `--lora-modules` as a convenience; what it cannot change later is
    the CAPACITY. `--enable-lora` allocates the machinery, `--max-lora-rank`
    sizes the slots, `--max-loras` decides how many distinct adapters may be
    live in one batch. Get those right and the list itself can grow at runtime.
    """
    # vLLM rejects an adapter whose rank exceeds --max-lora-rank (default 16),
    # and the message points at the flag rather than the adapter, so raise the
    # ceiling to what vLLM supports rather than have a trained LoRA refused.
    #
    # --max-loras default is 1, so with several registered, requests naming
    # different ones cannot batch together and serialise instead — the cost
    # lands on latency, not on an error, which is why it goes unnoticed. The
    # floor of 4 is for a pool that has no adapters YET: the engine must be able
    # to batch across the ones it is about to be given.
    return ["--max-lora-rank", "64",
            "--max-loras", str(max(4, min(n_mods, 8)) if n_mods else 4)]


def _lora_args(model_config: dict) -> list:
    """`--enable-lora --lora-modules …` for the adapters this engine must serve."""
    try:
        from codai.models.text_loras import describe, is_portable, local_path
        specs = _sibling_lora_specs(model_config or {})
    except Exception:
        specs = []
    mods, usable = [], []
    for spec in specs:
        src = str(spec.get("source") or "")
        if src.startswith(("sha256:", "name:")):
            # An uploaded adapter is named by CONTENT id, which is not a path: it
            # has to be looked up in the blob store first. Passing the id straight
            # through produced `--lora-modules name=sha256:…`, which vLLM cannot
            # open — and the launch failed minutes later, inside a pod.
            resolved = local_path(src)
            if not resolved:
                print(f"[lora] {spec.get('name')!r}: adapter {src} is not in this "
                      f"machine's blob store — serving without it", flush=True)
                continue
        else:
            # A path or a repo id is passed as configured: vLLM resolves both,
            # and second-guessing a path here would drop an adapter that a
            # mount makes appear a moment later.
            resolved = src
        mods.append(f"{spec['name']}={resolved}")
        usable.append(spec)
    if not mods:
        if not _serves_a_pool(model_config):
            return []
        # A pool pod with no adapter yet. `--enable-lora` cannot be turned on
        # afterwards, so an engine launched without it has to RELOAD the base
        # model — minutes, on a card being paid for by the hour — the first
        # time an adapter arrives. Opening the door costs a buffer allocation
        # and nothing else, and only a model that declares a pool pays it.
        print("[lora] no adapters yet, but this engine serves a pool — "
              "starting with LoRA enabled so one can be added without a "
              "reload", flush=True)
        return ["--enable-lora"] + _lora_capacity_args(0)
    args = ["--enable-lora", "--lora-modules"] + mods
    args += _lora_capacity_args(len(mods))
    print(f"[lora] vLLM will serve adapters: {describe(usable)}", flush=True)
    return args


def _model_config_for(model_name: str) -> dict:
    """The model's models.json entry — vLLM serves models from the model list, so
    its LoRA settings live on the model, not on the vLLM backend config."""
    try:
        from codai.models.manager import _model_entry_for
        return _model_entry_for(model_name) or {}
    except Exception:
        return {}


#: (engine url, adapter name) -> the source that engine was given. A retrain
#: keeps the name and changes the bytes, so "does it serve this name?" is not
#: the same question as "does it serve THIS adapter?".
_LORA_LOADED: dict = {}


def _adapter_source(model_config: dict, served_name: str) -> str:
    """The source configured for ``served_name`` right now, or ''."""
    want = (served_name or "").strip()
    if not want:
        return ""
    for spec in _sibling_lora_specs(model_config or {}):
        if str(spec.get("name") or "") == want:
            return str(spec.get("source") or "")
    return ""


def _note_loaded(url: str, model_config: dict, served_name: str) -> None:
    src = _adapter_source(model_config, served_name)
    if src:
        _LORA_LOADED[(url, served_name)] = src


def _adapter_is_stale(url: str, model_config: dict, served_name: str) -> bool:
    """Does this engine hold an OLDER version of the adapter it answers to?

    Retraining ships under the same name. Without this the engine kept the
    adapter it was launched with and answered confidently from stale weights —
    the update looked applied, because the catalogue had it and the name worked.

    Unknown counts as NOT stale: an engine we have no record for is one we did
    not load, and unloading an adapter on a guess is worse than serving it.
    """
    want = _adapter_source(model_config, served_name)
    if not want:
        return False
    had = _LORA_LOADED.get((url, served_name))
    return bool(had) and had != want


def _unload_lora_runtime(url: str, served_name: str) -> bool:
    """Drop an adapter from a running engine, so a new version can take its name."""
    import json as _json
    import urllib.request

    body = _json.dumps({"lora_name": served_name}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/unload_lora_adapter",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=120)
    except Exception as exc:
        print(f"[lora] could not unload {served_name!r} from {url} ({exc})",
              flush=True)
        return False
    _LORA_LOADED.pop((url, served_name), None)
    return True


def _load_lora_runtime(url: str, model_config: dict, served_name: str) -> bool:
    """Register an adapter into a RUNNING engine. True when it will now serve.

    vLLM takes `--lora-modules` at launch, so for a long time the only way to
    add one was to restart — a full reload of the base model, minutes, on a card
    billed by the hour, and every request in flight lost. `/v1/load_lora_adapter`
    does it in place (vLLM >= 0.9, behind VLLM_ALLOW_RUNTIME_LORA_UPDATING,
    which the launch sets whenever it enabled LoRA).

    Returns False for every reason it might not work — an engine started without
    `--enable-lora`, an older vLLM, the flag unset, an adapter whose rank exceeds
    the slot size — and the caller falls back to the restart. A pod rented an
    hour ago may be running an older image than the one we would launch today,
    so "it should be supported" is not something this can assume.
    """
    import json as _json
    import urllib.error
    import urllib.request

    want = (served_name or "").strip()
    if not want:
        return False
    spec = next((s for s in _sibling_lora_specs(model_config or {})
                 if str(s.get("name") or "") == want), None)
    if spec is None:
        print(f"[lora] {want!r} is not an adapter of this model here — nothing "
              f"to add to the running engine", flush=True)
        return False
    src = str(spec.get("source") or "")
    try:
        from codai.models.text_loras import local_path
        path = local_path(src) if src.startswith(("sha256:", "name:")) else src
    except Exception as exc:
        print(f"[lora] could not resolve {src} ({exc})", flush=True)
        path = ""
    if not path:
        # Said out loud because the alternative is this function returning
        # False with no explanation, and the engine then serving the BASE
        # under the adapter's name. That is exactly how an id mismatch between
        # the catalogue and the blob store stayed invisible for three releases.
        print(f"[lora] {want!r}: {src} is not in this machine's blob store — "
              f"cannot add it to the running engine. The catalogue and the "
              f"store disagree about this adapter's id.", flush=True)
        return False
    body = _json.dumps({"lora_name": want, "lora_path": path}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/load_lora_adapter",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:200]
        except Exception:
            pass
        print(f"[lora] {url} refused {want!r} at runtime ({exc.code} {detail}) "
              f"— falling back to a restart", flush=True)
        return False
    except Exception as exc:
        print(f"[lora] could not reach {url} to add {want!r} ({exc}) — falling "
              f"back to a restart", flush=True)
        return False
    print(f"[lora] added {want!r} to the running engine on {url} — no reload",
          flush=True)
    _LORA_LOADED[(url, want)] = str(spec.get("source") or "")
    return True


def _serves_name(url: str, name: str):
    """True / False / None — the last meaning "could not tell".

    The difference matters: not serving a name costs a relaunch, and failing to
    ASK looks identical from here. Restarting a loaded 27B because one probe
    timed out would be far worse than leaving the doubt, so an unanswered probe
    is never a verdict.
    """
    import requests
    try:
        r = requests.get(url + "/v1/models", timeout=5)
        data = r.json().get("data")
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    ids = [str(m.get("id", "")) for m in data if isinstance(m, dict)]
    tail = name.rstrip("/").split("/")[-1]
    return any(i == name or i.rstrip("/").split("/")[-1] == tail for i in ids)


def _remote_serves(url: str, name: str) -> bool:
    """True when a remote vLLM's /v1/models lists ``name`` (or a suffix match)."""
    import requests
    try:
        r = requests.get(url + "/v1/models", timeout=5)
        ids = [str(m.get("id", "")) for m in (r.json().get("data") or [])]
    except Exception:
        return False
    tail = name.rstrip("/").split("/")[-1]
    return any(i == name or i.rstrip("/").split("/")[-1] == tail for i in ids)


def needs_ray(cfg) -> bool:
    """Multi-node: other machines' GPUs join through a ray cluster."""
    from codai.cluster.multinode import parse_nodes
    if (getattr(cfg, "distributed_executor_backend", "") or "").strip().lower() == "ray":
        return True
    if int(getattr(cfg, "pipeline_parallel_size", 1) or 1) > 1:
        return True
    if (getattr(cfg, "ray_address", "") or "").strip():
        return True
    return bool(parse_nodes(getattr(cfg, "nodes", None)))


def _local_gpu_count(env: dict) -> int:
    vis = (env.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if vis:
        return len([x for x in vis.split(",") if x.strip()])
    try:
        from codai.backends.gpu_probe import total_memory_bytes
        return max(1, len(total_memory_bytes()))
    except Exception:
        return 1


def _start_ray(cfg, py: str, env: dict):
    """Bring the ray cluster up for this launch; returns (RayCluster, address)."""
    from codai.cluster.multinode import RayCluster
    from codai.cluster.rpc import advertise_host
    adv = ""
    try:
        from codai.admin.routes import config_manager
        adv = getattr(getattr(config_manager.config, "cluster", None), "advertise_host", "") or ""
    except Exception:
        pass
    adv = advertise_host(adv)
    tp = max(1, int(getattr(cfg, "tensor_parallel_size", 1) or 1))
    pp = max(1, int(getattr(cfg, "pipeline_parallel_size", 1) or 1))
    cluster = RayCluster(py, adv, port=int(getattr(cfg, "ray_port", 6379) or 6379),
                         address=getattr(cfg, "ray_address", "") or "",
                         nodes=getattr(cfg, "nodes", None),
                         ready_timeout_s=float(getattr(cfg, "nodes_ready_timeout_s", 600) or 600))
    address = cluster.start(gpus_needed=tp * pp, local_gpus=_local_gpu_count(env))
    env.setdefault("VLLM_HOST_IP", adv)
    env["RAY_ADDRESS"] = address
    return cluster, address


class _StillStarting(Exception):
    """Another caller is starting this engine; wait for it instead."""

    def __init__(self, svc):
        super().__init__("vLLM still starting")
        self.svc = svc


def _mark_ready(svc_key: str, proc) -> None:
    with _lock:
        svc = _services.get(svc_key)
        if svc and svc.get("proc") is proc:
            svc["ready"] = True
            ev = svc.get("ready_event")
            if ev is not None:
                ev.set()


def _wait_for_start(svc: dict, svc_key: str, ready_timeout: float) -> str:
    """Wait for an engine another caller is starting. Its own launcher reports
    failures; this only needs to know whether it came up."""
    deadline = time.time() + ready_timeout
    ev = svc.get("ready_event")
    while time.time() < deadline:
        if svc["proc"].poll() is not None:
            raise RuntimeError(f"vLLM for {svc_key} exited while another caller "
                               f"was starting it")
        if svc.get("ready") or (ev is not None and ev.wait(2.0)):
            return svc["url"]
        if ev is None:
            time.sleep(2)
    raise RuntimeError(f"vLLM for {svc_key} did not become ready in time "
                       f"(started by another caller)")


_BUSY_SAID: dict = {}


def _say_busy(svc_key: str, url: str) -> None:
    now = time.time()
    if now - _BUSY_SAID.get(svc_key, 0.0) > 60:
        _BUSY_SAID[svc_key] = now
        print(f"[vllm] {svc_key} at {url} is slow to answer its health check "
              f"but its process is alive — treating it as BUSY and keeping it, "
              f"not starting a second engine on the same card", flush=True)


def _ensure_service_once(cfg, model_path: Optional[str] = None,
                         served_name: Optional[str] = None,
                         ready_timeout: float = 3600.0,
                         gmu_absolute: float = 0.0,
                         max_model_len: Optional[int] = None,
                         max_num_batched_tokens: Optional[int] = None,
                         max_num_seqs: Optional[int] = None) -> str:
    """Launch (or reuse) a vLLM OpenAI server for a model; return its base URL.

    ``model_path``/``served_name`` override the config (used by the OCR subsystem to serve
    surya-2 on its own vLLM instance alongside any LLM instance)."""
    # An explicit `service_url` points at a vLLM running somewhere else — another
    # host, a container, a rented RunPod pod (where vLLM's own image is the ready-
    # made option) — so nothing is spawned or downloaded here. This function
    # already returns a URL, so its callers cannot tell the difference.
    _remote = (str(getattr(cfg, "service_url", "") or "")
               or os.environ.get("CODERAI_VLLM_SERVICE_URL") or "").strip()
    if _remote:
        _remote = _remote.rstrip("/")
        if not _health_ok(_remote):
            raise RuntimeError(
                f"configured service_url {_remote} is not answering its health check")
        # A remote vLLM serves whatever it was started with. When the caller asks
        # for a specific model (the OCR path does, for surya-2), make sure that
        # model is actually there rather than silently proxying to the wrong one.
        _want = (served_name or model_path or "").strip()
        if _want and not _remote_serves(_remote, _want):
            raise RuntimeError(
                f"configured service_url {_remote} does not serve {_want}")
        print(f"[vllm] using the configured remote service at {_remote}", flush=True)
        return _remote

    resolved, svc_key = resolve_service_key(cfg, model_path)
    try:
        with _lock:
            svc = _services.get(svc_key)
            _alive = bool(svc) and svc["proc"].poll() is None
            if _alive and not svc.get("ready", True):
                # STILL STARTING. Waiting for it is the only right answer: a
                # second launch puts two engines on one card, each sized for
                # all of it -- measured on a rented A40, two EngineCores one
                # second apart, then `-9` kills and a relaunch every few
                # seconds for twenty minutes.
                raise _StillStarting(svc)
            if _alive and not _health_ok(svc["url"]) and not _health_ok(
                    svc["url"], timeout=15):
                # BUSY, NOT DEAD. A live engine under load can miss a 3 s
                # probe; replacing it then is what started the storm above
                # (and the "10.22 of 47 GiB free" crash): the old process
                # keeps the card while the new one cannot fit. It is still
                # the engine for this model, so hand it out.
                _say_busy(svc_key, svc["url"])
                return svc["url"]
            if _alive and _health_ok(svc["url"]):
                # Alive is not the same as useful. An engine serves the names it was
                # LAUNCHED with, and this instance may have started before the name
                # now being asked for existed here at all — a pod is taught models
                # after it boots, and an adapter registered afterwards is not in any
                # running engine's --lora-modules. Reusing it then answers 404 for
                # as long as the engine lives.
                #
                # Only an answered probe that omits the name forces a relaunch: a
                # probe that fails says nothing, and restarting a loaded 27B on a
                # timeout would be far worse than the doubt.
                mcfg_entry = _model_config_for(resolved or model_path or "")
                if not served_name or _serves_name(svc["url"], served_name) is not False:
                    # It answers to the name. Does it answer with the adapter that
                    # is configured NOW? A retrain keeps the name.
                    if not _adapter_is_stale(svc["url"], mcfg_entry, served_name):
                        return svc["url"]
                    print(f"[lora] {served_name!r} on {svc['url']} is an older "
                          f"version of the adapter — swapping it in the running "
                          f"engine", flush=True)
                    if _unload_lora_runtime(svc["url"], served_name) \
                            and _load_lora_runtime(svc["url"], mcfg_entry, served_name):
                        return svc["url"]
                    print(f"[lora] could not swap {served_name!r} in place — "
                          f"restarting the engine so it picks up the new adapter",
                          flush=True)
                # An ADAPTER can usually just be handed to the running engine. Try
                # that before throwing away a loaded base model.
                elif _load_lora_runtime(svc["url"], mcfg_entry, served_name):
                    return svc["url"]
                print(f"[vllm] the engine on {svc['url']} does not serve "
                      f"{served_name!r} — it started before this model was known "
                      f"here; restarting it so one engine serves the base and every "
                      f"adapter of it", flush=True)
                _services.pop(svc_key, None)
                _stop_proc(svc, svc_key)
            elif svc:
                # The process is gone. Stop what is left of it (ray, pipes)
                # rather than only forgetting it.
                _services.pop(svc_key, None)
                try:
                    _stop_proc(svc, svc_key)
                except Exception:
                    pass

            py = ensure_built(cfg)
            model = resolved or (getattr(cfg, "model_path", "") or "").strip()
            if not model:
                raise RuntimeError(
                    "vLLM: no model resolved. Set vllm.model_path (an HF model dir or id).")

            host = (getattr(cfg, "host", "127.0.0.1") or "127.0.0.1").strip()
            port = int(getattr(cfg, "port", 0) or 0) or _free_port()
            url_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
            url = f"http://{url_host}:{port}"
            # The share is resolved by the caller (ensure_service, which may be escalating
            # after a too-small budget) and remembered here: what this instance actually
            # reserved is what stopping it frees, and the eviction path has to be told the
            # truth or it over- or under-evicts for the next model.
            gmu = float(gmu_absolute or 0.0)
            cmd = _launch_cmd(py, cfg, host, port, model, served_name,
                              model_config=_model_config_for(resolved or model),
                              gpu_memory_utilization=gmu,
                              max_model_len=max_model_len,
                              max_num_batched_tokens=max_num_batched_tokens,
                              max_num_seqs=max_num_seqs)

            env = os.environ.copy()
            # flashinfer JIT-compiles CUDA kernels with ninja at runtime, which fails on
            # hosts without a full build toolchain (nvcc/gcc wired for it). Default it OFF so
            # vLLM uses FLASH_ATTN + a native sampler out of the box; the user can re-enable
            # via extra_env. (Set before extra_env so an explicit override wins.)
            env.setdefault("VLLM_USE_FLASHINFER", "0")
            env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
            # An adapter can then be registered into the RUNNING engine
            # (/v1/load_lora_adapter) instead of costing a reload of the base
            # model. Only where the launch actually enabled LoRA: the flag is
            # meaningless otherwise, and setting it everywhere would be a claim we
            # cannot honour.
            if "--enable-lora" in cmd:
                env.setdefault("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "1")
            # Pin this vLLM instance to specific NVIDIA GPU(s) if configured (CUDA-only).
            gpu = (getattr(cfg, "gpu", "") or "").strip()
            if gpu:
                env["CUDA_VISIBLE_DEVICES"] = gpu
            extra_env = (getattr(cfg, "extra_env", "") or "").strip()
            applied = {}
            if extra_env:
                for tok in shlex.split(extra_env):
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        if k.strip():
                            env[k.strip()] = v; applied[k.strip()] = v
            ray = None
            if needs_ray(cfg):
                ray, _addr = _start_ray(cfg, py, env)
            print(f"[vllm] launching: {' '.join(cmd)}"
                  + (f"  ({' '.join(f'{k}={v}' for k, v in applied.items())})" if applied else ""),
                  flush=True)
            tail = collections.deque(maxlen=80)
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, env=env)
            except Exception:
                if ray is not None:
                    ray.stop()
                raise
            meta = {}
            threading.Thread(target=_pump_logs, args=(proc, tail, meta), daemon=True).start()
            _services[svc_key] = {"proc": proc, "port": port, "url": url, "ray": ray,
                                  "gmu": gmu, "meta": meta, "ready": False,
                                  "ready_event": threading.Event()}
            # Which VERSION of each adapter this engine was launched with, so a
            # retrain under the same name is recognised as a different adapter
            # rather than a name the engine already answers to.
            _LORA_LOADED.pop((url, served_name or ""), None)
            for _spec in _sibling_lora_specs(_model_config_for(resolved or model) or {}):
                _nm = str(_spec.get("name") or "")
                if _nm:
                    _LORA_LOADED[(url, _nm)] = str(_spec.get("source") or "")

        def _tail_msg():
            # The last six lines of a vLLM crash are the wrapper's own traceback
            # ("Engine core initialization failed. See root cause above.") — the
            # root cause IS above, and on a pod this message is all anyone gets.
            # Prefer the lines that name an error; fall back to a longer tail.
            lines = [l.strip() for l in tail if l.strip()]
            keyed = [l for l in lines if any(k in l for k in ("Error", "error:", "not found",
                                                              "No such", "CUDA", "Traceback"))]
            chosen = (keyed[-8:] if keyed else []) + lines[-6:]
            seen, out = set(), []
            for l in chosen:
                if l not in seen:
                    seen.add(l); out.append(l[:300])
            joined = " | ".join(out)
            return f". Last output: {joined}" if joined else ""

        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                stop_service(svc_key)
                raise RuntimeError(
                    f"vLLM exited (code {proc.returncode}) before becoming ready" + _tail_msg())
            if _health_ok(url):
                print(f"[vllm] service ready for {svc_key} at {url}", flush=True)
                _mark_ready(svc_key, proc)
                return url
            time.sleep(2)
        stop_service(svc_key)
        raise RuntimeError(f"vLLM for {svc_key} did not become ready in time" + _tail_msg())
    except _StillStarting as st:
        return _wait_for_start(st.svc, svc_key, ready_timeout)


# A budget too small for the engine's own footprint. vLLM says this whatever the model,
# card or context is, which is why the response is to measure rather than to guess.
_KV_EXHAUSTED = re.compile(
    r"No available memory for the cache blocks"
    r"|Available KV cache memory: *-"
    r"|less than desired GPU memory utilization",
    re.IGNORECASE)

# How much to add per attempt, and the most to try. Each attempt costs a real boot
# (minutes for a VLM), so the steps are coarse.
_GMU_ESCALATION_STEP = 0.12
_GMU_MAX_ATTEMPTS = 4

# vLLM reports what was left for the cache; this is how that figure is read back.
_KV_REPORT = re.compile(r"Available KV cache memory:\s*(-?[\d.]+)\s*GiB", re.IGNORECASE)
# ...and how many TOKENS that bought, which is the only way to turn a share into a
# window: bytes-per-token depends on the model's attention shape (this hybrid puts
# linear attention on most layers) and cannot be derived from the config.
_KV_TOKENS = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens", re.IGNORECASE)
# How much KV cache an instance should end up with, how much spare has to be on the table
# before shrinking is worth a move, and the margin kept when proposing a smaller share.
#
# 2 GiB is the right floor for a page-at-a-time VLM, and catastrophic for an LLM
# serving long conversations: measured on an A40 2026-10-22, 2 GiB holds ~32,000
# tokens of this model's cache, while the engine was serving a 131,072-token
# window -- not even one full request. The learner proposed exactly that (0.900
# -> 0.460, "would still give ~2.0 GiB") the moment the engine booted, and a
# vLLM whose cache cannot hold max_model_len refuses to start at all. So the
# target is a FLOOR, raised to whatever holds the window this engine is actually
# serving, times the concurrency below.
_TARGET_KV_GB = 2.0
# Full-length requests the cache must be able to hold at once before any share is
# given back. 2 is the smallest number that is not "one request at a time".
_KV_MIN_CONCURRENCY = 2.0
_SHRINK_MIN_SLACK_GB = 1.0
_SHRINK_MARGIN = 0.03


def _learned_gmu_path():
    try:
        from codai.platform_paths import legacy_style_config_dir
        return legacy_style_config_dir() / "vllm_gmu_learned.json"
    except Exception:
        return None


def _learned_gmu_key(cfg, model: str, ctx, mnbt) -> str:
    """What a learned share is valid FOR.

    The footprint depends on the model, how much memory the card has, and the limits it
    was profiled under — so a value learned for surya-ocr-2 on a 24 GB card at ctx 18432
    says nothing about olmOCR-2, or about the same model on a 48 GB card. Keying on all
    of it is what keeps this generic instead of a number that happened to fit once."""
    try:
        from codai.models.manager import multi_model_manager
        total = round(float(multi_model_manager._total_vram_gb() or 0.0), 1)
    except Exception:
        total = 0.0
    return f"{model}|{total}|{int(ctx or 0)}|{int(mnbt or 0)}"


def _load_learned(key: str) -> dict:
    """``{"ok": share that booted, "failed": highest share that could not hold the cache}``

    Both halves matter: ``ok`` lets a later start skip rediscovery (and lets a measured
    share come DOWN below the configured bid), while ``failed`` is a floor that stops the
    downward half ever proposing a budget already proven dead — which is what would
    otherwise oscillate."""
    p = _learned_gmu_path()
    if not p:
        return {}
    try:
        with open(p) as f:
            rec = (json.load(f) or {}).get(key)
    except Exception:
        return {}
    if isinstance(rec, (int, float)):          # the flat format this used to write
        return {"ok": float(rec)}
    return rec if isinstance(rec, dict) else {}


def _save_learned(key: str, ok: float = None, failed: float = None) -> None:
    """Record what booted and/or what was too small. Best-effort: a read-only config dir
    costs a few minutes of re-discovery, not a failure."""
    p = _learned_gmu_path()
    if not p:
        return
    try:
        data = {}
        try:
            with open(p) as f:
                data = json.load(f) or {}
        except Exception:
            data = {}
        rec = data.get(key)
        if isinstance(rec, (int, float)):
            rec = {"ok": float(rec)}
        if not isinstance(rec, dict):
            rec = {}
        before = dict(rec)
        if ok is not None and ok > 0:
            rec["ok"] = round(float(ok), 3)
        if failed is not None and failed > 0:
            rec["failed"] = round(max(float(failed), float(rec.get("failed") or 0.0)), 3)
        if rec == before:
            return
        data[key] = rec
        tmp = str(p) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, p)
        print(f"[vllm] learned {rec} for {key}", flush=True)
    except Exception as e:
        print(f"[vllm] could not persist the learned share: {e}", flush=True)


def _starting_gmu(configured: float, rec: dict) -> float:
    """Where to begin: what booted before if we know it, else the configured bid — never
    at or below a share that has already failed."""
    start = float(rec.get("ok") or 0.0) or float(configured or 0.0)
    failed = float(rec.get("failed") or 0.0)
    if failed > 0:
        start = max(start, min(failed + _GMU_ESCALATION_STEP, GMU_CEILING))
    return min(max(start, 0.0), GMU_CEILING)


def _kv_floor_gb(ctx, kv_gib, kv_tokens, target_kv_gb: float,
                 min_concurrency: float = None) -> float:
    """The smallest cache worth keeping: ``target_kv_gb``, or enough to hold
    ``min_concurrency`` requests of ``ctx`` tokens when that is larger.

    Bytes per token come from the boot report (``kv_gib`` for ``kv_tokens``)
    because they depend on the model's attention shape -- this hybrid runs
    linear attention on most of its layers -- and are not derivable from the
    config. With no report to read, the target is returned unchanged: a floor
    invented from a guess could refuse to shrink anything, for ever.
    """
    try:
        ctx = int(ctx or 0)
        kv_gib = float(kv_gib or 0)
        kv_tokens = int(kv_tokens or 0)
    except (TypeError, ValueError):
        return float(target_kv_gb)
    if ctx <= 0 or kv_gib <= 0 or kv_tokens <= 0:
        return float(target_kv_gb)
    try:
        conc = float(_KV_MIN_CONCURRENCY if min_concurrency is None
                     else min_concurrency)
    except (TypeError, ValueError):
        conc = _KV_MIN_CONCURRENCY
    if conc <= 0:                 # 0 opts out: keep only the flat target
        return float(target_kv_gb)
    per_token_gb = kv_gib / float(kv_tokens)
    return max(float(target_kv_gb), ctx * conc * per_token_gb)


def _shrink_candidate(gmu: float, kv_gib: float, target_kv_gb: float,
                      rec: dict) -> Optional[float]:
    """A smaller share that would still leave ``target_kv_gb`` of cache, or None.

    Escalation alone only ever ratchets upward, so an opening bid that is too generous
    would have the engine sit on memory it does not use — 0.60 of this card when it needs
    0.25 is 8 GB another model could have had. The successful boot reports the cache it
    got, which gives the footprint directly: ``footprint = budget - kv``. Kept above any
    share known to fail, and with a margin, so shrinking cannot trade a working budget
    for a dead one."""
    try:
        from codai.models.manager import multi_model_manager
        total = float(multi_model_manager._total_vram_gb() or 0.0)
    except Exception:
        return None
    if total <= 0 or kv_gib is None:
        return None
    slack = float(kv_gib) - float(target_kv_gb)
    if slack <= _SHRINK_MIN_SLACK_GB:          # not enough spare cache to be worth a move
        return None
    footprint = (gmu * total) - float(kv_gib)
    if footprint <= 0:
        return None
    cand = ((footprint + float(target_kv_gb)) / total) + _SHRINK_MARGIN
    floor = float(rec.get("failed") or 0.0)
    if floor > 0:
        cand = max(cand, min(floor + _GMU_ESCALATION_STEP, GMU_CEILING))
    cand = min(max(cand, 0.05), GMU_CEILING)
    if cand >= gmu - _SHRINK_MARGIN:           # not meaningfully smaller
        return None
    return round(cand, 3)


def _service_meta(cfg, model_path=None, served_name=None) -> dict:
    _resolved, svc_key = resolve_service_key(cfg, model_path)
    with _lock:
        svc = _services.get(svc_key) or {}
    return svc.get("meta") or {}


def ensure_service(cfg, model_path: Optional[str] = None,
                   served_name: Optional[str] = None,
                   ready_timeout: float = 3600.0,
                   gpu_memory_utilization: Optional[float] = None,
                   max_model_len: Optional[int] = None,
                   max_num_batched_tokens: Optional[int] = None,
                   max_num_seqs: Optional[int] = None,
                   target_kv_gb: Optional[float] = None) -> str:
    """Launch (or reuse) a vLLM OpenAI server for a model; return its base URL.

    The configured share is a STARTING POINT, not a verdict. An engine's real footprint
    is not knowable up front — it depends on the model, the card, the context and the
    multimodal limits it profiles for (surya-ocr-2 on a 24 GB 3090 needs ~10.9 GiB before
    a single KV block, almost all of it the vision encoder's profiling peak; another model
    or card is another number). So when vLLM reports the budget cannot hold the cache,
    this escalates the share and tries again, asks the evictor for the larger figure
    first, and REMEMBERS what worked, keyed by model+card+limits. That way any model,
    on any card, serving any kind of request, converges on a share that fits instead of
    depending on a constant somebody measured once.

    ``model_path``/``served_name`` override the config (used by the OCR subsystem to serve
    surya-2 on its own vLLM instance alongside any LLM instance)."""
    configured = resolve_gmu(cfg, gpu_memory_utilization)
    ctx = int(max_model_len or 0) or int(getattr(cfg, "ctx", 0) or 0)
    key = _learned_gmu_key(cfg, (served_name or model_path or "vllm"), ctx,
                           max_num_batched_tokens)
    rec = _load_learned(key)
    # The caller's figure wins (the OCR subsystem passes its own), then this
    # model's `vllm` block, then the global default.
    if target_kv_gb is not None:
        target_kv = float(target_kv_gb)
    else:
        try:
            target_kv = float(getattr(cfg, "kv_target_gb", 0) or 0) or _TARGET_KV_GB
        except (TypeError, ValueError):
            target_kv = _TARGET_KV_GB
    try:
        kv_conc = float(getattr(cfg, "kv_min_concurrency", _KV_MIN_CONCURRENCY))
    except (TypeError, ValueError):
        kv_conc = _KV_MIN_CONCURRENCY

    if configured <= 0 and not rec:
        # No share configured anywhere: nothing to size, let vLLM use its own default.
        return _ensure_service_once(
            cfg, model_path, served_name, ready_timeout, 0.0,
            max_model_len, max_num_batched_tokens, max_num_seqs)

    gmu = _starting_gmu(configured, rec)
    if rec and abs(gmu - configured) > 0.005:
        print(f"[vllm] starting at {gmu:.3f} rather than the configured {configured:.3f} "
              f"— measured previously for {key}: {rec}", flush=True)
    last = None
    for attempt in range(1, _GMU_MAX_ATTEMPTS + 1):
        try:
            url = _ensure_service_once(
                cfg, model_path, served_name, ready_timeout, gmu,
                max_model_len, max_num_batched_tokens, max_num_seqs)
        except RuntimeError as exc:
            last = exc
            if not _KV_EXHAUSTED.search(str(exc)):
                raise
            _save_learned(key, failed=gmu)
            nxt = min(gmu + _GMU_ESCALATION_STEP, GMU_CEILING)
            if gmu >= GMU_CEILING - 1e-6 or nxt <= gmu + 1e-6:
                raise
            print(f"[vllm] {served_name or model_path}: a {gmu:.3f} budget cannot hold "
                  f"the KV cache (attempt {attempt}/{_GMU_MAX_ATTEMPTS}); retrying at "
                  f"{nxt:.3f} — the engine's footprint is larger than that share, which "
                  f"is what this measures", flush=True)
            # The bigger budget needs the room to exist before vLLM asks for it.
            try:
                from codai.models.manager import multi_model_manager
                multi_model_manager._evict_models_for_vram(
                    float(prelaunch_free_gb(cfg, nxt)))
            except Exception as e:
                print(f"[vllm] evict-before-retry skipped: {e}", flush=True)
            gmu = nxt
            continue
        # Booted. Record it, and see whether the cache it got says a smaller share would
        # have done — escalation alone never gives memory back.
        _save_learned(key, ok=gmu)
        try:
            _meta = _service_meta(cfg, model_path, served_name) or {}
            kv = _meta.get("kv_gib")
            # The floor the window imposes. Without it the learner hands back
            # memory the engine needs to serve a single request of the size it
            # was configured for, and the NEXT start of this model is the one
            # that fails.
            floor_kv = _kv_floor_gb(ctx, kv, _meta.get("kv_tokens"), target_kv,
                                    kv_conc)
            if floor_kv > target_kv:
                print(f"[vllm] {served_name or model_path}: keeping at least "
                      f"{floor_kv:.2f} GiB of cache — {kv_conc:g}x a "
                      f"{ctx}-token window at the measured "
                      f"{(kv / _meta['kv_tokens'] * 1024 * 1024):.0f} KiB/token",
                      flush=True)
            cand = _shrink_candidate(gmu, kv, floor_kv, _load_learned(key))
            if cand is not None:
                print(f"[vllm] {served_name or model_path}: {kv:.2f} GiB of cache on a "
                      f"{gmu:.3f} share leaves room to spare; {cand:.3f} would still "
                      f"give ~{floor_kv:.1f} GiB — using it from the next start",
                      flush=True)
                _save_learned(key, ok=cand)
        except Exception as e:
            print(f"[vllm] could not size down from the reported cache: {e}", flush=True)
        return url
    raise last if last is not None else RuntimeError("vLLM did not start")


def _stop_proc(svc: dict, svc_key: str = "") -> None:
    """Tear a service's process (and its ray cluster) down.

    Separate from :func:`stop_service` because it takes no lock: a caller that
    already holds ``_lock`` — replacing an engine in place — would deadlock on
    it.
    """
    if not svc:
        return
    proc = svc["proc"]
    if proc.poll() is None:
        try:
            proc.terminate(); proc.wait(timeout=10)
        except Exception:
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass
    ray = svc.get("ray")
    if ray is not None:
        try:
            ray.stop()
        except Exception as exc:
            print(f"[vllm] ray teardown for {svc_key} failed: {exc}", flush=True)
    print(f"[vllm] service for {svc_key or svc.get('url')} stopped", flush=True)


def stop_service(svc_key: str) -> None:
    with _lock:
        svc = _services.pop(svc_key, None)
    _stop_proc(svc, svc_key)


def stop_all() -> None:
    for k in list(_services.keys()):
        stop_service(k)


def is_running(cfg, model_path: Optional[str] = None, served_name: Optional[str] = None) -> bool:
    """True if a vLLM service for this (model_path, served_name) is currently up."""
    _resolved, svc_key = resolve_service_key(cfg, model_path)
    with _lock:
        svc = _services.get(svc_key)
        return bool(svc and svc["proc"].poll() is None)


# vLLM must not be told to budget the whole card: the driver context, fragmentation and
# any allocation made between our measurement and its own need somewhere to live.
GMU_CEILING = 0.92


def _used_by_others_gb() -> float:
    """VRAM (GB) currently held on this box by anything that is NOT the vLLM we are
    about to launch — resident embedders, an LLM, another process' context."""
    try:
        from codai.models.manager import multi_model_manager
        total = float(multi_model_manager._total_vram_gb() or 0.0)
        free = float(multi_model_manager._get_free_vram_gb() or 0.0)
        if total <= 0 or free >= 999.0:   # 999 = "unknown, assume enough"
            return 0.0
        return max(0.0, total - free)
    except Exception:
        return 0.0


def resolve_gmu(cfg, gpu_memory_utilization: Optional[float] = None) -> float:
    """The absolute ``--gpu-memory-utilization`` to launch with.

    A side job (an OCR VLM engine) passes the share it wants for ITSELF; the backend's
    own configured value is used as-is. Both are fractions of the card's TOTAL memory,
    which is what the flag means — see :func:`effective_gmu` for why nothing is added
    back for other residents any more."""
    share = float(gpu_memory_utilization or 0)
    if share > 0:
        return effective_gmu(share)
    return float(getattr(cfg, "gpu_memory_utilization", 0) or 0)


def effective_gmu(self_share: float) -> float:
    """The share this instance may use, clamped to :data:`GMU_CEILING`, plus a log line
    naming what else is on the card.

    Measured on a 24 GB 3090 (2026-10-05, two live boots of surya-ocr-2): budget
    9.57 GiB gave ``Available KV cache memory: -1.54 GiB``, budget 14.31 GiB gave
    ``+3.18 GiB`` — a fixed footprint of 11.11 and 11.13 GiB respectively. Two things
    follow. First, vLLM charges only its OWN allocations against ``total × gmu``; if
    other residents counted too, those two boots would imply wildly different activation
    peaks (8.3 vs 3.6 GiB) instead of the same 11.1. So this function does NOT add back
    what others hold: doing so made the budget depend on unrelated residents and asked
    for ``share×total + used`` for this instance while ``used`` stayed held by someone
    else. The card is cleared by the caller's evict-first pass
    (:func:`prelaunch_free_gb`) instead, which is the honest mechanism.

    Second, the share has to actually cover the footprint: 0.35 of this card is 8.4 GiB
    against an 11.1 GiB floor, so it could never start, which is why ``0.35`` crash-looped
    for hours and then appeared to "work" only when 6.1 GB of other residents inflated
    the old add-back to 0.604. Hence the 0.60 default and the batched-token cap that
    brings the 9.4 GiB activation peak down."""
    try:
        share = max(0.0, float(self_share or 0.0))
    except Exception:
        return 0.0
    if share <= 0:
        return 0.0
    eff = min(share, GMU_CEILING)
    try:
        from codai.models.manager import multi_model_manager
        total = float(multi_model_manager._total_vram_gb() or 0.0)
    except Exception:
        total = 0.0
    if total > 0:
        used = _used_by_others_gb()
        print(f"[vllm] gpu-memory-utilization {eff:.3f} = {eff * total:.1f} GB of a "
              f"{total:.1f} GB card for this instance; {used:.1f} GB currently held by "
              f"others (the evict-first pass should have cleared what it needed)",
              flush=True)
    return eff


def prelaunch_free_gb(cfg, gpu_memory_utilization: Optional[float] = None) -> float:
    """How much VRAM must be FREE before launching this vLLM for the side-job share to
    fit under :data:`GMU_CEILING` — i.e. what to ask the evictor for.

    It is the engine's own need plus the headroom the ceiling reserves, so a crowded
    card gets models evicted instead of :func:`effective_gmu` clamping and vLLM dying
    on KV cache again."""
    gmu = float(gpu_memory_utilization or 0) or float(getattr(cfg, "gpu_memory_utilization", 0) or 0)
    if gmu <= 0:
        return planned_vram_gb(cfg, gpu_memory_utilization)
    try:
        from codai.models.manager import multi_model_manager
        total = float(multi_model_manager._total_vram_gb() or 24.0)
    except Exception:
        total = 24.0
    total = total or 24.0
    return max(0.5, (min(gmu, GMU_CEILING) + (1.0 - GMU_CEILING)) * total)


def planned_vram_gb(cfg, gpu_memory_utilization: Optional[float] = None) -> float:
    """How much VRAM a vLLM instance started from ``cfg`` will claim on this box.

    vLLM allocates ``gpu_memory_utilization`` × the card's TOTAL memory up front and
    refuses to start when that much is not free — so this is also what must be freed
    BEFORE launching it. Callers that boot vLLM for a side job (the OCR VLM engines) use
    it to ask the model manager for room first; without that, Surya-2 crash-looped on a
    3090 with 11 GB free against a 14 GB demand."""
    gmu = float(gpu_memory_utilization or 0) or float(getattr(cfg, "gpu_memory_utilization", 0) or 0)
    if gmu <= 0:
        return 4.6   # small VLM fallback estimate
    try:
        from codai.models.manager import multi_model_manager
        total = multi_model_manager._total_vram_gb()
    except Exception:
        total = 24.0
    return max(0.5, gmu * float(total or 24.0))


def stop_service_for(cfg, model_path: Optional[str] = None,
                     served_name: Optional[str] = None,
                     gpu_memory_utilization: Optional[float] = None) -> float:
    """Stop the vLLM service for ONE specific (model_path, served_name), if running.

    Used by the OCR subsystem as a VRAM releaser: the managed Surya-2 vLLM subprocess
    isn't a manager-tracked model, so this is what lets on-request eviction reclaim its
    VRAM. Returns an estimate of the GB it was holding (gpu_memory_utilization × total
    VRAM), 0.0 if nothing was running."""
    _resolved, svc_key = resolve_service_key(cfg, model_path)
    with _lock:
        svc = _services.get(svc_key)
        running = bool(svc and svc["proc"].poll() is None)
        launched_gmu = float((svc or {}).get("gmu") or 0.0)
    if not running:
        return 0.0
    stop_service(svc_key)
    # What it reserved is what we just freed. The share the caller passes is the engine's
    # OWN share; the instance was launched with that plus whatever else held the card
    # (effective_gmu), so reporting the caller's figure would under-count the release and
    # send the eviction path looking for memory that is already free.
    if launched_gmu > 0:
        try:
            from codai.models.manager import multi_model_manager
            total = float(multi_model_manager._total_vram_gb() or 24.0)
        except Exception:
            total = 24.0
        return max(0.5, launched_gmu * (total or 24.0))
    return planned_vram_gb(cfg, gpu_memory_utilization)


import atexit as _atexit
_atexit.register(stop_all)
