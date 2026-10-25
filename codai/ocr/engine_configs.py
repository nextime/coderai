# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Several published ids for one OCR engine.

Image models have had this for a long time: one checkpoint on disk, several
entries in the catalogue, each with its own settings, its own defaults and its
own line in ``/v1/models`` -- and one copy of the weights in VRAM. OCR engines
never did. They are a fixed registry of five names, each a singleton keyed by
that name, reading its settings out of flat ``ocr.<engine>_*`` fields. So there
was exactly one way to run surya on an install, and a caller who wanted the
same engine with layout on and another with it off had no way to ask.

This module is the name layer that fixes it. A configured id resolves to
*(base engine, overrides)*; the overrides are applied to the shared
``OcrConfig`` with ``dataclasses.replace``, and the result is what that id's
engine pool is built with. Engines read every setting from the config they were
given, so none of them needs to know that named configurations exist -- the
same trick ``MultiOcrManager._cfg_for`` already plays with a request's
language.

TWO IDS ARE TWO POOLS. That is deliberate and it is the cost: ids differing in
what the engine LOADS must not share a pool, because an instance is built for
its settings and cannot be re-settinged afterwards. Ids differing only in what
they ask of an already-loaded model still pay for a second pool today, which is
worth knowing before configuring ten of them on one card.

A key in ``overrides`` that is not a real ``OcrConfig`` field is REFUSED. The
alternative is an id that looks configured, serves the base settings, and
differs from its sibling in nothing at all -- which is the same class of
silence as an engine advertising regions it does not return.
"""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple


def configured(cfg) -> Dict[str, dict]:
    """``{id: {"engine": base, "overrides": {...}}}`` from ``ocr.engine_configs``.

    Malformed entries are skipped rather than fatal: a bad line in the list
    must not take the whole OCR subsystem down, and the id simply does not
    resolve, which is reported where it is asked for.
    """
    out: Dict[str, dict] = {}
    for raw in (getattr(cfg, "engine_configs", None) or []):
        if not isinstance(raw, dict):
            continue
        cid = str(raw.get("id") or "").strip().lower()
        base = str(raw.get("engine") or "").strip().lower()
        if not cid or not base or cid == base:
            # An id equal to its own engine name is not a second
            # configuration, it is a redefinition of the first.
            continue
        ov = raw.get("overrides")
        out[cid] = {"engine": base,
                    "overrides": dict(ov) if isinstance(ov, dict) else {}}
    return out


def ids_for(cfg, base: str) -> List[str]:
    """The configured ids that serve engine ``base``, in config order."""
    return [cid for cid, spec in configured(cfg).items()
            if spec["engine"] == str(base or "").strip().lower()]


def resolve(cfg, requested: Optional[str]) -> Tuple[str, str, dict]:
    """``(config_id, base_engine, overrides)`` for what the caller asked for.

    A bare engine name, an unknown name, or nothing at all gives
    ``(name, name, {})`` -- this function only answers "is this a named
    configuration", and leaves "is this a usable engine" to the caller that
    already asks it (``resolve_engine``), so an unknown name keeps producing
    the one refusal it always did rather than two different ones.
    """
    name = str(requested or "").strip().lower()
    spec = configured(cfg).get(name)
    if not spec:
        return name, name, {}
    return name, spec["engine"], dict(spec["overrides"])


def effective(cfg, overrides: Optional[dict]):
    """``cfg`` with this configuration's overrides applied.

    Raises ValueError naming the field when an override is not a real config
    field, since that is a configuration mistake and the only way to see it is
    to be told.
    """
    if not overrides:
        return cfg
    try:
        fields = {f.name for f in dataclasses.fields(cfg)}
    except Exception:
        # Not a dataclass: nothing can be replaced, and pretending the
        # overrides applied would serve the base settings under another name.
        raise ValueError("this OCR config cannot carry per-id overrides")
    unknown = sorted(k for k in overrides if k not in fields)
    if unknown:
        raise ValueError(
            "unknown OCR setting(s) in engine_configs overrides: "
            + ", ".join(unknown))
    return dataclasses.replace(cfg, **dict(overrides))


def describe(cfg, cid: str) -> str:
    """``"surya (surya_layout=True)"`` — for a log line or a catalogue row."""
    _cid, base, ov = resolve(cfg, cid)
    if not ov:
        return base
    inner = ", ".join(f"{k}={v!r}" for k, v in sorted(ov.items()))
    return f"{base} ({inner})"
