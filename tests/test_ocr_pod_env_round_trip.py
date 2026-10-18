"""Every OCR setting a pod is SENT must be one the pod READS.

A rented pod has no config.json and no models.json: environment is the only
channel it has. So a setting that is sent and never read is not a cosmetic gap,
it silently reverts that pod to a dataclass default while the admin UI shows
the operator's value.

Measured on one A40, 2026-10-18: the orchestrator sent
CODERAI_OCR_CHANDRA_INSTANCES=128 and nothing read it, so the pod logged

    [ocr] engine 'chandra': 1 instance(s) ready

and served one page at a time -- vLLM reporting `Running: 1, Waiting: 0` at
1.6% KV cache and 55 tokens/s with 32 pages in flight, against surya's ~2000
tokens/s on the same card. It read as "chandra is 36x slower than surya". It
was one page at a time.

The surya equivalents had all been mapped, which is why only surya worked.
These tests are deliberately generic: a per-name assertion list would have
passed before the fix for surya and said nothing about the other two engines.
"""
import re

import pytest

SRC = open("codai/config.py").read()
POD = open("codai/api/runpod_worker.py").read()

#: Settings that are per-engine tuning rather than wiring. If the orchestrator
#: sends one of these for an engine, the pod must read it back.
TUNING = ("INSTANCES", "MAX_TOKENS", "MAX_CONTINUATIONS", "MAX_RETRIES", "DPI")
ENGINES = ("SURYA", "CHANDRA", "OLMOCR")


def read_env_names():
    """Every CODERAI_OCR_* name config.py reads out of the environment."""
    names = set(re.findall(r'"(CODERAI_OCR_[A-Z0-9_]+)"', SRC))
    # also the f-string built ones, e.g. f"CODERAI_OCR_{name.upper()}_ENABLED"
    for suffix in re.findall(r'f"CODERAI_OCR_\{[^}]+\}_([A-Z0-9_]+)"', SRC):
        for eng in ENGINES:
            names.add(f"CODERAI_OCR_{eng}_{suffix}")
    return names


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("knob", TUNING)
def test_each_engine_tuning_knob_is_read(engine, knob):
    """Named with the engine and knob so a failure says which one is missing."""
    name = f"CODERAI_OCR_{engine}_{knob}"
    if name not in POD and name not in SRC:
        pytest.skip(f"{name} is not sent to pods, so it need not be read")
    assert name in read_env_names(), (
        f"{name} is sent to pods but config.py never reads it back; a pod has "
        f"no config file, so that setting silently reverts to its default")


def test_the_regression_itself():
    """CHANDRA_INSTANCES specifically: this is the one that cost a day."""
    assert "CODERAI_OCR_CHANDRA_INSTANCES" in read_env_names()


def test_olmocr_instances_too():
    """The same gap, same cause, for the third engine -- found by looking
    rather than by waiting for it to be measured."""
    assert "CODERAI_OCR_OLMOCR_INSTANCES" in read_env_names()


def test_each_engine_can_have_its_concurrency_set_from_env():
    """The property that actually matters, stated once per engine."""
    names = read_env_names()
    for eng, attr in (("SURYA", "surya_instances"),
                      ("CHANDRA", "chandra_instances"),
                      ("OLMOCR", "olmocr_instances")):
        assert f"CODERAI_OCR_{eng}_INSTANCES" in names, eng
        assert f'"{attr}"' in SRC, f"{attr} must be the attribute it maps to"


def test_mapped_attributes_exist_on_the_config():
    """A row pointing at a misspelled attribute would set nothing, silently."""
    from codai.config import OcrConfig
    cfg = OcrConfig()
    rows = re.findall(r'\("CODERAI_OCR_[A-Z0-9_]+",\s*\n?\s*"([a-z0-9_]+)"', SRC)
    assert rows, "no env rows found — the parse is wrong, not the config"
    missing = [a for a in set(rows) if not hasattr(cfg, a)]
    assert not missing, f"env rows map to attributes that do not exist: {missing}"
