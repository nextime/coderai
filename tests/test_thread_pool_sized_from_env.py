"""A rented pod sizes its thread pools from its ENVIRONMENT.

A pod has no config.json; its settings arrive as env vars, and the thread
pools are built before the config manager has read them. So on every pod
`_configured_concurrency()` returned 0 and the default pool took its floor of
32 -- and every page on a pod runs in that pool. Measured 2026-10-19: the
orchestrator dispatched 64 pages per pod, every card reported `Running: 32`,
GPU KV cache sat at 4-10%, surya was ready for 512 and vLLM for 72x.
"""
import pytest

from codai.api import thread_pool as tp


def _no_config(monkeypatch):
    """Make "no config is readable" true, whatever an earlier test left set."""
    import importlib
    for mod in ("codai.admin.routes", "codai.config"):
        try:
            m = importlib.import_module(mod)
            monkeypatch.setattr(m, "config_manager", None, raising=False)
        except Exception:
            pass


def test_the_env_sizes_the_pools_when_no_config_is_readable(monkeypatch):
    _no_config(monkeypatch)
    monkeypatch.setenv("CODERAI_OCR_MAX_CONCURRENCY", "4096")
    monkeypatch.setattr(tp, "task_budget", lambda: 0)
    assert tp._configured_concurrency() >= 4096
    general, pod = tp.pool_sizes()
    assert general > 32, "the floor is what capped every pod at 32 pages"


def test_no_env_and_no_config_still_takes_the_floor(monkeypatch):
    monkeypatch.delenv("CODERAI_OCR_MAX_CONCURRENCY", raising=False)
    import importlib
    for mod in ("codai.admin.routes", "codai.config"):
        try:
            m = importlib.import_module(mod)
            monkeypatch.setattr(m, "config_manager", None, raising=False)
        except Exception:
            pass
    assert tp._configured_concurrency() == 0
    assert tp.pool_sizes()[0] == tp._MIN_DEFAULT


def test_a_malformed_env_does_not_break_sizing(monkeypatch):
    monkeypatch.setenv("CODERAI_OCR_MAX_CONCURRENCY", "lots")
    import importlib
    for mod in ("codai.admin.routes", "codai.config"):
        try:
            m = importlib.import_module(mod)
            monkeypatch.setattr(m, "config_manager", None, raising=False)
        except Exception:
            pass
    assert tp._configured_concurrency() == 0
