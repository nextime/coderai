"""One pool, pods with different bearers, and a token that survives a restart.

2026-10-20, qwen38: a restart generated a FRESH random pool token (none is
configured), adopted a pod still carrying the old one, and rented a second pod
with the new one. The pool has one `api_key`, so whichever token it held 401'd
the other pod -- "Invalid API key" on every request until both were destroyed.

Two causes, two fixes:
  * the token belongs to the POD (PodHandle.api_key), handed out with the
    address, so a pool can hold pods launched with different ones;
  * the pool's own token is PERSISTED, so a restart does not invent one and
    orphan every pod it already rented.
"""
import json
import types

import pytest

from codai.api import runpod_worker as rw


# ------------------------------------------------- the key travels with the pod #

def _pool():
    p = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    p.api_key = "pool-key"
    p._pod_keys = {}
    p.pods = []
    p._cv = __import__("threading").Condition(__import__("threading").RLock())
    return p


def _handle(pid, key=""):
    return rw.PodHandle(pod_id=pid, url=f"https://{pid}:1/v1", hourly_usd=0.59,
                        started_at=0.0, api_key=key)


def test_a_pod_with_its_own_token_keeps_it():
    p = _pool()
    assert p.pod_key(_handle("adopted", "its-own-key")) == "its-own-key"


def test_a_pod_without_one_uses_the_pools():
    p = _pool()
    assert p.pod_key(_handle("rented")) == "pool-key"


def test_two_pods_in_one_pool_can_hold_different_tokens():
    """THE BUG: one pool-wide key 401s whichever pod does not share it."""
    p = _pool()
    assert p.pod_key(_handle("a", "key-a")) == "key-a"
    assert p.pod_key(_handle("b", "key-b")) == "key-b"


def test_a_pod_that_comes_back_is_still_reachable():
    """Quarantine and re-adoption drop and re-add the handle; the token must
    not be lost with it."""
    p = _pool()
    p._remember_pod_key("x1", "remembered")
    assert p.pod_key(_handle("x1")) == "remembered"


def test_no_pod_means_the_pools_own_key():
    assert _pool().pod_key(None) == "pool-key"


def test_the_remembered_keys_are_bounded():
    p = _pool()
    for i in range(700):
        p._remember_pod_key(f"p{i}", f"k{i}")
    assert len(p._pod_keys) <= 512


# ------------------------------------------------------------ the request path #

def test_the_backend_sends_the_acquired_pods_key():
    from codai.backends.runpod import RunpodBackend
    b = RunpodBackend.__new__(RunpodBackend)
    b._pool = _pool()
    b._headers = {}
    # The bearer the acquire returned wins over anything read from the pool.
    assert b._auth("acquired-key") == {"Authorization": "Bearer acquired-key"}


def test_without_one_it_falls_back_to_the_last_pod_then_the_pool():
    from codai.backends.runpod import RunpodBackend
    b = RunpodBackend.__new__(RunpodBackend)
    b._pool = _pool()
    b._headers = {}
    b._last_pod = _handle("p9", "last-pod-key")
    assert b._auth() == {"Authorization": "Bearer last-pod-key"}
    b._last_pod = None
    assert b._auth() == {"Authorization": "Bearer pool-key"}


def test_the_key_is_handed_out_with_the_address_not_read_after():
    """`_last_pod` is ONE attribute shared by every concurrent request, so a
    key read after the acquire can belong to another request's pod."""
    import inspect
    from codai.backends.runpod import RunpodBackend
    src = inspect.getsource(RunpodBackend._acquire_endpoint)
    assert "self._pool_key(pod)" in src
    for fn in (RunpodBackend.generate_chat, RunpodBackend.generate_chat_stream):
        s = inspect.getsource(fn)
        assert "base, release, bearer" in s
        assert "bearer" in s.split("_acquire_endpoint")[1]
    from codai.models import placement
    s = inspect.getsource(placement._pod_candidate)
    assert "base, release, bearer = backend._acquire_endpoint()" in s
    assert "backend._auth(bearer)" in s


def test_serverless_has_no_pod_key():
    from codai.backends.runpod import RunpodBackend
    b = RunpodBackend.__new__(RunpodBackend)
    b._mode = "serverless"
    b._base = lambda: "https://x/openai/v1"
    base, release, bearer = b._acquire_endpoint()
    assert (base, bearer) == ("https://x/openai/v1", "")
    release()


# ------------------------------------------------- the pool token is persisted #

def test_the_token_is_minted_once_and_reused(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rw, "_pod_registry_path", lambda: str(tmp_path / "pods.json"))
    first = rw._pool_token("datalab-to/surya-ocr-2")
    assert first.startswith("cra-")
    assert "minted" in capsys.readouterr().out
    # A RESTART: same file, same token, and nothing minted.
    assert rw._pool_token("datalab-to/surya-ocr-2") == first
    assert "minted" not in capsys.readouterr().out


def test_each_pool_gets_its_own(monkeypatch, tmp_path):
    monkeypatch.setattr(rw, "_pod_registry_path", lambda: str(tmp_path / "pods.json"))
    assert rw._pool_token("a") != rw._pool_token("b")


def test_it_is_not_world_readable(monkeypatch, tmp_path):
    monkeypatch.setattr(rw, "_pod_registry_path", lambda: str(tmp_path / "pods.json"))
    rw._pool_token("a")
    path = tmp_path / rw._POOL_TOKENS_FILE
    assert path.exists()
    assert oct(path.stat().st_mode)[-3:] == "600"


def test_an_unwritable_dir_still_serves(monkeypatch, tmp_path, capsys):
    """Serving with a per-process token beats not serving."""
    monkeypatch.setattr(rw, "_pod_registry_path", lambda: "/proc/nope/pods.json")
    tok = rw._pool_token("a")
    assert tok.startswith("cra-")
    assert "could not save" in capsys.readouterr().out


def test_a_configured_key_is_never_overridden(monkeypatch, tmp_path):
    import inspect
    src = inspect.getsource(rw.RunpodPodPool.__init__)
    i = src.index('self.api_key = (getattr(mcfg, "api_key"')
    blk = src[i:i + 900]
    assert "if not self.api_key" in blk
    assert "_pool_token(" in blk
    assert "secrets.token_urlsafe" not in blk       # no per-process token


# --------------------------------- a restart renting beside its own boot #
#
# Reverted, deliberately. Treating an orphaned mid-boot pod as "on the way"
# stops a restart renting a second pod -- and reintroduces the freeze of
# test_runpod_restart_orphans (a dead owner's url-less marker whose pod never
# came up blocked renting for twenty minutes). Renting twice costs cents; a
# pool that will not rent serves nothing. The surplus belongs to scale-down.


def test_waiting_on_a_dead_owners_boot_stays_out(monkeypatch, tmp_path):
    now = __import__("time").time()
    path = tmp_path / "runpod-pods.json"
    path.write_text(json.dumps({"booting1": {
        "pool": "m", "proc": "gone", "url": "", "at": now - 300,
        "created": now - 300}}))
    monkeypatch.setattr(rw, "_pod_registry_path", lambda: str(path))
    assert rw.sibling_is_provisioning("m") is False
