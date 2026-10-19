"""The endpoints that say what the server is doing must not need a thread the
request path can take.

Measured on a live orchestrator during a pod boot wave:

    /healthz            200 in   2.9 ms   (no thread needed)
    /v1/models          200 in   6   ms   (no thread needed)
    /metrics            200 in 180   ms   (no thread needed)
    /v1/runpod/status   no answer in 25 s (needs a thread)
    /v1/runpod/scale    no answer in 25 s (needs a thread)

Nothing was broken. The two endpoints that stopped answering are the only two
that hand work to a thread, and they drew that thread from the DEFAULT
executor -- the pool the request path also draws on. A client reasonably read
25 s of silence as a dead server; worse, it meant a client lease could not be
RELEASED while the fleet was busy, so pods kept billing with no way to stop
them short of the RunPod console.

Three properties, each the fix for one way that failed:

  1. the control plane has its OWN pool, which nothing in the request path uses
  2. the pdfium split no longer occupies the general pool, where hundreds of
     documents each took a thread and then queued on one global lock anyway
  3. every source in the fleet snapshot is bounded, so the endpoint answers
     late-but-partial instead of not at all
"""
import inspect

import pytest

from codai.api import thread_pool as tp


# --- 1. a reserved pool -----------------------------------------------------

def test_the_control_pool_exists_and_is_separate():
    ctrl = tp.control_executor()
    assert ctrl is not tp.pod_executor()
    assert ctrl is not tp.lease_executor()
    assert ctrl is not tp.pdfium_executor()


def test_the_control_pool_is_built_once():
    assert tp.control_executor() is tp.control_executor()


def test_the_control_pool_is_small_and_fixed():
    """Its value is being empty when everything else is full, so it must not
    be sized from anything that grows with load."""
    src = inspect.getsource(tp.control_executor)
    assert "_CONTROL_WORKERS" in src
    assert "_configured_concurrency" not in src
    assert "task_budget" not in src
    assert 0 < tp._CONTROL_WORKERS <= 64


def test_the_fixed_pools_are_paid_for_out_of_the_budget():
    """A reserved pool that was never subtracted would be reserved from
    threads the sized pools had already been promised."""
    src = inspect.getsource(tp.pool_sizes)
    assert "_CONTROL_WORKERS" in src and "_PDFIUM_WORKERS" in src


def test_the_control_plane_endpoints_use_it():
    """All three: status, scale and release. A teardown that cannot get a
    thread is a pod that cannot be stopped."""
    src = open("codai/api/app.py").read()
    scale = src[src.index("async def runpod_scale("):]
    scale = scale[:scale.index("@app.get")]
    assert "run_on_control_pool(" in scale and "set_client_pods" in scale
    assert "asyncio.to_thread" not in scale
    assert "run_on_control_pool(runpod_worker.release_pod" in src
    assert "run_on_control_pool(fn)" in src
    body = src[src.index("async def runpod_spend"):src.index("async def runpod_status")]
    assert "asyncio.to_thread" not in body, \
        "the snapshot must not draw from the pool the request path uses"


def test_the_admin_stats_page_uses_it_too():
    src = open("codai/admin/routes.py").read()
    body = src[src.index("async def api_runpod_stats"):]
    body = body[:body.index("@router.get")]
    assert "run_on_control_pool" in body
    assert "asyncio.gather" in body, "and does not run its three sources inline"


def test_nothing_in_the_request_path_uses_the_control_pool():
    """The reservation is the whole mechanism. One page forward on this pool
    and a busy server is silent again."""
    import subprocess
    out = subprocess.run(
        ["grep", "-rn", "run_on_control_pool\\|control_executor",
         "codai/ocr/", "codai/api/text.py", "codai/api/ocr.py",
         "codai/cluster/"],
        capture_output=True, text=True).stdout
    assert out.strip() == "", out


# --- 2. pdfium queues where it is already serialised ------------------------

def test_the_pdfium_pool_is_as_narrow_as_pdfium_is():
    """pdfium admits ONE caller; threads past the second buy no parallelism,
    they only decide which pool the queue forms in."""
    assert 1 <= tp._PDFIUM_WORKERS <= 4


def test_the_document_split_runs_on_it():
    src = open("codai/ocr/manager.py").read()
    assert "run_on_pdfium_pool(lambda: list(iter_pdf_page_pdfs(data)))" in src
    assert "asyncio.to_thread(lambda: list(iter_pdf_page_pdfs(data)))" not in src


def test_pdfium_really_is_single_caller():
    """Guards the reasoning above rather than the code: if the global lock
    ever goes away, a two-wide pool becomes a throughput ceiling."""
    src = open("codai/ocr/base.py").read()
    assert "_PDFIUM_LOCK = threading.RLock()" in src
    split = src[src.index("def iter_pdf_page_pdfs"):]
    split = split[:split.index("def render_pages")]
    assert "with _PDFIUM_LOCK:" in split


# --- 3. the snapshot cannot hang -------------------------------------------

def test_each_source_has_its_own_deadline():
    """One function on one thread meant the slowest source decided the whole
    answer, and there was no bound between them at all."""
    src = open("codai/api/app.py").read()
    assert "SNAPSHOT_PART_TIMEOUT_S" in src
    body = src[src.index("async def runpod_spend"):src.index("async def runpod_status")]
    assert "asyncio.wait_for(" in body
    assert "asyncio.gather(" in body


def test_a_source_that_times_out_is_named_not_hidden():
    """A zero meaning 'nothing is running' and a zero meaning 'we could not
    look' must not read the same: a monitor that cannot tell them apart
    reports an idle fleet during an incident."""
    src = open("codai/api/app.py").read()
    body = src[src.index("async def runpod_spend"):src.index("async def runpod_status")]
    assert 'out["degraded"] = degraded' in body
    assert "did not answer in" in body


def test_the_schedule_snapshot_locks_are_bounded():
    """This was the unbounded half: pods_status() gave up on a busy pool,
    pools_schedule_status() waited for it, and the endpoint hung either way."""
    from codai.api import runpod_worker as rw
    src = inspect.getsource(rw.pools_schedule_status)
    assert "_lock_for_read(_pools_lock)" in src
    assert "_lock_for_read(pool._cv)" in src
    assert "with _pools_lock:" not in src
    assert "with pool._cv:" not in src


def test_a_pool_that_cannot_be_read_says_so():
    """Same shape as pods_status: a row naming the pool, not a missing pool."""
    from codai.api import runpod_worker as rw
    src = inspect.getsource(rw.pools_schedule_status)
    assert '"busy": True' in src
    assert "was locked when this" in src


def test_the_ledger_snapshot_does_not_wait_forever():
    """totals() shared one lock with record_spend, which WRITES the file, and
    terminations arrive in bursts during exactly the wave being polled."""
    from codai.api import runpod_ledger as rl
    src = inspect.getsource(rl.totals)
    assert "_lock.acquire(timeout=" in src
    assert "degraded" in src


def test_enforcement_still_takes_the_ledger_lock_properly():
    """A decision made from data that may be mid-write is worse than one made
    late. Only the SNAPSHOT may degrade."""
    from codai.api import runpod_ledger as rl
    for fn in (rl.spend_in_period, rl.global_spend_in_period, rl.record_spend):
        src = inspect.getsource(fn)
        assert "with _lock:" in src, fn.__name__
        assert "timeout" not in src, fn.__name__
