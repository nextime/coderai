"""A pod's log has to outlive the pod.

A rented pod is the one machine in the system whose logs cannot be read from
outside: RunPod publishes no pod-log API (the REST spec lists 23 routes and none
of them serve logs, which is why every attempt returns HTTP 400), and `boot.sh`
writes its phases to a file and then execs uvicorn, whose output goes to the
container's stdout and nowhere else. So `/boot` covered the way up and nothing
after it.

That cost an evening. Three pods were investigated in one session and two were
already destroyed by the time there was a question to ask — one reclaimed by
RunPod mid-request, taking the only evidence with it — and "the serving hangs"
stayed a hypothesis for an hour because the machine that knew was first
unreachable and then absent. (It was not hanging: a page took 807s, mostly
weights download, and every client had given up at 180-300s.)

So the pod keeps a bounded ring of its own output and serves it with a cursor,
and the renting install drains that into a file it keeps AFTER the pod dies.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import podlog                                      # noqa: E402
from codai.api import runpod_worker as rw                         # noqa: E402


def _fresh():
    podlog._lines.clear()
    podlog._seq = 0


# ----------------------------------------------------------------- the ring #
def test_lines_come_back_with_a_cursor():
    _fresh()
    podlog.record("boot +0.1s container started")
    podlog.record("vllm: loading weights")
    out = podlog.tail()
    assert out["lines"] == ["boot +0.1s container started", "vllm: loading weights"]
    assert out["cursor"] == 2


def test_the_cursor_only_returns_what_is_new():
    """The drain runs every 15s; re-sending everything each time would make the
    kept file mostly duplicates."""
    _fresh()
    podlog.record("one")
    first = podlog.tail()
    podlog.record("two")
    second = podlog.tail(since=first["cursor"])
    assert second["lines"] == ["two"]


def test_nothing_new_is_an_empty_answer_not_a_repeat():
    _fresh()
    podlog.record("one")
    c = podlog.tail()["cursor"]
    assert podlog.tail(since=c)["lines"] == []


def test_the_ring_is_bounded():
    """A pod's disk is rented too, and this must never be why one falls over."""
    _fresh()
    assert podlog._lines.maxlen == podlog.MAX_LINES
    for i in range(podlog.MAX_LINES + 50):
        podlog.record(f"line {i}")
    assert len(podlog._lines) == podlog.MAX_LINES


def test_eviction_is_reported_rather_than_hidden():
    """Silence and "nothing happened" must not look the same: a caller that
    fell behind needs to know its gap is a gap."""
    _fresh()
    podlog._lines.clear()
    for i in range(5):
        podlog.record(f"line {i}")
    # Still holding everything: a caller at cursor 1 has missed nothing.
    assert podlog.tail(since=1)["dropped"] == 0
    # Now roll lines 2 and 3 out from under that caller: a real gap.
    podlog._lines.popleft()                 # seq 1, already seen
    podlog._lines.popleft()                 # seq 2, never seen
    podlog._lines.popleft()                 # seq 3, never seen
    out = podlog.tail(since=1)
    assert out["dropped"] == 2
    assert out["lines"] == ["line 3", "line 4"]


def test_blank_lines_are_not_recorded():
    _fresh()
    podlog.record("")
    podlog.record("\n")
    assert podlog.tail()["lines"] == []


# ------------------------------------------------- capturing without breaking #
def test_the_tee_writes_through_to_the_real_stream():
    """Printing is how everything else in coderai reports itself. A logging aid
    that can break it would be worse than none."""
    import io as _io
    _fresh()
    real = _io.StringIO()
    tee = podlog._Tee(real)
    tee.write("hello\n")
    assert real.getvalue() == "hello\n"
    assert podlog.tail()["lines"] == ["hello"]


def test_a_broken_real_stream_does_not_raise():
    _fresh()

    class _Broken:
        def write(self, s):
            raise OSError("stream closed")

        def flush(self):
            raise OSError("stream closed")
    tee = podlog._Tee(_Broken())
    tee.write("still recorded\n")            # must not raise
    tee.flush()
    assert podlog.tail()["lines"] == ["still recorded"]


def test_a_writer_that_never_sends_a_newline_is_still_captured():
    import io as _io
    _fresh()
    tee = podlog._Tee(_io.StringIO())
    tee.write("x" * 9000)
    assert podlog.tail()["lines"]


def test_install_is_idempotent():
    assert podlog.install() in (True, False)
    before = podlog._installed
    podlog.install()
    assert podlog._installed == before


# ------------------------------------------------------------- the drain side #
class _Reply:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p


def _pool(tmp_path, monkeypatch, reply):
    import threading
    pool = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    pool._cv = threading.Condition()
    pool.model_key = "m"
    pool.api_key = "tok"
    pool.pods = []
    monkeypatch.setattr(rw, "_pod_registry_path",
                        lambda: str(tmp_path / "runpod-pods.json"))
    seen = {}

    def _get(url, headers=None, timeout=None, **kw):
        seen["url"] = url
        seen["headers"] = headers or {}
        return reply
    from codai.api import pod_http
    monkeypatch.setattr(pod_http, "get", _get)
    return pool, seen


def _pod(pid="p1", url="https://10.0.0.2:8000", cursor=0):
    return rw.PodHandle(pod_id=pid, url=url, hourly_usd=0.6, started_at=0.0,
                        log_cursor=cursor)


def test_the_drain_writes_a_file_named_for_the_pod(tmp_path, monkeypatch):
    pool, _ = _pool(tmp_path, monkeypatch,
                    _Reply({"cursor": 2, "lines": ["a", "b"]}))
    pod = _pod()
    pool._drain_pod_log(pod)
    kept = pathlib.Path(pool._pod_log_path("p1"))
    assert kept.read_text().splitlines() == ["a", "b"]


def test_it_asks_only_for_what_it_has_not_seen(tmp_path, monkeypatch):
    pool, seen = _pool(tmp_path, monkeypatch, _Reply({"cursor": 9, "lines": []}))
    pool._drain_pod_log(_pod(cursor=7))
    assert "since=7" in seen["url"]


def test_the_cursor_advances_so_the_file_does_not_repeat(tmp_path, monkeypatch):
    pool, _ = _pool(tmp_path, monkeypatch,
                    _Reply({"cursor": 5, "lines": ["x"]}))
    pod = _pod()
    pool._drain_pod_log(pod)
    assert pod.log_cursor == 5


def test_the_pods_bearer_token_is_sent(tmp_path, monkeypatch):
    """A pod is bearer-locked; an unauthenticated drain would get nothing."""
    pool, seen = _pool(tmp_path, monkeypatch, _Reply({"cursor": 1, "lines": ["a"]}))
    pool._drain_pod_log(_pod())
    assert seen["headers"]["Authorization"] == "Bearer tok"


def test_an_older_image_without_logs_is_not_an_error(tmp_path, monkeypatch):
    """A pod built before this exists answers 404; the drain must shrug."""
    pool, _ = _pool(tmp_path, monkeypatch, _Reply({}, status=404))
    pod = _pod()
    pool._drain_pod_log(pod)                 # must not raise
    assert not pathlib.Path(pool._pod_log_path("p1")).exists()


def test_an_unreachable_pod_is_not_the_drains_problem(tmp_path, monkeypatch):
    from codai.api import pod_http
    pool, _ = _pool(tmp_path, monkeypatch, _Reply({}))

    def _boom(*a, **kw):
        raise OSError("connection refused")
    monkeypatch.setattr(pod_http, "get", _boom)
    pool._drain_pod_log(_pod())              # the health machinery owns that


def test_a_gap_in_the_pods_ring_is_written_into_the_file(tmp_path, monkeypatch):
    pool, _ = _pool(tmp_path, monkeypatch,
                    _Reply({"cursor": 99, "dropped": 7, "lines": ["after"]}))
    pool._drain_pod_log(_pod(cursor=1))
    text = pathlib.Path(pool._pod_log_path("p1")).read_text()
    assert "7 line(s) rolled out" in text and "after" in text


def test_the_kept_file_is_trimmed_rather_than_growing_for_ever(tmp_path,
                                                               monkeypatch):
    pool, _ = _pool(tmp_path, monkeypatch, _Reply({"cursor": 1, "lines": ["z"]}))
    monkeypatch.setattr(rw, "_POD_LOG_MAX_BYTES", 100)
    monkeypatch.setattr(rw, "_POD_LOG_KEEP_LINES", 3)
    path = pathlib.Path(pool._pod_log_path("p1"))
    path.write_text("old\n" * 200)
    pool._drain_pod_log(_pod())
    assert len(path.read_text().splitlines()) <= 4


def test_the_file_survives_the_pod(tmp_path, monkeypatch):
    """The whole point. A pod reclaimed mid-request took its evidence with it;
    the file is what is left to read afterwards."""
    pool, _ = _pool(tmp_path, monkeypatch,
                    _Reply({"cursor": 3, "lines": ["last words"]}))
    pod = _pod()
    pool.pods.append(pod)
    pool._drain_pod_log(pod)
    pool.pods.clear()                        # pod gone
    assert "last words" in pathlib.Path(pool._pod_log_path("p1")).read_text()


def test_maintenance_drains_before_it_tears_anything_down():
    """The interesting lines are the last ones, and they are only readable while
    the pod still exists."""
    import inspect
    src = inspect.getsource(rw.RunpodPodPool.maintain)
    assert "_drain_pod_log" in src
    assert src.index("_drain_pod_log") < src.index("_scale_down")
