"""Growing a pool must try adoption before renting, like every other path.

Live, 2026-10-13 23:10. A run started with two of this pool's pods already
running (ours, orphaned by a restart). The cold path adopted one — correctly —
and then the pool grew:

  23:10:12 [runpod] adopting pod sncrmkqdnd7c57's bearer token
  23:10:12 [runpod] reusing the pod already running … instead of renting another
  23:10:12 'datalab-to/surya-ocr-2': growing to meet 1 pod(s) of demand
  23:10:13 provisioning pod … (a 30-candidate capacity walk, then an RTX A6000)

…renting a card beside a second, healthy, idle pod it already owned. Adoption
was attempted in three places — `ensure_ready`, `acquire` with nothing healthy,
and the warm floor — and the growth path was not one of them, so growth always
rented. The second pod billed for nothing until the reaper's grace expired.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import runpod_worker as rw                          # noqa: E402


class _Pool:
    """Just the growth entry point, over recorded decisions."""

    model_key = "m"
    _grow_in_background = rw.RunpodPodPool._grow_in_background

    def __init__(self, adopts):
        import threading
        self._cv = threading.Condition()
        self._booting = 1
        self._adopts = adopts
        self.calls = []

    def _adopt_shared_pod(self):
        self.calls.append("adopt")
        return self._adopts

    def _provision_one(self, cancellable=False):
        self.calls.append("rent")


def _grow(adopts):
    import time
    pool = _Pool(adopts)
    pool._grow_in_background()
    for _ in range(200):                     # it runs on its own thread
        if pool.calls and (adopts or "rent" in pool.calls):
            break
        time.sleep(0.01)
    time.sleep(0.05)
    return pool


def test_growth_adopts_a_running_pod_instead_of_renting():
    pool = _grow(True)
    assert pool.calls == ["adopt"], "a card was rented beside a pod we own"


def test_growth_still_rents_when_there_is_nothing_to_adopt():
    pool = _grow(False)
    assert pool.calls == ["adopt", "rent"]


def test_the_booting_count_is_released_either_way():
    # Left standing, it would block every later growth: _should_grow refuses
    # while `_booting >= scale_up_max_parallel_boots`.
    assert _grow(True)._booting == 0
    assert _grow(False)._booting == 0


def test_every_provisioning_path_tries_adoption_first():
    """Pinned against the source: a fourth path added later must not forget.

    `seed_region` is deliberately excluded — it exists to fill a region's volume
    by booting a pod THERE, and adopting a pod elsewhere would not do that.
    """
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    # 5 call sites: ensure_ready, acquire, growth, warm floor, seed_region.
    # Two of the five now pass `cancellable=True` -- the anticipatory
    # boots, which hand their pod back if the fleet is released while
    # they are on the way. The inline one a request is waiting on never
    # is. Counted together: every call site still adopts first.
    assert src.count("self._provision_one(") == 5
    assert src.count("self._provision_one(cancellable=True)") == 2

    # Per PATH rather than a global count of one spelling. The warm floor stopped
    # matching `if not self._adopt_shared_pod():` in 0.3.69 when it became a
    # BOUNDED loop (adoption can report success without adding a pod, and the
    # `while` form span forever the first time it did) -- the intent held, the
    # string did not, and a count cannot tell those apart.
    def _body(start: str, end: str) -> str:
        return src.split(start)[1].split(end)[0]

    paths = {
        "ensure_ready": _body("def ensure_ready", "\n    def "),
        "acquire": _body("def acquire", "\n    def "),
        "growth": _body("def _grow_in_background", "\n    def "),
        "warm floor": _body("# Keep min_pods warm.", "def _warm_provision_one"),
    }
    for name, body in paths.items():
        assert "_adopt_shared_pod()" in body, f"{name} does not try adoption"
        if "_provision_one" in body:
            assert body.index("_adopt_shared_pod()") < body.index("_provision_one"), (
                f"{name} rents before it tries adoption")
