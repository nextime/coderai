"""A training pod showed dashes for VRAM, RAM, GPU and CPU for its whole life.

The admin page pulls telemetry FROM the pods: `GET /internal/engine-state` on
each, listed by `held_pod_endpoints()`. That walked `_pools` only, and a
training pod belongs to a job, never to a pool -- so nothing ever asked it.
And even asked, the trainer image did not serve that path: it had `/healthz`,
`/status`, `/adapter`, `/dataset`, `/round` and nothing else.

Both halves were needed. This is the fourth thing the trainer lost by not
being a pool, after cleanup, visibility and its label.
"""

import ast
import inspect
from pathlib import Path

import pytest

from codai.api import runpod_worker as rw


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "packaging/runpod/text_lora_pod_runner.py"


# ------------------------------------------------ the pod is in the list ----

def test_fleet_pods_are_listed_for_telemetry():
    src = inspect.getsource(rw.held_pod_endpoints)
    assert "FLEET_POOL_PREFIX" in src


def test_only_a_pod_with_an_address_is_asked():
    """Asking one that has not published a port yet is a request that cannot
    succeed, and the round is time-bounded for every other pod in it."""
    src = inspect.getsource(rw.held_pod_endpoints)
    assert "not url or not _is_this_process" in src


def test_a_pool_pod_is_not_listed_twice():
    src = inspect.getsource(rw.held_pod_endpoints)
    assert "in seen" in src


def test_the_fleet_pods_bearer_is_used_not_the_pools():
    """Each trainer pod has its own minted secret; a pool's shared token would
    be answered 401."""
    src = inspect.getsource(rw.held_pod_endpoints)
    assert 'info.get("api_key")' in src


def test_this_list_still_stays_server_side():
    """Its docstring's own reason for existing separately from pods_status: a
    bearer must never reach a browser."""
    doc = rw.held_pod_endpoints.__doc__ or ""
    assert "bearer" in doc


# ---------------------------------------------- the pod answers the ask ----

def _runner_ast():
    return ast.parse(RUNNER.read_text())


def test_the_runner_serves_the_endpoint_the_collector_asks_for():
    src = RUNNER.read_text()
    assert '"/internal/engine-state"' in src
    assert "_engine_state()" in src


def test_the_runner_parses():
    _runner_ast()      # a syntax error here is a pod that will not boot


def test_the_helper_is_module_level():
    """It was first written inside the handler class, where it could not be
    called as a function."""
    tree = _runner_ast()
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert "_engine_state" in names


def test_it_reports_free_and_total_per_card():
    """The collector derives USED from free and total, because a context-free
    query returns free -- reporting used would be read as free."""
    body = _engine_state_body()
    assert '"free"' in body and '"total"' in body
    assert '"devices"' in body


def _engine_state_body():
    """The whole function, bounded by the marker after it rather than a guessed
    character count -- a short window silently passed or failed on where the
    text happened to end."""
    src = RUNNER.read_text()
    i = src.index("def _engine_state")
    return src[i:src.index("_CPU_LAST = None", i)]


def test_host_ram_and_cpu_are_under_load():
    """`_fetch` reads these from a `load` block and ignores them anywhere
    else, so the shape is the contract."""
    body = _engine_state_body()
    for k in ("ram_total_gb", "ram_free_gb", "ram_used_gb", "cpu_util",
              "cpu_cores"):
        assert k in body, k
    assert 'out["load"] = load' in body


def test_cpu_is_a_delta_not_a_total_since_boot():
    """A single /proc/stat read gives totals since boot and would report the
    same number forever."""
    src = RUNNER.read_text()
    assert "_CPU_LAST" in src


def test_served_request_counters_are_left_out():
    """Zero served requests would read as an idle SERVING pod, which is the
    opposite of the truth for a trainer. Keyed on the QUOTED name: the
    docstring names it in prose precisely to explain the omission."""
    assert '"requests_done"' not in _engine_state_body()


def test_telemetry_never_costs_the_training_run():
    """Every probe is best-effort: a driver without NVML, a container without
    /proc/meminfo, an import that is not there."""
    assert _engine_state_body().count("except Exception") >= 4


def test_vram_is_reported_in_gigabytes():
    """`mem_get_info` returns BYTES and the collector's fields are named *_gb
    with no conversion anywhere, so reporting bytes put a number 1e9 too large
    on the page under a GB label. Caught by feeding the payload through the
    real collector instead of reading the code."""
    body = _engine_state_body()
    assert "round(free / 1e9, 2)" in body
    assert "round(total / 1e9, 2)" in body


def test_the_card_is_named_and_indexed_like_the_canonical_block():
    """codai.api.app reports index and name per device so a reader can see
    WHICH card is busy."""
    body = _engine_state_body()
    assert '"index": i' in body and "get_device_name(i)" in body


def test_the_totals_sit_beside_the_devices():
    """A reader that looks at `vram.total` rather than walking `devices` must
    get the same answer."""
    body = _engine_state_body()
    for k in ("used", "free", "total", "gpu", "device_count"):
        assert f'"{k}"' in body, k


def test_the_payload_survives_the_real_collector():
    """The contract that matters: not the field names I chose, but what
    pod_telemetry actually extracts from them."""
    import importlib.util
    from codai.api.pod_telemetry import cards_from_engine_state, summarise
    spec = importlib.util.spec_from_file_location("runner_mod", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    state = mod._engine_state()
    cards = cards_from_engine_state(state)
    if not cards:
        pytest.skip("no CUDA device on this machine to report")
    got = summarise(cards)
    assert got.get("gpu_cards", 0) >= 1
    # GB, not bytes: a 1e9 error would show here as a vram_total in the
    # billions on hardware that has tens of gigabytes.
    assert 0 < got["vram_total_gb"] < 10_000
    assert 0 <= got["vram_used_gb"] <= got["vram_total_gb"]
