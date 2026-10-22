"""The two limits that actually bind a pod: host memory, and the KV cache.

Reserved VRAM is not one of them. vLLM claims its gpu_memory_utilization share
up front, so a pod reads ~90% VRAM from the moment it boots whether it is
serving one request or sixty-four -- measured 2026-10-22: 23.22 GiB weights +
3.49 activation + 2.61 CUDA graphs + 13.27 KV on a 48 GB A40, with
gpu_memory_utilization 0.9. A bar pinned at 90% says nothing about load.

What binds, measured on an OCR pod 2026-10-18: host RAM. 43 GB free at 32
in-flight, 12.95 GB at 64, OOM-killed at 128 -- the GPU was never the limit.
And on the GPU side it is KV occupancy, with PREEMPTIONS as the proof the cache
ran out: vLLM evicts a running request and recomputes it later, so throughput
collapses while every other counter still looks busy.
"""
import pytest

from codai.api import serving_load as sl


def setup_function():
    sl.forget()


BODY = """
# HELP vllm:kv_cache_usage_perc GPU KV-cache usage
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.265
vllm:num_requests_running{engine="0",model_name="m"} 75
vllm:num_requests_waiting{engine="0",model_name="m"} 12
vllm:num_preemptions{engine="0",model_name="m"} 4
vllm:prefix_cache_hits{engine="0",model_name="m"} 250
vllm:prefix_cache_queries{engine="0",model_name="m"} 1000
vllm:something_else{engine="0"} 999
"""


def test_the_ladders_numbers_come_back():
    """These are the figures the 64-in-flight rung actually produced."""
    out = sl.parse_prometheus(BODY)
    assert out["kv_cache_pct"] == 26.5      # reported 0..1, surfaced as a %
    assert out["running"] == 75
    assert out["waiting"] == 12
    assert out["preemptions"] == 4
    assert out["prefix_hit_pct"] == 25.0


def test_unrelated_metrics_are_ignored():
    assert "something_else" not in sl.parse_prometheus(BODY)


def test_counters_sum_across_engines_but_a_fraction_does_not():
    """Adding two engines' cache FRACTIONS would give a meaningless >100%."""
    body = ('vllm:kv_cache_usage_perc{engine="0"} 0.5\n'
            'vllm:kv_cache_usage_perc{engine="1"} 0.3\n'
            'vllm:num_requests_running{engine="0"} 4\n'
            'vllm:num_requests_running{engine="1"} 6\n')
    out = sl.parse_prometheus(body)
    assert out["kv_cache_pct"] == 50.0
    assert out["running"] == 10


def test_a_metric_without_labels_parses():
    assert sl.parse_prometheus("vllm:num_requests_running 3")["running"] == 3


def test_junk_is_survivable():
    assert sl.parse_prometheus("") == {}
    assert sl.parse_prometheus(None) == {}
    assert sl.parse_prometheus("vllm:num_requests_running notanumber") == {}


def test_no_queries_means_no_hit_rate():
    """0/0 is not 0%."""
    out = sl.parse_prometheus('vllm:prefix_cache_queries{e="0"} 0')
    assert "prefix_hit_pct" not in out


# --------------------------------------------------------------- host memory #

def test_the_cgroup_limit_wins_over_meminfo(tmp_path, monkeypatch):
    """THE TRAP: in a container /proc/meminfo reports the HOST's memory, so a
    pod with a 60 GB limit on a 1 TB machine looks like it has hundreds of GB
    free right up to the moment the kernel kills it."""
    real_open = open

    def _open(path, *a, **k):
        if path == "/sys/fs/cgroup/memory.current":
            return real_open(tmp_path / "cur", *a, **k)
        if path == "/sys/fs/cgroup/memory.max":
            return real_open(tmp_path / "max", *a, **k)
        return real_open(path, *a, **k)
    (tmp_path / "cur").write_text("48000000000\n")
    (tmp_path / "max").write_text("60000000000\n")
    monkeypatch.setattr("builtins.open", _open)
    out = sl.host_memory()
    assert out["ram_source"] == "cgroup"
    assert out["ram_total_gb"] == 60.0
    assert out["ram_used_gb"] == 48.0
    assert out["ram_free_gb"] == 12.0


def test_an_unlimited_cgroup_falls_back_to_meminfo(tmp_path, monkeypatch):
    real_open = open

    def _open(path, *a, **k):
        if path == "/sys/fs/cgroup/memory.max":
            return real_open(tmp_path / "max", *a, **k)
        if path == "/sys/fs/cgroup/memory.current":
            raise FileNotFoundError(path)
        return real_open(path, *a, **k)
    (tmp_path / "max").write_text("max\n")
    monkeypatch.setattr("builtins.open", _open)
    out = sl.host_memory()
    assert out.get("ram_source") in ("meminfo", None)


def test_host_memory_is_read_on_this_machine():
    """Not mocked: the numbers must be plausible wherever this runs."""
    out = sl.host_memory()
    if out:
        assert out["ram_total_gb"] > 0
        assert 0 <= out["ram_used_gb"] <= out["ram_total_gb"] * 1.05


# -------------------------------------------------------------- the fetching #

def test_the_engine_is_asked_for_its_metrics(monkeypatch):
    seen = {}

    class _R:
        status_code = 200
        text = BODY
    def _get(url, timeout=None):
        seen.update(url=url, timeout=timeout)
        return _R()
    monkeypatch.setattr("requests.get", _get)
    assert sl.engine_load("http://127.0.0.1:44533")["running"] == 75
    assert seen["url"] == "http://127.0.0.1:44533/metrics"


def test_a_fresh_read_is_reused(monkeypatch):
    calls = []

    class _R:
        status_code = 200
        text = BODY
    monkeypatch.setattr("requests.get", lambda *a, **k: calls.append(1) or _R())
    sl.engine_load("http://e:1")
    sl.engine_load("http://e:1")
    assert len(calls) == 1


def test_an_engine_that_does_not_answer_is_not_a_failure(monkeypatch):
    def _boom(*a, **k):
        raise OSError("refused")
    monkeypatch.setattr("requests.get", _boom)
    assert sl.engine_load("http://e:1") == {}


def test_no_engine_url_means_host_memory_only(monkeypatch):
    out = sl.snapshot("")
    assert "kv_cache_pct" not in out
    assert "running" not in out
