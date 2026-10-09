"""Training has to be visible to the thing that decides it is safe to restart.

The unattended upgrade waits for idleness before restarting, and it measured
idleness as requests in flight. A LoRA training run is started by a request that
returns immediately and then works for hours, so ``coderai_engine_inflight`` is
0 the whole time — an hourly upgrade would have restarted straight through one,
which is exactly the standing "never restart during training" rule.

Nothing was missing in the training path: ``codai/api/loras.py`` already
registers a ``training`` task, the engine already ships active tasks in
``/internal/engine-state``, and the front already stores them on
``EngineEntry.tasks``. The gap was that ``/metrics`` never exposed them and the
probe never looked. So this covers the new gauge and the probe's use of it.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SH = (ROOT / "packaging/linux/launcher/coderai-autoupgrade").read_text()


class _Engine:
    role = "engine"
    remote = False

    def __init__(self, name, inflight=0, tasks=None):
        self.name = name
        self.backend = "nvidia"
        self.healthy = True
        self.inflight = inflight
        self.vram = {}
        self.loaded_models = set()
        self.tasks = tasks or []


class _Registry:
    def __init__(self, engines):
        self._e = engines

    def all(self):
        return self._e


class _Front:
    metrics = None
    supervisor = None

    def __init__(self, engines):
        self.registry = _Registry(engines)


def _render(engines) -> str:
    from codai.frontproxy.metrics import render
    return render(_Front(engines))


def _gauge(text, name):
    for line in text.splitlines():
        if line.startswith(name + " ") or line.startswith(name + "{"):
            return float(line.rsplit(" ", 1)[1])
    return None


# --------------------------------------------------------------- the new gauge

def test_a_training_run_with_no_request_in_flight_is_visible():
    """The whole point: inflight 0, but the card is busy."""
    out = _render([_Engine("nvidia", inflight=0,
                           tasks=[{"kind": "training", "status": "running"}])])
    assert _gauge(out, "coderai_jobs_active") == 1
    assert _gauge(out, "coderai_requests_active") == 0


def test_the_per_engine_series_is_labelled_like_the_others():
    out = _render([_Engine("nvidia", tasks=[{"kind": "training", "status": "running"}])])
    assert 'coderai_engine_jobs_active{engine="nvidia",backend="nvidia",remote="0"} 1' in out


def test_finished_tasks_are_not_counted():
    """The engine reports recent terminal tasks too, for the Tasks page history."""
    out = _render([_Engine("nvidia", tasks=[
        {"kind": "training", "status": "done"},
        {"kind": "image", "status": "error"},
        {"kind": "video", "status": "cancelled"},
    ])])
    assert _gauge(out, "coderai_jobs_active") == 0


def test_queued_and_paused_count_as_busy():
    """A thermally paused generation has not finished; restarting loses it."""
    for state in ("running", "queued", "paused"):
        out = _render([_Engine("nvidia", tasks=[{"kind": "image", "status": state}])])
        assert _gauge(out, "coderai_jobs_active") == 1, state


def test_jobs_are_summed_across_engines():
    out = _render([
        _Engine("nvidia", tasks=[{"kind": "training", "status": "running"}]),
        _Engine("radeon", tasks=[{"kind": "image", "status": "running"},
                                 {"kind": "image", "status": "done"}]),
    ])
    assert _gauge(out, "coderai_jobs_active") == 2


def test_an_engine_with_no_tasks_reports_zero_not_nothing():
    """A missing series would read as "unknown" at the probe and fail closed."""
    out = _render([_Engine("nvidia")])
    assert _gauge(out, "coderai_jobs_active") == 0
    assert "coderai_engine_jobs_active" in out


def test_junk_in_the_task_list_does_not_break_the_scrape():
    """It is whatever another process put in a status payload."""
    out = _render([_Engine("nvidia", tasks=[None, "nonsense", 42,
                                            {"status": "running"}])])
    assert _gauge(out, "coderai_jobs_active") == 1


def test_the_gauge_says_it_overlaps_the_request_count():
    """A generation is BOTH a request in flight and a task; anyone summing the
    two would double-count, so the exposition has to say so."""
    out = _render([_Engine("nvidia", inflight=1,
                           tasks=[{"kind": "image", "status": "running"}])])
    assert _gauge(out, "coderai_requests_active") == 1
    assert _gauge(out, "coderai_jobs_active") == 1
    assert "overlaps" in out


# ------------------------------------------------------------------- the probe

def _awk(metrics_text: str) -> str:
    """Run the probe's awk exactly as the script has it."""
    import re
    import subprocess
    prog = re.search(r"\| awk '(.*?)'\n", SH, re.S).group(1)
    r = subprocess.run(["awk", prog], input=metrics_text, capture_output=True, text=True)
    return r.stdout.strip()


def test_the_probe_reads_both_numbers_from_one_scrape():
    assert _awk('coderai_engine_inflight{engine="a"} 2\n'
                'coderai_jobs_active 5\n') == "2 5"


def test_the_probe_sees_training_when_nothing_is_in_flight():
    assert _awk('coderai_engine_inflight{engine="a"} 0\n'
                'coderai_jobs_active 1\n') == "0 1"


def test_an_older_front_without_the_gauge_says_so_instead_of_zero():
    """Claiming 0 would silently restore the old blind spot."""
    assert _awk('coderai_engine_inflight{engine="a"} 0\n') == "0 -"


def test_an_unreadable_scrape_is_unknown_on_both():
    assert _awk("# nothing useful here\n") == "unknown unknown"


def test_the_probe_takes_the_max_not_the_sum():
    """Summing would double-count a generation, which is both."""
    blk = SH.split("inflight(){")[1].split("guard_idle()")[0]
    assert '[ "$j" -gt "$best" ] && best="$j"' in blk
    assert '[ "$r" -gt "$best" ] && best="$r"' in blk
    assert "MAXIMUM of the sources, never the sum" in SH


def test_a_missing_jobs_gauge_is_reported_in_the_detail():
    blk = SH.split("inflight(){")[1].split("guard_idle()")[0]
    assert "front too old" in blk


def test_the_training_path_already_registers_so_nothing_was_changed_there():
    """Guards the premise: if this registration ever goes away, the gauge goes
    quiet and the blind spot comes back without anything failing."""
    loras = (ROOT / "codai/api/loras.py").read_text()
    assert 'task_registry.register("training"' in loras
    assert 'status="running"' in loras.split('task_registry.register("training"')[1][:400]


def test_the_engine_still_reports_active_tasks_to_the_front():
    """The other half of the premise: the gauge is built from this payload."""
    app = (ROOT / "codai/api/app.py").read_text()
    assert '"tasks": tasks' in app
    blk = app.split("from codai.tasks import task_registry")[1][:900]
    assert '("running", "queued", "paused")' in blk
