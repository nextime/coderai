"""Fleet status for an external monitor: how many pods, where, which model.

Admin-scoped only — the same gate as the spend figures, because this is the
same document.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _rows():
    return [
        {"state": "ready", "healthy": True, "model": "surya", "data_center": "EU-SE-1",
         "gpu": "A40", "gpu_count": 1, "inflight": 2, "hourly_usd": 1.09,
         "cloud_type": "SECURE", "is_spot": False},
        {"state": "ready", "healthy": True, "model": "surya", "data_center": "EU-SE-1",
         "gpu": "A40", "gpu_count": 1, "inflight": 0, "hourly_usd": 1.09,
         "cloud_type": "SECURE", "is_spot": False},
        {"state": "booting", "healthy": False, "model": "qwen", "data_center": "EU-RO-1",
         "gpu": "A100", "gpu_count": 2, "inflight": 0, "hourly_usd": 2.40,
         "cloud_type": "SECURE", "is_spot": True},
        {"state": "ready", "healthy": False, "model": "qwen", "data_center": "EU-RO-1",
         "gpu": "A100", "gpu_count": 2, "inflight": 0, "hourly_usd": 2.40,
         "cloud_type": "SECURE", "is_spot": False},
    ]


def test_the_summary_answers_how_many_where_and_which_model():
    from codai.api.runpod_worker import fleet_summary
    s = fleet_summary(_rows())
    assert s["pods_total"] == 4
    assert s["pods_ready"] == 2          # healthy and serving
    assert s["pods_booting"] == 1
    assert s["pods_unhealthy"] == 1      # ready-state but failed its probe
    assert s["by_model"] == {"surya": 2, "qwen": 2}
    assert s["by_region"] == {"EU-SE-1": 2, "EU-RO-1": 2}
    assert s["by_gpu"] == {"A40": 2, "A100": 2}
    assert s["spot_pods"] == 1


def test_the_hourly_rate_counts_only_what_is_actually_serving():
    """A booting or unhealthy pod still bills, but reporting it as capacity
    would overstate what the fleet can do; the ledger covers real spend."""
    from codai.api.runpod_worker import fleet_summary
    s = fleet_summary(_rows())
    assert s["hourly_usd"] == round(1.09 * 2, 4)
    assert s["gpus_in_use"] == 2         # multi-GPU pods count their cards
    assert s["inflight"] == 2


def test_an_unknown_region_is_labelled_not_dropped():
    from codai.api.runpod_worker import fleet_summary
    s = fleet_summary([{"state": "ready", "healthy": True, "model": "m",
                        "gpu": "", "gpu_count": 1, "hourly_usd": 0.5}])
    assert s["by_region"] == {"unknown": 1}
    assert s["by_gpu"] == {"unknown": 1}


def test_each_pod_reports_where_it_is_and_what_it_serves():
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    blk = src.split("def pods_status")[1].split("def ")[0]
    for field in ('"data_center"', '"cloud_type"', '"pool"', '"engine"',
                  '"gpu_count"', '"model"'):
        assert field in blk, field
    # and the placement has to be carried on the pod to be reportable
    assert "data_center: str = \"\"" in src.split("class PodHandle")[1][:600]
    # The handle gets its region from the pool's own resolver. It used to read
    # a local `dc` that did not exist in this function, so the pod was created
    # and then the handle raised NameError — the A40 billed untracked.
    assert "data_center=self._data_center()" in src


def test_status_and_spend_are_the_same_document_behind_the_same_gate():
    src = (ROOT / "codai/api/app.py").read_text()
    assert '@app.get("/v1/runpod/status"' in src
    blk = src.split('async def runpod_status')[1].split("@app.get")[0]
    assert "return await runpod_spend(request)" in blk, \
        "one implementation, so the two can never disagree"
    spend = src.split('async def runpod_spend')[1].split("@app.get")[0]
    assert "_admin_scoped_token(request)" in spend and "status_code=403" in spend


def test_the_summary_is_in_the_payload():
    src = (ROOT / "codai/api/app.py").read_text()
    blk = src.split('async def runpod_spend')[1].split("@app.get")[0]
    assert "fleet_summary(pods)" in blk
    assert '"summary": summary' in blk
