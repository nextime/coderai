"""A distributed run, simulated: pods call in, rounds average, nothing leaks.

Every message in a distributed text-LoRA run is a POD CALLING IN. Rented pods
in different datacenters cannot address one another and nothing can address
them, so a design where the orchestrator initiates does not run at all. These
three endpoints are therefore the whole network surface, and this exercises
them the way pods would -- including the cases that are only visible in a
protocol: a straggler, a shard that dies, and a pod asking for an average the
instant the next round opens.
"""

import json
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

torch = pytest.importorskip("torch")
from safetensors.torch import load_file, save_file

from codai.api import loras
from codai.api import text_lora_fleet as flt
from codai.api import text_lora_rounds as tlr
from codai.api import text_lora_trainer as tlt


class FakeClient:
    def __init__(self):
        self.terminated = []
        self._n = 0

    def create_pod(self, **kw):
        self._n += 1
        return f"pod{self._n}"

    def terminate_pod(self, pod_id, attempts=4):
        self.terminated.append(pod_id)


class Enc:
    """Just enough of an encoded example for the plan to count them."""

    def __init__(self, i):
        self.input_ids = [1, 2]
        self.labels = [-100, 2]
        self.weight = 1.0
        self.line = i


@pytest.fixture
def run(monkeypatch, tmp_path):
    """A coordinator with three shards, registered so the endpoints find it."""
    monkeypatch.setattr(loras, "_loras_dir", lambda: str(tmp_path), raising=False)
    plan = tlr.RoundPlan(n_shards=3, seed=5)
    tracker = tlr.RoundTracker(plan, min_fraction=0.99, deadline_s=600)
    spec = flt.TrainerPodSpec(image="img", gpu_type_id="NVIDIA A40",
                              hourly_usd=0.45, max_pod_hourly_usd=0.90)
    client = FakeClient()
    fleet = flt.TrainerFleet(client, "job-d", spec)
    coord = loras._TextTrainCoordinator(
        "job-d", plan=plan, tracker=tracker, fleet=fleet,
        encoded=[Enc(i) for i in range(30)],
        out_dir=str(tmp_path / "out"), settings=tlt.QLoraSettings(),
        dataset_id="sha256:" + "a" * 64, base="Qwen/Qwen3.8-27B",
        total_rounds=3)
    fleet.provision([0, 1, 2])
    with loras._coord_lock:
        loras._train_coordinators["job-d"] = coord
    app = FastAPI()
    app.include_router(loras.router)
    yield {"client": TestClient(app), "coord": coord, "fleet": fleet,
           "runpod": client, "tmp": tmp_path}
    with loras._coord_lock:
        loras._train_coordinators.pop("job-d", None)


def _hdr(coord):
    return {"Authorization": f"Bearer {coord.secret}"}


def _adapter_bytes(tmp_path, value, name="a.safetensors"):
    p = tmp_path / name
    save_file({"lora_A.weight": torch.full((2, 2), float(value))}, str(p))
    return p.read_bytes()


# ------------------------------------------------------------------ auth -----

def test_a_pod_without_the_job_credential_gets_nothing(run):
    c = run["client"]
    assert c.get("/v1/loras/train/text/job-d/shard?shard=0").status_code == 401
    assert c.get("/v1/loras/train/text/job-d/shard?shard=0",
                 headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_the_credential_is_per_job_not_per_install(run):
    """A training pod holds its credential for as long as it lives; the blast
    radius should be one job."""
    coord = run["coord"]
    assert len(coord.secret) >= 32
    assert coord.secret not in json.dumps(coord.assignment(0))


def test_an_unknown_job_is_a_404_not_a_401(run):
    assert run["client"].get("/v1/loras/train/text/nope/shard?shard=0",
                             headers=_hdr(run["coord"])).status_code == 404


# -------------------------------------------------------------- assignment ---

def test_a_pod_is_told_its_shard_round_and_indices(run):
    got = run["client"].get("/v1/loras/train/text/job-d/shard?shard=1",
                            headers=_hdr(run["coord"])).json()
    assert got["shard"] == 1 and got["round"] == 0 and got["n_shards"] == 3
    assert got["resume_round"] is None, "round 0 starts from the base"
    # A per-ROUND slice, not the whole shard: ten examples per shard over three
    # rounds is four, four, two.
    assert len(got["indices"]) == 4
    assert got["base"] == "Qwen/Qwen3.8-27B"
    assert got["settings"]["rank"] == 16 and got["settings"]["max_len"] == 8192


def test_every_example_is_trained_exactly_once_across_shards_and_rounds(run):
    """The property that matters, over the wire rather than in the planner: no
    example is trained twice and none is skipped. An overlap wastes GPU hours
    on a lesson already learned; a gap silently drops data."""
    c, coord = run["client"], run["coord"]
    seen = []
    for rnd in range(3):
        for s in range(3):
            seen += c.get(f"/v1/loras/train/text/job-d/shard?shard={s}",
                          headers=_hdr(coord)).json()["indices"]
        coord.tracker.round = rnd + 1
    assert sorted(seen) == list(range(30))


def test_asking_for_work_counts_as_a_checkin(run):
    """A pod that is asking what to do is alive, and the reaper must know."""
    fleet = run["fleet"]
    fleet.pods[0].last_checkin = 0.0
    run["client"].get("/v1/loras/train/text/job-d/shard?shard=0",
                      headers=_hdr(run["coord"]))
    assert fleet.pods[0].last_checkin > 0


# ----------------------------------------------------------- the round -------

def test_a_full_round_averages_and_advances(run):
    c, coord, tmp = run["client"], run["coord"], run["tmp"]
    for shard, value in ((0, 0.0), (1, 3.0), (2, 3.0)):
        r = c.post(f"/v1/loras/train/text/job-d/round?shard={shard}&round=0"
                   f"&examples=10", headers=_hdr(coord),
                   content=_adapter_bytes(tmp, value, f"s{shard}.safetensors"))
        assert r.status_code == 200, r.text
    assert coord.close_round_if_ready() == 0
    assert coord.tracker.round == 1
    avg = load_file(coord.average_path(0))["lora_A.weight"]
    assert avg.flatten()[0].item() == pytest.approx(2.0)


def test_the_average_is_weighted_by_the_examples_each_pod_reported(run):
    c, coord, tmp = run["client"], run["coord"], run["tmp"]
    c.post("/v1/loras/train/text/job-d/round?shard=0&round=0&examples=100",
           headers=_hdr(coord), content=_adapter_bytes(tmp, 0.0, "x0.safetensors"))
    c.post("/v1/loras/train/text/job-d/round?shard=1&round=0&examples=300",
           headers=_hdr(coord), content=_adapter_bytes(tmp, 4.0, "x1.safetensors"))
    c.post("/v1/loras/train/text/job-d/round?shard=2&round=0&examples=0",
           headers=_hdr(coord), content=_adapter_bytes(tmp, 99.0, "x2.safetensors"))
    coord.close_round_if_ready()
    avg = load_file(coord.average_path(0))["lora_A.weight"].flatten()[0].item()
    assert avg == pytest.approx(3.0), "a shard that trained nothing gets no vote"


def test_a_pod_can_fetch_the_average_it_is_meant_to_resume_from(run):
    c, coord, tmp = run["client"], run["coord"], run["tmp"]
    assert c.get("/v1/loras/train/text/job-d/round/0/adapter",
                 headers=_hdr(coord)).status_code == 404
    for s in range(3):
        c.post(f"/v1/loras/train/text/job-d/round?shard={s}&round=0&examples=10",
               headers=_hdr(coord), content=_adapter_bytes(tmp, s, f"y{s}.safetensors"))
    coord.close_round_if_ready()
    got = c.get("/v1/loras/train/text/job-d/round/0/adapter", headers=_hdr(coord))
    assert got.status_code == 200 and len(got.content) > 0


def test_the_average_exists_before_the_next_round_opens(run):
    """A pod asking for round k's adapter the instant k+1 opens must get the
    file, not a 404: the two are not atomic from its side."""
    c, coord, tmp = run["client"], run["coord"], run["tmp"]
    for s in range(3):
        c.post(f"/v1/loras/train/text/job-d/round?shard={s}&round=0&examples=10",
               headers=_hdr(coord), content=_adapter_bytes(tmp, s, f"z{s}.safetensors"))
    coord.close_round_if_ready()
    assert coord.tracker.round == 1
    assert os.path.isfile(coord.average_path(0))


def test_the_next_round_tells_pods_where_to_resume_from(run):
    c, coord, tmp = run["client"], run["coord"], run["tmp"]
    for s in range(3):
        c.post(f"/v1/loras/train/text/job-d/round?shard={s}&round=0&examples=10",
               headers=_hdr(coord), content=_adapter_bytes(tmp, s, f"w{s}.safetensors"))
    coord.close_round_if_ready()
    got = c.get("/v1/loras/train/text/job-d/shard?shard=0",
                headers=_hdr(coord)).json()
    assert got["round"] == 1 and got["resume_round"] == 0


# ------------------------------------------------------ out-of-step pods -----

def test_a_stale_round_is_a_409_so_the_pod_resyncs(run):
    """Not a 400: the body is fine, the pod is behind. It should re-ask for an
    assignment rather than retry the same upload."""
    c, coord, tmp = run["client"], run["coord"], run["tmp"]
    for s in range(3):
        c.post(f"/v1/loras/train/text/job-d/round?shard={s}&round=0&examples=10",
               headers=_hdr(coord), content=_adapter_bytes(tmp, s, f"q{s}.safetensors"))
    coord.close_round_if_ready()
    late = c.post("/v1/loras/train/text/job-d/round?shard=1&round=0&examples=10",
                  headers=_hdr(coord), content=_adapter_bytes(tmp, 7.0, "late.safetensors"))
    assert late.status_code == 409 and "stale" in late.json()["detail"]


def test_an_empty_adapter_body_is_refused(run):
    r = run["client"].post("/v1/loras/train/text/job-d/round?shard=0&round=0",
                           headers=_hdr(run["coord"]), content=b"")
    assert r.status_code == 400


def test_a_round_closes_without_a_dead_shard_once_the_deadline_passes(run):
    """Waiting for every shard is what turned one dead pod into a stalled
    fleet. Two of three, past the deadline, still makes progress."""
    c, coord, tmp = run["client"], run["coord"], run["tmp"]
    import time as _t
    for s in (0, 2):
        c.post(f"/v1/loras/train/text/job-d/round?shard={s}&round=0&examples=10",
               headers=_hdr(coord), content=_adapter_bytes(tmp, s, f"d{s}.safetensors"))
    # Opened ten seconds ago against a ten-minute deadline: two of three have
    # arrived and the third still has time.
    coord.tracker._opened_at[0] = _t.time() - 10
    assert coord.close_round_if_ready() is None, "not yet: inside the deadline"
    coord.tracker.deadline_s = 1.0                 # the deadline has now passed
    assert coord.close_round_if_ready() == 0
    assert coord.tracker.round == 1


def test_a_silent_pod_is_released_and_its_shard_reported(run):
    coord, fleet, runpod = run["coord"], run["fleet"], run["runpod"]
    import time as _t
    fleet.spec.max_silence_s = 60.0
    # An hour ago, not 0.0: `silent_for` reads a zero last_checkin as "never
    # checked in" and measures from the pod's start instead, so 0.0 would mean
    # a pod rented a moment ago rather than one that has gone quiet.
    for p in fleet.pods.values():
        p.last_checkin = _t.time() - 3600
    gone = coord.reap_the_dead()
    assert sorted(gone) == [0, 1, 2]
    assert sorted(runpod.terminated) == ["pod1", "pod2", "pod3"]
    assert flt.orphans() == [], "released means no longer billing and no longer listed"


# -------------------------------------------------- the pod's exit signal ----

def test_a_finished_job_tells_the_pod_to_stop(run):
    """Without this a pod polls forever on a job that is over -- which is a pod
    billing for nothing, the exact failure the fleet's reaper exists to catch
    and the one it should never have to."""
    c, coord, tmp = run["client"], run["coord"], run["tmp"]
    for rnd in range(3):
        for s in range(3):
            c.post(f"/v1/loras/train/text/job-d/round?shard={s}&round={rnd}"
                   f"&examples=10", headers=_hdr(coord),
                   content=_adapter_bytes(tmp, s + rnd, f"f{rnd}{s}.safetensors"))
        assert coord.close_round_if_ready() == rnd
    got = c.get("/v1/loras/train/text/job-d/shard?shard=0",
                headers=_hdr(coord)).json()
    assert got["finished"] is True
    assert "indices" not in got, "no work is handed out after the last round"


def test_a_round_slice_is_a_third_of_the_shard_over_three_rounds(run):
    """Round length is chosen from measured throughput, so a shard is cut into
    per-round slices -- and over a full cycle a pod still sees its whole shard
    exactly once."""
    c, coord = run["client"], run["coord"]
    seen = []
    for rnd in range(3):
        got = c.get("/v1/loras/train/text/job-d/shard?shard=0",
                    headers=_hdr(coord)).json()
        assert len(got["indices"]) <= 4, "a third of ten, rounded up"
        seen += got["indices"]
        coord.tracker.round = rnd + 1          # pretend the round closed
    whole = coord.plan.indices_for(0, 30)
    assert sorted(seen) == sorted(whole), "the full shard, once, across rounds"


def test_the_pod_runner_is_served_to_an_authenticated_pod(run):
    """Served rather than baked, so the protocol can change without
    republishing the image and the two can never disagree."""
    c, coord = run["client"], run["coord"]
    assert c.get("/v1/loras/train/text/job-d/runner").status_code == 401
    got = c.get("/v1/loras/train/text/job-d/runner", headers=_hdr(coord))
    assert got.status_code == 200
    assert "CODERAI_TRAIN_SHARD" in got.text and "def main" in got.text


def test_the_dataset_is_fetchable_by_a_pod_but_not_anonymously(run, tmp_path):
    from codai.api import text_lora_dataset as tld
    body = b'{"messages": [{"role": "user", "content": "q"}], "answer": "a"}\n'
    hexd, _existed, _n = tld.save_dataset_stream([body])
    c, coord = run["client"], run["coord"]
    assert c.get(f"/v1/loras/datasets/sha256:{hexd}/content").status_code == 401
    got = c.get(f"/v1/loras/datasets/sha256:{hexd}/content", headers=_hdr(coord))
    assert got.status_code == 200 and got.content == body
