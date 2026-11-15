"""The text-LoRA route: its own body, the shared job machinery underneath.

`POST /v1/loras/train/text` is a sibling of the diffusion trainer rather than a
`target: "text"` on it, because the two bodies share almost nothing -- images,
a resolution and a frame count against a JSONL of chat examples -- and one
model carrying both makes every field conditional on the target, which is a
schema that can no longer tell a client what to send.

What it DOES share is the job record, the session that survives a client
restart, resume, cancel/pause through the Tasks page and the adapter registry.
These tests pin the seam: a text job is a job like any other, and the dataset
pass refuses the one failure that otherwise looks like success -- training on a
fraction of the data because the length limit, the template or the field names
were wrong.
"""

import hashlib
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from codai.api import loras
from codai.api import text_lora_dataset as tld


class Tok:
    eos_token_id = 7

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False):
        return "P:" + "".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids = [ord(c) for c in text]
        if not return_offsets_mapping:
            return {"input_ids": ids}
        return {"input_ids": ids,
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


@pytest.fixture
def client(monkeypatch, tmp_path):
    """The router on a bare app, with auth and the store pointed at tmp."""
    monkeypatch.setattr(loras, "_require_api_auth", lambda: True, raising=False)
    monkeypatch.setattr(loras, "_loras_dir", lambda: str(tmp_path), raising=False)
    app = FastAPI()
    app.include_router(loras.router)
    app.dependency_overrides[loras._require_api_auth] = lambda: True
    return TestClient(app)


def _jsonl_bytes(rows):
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode()


ROWS = [{"messages": [{"role": "user", "content": "q"}], "answer": "abc",
         "mask": [[0, 1]], "weight": 0.3333},
        {"messages": [{"role": "user", "content": "r"}], "answer": "defg",
         "weight": 1.0}]


# ------------------------------------------------------------- the store -----

def test_a_dataset_is_stored_under_the_hash_the_client_can_compute(client):
    """So a 119 MB file is never sent twice: the id is the plain sha256 of the
    bytes, which the client already knows before uploading."""
    body = _jsonl_bytes(ROWS)
    r = client.post("/v1/loras/datasets", content=body)
    assert r.status_code == 200, r.text
    assert r.json()["id"] == "sha256:" + hashlib.sha256(body).hexdigest()
    assert r.json()["existed"] is False and r.json()["bytes"] == len(body)


def test_resending_the_same_dataset_is_free(client):
    body = _jsonl_bytes(ROWS)
    first = client.post("/v1/loras/datasets", content=body).json()
    again = client.post("/v1/loras/datasets", content=body).json()
    assert again["id"] == first["id"] and again["existed"] is True


def test_a_client_can_ask_before_uploading(client):
    body = _jsonl_bytes(ROWS)
    ident = "sha256:" + hashlib.sha256(body).hexdigest()
    assert client.get(f"/v1/loras/datasets/{ident}").status_code == 404
    client.post("/v1/loras/datasets", content=body)
    got = client.get(f"/v1/loras/datasets/{ident}")
    assert got.status_code == 200 and got.json()["exists"] is True


def test_an_empty_upload_is_refused(client):
    assert client.post("/v1/loras/datasets", content=b"").status_code == 400


def test_a_dropped_upload_leaves_nothing_to_mistake_for_a_dataset(tmp_path, monkeypatch):
    """A connection that dies at 80 MB must not leave a file a later run reads
    as complete -- the store renames only once the whole body has arrived."""
    monkeypatch.setattr(loras, "_loras_dir", lambda: str(tmp_path), raising=False)

    def _chunks():
        yield b'{"messages": [], "answer": "a"}\n'
        raise ConnectionResetError("client vanished")

    with pytest.raises(ConnectionResetError):
        tld.save_dataset_stream(_chunks())
    import os
    assert not [f for f in os.listdir(tld.dataset_dir()) if f.endswith(".jsonl")]


def test_a_malformed_id_is_not_a_path_traversal(client):
    for bad in ("../../etc/passwd", "sha256:../x", "nothex"):
        assert client.get(f"/v1/loras/datasets/{bad}").status_code in (404, 400)


# -------------------------------------------------------------- the route ----

def test_the_route_refuses_a_dataset_it_cannot_find(client):
    r = client.post("/v1/loras/train/text",
                    json={"name": "a", "base_model": "b",
                          "dataset": "/nope/missing.jsonl"})
    assert r.status_code == 400 and "not found" in r.json()["detail"]


def test_the_route_requires_a_dataset(client):
    r = client.post("/v1/loras/train/text", json={"name": "a", "base_model": "b"})
    assert r.status_code == 400
    assert "dataset" in r.json()["detail"]


def test_an_uploaded_dataset_is_usable_by_id(client, tmp_path, monkeypatch):
    """The point of the store: a client on another machine trains from an id."""
    body = _jsonl_bytes(ROWS)
    ident = client.post("/v1/loras/datasets", content=body).json()["id"]
    monkeypatch.setattr(loras, "_run_text_train_job",
                        _fake_run(lambda: {"ok": 1}), raising=False)
    r = client.post("/v1/loras/train/text",
                    json={"name": "adapter", "base_model": "b",
                          "dataset_id": ident, "wait": False})
    assert r.status_code == 200, r.text
    assert r.json()["job_id"].startswith("job-")


def test_a_bad_name_is_refused_before_anything_is_read(client):
    for name in ("../escape", "a/b", ""):
        r = client.post("/v1/loras/train/text",
                        json={"name": name, "base_model": "b", "dataset": "/x"})
        assert r.status_code == 400
        assert "name" in r.json()["detail"].lower()


def _fake_run(fn):
    async def _run(req, job_id, dataset_path):
        return fn()
    return _run


def test_a_text_job_is_a_job_like_any_other(client, tmp_path, monkeypatch):
    """It must appear in the same record the Tasks page and /v1/loras/progress
    read, or a text run is invisible to everything that watches training."""
    body = _jsonl_bytes(ROWS)
    ident = client.post("/v1/loras/datasets", content=body).json()["id"]
    monkeypatch.setattr(loras, "_run_text_train_job",
                        _fake_run(lambda: {}), raising=False)
    job = client.post("/v1/loras/train/text",
                      json={"name": "adapter", "base_model": "qwen",
                            "dataset_id": ident}).json()
    rec = client.get(f"/v1/loras/progress?job={job['job_id']}").json()
    assert rec["target"] == "text"
    assert rec["name"] == "adapter" and rec["base_model"] == "qwen"
    # and recoverable by session alone, after a client lost the job id
    by_session = client.get(f"/v1/loras/progress?session={job['session']}").json()
    assert by_session["job_id"] == job["job_id"]


def test_a_text_job_can_be_cancelled_through_the_shared_endpoint(client, monkeypatch):
    body = _jsonl_bytes(ROWS)
    ident = client.post("/v1/loras/datasets", content=body).json()["id"]
    monkeypatch.setattr(loras, "_run_text_train_job",
                        _fake_run(lambda: {}), raising=False)
    job = client.post("/v1/loras/train/text",
                      json={"name": "adapter", "base_model": "b",
                            "dataset_id": ident}).json()
    assert client.post(f"/v1/loras/jobs/{job['job_id']}/cancel").status_code == 200


# --------------------------------------------------- the dataset-pass guard --

def _req(**kw):
    base = {"name": "a", "base_model": "b", "dataset": "/x"}
    base.update(kw)
    return loras.TextLoraTrainRequest(**base)


def test_the_pass_reports_what_would_be_trained(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_bytes(_jsonl_bytes(ROWS))
    enc, stats = loras._text_dataset_pass(_req(max_len=4096), str(path), Tok())
    assert stats.examples == 2 and not stats.skipped
    assert stats.masked_tokens == 1          # the one masked character
    assert stats.trained_tokens > 0


def test_training_on_a_fraction_of_the_data_is_refused(tmp_path):
    """The failure this guard exists for: a wrong max_len, template or field
    name silently shrinks the dataset, the run succeeds, and nothing
    downstream can tell."""
    rows = [ROWS[0]] + [{"messages": [{"role": "user", "content": "q"}],
                         "answer": "x" * 200} for _ in range(5)]
    path = tmp_path / "d.jsonl"
    path.write_bytes(_jsonl_bytes(rows))
    with pytest.raises(ValueError) as e:
        loras._text_dataset_pass(_req(max_len=30), str(path), Tok())
    msg = str(e.value)
    assert "5 of 6 examples were skipped" in msg
    assert "83.3%" in msg and "5.0% limit" in msg
    assert "line" in msg, "name the first offenders so the cause is findable"


def test_the_refusal_can_be_overridden_deliberately(tmp_path):
    rows = [ROWS[0]] + [{"messages": [{"role": "user", "content": "q"}],
                         "answer": "x" * 200} for _ in range(5)]
    path = tmp_path / "d.jsonl"
    path.write_bytes(_jsonl_bytes(rows))
    enc, stats = loras._text_dataset_pass(
        _req(max_len=30, max_skipped_fraction=0.9), str(path), Tok())
    assert stats.examples == 1 and len(stats.skipped) == 5


def test_an_empty_dataset_is_an_error_not_an_empty_run(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_bytes(b"\n\n")
    with pytest.raises(ValueError, match="no usable examples"):
        loras._text_dataset_pass(_req(), str(path), Tok())


def test_a_field_map_is_honoured_by_the_pass(tmp_path):
    rows = [{"msgs": [{"role": "user", "content": "q"}], "out": "abc", "w": 0.5}]
    path = tmp_path / "d.jsonl"
    path.write_bytes(_jsonl_bytes(rows))
    enc, stats = loras._text_dataset_pass(
        _req(field_map={"messages": "msgs", "answer": "out", "weight": "w"}),
        str(path), Tok())
    assert stats.examples == 1 and enc[0].weight == 0.5
