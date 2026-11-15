"""A client that keeps its own voiceprints must not have to enrol them.

``/v1/audio/speaker-verify`` could only verify against an *enrolled* speaker, so
a caller holding its own reference vector had two bad choices: enrol into our
store (state it did not want, in a registry it does not own) or reimplement
cosine similarity and the threshold itself — which is the thing the endpoint
exists to spare every client.

So ``reference`` takes the vector directly. The interesting cases are the
refusals: a reference from a different embedding model must not quietly compare
as noise, and a malformed one must read as the caller's 400, never our 500.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import codai.api.transcriptions as tr
from codai.api import speaker_registry


DIM = 8
VEC_A = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
VEC_B = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
VEC_NEAR_A = [0.98, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.fixture
def client(monkeypatch, tmp_path):
    """The router with the embedder stubbed: the clip always embeds to VEC_A.

    Verification is cosine arithmetic over two vectors; using a real ECAPA model
    here would test speechbrain, not this endpoint.
    """
    monkeypatch.setenv("CODERAI_SPEAKER_STORE", str(tmp_path / "speakers.json"))

    async def _fake_embed(file, backend, model):
        return {"embedding": list(VEC_A), "backend": backend or "ecapa",
                "model": model or "stub", "dim": DIM}

    monkeypatch.setattr(tr, "_embed_upload", _fake_embed)
    app = FastAPI()
    app.include_router(tr.router)
    return TestClient(app)


def _post(client, **form):
    return client.post("/v1/audio/speaker-verify",
                       files={"file": ("clip.wav", b"RIFF-not-really")},
                       data=form)


def test_a_reference_vector_verifies_without_enrolling_anything(client, tmp_path):
    r = _post(client, reference=json.dumps(VEC_A))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["same_speaker"] is True
    assert body["score"] == pytest.approx(1.0, abs=1e-6)
    assert body["reference"] == "vector" and body["dim"] == DIM
    # The point of the feature: no state was created on our side.
    assert speaker_registry.list_speakers() == []
    assert not (tmp_path / "speakers.json").exists()


def test_a_different_speaker_is_refused(client):
    body = _post(client, reference=json.dumps(VEC_B)).json()
    assert body["same_speaker"] is False
    assert body["score"] == pytest.approx(0.0, abs=1e-6)


def test_the_threshold_decides(client):
    """Same pair of vectors, opposite verdicts — so the caller's threshold is
    what is being applied, not a constant that happens to agree with it."""
    ref = json.dumps(VEC_NEAR_A)
    loose = _post(client, reference=ref, threshold=0.25).json()
    strict = _post(client, reference=ref, threshold=0.99).json()
    assert loose["same_speaker"] is True
    assert strict["same_speaker"] is False
    assert loose["score"] == pytest.approx(strict["score"])
    assert strict["threshold"] == 0.99


def test_the_embedding_response_can_be_pasted_back_unchanged(client):
    """What a caller has in hand is whatever /v1/audio/speaker-embeddings gave
    it. Handing that back is the first thing anyone tries."""
    whole = {"object": "list", "model": "ecapa", "backend": "ecapa", "dim": DIM,
             "data": [{"object": "embedding", "index": 0, "embedding": VEC_A}]}
    r = _post(client, reference=json.dumps(whole))
    assert r.status_code == 200, r.text
    assert r.json()["same_speaker"] is True


def test_an_object_with_an_embedding_key_works(client):
    r = _post(client, reference=json.dumps({"embedding": VEC_A}))
    assert r.status_code == 200, r.text
    assert r.json()["same_speaker"] is True


def test_a_reference_from_another_model_is_refused_not_compared(client):
    """192-d ECAPA against a 512-d vector is not a near miss; it is a different
    space. Comparing the overlap would answer with confident noise."""
    r = _post(client, reference=json.dumps([0.1] * (DIM + 4)))
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert str(DIM + 4) in detail and str(DIM) in detail
    assert "different embedding model" in detail


def test_malformed_json_is_the_callers_error(client):
    r = _post(client, reference="[1.0, 2.0,")
    assert r.status_code == 400
    assert "not valid JSON" in r.json()["detail"]


@pytest.mark.parametrize("bad,expect", [
    (json.dumps(["a", "b"]), "not a number"),
    (json.dumps([]), "empty vector"),
    (json.dumps({"nope": 1}), "no 'embedding'"),
    (json.dumps("a string"), "JSON array of numbers"),
    (json.dumps([True, False]), "not a number"),
])
def test_every_malformed_reference_is_a_400(client, bad, expect):
    r = _post(client, reference=bad)
    assert r.status_code == 400, r.text
    assert expect in r.json()["detail"]


def test_a_reference_is_required(client):
    r = _post(client)
    assert r.status_code == 400
    assert "'name'" in r.json()["detail"] and "'reference'" in r.json()["detail"]


def test_naming_both_is_refused_rather_than_one_silently_winning(client):
    r = _post(client, name="ana", reference=json.dumps(VEC_A))
    assert r.status_code == 400
    assert "not both" in r.json()["detail"]


def test_the_enrolled_path_still_works(client):
    """The feature is additive: `name` must behave exactly as before."""
    speaker_registry.enroll("ana", VEC_A, "ecapa", "stub")
    r = _post(client, name="ana")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["same_speaker"] is True and body["name"] == "ana"
    assert body["reference"] == "enrolled" and body["backend"] == "ecapa"


def test_an_unknown_enrolled_name_is_still_a_404(client):
    r = _post(client, name="nobody")
    assert r.status_code == 404
    assert "nobody" in r.json()["detail"]


def test_a_bad_reference_is_refused_before_the_audio_is_embedded(monkeypatch, tmp_path):
    """Embedding a clip costs a GPU round-trip. A request that cannot succeed
    should not pay for one."""
    monkeypatch.setenv("CODERAI_SPEAKER_STORE", str(tmp_path / "speakers.json"))
    calls = []

    async def _counting_embed(file, backend, model):
        calls.append(1)
        return {"embedding": list(VEC_A), "backend": "ecapa", "model": "stub", "dim": DIM}

    monkeypatch.setattr(tr, "_embed_upload", _counting_embed)
    app = FastAPI()
    app.include_router(tr.router)
    c = TestClient(app)
    assert _post(c).status_code == 400
    assert _post(c, name="x", reference="[1]").status_code == 400
    assert calls == []
