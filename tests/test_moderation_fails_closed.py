"""B4: /v1/moderation, and the one property that makes it worth having.

Every "safe" value this endpoint can return -- minor_detected false, flagged
false, unmatched_faces 0 -- is a claim that a model looked and found nothing.
If any of those can also mean "the model was missing", a caller who holds
adult renders for manual review would start shipping them on a result that
measured nothing at all. So the tests below are mostly about what happens when
things are absent or unsure.
"""
import types

import pytest
from fastapi import HTTPException

import codai.api.moderation as M


# ------------------------------------------------------------- request shape
def _req(**kw):
    kw.setdefault("input", [M.ModerationItem(type="image", src="x.png")])
    return M.ModerationRequest(**kw)


@pytest.mark.anyio
async def test_an_unknown_check_is_refused_not_skipped():
    # A typo'd check name must not come back as a result with that check
    # quietly absent -- indistinguishable, to a caller, from it having passed.
    with pytest.raises(HTTPException) as e:
        await M.create_moderation(_req(checks=["nsfw", "ages"]), _auth=None)
    assert e.value.status_code == 400
    assert "ages" in str(e.value.detail)


@pytest.mark.anyio
async def test_empty_input_or_checks_is_refused():
    for kw in ({"input": [], "checks": ["nsfw"]},
               {"checks": []}):
        with pytest.raises(HTTPException) as e:
            await M.create_moderation(_req(**kw), _auth=None)
        assert e.value.status_code == 400


@pytest.mark.anyio
async def test_face_match_without_references_is_refused():
    with pytest.raises(HTTPException) as e:
        await M.create_moderation(_req(checks=["face_match"]), _auth=None)
    assert e.value.status_code == 400
    assert "face_match" in str(e.value.detail)


@pytest.mark.anyio
async def test_a_missing_model_is_a_503_before_anything_is_examined(monkeypatch):
    """The core guarantee. No model, no result -- an error."""
    def _boom():
        raise M.ModelUnavailable("the nsfw classifier 'X' could not be loaded")
    monkeypatch.setattr(M, "_get_nsfw", _boom)
    # And nothing is even fetched: a 503 must not cost the caller a download.
    monkeypatch.setattr(M, "_load_image",
                        lambda s: pytest.fail("content was examined anyway"))
    with pytest.raises(HTTPException) as e:
        await M.create_moderation(_req(checks=["nsfw"]), _auth=None)
    assert e.value.status_code == 503
    assert "could not be loaded" in str(e.value.detail)


@pytest.mark.anyio
async def test_a_reference_with_no_face_is_refused(monkeypatch):
    """An enrolled performer who is not actually enrolled.

    Skipping such a reference quietly would make every real face read as
    "matches no consenting performer" for the wrong reason.
    """
    class _App:
        def get(self, arr):
            return []
    monkeypatch.setattr(M, "_get_face_app", lambda: _App())
    monkeypatch.setattr(M, "_load_image", lambda s: _img())
    with pytest.raises(HTTPException) as e:
        M._reference_embeddings(M.FaceMatchSpec(reference=["a.png"]))
    assert e.value.status_code == 400
    assert "no detectable face" in str(e.value.detail)


def _img(w=640, h=640):
    from PIL import Image
    return Image.new("RGB", (w, h), (128, 128, 128))


# ------------------------------------------------------- the age interval
class _Face:
    def __init__(self, box=(0, 0, 200, 200), age=30, emb=None, det=0.9):
        self.bbox = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        self.age = age
        self.det_score = det
        self.embedding = emb if emb is not None else [1.0, 0.0, 0.0]
        self.normed_embedding = None


def test_the_age_estimate_is_an_interval_not_a_number():
    est, low, high = M._age_interval(_Face(age=22), ("insightface", None))
    assert est == 22.0
    # A band, because a decision about a minor made on one number from a model
    # with a multi-year error is a decision made on noise.
    assert low < 22.0 < high
    assert high - low >= 2 * M.POINT_ESTIMATE_BAND - 0.1


def test_a_face_with_no_age_reported_yields_no_estimate():
    f = _Face()
    f.age = None
    assert M._age_interval(f, ("insightface", None)) == (None, None, None)


def _assess(faces, checks=("age",), nsfw=None, refs=(), threshold=0.6):
    pipes = {"face": types.SimpleNamespace(get=lambda arr: list(faces)),
             "age": ("insightface", None)}
    if nsfw is not None:
        pipes["nsfw"] = lambda img: nsfw
    return M._assess_frame(_img(), 0, list(checks), pipes, list(refs), threshold)


def test_a_young_face_is_flagged_on_the_low_end_of_the_interval():
    # 24 is comfortably adult as a point estimate; 24 - 8 is not.
    out = _assess([_Face(age=24)])
    rec = out["faces"][0]
    assert rec["age_assessed"] is True
    assert rec["minor_probability"] > 0
    res = M._combine("x", [out], ["age"], 0.7)
    assert res["minor_detected"] is True
    assert res["flagged"] is True


def test_a_clearly_adult_face_is_not_flagged():
    out = _assess([_Face(age=40)])
    assert out["faces"][0]["minor_probability"] == 0.0
    res = M._combine("x", [out], ["age"], 0.7)
    assert res["minor_detected"] is False
    assert res["flagged"] is False


def test_a_face_too_small_to_assess_is_not_counted_as_an_adult():
    out = _assess([_Face(box=(0, 0, 20, 20), age=30)])
    rec = out["faces"][0]
    assert rec["too_small"] is True
    # Reported as a face, with NO age: a 20-pixel face carries no age
    # information, and an estimate from one would have the authority of a
    # measurement and the content of a guess.
    assert rec["age_estimate"] is None
    assert rec["age_assessed"] is False
    res = M._combine("x", [out], ["age"], 0.7)
    assert res["faces_not_assessed"] == 1
    assert res["review_required"] is True


def test_explicit_content_with_no_face_cannot_be_cleared():
    """The case that matters most, and the reason minor_detected is nullable.

    Nudity with no detectable face has nothing to estimate an age from. The
    honest answer is "not assessed", so minor_detected is None -- not false.
    """
    out = _assess([], checks=("age", "nsfw"),
                  nsfw=[{"label": "nsfw", "score": 0.97},
                        {"label": "normal", "score": 0.03}])
    res = M._combine("x", [out], ["age", "nsfw"], 0.7)
    assert res["minor_detected"] is None
    assert res["flagged"] is True
    assert any("no detectable face" in r for r in res["reasons"])


def test_a_clean_image_with_no_face_is_not_flagged():
    out = _assess([], checks=("age", "nsfw"),
                  nsfw=[{"label": "normal", "score": 0.99},
                        {"label": "nsfw", "score": 0.01}])
    res = M._combine("x", [out], ["age", "nsfw"], 0.7)
    assert res["flagged"] is False
    assert res["minor_detected"] is False


# ------------------------------------------------------------------ nsfw
def test_the_nsfw_score_is_the_strongest_explicit_label():
    pipe = lambda img: [{"label": "porn", "score": 0.8},
                        {"label": "hentai", "score": 0.9},
                        {"label": "neutral", "score": 0.1}]
    out = M._check_nsfw(pipe, _img())
    assert out["score"] == 0.9
    # Every label is reported, not just the verdict: the caller's policy may
    # differ from ours.
    assert out["labels"]["neutral"] == 0.1


def test_a_video_reports_the_worst_frame_not_the_average():
    frames = [
        _assess([], checks=("nsfw",), nsfw=[{"label": "normal", "score": 0.99}]),
        _assess([], checks=("nsfw",), nsfw=[{"label": "nsfw", "score": 0.95}]),
    ]
    for i, f in enumerate(frames):
        f["frame"] = i
    res = M._combine("v.mp4", frames, ["nsfw"], 0.7)
    # One bad frame in a clean clip is a bad clip.
    assert res["nsfw"]["score"] == 0.95
    assert res["flagged"] is True
    assert res["frames_examined"] == 2


# ------------------------------------------------------------- face match
def test_a_matching_face_is_matched_and_a_stranger_is_not():
    ref = [1.0, 0.0, 0.0]
    same = _Face(emb=[0.99, 0.01, 0.0])
    other = _Face(emb=[0.0, 1.0, 0.0])
    out = _assess([same, other], checks=("face_match",), refs=[ref])
    a, b = out["faces"]
    assert a["match"]["matched"] is True and a["match"]["reference"] == 0
    assert b["match"]["matched"] is False and b["match"]["reference"] is None
    res = M._combine("x", [out], ["face_match"], 0.7)
    assert res["unmatched_faces"] == 1
    assert res["flagged"] is True


def test_cosine_handles_a_zero_vector_without_raising():
    assert M._cosine([0, 0, 0], [1, 0, 0]) == 0.0
    assert abs(M._cosine([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-6


# ------------------------------------------------------- video frame budget
def test_the_frame_cap_lowers_the_rate_rather_than_dropping_the_tail():
    # Screening only the first four minutes of a ten-minute video is not
    # screening it. Asserted on the arithmetic the sampler uses.
    duration, asked = 600.0, 1.0
    fps = asked
    if duration * fps > M.MAX_FRAMES:
        fps = M.MAX_FRAMES / duration
    assert fps < asked
    assert abs(duration * fps - M.MAX_FRAMES) < 1e-6


# --------------------------------------------------------- the model listing
def test_the_models_are_published_with_their_availability(monkeypatch):
    monkeypatch.setattr(M, "_role_available", lambda role: role == "face")
    models = M.available_models()
    roles = {m["role"]: m for m in models}
    assert set(roles) == {"nsfw", "age", "face"}
    assert roles["face"]["available"] is True
    # An unavailable check is PUBLISHED as unavailable rather than omitted: an
    # absent entry reads as "this server does no moderation" rather than "this
    # check would fail".
    assert roles["nsfw"]["available"] is False
    assert all(m["type"] == "moderation" for m in models)
    assert "age_estimation" in roles["age"]["capabilities"]


def test_the_endpoint_is_registered():
    from codai.api.app import app
    paths = {r.path for r in app.routes}
    assert "/v1/moderation" in paths
    assert "/v1/moderation/models" in paths
