# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""``POST /v1/moderation`` — screen generated images and video before release.

Four independent checks over an image, or over frames sampled from a video:

``nsfw``
    How explicit the picture is, from an image classifier, with the model's own
    label scores passed through.
``faces``
    Where the faces are, one entry per face per sampled frame.
``age``
    An age estimate per face, as a point estimate AND an interval. The interval
    is what decisions are made on; a point estimate alone invites treating
    "19" as settled.
``face_match``
    Whether each face matches one of a set of reference images — the consenting
    performer — by cosine similarity between face embeddings.

THE RULE THIS MODULE IS BUILT AROUND: it fails CLOSED. Every "safe" field in
the response -- ``minor_detected: false``, ``flagged: false``,
``unmatched_faces: 0`` -- is a statement that a model actually examined the
content and found nothing. If a model needed for a requested check is missing,
broken, or unsure, this returns an error or flags for review; it never returns
the reassuring value by default. A moderation endpoint that answers
``minor_detected: false`` because its age model failed to load is worse than
no endpoint at all, because a caller would ship on it.

Two consequences of that rule worth stating plainly, because they are not
oversights:

* An explicit image in which NO face was found cannot be cleared on age. There
  is nothing to estimate an age from, so the result carries
  ``review_required`` and ``flagged`` is true. A body without a face is exactly
  the case the caller most needs told.
* ``minor_detected`` keys on the LOW end of the age interval, not the point
  estimate. A face estimated at 21 ± 5 has a credible chance of being under
  18 and is flagged.

None of this is a legal opinion or a substitute for the caller's own duties;
it is a screen, and the response says what was measured and by which model so
the caller can tell the difference.
"""

import asyncio
import base64
import io
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from codai.api.loras import _require_api_auth

router = APIRouter()

#: The checks a caller may ask for. Anything else is a 400 rather than a
#: silently skipped check -- a typo'd check name must not read as "passed".
CHECKS = ("nsfw", "age", "faces", "face_match")

#: Default frames per second to sample from a video. One is plenty for
#: screening: an offending frame that appears for under a second still appears.
DEFAULT_SAMPLE_FPS = 1.0

#: Hard cap on frames examined per video, so a long upload cannot pin the GPU
#: indefinitely. Sampling is uniform across the whole duration when the cap
#: binds, never just the opening seconds.
MAX_FRAMES = 240

#: Below this, a face is too small for an age estimate to mean anything. Such
#: a face is still REPORTED (it is a face), with a null age and
#: ``too_small: true``, and it counts as "could not be assessed" rather than
#: "adult".
MIN_FACE_PX = 48

#: Cosine similarity at or above which two embeddings are the same person,
#: when the caller names no threshold. 0.6 or so is where buffalo_l's
#: recognition head separates identities; the caller can raise it.
DEFAULT_MATCH_THRESHOLD = 0.6

#: An age interval's half-width when the estimator gives only a point value.
#: insightface's genderage head reports a single number; treating it as exact
#: would defeat the interval, so a conservative band is applied. Published
#: evaluations of that head put its mean absolute error around 4-5 years, and
#: the band is wider than the MAE on purpose: this is the number a
#: minor/adult decision is made on.
POINT_ESTIMATE_BAND = 8


# =============================================================================
# Request / response models
# =============================================================================

class ModerationItem(BaseModel):
    type: str = "image"                       # image | video
    src: str                                  # /v1/files/..., URL, path or b64
    sample_fps: Optional[float] = None        # video only
    model_config = ConfigDict(extra="allow")


class FaceMatchSpec(BaseModel):
    reference: List[str] = Field(default_factory=list)
    threshold: float = DEFAULT_MATCH_THRESHOLD
    # Saved character profiles may stand in for reference images, so a caller
    # who has already enrolled a performer does not resend their photos.
    character_profiles: List[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="allow")


class ModerationRequest(BaseModel):
    input: List[ModerationItem] = Field(default_factory=list)
    checks: List[str] = Field(default_factory=lambda: ["nsfw", "age", "faces"])
    face_match: Optional[FaceMatchSpec] = None
    # Score at or above which the nsfw classifier's explicit labels flag an
    # item. Only affects `flagged`; the scores are always reported in full.
    nsfw_threshold: float = 0.7
    model_config = ConfigDict(extra="allow")


# =============================================================================
# Model access
# =============================================================================

class ModelUnavailable(RuntimeError):
    """A model a requested check needs is not usable.

    Raised rather than degraded, because the degraded answer for every check
    here is the dangerous one.
    """


_face_app = None
_nsfw_pipe = None
_mivolo = None


def _config_section() -> List[dict]:
    """The operator's ``moderation_models`` entries, or an empty list."""
    try:
        from codai.admin.routes import config_manager
        md = getattr(config_manager, "models_data", {}) or {}
        return [e for e in (md.get("moderation_models") or [])
                if isinstance(e, dict)]
    except Exception:
        return []


def _configured(role: str) -> Optional[dict]:
    """The configured entry for a role (``nsfw``, ``age``, ``face``)."""
    for e in _config_section():
        if str(e.get("role") or "").strip().lower() == role:
            return e
    return None


def available_models() -> List[dict]:
    """What ``/v1/models`` should publish as type ``moderation``.

    Each entry says which check it serves and whether it is actually loadable
    right now, so a caller can see a check is unavailable BEFORE sending
    content for it.
    """
    out = []
    for role, caps, default in (
        ("nsfw", ["moderation", "nsfw_detection"], _default_nsfw_model()),
        ("age", ["moderation", "age_estimation"], _default_age_model()),
        ("face", ["moderation", "face_detection", "face_embedding"], "buffalo_l"),
    ):
        cfg = _configured(role) or {}
        out.append({
            "id": cfg.get("id") or cfg.get("path") or default,
            "role": role,
            "type": "moderation",
            "capabilities": caps,
            "available": _role_available(role),
        })
    return out


def _default_nsfw_model() -> str:
    return os.environ.get("CODERAI_NSFW_MODEL") or "Falconsai/nsfw_image_detection"


def _default_age_model() -> str:
    # insightface's buffalo_l bundle carries a genderage head, and the image
    # already ships insightface for face swapping -- so age needs no extra
    # download by default. A configured MiVOLO entry overrides it and is the
    # better estimator where the operator has installed one.
    return "insightface:genderage"


#: Cached availability per role. Cleared by nothing: a model that appears is
#: picked up on the next process start, which is when its weights could have
#: arrived anyway.
_avail_cache: Dict[str, bool] = {}


def _weights_present(name: str) -> bool:
    """Whether these weights are on this machine, WITHOUT loading them.

    A local directory, or a HuggingFace repo already in the hub cache. The
    point is that this must not download and must not construct a pipeline.
    """
    if not name:
        return False
    if os.path.isdir(name) or os.path.isfile(name):
        return True
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(name, local_files_only=True)
        return True
    except Exception:
        return False


def _role_available(role: str) -> bool:
    """Whether a check could run, decided CHEAPLY.

    This must never load a model. :func:`available_models` is called from the
    ``/v1/models`` listing, and an earlier version of this function answered
    the question by actually constructing each pipeline -- so listing the
    models pulled a 350 MB classifier onto the GPU, on the event loop, every
    time anything asked what models exist. A test run caught it loading the
    nsfw classifier dozens of times.

    So: is the package importable, and are the weights already here. That can
    be wrong in one direction -- present weights that fail to load -- and that
    direction is safe, because the moderation call itself still fails closed
    with a 503 naming the model.
    """
    if role in _avail_cache:
        return _avail_cache[role]
    ok = False
    try:
        if role == "face":
            import importlib.util
            ok = importlib.util.find_spec("insightface") is not None
        elif role == "nsfw":
            import importlib.util
            ok = (importlib.util.find_spec("transformers") is not None
                  and _weights_present(
                      (_configured("nsfw") or {}).get("path")
                      or (_configured("nsfw") or {}).get("id")
                      or _default_nsfw_model()))
        elif role == "age":
            cfg = _configured("age") or {}
            name = str(cfg.get("path") or cfg.get("id") or "").strip()
            if not name or name.startswith("insightface"):
                import importlib.util
                ok = importlib.util.find_spec("insightface") is not None
            else:
                ok = _weights_present(name)
    except Exception:
        ok = False
    _avail_cache[role] = ok
    return ok


def _get_face_app():
    """insightface FaceAnalysis: detection, 512-d embedding and age/gender."""
    global _face_app
    if _face_app is not None:
        return _face_app
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise ModelUnavailable(
            "insightface is not installed, so faces cannot be detected. "
            "Install it (it is in the coderai image) or configure a "
            "moderation_models entry with role 'face'.") from e
    cfg = _configured("face") or {}
    name = cfg.get("path") or cfg.get("id") or "buffalo_l"
    try:
        app = FaceAnalysis(name=name, providers=[
            "CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as e:
        raise ModelUnavailable(f"face model {name!r} could not be loaded: {e}") from e
    _face_app = app
    return _face_app


def _get_nsfw():
    """An image classifier whose labels describe explicitness."""
    global _nsfw_pipe
    if _nsfw_pipe is not None:
        return _nsfw_pipe
    cfg = _configured("nsfw") or {}
    name = cfg.get("path") or cfg.get("id") or _default_nsfw_model()
    try:
        from transformers import pipeline
    except ImportError as e:
        raise ModelUnavailable("transformers is not installed") from e
    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except Exception:
        device = -1
    try:
        _nsfw_pipe = pipeline("image-classification", model=name, device=device)
    except Exception as e:
        raise ModelUnavailable(
            f"the nsfw classifier {name!r} could not be loaded: {e}. Download it "
            f"or configure a moderation_models entry with role 'nsfw'.") from e
    return _nsfw_pipe


def _get_age():
    """The age estimator: a configured MiVOLO, else insightface's own head."""
    global _mivolo
    cfg = _configured("age") or {}
    name = str(cfg.get("path") or cfg.get("id") or "").strip()
    if not name or name.startswith("insightface"):
        # The face app IS the estimator; it reports `age` per face.
        _get_face_app()
        return ("insightface", None)
    if _mivolo is not None:
        return _mivolo
    try:
        from transformers import pipeline
        _mivolo = ("pipeline", pipeline("image-classification", model=name))
    except Exception as e:
        raise ModelUnavailable(
            f"the age estimator {name!r} could not be loaded: {e}") from e
    return _mivolo


# =============================================================================
# Media handling
# =============================================================================

def _load_image(src: str):
    """Any reference a caller may send → a PIL RGB image."""
    from PIL import Image

    from codai.compose import media
    if not src:
        raise HTTPException(status_code=400, detail="an input item has no src")
    s = src.strip()
    if s.startswith("data:") or (len(s) > 512 and "/" not in s[:64]):
        raw = s.split(",", 1)[1] if s.startswith("data:") else s
        try:
            return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400,
                                detail=f"could not decode image data: {e}")
    with tempfile.TemporaryDirectory() as d:
        try:
            path = media.resolve(s, d, "mod")
            return Image.open(path).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400,
                                detail=f"could not fetch {s!r}: {e}")


def _sample_video(src: str, sample_fps: float, workdir: str) -> List[tuple]:
    """Frames from a video as ``[(frame_index, PIL image), ...]``.

    Sampled uniformly across the WHOLE duration. When the frame cap binds, the
    sample rate is reduced rather than the tail being dropped: screening only
    the first four minutes of a ten-minute video is not screening it.
    """
    from PIL import Image

    from codai.compose import media
    path = media.resolve(src, workdir, "modvid")
    try:
        info = media.probe(path)
        duration = float(info.get("duration") or 0.0)
    except Exception:
        duration = 0.0
    fps = max(0.01, float(sample_fps or DEFAULT_SAMPLE_FPS))
    if duration > 0 and duration * fps > MAX_FRAMES:
        fps = MAX_FRAMES / duration
    out_dir = os.path.join(workdir, "frames")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [media.ffmpeg_bin(), "-v", "error", "-i", path,
           "-vf", f"fps={fps:.6f}", "-vsync", "0",
           os.path.join(out_dir, "f%06d.png")]
    try:
        subprocess.run(cmd, capture_output=True, timeout=1800, check=True)
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"could not sample frames from {src!r}: {e}")
    frames = []
    for i, fn in enumerate(sorted(os.listdir(out_dir))):
        if not fn.endswith(".png"):
            continue
        # The frame number a caller can seek to, not the sample's ordinal.
        idx = int(round(i / fps * (float(info.get("fps") or fps) if duration else fps))) \
            if duration else i
        frames.append((idx, Image.open(os.path.join(out_dir, fn)).convert("RGB")))
    if not frames:
        raise HTTPException(status_code=400,
                            detail=f"no frames could be read from {src!r}")
    return frames


# =============================================================================
# The checks
# =============================================================================

#: Labels that mean "explicit", across the classifiers in common use. Matched
#: case-insensitively on the label text, so a model with its own vocabulary
#: still contributes as long as it says one of these words.
_EXPLICIT_LABELS = ("nsfw", "porn", "sexy", "hentai", "explicit", "nudity",
                    "nude", "suggestive")


def _check_nsfw(pipe, image) -> dict:
    raw = pipe(image)
    labels = {}
    for row in (raw or []):
        labels[str(row.get("label"))] = round(float(row.get("score") or 0.0), 4)
    score = 0.0
    for lab, sc in labels.items():
        if any(w in lab.lower() for w in _EXPLICIT_LABELS):
            score = max(score, sc)
    return {"score": round(score, 4), "labels": labels}


def _cosine(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype="float32").ravel()
    b = np.asarray(b, dtype="float32").ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(a.dot(b) / (na * nb))


def _reference_embeddings(spec: Optional[FaceMatchSpec]) -> List[Any]:
    """One embedding per reference image of the consenting performer."""
    import numpy as np
    if spec is None:
        return []
    srcs = list(spec.reference or [])
    # Each named profile is resolved separately and its failure is REPORTED.
    # This was a bare `except Exception: pass`, which turned every possible
    # cause -- a misspelt name, a profile with no image files, a
    # characters directory the server resolves elsewhere -- into the single
    # message "face_match was requested with no reference images". That told
    # the caller the opposite of the truth (they had supplied references) and
    # left no way to find out what went wrong; it cost a long debugging
    # session on a live server to establish which of those it was.
    problems: List[str] = []
    for name in (spec.character_profiles or []):
        try:
            from codai.api.characters import resolve_character_profiles
            got = resolve_character_profiles([name])
        except Exception as exc:
            problems.append(f"{name!r}: {type(exc).__name__}: {exc}")
            continue
        if not got:
            problems.append(
                f"{name!r}: no reference images (is the profile name right, "
                f"and does it have images saved?)")
            continue
        srcs += got
    if not srcs:
        detail = "face_match was requested with no usable reference images"
        if problems:
            detail += ": " + "; ".join(problems)
        elif not spec.character_profiles:
            detail = ("face_match was requested with neither `reference` nor "
                      "`character_profiles`")
        raise HTTPException(status_code=400, detail=detail)
    if problems:
        # Some resolved and some did not. Not fatal -- there ARE references to
        # match against -- but a performer the caller believes is enrolled and
        # is not would make a genuine face read as unmatched, so it is said.
        print(f"[moderation] some face_match profiles were unusable: "
              f"{'; '.join(problems)}", flush=True)
    app = _get_face_app()
    embs = []
    for i, s in enumerate(srcs):
        img = _load_image(s)
        faces = app.get(np.asarray(img)[:, :, ::-1])     # insightface wants BGR
        if not faces:
            # Not skipped quietly: a reference with no detectable face means
            # the caller believes a performer is enrolled who is not, and
            # every face would then read as "unmatched" for the wrong reason.
            raise HTTPException(
                status_code=400,
                detail=f"face_match reference #{i} has no detectable face")
        faces.sort(key=lambda f: -float(
            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
        embs.append(getattr(faces[0], "normed_embedding", None)
                    if getattr(faces[0], "normed_embedding", None) is not None
                    else faces[0].embedding)
    return embs


def _age_interval(face, age_mode) -> tuple:
    """``(estimate, low, high)`` for one face, or ``(None, None, None)``.

    The interval is the point of this function. insightface's head reports one
    number; a decision about a minor made on one number, from a model with a
    multi-year error, is a decision made on noise. So a band is applied and
    the LOW end is what the minor rule reads.
    """
    kind = age_mode[0] if isinstance(age_mode, tuple) else "insightface"
    if kind == "insightface":
        a = getattr(face, "age", None)
        if a is None:
            return (None, None, None)
        a = float(a)
        return (round(a, 1), max(0.0, round(a - POINT_ESTIMATE_BAND, 1)),
                round(a + POINT_ESTIMATE_BAND, 1))
    return (None, None, None)


def _assess_frame(image, frame_index, checks, pipes, refs, threshold) -> dict:
    """Every requested check over one frame."""
    import numpy as np
    out = {"frame": frame_index, "faces": [], "nsfw": None}

    if "nsfw" in checks:
        out["nsfw"] = _check_nsfw(pipes["nsfw"], image)

    if not ({"faces", "age", "face_match"} & set(checks)):
        return out

    app = pipes["face"]
    faces = app.get(np.asarray(image)[:, :, ::-1])
    for f in faces:
        x1, y1, x2, y2 = [float(v) for v in f.bbox]
        w, h = x2 - x1, y2 - y1
        rec = {"frame": frame_index,
               "box": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
               "det_score": round(float(getattr(f, "det_score", 0.0)), 4)}
        too_small = min(w, h) < MIN_FACE_PX
        if too_small:
            rec["too_small"] = True

        if "age" in checks:
            est, low, high = _age_interval(f, pipes.get("age"))
            if too_small:
                # Reported, not estimated: a 20-pixel face carries no age
                # information, and an estimate from one would be a number
                # with the authority of a measurement and the content of a
                # guess.
                est = low = high = None
            rec["age_estimate"] = est
            rec["age_low"] = low
            rec["age_high"] = high
            if low is None:
                rec["minor_probability"] = None
                rec["age_assessed"] = False
            else:
                rec["age_assessed"] = True
                # A credible chance of being under 18, expressed as how far
                # into the interval 18 falls. 1.0 when the whole interval is
                # below 18, 0.0 when the whole interval is above it.
                span = max(1e-6, (high or 0.0) - low)
                rec["minor_probability"] = round(
                    min(1.0, max(0.0, (18.0 - low) / span)), 4)

        if "face_match" in checks and refs:
            emb = (getattr(f, "normed_embedding", None)
                   if getattr(f, "normed_embedding", None) is not None
                   else f.embedding)
            best_i, best_s = -1, -1.0
            for i, r in enumerate(refs):
                sim = _cosine(emb, r)
                if sim > best_s:
                    best_i, best_s = i, sim
            rec["match"] = {"reference": best_i if best_s >= threshold else None,
                            "similarity": round(best_s, 4),
                            "matched": bool(best_s >= threshold)}
        out["faces"].append(rec)
    return out


def _combine(item_src: str, frames: List[dict], checks, nsfw_threshold) -> dict:
    """Roll every sampled frame up into one verdict for the item."""
    faces = [f for fr in frames for f in (fr.get("faces") or [])]
    res: Dict[str, Any] = {"src": item_src, "faces": faces,
                           "frames_examined": len(frames)}
    reasons: List[str] = []

    nsfw_score = 0.0
    if "nsfw" in checks:
        best = None
        for fr in frames:
            n = fr.get("nsfw") or {}
            if best is None or float(n.get("score") or 0) > float(best.get("score") or 0):
                best = n
        res["nsfw"] = best or {"score": 0.0, "labels": {}}
        nsfw_score = float(res["nsfw"].get("score") or 0.0)
        if nsfw_score >= float(nsfw_threshold):
            reasons.append(f"explicit content (score {nsfw_score:.2f})")

    if "age" in checks:
        assessed = [f for f in faces if f.get("age_assessed")]
        unassessed = [f for f in faces if f.get("age_assessed") is False]
        lows = [f["age_low"] for f in assessed if f.get("age_low") is not None]
        res["min_age_estimate"] = min(lows) if lows else None
        # Caution, deliberately: a credible chance, not a likely one.
        minor = any((f.get("minor_probability") or 0) > 0.0 for f in assessed)
        res["minor_detected"] = bool(minor)
        if minor:
            reasons.append("a face may be under 18")
        if unassessed:
            # A face that could not be aged is not an adult face.
            res["faces_not_assessed"] = len(unassessed)
            reasons.append(f"{len(unassessed)} face(s) too small to assess for age")
        if not faces and nsfw_score >= float(nsfw_threshold):
            # The case that matters most: explicit content with no face to
            # estimate an age from. It cannot be cleared, so it is not.
            res["minor_detected"] = None
            reasons.append("explicit content with no detectable face: age "
                           "could not be assessed")

    if "face_match" in checks:
        unmatched = [f for f in faces if not (f.get("match") or {}).get("matched")]
        res["unmatched_faces"] = len(unmatched)
        if unmatched:
            reasons.append(f"{len(unmatched)} face(s) match no consenting "
                           f"performer reference")

    res["review_required"] = bool(reasons)
    res["flagged"] = bool(reasons)
    res["reasons"] = reasons
    return res


# =============================================================================
# Endpoint
# =============================================================================

@router.post("/v1/moderation", summary="Screen images and video before release")
async def create_moderation(request: ModerationRequest,
                            _auth=Depends(_require_api_auth)):
    """Run the requested checks over every input item.

    Fails closed. A requested check whose model cannot be loaded is a **503**
    naming the model, not a result with the check quietly absent: a caller
    cannot tell a missing check from a passed one, and would ship on it.
    """
    if not request.input:
        raise HTTPException(status_code=400, detail="`input` is empty")
    checks = [str(c).strip().lower() for c in (request.checks or []) if str(c).strip()]
    unknown = [c for c in checks if c not in CHECKS]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"unknown check(s) {unknown}; valid: {list(CHECKS)}")
    if not checks:
        raise HTTPException(status_code=400, detail="`checks` is empty")
    if "face_match" in checks and not request.face_match:
        raise HTTPException(
            status_code=400,
            detail="face_match was requested without a `face_match` block")

    # Load every model the requested checks need BEFORE examining anything, so
    # a missing model is one clear error rather than a partial result.
    pipes: Dict[str, Any] = {}
    try:
        if "nsfw" in checks:
            pipes["nsfw"] = await asyncio.to_thread(_get_nsfw)
        if {"faces", "age", "face_match"} & set(checks):
            pipes["face"] = await asyncio.to_thread(_get_face_app)
        if "age" in checks:
            pipes["age"] = await asyncio.to_thread(_get_age)
    except ModelUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))

    threshold = float((request.face_match.threshold if request.face_match
                       else DEFAULT_MATCH_THRESHOLD) or DEFAULT_MATCH_THRESHOLD)
    refs = []
    if "face_match" in checks:
        refs = await asyncio.to_thread(_reference_embeddings, request.face_match)

    results = []
    for item in request.input:
        kind = str(item.type or "image").strip().lower()

        def _work(item=item, kind=kind):
            if kind == "video":
                with tempfile.TemporaryDirectory() as d:
                    frames = _sample_video(
                        item.src, item.sample_fps or DEFAULT_SAMPLE_FPS, d)
                    assessed = [_assess_frame(img, idx, checks, pipes, refs,
                                              threshold)
                                for idx, img in frames]
            else:
                assessed = [_assess_frame(_load_image(item.src), 0, checks,
                                          pipes, refs, threshold)]
            return _combine(item.src, assessed, checks, request.nsfw_threshold)

        results.append(await asyncio.to_thread(_work))

    return {
        "results": results,
        "flagged": any(r.get("flagged") for r in results),
        "checks": checks,
        "models": {m["role"]: m["id"] for m in available_models()
                   if m["role"] in ("nsfw", "age", "face")},
    }


@router.get("/v1/moderation/models", summary="Moderation models and availability")
async def moderation_models(_auth=Depends(_require_api_auth)):
    """Which check each model serves, and whether it can be loaded right now.

    Worth asking before sending content: an unavailable check is a 503 at
    moderation time, and a caller holding adult renders for manual review
    needs to know that in advance.
    """
    models = await asyncio.to_thread(available_models)
    return {"models": models,
            "unavailable": [m["role"] for m in models if not m["available"]]}
