"""What a video model can actually emit — frame rate, legal frame counts, continuation.

Three places already knew a piece of this and none of them could tell a client:
``codai/api/video.py:_snap_wan_frames`` (the Wan VAE's 4k+1 grid), ``tools/h3_common``
(MiniMax-H3's 17n+5 clip length) and ``tools/longcat_common`` (LongCat's 93-frame
segment, which a continuation extends by 80). So a caller asking for 250 frames was
silently rounded, and a caller trying to PLAN around the model — how long one scene
can be, how many scenes fill a minute — had no choice but to hardcode a table per
family and keep it in sync by hand. ``tools/gen_township_fighters.py`` did exactly
that, and drifted.

The geometry therefore lives here once, is published per model on ``/v1/models``
(``ModelInfo.video``), and is what the server itself snaps with.

A family's legal TOTAL frame counts are ``frame_base + frame_step * k`` for k >= 0:

  wan      1 + 4k    the VAE's 4k+1. One render holds ~81 frames; a longer clip is
                     chained client-side from each part's last frame.
  longcat  93 + 80k  one 93-frame segment, and each continuation re-renders 13 of
                     its frames so it only ADDS 80 (``longcat_common.frames_for``).
                     80 divides by 16, which is why every point on this grid is also
                     a count block-sparse attention can tile (16n+13): snapping to
                     the segment grid and keeping BSA on are the same act.
  h3       5 + 17k   the video VAE's clip length, clamped to 5-15 s.

The numbers are literals rather than imports because ``codai`` runs on a different
Python than the LongCat and H3 workers (3.10 venvs that cannot import ``codai.*``),
and loading their modules by path just to read six constants would drag their
dependencies into every ``/v1/models`` call. ``tests/test_video_geometry.py`` pins
each one against its source of truth so the duplication cannot drift silently.
"""

from typing import Optional

# Families, most specific first: a model is matched by the engine predicate the
# generation path itself uses (so geometry and routing can never disagree), and
# only then by name.
FAMILIES = {
    "longcat": {
        "family": "longcat",
        "label": "LongCat-Video",
        # longcat_common: DEFAULT_FPS, DEFAULT_NUM_FRAMES, DEFAULT_COND_FRAMES
        "native_fps": 15,
        "segment_frames": 93,
        "cond_frames": 13,
        "frame_base": 93,
        "frame_step": 80,
        "min_frames": 93,
        "max_frames": None,            # the server runs the segment loop; no ceiling
        "max_frames_per_render": 93,
        "default_frames": 93,
        # Block-sparse attention tiles the latent in (4, 4, 4) chunks over a 16px
        # VAE cell, so a side that is not a multiple of 64 silently falls back to
        # dense attention. 16 would decode; 64 is what stays fast.
        "side_multiple": 64,
        "continuation": "native",
    },
    "h3": {
        "family": "h3",
        "label": "MiniMax-H3",
        # h3_common: H3_FPS, H3_FRAMES_CHUNK, H3_FRAMES_REMAINDER, H3_MIN/MAX_SECONDS
        "native_fps": 24,
        "segment_frames": None,
        "cond_frames": 0,
        "frame_base": 5,
        "frame_step": 17,
        "min_frames": 124,             # ceil(5 s) onto the grid
        "max_frames": 345,             # floor(15 s) onto the grid
        "max_frames_per_render": 345,
        "default_frames": 124,
        "side_multiple": 32,           # H3_CANVAS_MULTIPLE
        "continuation": "none",
    },
    "wan": {
        "family": "wan",
        "label": "Wan2.x",
        "native_fps": 16,
        "segment_frames": None,
        "cond_frames": 0,
        "frame_base": 1,
        "frame_step": 4,
        "min_frames": 5,
        "max_frames": None,            # chained client-side, so no total ceiling
        # Wan2.2-A14B holds coherence to ~81 frames in ONE call; past that the
        # frames visibly jump, which is why longer clips are chained.
        "max_frames_per_render": 81,
        "default_frames": None,        # the request decides; the server has no one default
        "side_multiple": 16,
        "continuation": "chained",
    },
}

# The family used when nothing matches. Wan is the historical shape of every
# diffusers video pipeline the server runs, and its 4k+1 is the common VAE rule.
DEFAULT_FAMILY = "wan"

# models.json keys that override a published field. A new checkpoint whose family
# this table has never heard of is configurable rather than wrong — config is the
# source of truth, here as everywhere else.
CONFIG_OVERRIDES = {
    "video_native_fps": ("native_fps", int),
    "video_frame_base": ("frame_base", int),
    "video_frame_step": ("frame_step", int),
    "video_min_frames": ("min_frames", int),
    "video_max_frames": ("max_frames", int),
    "video_max_frames_per_render": ("max_frames_per_render", int),
    "video_side_multiple": ("side_multiple", int),
    "video_segment_frames": ("segment_frames", int),
    "video_cond_frames": ("cond_frames", int),
}


def family_for(model_name: str = "", config: dict = None) -> str:
    """Which geometry family serves this model.

    Asks the workers' own predicates first so a model routed to the LongCat worker
    can never be described with Wan's geometry — the backend pin, the alias and the
    path all count, exactly as they do when the request is dispatched.
    """
    cfg = config or {}
    try:
        from codai.api import longcat_worker
        if longcat_worker.is_longcat_model(model_name, cfg):
            return "longcat"
    except Exception:
        if "longcat" in str(model_name or "").lower():
            return "longcat"
    try:
        from codai.api import h3_worker
        if h3_worker.is_h3_model(model_name, cfg):
            return "h3"
    except Exception:
        if "minimax-h3" in str(model_name or "").lower().replace("_", "-"):
            return "h3"
    return DEFAULT_FAMILY


def geometry_for(model_name: str = "", config: dict = None) -> dict:
    """The published geometry for one video model: family defaults + config overrides."""
    cfg = config or {}
    raw = cfg.get("_raw_cfg") if isinstance(cfg.get("_raw_cfg"), dict) else None
    merged = dict(raw or {})
    for k, v in cfg.items():
        if k != "_raw_cfg":
            merged.setdefault(k, v)
    geom = dict(FAMILIES[family_for(model_name, merged)])
    for key, (field, cast) in CONFIG_OVERRIDES.items():
        if merged.get(key) in (None, ""):
            continue
        try:
            geom[field] = cast(merged[key])
        except (TypeError, ValueError):
            continue
    return geom


def snap_frames(num_frames, geom: dict, mode: str = "nearest") -> int:
    """``num_frames`` moved onto the family's legal grid and into its limits.

    ``mode`` "nearest" is for a planner choosing a length (a 2-frame move either way
    is not worth a whole extra segment); "up" is for honouring a caller who asked for
    at least this much. Either way the result is a count the VAE can decode — asking
    for one it cannot is how a request comes back a different length than it ordered.
    """
    base = int(geom.get("frame_base") or 1)
    step = max(1, int(geom.get("frame_step") or 1))
    lo = int(geom.get("min_frames") or base)
    hi = geom.get("max_frames")
    try:
        n = int(num_frames)
    except (TypeError, ValueError):
        n = lo
    k = (n - base) / float(step)
    k = int(-(-k // 1)) if mode == "up" else int(round(k))
    out = base + step * max(0, k)
    while out < lo:
        out += step
    if hi:
        while out > int(hi):
            out -= step
    return max(base, out)


def segments_for(total_frames, geom: dict) -> int:
    """How many model calls a total of ``total_frames`` takes on this family.

    LongCat's segment loop runs server-side (one request, native continuation); Wan's
    chaining runs in the client. The arithmetic is the same either way: the first call
    yields ``max_frames_per_render`` and each later one adds that minus the tail it
    re-renders.
    """
    per_call = int(geom.get("max_frames_per_render") or 0)
    if per_call <= 0:
        return 1
    added = max(1, per_call - int(geom.get("cond_frames") or 0))
    try:
        total = int(total_frames)
    except (TypeError, ValueError):
        return 1
    if total <= per_call:
        return 1
    return 1 + (total - per_call + added - 1) // added


def frames_for(segments, geom: dict) -> int:
    """The inverse of :func:`segments_for` — what ``segments`` calls actually yield."""
    per_call = int(geom.get("max_frames_per_render") or 0) or 1
    added = max(1, per_call - int(geom.get("cond_frames") or 0))
    n = max(1, int(segments or 1))
    return per_call + (n - 1) * added


def seconds_to_frames(seconds, geom: dict, fps: Optional[int] = None,
                      mode: str = "nearest") -> int:
    """A duration in seconds as a legal frame count at ``fps`` (default: the model's own).

    Seconds are the honest unit for a plan: a scene is "about twelve seconds", and
    pinning frames instead means the duration silently changes the moment the output
    rate does.
    """
    rate = int(fps or geom.get("native_fps") or 16)
    try:
        want = float(seconds) * max(1, rate)
    except (TypeError, ValueError):
        want = 0
    return snap_frames(want, geom, mode=mode)
