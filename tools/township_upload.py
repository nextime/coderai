"""Township Combat League upload helpers for the township fighter generator.

This module is intentionally self-contained and free of any dependency on the
big ``gen_township_fighters`` script so it can be unit-tested on its own. It
provides four pieces of functionality used by the web UI / CLI:

* ``generate_odds`` — produce a full set of betting odds for a match within the
  league's allowed ranges, guaranteed to survive the server's anti-arbitrage
  ("sure bet") checks. It samples within the ranges, runs the exact same checks
  the server runs, and retries up to ``max_tries`` times.
* ``check_arbitrage`` — a faithful re-implementation of the server's
  ``SureBetAnalyzer`` so the generator never ships odds the server would reject.
* ``build_match_zip`` — collect the nine outcome videos for a match, rename them
  to the canonical names the server expects (OVER/UNDER/WIN1/…/DRAW) using the
  highest-quality enhanced variant available, and pack them into a ZIP.
* ``upload_match`` — create the match on the server, push its ZIP in proxy-safe
  chunks, and finalize it so it goes live.

The two extractions and their payout matrix (mirrored from the server):

* First extraction:  ``under`` / ``over`` (exactly one wins).
* Second extraction: ``win1, win2, ko1, ko2, ret1, ret2, draw`` where
    - ko1 → fighter 2 wins by KO   (pays ko1 and win2)
    - ko2 → fighter 1 wins by KO   (pays ko2 and win1)
    - ret1 → fighter 2 wins by retirement (pays ret1 and win2)
    - ret2 → fighter 1 wins by retirement (pays ret2 and win1)
    - win1 → fighter 1 wins on points
    - win2 → fighter 2 wins on points
    - draw → draw
"""

from __future__ import annotations

import os
import random
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import requests


# ---------------------------------------------------------------------------
# Odds ranges (inclusive), in the order the league specifies. Values are always
# rounded to two decimals.
# ---------------------------------------------------------------------------
ODDS_RANGES: Dict[str, Tuple[float, float]] = {
    "under": (1.00, 2.00),
    "over": (1.00, 2.00),
    "win1": (1.00, 3.50),
    "win2": (1.00, 3.50),
    "ko1": (2.50, 7.00),
    "ko2": (2.50, 7.00),
    "ret1": (2.50, 7.00),
    "ret2": (2.50, 7.00),
    "draw": (1.50, 5.50),
}

OUTCOME_COLUMNS: List[str] = list(ODDS_RANGES.keys())


def default_ranges() -> Dict[str, Tuple[float, float]]:
    """A fresh copy of the built-in odds ranges."""
    return {k: (lo, hi) for k, (lo, hi) in ODDS_RANGES.items()}


def _range_dict_to_json(ranges: Dict[str, Tuple[float, float]]) -> Dict[str, List[float]]:
    """Convert a range map of tuples to a JSON-friendly map of ``[min, max]``
    lists (so it round-trips cleanly through config files)."""
    return {col: [round(float(lo), 2), round(float(hi), 2)]
            for col, (lo, hi) in ranges.items()}


def merge_ranges(overrides: Optional[Dict]) -> Dict[str, Tuple[float, float]]:
    """Merge user-supplied range overrides onto the defaults.

    ``overrides`` may be a mapping of ``column -> [min, max]`` (or ``(min, max)``);
    unknown columns and malformed pairs are ignored so a partial/old config stays
    usable. Returns a complete, validated range map (min <= max, all columns).
    """
    ranges = default_ranges()
    if isinstance(overrides, dict):
        for col, pair in overrides.items():
            if col not in ranges:
                continue
            try:
                lo, hi = float(pair[0]), float(pair[1])
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            if lo > hi:
                lo, hi = hi, lo
            ranges[col] = (round(lo, 2), round(hi, 2))
    return ranges


# ---------------------------------------------------------------------------
# Anti-arbitrage ("sure bet") checks — mirror of app/utils/sure_bet_analyzer.py
# ---------------------------------------------------------------------------
MIN_SURE_BET_ODDS_FIRST = 2.0    # under/over: sure bet if BOTH > this
MIN_SURE_BET_ODDS_SECOND = 3.0   # win1/win2/draw: sure bet if ALL > this
MIN_SURE_BET_PRODUCT = 4.0       # 2-outcome scenarios: sure bet if product > this


def check_arbitrage(odds: Dict[str, float]) -> Tuple[bool, str]:
    """Return ``(ok, reason)``. ``ok`` is True when the odds are SAFE (no sure
    bet). ``reason`` describes the first failing check when not ok.

    This replicates every check the server's ``SureBetAnalyzer.analyze_match``
    performs, using the same strict ``>`` comparisons so anything we accept the
    server will too.
    """
    under = odds.get("under")
    over = odds.get("over")
    win1 = odds.get("win1")
    win2 = odds.get("win2")
    ko1 = odds.get("ko1")
    ko2 = odds.get("ko2")
    ret1 = odds.get("ret1")
    ret2 = odds.get("ret2")
    draw = odds.get("draw")

    # First extraction: under/over.
    if under and over and under > MIN_SURE_BET_ODDS_FIRST and over > MIN_SURE_BET_ODDS_FIRST:
        return False, f"under ({under}) and over ({over}) both > {MIN_SURE_BET_ODDS_FIRST}"

    # Second extraction: win1/win2/draw all above threshold (one always wins).
    if (win1 and win2 and draw and win1 > MIN_SURE_BET_ODDS_SECOND
            and win2 > MIN_SURE_BET_ODDS_SECOND and draw > MIN_SURE_BET_ODDS_SECOND):
        return False, (f"win1 ({win1}), win2 ({win2}) and draw ({draw}) all "
                       f"> {MIN_SURE_BET_ODDS_SECOND}")

    # Two-outcome scenarios: product must not exceed the threshold.
    pairs = [
        ("ko1", ko1, "win2", win2),
        ("ko2", ko2, "win1", win1),
        ("ret1", ret1, "win2", win2),
        ("ret2", ret2, "win1", win1),
    ]
    for an, a, bn, b in pairs:
        if a and b and a * b > MIN_SURE_BET_PRODUCT:
            return False, f"{an} ({a}) * {bn} ({b}) = {a * b:.2f} > {MIN_SURE_BET_PRODUCT}"

    # Matrix safety margin: must be > 1.0.
    if all(v for v in (win1, win2, ko1, ko2, ret1, ret2, draw)):
        margin = (1.0 / draw
                  + 1.0 / win1 + 1.0 / ko2 + 1.0 / ret2
                  + 1.0 / win2 + 1.0 / ko1 + 1.0 / ret1)
        if margin <= 1.0:
            return False, f"matrix safety margin {margin:.4f} <= 1.0"

    return True, "ok"


def _r2(lo: float, hi: float, rng: random.Random) -> float:
    """Uniform sample in [lo, hi] rounded to two decimals (clamped into range)."""
    if hi < lo:
        hi = lo
    v = round(rng.uniform(lo, hi), 2)
    return min(max(v, round(lo, 2)), round(hi, 2))


def generate_odds(seed: Optional[int] = None, max_tries: int = 10,
                  ranges: Optional[Dict] = None) -> Dict[str, float]:
    """Generate a full, arbitrage-safe set of odds.

    Sampling is constraint-aware so a valid set is usually found on the first
    try: the KO/RET odds are drawn first, then each win odd is capped so its
    product with every coupled KO/RET stays at/under the threshold. After
    rounding the result is verified with :func:`check_arbitrage` exactly as the
    server would, and re-rolled up to ``max_tries`` times if rounding nudged a
    value over the line.

    Raises ``RuntimeError`` if no safe set is found within ``max_tries``.
    """
    rng = random.Random(seed)
    last_reason = "no attempts made"
    rg = merge_ranges(ranges)

    # The win and KO/RET odds are coupled by the product cap: win1 pairs with
    # ko2/ret2 and win2 pairs with ko1/ret1, and each product must stay at/under
    # MIN_SURE_BET_PRODUCT. Because the win minimum is ~1.0, a coupled KO/RET of
    # the range maximum (e.g. 7.0) can NEVER avoid a sure bet — so the usable
    # KO/RET ceiling is driven by the chosen win odd. We therefore draw the wins
    # first (low enough that a KO/RET at its floor still fits) and then bound each
    # KO/RET by the product cap. A small safety margin absorbs 2-decimal rounding.
    safe_product = MIN_SURE_BET_PRODUCT - 0.05
    ko_ret_lo = min(rg["ko1"][0], rg["ko2"][0], rg["ret1"][0], rg["ret2"][0])
    # Largest win that still leaves room for a KO/RET at its lowest floor.
    win_cap1 = min(rg["win1"][1], safe_product / ko_ret_lo)
    win_cap2 = min(rg["win2"][1], safe_product / ko_ret_lo)

    for _ in range(max_tries):
        win1 = _r2(rg["win1"][0], win_cap1, rng)
        win2 = _r2(rg["win2"][0], win_cap2, rng)

        ko1 = _r2(rg["ko1"][0], min(rg["ko1"][1], safe_product / win2), rng)
        ret1 = _r2(rg["ret1"][0], min(rg["ret1"][1], safe_product / win2), rng)
        ko2 = _r2(rg["ko2"][0], min(rg["ko2"][1], safe_product / win1), rng)
        ret2 = _r2(rg["ret2"][0], min(rg["ret2"][1], safe_product / win1), rng)

        draw = _r2(*rg["draw"], rng)
        under = _r2(*rg["under"], rng)
        over = _r2(*rg["over"], rng)

        odds = {
            "under": under, "over": over,
            "win1": win1, "win2": win2,
            "ko1": ko1, "ko2": ko2,
            "ret1": ret1, "ret2": ret2,
            "draw": draw,
        }
        ok, reason = check_arbitrage(odds)
        if ok:
            return odds
        last_reason = reason

    raise RuntimeError(
        f"could not generate arbitrage-safe odds in {max_tries} tries "
        f"(last failure: {last_reason})"
    )


# ---------------------------------------------------------------------------
# ZIP packing
# ---------------------------------------------------------------------------
# Township internal outcome name -> server column / file basename, per fighter.
# KO1/RET1 mean fighter 2 wins; KO2/RET2 mean fighter 1 wins (server matrix).

def _video_variants(p: Path) -> List[Tuple[int, Path]]:
    """Return [(weight, path)] for a base video and its enhanced siblings,
    weakest first. Mirrors the generator's variant detection (``_2x`` upscales,
    ``_2xfps`` interpolation, combinations thereof)."""
    import re
    out: List[Tuple[int, Path]] = [(0, p)]
    stem = p.stem
    for sib in p.parent.glob(f"{stem}_*{p.suffix}"):
        if sib == p:
            continue
        suf = sib.stem[len(stem) + 1:]
        toks = suf.split("_")
        if not toks or not all(re.fullmatch(r"\d+x(fps)?", t) for t in toks):
            continue
        weight = 0
        for t in toks:
            n = int(re.match(r"\d+", t).group())
            weight += n if t.endswith("xfps") else n * 10
        out.append((weight, sib))
    out.sort(key=lambda t: t[0])
    return out


def _best_variant(base: Path) -> Optional[Path]:
    """Highest-quality existing variant for a base video path, or None if the
    base file itself does not exist."""
    if not base.exists():
        return None
    variants = _video_variants(base)
    return variants[-1][1] if variants else base


def resolve_match_videos(out_dir, match_name: str, f1: str, f2: str) -> Tuple[Dict[str, Path], List[str]]:
    """Map each server ZIP filename to the best local video for the match.

    Returns ``(found, missing)`` where ``found`` maps ``"OVER.mp4" -> Path`` and
    ``missing`` lists the ZIP names with no source video.
    """
    vdir = Path(out_dir) / "videos"

    # The draw is stored once per match under one fighter's name; find whichever.
    draw_src = None
    for cand in sorted(vdir.glob(f"{match_name}_*_draw.mp4")):
        # Exclude enhanced siblings (handled by _best_variant); take the base.
        suf = cand.stem[len(match_name) + 1:]
        # base draw files end with "_draw" exactly (no trailing _NxfpS tokens).
        if suf.endswith("_draw"):
            draw_src = cand
            break

    spec = {
        "OVER.mp4": vdir / f"{match_name}_long.mp4",
        "UNDER.mp4": vdir / f"{match_name}_short.mp4",
        "WIN1.mp4": vdir / f"{match_name}_{f1}_win.mp4",
        "WIN2.mp4": vdir / f"{match_name}_{f2}_win.mp4",
        "KO1.mp4": vdir / f"{match_name}_{f2}_ko_win.mp4",
        "KO2.mp4": vdir / f"{match_name}_{f1}_ko_win.mp4",
        "RET1.mp4": vdir / f"{match_name}_{f2}_retire.mp4",
        "RET2.mp4": vdir / f"{match_name}_{f1}_retire.mp4",
        "DRAW.mp4": draw_src,
    }

    found: Dict[str, Path] = {}
    missing: List[str] = []
    for name, base in spec.items():
        best = _best_variant(base) if base else None
        if best is not None and best.exists():
            found[name] = best
        else:
            missing.append(name)
    return found, missing


def match_video_signature(out_dir, match_name: str, f1: str, f2: str) -> str:
    """A stable signature over the match's source videos (size + mtime of the
    best variant for each ZIP slot). Changes whenever any video is re-rendered
    or re-enhanced, so a stored 'uploaded' state can be invalidated."""
    import hashlib
    found, missing = resolve_match_videos(out_dir, match_name, f1, f2)
    h = hashlib.sha1()
    for name in sorted(set(list(found.keys()) + missing)):
        p = found.get(name)
        if p is not None:
            st = p.stat()
            h.update(f"{name}:{p.name}:{st.st_size}:{int(st.st_mtime)}".encode())
        else:
            h.update(f"{name}:MISSING".encode())
    return h.hexdigest()


def build_match_zip(out_dir, match_name: str, f1: str, f2: str,
                    dest_zip) -> Tuple[bool, List[str]]:
    """Pack the match's nine renamed videos into ``dest_zip``.

    Returns ``(ok, missing)``. ``ok`` is True only when all nine outcome videos
    were found and written. When some are missing the ZIP is still written with
    whatever exists so partial review is possible, but ``ok`` is False.
    """
    found, missing = resolve_match_videos(out_dir, match_name, f1, f2)
    dest_zip = Path(dest_zip)
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_zip.with_suffix(dest_zip.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, path in found.items():
            zf.write(path, arcname=name)
    tmp.replace(dest_zip)
    return (len(missing) == 0), missing


# ---------------------------------------------------------------------------
# Upload — create match, chunk the ZIP (proxy-safe), finalize
# ---------------------------------------------------------------------------
def upload_match(endpoint: str, token: str, fixture_id: str,
                 meta: Dict[str, str], odds: Dict[str, float], zip_path,
                 progress_cb: Optional[Callable[[float, str], None]] = None,
                 chunk_size: int = 4 * 1024 * 1024,
                 timeout: int = 600) -> Dict:
    """Upload one prepared match to the Township Combat League server.

    ``meta`` must contain ``fighter1_township``, ``fighter2_township`` and
    ``venue_kampala_township`` (optionally ``start_time``/``end_time`` ISO
    strings). ``odds`` is the column→float map. The ZIP is uploaded in chunks so
    it survives request-size limits on reverse proxies.

    Returns the server's finalize JSON (includes ``match_id``/``match_number``).
    Raises ``RuntimeError`` on any failure.
    """
    def _emit(frac: float, label: str):
        if progress_cb:
            try:
                progress_cb(frac, label)
            except Exception:
                pass

    base = endpoint.rstrip("/")
    headers = {"Authorization": f"Bearer {token}"}
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise RuntimeError(f"ZIP not found: {zip_path}")

    # 1) Create the match (carries the odds).
    _emit(0.0, "creating match…")
    body = {
        "fighter1_township": meta["fighter1_township"],
        "fighter2_township": meta["fighter2_township"],
        "venue_kampala_township": meta["venue_kampala_township"],
        "outcomes": {k: float(v) for k, v in odds.items()},
    }
    for opt in ("start_time", "end_time", "result"):
        if meta.get(opt):
            body[opt] = meta[opt]

    r = requests.post(f"{base}/api/fixture/{fixture_id}/match",
                      json=body, headers=headers, timeout=timeout)
    data = _json_or_raise(r, "create match")
    match_id = data.get("match_id")
    if not match_id:
        raise RuntimeError(f"create match: no match_id in response ({data})")

    # 2) Upload the ZIP in chunks.
    upload_id = uuid.uuid4().hex
    file_name = zip_path.name
    total = zip_path.stat().st_size
    total_chunks = max(1, (total + chunk_size - 1) // chunk_size)
    sent = 0
    with open(zip_path, "rb") as fh:
        for idx in range(total_chunks):
            blob = fh.read(chunk_size)
            files = {"chunk": (f"chunk_{idx}", blob, "application/octet-stream")}
            form = {
                "chunkIndex": str(idx),
                "totalChunks": str(total_chunks),
                "uploadId": upload_id,
                "fileName": file_name,
            }
            cr = requests.post(
                f"{base}/api/fixture/match/{match_id}/zip/chunk",
                data=form, files=files, headers=headers, timeout=timeout)
            _json_or_raise(cr, f"chunk {idx + 1}/{total_chunks}")
            sent += len(blob)
            _emit(0.05 + 0.85 * (sent / total if total else 1.0),
                  f"uploading {sent // (1024 * 1024)}/{total // (1024 * 1024)} MB")

    # 3) Finalize → assemble + go live.
    _emit(0.95, "finalizing…")
    fr = requests.post(
        f"{base}/api/fixture/match/{match_id}/zip/finalize",
        json={"uploadId": upload_id, "fileName": file_name},
        headers=headers, timeout=timeout)
    result = _json_or_raise(fr, "finalize")
    _emit(1.0, "done")
    result.setdefault("match_id", match_id)
    return result


def _json_or_raise(resp: "requests.Response", what: str) -> Dict:
    try:
        data = resp.json()
    except ValueError:
        data = {}
    if resp.status_code >= 400 or (isinstance(data, dict) and data.get("success") is False):
        msg = (data.get("error") or data.get("details")
               or f"HTTP {resp.status_code}") if isinstance(data, dict) else f"HTTP {resp.status_code}"
        raise RuntimeError(f"{what} failed: {msg}")
    return data if isinstance(data, dict) else {}
