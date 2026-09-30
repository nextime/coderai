# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Captions: when each word is said, how the lines break, what libass draws.

Two rules from the spec shape all of this. **The text shown is the text the
caller gave** — speech recognition is used only to find out *when* each word is
spoken, never to rewrite it; a mis-heard word must not reach the screen. And
**the lines stay out of the platform's furniture**: a safe area of ~10% top and
bottom, ~6% at the sides, which is where TikTok, Reels and Shorts put their own
buttons.

Word timings come from one of two places:

* ``stt`` — word timestamps from a speech model transcribing the narration we
  just synthesised, aligned onto the caller's words (the transcript's words are
  matched positionally, so a mis-heard word still gets its slot);
* ``estimate`` — a duration per word proportional to its length, which is what
  you get with no STT model configured and is good enough for a 4-words-a-line
  reel.
"""

import os
import re
from typing import List, Optional

# Presets: the look, not the position. Each is a small set of ASS style values
# plus whether the active word is highlighted word-by-word.
PRESETS = {
    "karaoke":  {"karaoke": True,  "bold": True,  "outline": 4.0, "shadow": 1.0,
                 "back": 0.0, "scale": 1.00, "spacing": 0.0, "upper": False},
    "bold":     {"karaoke": True,  "bold": True,  "outline": 5.0, "shadow": 0.0,
                 "back": 0.0, "scale": 1.08, "spacing": 0.5, "upper": True},
    "classic":  {"karaoke": False, "bold": False, "outline": 2.5, "shadow": 1.0,
                 "back": 0.0, "scale": 0.92, "spacing": 0.0, "upper": False},
    "boxed":    {"karaoke": False, "bold": True,  "outline": 0.0, "shadow": 0.0,
                 "back": 0.75, "scale": 0.95, "spacing": 0.0, "upper": False},
    "neon":     {"karaoke": True,  "bold": True,  "outline": 3.0, "shadow": 3.0,
                 "back": 0.0, "scale": 1.02, "spacing": 1.0, "upper": True},
    "minimal":  {"karaoke": False, "bold": False, "outline": 1.5, "shadow": 0.0,
                 "back": 0.0, "scale": 0.82, "spacing": 0.0, "upper": False},
}

# ASS alignment (numpad): 2 = bottom-centre, 5 = middle-centre, 8 = top-centre.
_ALIGN = {"bottom": 2, "center": 5, "middle": 5, "top": 8}

_WORD_RE = re.compile(r"\S+")


# ------------------------------------------------------------------- words
def split_words(text: str) -> List[str]:
    return _WORD_RE.findall(text or "")


def estimate_words(text: str, start: float, duration: float) -> List[dict]:
    """Spread ``text``'s words over ``duration`` proportionally to their length.

    Every word gets a floor of 90 ms plus time proportional to its characters,
    so "a" does not flash by in one frame and a long word is not clipped."""
    words = split_words(text)
    if not words or duration <= 0:
        return []
    floor = min(0.09, duration / (len(words) * 2.0))
    weights = [len(w) + 1.0 for w in words]
    total_w = sum(weights)
    spare = max(0.0, duration - floor * len(words))
    out, t = [], start
    for w, weight in zip(words, weights):
        d = floor + spare * (weight / total_w)
        out.append({"word": w, "start": round(t, 3), "end": round(t + d, 3)})
        t += d
    if out:
        out[-1]["end"] = round(start + duration, 3)
    return out


def _norm(word: str) -> str:
    return re.sub(r"[^\w']+", "", (word or "").lower(), flags=re.UNICODE)


def align_words(text: str, stt_words: List[dict], start: float,
                duration: float) -> List[dict]:
    """Put the caller's words on the STT words' clock.

    The two lists rarely match one-to-one: a model hears "3" as "three", joins
    "voice over" or drops a filler. So we walk both and consume STT words per
    caller word, greedily matching on the normalised form and falling back to
    proportional splitting for any caller word left without a timestamp. The
    result always covers exactly ``[start, start+duration]`` and never contains
    a word the caller did not write."""
    words = split_words(text)
    if not words:
        return []
    stt = [w for w in (stt_words or [])
           if isinstance(w, dict) and (w.get("word") or "").strip()]
    if not stt:
        return estimate_words(text, start, duration)
    # STT timestamps are relative to the clip we sent; shift onto the timeline.
    times = []
    for w in stt:
        try:
            s = float(w.get("start") or 0.0)
            e = float(w.get("end") or s)
        except (TypeError, ValueError):
            continue
        times.append({"t": _norm(str(w.get("word"))), "start": start + s,
                      "end": start + max(e, s)})
    if not times:
        return estimate_words(text, start, duration)

    out: List[Optional[dict]] = [None] * len(words)
    j = 0
    for i, word in enumerate(words):
        if j >= len(times):
            break
        target = _norm(word)
        # Normally each caller word takes the next STT word, even when the model
        # heard something else ("3" for "three") — a substitution still marks
        # when that word was said. Two refinements:
        hit = j
        if target and times[j]["t"] != target:
            # the model inserted a word the caller did not write ("um"): skip it
            # when the real match is just behind it;
            for k in range(j + 1, min(j + 3, len(times))):
                if times[k]["t"] == target:
                    hit = k
                    break
        end_idx = hit
        acc = times[hit]["t"]
        # one caller word may have been heard as several ("voiceover" → "voice
        # over"): merge while the pieces still spell the caller's word.
        while (end_idx + 1 < len(times) and target and acc != target
               and target.startswith(acc)
               and target.startswith(acc + times[end_idx + 1]["t"])):
            end_idx += 1
            acc += times[end_idx]["t"]
        out[i] = {"word": word, "start": times[hit]["start"], "end": times[end_idx]["end"]}
        j = end_idx + 1

    # Fill the gaps (unmatched tails) by splitting the space around them.
    last_end = start
    for i, rec in enumerate(out):
        if rec is not None:
            last_end = max(last_end, rec["end"])
            continue
        # Find the next anchored word to know how much room there is.
        nxt_start = None
        for k in range(i + 1, len(out)):
            if out[k] is not None:
                nxt_start = out[k]["start"]
                break
        room_end = nxt_start if nxt_start is not None else start + duration
        # The contiguous run of unanchored words starting here shares the space.
        run = []
        k = i
        while k < len(out) and out[k] is None:
            run.append(k)
            k += 1
        span = max(0.05, room_end - last_end)
        filler = estimate_words(" ".join(words[r] for r in run), last_end, span)
        for r, f in zip(run, filler):
            out[r] = {"word": words[r], "start": f["start"], "end": f["end"]}
        last_end = out[run[-1]]["end"] if run else last_end

    fixed = [w for w in out if w]
    # Monotonic, inside the scene, non-zero length.
    prev = start
    for w in fixed:
        w["start"] = round(max(prev, min(w["start"], start + duration)), 3)
        w["end"] = round(max(w["start"] + 0.04, min(w["end"], start + duration)), 3)
        prev = w["end"]
    if fixed:
        fixed[-1]["end"] = round(min(start + duration, max(fixed[-1]["end"],
                                                           fixed[-1]["start"] + 0.04)), 3)
    return fixed


# ------------------------------------------------------------------- lines
def break_lines(words: List[dict], max_words_per_line: int = 4,
                max_lines: int = 2) -> List[dict]:
    """Group timed words into caption events of at most ``max_lines`` lines of
    ``max_words_per_line`` words, each with its own start/end."""
    per_line = max(1, int(max_words_per_line or 4))
    lines_max = max(1, int(max_lines or 1))
    per_event = per_line * lines_max
    out = []
    for i in range(0, len(words), per_event):
        chunk = words[i:i + per_event]
        if not chunk:
            continue
        rows = [chunk[k:k + per_line] for k in range(0, len(chunk), per_line)]
        ev = {
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "words": chunk,
            "rows": [[w["word"] for w in row] for row in rows],
            "text": " ".join(w["word"] for w in chunk),
        }
        # Words carry the margin their scene needs (a bottom presenter bubble);
        # an event takes the largest of the ones it groups.
        mv = max((int(w.get("margin_v") or 0) for w in chunk), default=0)
        if mv:
            ev["margin_v"] = mv
        out.append(ev)
    return out


# --------------------------------------------------------------------- ASS
def _ass_colour(hex_colour: str, alpha: int = 0) -> str:
    """``#RRGGBB`` → ASS ``&HAABBGGRR``."""
    h = (hex_colour or "").strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        h = "FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _ass_time(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _ass_escape(text: str) -> str:
    return (text or "").replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def build_ass(events: List[dict], *, width: int, height: int, font: str = "Montserrat",
              font_scale: float = 1.0, colour: str = "#FFFFFF",
              highlight: str = "#FFE14D", outline_colour: str = "#000000",
              position: str = "center", preset: str = "karaoke",
              uppercase: bool = False) -> str:
    """The ASS subtitle file the renderer burns in.

    Sizes are derived from the canvas height so the same request looks the same
    at 1080x1920 and 720x1280; margins keep every line inside the safe area."""
    p = PRESETS.get((preset or "karaoke").lower(), PRESETS["karaoke"])
    upper = bool(uppercase) or p["upper"]
    base = height * 0.052 * p["scale"] * float(font_scale or 1.0)
    size = max(12, int(round(base)))
    align = _ALIGN.get((position or "center").lower(), 5)
    margin_v = int(round(height * 0.10))
    margin_h = int(round(width * 0.06))
    primary = _ass_colour(colour)
    hi = _ass_colour(highlight)
    outline = _ass_colour(outline_colour)
    back = _ass_colour("#000000", alpha=int(round((1.0 - p["back"]) * 255)) if p["back"] else 255)
    border_style = 3 if p["back"] else 1        # 3 = opaque box
    head = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {int(width)}",
        f"PlayResY: {int(height)}",
        "WrapStyle: 2",                          # no automatic wrapping: we break lines
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: cap,{font or 'DejaVu Sans'},{size},{primary},{hi},{outline},{back},"
        f"{-1 if p['bold'] else 0},0,0,0,100,100,{p['spacing']},0,"
        f"{border_style},{p['outline'] * size / 40.0:.2f},{p['shadow'] * size / 40.0:.2f},"
        f"{align},{margin_h},{margin_h},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    body = []
    for ev in events:
        rows = ev.get("rows") or [[w["word"] for w in ev.get("words", [])]]
        # A scene with a presenter bubble at the bottom pushes its captions up:
        # ASS takes a per-event MarginV, so only those events move.
        mv = int(ev.get("margin_v") or 0)
        if not p["karaoke"]:
            text = "\\N".join(" ".join(_ass_escape(w.upper() if upper else w) for w in row)
                              for row in rows)
            body.append(f"Dialogue: 0,{_ass_time(ev['start'])},{_ass_time(ev['end'])},"
                        f"cap,,0,0,{mv},,{text}")
            continue
        # Karaoke: one event per word, the whole line drawn each time with the
        # active word in the highlight colour. Simpler than \k timings and it
        # renders identically on every libass version.
        flat = ev.get("words") or []
        for idx, active in enumerate(flat):
            parts = []
            n = 0
            for row in rows:
                row_out = []
                for w in row:
                    shown = _ass_escape(w.upper() if upper else w)
                    row_out.append(f"{{\\c{hi}}}{shown}{{\\c{primary}}}"
                                   if n == idx else shown)
                    n += 1
                parts.append(" ".join(row_out))
            start = active["start"]
            end = flat[idx + 1]["start"] if idx + 1 < len(flat) else ev["end"]
            if end <= start:
                end = start + 0.04
            body.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},"
                        f"cap,,0,0,{mv},,{{\\c{primary}}}" + "\\N".join(parts))
    return "\n".join(head + body) + "\n"


# ---------------------------------------------------------------- SRT / VTT
def _srt_time(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def _vtt_time(t: float) -> str:
    return _srt_time(t).replace(",", ".")


def build_srt(events: List[dict]) -> str:
    out = []
    for i, ev in enumerate(events, 1):
        text = "\n".join(" ".join(row) for row in (ev.get("rows") or [[ev.get("text", "")]]))
        out.append(f"{i}\n{_srt_time(ev['start'])} --> {_srt_time(ev['end'])}\n{text}\n")
    return "\n".join(out)


def build_vtt(events: List[dict]) -> str:
    out = ["WEBVTT", ""]
    for ev in events:
        text = "\n".join(" ".join(row) for row in (ev.get("rows") or [[ev.get("text", "")]]))
        out.append(f"{_vtt_time(ev['start'])} --> {_vtt_time(ev['end'])}\n{text}\n")
    return "\n".join(out)


def write_files(events: List[dict], out_dir: str, stem: str) -> dict:
    """Write ``<stem>.srt`` / ``.vtt`` next to each other; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for kind, text in (("srt", build_srt(events)), ("vtt", build_vtt(events))):
        p = os.path.join(out_dir, f"{stem}.{kind}")
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        paths[kind] = p
    return paths
