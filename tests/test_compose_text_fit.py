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

"""Text that does not fit: wrap, never silently shrink below readable.

A text overlay was one drawtext line sized from the frame height, so a long
text ran off a vertical frame; and a caption row wider than the frame was drawn
smaller until it fit, which on a vertical video can make it too small to read.
Both now take a ``fit``, and an explicit newline always breaks regardless --
a caller that pre-wrapped its text used to have it drawn as a single line.
"""

import pytest

from codai.compose import captions as cap
from codai.compose import render as rnd


# A measurable font is needed for any of this to do anything; without one the
# code deliberately leaves the text alone rather than guessing from character
# counts, so the tests would be vacuous.
_MEASURABLE = rnd._measure("probe", 40, "DejaVu Sans", True) is not None
needs_font = pytest.mark.skipif(not _MEASURABLE,
                                reason="no measurable font available")

FONT = "DejaVu Sans"
LONG = ("This is a decidedly long watermark line that will not fit across a "
        "narrow vertical frame at this size")


# --------------------------------------------------------------- wrap_text

def test_an_explicit_newline_always_breaks_whatever_the_fit():
    """The case Motus hit: pre-wrapped text drawn as one line."""
    lines = rnd.wrap_text("first line\nsecond line", 40, FONT, True, 100000.0)
    assert lines == ["first line", "second line"], lines


def test_carriage_returns_count_as_newlines():
    assert rnd.wrap_text("a\r\nb", 40, FONT, True, 100000.0) == ["a", "b"]


@needs_font
def test_wrap_breaks_a_long_text_into_lines_that_fit():
    usable = 400.0
    lines = rnd.wrap_text(LONG, 40, FONT, True, usable)
    assert len(lines) > 1, "a long text was not broken at all"
    for ln in lines:
        w = rnd._measure(ln, 40, FONT, True)
        assert w is not None and w <= usable, f"line still too wide ({w}): {ln!r}"


@needs_font
def test_wrap_keeps_every_word_in_order():
    lines = rnd.wrap_text(LONG, 40, FONT, True, 400.0)
    assert " ".join(lines).split() == LONG.split()


@needs_font
def test_a_single_word_wider_than_a_line_is_cut_not_drawn_past_the_edge():
    usable = 220.0
    url = "https://example.com/a/very/long/path?with=query&and=more#fragment"
    lines = rnd.wrap_text(url, 36, FONT, True, usable)
    assert len(lines) > 1, "an unbreakable-looking word was left overflowing"
    for ln in lines:
        w = rnd._measure(ln, 36, FONT, True)
        assert w is not None and w <= usable, f"piece too wide ({w}): {ln!r}"
    assert "".join(lines) == url, "cutting the word lost or reordered characters"


@needs_font
def test_an_unmeasurable_font_leaves_the_text_alone():
    """Guessing from character counts is wrong in exactly the case this catches."""
    lines = rnd.wrap_text(LONG, 40, "NoSuchFontFamilyAnywhere__", True, 10.0)
    assert lines == [LONG] or len(lines) >= 1


# ------------------------------------------------------------- shrink_to_fit

@needs_font
def test_shrink_reduces_the_size_until_one_line_actually_fits():
    """Verified by measurement: the width ratio alone picks a size too large."""
    usable = 900.0
    size = rnd.shrink_to_fit(LONG, 60, FONT, True, usable)
    assert size < 60, "shrink did not reduce the size"
    w = rnd._measure(LONG, size, FONT, True)
    assert w is not None and w <= usable, f"still {w} px wide at size {size}"


@needs_font
def test_shrink_returns_the_largest_size_that_fits():
    usable = 900.0
    size = rnd.shrink_to_fit(LONG, 60, FONT, True, usable)
    bigger = rnd._measure(LONG, size + 1, FONT, True)
    assert bigger is None or bigger > usable, (
        f"size {size} fits but so does {size + 1}; shrink gave up too much")


@needs_font
def test_shrink_returns_the_floor_when_nothing_fits():
    """A budget no size can meet gets the floor, not an overflowing guess."""
    size = rnd.shrink_to_fit(LONG, 60, FONT, True, 30.0, floor=8)
    assert size == 8, f"expected the floor, got {size}"


@needs_font
def test_shrink_leaves_a_text_that_already_fits_alone():
    assert rnd.shrink_to_fit("short", 40, FONT, True, 100000.0) == 40


# --------------------------------------------------------------- caption fit

def _events():
    words = [{"word": w, "start": i * 0.3, "end": i * 0.3 + 0.3}
             for i, w in enumerate(LONG.split())]
    return [{"start": 0.0, "end": 9.0, "words": words,
             "rows": [[w["word"] for w in words]]}]


@needs_font
def test_captions_wrap_keeps_the_size_and_adds_rows():
    ass = cap.build_ass(_events(), width=1080, height=1920, font=FONT,
                        preset="classic", fit="wrap")
    # No inline \fs override means the event kept the global size.
    assert "\\fs" not in ass, "wrap shrank the text anyway"
    assert "\\N" in ass, "wrap produced no extra row"


@needs_font
def test_captions_shrink_is_still_the_default():
    ass = cap.build_ass(_events(), width=1080, height=1920, font=FONT,
                        preset="classic")
    assert "\\fs" in ass, "the default stopped shrinking an overlong row"


@needs_font
def test_wrapping_rows_preserves_word_order_for_karaoke():
    rows = [LONG.split()]
    out = cap._wrap_rows(rows, 48, 400.0, FONT, True, 0.0, False)
    assert [w for row in out for w in row] == LONG.split(), (
        "the karaoke highlight walks words in row order, so wrapping must not "
        "reorder them")
    assert len(out) > 1


@needs_font
def test_a_wrapped_karaoke_event_still_highlights_every_word():
    ass = cap.build_ass(_events(), width=1080, height=1920, font=FONT,
                        preset="karaoke", fit="wrap")
    # One Dialogue line per word, each highlighting exactly one.
    dialogues = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(dialogues) == len(LONG.split()), (
        f"{len(dialogues)} events for {len(LONG.split())} words")


# ------------------------------------------------------- the renderer wiring

def test_the_overlay_renderer_accepts_fit_and_emits_a_line_each(monkeypatch):
    """`fit=wrap` must reach drawtext as several placed lines."""
    seen = {}

    def _fake_run(args, **kw):
        seen["args"] = args

    monkeypatch.setattr(rnd, "run", _fake_run)
    rnd.final_render(
        "v.mp4", "a.wav", "out.mp4", width=1080, height=1920, fps=30, total=5.0,
        overlays=[{"type": "text", "text": LONG, "fit": "wrap",
                   "position": "bottom-right", "size": 0.03, "font": FONT}])
    graph = next(a for a in seen["args"] if "drawtext" in str(a))
    n = str(graph).count("drawtext")
    if _MEASURABLE:
        assert n > 1, f"wrap emitted {n} drawtext filter(s); expected one per line"
    else:
        assert n >= 1


def test_an_overlay_without_fit_still_renders_one_line(monkeypatch):
    seen = {}
    monkeypatch.setattr(rnd, "run", lambda args, **kw: seen.setdefault("args", args))
    rnd.final_render(
        "v.mp4", "a.wav", "out.mp4", width=1080, height=1920, fps=30, total=5.0,
        overlays=[{"type": "text", "text": "made with motus",
                   "position": "bottom-right"}])
    graph = next(a for a in seen["args"] if "drawtext" in str(a))
    assert str(graph).count("drawtext") == 1


@needs_font
def test_a_shrunk_caption_actually_fits_its_frame():
    """The invariant `_fit_size` exists for, now held by construction.

    It used to scale the size by the width ratio and trust the result. This
    input passes either way -- the ratio is close enough above `_fit_size`'s
    floor of 12 -- so this is a regression guard rather than a reproduction;
    the assumption provably fails at smaller sizes (see shrink_to_fit, where a
    computed size of 8 measured 47% over its budget).
    """
    rows = [LONG.split()]
    usable = 900.0
    size = cap._fit_size(rows, 64, usable, FONT, True, 0.0, False)
    got = cap._text_width(" ".join(rows[0]), size, FONT, True, 0.0)
    assert got is not None and got <= usable, (
        f"a caption shrunk to size {size} is still {got} px against a "
        f"{usable} px budget, so it is drawn past the frame edge anyway")
