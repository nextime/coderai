"""Server-side composition: ``POST /v1/video/compose`` and ``POST /v1/files/upload``.

These tests really run ffmpeg — a composition that "works" in mocks and produces
an unplayable file is worthless — but never a model: narration is supplied as
pre-made audio, captions are timed by estimation, and music is a sine wave. So
the whole pipeline (visual normalisation, concat and crossfade, narration
padding, ducking, burned captions, thumbnail, SRT/VTT) is exercised end to end
on any machine with ffmpeg and no GPU.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codai.compose import captions as cap
from codai.compose import media
from codai.compose import render as rnd

pytestmark = pytest.mark.skipif(media.ffmpeg_bin() is None
                                or subprocess.run(["which", "ffmpeg"],
                                                  capture_output=True).returncode != 0,
                                reason="composition needs ffmpeg on PATH")


# --------------------------------------------------------------- fixtures
def _tone(path, seconds=1.0, freq=440):
    subprocess.run([media.ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
                    "-ac", "2", "-ar", "48000", str(path)], check=True)
    return str(path)


def _clip(path, seconds=1.0, w=320, h=240, colour="red"):
    subprocess.run([media.ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"color=c={colour}:s={w}x{h}:r=25:d={seconds}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


def _uri(path):
    """A local fixture file as a client would send it: a data URI. Compose
    deliberately refuses bare server paths, so tests use the real wire form."""
    import base64
    import mimetypes
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def _png(path, w=64, h=64, colour=(200, 40, 90)):
    from PIL import Image
    Image.new("RGB", (w, h), colour).save(str(path))
    return str(path)


@pytest.fixture
def files_dir(tmp_path, monkeypatch):
    d = tmp_path / "out"
    d.mkdir()
    media.set_global_file_path(str(d))
    yield d
    media.set_global_file_path("")


# =================================================================== words
def test_estimated_word_timings_fill_the_scene_exactly():
    words = cap.estimate_words("Octopuses have three hearts and blue blood.", 2.0, 4.0)
    assert [w["word"] for w in words] == ["Octopuses", "have", "three", "hearts",
                                          "and", "blue", "blood."]
    assert words[0]["start"] == 2.0
    assert words[-1]["end"] == pytest.approx(6.0, abs=0.01)
    # Monotonic, nothing zero-length, longer words get more time than "and".
    for a, b in zip(words, words[1:]):
        assert a["end"] <= b["start"] + 1e-6 and a["end"] > a["start"]
    assert (words[0]["end"] - words[0]["start"]) > (words[4]["end"] - words[4]["start"])
    assert cap.estimate_words("", 0.0, 3.0) == []


def test_stt_timings_are_aligned_onto_the_callers_words():
    # The model heard "3" as "three" and split "voiceover"; the caption must still
    # show what the caller wrote, on the model's clock.
    stt = [{"word": "octopuses", "start": 0.0, "end": 0.5},
           {"word": "have", "start": 0.5, "end": 0.7},
           {"word": "three", "start": 0.7, "end": 1.1},
           {"word": "hearts", "start": 1.1, "end": 1.6}]
    out = cap.align_words("Octopuses have 3 hearts", stt, 10.0, 2.0)
    assert [w["word"] for w in out] == ["Octopuses", "have", "3", "hearts"]
    assert out[0]["start"] == pytest.approx(10.0, abs=0.01)
    assert out[2]["start"] == pytest.approx(10.7, abs=0.05)     # "3" got "three"'s slot
    assert out[-1]["end"] <= 12.0 + 1e-6
    # No STT at all → estimation, never an empty caption track.
    assert len(cap.align_words("two words here", [], 0.0, 1.5)) == 3


def test_lines_break_and_srt_matches_the_text():
    words = cap.estimate_words("one two three four five six seven eight", 0.0, 8.0)
    events = cap.break_lines(words, max_words_per_line=2, max_lines=2)
    assert [ev["rows"] for ev in events] == [[["one", "two"], ["three", "four"]],
                                            [["five", "six"], ["seven", "eight"]]]
    assert events[0]["start"] == 0.0 and events[-1]["end"] == pytest.approx(8.0, abs=0.01)
    srt = cap.build_srt(events)
    assert srt.startswith("1\n00:00:00,000 --> ")
    assert "one two\nthree four" in srt
    vtt = cap.build_vtt(events)
    assert vtt.startswith("WEBVTT") and "-->" in vtt and "," not in vtt.split("\n")[2]


def test_ass_is_karaoke_safe_area_and_never_leaks_the_stt_text():
    words = cap.estimate_words("hello brave new world", 0.0, 4.0)
    events = cap.break_lines(words, 2, 2)
    ass = cap.build_ass(events, width=1080, height=1920, font="Montserrat",
                        colour="#FFFFFF", highlight="#FFE14D", position="center",
                        preset="karaoke")
    assert "PlayResX: 1080" and "PlayResY: 1920" in ass
    # Safe area: 10% vertical, 6% horizontal margins, centre alignment.
    style = [l for l in ass.splitlines() if l.startswith("Style: cap,")][0]
    assert style.endswith(",65,65,192,1") and ",5," in style   # 6% sides, 10% vertical
    # One event per word (karaoke) and the highlight colour appears in each.
    dialogues = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(dialogues) == 4
    assert all("&H004DE1FF" in d for d in dialogues)      # #FFE14D as ASS BGR
    # classic preset: one event per line, no per-word highlight
    plain = cap.build_ass(events, width=1080, height=1920, preset="classic")
    # One event for the whole 2-line block, the rows joined with ASS's \N.
    assert len([l for l in plain.splitlines() if l.startswith("Dialogue:")]) == 1
    assert "hello brave\\Nnew world" in plain


def test_uppercase_and_box_presets_change_the_style_not_the_words():
    events = cap.break_lines(cap.estimate_words("keep it real", 0.0, 2.0), 3, 1)
    boxed = cap.build_ass(events, width=720, height=1280, preset="boxed", uppercase=True)
    assert "KEEP IT REAL" in boxed
    style = [l for l in boxed.splitlines() if l.startswith("Style: cap,")][0]
    assert ",3," in style          # BorderStyle 3 = opaque box


# =================================================================== media
def test_mime_sniffing_beats_the_clients_content_type(tmp_path):
    png = Path(_png(tmp_path / "x.png"))
    assert media.sniff_mime(png.read_bytes(), "application/octet-stream") == "image/png"
    assert media.kind_of("image/png") == "image"
    wav = Path(_tone(tmp_path / "a.wav", 0.2))
    assert media.sniff_mime(wav.read_bytes()) == "audio/wav"
    mp4 = Path(_clip(tmp_path / "v.mp4", 0.2))
    assert media.sniff_mime(mp4.read_bytes()) == "video/mp4"
    assert media.ext_for("video/mp4") == ".mp4"
    assert media.kind_of("text/plain") == "other"


def test_references_resolve_from_data_uris_files_urls_and_base64(tmp_path, files_dir):
    import base64
    png = Path(_png(tmp_path / "y.png")).read_bytes()
    work = str(tmp_path / "work")
    p1 = media.resolve("data:image/png;base64," + base64.b64encode(png).decode(),
                       work, "a")
    assert Path(p1).read_bytes() == png
    p2 = media.resolve(base64.b64encode(png).decode(), work, "b")
    assert Path(p2).read_bytes() == png
    # A file this install already serves is used where it lies, not re-downloaded.
    (files_dir / "held.png").write_bytes(png)
    assert media.resolve("/v1/files/held.png", work, "c") == str(files_dir / "held.png")
    assert media.resolve("http://elsewhere.invalid:1/v1/files/held.png", work, "d") \
        == str(files_dir / "held.png")
    # Path traversal through the files URL is refused.
    assert media.local_path_for_files_url("/v1/files/../../etc/passwd") is None
    with pytest.raises(media.MediaError):
        media.resolve("/etc/passwd", work, "e")
    with pytest.raises(media.MediaError):
        media.resolve("not base64 at all !!", work, "f")


def test_probe_reports_what_the_renderer_needs(tmp_path):
    mp4 = _clip(tmp_path / "p.mp4", seconds=1.5, w=640, h=360)
    info = media.probe(mp4)
    assert info["has_video"] and not info["has_audio"]
    assert info["width"] == 640 and info["height"] == 360
    assert info["duration"] == pytest.approx(1.5, abs=0.15)
    assert info["fps"] == pytest.approx(25, abs=1)
    wav = _tone(tmp_path / "p.wav", 0.7)
    a = media.probe(wav)
    assert a["has_audio"] and not a["has_video"]
    assert media.duration_of(wav) == pytest.approx(0.7, abs=0.05)
    with pytest.raises(media.MediaError):
        media.probe(str(tmp_path / "nope.mp4"))


# ================================================================== render
def test_fit_filters_cover_contain_and_blur_pad():
    assert "crop=1080:1920" in rnd.fit_filter(1080, 1920, "cover")
    assert "pad=1080:1920" in rnd.fit_filter(1080, 1920, "contain")
    blur = rnd.fit_filter(1080, 1920, "blur_pad")
    assert "gblur" in blur and "overlay" in blur
    assert rnd.fit_filter(100, 100, "nonsense") == rnd.fit_filter(100, 100, "cover")


def test_ken_burns_alternates_direction_and_respects_the_zoom_range():
    a = rnd.ken_burns_filter(1080, 1920, 30, 2.0, 1.0, 1.2, "auto", index=0)
    b = rnd.ken_burns_filter(1080, 1920, 30, 2.0, 1.0, 1.2, "auto", index=1)
    assert "zoompan" in a and "d=60" in a and "s=1080x1920" in a
    assert a != b                                   # right, then left
    assert "2160" in a                              # oversampled before zoompan
    down = rnd.ken_burns_filter(720, 1280, 25, 1.0, 1.2, 1.0, "none", 0)
    assert "max(" in down                           # zooming out
    assert "zoompan" in rnd.ken_burns_filter(720, 1280, 25, 1.0, 1.0, 1.0, "up", 0)


def test_a_visual_of_each_type_becomes_a_clip_of_the_right_length(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    cases = [
        ({"type": "color", "color": "#112233"}, 0.6),
        ({"type": "gradient", "colors": ["#6D28D9", "#DB2777"]}, 0.5),
        ({"type": "image", "_path": _png(tmp_path / "i.png", 200, 300)}, 0.8),
        ({"type": "video", "_path": _clip(tmp_path / "v.mp4", 2.0),
          "_probe": None, "trim_start": 0.5}, 0.7),
    ]
    for n, (spec, dur) in enumerate(cases):
        if spec.get("type") == "video":
            spec["_probe"] = media.probe(spec["_path"])
        out = str(work / f"c{n}.mp4")
        rnd.build_visual(spec, out, width=240, height=426, fps=24, duration=dur,
                         fit="cover", index=n, workdir=str(work))
        info = media.probe(out)
        assert info["width"] == 240 and info["height"] == 426
        assert info["duration"] == pytest.approx(dur, abs=0.12), spec["type"]
        assert not info["has_audio"]


def test_a_short_video_is_looped_to_fill_its_slot(tmp_path):
    src = _clip(tmp_path / "short.mp4", seconds=0.4)
    out = str(tmp_path / "looped.mp4")
    rnd.build_visual({"type": "video", "_path": src, "_probe": media.probe(src)},
                     out, width=160, height=284, fps=24, duration=1.6,
                     workdir=str(tmp_path))
    assert media.probe(out)["duration"] == pytest.approx(1.6, abs=0.12)


def test_clips_join_by_concat_and_by_crossfade_without_losing_time(tmp_path):
    work = tmp_path / "j"
    work.mkdir()
    clips, durs = [], []
    for i, colour in enumerate(("red", "green", "blue")):
        p = str(work / f"{i}.mp4")
        rnd.build_visual({"type": "color", "color": colour}, p, width=160, height=284,
                         fps=24, duration=1.0 + 0.3, workdir=str(work), index=i)
        clips.append(p)
        durs.append(1.3)
    joined = str(work / "concat.mp4")
    rnd.concat_clips(clips, joined, str(work))
    assert media.probe(joined)["duration"] == pytest.approx(3.9, abs=0.15)
    faded = str(work / "xfade.mp4")
    rnd.xfade_clips(clips, durs, faded, kind="crossfade", dur=0.3, fps=24)
    # 3 clips of 1.3 with 0.3 of overlap each → 3*1.0 + 0.3 of tail.
    assert media.probe(faded)["duration"] == pytest.approx(3.3, abs=0.2)


def test_narration_is_padded_per_scene_and_music_is_ducked(tmp_path):
    work = tmp_path / "a"
    work.mkdir()
    v1 = _tone(work / "v1.wav", 0.5, 300)
    padded = rnd.pad_to(v1, str(work / "p1.wav"), 1.5)
    assert media.duration_of(padded) == pytest.approx(1.5, abs=0.05)
    sil = rnd.silence(str(work / "s.wav"), 0.8)
    joined = rnd.concat_audio([padded, sil], str(work / "n.wav"), str(work))
    assert media.duration_of(joined) == pytest.approx(2.3, abs=0.08)
    normed = rnd.loudnorm(joined, str(work / "nl.wav"))
    assert media.duration_of(normed) == pytest.approx(2.3, abs=0.15)
    bed = rnd.build_music_bed(_tone(work / "m.wav", 0.6, 120), str(work / "bed.wav"),
                              total=2.3, volume=0.2, loop=True, fade_in=0.2, fade_out=0.4)
    assert media.duration_of(bed) == pytest.approx(2.3, abs=0.08)
    mixed = rnd.mix_narration_music(normed, bed, str(work / "mix.wav"), duck=True)
    assert media.duration_of(mixed) == pytest.approx(2.3, abs=0.12)
    alone = rnd.mix_narration_music(normed, None, str(work / "solo.wav"))
    assert media.duration_of(alone) == pytest.approx(2.3, abs=0.12)


def test_the_final_render_is_uploadable_and_carries_captions_and_overlays(tmp_path):
    work = tmp_path / "f"
    work.mkdir()
    silent = str(work / "v.mp4")
    rnd.build_visual({"type": "gradient"}, silent, width=240, height=426, fps=24,
                     duration=2.0, workdir=str(work))
    audio = rnd.mix_narration_music(_tone(work / "n.wav", 2.0), None, str(work / "a.wav"))
    events = cap.break_lines(cap.estimate_words("burned in captions here", 0.0, 2.0), 2, 2)
    ass = work / "c.ass"
    ass.write_text(cap.build_ass(events, width=240, height=426, font="DejaVu Sans"))
    out = str(work / "final.mp4")
    rnd.final_render(silent, audio, out, width=240, height=426, fps=24, total=2.0,
                     ass_file=str(ass),
                     overlays=[{"type": "text", "text": "made with motus",
                               "position": "bottom-right", "opacity": 0.6, "size": 0.04}],
                     crf=28)
    info = media.probe(out)
    assert info["has_video"] and info["has_audio"]
    assert info["duration"] == pytest.approx(2.0, abs=0.15)
    # H.264 High + AAC + faststart is what the platforms take without re-encoding.
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                            "-print_format", "json", out], capture_output=True)
    doc = json.loads(probe.stdout)
    vs = [s for s in doc["streams"] if s["codec_type"] == "video"][0]
    aud = [s for s in doc["streams"] if s["codec_type"] == "audio"][0]
    assert vs["codec_name"] == "h264" and vs["profile"] == "High"
    assert vs["pix_fmt"] == "yuv420p"
    assert aud["codec_name"] == "aac" and int(aud["sample_rate"]) == 48000
    thumb = str(work / "t.jpg")
    rnd.thumbnail(out, thumb, at=0.5, width=240, height=426, text="3 hearts?!")
    assert os.path.getsize(thumb) > 500


def test_a_cancelled_render_kills_its_ffmpeg(tmp_path):
    import threading
    ev = threading.Event()
    ev.set()
    with pytest.raises(rnd.Cancelled):
        rnd.run(["-f", "lavfi", "-i", "color=c=black:s=64x64:r=30:d=30",
                 "-c:v", "libx264", str(tmp_path / "never.mp4")], cancel=ev)
    with pytest.raises(rnd.RenderError):
        rnd.run(["-i", str(tmp_path / "missing.mp4"), str(tmp_path / "x.mp4")])


# ============================================================ the whole job
def _spec(tmp_path, **over):
    voice = _tone(tmp_path / "vo1.wav", 1.2, 330)
    voice2 = _tone(tmp_path / "vo2.wav", 0.8, 420)
    spec = {
        "canvas": {"width": 160, "height": 284, "fps": 24},
        "scenes": [
            {"text": "Octopuses have three hearts.", "audio": _uri(voice), "padding": 0.2,
             "visuals": [{"type": "gradient", "colors": ["#6D28D9", "#DB2777"]},
                         {"type": "image", "src": _data_uri_png(tmp_path)}]},
            {"text": "And blue blood.", "audio": _uri(voice2), "padding": 0.1,
             "min_duration": 1.5,
             "visuals": [{"type": "color", "color": "#111827"}]},
        ],
        "captions": {"enabled": True, "timing": "estimate", "preset": "karaoke",
                     "font": "DejaVu Sans", "max_words_per_line": 2},
        "overlays": [{"type": "text", "text": "motus", "position": "bottom-right"}],
        "thumbnail": {"at": 0.3, "text": "3 hearts?!"},
        "output": {"crf": 30},
        "outputs": ["video", "thumbnail", "srt", "vtt", "narration"],
        "async": True,
    }
    spec.update(over)
    return spec


def _data_uri_png(tmp_path):
    import base64
    data = Path(_png(tmp_path / "scene.png", 120, 200)).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _client(files_dir):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from codai.api import compose as compose_mod
    from codai.api import uploads as uploads_mod
    app = FastAPI()
    app.include_router(compose_mod.router)
    app.include_router(uploads_mod.router)
    return TestClient(app), compose_mod


def _wait(client, job_id, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/v1/video/compose/{job_id}")
        assert r.status_code == 200
        doc = r.json()
        if doc["status"] in ("done", "failed", "cancelled"):
            return doc
        time.sleep(0.4)
    raise AssertionError("composition did not finish in time")


def test_a_composition_job_produces_every_artefact(tmp_path, files_dir):
    client, _ = _client(files_dir)
    r = client.post("/v1/video/compose", json=_spec(tmp_path))
    assert r.status_code == 202, r.text
    job_id = r.json()["id"]
    assert r.json()["status"] == "queued" and job_id.startswith("cmp_")
    doc = _wait(client, job_id)
    assert doc["status"] == "done", doc
    assert doc["progress"] == 100.0 and doc["error"] is None
    res = doc["result"]

    # Scene timing is narration-driven: 1.2+0.2, then max(0.8+0.1, 1.5).
    assert [s["start"] for s in res["scenes"]] == [0.0, pytest.approx(1.4, abs=0.1)]
    assert res["scenes"][1]["end"] == pytest.approx(2.9, abs=0.15)
    assert res["duration"] == pytest.approx(2.9, abs=0.15)

    for key in ("video", "thumbnail", "srt", "vtt", "narration"):
        assert key in res, key
        name = res[key]["path"].split("/")[-1]
        assert (files_dir / name).is_file()
        assert res[key]["url"].endswith(res[key]["path"]) or res[key]["url"].startswith("/v1/")
    vid = files_dir / res["video"]["path"].split("/")[-1]
    info = media.probe(str(vid))
    assert info["has_video"] and info["has_audio"]
    assert (info["width"], info["height"]) == (160, 284)
    assert info["duration"] == pytest.approx(2.9, abs=0.25)
    assert res["video"]["duration"] == pytest.approx(info["duration"], abs=0.05)
    # The captions carry the caller's text, not a transcription.
    srt = (files_dir / res["srt"]["path"].split("/")[-1]).read_text()
    assert "Octopuses have" in srt and "three hearts." in srt
    assert "And blue" in srt and "blood." in srt        # 2 words per line, as asked
    # The download endpoint of the real app serves them by name (path shape).
    assert res["video"]["path"].startswith("/v1/files/cmp_")


def test_the_blocking_form_returns_the_finished_document(tmp_path, files_dir):
    client, _ = _client(files_dir)
    spec = _spec(tmp_path, **{"async": False, "outputs": ["video"],
                              "captions": {"enabled": False}})
    spec["scenes"] = spec["scenes"][:1]
    r = client.post("/v1/video/compose", json=spec)
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["status"] == "done" and doc["result"]["video"]["size"] > 1000
    assert "thumbnail" not in doc["result"] and "srt" not in doc["result"]


def test_a_broken_visual_becomes_a_gradient_and_a_warning(tmp_path, files_dir):
    client, _ = _client(files_dir)
    spec = _spec(tmp_path, captions={"enabled": False}, outputs=["video"])
    spec["scenes"] = [{
        "text": "one scene", "audio": _uri(_tone(tmp_path / "v3.wav", 0.6)),
        "visuals": [{"type": "video", "src": "http://127.0.0.1:1/nope.mp4"},
                    {"type": "image", "src": "data:image/png;base64,bm90YW5pbWFnZQ=="}],
    }]
    doc = _wait(client, client.post("/v1/video/compose", json=spec).json()["id"])
    assert doc["status"] == "done", doc
    assert len(doc["warnings"]) >= 2
    assert any("gradient" in w for w in doc["warnings"])
    vid = files_dir / doc["result"]["video"]["path"].split("/")[-1]
    assert media.probe(str(vid))["has_video"]


def test_crossfade_and_music_paths_render(tmp_path, files_dir):
    client, _ = _client(files_dir)
    spec = _spec(tmp_path, captions={"enabled": False}, outputs=["video"],
                 transition={"type": "crossfade", "duration": 0.25},
                 music={"src": _uri(_tone(tmp_path / "bed.wav", 0.5, 110)), "volume": 0.2,
                        "duck": True, "loop": True, "fade_in": 0.2, "fade_out": 0.3})
    doc = _wait(client, client.post("/v1/video/compose", json=spec).json()["id"])
    assert doc["status"] == "done", doc
    vid = files_dir / doc["result"]["video"]["path"].split("/")[-1]
    assert media.probe(str(vid))["has_audio"]


def test_cancel_stops_a_running_job(tmp_path, files_dir):
    client, compose_mod = _client(files_dir)
    # Long enough that cancellation lands mid-render.
    spec = _spec(tmp_path, captions={"enabled": False}, outputs=["video"])
    spec["canvas"] = {"width": 720, "height": 1280, "fps": 30}
    spec["scenes"] = [{"text": "x", "audio": _uri(_tone(tmp_path / "long.wav", 8.0)),
                       "visuals": [{"type": "gradient"}]}]
    job_id = client.post("/v1/video/compose", json=spec).json()["id"]
    time.sleep(1.0)
    r = client.post(f"/v1/video/compose/{job_id}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    doc = _wait(client, job_id, timeout=60)
    assert doc["status"] == "cancelled"
    assert client.post("/v1/video/compose/cmp_nope/cancel").status_code == 404
    assert client.get("/v1/video/compose/cmp_nope").status_code == 404


def test_the_spec_is_validated_with_the_offending_field_named(tmp_path, files_dir):
    client, _ = _client(files_dir)

    def err(spec):
        r = client.post("/v1/video/compose", json=spec)
        assert r.status_code == 400, r.text
        return r.json()["detail"]

    assert "at least one scene" in err({"scenes": []})
    assert "scenes[0].visuals" in err({"scenes": [{"text": "hi", "visuals": []}]})
    assert "explicit" in err({"scenes": [{"visuals": [{"type": "color"}]}]})
    d = err({"scenes": [{"text": "a", "visuals": [{"type": "hologram"}]}]})
    assert "scenes[0].visuals[0].type" in d and "hologram" in d
    assert "src" in err({"scenes": [{"text": "a", "visuals": [{"type": "video"}]}]})
    good = {"scenes": [{"text": "a", "audio": _uri(_tone(tmp_path / "ok.wav", 0.3)),
                        "visuals": [{"type": "color"}]}]}
    assert "even" in err({**good, "canvas": {"width": 101, "height": 100}})
    assert "visual_fit" in err({**good, "visual_fit": "squeeze"})
    assert "transition.type" in err({**good, "transition": {"type": "swirl"}})
    assert "captions.preset" in err({**good, "captions": {"preset": "sparkles"}})
    assert "outputs" in err({**good, "outputs": ["video", "gif"]})
    assert "output.format" in err({**good, "output": {"format": "avi"}})


def test_missing_models_answer_501_not_a_failed_job(tmp_path, files_dir, monkeypatch):
    client, compose_mod = _client(files_dir)
    monkeypatch.setattr(compose_mod, "_model_ids", lambda mtype: [])
    base = {"scenes": [{"text": "a", "audio": _uri(_tone(tmp_path / "m.wav", 0.3)),
                        "visuals": [{"type": "color"}]}]}
    r = client.post("/v1/video/compose", json={**base, "music": {"generate": {"prompt": "x"}}})
    assert r.status_code == 501 and "audio-generation model" in r.json()["detail"]
    r = client.post("/v1/video/compose",
                    json={**base, "captions": {"enabled": True, "timing": "stt"}})
    assert r.status_code == 501 and "speech-to-text" in r.json()["detail"]
    # Text with no audio and no TTS model: 501 rather than a job that fails later.
    r = client.post("/v1/video/compose",
                    json={"scenes": [{"text": "a", "visuals": [{"type": "color"}]}]})
    assert r.status_code == 501 and "TTS model" in r.json()["detail"]


def test_narration_is_synthesised_through_the_tts_endpoint(tmp_path, files_dir, monkeypatch):
    """No TTS model here, so the endpoint is stubbed — what is tested is that
    compose calls it once per scene with that scene's text and voice."""
    import base64
    client, compose_mod = _client(files_dir)
    monkeypatch.setattr(compose_mod, "_model_ids",
                        lambda mtype: ["kokoro"] if mtype == "tts" else [])
    calls = []
    wav = Path(_tone(tmp_path / "synth.wav", 0.5)).read_bytes()

    async def fake_speech(req, http_request=None):
        calls.append({"input": req.input, "voice": req.voice, "model": req.model,
                      "speed": req.speed, "language": getattr(req, "language", None)})
        return {"audio": base64.b64encode(wav).decode()}

    import codai.api.tts as tts_mod
    monkeypatch.setattr(tts_mod, "create_speech", fake_speech)
    spec = {"canvas": {"width": 160, "height": 284, "fps": 24},
            "voice": {"engine": "tts", "model": "kokoro", "voice": "af_sarah",
                      "language": "en-us", "speed": 1.1},
            "captions": {"enabled": True, "timing": "estimate", "font": "DejaVu Sans"},
            "outputs": ["video", "srt"],
            "scenes": [{"text": "first line", "visuals": [{"type": "color"}]},
                       {"text": "second line", "visuals": [{"type": "gradient"}]}]}
    doc = _wait(client, client.post("/v1/video/compose", json=spec).json()["id"])
    assert doc["status"] == "done", doc
    assert [c["input"] for c in calls] == ["first line", "second line"]
    assert calls[0]["voice"] == "af_sarah" and calls[0]["model"] == "kokoro"
    assert calls[0]["speed"] == 1.1 and calls[0]["language"] == "en-us"
    srt = (files_dir / doc["result"]["srt"]["path"].split("/")[-1]).read_text()
    assert "first line" in srt and "second line" in srt


def test_stt_timings_are_used_when_a_speech_model_is_configured(tmp_path, files_dir,
                                                                monkeypatch):
    client, compose_mod = _client(files_dir)
    monkeypatch.setattr(compose_mod, "_model_ids",
                        lambda mtype: ["whisper-large-v3"] if mtype == "audio" else [])
    seen = {}

    def fake_words(loop, path, model, language):
        seen["model"] = model
        return [{"word": "hello", "start": 0.0, "end": 0.9},
                {"word": "world", "start": 0.9, "end": 1.0}]

    monkeypatch.setattr(compose_mod, "_stt_words", fake_words)
    spec = {"canvas": {"width": 160, "height": 284, "fps": 24},
            "captions": {"enabled": True, "timing": "stt", "stt_model": "whisper-large-v3",
                         "font": "DejaVu Sans", "max_words_per_line": 2},
            "outputs": ["video", "srt"],
            "scenes": [{"text": "hello world", "audio": _uri(_tone(tmp_path / "s.wav", 1.0)),
                        "padding": 0.0, "visuals": [{"type": "color"}]}]}
    doc = _wait(client, client.post("/v1/video/compose", json=spec).json()["id"])
    assert doc["status"] == "done", doc
    assert seen["model"] == "whisper-large-v3"
    srt = (files_dir / doc["result"]["srt"]["path"].split("/")[-1]).read_text()
    # "world" starts at 0.9 per the model, not at the halfway point of the clip.
    assert "00:00:00,000 --> 00:00:01,0" in srt and "hello world" in srt


# ================================================================= uploads
def test_upload_is_content_addressed_and_dedupes(tmp_path, files_dir):
    client, _ = _client(files_dir)
    png = Path(_png(tmp_path / "u.png")).read_bytes()
    r = client.post("/v1/files/upload", files={"file": ("u.png", png, "image/png")})
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["id"].startswith("sha256:") and len(doc["id"]) == 71
    assert doc["mime"] == "image/png" and doc["kind"] == "image"
    assert doc["bytes"] == len(png) and doc["existed"] is False
    assert doc["path"].startswith("/v1/files/up-")
    assert (files_dir / doc["path"].split("/")[-1]).read_bytes() == png

    again = client.post("/v1/files/upload",
                        files={"file": ("other-name.png", png, "application/octet-stream")})
    assert again.json()["existed"] is True and again.json()["id"] == doc["id"]

    # The existence check lets a client skip the upload entirely.
    h = doc["id"].split(":")[1]
    assert client.get(f"/v1/files/blob/{h}").json()["exists"] is True
    assert client.get(f"/v1/files/blob/sha256:{h}").status_code == 200
    assert client.get("/v1/files/blob/" + "0" * 64).status_code == 404
    assert client.get("/v1/files/blob/not-a-hash").status_code == 404


def test_upload_accepts_json_and_raw_bodies_and_refuses_junk(tmp_path, files_dir):
    import base64
    client, _ = _client(files_dir)
    wav = Path(_tone(tmp_path / "u.wav", 0.2)).read_bytes()
    r = client.post("/v1/files/upload",
                    json={"file": base64.b64encode(wav).decode(), "filename": "u.wav"})
    assert r.status_code == 200 and r.json()["kind"] == "audio"
    mp4 = Path(_clip(tmp_path / "u.mp4", 0.3)).read_bytes()
    r = client.post("/v1/files/upload", content=mp4,
                    headers={"content-type": "video/mp4"})
    assert r.status_code == 200 and r.json()["mime"] == "video/mp4"
    r = client.post("/v1/files/upload", content=b"#!/bin/sh\nrm -rf /\n",
                    headers={"content-type": "application/x-sh"})
    assert r.status_code == 415 and "unsupported media type" in r.json()["detail"]
    assert client.post("/v1/files/upload", content=b"",
                       headers={"content-type": "video/mp4"}).status_code == 400


def test_an_uploaded_file_can_be_used_as_a_visual(tmp_path, files_dir):
    client, _ = _client(files_dir)
    clip = Path(_clip(tmp_path / "stock.mp4", 1.0, 320, 180, "blue")).read_bytes()
    up = client.post("/v1/files/upload",
                     files={"file": ("stock.mp4", clip, "video/mp4")}).json()
    spec = {"canvas": {"width": 160, "height": 284, "fps": 24},
            "captions": {"enabled": False}, "outputs": ["video"],
            "scenes": [{"text": "with my own footage",
                        "audio": _uri(_tone(tmp_path / "own.wav", 0.8)),
                        "visuals": [{"type": "video", "src": up["path"]}]}]}
    doc = _wait(client, client.post("/v1/video/compose", json=spec).json()["id"])
    assert doc["status"] == "done", doc
    assert doc["warnings"] == []          # resolved locally, nothing downloaded


def test_compose_and_uploads_are_never_forwarded_to_a_remote():
    """A pod has ffmpeg but not this install's models or files: composition and
    uploads must stay here even when a video capability is remoted."""
    from codai.api.remote_gateway import capability_for, resolve_target
    assert capability_for("/v1/video/compose") == "video"     # by prefix…
    for path in ("/v1/video/compose", "/v1/video/compose/cmp_1",
                 "/v1/video/compose/cmp_1/cancel", "/v1/files/upload",
                 "/v1/files/blob/" + "a" * 64):
        assert resolve_target(path, "POST", "", b"", "application/json") is None, path


def test_a_cloned_voice_profile_is_used_for_narration(tmp_path, files_dir, monkeypatch):
    """engine=clone goes through /v1/audio/clone, whose answer is a url when the
    install has an output directory and inline audio when it does not."""
    client, compose_mod = _client(files_dir)
    monkeypatch.setattr(compose_mod, "_model_ids", lambda mtype: [])
    monkeypatch.setattr(compose_mod, "_voice_available", lambda req: True)
    wav = Path(_tone(tmp_path / "cloned.wav", 0.6)).read_bytes()
    (files_dir / "cloned-out.wav").write_bytes(wav)
    seen = []

    async def fake_clone(req, http_request=None):
        seen.append({"text": req.text, "voice_name": req.voice_name, "speed": req.speed})
        return {"created": 0, "data": [{"url": "/v1/files/cloned-out.wav"}], "engine": "f5"}

    import codai.api.voice_clone as vc
    monkeypatch.setattr(vc, "clone_voice", fake_clone)
    spec = {"canvas": {"width": 160, "height": 284, "fps": 24},
            "voice": {"engine": "clone", "voice_name": "nextime", "speed": 0.9},
            "captions": {"enabled": False}, "outputs": ["video", "narration"],
            "scenes": [{"text": "in my own voice", "visuals": [{"type": "color"}]}]}
    doc = _wait(client, client.post("/v1/video/compose", json=spec).json()["id"])
    assert doc["status"] == "done", doc
    assert seen == [{"text": "in my own voice", "voice_name": "nextime", "speed": 0.9}]
    # The scene is as long as the cloned clip (0.6s) plus the default padding.
    assert doc["result"]["duration"] == pytest.approx(0.75, abs=0.1)
