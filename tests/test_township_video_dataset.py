"""The SimpleTuner dataset the township tool builds from its own renders.

LongCat LoRA training refuses a folder of stills — it wants captioned video clips.
The tool has already rendered exactly that: every clip came FROM a prompt, kept
beside it in videos/prompts.json. These check the dataset assembled from them is
one SimpleTuner will accept, and that the refusals stay refusals.
"""
import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "tw_ds", ROOT / "tools" / "gen_township_fighters.py")
TW = importlib.util.module_from_spec(_spec)
sys.modules["tw_ds"] = TW
_spec.loader.exec_module(TW)


def _render(tmp_path, clips, outcomes=(), frames=120):
    """A fake township output folder: a plan plus the mp4s it says were rendered."""
    videos = tmp_path / "videos"
    videos.mkdir(parents=True)
    plan = {"fps": 15, "fight_plan": [], "outcome_plan": []}
    match = {"match_name": "match_a_vs_b", "env": "the_yard", "clips": []}
    for idx, fighters in enumerate(clips):
        match["clips"].append({"idx": idx, "nf": frames, "fighters": list(fighters),
                               "prompt": f"clip {idx} with {'+'.join(fighters)}"})
        (videos / f"match_a_vs_b_clip{idx:02d}.mp4").write_bytes(b"\x00")
    plan["fight_plan"].append(match)
    for fighter, opponent, outcome in outcomes:
        plan["outcome_plan"].append({"match_name": "match_a_vs_b", "fighter": fighter,
                                     "opponent": opponent, "outcome": outcome,
                                     "env": "the_yard", "nf": frames,
                                     "prompt": f"{fighter} {outcome}"})
        (videos / f"match_a_vs_b_{fighter}_{outcome}.mp4").write_bytes(b"\x00")
    (videos / "prompts.json").write_text(json.dumps(plan))
    return tmp_path


def test_a_clip_is_paired_with_the_prompt_it_was_rendered_from(tmp_path):
    out = _render(tmp_path, [("a",), ("a", "b")])
    found = TW._dataset_clip_sources(out)
    assert [f["prompt"] for f in found] == ["clip 0 with a", "clip 1 with a+b"]


def test_a_plan_entry_with_no_rendered_file_is_not_offered(tmp_path):
    out = _render(tmp_path, [("a",), ("a",)])
    (out / "videos" / "match_a_vs_b_clip01.mp4").unlink()
    assert len(TW._dataset_clip_sources(out)) == 1


def test_upscaled_derivatives_are_never_picked_up(tmp_path):
    """_2x/_2xfps files are upscaled and frame-interpolated output, not what the
    model generated; training on them teaches the upscaler's artefacts."""
    out = _render(tmp_path, [("a",)])
    (out / "videos" / "match_a_vs_b_clip00_2x_2xfps.mp4").write_bytes(b"\x00")
    sources = TW._dataset_clip_sources(out)
    assert len(sources) == 1
    assert "2x" not in sources[0]["path"].name


def test_an_outcome_is_training_data_for_both_fighters(tmp_path):
    out = _render(tmp_path, [], outcomes=[("a", "b", "ko_win")])
    got = TW._dataset_clip_sources(out)
    assert len(got) == 1
    assert set(got[0]["fighters"]) == {"a", "b"}


def test_a_dataset_holds_only_the_clips_that_profile_appears_in(tmp_path):
    out = _render(tmp_path, [("a",), ("b",), ("a", "b")])
    cfg, why = TW.build_video_dataset(out, "character", "a", "slug",
                                      num_frames=93, min_clips=1)
    assert cfg is not None, why
    clips = sorted((cfg.parent / "clips").glob("*.mp4"))
    assert len(clips) == 2
    captions = sorted(p.read_text().strip() for p in (cfg.parent / "clips").glob("*.txt"))
    assert captions == ["clip 0 with a", "clip 2 with a+b"]


def test_an_environment_dataset_selects_by_location(tmp_path):
    out = _render(tmp_path, [("a",), ("b",)])
    cfg, _ = TW.build_video_dataset(out, "environment", "the_yard", "slug",
                                    num_frames=93, min_clips=1)
    assert len(list((cfg.parent / "clips").glob("*.mp4"))) == 2
    assert TW.build_video_dataset(out, "environment", "elsewhere", "slug",
                                  num_frames=93, min_clips=1)[0] is None


def test_the_config_is_the_shape_simpletuner_documents(tmp_path):
    out = _render(tmp_path, [("a",), ("a",)])
    cfg, _ = TW.build_video_dataset(out, "character", "a", "slug",
                                    num_frames=93, resolution=480, min_clips=1)
    backends = json.loads(cfg.read_text())
    video = [b for b in backends if b["dataset_type"] == "video"]
    embeds = [b for b in backends if b["dataset_type"] == "text_embeds"]
    assert len(video) == 1 and len(embeds) == 1
    # Exactly one text_embeds backend must carry default: true.
    assert sum(1 for b in embeds if b.get("default")) == 1
    v = video[0]
    assert v["type"] == "local"
    assert v["caption_strategy"] == "textfile"       # the .txt beside each clip
    assert v["video"]["num_frames"] == 93
    assert v["video"]["min_frames"] == 93
    assert pathlib.Path(v["instance_data_dir"]).is_dir()
    assert pathlib.Path(v["cache_dir_vae"]).is_dir()
    assert pathlib.Path(embeds[0]["cache_dir"]).is_dir()


def test_clips_are_symlinked_not_copied(tmp_path):
    """A fighter is in dozens of clips; copying would duplicate the whole render
    once per fighter."""
    out = _render(tmp_path, [("a",), ("a",)])
    cfg, _ = TW.build_video_dataset(out, "character", "a", "slug",
                                    num_frames=93, min_clips=1)
    assert all(p.is_symlink() for p in (cfg.parent / "clips").glob("*.mp4"))


def test_clips_shorter_than_a_segment_are_refused_with_a_reason(tmp_path):
    """SimpleTuner silently drops a clip below min_frames, so an all-short dataset
    would train on nothing at all."""
    out = _render(tmp_path, [("a",), ("a",)], frames=50)
    cfg, why = TW.build_video_dataset(out, "character", "a", "slug",
                                      num_frames=93, min_clips=1)
    assert cfg is None
    assert "shorter than 93" in why and "longest 50" in why


def test_too_few_clips_is_refused_rather_than_trained_on(tmp_path):
    out = _render(tmp_path, [("a",)])
    cfg, why = TW.build_video_dataset(out, "character", "a", "slug",
                                      num_frames=93, min_clips=8)
    assert cfg is None and "need 8" in why


def test_nothing_rendered_yet_says_so(tmp_path):
    (tmp_path / "videos").mkdir()
    cfg, why = TW.build_video_dataset(tmp_path, "character", "a", "slug")
    assert cfg is None and "no rendered clips" in why


def test_a_rebuild_drops_clips_that_no_longer_exist(tmp_path):
    out = _render(tmp_path, [("a",), ("a",)])
    cfg, _ = TW.build_video_dataset(out, "character", "a", "slug",
                                    num_frames=93, min_clips=1)
    assert len(list((cfg.parent / "clips").glob("*.mp4"))) == 2
    (out / "videos" / "match_a_vs_b_clip01.mp4").unlink()
    cfg, _ = TW.build_video_dataset(out, "character", "a", "slug",
                                    num_frames=93, min_clips=1)
    links = list((cfg.parent / "clips").glob("*.mp4"))
    assert len(links) == 1 and links[0].resolve().exists()


def test_the_default_frame_count_obeys_the_vae_rule():
    assert (TW.DATASET_FRAMES - 1) % 4 == 0


@pytest.mark.parametrize("model,expected", [
    ("longcat", True), ("meituan-longcat/LongCat-Video", True),
    ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", False), ("", False), (None, False),
])
def test_longcat_detection(model, expected):
    assert TW._is_longcat_model(model) is expected


def test_both_training_paths_build_a_dataset_for_longcat():
    """The batch stage (step 4b) and the per-profile Train button are separate code
    paths. Only the first built a dataset, so training one fighter from the
    Characters page still reached the server with no dataset_config and was
    refused — the same error, from the button nobody had wired.
    """
    src = (ROOT / "tools" / "gen_township_fighters.py").read_text(encoding="utf-8")
    # the batch stage
    assert "dataset_config=(str(dataset) if dataset else None)" in src
    # the per-profile button
    assert 'kwargs["dataset_config"] = str(_ds)' in src
    assert src.count("build_video_dataset(") >= 3   # definition + both call sites


def test_the_per_profile_path_explains_what_to_do_when_it_cannot_build_one():
    src = (ROOT / "tools" / "gen_township_fighters.py").read_text(encoding="utf-8")
    assert "cannot train a LongCat video LoRA" in src
    assert "Render some match clips first" in src
