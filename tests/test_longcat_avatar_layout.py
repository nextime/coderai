"""An avatar checkpoint is a transformer and an audio encoder, nothing else.

LongCat-Video-Avatar-1.5 ships base_model/, base_model_int8/, lora/dmd_lora, scheduler/
and whisper-large-v3/ — and NO tokenizer, text_encoder or vae. Its README says so in
its frontmatter (`base_model: meituan-longcat/LongCat-Video`) and upstream's own
instructions download both repos. Validating it against the base checkpoint's layout
reports four missing directories that were never supposed to be there.
"""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMON = ROOT / "tools" / "longcat_common.py"
SERVICE = ROOT / "tools" / "longcat_service.py"


@pytest.fixture
def LC():
    spec = importlib.util.spec_from_file_location("lc_avatar_layout", COMMON)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _base_checkpoint(path):
    for sub in ("tokenizer", "text_encoder", "vae", "scheduler", "dit"):
        (path / sub).mkdir(parents=True)
    return path


def _avatar_checkpoint(path, int8=True, bf16=True):
    (path / "scheduler").mkdir(parents=True)
    if bf16:
        (path / "base_model").mkdir()
    if int8:
        (path / "base_model_int8").mkdir()
    (path / "lora").mkdir()
    (path / "lora" / "dmd_lora.safetensors").write_bytes(b"")
    return path


# ---------------------------------------------------------------- family detection

@pytest.mark.parametrize("ref,want", [
    ("meituan-longcat/LongCat-Video", ""),
    ("meituan-longcat/LongCat-Video-Avatar", "avatar"),
    ("meituan-longcat/LongCat-Video-Avatar-1.5", "avatar-1.5"),
    ("/AI/x/models--meituan-longcat--LongCat-Video-Avatar-1.5/snapshots/a", "avatar-1.5"),
])
def test_the_family_comes_from_the_path(LC, ref, want):
    assert LC.family_of(ref) == want


# ---------------------------------------------------------------- the dit subfolder

def test_the_avatar_transformer_is_under_base_model(LC):
    """The base checkpoint uses dit/; loading an avatar with subfolder="dit" fails."""
    assert LC.dit_subdir("avatar-1.5") == "base_model"
    assert LC.dit_subdir("avatar") == "base_model"
    assert LC.dit_subdir("") == "dit"


# ---------------------------------------------------------------- shared components

def test_the_base_checkpoint_shares_with_itself(LC, tmp_path):
    base = _base_checkpoint(tmp_path / "base")
    assert LC.shared_checkpoint(str(base)) == str(base)


def test_an_avatar_borrows_from_the_configured_checkpoint(LC, tmp_path):
    base = _base_checkpoint(tmp_path / "base")
    avatar = _avatar_checkpoint(tmp_path / "LongCat-Video-Avatar-1.5")
    assert LC.shared_checkpoint(str(avatar), str(base)) == str(base)


def test_an_avatar_with_nothing_configured_looks_for_the_upstream_base(LC, tmp_path,
                                                                      monkeypatch):
    """Unresolvable comes back as the repo id, which the validator then names."""
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty"))
    # resolve_checkpoint also falls back to ~/.cache/huggingface/hub, which on this
    # machine is the real store — point HOME somewhere empty so the test is about the
    # fallback and not about what happens to be downloaded.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    avatar = _avatar_checkpoint(tmp_path / "LongCat-Video-Avatar-1.5")
    assert LC.shared_checkpoint(str(avatar)) == LC.BASE_REPO


# ---------------------------------------------------------------- validation

def test_a_valid_avatar_checkpoint_passes_with_a_base(LC, tmp_path):
    base = _base_checkpoint(tmp_path / "base")
    avatar = _avatar_checkpoint(tmp_path / "LongCat-Video-Avatar-1.5")
    assert LC.checkpoint_problems(str(avatar), (), "avatar-1.5", str(base)) == []


def test_the_avatar_is_not_asked_for_components_it_never_ships(LC, tmp_path):
    """The bug this fixes: four 'missing' directories that are not part of the repo."""
    base = _base_checkpoint(tmp_path / "base")
    avatar = _avatar_checkpoint(tmp_path / "LongCat-Video-Avatar-1.5")
    problems = " ".join(LC.checkpoint_problems(str(avatar), (), "avatar-1.5", str(base)))
    for never_there in ("tokenizer/", "text_encoder/", "vae/", "dit/"):
        assert f"missing {never_there} in {avatar}" not in problems


def test_an_avatar_without_any_transformer_is_caught(LC, tmp_path):
    base = _base_checkpoint(tmp_path / "base")
    avatar = _avatar_checkpoint(tmp_path / "LongCat-Video-Avatar-1.5",
                                int8=False, bf16=False)
    problems = LC.checkpoint_problems(str(avatar), (), "avatar-1.5", str(base))
    assert any("needs at least one transformer" in p for p in problems)


@pytest.mark.parametrize("int8,bf16", [(True, False), (False, True)])
def test_either_transformer_alone_is_enough(LC, tmp_path, int8, bf16):
    """We deliberately skip base_model/ on a 24 GB card — int8 alone must validate."""
    base = _base_checkpoint(tmp_path / "base")
    avatar = _avatar_checkpoint(tmp_path / "LongCat-Video-Avatar-1.5",
                                int8=int8, bf16=bf16)
    assert LC.checkpoint_problems(str(avatar), (), "avatar-1.5", str(base)) == []


def test_a_missing_base_checkpoint_is_reported_as_such(LC, tmp_path):
    """Not as "missing tokenizer/" inside the avatar, which sends you looking in the
    wrong repo."""
    avatar = _avatar_checkpoint(tmp_path / "LongCat-Video-Avatar-1.5")
    problems = LC.checkpoint_problems(str(avatar), (), "avatar-1.5",
                                      str(tmp_path / "nope"))
    assert any("base checkpoint" in p for p in problems)


def test_an_incomplete_base_checkpoint_names_the_base(LC, tmp_path):
    base = tmp_path / "base"
    (base / "tokenizer").mkdir(parents=True)      # vae and text_encoder missing
    avatar = _avatar_checkpoint(tmp_path / "LongCat-Video-Avatar-1.5")
    problems = LC.checkpoint_problems(str(avatar), (), "avatar-1.5", str(base))
    assert any("text_encoder/ in the base checkpoint" in p for p in problems)
    assert any("vae/ in the base checkpoint" in p for p in problems)


def test_the_base_checkpoint_still_validates_the_old_way(LC, tmp_path):
    base = _base_checkpoint(tmp_path / "base")
    assert LC.checkpoint_problems(str(base), (), "", str(base)) == []
    (base / "vae").rmdir()
    assert any("missing vae/" in p for p in LC.checkpoint_problems(str(base), (), ""))


# ---------------------------------------------------------------- service wiring

def test_the_service_loads_shared_components_from_the_shared_root():
    src = SERVICE.read_text(encoding="utf-8")
    body = src[src.index("def load_pipeline"):src.index("def _cp_split_hw")]
    for component in ("tokenizer", "text_encoder", "vae"):
        assert f'shared, subfolder="{component}"' in body, component
    # the scheduler is the avatar's OWN
    assert 'root, subfolder="scheduler"' in body
    assert "LC.dit_subdir(family)" in body
