"""A repository may run its own code here only if it was allow-listed.

Some published pipelines ship the pipeline class in the repo rather than in
diffusers. MOSS-Image's own `model_index.json` says so:

    "_class_name": "MOSSImagePipeline",
    "transformer": ["pipeline_moss_image", "MOSSImageTransformer"],
    "scheduler":   ["pipeline_moss_image", "MOSSImageScheduler"]

Loading it means executing `pipeline_moss_image` from the repository. So the
capability is necessary -- and blanket `trust_remote_code=True` is not, because
that would run arbitrary Python from any checkpoint an operator named. Hence an
allow-list that starts EMPTY, and a refusal that names the model rather than a
quiet load without the pipeline it asked for.
"""
import pytest

from codai.config import Config, ModelsConfig
from codai.models import remote_code


def _cfg(allow=None):
    c = Config()
    c.models = ModelsConfig(trust_remote_code=list(allow or []))
    return c


# --------------------------------------------------------------------------- #
#  the default is no
# --------------------------------------------------------------------------- #

def test_nothing_is_trusted_by_default():
    assert ModelsConfig().trust_remote_code == []
    assert remote_code.allowed(_cfg(), "OpenMOSS-Team/MOSS-Image-SFT") is False


def test_an_exact_id_is_honoured():
    cfg = _cfg(["OpenMOSS-Team/MOSS-Image-SFT"])
    assert remote_code.allowed(cfg, "OpenMOSS-Team/MOSS-Image-SFT") is True
    assert remote_code.allowed(cfg, "OpenMOSS-Team/Something-Else") is False


def test_a_publisher_prefix_covers_its_family():
    """`lodestones/*` covers Kromoss and its siblings, which is how an install
    trusts a publisher rather than re-listing every fine-tune."""
    cfg = _cfg(["lodestones/*"])
    assert remote_code.allowed(cfg, "lodestones/Kromoss") is True
    assert remote_code.allowed(cfg, "lodestones/Kroma") is True
    assert remote_code.allowed(cfg, "someoneelse/Kromoss") is False


def test_case_does_not_decide_it():
    cfg = _cfg(["OpenMOSS-Team/MOSS-Image-SFT"])
    assert remote_code.allowed(cfg, "openmoss-team/moss-image-sft") is True


def test_a_star_is_a_deliberate_any():
    assert remote_code.allowed(_cfg(["*"]), "whoever/whatever") is True


def test_an_empty_name_is_never_allowed():
    assert remote_code.allowed(_cfg(["*"]), "") is False
    assert remote_code.allowed(_cfg(["*"]), None) is False


def test_a_string_instead_of_a_list_still_works():
    """Config files get hand-edited; one id written as a string is obvious
    intent, and refusing it on a technicality would just confuse."""
    assert remote_code.allowed(_cfg("lodestones/*"), "lodestones/Kromoss") is True


# --------------------------------------------------------------------------- #
#  refusing, and what the refusal says
# --------------------------------------------------------------------------- #

def test_a_model_that_needs_remote_code_is_refused_by_name():
    with pytest.raises(PermissionError, match="MOSS-Image-SFT"):
        remote_code.require(_cfg(), "OpenMOSS-Team/MOSS-Image-SFT")


def test_the_refusal_names_the_setting_and_the_value_to_add():
    with pytest.raises(PermissionError) as e:
        remote_code.require(_cfg(), "OpenMOSS-Team/MOSS-Image-SFT")
    msg = str(e.value)
    assert "models.trust_remote_code" in msg
    assert "OpenMOSS-Team/*" in msg, "the operator should not have to guess the syntax"
    assert "executes Python" in msg, "and should know what they are agreeing to"


# --------------------------------------------------------------------------- #
#  the loader kwargs
# --------------------------------------------------------------------------- #

def test_a_model_asking_for_nothing_changes_nothing():
    assert remote_code.kwargs_for(_cfg(), "org/plain-model", None) == {}
    assert remote_code.kwargs_for(_cfg(), "org/plain-model", "") == {}


def test_an_allowed_model_gets_both_kwargs():
    got = remote_code.kwargs_for(_cfg(["OpenMOSS-Team/*"]),
                                 "OpenMOSS-Team/MOSS-Image-SFT",
                                 "pipeline_moss_image")
    assert got == {"custom_pipeline": "pipeline_moss_image",
                   "trust_remote_code": True}


def test_asking_without_permission_raises_rather_than_loading_plain():
    """Loading it without its pipeline would succeed and serve something that
    is not the model."""
    with pytest.raises(PermissionError):
        remote_code.kwargs_for(_cfg(), "OpenMOSS-Team/MOSS-Image-SFT",
                               "pipeline_moss_image")


# --------------------------------------------------------------------------- #
#  and the image loader has to use it
# --------------------------------------------------------------------------- #

def test_the_image_loader_asks_before_any_weights_move():
    src = open("codai/api/images.py").read()
    assert "remote_code.kwargs_for(" in src
    head = src.index("_img_remote = {}")
    # Resolved before the pipeline load, not during it.
    assert head < src.index("build_single_file_pipeline(model_name")


def test_a_permission_error_is_not_swallowed_by_the_loader():
    """The surrounding try/except logs and continues for ordinary surprises;
    a refusal must not be one of them."""
    src = open("codai/api/images.py").read()
    head = src.index("_img_remote = {}")
    # Bounded by the next statement after the block, not by a character count:
    # a fixed window silently stops covering the code it was written for the
    # moment anything is added inside it, which is how this test broke when the
    # declared-class check went in.
    body = src[head:src.index("_img_comp_models", head)]
    assert "except PermissionError:" in body and "raise" in body


def test_both_load_paths_pass_the_kwargs():
    src = open("codai/api/images.py").read()
    assert src.count("_xtra.update(_img_remote)") == 2


def test_such_a_model_is_not_tried_as_sdxl_first():
    """Every custom-pipeline model would fail that attempt, burying the real
    error under a class mismatch."""
    src = open("codai/api/images.py").read()
    assert "if pipeline is None and _img_remote:" in src


def test_the_field_is_savable_from_the_admin_page():
    assert '"custom_pipeline",' in open("codai/admin/routes.py").read()


# ---------------------------------------------------------------------------
# A repo that declares its own class without any card saying so
# ---------------------------------------------------------------------------

def test_a_repo_declaring_an_unavailable_class_is_detected(monkeypatch):
    """MOSS-Image and Kromoss, exactly as they failed: no card named a
    custom_pipeline, so nothing fired, and the load died on
    'module diffusers has no attribute MOSSImagePipeline'."""
    from codai.models import remote_code as R
    monkeypatch.setattr(R, "declared_pipeline_class",
                        lambda repo: "MOSSImagePipeline")
    kw = R.kwargs_for_declared_class(_cfg(["openmoss-team/*"]),
                                     "OpenMOSS-Team/MOSS-Image-SFT")
    assert kw["trust_remote_code"] is True
    assert kw["custom_pipeline"] == "OpenMOSS-Team/MOSS-Image-SFT"
    assert kw["_declared_class"] == "MOSSImagePipeline"


def test_such_a_repo_is_refused_by_name_when_not_allow_listed(monkeypatch):
    """The refusal has to name the setting — that is the whole reason the
    allow-list is spelled out rather than implied."""
    from codai.models import remote_code as R
    monkeypatch.setattr(R, "declared_pipeline_class",
                        lambda repo: "MOSSImagePipeline")
    with pytest.raises(PermissionError) as ei:
        R.kwargs_for_declared_class(_cfg([]), "lodestones/Kromoss")
    msg = str(ei.value)
    assert "models.trust_remote_code" in msg
    assert "lodestones/*" in msg
    assert "MOSSImagePipeline" in msg


def test_an_ordinary_repo_pays_nothing(monkeypatch):
    """The class diffusers already has must not trigger any of this: every
    normal image model goes through this code path."""
    from codai.models import remote_code as R
    monkeypatch.setattr(R, "declared_pipeline_class",
                        lambda repo: "StableDiffusionXLPipeline")
    assert R.kwargs_for_declared_class(_cfg([]), "any/sdxl") == {}


def test_a_repo_that_declares_nothing_is_left_alone(monkeypatch):
    from codai.models import remote_code as R
    monkeypatch.setattr(R, "declared_pipeline_class", lambda repo: None)
    assert R.kwargs_for_declared_class(_cfg([]), "any/model") == {}


def test_the_marker_never_reaches_from_pretrained():
    """_declared_class is ours, for the log line. Passing it on would be a
    TypeError from diffusers at the worst possible moment."""
    src = open("codai/api/images.py").read()
    head = src.index("_img_remote = {}")
    body = src[head:src.index("_img_comp_models", head)]
    assert '_img_remote.pop("_declared_class"' in body


def test_the_loader_asks_the_repository_when_no_card_did():
    src = open("codai/api/images.py").read()
    assert "kwargs_for_declared_class(" in src
