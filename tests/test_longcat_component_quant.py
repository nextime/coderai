"""Quantising the components that upstream does not quantise for us.

The DiT has upstream's own INT8 path. The UMT5-XXL text encoder does not, and at bf16
it is ~11 GB — 45% of why a 13.6 GB INT8 DiT still overflows a 24 GB card (13.6 + 11.2
+ VAE = 25.1 GB before activations). Quantised it is ~5.6 GB at INT8 or ~2.8 GB at NF4,
which is the difference between offloading and not.
"""
import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMON = ROOT / "tools" / "longcat_common.py"
SERVICE = ROOT / "tools" / "longcat_service.py"
WORKER = ROOT / "codai" / "api" / "longcat_worker.py"


@pytest.fixture
def LC():
    spec = importlib.util.spec_from_file_location("lc_cq", COMMON)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ the contract

def test_none_means_keep_the_pipeline_dtype(LC):
    assert LC.component_quant_bpe("none") is None
    assert LC.component_quant_bpe("") is None
    assert LC.component_quant_bpe(None) is None


@pytest.mark.parametrize("kind,bpe", [("int8", 1.0), ("nf4", 0.5), ("fp4", 0.5)])
def test_the_widths_are_what_the_card_will_hold(LC, kind, bpe):
    assert LC.component_quant_bpe(kind) == bpe


def test_an_unknown_setting_is_refused_with_the_options(LC):
    problems = LC.component_quant_problems("text_encoder_quant", "4bit-ish")
    assert problems and "int8" in problems[0] and "nf4" in problems[0]


def test_a_valid_setting_passes(LC):
    for kind in ("none", "int8", "nf4", "fp4", "NF4", " int8 "):
        assert LC.component_quant_problems("text_encoder_quant", kind) == []


# ------------------------------------------------------------------ the arithmetic

def test_quantising_the_encoder_is_what_makes_it_fit(LC):
    """The whole justification, in numbers: 22 GB of fp32 on disk becomes 11 GB at bf16
    and does not fit beside the DiT; at NF4 it does."""
    disk_gb, load_bpe, stored_bpe = 22.0, 2.0, 4.0
    bf16 = disk_gb * (load_bpe / stored_bpe)
    nf4 = disk_gb * (LC.component_quant_bpe("nf4") / stored_bpe)
    int8 = disk_gb * (LC.component_quant_bpe("int8") / stored_bpe)
    dit, vae = 13.6, 0.25
    assert dit + bf16 + vae > 23.5, "this is the configuration that OOMs"
    assert dit + int8 + vae < 23.5
    assert dit + nf4 + vae < 20


# ------------------------------------------------------------------ the service

def test_the_service_refuses_a_bad_value_before_loading_anything():
    src = SERVICE.read_text(encoding="utf-8")
    assert "component_quant_problems" in src
    main = src[src.index('ap.add_argument("--text-encoder-quant"'):]
    assert "SystemExit" in main, "a typo must fail at startup, not after a 7-minute load"


def test_a_quantised_encoder_is_never_offloaded():
    """bitsandbytes quantises as weights land on CUDA and is not built to move them
    back; it is also small enough that nothing needs to take turns."""
    src = SERVICE.read_text(encoding="utf-8")
    body = src[src.index("def _place_pipeline"):src.index("def _load_quantized_base_dit")]
    assert "_encoder_is_quantised(pipe)" in body
    # it must return before reaching the swap wrapper
    assert body.index("_encoder_is_quantised(pipe)") < body.index("swap=True")


def test_quantisation_is_detected_from_the_model_not_the_config():
    """A pipeline loaded quantised must not be offloaded by a config that changed since."""
    src = SERVICE.read_text(encoding="utf-8")
    body = src[src.index("def _encoder_is_quantised"):src.index("def _quant_config")]
    assert "Linear4bit" in body and "Linear8bitLt" in body
    assert "text_encoder_quant" not in body


def test_missing_bitsandbytes_is_a_clear_error():
    src = SERVICE.read_text(encoding="utf-8")
    body = src[src.index("def _quant_config"):src.index("def _wrap_encode_prompt")]
    assert "needs bitsandbytes" in body
    assert "requirements-longcat.txt" in body, "say where to fix it"


def test_a_quantised_load_passes_a_device_map():
    """bitsandbytes quantises on the way to the card; without device_map the weights
    never get there and the config is silently ignored."""
    src = SERVICE.read_text(encoding="utf-8")
    assert "device_map" in src[src.index("te_quant ="):src.index("UMT5EncoderModel.from_pretrained")+200]


# ------------------------------------------------------------------ plumbing

def test_it_is_configurable_per_model_and_server_wide():
    src = WORKER.read_text(encoding="utf-8")
    assert '"--text-encoder-quant"' in src
    body = src[src.index('te_q = str(config.get("text_encoder_quant")'):]
    # per-model first: one checkpoint trading prompt fidelity for room must not force
    # that trade on every other model on the box
    assert body.index('config.get("text_encoder_quant")') < \
        body.index('getattr(sec, "text_encoder_quant"')


def test_it_is_documented_in_the_config_keys():
    src = WORKER.read_text(encoding="utf-8")
    assert "text_encoder_quant" in src[:src.index("import collections")]


def test_bitsandbytes_is_pinned_and_verified_in_both_images():
    reqs = (ROOT / "requirements-longcat.txt").read_text(encoding="utf-8")
    assert "bitsandbytes==" in reqs
    pod = (ROOT / "packaging/runpod/Dockerfile.capability-video-longcat").read_text()
    assert "bitsandbytes" in pod
    smoke = (ROOT / "packaging/linux/smoke_test_services.sh").read_text()
    assert "bitsandbytes" in smoke


# ------------------------------------------------------------------ shape of the helpers
# Inserting these functions by anchor put one of them between @contextlib.contextmanager
# and the function it was meant to decorate, so _encoder_is_quantised returned a context
# manager (truthy — every pipeline looked quantised) and _bsa_for stopped being one.
# Cheap guards, because the symptom was five unrelated test failures.

def test_the_quantisation_check_returns_a_bool(service_mod):
    import inspect
    fn = service_mod._encoder_is_quantised
    assert not hasattr(fn, "__wrapped__"), "something is decorating it"
    assert not inspect.isgeneratorfunction(fn)

    class _NoModules:
        pass

    class _Pipe:
        text_encoder = _NoModules()

    assert service_mod._encoder_is_quantised(_Pipe()) is False
    assert service_mod._encoder_is_quantised(type("P", (), {"text_encoder": None})()) is False


def test_the_bsa_guard_is_a_context_manager(service_mod):
    import contextlib
    cm = service_mod._bsa_for(type("P", (), {"dit": None})(), {})
    assert hasattr(cm, "__enter__") and hasattr(cm, "__exit__")
    with cm as active:
        assert active is False          # no dit, so nothing to tile


@pytest.fixture
def service_mod(monkeypatch):
    spec = importlib.util.spec_from_file_location("lc_svc_shape", SERVICE)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    return mod
