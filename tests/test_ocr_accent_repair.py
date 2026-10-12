"""Rebuild a surya reply's text from the model's own bytes when the text is corrupt.

The measured fault, on the Aruba install, 2026-10-12: a page of crisp synthetic
Italian (vector glyphs, 34pt, 250 DPI) came back from surya-ocr-2 with 29 U+FFFD --
one per accented letter, zero accents surviving -- while

  * the model's own byte stream decoded cleanly as UTF-8 (the byte trace reported
    `conclusive: true`, no findings, `blame: "the serving path"`),
  * the model reported ~96% confidence on every line,
  * chandra read the identical pixels with all 12 accents intact,
  * and it was ONE model call, so concurrency was not involved.

So the characters are lost after the model and before `message.content`, which is the
field surya reads -- vLLM 0.27.1's detokenization of surya's byte-level vocabulary,
where surya itself pins `vllm/vllm-openai:v0.20.1`. The correct bytes arrive in the
same response's logprobs, so the reply is rebuilt from them.

Rewriting a model's output is a serious thing to do, so these tests are mostly about
the guards: the repair must be impossible to apply to anything but this exact fault.
"""

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _worker_module():
    path = ROOT / "codai" / "ocr" / "workers" / "ocr_worker.py"
    spec = importlib.util.spec_from_file_location("ocr_worker_repair_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _worker_module()
UN = "�"


def _tokens_for(text):
    """Byte-level tokens the way surya-ocr-2 emits them: one token per UTF-8 byte."""
    return [{"token": "?", "bytes": [b], "logprob": -0.01} for b in text.encode()]


# ------------------------------------------------------------------ the repair #
def test_the_accents_come_back():
    good = "La società ha facoltà di più attività"
    broken = "La societ? ha facolt? di pi? attivit?".replace("?", UN)
    fixed, n = W.repair_content(broken, _tokens_for(good))
    assert fixed == good
    assert n == 4


def test_it_recovers_the_real_production_line():
    good = ("La società ha facoltà di più attività: perciò è così che "
            "l'unità perché già oltre.")
    broken = "".join(UN if c in "àèéìòù" else c for c in good)
    fixed, n = W.repair_content(broken, _tokens_for(good))
    assert fixed == good
    assert n == broken.count(UN) == 10


def test_uppercase_accents_too():
    good = "PIÙ FACOLTÀ PERCHÉ COSÌ È SOCIETÀ"
    broken = "".join(UN if c in "ÀÈÉÌÒÙ" else c for c in good)
    fixed, _ = W.repair_content(broken, _tokens_for(good))
    assert fixed == good


def test_html_survives_the_repair():
    """surya replies in HTML, and the repair happens before it is parsed — so the
    tags have to come through untouched or the blocks fall apart."""
    good = "<p>La società è <b>più</b> antica</p>"
    broken = good.replace("à", UN).replace("è", UN).replace("ù", UN)
    fixed, _ = W.repair_content(broken, _tokens_for(good))
    assert fixed == good


def test_the_count_is_what_was_recovered():
    good = "àèì"
    fixed, n = W.repair_content(UN * 3, _tokens_for(good))
    assert fixed == good and n == 3


# ------------------------------------------------------------------ the guards #
def test_clean_text_is_left_completely_alone():
    """The common case. No marker, no repair, not even a reassembly attempt."""
    assert W.repair_content("La società è antica", _tokens_for("anything else")) \
        == (None, 0)


def test_a_mismatched_ascii_skeleton_is_refused():
    """THE guard that makes this a repair and not a substitution. If the bytes
    decode to different TEXT — a truncation, the wrong block, a shifted offset —
    the skeletons differ and the original is kept, markers and all."""
    broken = "La societ" + UN + " ha"
    other = "Un documento completamente diverso"
    assert W.repair_content(broken, _tokens_for(other)) == (None, 0)


def test_a_repair_may_only_change_non_ascii_characters():
    """Stated as a property: whatever comes back, its ASCII skeleton is identical."""
    good = "Perché la società è così: più unità"
    broken = "".join(UN if ord(c) > 127 else c for c in good)
    fixed, _ = W.repair_content(broken, _tokens_for(good))
    assert W._ascii_skeleton(fixed) == W._ascii_skeleton(broken)


def test_an_undecodable_byte_stream_is_refused():
    """A lone lead byte: the model really did emit something broken, and then the
    markers are honest. Repairing would invent characters."""
    tokens = [{"token": "a", "bytes": [97]}, {"token": "?", "bytes": [195]}]
    assert W.repair_content("a" + UN, tokens) == (None, 0)


def test_a_reassembly_that_still_contains_a_marker_is_refused():
    """The server had already substituted it, so the bytes carry it too — there is
    nothing to recover and replacing like with like would just hide the problem."""
    tokens = _tokens_for("pi" + UN)
    assert W.repair_content("pi" + UN, tokens) == (None, 0)


def test_missing_bytes_mean_no_repair():
    """A server that reports logprobs without a bytes field gives us nothing to
    rebuild from."""
    tokens = [{"token": "p", "logprob": -0.1}, {"token": "?", "logprob": -0.2}]
    assert W.repair_content("p" + UN, tokens) == (None, 0)


def test_no_logprobs_mean_no_repair():
    assert W.repair_content("pi" + UN, None) == (None, 0)
    assert W.repair_content("pi" + UN, []) == (None, 0)


def test_empty_input_is_handled():
    assert W.repair_content("", _tokens_for("x")) == (None, 0)
    assert W.repair_content(None, _tokens_for("x")) == (None, 0)


def test_reassemble_is_strict_about_utf8():
    assert W._reassemble([{"token": "x", "bytes": [195]}]) is None     # lone lead
    assert W._reassemble([{"token": "x", "bytes": [160]}]) is None     # lone cont.
    assert W._reassemble(_tokens_for("società")) == "società"


def test_a_three_byte_character_reassembles():
    """The em dash survived the production fault, so it must survive the repair."""
    assert W._reassemble(_tokens_for("a — b")) == "a — b"


# ----------------------------------------------------------- it is switchable #
def test_the_worker_repairs_by_default():
    assert W.SuryaWorker({})._repair is True
    assert W.SuryaWorker({"repair_accents": False})._repair is False


def test_the_hook_installs_for_the_repair_even_with_the_trace_off():
    """The repair needs the same interception point as the trace, so turning the
    trace off must not disable it."""
    w = W.SuryaWorker({"repair_accents": True, "token_trace": False})
    w._install_trace_hook()        # surya is absent here; must not raise


def test_neither_feature_means_no_patching():
    import inspect
    src = inspect.getsource(W.SuryaWorker._install_trace_hook)
    assert "if not (self._trace or self._repair):" in src


# ------------------------------------------------------- it reports what it did #
def test_a_repaired_page_says_so(monkeypatch):
    """The text is no longer byte-for-byte what the server sent; anyone comparing
    the two needs to know why."""
    from PIL import Image
    w = W.SuryaWorker({"repair_accents": True})
    w._mode = "predictor"

    def _predict(img):
        # The hook registers the repair DURING the call; ocr() clears the list
        # first, so pre-populating it would prove nothing.
        w._repairs.append({"recovered": 4})
        return [{"blocks": [{"html": "<p>società</p>",
                             "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                             "confidence": 0.96, "reading_order": 0}]}]
    monkeypatch.setattr(w, "_predict", _predict)
    page = w.ocr(Image.new("RGB", (10, 10)))
    assert page["accents_repaired"]["blocks"] == 1
    assert page["accents_repaired"]["recovered"] == 4
    assert "per-token bytes" in page["accents_repaired"]["how"]


def test_an_unrepaired_page_says_nothing(monkeypatch):
    from PIL import Image
    w = W.SuryaWorker({"repair_accents": True})
    w._mode = "predictor"
    monkeypatch.setattr(w, "_predict", lambda img: [
        {"blocks": [{"html": "<p>ok</p>", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                     "confidence": 0.9, "reading_order": 0}]}])
    assert "accents_repaired" not in w.ocr(Image.new("RGB", (10, 10)))


def test_repairs_are_cleared_between_pages(monkeypatch):
    from PIL import Image
    w = W.SuryaWorker({"repair_accents": True})
    w._mode = "predictor"
    monkeypatch.setattr(w, "_predict", lambda img: [
        {"blocks": [{"html": "<p>ok</p>", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                     "confidence": 0.9, "reading_order": 0}]}])
    w._repairs.append({"recovered": 1})
    w.ocr(Image.new("RGB", (10, 10)))
    assert w._repairs == []


# ---------------------------------------------------------------- and the wiring #
def test_the_engine_tells_the_worker():
    from codai.config import OcrConfig
    from codai.ocr.surya import SuryaEngine
    assert SuryaEngine(OcrConfig())._worker_opts()["repair_accents"] is True
    cfg = OcrConfig()
    cfg.surya_repair_accents = False
    assert SuryaEngine(cfg)._worker_opts()["repair_accents"] is False


def test_toggling_it_rebuilds_the_pool():
    from codai.config import OcrConfig
    from codai.ocr.manager import OcrManager
    on, off = OcrConfig(), OcrConfig()
    off.surya_repair_accents = False
    assert OcrManager._pool_params(on) != OcrManager._pool_params(off)


def test_it_defaults_on():
    from codai.config import OcrConfig
    assert OcrConfig().surya_repair_accents is True


# ------------------------------------------- option 2: the pinned-vLLM image #
def test_the_vllm_ocr_profile_exists_and_pins_the_version_surya_validates():
    """Surya spawns `vllm/vllm-openai:v0.20.1` (surya/settings.py:96). Serving the
    same weights on 0.27.1 is what loses the accents, so the second half of the
    answer is an image on the version its authors test."""
    import pathlib
    prof = pathlib.Path(__file__).resolve().parents[1] / "packaging/runpod/profiles"
    reqs = (prof / "vllm-ocr.venv-vllm.txt").read_text()
    assert "vllm==0.20.1+cu129" in reqs
    # The torch triple is vLLM 0.20.1's own metadata, not a guess.
    assert "torch==2.11.0+cu129" in reqs
    assert "torchvision==0.26.0+cu129" in reqs
    assert "torchaudio==2.11.0+cu129" in reqs
    # uv, because pip cannot resolve this pin set (as for the 0.27.1 profile).
    assert (prof / "vllm-ocr.venv-vllm.uv").exists()
    # The light core, like the profile it is derived from.
    assert (prof / "vllm-ocr.light").exists()
    # Triton needs a C compiler for its launcher stubs.
    assert "gcc" in (prof / "vllm-ocr.apt").read_text()


def test_the_build_check_refuses_a_silent_version_drift():
    """The whole point of the image is the pin. If uv resolved something newer the
    build has to fail rather than produce an image that looks right."""
    import pathlib
    prof = pathlib.Path(__file__).resolve().parents[1] / "packaging/runpod/profiles"
    check = (prof / "vllm-ocr.venv-vllm.check").read_text()
    assert 'startswith("0.20.")' in check
    # ...but it must not fail the build on the NAME of a private vLLM symbol.
    assert "except ImportError" in check


def test_the_profile_is_discoverable_by_the_builder():
    """Profiles are found by globbing *.txt, so the file's existence is the
    registration — but the .txt must not be mistaken for a venv spec."""
    import pathlib
    prof = pathlib.Path(__file__).resolve().parents[1] / "packaging/runpod/profiles"
    names = sorted(p.stem for p in prof.glob("*.txt")
                   if not p.stem.startswith("core") and ".venv-" not in p.stem)
    assert "vllm-ocr" in names
    assert "vllm" in names          # the 0.27.1 image is untouched


def test_a_model_can_select_the_new_image():
    from codai.api.runpod_worker import (_CODERAI_SERVER_ENGINES,
                                         default_capability_image,
                                         engine_pod_backend)

    class _M:
        engine = "coderai-vllm-ocr"
    assert _CODERAI_SERVER_ENGINES["coderai-vllm-ocr"] == "vllm-ocr"
    assert default_capability_image("vllm-ocr") == \
        "ghcr.io/nextime/coderai-vllm-ocr:latest"
    # It is still coderai's vLLM BACKEND; only the image's vLLM differs.
    assert engine_pod_backend({}, _M()) == "vllm"


def test_the_existing_vllm_image_is_unchanged():
    """A model on `coderai-vllm` must not be quietly moved to the pinned image."""
    from codai.api.runpod_worker import default_capability_image
    assert default_capability_image("vllm") == "ghcr.io/nextime/coderai-vllm:latest"


# ------------------- token ids: the lossless channel on vLLM 0.27.1 #
def test_token_ids_are_read_from_the_vllm_form():
    """vLLM renders a token it cannot JSON-encode as `token_id:N` when launched
    with --return-tokens-as-token-ids. Those are exactly surya's byte-level
    tokens — the ones it otherwise reports as `bytes: []` and U+FFFD."""
    toks = [{"token": "token_id:2131"}, {"token": "token_id:2104"}]
    assert W._token_ids(toks) == [2131, 2104]


def test_rendered_tokens_are_not_mistaken_for_ids():
    assert W._token_ids([{"token": "a"}, {"token": "token_id:5"}]) is None
    assert W._token_ids([{"token": "token_id:x"}]) is None
    assert W._token_ids([]) is None
    assert W._token_ids(None) is None


def test_ids_are_all_or_nothing():
    """A partial id list would decode to a partial page, which is worse than no
    repair at all."""
    assert W._token_ids([{"token": "token_id:1"}, {"token": "b"}]) is None


def test_the_repair_prefers_ids_over_bytes(monkeypatch):
    """On 0.27.1 the bytes are empty for the very tokens that matter, so the id
    route has to be tried first or the repair falls back to a dead end."""
    UN = "�"
    monkeypatch.setattr(W, "_decode_ids", lambda ids: "società" if ids else None)
    toks = [{"token": "token_id:1", "bytes": []}]
    fixed, n = W.repair_content("societ" + UN, toks)
    assert fixed == "società"
    assert n == 1


def test_a_bad_id_decode_falls_back_to_the_byte_route(monkeypatch):
    """The byte route needs no server flag, so it stays as the fallback for any
    stack that reports bytes honestly."""
    UN = "�"
    monkeypatch.setattr(W, "_decode_ids", lambda ids: None)
    good = "società"
    toks = [{"token": "x", "bytes": [b]} for b in good.encode()]
    fixed, _ = W.repair_content("societ" + UN, toks)
    assert fixed == good


def test_an_id_decode_that_still_contains_a_marker_is_rejected(monkeypatch):
    UN = "�"
    monkeypatch.setattr(W, "_decode_ids", lambda ids: "societ" + UN)
    assert W.repair_content("societ" + UN, [{"token": "token_id:1"}]) == (None, 0)


def test_the_skeleton_guard_still_applies_to_the_id_route(monkeypatch):
    """Ids are lossless but the decode could still be of the wrong call."""
    UN = "�"
    monkeypatch.setattr(W, "_decode_ids", lambda ids: "qualcosa di diverso")
    assert W.repair_content("societ" + UN, [{"token": "token_id:1"}]) == (None, 0)


# ------------- the other half: the proxy must stop dropping the field #
def test_the_vllm_backend_forwards_the_token_ids_request():
    """MEASURED: sending `return_tokens_as_token_ids` to a pod changed nothing,
    because `_chat_payload` builds an explicit whitelist and silently dropped it —
    the exact failure its own neighbouring comment warns about ("a proxy that
    edits the request decides for it")."""
    from codai.backends.vllm import VllmBackend
    from codai.config import VllmConfig

    be = VllmBackend.__new__(VllmBackend)
    be._cfg = VllmConfig()
    be._model_id = "m"
    be._served_name = "m"

    off = be._chat_payload([], None, 0.0, 1.0, None, False, logprobs=True)
    assert "return_tokens_as_token_ids" not in off

    be._cfg.return_token_ids = True
    on = be._chat_payload([], None, 0.0, 1.0, None, False, logprobs=True)
    assert on["return_tokens_as_token_ids"] is True


def test_the_field_is_only_sent_with_logprobs():
    """It describes how logprob tokens are rendered, so without logprobs it is
    noise on the wire."""
    from codai.backends.vllm import VllmBackend
    from codai.config import VllmConfig

    be = VllmBackend.__new__(VllmBackend)
    be._cfg = VllmConfig()
    be._cfg.return_token_ids = True
    be._model_id = be._served_name = "m"
    assert "return_tokens_as_token_ids" not in be._chat_payload(
        [], None, 0.0, 1.0, None, False)


def test_it_is_off_by_default():
    """It changes what the `token` field looks like, and only one consumer reads
    it."""
    from codai.config import VllmConfig
    assert VllmConfig().return_token_ids is False


def test_it_can_be_set_per_model():
    """So the one checkpoint with a byte-level vocabulary can ask for ids without
    changing what every other vLLM model reports."""
    from codai.backends.overrides import VLLM_FIELDS
    assert "return_token_ids" in VLLM_FIELDS


def test_a_pod_can_be_told_to_return_token_ids():
    """The pod's vLLM backend is configured by what this install forwards, so the
    field has to be in the forwarded set or a pod could never ask for ids."""
    from codai.api.runpod_worker import _ENGINE_FORWARDED_FIELDS
    assert "return_token_ids" in _ENGINE_FORWARDED_FIELDS["vllm"]


def test_forwarding_it_cannot_affect_an_install_that_did_not_ask():
    """engine_config_for_pod skips falsey values, so adding the field to the
    forwarded set is inert until someone sets it True."""
    import inspect
    from codai.api import runpod_worker
    src = inspect.getsource(runpod_worker.engine_config_for_pod)
    assert "if v in (None, \"\", 0, False):" in src
