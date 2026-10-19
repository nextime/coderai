"""A pod setting that is a BOOLEAN must be able to say "off".

The pod env builder filters out any value equal to 0:

    if _val not in (None, "", 0, 0.0):
        env[_var] = str(_val)

and in Python `False == 0`. So a flag sent through that loop can only ever be
turned ON: `early_loop_abort: false` was silently discarded and the pod ran
with the feature enabled. The symptom is the worst kind -- the setting appears
in the config, the operator believes it applied, and the behaviour is the
opposite. It also made the feature impossible to A/B, which is how it was
found.

Booleans therefore go through their own loop, which writes "1" or "0"
explicitly. Two flags already did; the new one did not.
"""
import inspect
import re

from codai.api import runpod_worker as rw


def _env_block() -> str:
    """The OCR half of `_plan`, which is where a pod's env is assembled."""
    src = inspect.getsource(rw._plan)
    i = src.index('env["CODERAI_OCR_ENABLED"] = "1"')
    return src[i:]


def test_the_int_filter_still_drops_zero():
    """Unchanged on purpose: for an INT, 0 means "use the engine's default",
    and sending it would override the default with a meaningless zero."""
    assert '_val not in (None, "", 0, 0.0)' in _env_block()


def test_false_is_zero_in_python():
    """The whole reason this file exists."""
    assert False == 0
    assert False in (None, "", 0, 0.0)


def test_the_boolean_loop_writes_one_or_zero():
    blk = _env_block()
    assert 'env[_var] = "1" if getattr(_o, _attr) else "0"' in blk


def test_no_boolean_ocr_setting_goes_through_the_int_loop():
    """A boolean in the int loop is a boolean that can only be turned ON.
    This is the guard against the next one."""
    blk = _env_block()
    int_loop = blk[:blk.index('for _var, _attr in (("CODERAI_OCR_SURYA_REPAIR_ACCENTS"')]
    from codai.config import OcrConfig
    bools = {k for k, v in getattr(OcrConfig, "__annotations__", {}).items()
             if v is bool or v == "bool"}
    assert bools, "no boolean OCR settings found — the check would be vacuous"
    for name in sorted(bools):
        assert f'"{name}"' not in int_loop, (
            f"{name} is a bool sent through the zero-dropping int loop, so "
            f"setting it false would be silently discarded")


def test_the_early_loop_abort_reaches_a_pod():
    blk = _env_block()
    assert "CODERAI_OCR_EARLY_LOOP_ABORT" in blk
    assert "CODERAI_OCR_EARLY_LOOP_MIN_CHARS" in blk
    assert "CODERAI_OCR_EARLY_LOOP_PERSISTENCE" in blk


def test_the_abort_switch_is_in_the_boolean_half():
    """Not in the int loop, whose filter would drop it when set to false."""
    blk = _env_block()
    bool_loop = blk[blk.index('for _var, _attr in (("CODERAI_OCR_SURYA_REPAIR_ACCENTS"'):]
    assert "CODERAI_OCR_EARLY_LOOP_ABORT" in bool_loop


def test_the_config_can_express_off():
    from codai.config import OcrConfig
    cfg = OcrConfig()
    assert cfg.early_loop_abort is True
    cfg.early_loop_abort = False
    assert cfg.early_loop_abort is False


# ---------------------------------------------------------------------------
# the READING side, which had the same bug in mirror image
# ---------------------------------------------------------------------------

def test_bool_of_the_string_zero_is_true():
    """The other half of this file's lesson. The writer dropped False because
    `False == 0`; the reader turned "0" into True because `bool("0")` is True.
    One flag, two opposite coercions, both wrong in the same direction: the
    feature could only ever be ON."""
    assert bool("0") is True


def test_the_abort_flag_is_parsed_as_a_flag_not_cast_as_a_type():
    src = open("codai/config.py").read()
    # not in the typed loop, where the cast would be bool("0") -> True
    assert '("CODERAI_OCR_EARLY_LOOP_ABORT", "early_loop_abort", bool)' not in src
    # in the _flag loop, which reads the string
    i = src.index('("CODERAI_OCR_SURYA_REPAIR_ACCENTS", "surya_repair_accents")')
    j = src.index("setattr(self.config.ocr, _attr, _flag(_env))")
    k = src.index('("CODERAI_OCR_EARLY_LOOP_ABORT", "early_loop_abort")')
    assert i < k < j, "the flag must sit in the _flag() loop"


def test_the_flag_parser_reads_zero_as_false():
    src = open("codai/config.py").read()
    blk = src[src.index("def _flag(name: str) -> bool:"):]
    blk = blk[:blk.index("\n\n")]
    assert '"1", "true", "yes", "on"' in blk
    # i.e. anything else, including "0" and "false", is False
    for val in ("0", "false", "no", "off", ""):
        assert val not in ("1", "true", "yes", "on")


def test_no_ocr_env_row_casts_with_bool():
    """A typed row whose cast is `bool` can never express false. Guard against
    the next one."""
    import re
    src = open("codai/config.py").read()
    rows = re.findall(r'\("CODERAI_[A-Z_]+",\s*"[a-z_0-9]+",\s*bool\)', src)
    assert rows == [], f"these rows cast with bool and cannot be turned off: {rows}"
