"""The card, the image and the money are the MODEL's configuration.

`/v1/runpod/scale` states the rule for itself: "there is deliberately no way
for a request to change a model's configuration". That is what makes a renting
route safe to expose -- a caller can start work without being able to pick a
bigger card, a different image, or a higher price than the operator set on the
Models page.

A request may NARROW what is configured: fewer pods, a lower ceiling. Asking
for less than you are allowed is always safe, and the narrowing dies with the
request. Asking for MORE is clamped and said out loud, because it is a client
that does not know the limit rather than an attack -- but it cannot take
effect.
"""

import pytest

from codai.api import loras
from codai.api import text_lora_fleet as flt


BLOCK = {"enabled": True, "min_vram_gb": 48, "cloud_type": "SECURE",
         "image": "ghcr.io/nextime/coderai-trainer-text:v0.27.1",
         "max_pods": 4, "rounds": 6, "max_pod_hourly_usd": 0.90,
         "max_fleet_hourly_usd": 3.60, "container_disk_gb": 150}


# ------------------------------------------------------------ the parsing ----

def test_a_models_block_becomes_the_configuration():
    c = flt.parse_model_lora_train(BLOCK)
    assert c.enabled and c.min_vram_gb == 48 and c.max_pods == 4
    assert c.max_pod_hourly_usd == 0.90 and c.container_disk_gb == 150
    assert c.image.endswith(":v0.27.1")


def test_a_model_without_the_block_is_not_trainable_on_pods():
    """Silence means off: a model nobody configured for rented training must
    not become trainable on rented GPUs by default."""
    assert flt.parse_model_lora_train(None).enabled is False
    assert flt.parse_model_lora_train({}).enabled is False


def test_an_unknown_key_is_ignored_not_fatal():
    c = flt.parse_model_lora_train({**BLOCK, "from_the_future": True})
    assert c.max_pods == 4


def test_a_malformed_value_keeps_the_default_rather_than_crashing():
    """An operator typo in one field must not make the model untrainable."""
    c = flt.parse_model_lora_train({**BLOCK, "max_pods": "four"})
    assert c.max_pods == 1 and c.min_vram_gb == 48


# ----------------------------------------------------------- the narrowing ---

def test_a_request_may_ask_for_fewer_pods():
    c = flt.narrow(flt.parse_model_lora_train(BLOCK), pods=2)
    assert c.max_pods == 2


def test_a_request_cannot_ask_for_more_pods(capsys):
    c = flt.narrow(flt.parse_model_lora_train(BLOCK), pods=99)
    assert c.max_pods == 4, "clamped to what the operator permits"
    assert "the ceiling is 4" in capsys.readouterr().out


def test_a_request_may_lower_the_ceiling_as_well_as_the_target():
    """Both halves of what a scale request can do: ask for fewer, and cap how
    many it would accept."""
    cfg = flt.parse_model_lora_train(BLOCK)        # operator permits 4
    assert flt.narrow(cfg, max_pods=2).max_pods == 2
    assert flt.narrow(cfg, pods=8, max_pods=2).max_pods == 2


def test_a_request_cannot_raise_the_ceiling(capsys):
    c = flt.narrow(flt.parse_model_lora_train(BLOCK), max_pods=64)
    assert c.max_pods == 4
    assert "permits 4" in capsys.readouterr().out


def test_a_request_may_lower_the_price_ceiling():
    c = flt.narrow(flt.parse_model_lora_train(BLOCK), max_pod_hourly_usd=0.40)
    assert c.max_pod_hourly_usd == 0.40


def test_a_request_cannot_raise_the_price_ceiling(capsys):
    """The one that matters: a client must not be able to rent an H100 by
    asking nicely."""
    c = flt.narrow(flt.parse_model_lora_train(BLOCK), max_pod_hourly_usd=9.99)
    assert c.max_pod_hourly_usd == 0.90
    assert "the model allows $0.90" in capsys.readouterr().out


def test_a_ceiling_can_be_introduced_where_the_model_set_none():
    """No configured ceiling means the account caps are the only limit, so a
    request adding one is narrowing, not widening."""
    c = flt.narrow(flt.parse_model_lora_train({**BLOCK, "max_pod_hourly_usd": 0}),
                   max_pod_hourly_usd=0.50)
    assert c.max_pod_hourly_usd == 0.50


def test_rounds_are_duration_not_fleet_size():
    """More rounds costs time, not a bigger fleet or a dearer card, so a
    request may raise them."""
    c = flt.narrow(flt.parse_model_lora_train(BLOCK), rounds=12)
    assert c.rounds == 12 and c.max_pods == 4


def test_narrowing_does_not_mutate_the_models_configuration():
    base = flt.parse_model_lora_train(BLOCK)
    flt.narrow(base, pods=1, max_pod_hourly_usd=0.10)
    assert base.max_pods == 4 and base.max_pod_hourly_usd == 0.90


def test_nothing_in_the_request_can_name_a_card_an_image_or_a_url():
    """The structural guarantee: those fields do not exist on the request, so
    no amount of client creativity reaches them."""
    fields = set(loras.TextLoraTrainRequest.model_fields)
    for forbidden in ("min_vram_gb", "gpu_types", "cloud_type", "image",
                      "callback_url", "network_volume_id", "container_disk_gb",
                      "pod_max_lifetime_s"):
        assert forbidden not in fields, f"{forbidden} is settable by a request"


def test_the_request_can_still_narrow_the_three_that_are_safe():
    fields = set(loras.TextLoraTrainRequest.model_fields)
    for allowed in ("pods", "rounds", "max_pod_hourly_usd",
                    "max_fleet_hourly_usd", "distributed"):
        assert allowed in fields


def test_asking_for_distributed_on_an_unconfigured_model_says_where_to_set_it():
    """The error has to name the fix, or an operator goes looking in the
    request schema for a field that deliberately is not there."""
    cfg = flt.config_for_model("a-model-with-no-block")
    assert cfg.enabled is False


# ------------------------------------------------- the Models page edits it --
#
# "It should be a configuration in the model page like all the others" -- so it
# has to survive the round trip through that page, and the admin save path is a
# WHITELIST: a key not on it is dropped silently, which is the precise shape of
# a knob that appears to save and does not.

def test_the_admin_save_path_accepts_the_block():
    """It is a whitelist. Without the key, every field on the page would look
    like it saved, reload as empty, and no error would appear anywhere."""
    import pathlib
    src = (pathlib.Path(loras.__file__).parent.parent / "admin" / "routes.py").read_text()
    assert '"lora_train",' in src


def _models_html():
    import pathlib
    return (pathlib.Path(loras.__file__).parent.parent / "admin" / "templates"
            / "models.html").read_text()


def _lt_fields():
    """The entries of _LT_FIELDS only -- the page has other arrays of its own,
    and a regex over the whole template matches them too."""
    import re
    html = _models_html()
    block = html[html.index("const _LT_FIELDS = ["):]
    block = block[:block.index("];")]
    return re.findall(r"\['([a-z-]+)',\s*'([a-z_]+)',", block)


def test_the_models_page_can_read_every_field_it_can_write():
    """A knob the form can SET and cannot READ is indistinguishable from one
    that did not save. Both directions come from one field list."""
    import re
    html = _models_html()
    assert "_populateLoraTrain" in html and "_collectLoraTrain" in html
    assert "lora_train: _collectLoraTrain()" in html, "never written on save"
    ids = set(re.findall(r'id="cfg-lt-([a-z-]+)"', html))
    listed = {sfx for sfx, _key in _lt_fields()}
    assert listed, "the field list is empty"
    assert listed <= ids, f"no input on the page for {sorted(listed - ids)}"


def test_every_configurable_field_is_a_real_config_field():
    """A page offering a knob the parser ignores is worse than no knob."""
    import dataclasses
    keys = {key for _sfx, key in _lt_fields()}
    known = {f.name for f in dataclasses.fields(flt.LoraTrainConfig)}
    assert keys and keys <= known, f"not in LoraTrainConfig: {sorted(keys - known)}"


def test_the_money_fields_are_on_the_page():
    """The whole point of moving this out of the request: an operator can see
    and set the ceiling."""
    keys = {key for _sfx, key in _lt_fields()}
    assert {"max_pod_hourly_usd", "max_fleet_hourly_usd", "min_vram_gb",
            "image", "max_pods"} <= keys


def test_an_off_block_is_absent_rather_than_full_of_defaults():
    """An absent block is how a model says 'not trainable on rented GPUs'.
    Writing one full of zeros would make every model look configured."""
    assert "if (!enabled) return null;" in _models_html()


# ------------------------------------------------- choosing a card by VRAM ---

class _Catalogue:
    def list_gpu_types(self):
        return [{"id": "NVIDIA A40", "display_name": "NVIDIA A40", "memory_gb": 48, "secure_price": 0.44,
                 "community_price": 0.35},
                {"id": "NVIDIA L40S", "display_name": "NVIDIA L40S", "memory_gb": 48, "secure_price": 0.86},
                {"id": "NVIDIA RTX 6000 Ada", "display_name": "NVIDIA RTX 6000 Ada", "memory_gb": 48, "secure_price": 0.77},
                {"id": "NVIDIA H100 PCIe", "display_name": "NVIDIA H100 PCIe", "memory_gb": 80, "secure_price": 2.69},
                {"id": "NVIDIA RTX 4090", "display_name": "NVIDIA RTX 4090", "memory_gb": 24, "secure_price": 0.34},
                {"id": "NVIDIA A100 40GB", "display_name": "NVIDIA A100 40GB", "memory_gb": 40, "secure_price": 0.80},
                {"id": "NVIDIA UNPRICED", "display_name": "NVIDIA UNPRICED", "memory_gb": 48}]


def _cards(**kw):
    cfg = flt.LoraTrainConfig(**{"min_vram_gb": 48, **kw})
    return flt.eligible_cards(_Catalogue(), cfg,
                              ceiling=kw.pop("ceiling", 0.90)
                              if "ceiling" in kw else 0.90)


def test_a_vram_floor_excludes_smaller_cards():
    ids = [c["id"] for c in _cards()]
    assert "NVIDIA RTX 4090" not in ids, "24 GB is under the floor"
    assert "NVIDIA A100 40GB" not in ids, "40 GB is under a 48 GB floor"


def test_the_price_ceiling_excludes_the_expensive_ones():
    ids = [c["id"] for c in _cards()]
    assert "NVIDIA H100 PCIe" not in ids, "$2.69 is over a $0.90 ceiling"


def test_the_cheapest_eligible_card_is_first():
    """Ranked, not picked: capacity moves, and the next-cheapest card is a
    better answer than a failed job."""
    cards = _cards()
    assert [c["id"] for c in cards] == ["NVIDIA A40", "NVIDIA RTX 6000 Ada",
                                        "NVIDIA L40S"]
    assert cards[0]["hourly_usd"] == 0.44


def test_an_unpriced_card_is_excluded_when_a_ceiling_is_set():
    """Renting a card the catalogue cannot price, under a price ceiling, is
    renting blind."""
    assert "NVIDIA UNPRICED" not in [c["id"] for c in _cards()]


def test_an_allow_list_narrows_but_cannot_widen():
    cfg = flt.LoraTrainConfig(min_vram_gb=48,
                              gpu_types="NVIDIA L40S, NVIDIA RTX 4090, NVIDIA H100 PCIe")
    ids = [c["id"] for c in flt.eligible_cards(_Catalogue(), cfg, ceiling=0.90)]
    assert ids == ["NVIDIA L40S"], "the 4090 is under the floor, the H100 over the ceiling"


def test_community_pricing_is_used_when_asked_for():
    cfg = flt.LoraTrainConfig(min_vram_gb=48, cloud_type="COMMUNITY")
    cards = flt.eligible_cards(_Catalogue(), cfg, ceiling=0.90)
    assert cards[0]["id"] == "NVIDIA A40" and cards[0]["hourly_usd"] == 0.35


def test_no_eligible_card_is_an_empty_list_not_a_guess():
    cfg = flt.LoraTrainConfig(min_vram_gb=200)
    assert flt.eligible_cards(_Catalogue(), cfg, ceiling=0.90) == []


# ------------------------------- the three defects a real run turned up ------
#
# A test_run was issued naming `Qwen/Qwen3.8-27B` -- the weights -- where the
# block sits on the entry aliased `qwen38`. The lookup found nothing, the block
# defaulted to DISABLED, and the job fell through to training LOCALLY on a
# GPU-less orchestrator: tens of gigabytes of download, then a CUDA error, and
# nothing in the log because the failure was only recorded on the job record.
# Three separate faults, each tested here.

class _CM:
    def __init__(self, entries):
        self.models_data = {"text_models": entries}


@pytest.fixture
def catalogue(monkeypatch):
    def _install(entries):
        import codai.admin.routes as ar
        import codai.models.manager as mgr
        cm = _CM(entries)
        monkeypatch.setattr(ar, "config_manager", cm, raising=False)
        monkeypatch.setattr(mgr, "config_manager", cm, raising=False)
        return cm
    return _install


def test_the_block_is_found_by_the_catalogue_alias(catalogue):
    catalogue([{"alias": "qwen38", "path": "", "lora_train": dict(BLOCK)}])
    assert flt.config_for_model("qwen38").enabled is True


def test_the_block_is_also_found_by_the_training_base(catalogue):
    """The defect: 'qwen38' and 'Qwen/Qwen3.8-27B' are one model to a human and
    two keys to code, and a request may reasonably say either."""
    block = {**BLOCK, "base_model": "Qwen/Qwen3.8-27B"}
    catalogue([{"alias": "qwen38", "path": "", "lora_train": block}])
    cfg = flt.config_for_model("Qwen/Qwen3.8-27B")
    assert cfg.enabled is True and cfg.max_pods == 4


def test_an_ambiguous_training_base_is_refused_not_guessed(catalogue):
    """Two models training the same base have two ceilings. Picking one
    silently is how a run uses the wrong one."""
    block = {**BLOCK, "base_model": "Qwen/Qwen3.8-27B"}
    catalogue([{"alias": "qwen38", "lora_train": block},
               {"alias": "qwen38-alt", "lora_train": {**block, "max_pods": 64}}])
    with pytest.raises(ValueError) as e:
        flt.config_for_model("Qwen/Qwen3.8-27B")
    assert "more than one model" in str(e.value)
    assert "qwen38" in str(e.value) and "qwen38-alt" in str(e.value)


def test_an_unknown_model_still_returns_a_disabled_block(catalogue):
    catalogue([{"alias": "something-else"}])
    assert flt.config_for_model("nobody").enabled is False


def test_asking_for_pods_without_a_block_refuses_rather_than_training_here():
    """The fall-through that cost tens of gigabytes: test_run and a bare `pods`
    did not take the explicit-distributed branch, so they trained locally."""
    import inspect
    src = inspect.getsource(loras)
    i = src.index("_asked_for_pods = any(")
    block = src[i:i + 500]
    for field in ("req.distributed", "req.test_run", "req.pods", "req.min_pods",
                  "req.max_pods"):
        assert field in block, f"{field} can still fall through"
    assert "has no `lora_train` block with " in src


def test_the_refusal_says_which_name_to_use():
    """The error has to name the fix, or the next person passes the weights
    again -- it is the natural thing to write."""
    import inspect
    src = inspect.getsource(loras)
    assert "an alias like " in src and "'qwen38' is right" in src


def test_local_training_refuses_before_downloading_on_a_gpuless_box():
    """Checked before the base is fetched: otherwise the failure arrives after
    the download as a CUDA error that names neither cause nor fix."""
    import inspect
    src = inspect.getsource(loras._text_train_blocking)
    # The CONDITION, not just the message: blanking the test leaves the string
    # in place, and an earlier version of this test passed on exactly that.
    assert "if not _has_gpu:" in src, "the guard is not a live condition"
    guard = src.index("if not _has_gpu:")
    fetch = src.index("tlt.run_training(")
    assert guard < fetch, "the GPU check must precede the load"
    assert "cuda.is_available()" in src


def test_a_failure_is_printed_and_not_only_recorded():
    """A job that died before renting wrote nothing to the log, so the only
    trace was a field a client had to ask for."""
    import inspect
    src = inspect.getsource(loras._detached_text_train)
    assert "FAILED" in src and "print(" in src


def test_the_request_string_never_reaches_from_pretrained_when_a_block_names_a_base():
    """The defect digesta found, and the one my own docstring had promised was
    fixed: the lookup worked, the SUBSTITUTION did not.

    `base_model: "qwen38"` found the entry and then handed "qwen38" straight to
    from_pretrained, which failed with "not a local folder and is not a valid
    model identifier" -- identically under dry_run, which is what proved it was
    resolution rather than training. So the order matters: the block has to be
    resolved before the tokenizer, not at the pod launch.
    """
    import inspect
    src = inspect.getsource(loras._text_train_blocking)
    i_cfg = src.index("config_for_model")
    i_base = src.index("base = _resolve_text_base")
    i_tok = src.index("AutoTokenizer.from_pretrained")
    assert i_cfg < i_base < i_tok, "the block must be read before the base loads"
    assert "_train_cfg.base_model or req.base_model" in src, (
        "the base must come from the block, falling back to the request")


def test_the_refusal_happens_before_anything_is_downloaded():
    """Asking for pods on a model with no block used to get as far as pulling
    the weights. The refusal belongs before the tokenizer, not after it."""
    import inspect
    src = inspect.getsource(loras._text_train_blocking)
    assert src.index("_asked_for_pods") < src.index("AutoTokenizer.from_pretrained")


# ------------- the catalogue a process holds is not the configuration --------
#
# digesta's second report: the error changed from "not a valid model
# identifier" to "has no lora_train block", while the block was right there in
# models.json -- read inside the production container, by a process started
# AFTER the file was written. Their hypothesis was exact: the lookup reads
# `codai.admin.routes.config_manager.models_data`, and an engine process where
# the admin module was never initialised holds nothing. The block was invisible
# while being correct, and my message asserted the opposite of the truth.

@pytest.fixture
def on_disk(monkeypatch, tmp_path):
    """A models.json on disk, and NO in-memory catalogue -- an engine process."""
    import json as _json
    entry = {"alias": "qwen38", "id": "qwen38-awq", "path": "",
             "lora_train": {"enabled": True, "base_model": "Qwen/Qwen3.8-27B",
                            "max_pods": 32, "min_vram_gb": 48,
                            "max_pod_hourly_usd": 0.9}}
    (tmp_path / "models.json").write_text(_json.dumps({"text_models": [entry]}))
    monkeypatch.setenv("CODERAI_CONFIG_DIR", str(tmp_path))
    import codai.admin.routes as ar
    monkeypatch.setattr(ar, "config_manager", None, raising=False)
    return tmp_path


@pytest.mark.parametrize("name", ["qwen38", "qwen38-awq", "Qwen/Qwen3.8-27B"])
def test_the_block_is_found_with_no_in_memory_catalogue(on_disk, name):
    """All three names, from the file, in a process that holds no catalogue."""
    cfg = flt.config_for_model(name)
    assert cfg.enabled is True, f"{name} did not resolve from disk"
    assert cfg.base_model == "Qwen/Qwen3.8-27B" and cfg.max_pods == 32


def test_memory_is_preferred_when_a_process_has_it(on_disk, monkeypatch):
    """The file is the fallback, not the source of record: an unsaved edit in
    the admin UI must still win for the process holding it."""
    import codai.admin.routes as ar

    class CM:
        models_data = {"text_models": [{"alias": "qwen38",
                                        "lora_train": {"enabled": True,
                                                       "max_pods": 7}}]}

    monkeypatch.setattr(ar, "config_manager", CM, raising=False)
    assert flt.config_for_model("qwen38").max_pods == 7


def test_the_error_names_where_it_looked(on_disk):
    """"There is no block" was true of the lookup and false of the disk, and it
    sent someone to check a file that was already right."""
    src = flt.catalogue_source()
    assert "models.json" in src and "entries" in src


def test_a_missing_models_json_says_so_rather_than_implying_a_missing_block(
        monkeypatch, tmp_path):
    monkeypatch.setenv("CODERAI_CONFIG_DIR", str(tmp_path / "nowhere"))
    import codai.admin.routes as ar
    monkeypatch.setattr(ar, "config_manager", None, raising=False)
    assert "DOES NOT EXIST" in flt.catalogue_source()


def test_a_dry_run_is_never_blocked_by_a_missing_block():
    """A dry run rents nothing. Refusing it withheld the one answer that is
    free -- and that free answer is what found all of these."""
    import inspect
    src = inspect.getsource(loras._text_train_blocking)
    i = src.index("_asked_for_pods and not _train_cfg.enabled")
    block = src[i:i + 1200]
    assert "if req.dry_run:" in block
    assert "Reported, not fatal" in block


def test_an_alias_resolves_to_its_path_with_no_in_memory_catalogue(on_disk):
    """The same fault, one layer down, caught by a probe rather than by them.

    Once the training entry's own `path` is the base, resolution depends
    entirely on finding that entry -- and the resolver read only the catalogue
    a process holds. In an engine with none, the alias went to from_pretrained
    verbatim: exactly the failure that had just been fixed one level up.
    """
    import json as _json
    entry = {"alias": "qwen38-train", "id": "qwen38-train",
             "path": "Qwen/Qwen3.8-27B", "enabled": False,
             "lora_train": {"enabled": True, "max_pods": 32}}
    (on_disk / "models.json").write_text(_json.dumps({"text_models": [entry]}))
    assert loras._resolve_text_base("qwen38-train") == "Qwen/Qwen3.8-27B"
    # and an HF id that is nobody's alias passes through untouched
    assert loras._resolve_text_base("Qwen/Qwen3.8-27B") == "Qwen/Qwen3.8-27B"


# ---- a populated catalogue is not a complete one ---------------------------
#
# The third report in a row, and the one my own tests could not have caught:
# they covered an EMPTY in-memory catalogue, and the real failure was a
# populated one. The loader excludes parked (enabled: false) entries, and a
# training-only entry is naturally parked -- nothing should serve it. So memory
# answered "no such entry", `if md: return md` treated that as authoritative,
# and the file holding the right answer was never opened.
#
# Found because catalogue_source() printed "read from the in-memory catalogue".
# A message that names where it looked is worth the line it costs.

@pytest.fixture
def parked(monkeypatch, tmp_path):
    """Production's exact shape: a parked training entry on disk, and an
    in-memory catalogue that has filtered it out."""
    import json as _json
    (tmp_path / "models.json").write_text(_json.dumps({"text_models": [
        {"alias": "qwen38", "id": "qwen38-awq", "enabled": True},
        {"alias": "qwen38-train", "id": "qwen38-train",
         "path": "Qwen/Qwen3.8-27B", "enabled": False,
         "lora_train": {"enabled": True, "base_model": "Qwen/Qwen3.8-27B",
                        "max_pods": 32, "min_vram_gb": 48}},
    ]}))
    monkeypatch.setenv("CODERAI_CONFIG_DIR", str(tmp_path))

    class CM:          # the loader's view: the parked entry is simply absent
        models_data = {"text_models": [{"alias": "qwen38", "id": "qwen38-awq",
                                        "enabled": True}]}

    import codai.admin.routes as ar
    monkeypatch.setattr(ar, "config_manager", CM, raising=False)
    return tmp_path


def test_a_parked_entrys_block_is_found_though_memory_filtered_it_out(parked):
    cfg = flt.config_for_model("qwen38-train")
    assert cfg.enabled is True, "a populated catalogue masked the file"
    assert cfg.base_model == "Qwen/Qwen3.8-27B" and cfg.max_pods == 32


def test_a_parked_entry_still_resolves_to_its_path(parked):
    """The resolver has the same two sources and needed the same fix."""
    assert loras._resolve_text_base("qwen38-train") == "Qwen/Qwen3.8-27B"


def test_the_training_entry_need_not_be_servable_to_be_configured(parked):
    """Why this is a code fix and not `enabled: true` on the entry: flipping
    that switch would offer a 55GB base to the serving path to keep a lookup
    happy. A config-only entry stays unservable."""
    import json as _json
    entry = _json.loads((parked / "models.json").read_text())["text_models"][1]
    assert entry["enabled"] is False
    assert flt.config_for_model("qwen38-train").enabled is True


def test_the_source_line_names_both_places(parked):
    """It said "the in-memory catalogue" while the file held the answer: true
    of what the lookup did, useless for working out why."""
    src = flt.catalogue_source()
    assert "in-memory" in src and "models.json" in src
    assert "parked ones excluded" in src


def test_a_block_is_found_by_training_base_from_the_file_too(parked):
    """The by-base scan reads both sources, so naming the weights still works."""
    cfg = flt.config_for_model("Qwen/Qwen3.8-27B")
    assert cfg.enabled is True and cfg.max_pods == 32
