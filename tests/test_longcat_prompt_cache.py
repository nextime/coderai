"""Caching prompt embeddings on disk, so the text encoder need not run — or load.

The UMT5-XXL encoder is ~11 GB and runs ONCE per request to produce a few MB of
embeddings that are a pure function of the prompt and the encoder. v0.2.76 made that
once per request instead of once per segment. This makes it once per prompt, ever.

The payoff is not the saved GPU pass, which is seconds. It is that a cached prompt
needs no encoder at all, so with the cache on the encoder is built LAZILY: ~11 GB of
host RAM and about a minute of load that a repeated prompt never spends. The whole
feature is therefore only worth having if the key is exact — an entry served to the
wrong pipeline is a silent quality change, which is worse than a miss.
"""
import importlib.util
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMON = ROOT / "tools" / "longcat_common.py"
SERVICE = ROOT / "tools" / "longcat_service.py"
WORKER = ROOT / "codai" / "api" / "longcat_worker.py"
ROUTES = ROOT / "codai" / "admin" / "routes.py"
TEMPLATE = ROOT / "codai" / "admin" / "templates" / "models.html"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def LC():
    return _load("lc_pc", COMMON)


@pytest.fixture
def SVC():
    return _load("lsvc_pc", SERVICE)


# ------------------------------------------------------------------ the contract

def test_it_is_off_unless_configured(LC):
    """Embeddings are derived data, but they are also MBs under the user's cache dir."""
    assert LC.PROMPT_CACHE_MODES[0] == "off"
    assert LC.prompt_cache_problems("off") == []
    assert LC.prompt_cache_problems("") == []


def test_an_unknown_mode_is_refused_by_name(LC):
    problems = LC.prompt_cache_problems("ram")
    assert problems and "ram" in problems[0] and "prompt_cache" in problems[0]


@pytest.mark.parametrize("bad", ["0", "-1", "nonsense"])
def test_a_cap_that_cannot_hold_anything_is_refused(LC, bad):
    assert LC.prompt_cache_problems("disk", bad)


def test_a_blank_cap_means_the_default(LC):
    assert LC.prompt_cache_problems("disk", "") == []
    assert LC.prompt_cache_problems("disk", None) == []
    assert LC.DEFAULT_PROMPT_CACHE_MAX_GB > 0


def test_the_default_dir_is_under_the_coderai_home(LC):
    assert LC.prompt_cache_dir().endswith("prompt_cache/longcat")
    assert LC.prompt_cache_dir("~/elsewhere").endswith("elsewhere")


# ------------------------------------------------------------------ the key

def _key(LC, **over):
    args = dict(checkpoint="/AI/hfcache/LongCat-Video", dtype="bfloat16",
                text_encoder_quant="none", args=(), kwargs={"prompt": "a cat"})
    args.update(over)
    return LC.prompt_cache_key(**args)


def test_the_same_call_keys_the_same_way(LC):
    assert _key(LC) == _key(LC)


def test_device_is_excluded_because_it_moves_the_answer_not_changes_it(LC):
    """Including it would split one entry into one per device, for identical numbers."""
    on_cuda = _key(LC, kwargs={"prompt": "a cat", "device": "cuda"})
    on_cpu = _key(LC, kwargs={"prompt": "a cat", "device": "cpu"})
    assert on_cuda == on_cpu == _key(LC)


def test_a_different_prompt_is_a_different_entry(LC):
    assert _key(LC) != _key(LC, kwargs={"prompt": "a dog"})


def test_the_negative_prompt_counts_too(LC):
    assert _key(LC) != _key(LC, kwargs={"prompt": "a cat", "negative_prompt": "blurry"})


def test_quantisation_cannot_be_served_from_a_bf16_entry(LC):
    """An NF4 encoder does not produce the bf16 encoder's numbers. Reusing one for the
    other is a silent quality change — the thing a cache must never do."""
    assert _key(LC) != _key(LC, text_encoder_quant="nf4")
    assert _key(LC, text_encoder_quant="int8") != _key(LC, text_encoder_quant="nf4")


def test_the_dtype_counts(LC):
    assert _key(LC) != _key(LC, dtype="float16")


def test_the_sequence_length_counts(LC):
    assert _key(LC) != _key(LC, kwargs={"prompt": "a cat", "max_sequence_length": 256})


def test_the_same_checkpoint_through_a_symlink_is_the_same_encoder(LC):
    """hfcache is reached by different paths on different machines and in the pod."""
    assert _key(LC, checkpoint="/other/mount/LongCat-Video") == _key(LC)


def test_a_different_checkpoint_is_not(LC):
    assert _key(LC, checkpoint="/AI/hfcache/LongCat-Video-Avatar-1.5") != _key(LC)


def test_the_format_version_invalidates_an_old_cache(LC):
    """A stored tuple whose shape changed must be ignored, not misread."""
    assert isinstance(LC.PROMPT_CACHE_FORMAT, int)
    assert "format" in __import__("inspect").getsource(LC.prompt_cache_key)


def test_an_unhashable_argument_still_keys(LC):
    """The in-memory cache gives up on these; the disk key must not, since a prompt
    list is both unhashable and the normal case for a batch."""
    assert _key(LC, kwargs={"prompt": ["a cat", "a dog"]}) \
        != _key(LC, kwargs={"prompt": ["a dog", "a cat"]})


# ------------------------------------------------------------------ the cache itself

torch = pytest.importorskip("torch", reason="the cache stores tensors")


@pytest.fixture
def cache(SVC, tmp_path):
    return SVC._PromptCache(str(tmp_path / "pc"), 1.0)


def _embeds():
    """The 4-tuple upstream's encode_prompt returns: embeds, mask, and the negatives."""
    return (torch.randn(1, 8, 16, dtype=torch.bfloat16),
            torch.ones(1, 8, dtype=torch.long), None, None)


def test_a_cold_cache_is_a_miss_not_an_error(cache):
    assert cache.get("nothing-here") is None
    assert cache.stats["misses"] == 1


def test_what_goes_in_comes_back_out(cache):
    value = _embeds()
    cache.put("k", value)
    got = cache.get("k", device="cpu", dtype=torch.bfloat16)
    assert got is not None
    assert torch.equal(got[0], value[0])


def test_the_attention_mask_is_not_cast_to_the_compute_dtype(cache):
    """It is an integer mask. Casting it to bf16 with the embeddings would corrupt it."""
    cache.put("k", _embeds())
    got = cache.get("k", device="cpu", dtype=torch.bfloat16)
    assert got[1].dtype == torch.long


def test_absent_negatives_stay_absent(cache):
    """do_classifier_free_guidance=False returns None for both; None is not a tensor."""
    cache.put("k", _embeds())
    got = cache.get("k", device="cpu", dtype=torch.bfloat16)
    assert got[2] is None and got[3] is None


def test_a_hit_is_counted_and_a_miss_is_not(cache):
    cache.put("k", _embeds())
    cache.get("k", device="cpu")
    assert cache.stats["hits"] == 1 and cache.stats["misses"] == 0


def test_an_unreadable_entry_is_a_miss_and_deletes_itself(cache, tmp_path):
    """Truncated by a crash mid-write, or written by an incompatible torch. A cache
    that cannot be read must not be able to break generation."""
    cache.put("k", _embeds())
    with open(cache._path("k"), "wb") as fh:
        fh.write(b"not a tensor")
    assert cache.get("k", device="cpu") is None
    assert not pathlib.Path(cache._path("k")).exists()
    assert cache.stats["errors"] == 1


def test_a_half_written_entry_is_never_visible(cache):
    """Written to a .tmp and renamed, so a reader sees all of it or none of it."""
    src = __import__("inspect").getsource(type(cache).put)
    assert "os.replace" in src and ".tmp" in src


def test_it_will_not_accept_a_pickle_from_its_own_directory(cache):
    """A cache dir is not a trusted source: weights_only, so an entry cannot execute."""
    src = __import__("inspect").getsource(type(cache).get)
    assert "weights_only=True" in src


def test_it_stays_under_its_cap_by_dropping_the_oldest(SVC, tmp_path):
    """A prompt list is the case that runs away, so the cap is on bytes, not count."""
    tiny = SVC._PromptCache(str(tmp_path / "tiny"), 0.000001)   # ~1 KB
    tiny.put("k", _embeds())
    assert tiny.stats["evicted"] >= 1
    assert tiny.get("k") is None


def test_a_hit_counts_as_a_use_for_eviction(cache):
    """LRU by mtime: a prompt used every day must not be evicted before a stale one."""
    src = __import__("inspect").getsource(type(cache).get)
    assert "os.utime" in src


def test_it_reports_what_is_actually_on_disk(cache):
    cache.put("k", _embeds())
    report = cache.report()
    assert report["entries"] == 1 and report["writes"] == 1
    assert report["size_mb"] >= 0 and report["dir"].endswith("pc")


def test_an_unwritable_cache_warns_once_and_keeps_generating(SVC, tmp_path):
    """A cache we cannot write is not a reason to fail a request."""
    c = SVC._PromptCache(str(tmp_path / "ro"), 1.0)
    c.dir = "/proc/nonexistent-cannot-write"      # a real write failure
    c.put("k", _embeds())
    c.put("k2", _embeds())
    assert c.stats["writes"] == 0 and c._warned is True


# ------------------------------------------------------------------ laziness

def test_the_encoder_is_not_built_until_something_misses(SVC):
    built = []

    class _Pipe:
        text_encoder = None

    lazy = SVC._LazyEncoder(lambda: built.append(1) or "encoder")
    assert built == [], "constructing the holder must not build 11 GB of weights"
    lazy.realise(_Pipe())
    assert built == [1]


def test_realising_twice_builds_once(SVC):
    built = []

    class _Pipe:
        text_encoder = None

    lazy = SVC._LazyEncoder(lambda: built.append(1) or "encoder")
    pipe = _Pipe()
    lazy.realise(pipe)
    lazy.realise(pipe)
    assert built == [1] and pipe.text_encoder == "encoder"


def test_it_is_not_a_transparent_proxy(SVC):
    """Upstream guards every use with `if self.text_encoder is not None`. A proxy that
    looked like a module would be moved to the card by pipe.to() and offloaded by
    _place_pipeline — building the very weights this exists to avoid. So the pipeline's
    text_encoder stays None and the holder sits beside it."""
    src = SERVICE.read_text(encoding="utf-8")
    assert "_state[\"lazy_encoder\"] = _LazyEncoder(" in src
    assert re.search(r"lazy_encoder.*\n.*text_encoder = None", src) or \
        "            text_encoder = None" in src


def test_laziness_follows_the_cache_and_nothing_else(SVC):
    """Off by default means the encoder loads eagerly, exactly as before."""
    src = SERVICE.read_text(encoding="utf-8")
    assert 'if _state.get("prompt_cache") is not None:' in src


def test_the_pipeline_is_dropped_with_its_deferred_encoder(SVC):
    """The holder closes over the load arguments; the pipeline it would attach to is
    gone. The DISK cache survives, which is the point of it being on disk."""
    src = SERVICE.read_text(encoding="utf-8")
    unload = src[src.index("def unload("):]
    assert '_state["lazy_encoder"] = None' in unload[:unload.index("\ndef ")]


def test_a_lazily_realised_quantised_encoder_is_never_swapped(SVC):
    """bitsandbytes quantises as weights land on CUDA and is not built to shuttle them
    back. A quantised encoder realised on a miss must not then be offloaded."""
    src = SERVICE.read_text(encoding="utf-8")
    wrap = src[src.index("def _wrap_encode_prompt("):]
    wrap = wrap[:wrap.index("\n@contextlib")]
    assert "_encoder_is_quantised(pipe)" in wrap


def test_the_disk_cache_is_consulted_before_the_encoder_is_touched(SVC):
    """Order matters: in-memory, then disk, then encode. Realising the encoder before
    the disk lookup would spend the 11 GB this feature exists to save."""
    wrap = SERVICE.read_text(encoding="utf-8")
    wrap = wrap[wrap.index("def _wrap_encode_prompt("):]
    assert wrap.index("cache.get(") < wrap.index("lazy.realise(")


# ------------------------------------------------------------------ orchestration

def test_the_service_validates_before_the_slow_load(SVC):
    """A bad value should not surface after a multi-minute load."""
    src = SERVICE.read_text(encoding="utf-8")
    assert "LC.prompt_cache_problems(" in src
    assert src.index("LC.prompt_cache_problems(") < src.index("load_pipeline(_state[\"model\"]")


@pytest.mark.parametrize("flag", ["--prompt-cache", "--prompt-cache-dir",
                                  "--prompt-cache-max-gb"])
def test_the_service_takes_the_flag(SVC, flag):
    assert flag in SERVICE.read_text(encoding="utf-8")


@pytest.mark.parametrize("flag", ["--prompt-cache", "--prompt-cache-dir",
                                  "--prompt-cache-max-gb"])
def test_the_worker_passes_it_through(flag):
    assert flag in WORKER.read_text(encoding="utf-8")


@pytest.mark.parametrize("key", ["prompt_cache", "prompt_cache_dir",
                                 "prompt_cache_max_gb"])
def test_the_key_is_documented_where_the_others_are(key):
    """The worker's header block is what anyone configuring this reads."""
    head = WORKER.read_text(encoding="utf-8")
    head = head[:head.index("import collections")]
    assert key in head


@pytest.mark.parametrize("key", ["prompt_cache", "prompt_cache_dir",
                                 "prompt_cache_max_gb"])
def test_the_key_survives_a_save_from_the_models_page(key):
    """api_model_configure rebuilds the entry from scratch, so a key missing from its
    whitelist is silently dropped on the next save — which is how H3's documented keys
    came to never survive."""
    src = ROUTES.read_text(encoding="utf-8")
    block = src[src.index("\"text_encoder_quant\", \"bsa\""):]
    assert f'"{key}"' in block[:block.index("):")]


@pytest.mark.parametrize("element", ["cfg-longcat-prompt-cache",
                                     "cfg-longcat-prompt-cache-dir",
                                     "cfg-longcat-prompt-cache-max"])
def test_the_models_page_can_set_it(element):
    """Three places, or the control exists and does nothing: the markup, the loader
    that fills it from the entry, and the serialiser that reads it back."""
    html = TEMPLATE.read_text(encoding="utf-8")
    assert html.count(f'id="{element}"') == 1, "markup"
    assert f"set('{element}'" in html, "not populated from the saved entry"
    assert f"'{element}')" in html.split("if (backend === 'longcat')")[1], "not saved"


def test_the_service_reports_the_cache_and_whether_the_encoder_loaded(SVC):
    """Laziness is invisible otherwise: /health must say if the 11 GB was spent."""
    src = SERVICE.read_text(encoding="utf-8")
    assert "text_encoder_loaded" in src
    assert "/prompt-cache" in src
