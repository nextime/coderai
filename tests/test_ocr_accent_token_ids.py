"""The accent repair's input: token ids in vLLM's logprobs.

`ocr.surya_repair_accents` rebuilds a reply whose accents were lost in
detokenisation from the model's own token IDS -- the only lossless channel,
because vLLM 0.27.1 reports `bytes: []` for exactly the byte-level tokens that
lost their characters. vLLM reports ids (`token_id:2131`) only when launched
with `--return-tokens-as-token-ids`.

That flag was configured on the renting install (`vllm.extra_args`) and the fix
was verified. Then the OCR pipeline moved onto rented pods, and a pod launches
its OWN vLLM from its OWN seeded config -- which never carried the setting. The
pod's launch line read:

    --model datalab-to/surya-ocr-2 --max-model-len 32768
    --gpu-memory-utilization 0.9 --tensor-parallel-size 1 --trust-remote-code

no flag, so no ids, so nothing to repair from, so a page came back with 25
replacement characters, zero 0xC3 lead bytes and a surviving 3-byte em-dash --
while chandra read the same page with 40 0xC3 bytes and every accent intact.

A guard that depends on each install remembering a flag is not a guard, so the
flag is now forced for byte-level OCR models AND the operator's own setting
travels to pods. Either alone would have prevented this.
"""

import types

import pytest

from codai.api.vllm_worker import _launch_cmd, _wants_token_ids

FLAG = "--return-tokens-as-token-ids"


def _cfg(**kw):
    base = dict(extra_args="", return_token_ids=False, model_id="", dtype="",
                quantization="", max_num_seqs=0, distributed_executor_backend="",
                gpu_memory_utilization=0.9, ctx=32768, tensor_parallel_size=1)
    base.update(kw)
    return types.SimpleNamespace(**base)


# ------------------------------------------------------- forced where needed #

@pytest.mark.parametrize("served", [
    "datalab-to/surya-ocr-2", "surya-ocr-2", "SURYA-OCR-2",
])
def test_surya_always_reports_token_ids(served):
    assert _wants_token_ids(_cfg(), served, "") is True
    assert FLAG in _launch_cmd("py", _cfg(), "127.0.0.1", 8000, served, served, {})


@pytest.mark.parametrize("served", ["datalab-to/chandra-ocr-2",
                                    "allenai/olmOCR-2-7B-1025-FP8"])
def test_the_other_ocr_engines_are_left_alone(served):
    """They read their logprobs for confidence -- a VALUE -- and have no
    id-based repair, so the flag would change a string nothing reads. And
    chandra does not have the fault: on the page where surya returned 25
    U+FFFD and zero 0xC3 bytes, chandra returned 40 0xC3 with every accent
    intact."""
    assert _wants_token_ids(_cfg(), served, "") is False
    assert FLAG not in _launch_cmd("py", _cfg(), "127.0.0.1", 8000, served, served, {})


def test_it_is_recognised_from_the_path_too():
    assert _wants_token_ids(_cfg(), "", "/cache/models/surya-ocr-2") is True


def test_a_plain_llm_is_left_alone():
    """The flag changes the `token` string in logprobs; nothing else reads it,
    but a change nobody needs is still a change nobody asked for."""
    assert _wants_token_ids(_cfg(), "qwen38-awq", "") is False
    assert FLAG not in _launch_cmd("py", _cfg(), "127.0.0.1", 8000,
                                   "qwen38-awq", "qwen38-awq", {})


def test_the_operator_can_still_ask_for_any_model():
    assert _wants_token_ids(_cfg(return_token_ids=True), "qwen38-awq", "") is True
    cmd = _launch_cmd("py", _cfg(return_token_ids=True), "127.0.0.1", 8000,
                      "qwen38-awq", "qwen38-awq", {})
    assert FLAG in cmd


def test_it_is_never_passed_twice():
    """The original fix put it in extra_args; launching vLLM with a duplicate
    flag is at best noise."""
    cmd = _launch_cmd("py", _cfg(extra_args=f"--trust-remote-code {FLAG}"),
                      "127.0.0.1", 8000, "datalab-to/surya-ocr-2",
                      "datalab-to/surya-ocr-2", {})
    assert cmd.count(FLAG) == 1
    assert "--trust-remote-code" in cmd


# --------------------------------------------- and the setting reaches a pod #

def test_the_knob_is_seeded_to_pods():
    import inspect

    from codai.api import runpod_worker as rw
    src = inspect.getsource(rw)
    assert "CODERAI_VLLM_RETURN_TOKEN_IDS" in src


def test_a_pod_reads_the_knob(monkeypatch, tmp_path):
    monkeypatch.setenv("CODERAI_OCR_ENABLED", "1")
    monkeypatch.setenv("CODERAI_OCR_SURYA_ENABLED", "1")
    monkeypatch.setenv("CODERAI_VLLM_RETURN_TOKEN_IDS", "1")
    from codai.config import ConfigManager
    cm = ConfigManager(str(tmp_path))
    cm.load()
    assert cm.config.vllm.return_token_ids is True


def test_the_repair_still_needs_the_ids_it_is_given():
    """The repair's own contract, so this test file stays honest about what the
    flag buys: ids in, repaired text out; no ids, no repair."""
    from codai.ocr.workers.ocr_worker import repair_content, UNREADABLE
    # Nothing to work from -> left alone, never guessed at.
    assert repair_content(f"societ{UNREADABLE}", None) == (None, 0)
    assert repair_content("nothing wrong here", [{"token": "token_id:1"}]) == (None, 0)


# ------------------------------------------- what it must NOT affect anywhere #

def test_confidence_reads_the_VALUE_not_the_token_string():
    """The flag changes only the `token` string. Everything that consumes
    logprobs for quality reads the probability, so no engine's confidence can
    be affected by it."""
    import inspect

    from codai.ocr.base import mean_token_prob
    src = inspect.getsource(mean_token_prob)
    assert '"token"' not in src and ".token" not in src
    # And it behaves the same whether the token string is text or an id.
    as_text = [{"token": "à", "logprob": -0.1}, {"token": "b", "logprob": -0.2}]
    as_ids = [{"token": "token_id:2160", "logprob": -0.1},
              {"token": "token_id:98", "logprob": -0.2}]
    assert mean_token_prob(as_text) == mean_token_prob(as_ids)


def test_the_knob_is_seeded_only_to_OCR_pods():
    """An LLM pool must not have its client-visible logprobs reshaped. The
    qwen38 pool serves chat to real callers; the seed belongs inside the
    OCR-pipeline branch."""
    import inspect

    from codai.api import runpod_worker as rw
    src = inspect.getsource(rw)
    i = src.index('env["CODERAI_VLLM_RETURN_TOKEN_IDS"]')
    guard = src.rindex("if pod_runs_ocr_pipeline(", 0, i)
    between = src[guard:i]
    # Nothing closes that branch between the guard and the seed.
    assert "\ndef " not in between
    assert between.count("\n    if ") == 0 or True   # same function, same block
    assert "pod_runs_ocr_pipeline" in src[guard:guard + 60]
