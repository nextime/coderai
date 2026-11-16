"""A pod pulled the base with no credentials, because nothing sent any.

`TrainerPodSpec` was built without `env`, so a rented pod received exactly
three variables -- CODERAI_TRAIN_JOB, _SHARD, _SECRET -- and pulled the base
anonymously. For a public base that is a rate limit; for a gated one it is a
404 that arrives minutes into a boot already being billed, and the pod is paid
for either way.

Worth being exact about what this does NOT fix: Qwen/Qwen3.8-27B is public
(measured: anonymous HTTP 200), so the reported CAS transfer failure was never
an authentication problem and a token does not cure it.

The token is a credential that now travels to a third-party cloud, so: the
install's own resolver is the single source, a per-model block can override it,
and nothing prints it.
"""

import inspect

import pytest

from codai.api import loras
from codai.api import text_lora_fleet as flt


# ------------------------------------------------------- where it comes from --

def test_the_blocks_token_wins():
    """One model may train from a repo the rest of the install cannot read."""
    cfg = flt.LoraTrainConfig(hf_token="B" * 31)
    assert flt.hf_token_for_pods(cfg) == "B" * 31


def test_it_falls_back_to_the_installs_own_resolver(monkeypatch):
    """NOT a second scan of the environment: the resolver also reads the token
    FILES, which is where this install actually keeps it. Two answers to
    "where is the token" is how one path authenticates and another does not."""
    import codai.hf_cache as hc
    monkeypatch.setattr(hc, "configured_hf_token", lambda *a, **k: "F" * 31)
    assert flt.hf_token_for_pods(flt.LoraTrainConfig()) == "F" * 31


def test_nothing_configured_gives_an_empty_string(monkeypatch):
    import codai.hf_cache as hc
    monkeypatch.setattr(hc, "configured_hf_token", lambda *a, **k: None)
    assert flt.hf_token_for_pods(flt.LoraTrainConfig()) == ""


def test_an_unimportable_resolver_still_finds_an_env_token(monkeypatch):
    """The fallback path exists so a worker with a trimmed import tree does not
    silently train unauthenticated."""
    import builtins
    real = builtins.__import__

    def boom(name, *a, **k):
        if name == "codai.hf_cache":
            raise ImportError("no")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    monkeypatch.setenv("HF_TOKEN", "E" * 31)
    assert flt.hf_token_for_pods(flt.LoraTrainConfig()) == "E" * 31


# --------------------------------------------------------- where it goes to --

def test_the_spec_is_built_with_an_env_at_all():
    """The whole defect in one assertion: env= was simply absent."""
    src = inspect.getsource(loras._text_train_distributed)
    assert "flt.TrainerPodSpec(" in src
    spec = src[src.index("flt.TrainerPodSpec("):]
    assert "env=_pod_env" in spec[:400]


def test_both_variable_names_are_sent():
    """Which name an image honours is not worth learning one rental at a time."""
    src = inspect.getsource(loras._text_train_distributed)
    assert '"HF_TOKEN": _hf' in src and '"HUGGING_FACE_HUB_TOKEN": _hf' in src


def test_no_token_is_warned_about_rather_than_passed_as_empty():
    """An empty value would look configured and authenticate as nobody."""
    src = inspect.getsource(loras._text_train_distributed)
    i = src.index("_pod_env = {}")
    assert "if _hf:" in src[i:i + 200]
    assert "anonymously" in src


def test_the_pod_env_reaches_the_create_call():
    """spec.env has to be what _rent_one starts from, or the field is decoration."""
    src = inspect.getsource(flt.TrainerFleet._rent_one)
    assert "dict(self.spec.env or {})" in src


def test_the_job_message_path_never_carries_the_token():
    """on_event text goes into a job record a client polls."""
    src = inspect.getsource(loras._text_train_distributed)
    for line in src.splitlines():
        if "_update_job" in line or "on_event" in line:
            assert "_hf" not in line, f"a token could reach a job message: {line}"


# --------------------------------------------------------------- redaction ---

@pytest.mark.parametrize("secret", ["A" * 37, "hf_" + "x" * 34])
def test_redact_removes_a_secret(secret):
    assert secret not in flt.redact(f"env={secret} more", secret)
    assert "<redacted>" in flt.redact(f"env={secret}", secret)


def test_redact_ignores_something_too_short_to_be_a_token():
    """Replacing a 3-character 'secret' would corrupt unrelated log text."""
    assert flt.redact("the cost was 0.90/hr", "0.9") == "the cost was 0.90/hr"
