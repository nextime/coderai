"""A trained text adapter reaching a RunPod pod, and being selectable per request.

This path existed on paper and had never been run end to end: a QLoRA trained for
a client sat on disk for ten days while a second service rented its own pod daily
to serve it over SSH. Five things were in the way, one test each below.

No GPU, no network, no pod: the blob store is a temp dir and the HTTP calls are
stubbed.
"""

import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codai.models import text_loras


ADAPTER = {"peft_type": "LORA", "r": 16, "lora_alpha": 32,
           "base_model_name_or_path": "Qwen/Qwen3.8-27B"}


@pytest.fixture
def adapter(tmp_path):
    """A PEFT adapter directory, the shape a QLoRA trainer actually writes."""
    d = tmp_path / "diogene-alpha"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps(ADAPTER))
    (d / "adapter_model.safetensors").write_bytes(b"\x00" * 4096)
    return d


@pytest.fixture
def blob_store(tmp_path, monkeypatch):
    """Point the content-addressed store at a temp dir."""
    from codai.api import loras as loras_api
    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setattr(loras_api, "_LORAS_DIR", str(store), raising=False)
    return store


# ---------------------------------------------------------------- the bundle

def test_an_adapter_directory_becomes_one_blob_with_a_stable_id(adapter):
    """The store holds single files; a PEFT adapter is a directory of them.

    It has to hash the same every time or a pod would be re-sent 159 MB because
    a timestamp moved, so the tar is built with fixed uid/gid/mtime/mode.
    """
    first = text_loras.bundle_dir(str(adapter))
    second = text_loras.bundle_dir(str(adapter))
    assert first and first == second
    assert text_loras.is_bundle(first)
    assert text_loras.blob_id(str(adapter)).startswith("sha256:")
    # Touching the file must not change the id: content is what is addressed.
    os.utime(adapter / "adapter_config.json", (1, 1))
    assert text_loras.bundle_dir(str(adapter)) == first


def test_the_bundle_carries_the_config_vllm_needs_to_load_it(adapter):
    """A bare .safetensors is not loadable: vLLM reads adapter_config.json for
    the rank and target modules. Sending only the weights would fail on the pod."""
    import io
    names = tarfile.open(fileobj=io.BytesIO(text_loras.bundle_dir(str(adapter)))).getnames()
    assert sorted(names) == ["adapter_config.json", "adapter_model.safetensors"]


def test_a_single_file_adapter_still_hashes_as_itself(tmp_path):
    f = tmp_path / "one.safetensors"
    f.write_bytes(b"abc")
    import hashlib
    assert text_loras.blob_id(str(f)) == "sha256:" + hashlib.sha256(b"abc").hexdigest()
    assert not text_loras.is_bundle(b"abc")


# ---------------------------------------------------------------- the receiver

def test_the_store_unpacks_a_bundle_so_vllm_gets_a_directory(adapter, blob_store):
    from codai.api import loras as loras_api
    data = text_loras.bundle_dir(str(adapter))
    h, existed = loras_api.save_lora_blob(data)
    assert not existed
    path = loras_api._lora_blob_path(h)
    assert os.path.isdir(path), "an adapter must resolve to a directory, not a tar"
    assert os.path.isfile(os.path.join(path, "adapter_config.json"))
    assert loras_api.lora_blob_exists(h)
    # Sending it twice costs nothing and does not re-extract.
    assert loras_api.save_lora_blob(data)[1] is True


def test_a_bundle_cannot_write_outside_the_blob_store(blob_store):
    """The bytes come off the network. A member named ../../x would land in the
    filesystem, so names are checked before anything is extracted."""
    import io
    from fastapi import HTTPException
    from codai.api import loras as loras_api
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        info = tarfile.TarInfo("../../escaped.txt")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"bad"))
    with pytest.raises(HTTPException):
        loras_api.save_lora_blob(buf.getvalue())
    assert not (blob_store.parent.parent / "escaped.txt").exists()


# ---------------------------------------------------------------- the sender

class _Resp:
    def __init__(self, code):
        self.status_code = code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


@pytest.fixture
def pod_calls(monkeypatch):
    """Count what the sender actually asks the pod for."""
    from codai.api import remote_gateway as rg
    calls = {"get": [], "post": []}

    class _Http:
        @staticmethod
        def get(url, **kw):
            calls["get"].append(url)
            return _Resp(404)            # the pod does not have it yet

        @staticmethod
        def post(url, **kw):
            calls["post"].append((url, len(kw.get("data") or b"")))
            return _Resp(200)

    monkeypatch.setattr(rg, "pod_http", _Http)
    rg._LORA_SENT.clear()
    return calls


def test_a_trained_adapter_directory_is_sent_instead_of_dropped(adapter, pod_calls):
    """It used to be reported and skipped — "publish it to HuggingFace" — which
    is why a client's private adapter could not run on a pod at all."""
    from codai.api import remote_gateway as rg
    blob = rg.push_lora_file("http://pod:8000", "k", str(adapter))
    assert blob.startswith("sha256:")
    assert len(pod_calls["post"]) == 1, "the bundle must be uploaded"
    assert pod_calls["post"][0][1] > 4096


def test_the_adapter_is_not_re_read_on_every_request(adapter, pod_calls):
    """A Target is rebuilt per request, so this ran per request: 159 MB re-read
    and re-hashed, plus a round trip, to learn the pod already had it."""
    from codai.api import remote_gateway as rg
    first = rg.push_lora_file("http://pod:8000", "k", str(adapter))
    before = (len(pod_calls["get"]), len(pod_calls["post"]))
    for _ in range(5):
        assert rg.push_lora_file("http://pod:8000", "k", str(adapter)) == first
    assert (len(pod_calls["get"]), len(pod_calls["post"])) == before, \
        "a repeat must not touch the pod at all"


def test_a_retrained_adapter_at_the_same_path_is_sent_again(adapter, pod_calls):
    """The cache must not serve stale weights: training writes the same path."""
    from codai.api import remote_gateway as rg
    first = rg.push_lora_file("http://pod:8000", "k", str(adapter))
    (adapter / "adapter_model.safetensors").write_bytes(b"\x01" * 8192)
    second = rg.push_lora_file("http://pod:8000", "k", str(adapter))
    assert second != first
    assert len(pod_calls["post"]) == 2


def test_a_new_pod_is_sent_the_adapter_it_does_not_have(adapter, pod_calls):
    from codai.api import remote_gateway as rg
    rg.push_lora_file("http://pod-a:8000", "k", str(adapter))
    rg.push_lora_file("http://pod-b:8000", "k", str(adapter))
    assert len(pod_calls["post"]) == 2


def test_the_entry_sent_to_the_pod_names_the_adapter_by_content_id(adapter, pod_calls):
    from codai.api import remote_gateway as rg
    entry = {"id": "diogene-alpha",
             "loras": [{"name": "diogene-alpha", "path": str(adapter), "weight": 1.0}]}
    out = rg.ensure_text_loras("http://pod:8000", "k", entry)
    assert out["loras"][0]["path"].startswith("sha256:")
    assert out["loras"][0]["name"] == "diogene-alpha"


# ---------------------------------------------------------------- the launch

def _models(*entries):
    class _CM:
        models_data = {"text_models": list(entries)}
    return _CM


def test_a_content_id_is_resolved_to_a_path_vllm_can_open(adapter, blob_store,
                                                          monkeypatch):
    """On the pod the adapter is named by hash, which is not a path. Passing it
    through produced `--lora-modules diogene-alpha=sha256:...`, which vLLM
    cannot open — and the launch failed minutes later, inside a pod."""
    from codai.api import loras as loras_api
    from codai.api import vllm_worker
    h, _ = loras_api.save_lora_blob(text_loras.bundle_dir(str(adapter)))
    entry = {"id": "diogene-alpha", "path": "base/model",
             "loras": [{"name": "diogene-alpha", "path": f"sha256:{h}"}]}
    monkeypatch.setattr("codai.admin.routes.config_manager", _models(entry),
                        raising=False)
    args = vllm_worker._lora_args(entry)
    assert "--enable-lora" in args
    mod = args[args.index("--lora-modules") + 1]
    name, _, src = mod.partition("=")
    assert name == "diogene-alpha"
    assert not src.startswith("sha256:"), f"unresolved content id: {mod}"
    assert os.path.isdir(src) and os.path.isfile(os.path.join(src, "adapter_config.json"))


def test_one_engine_serves_the_base_and_every_adapter_of_it(adapter, blob_store,
                                                            monkeypatch):
    """vLLM registers adapters at LAUNCH and one vLLM serves one base. The pod
    created for the base alone got no --enable-lora, so a later request for an
    adapter of it had nowhere to go — the whole point of a shared pool."""
    from codai.api import loras as loras_api
    from codai.api import vllm_worker
    h, _ = loras_api.save_lora_blob(text_loras.bundle_dir(str(adapter)))
    base = {"id": "qwen38-awq", "path": "qwen/base",
            "runpod": {"hf_repo": "cyankiwi/Qwen3.8-27B-AWQ-INT4"}}
    tuned = {"id": "diogene-alpha", "path": "qwen/base",
             "loras": [{"name": "diogene-alpha", "path": f"sha256:{h}"}]}
    monkeypatch.setattr("codai.admin.routes.config_manager", _models(base, tuned),
                        raising=False)
    args = vllm_worker._lora_args(base)          # launched for the BASE
    assert "--enable-lora" in args, "the base's engine must carry its adapters"
    mods = args[args.index("--lora-modules") + 1:]
    assert any(m.startswith("diogene-alpha=") for m in mods), mods


def test_an_adapter_is_served_under_the_name_requests_ask_for(adapter, blob_store,
                                                              monkeypatch):
    """The request says model=diogene-alpha, so that must be the module name —
    not the adapter file's own name."""
    from codai.api import loras as loras_api
    from codai.api import vllm_worker
    h, _ = loras_api.save_lora_blob(text_loras.bundle_dir(str(adapter)))
    base = {"id": "qwen38-awq", "path": "qwen/base"}
    tuned = {"id": "diogene-alpha-v2-candidato", "path": "qwen/base",
             "loras": [{"name": "whatever-the-file-was", "path": f"sha256:{h}"}]}
    monkeypatch.setattr("codai.admin.routes.config_manager", _models(base, tuned),
                        raising=False)
    mods = vllm_worker._lora_args(base)
    assert any(m.startswith("diogene-alpha-v2-candidato=") for m in mods)


def test_several_adapters_can_be_active_in_one_batch(adapter, blob_store, monkeypatch):
    """--max-loras defaults to 1, so requests naming different adapters cannot
    batch together and serialise instead. The cost lands on latency, not on an
    error, which is why it goes unnoticed."""
    from codai.api import loras as loras_api
    from codai.api import vllm_worker
    h, _ = loras_api.save_lora_blob(text_loras.bundle_dir(str(adapter)))
    base = {"id": "qwen38-awq", "path": "qwen/base"}
    tuned = [{"id": f"diogene-v{i}", "path": "qwen/base",
              "loras": [{"name": f"v{i}", "path": f"sha256:{h}"}]} for i in (1, 2, 3)]
    monkeypatch.setattr("codai.admin.routes.config_manager", _models(base, *tuned),
                        raising=False)
    args = vllm_worker._lora_args(base)
    assert "--max-loras" in args
    assert int(args[args.index("--max-loras") + 1]) >= 3
    assert int(args[args.index("--max-lora-rank") + 1]) >= 16


def test_an_adapter_that_is_not_here_is_reported_not_guessed(monkeypatch):
    """Better a model served without its adapter, said out loud, than a launch
    command that fails inside a pod on a path nobody can inspect."""
    from codai.api import vllm_worker
    entry = {"id": "diogene-alpha", "path": "qwen/base",
             "loras": [{"name": "diogene-alpha", "path": "sha256:" + "0" * 64}]}
    monkeypatch.setattr("codai.admin.routes.config_manager", _models(entry),
                        raising=False)
    assert vllm_worker._lora_args(entry) == []


def test_a_huggingface_adapter_still_travels_as_a_repo_id(monkeypatch):
    """A repo id needs no blob store: the pod downloads it itself."""
    from codai.api import vllm_worker
    entry = {"id": "tuned", "path": "qwen/base",
             "loras": [{"name": "tuned", "path": "someone/an-adapter"}]}
    monkeypatch.setattr("codai.admin.routes.config_manager", _models(entry),
                        raising=False)
    mods = vllm_worker._lora_args(entry)
    assert "tuned=someone/an-adapter" in mods
