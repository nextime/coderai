"""Serving the same model is not enough to inherit a pod.

Adoption reuses a pod another process (or this one, before a restart) rented for
the same pool, instead of renting a second card. The registry recorded the pool
and nothing about what the pod was RUNNING, so two silent mistakes were possible:

  * **the image.** The same model on a pinned tag and on `:latest` is different
    code. A pool configured for one could inherit a pod running the other and
    nobody could tell from outside — a whole evening went into chasing an OCR
    result from a pod that turned out not to be running what everyone assumed;
  * **the card.** A pod rented as a FALLBACK is on a card the configuration
    accepts only when its preferred one is unavailable. Inheriting one would
    serve a lease that asked for an A40 *and nothing else* (v0.3.54) from an
    RTX A6000, which is exactly the comparison that lease exists to protect.

So both are recorded at creation and both are checked before adopting, and an
entry that records neither is refused rather than trusted.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import runpod_worker as rw                          # noqa: E402

IMG = "ghcr.io/nextime/coderai-vllm:0.3.53"


def _registry(monkeypatch, tmp_path, entries):
    path = tmp_path / "runpod-pods.json"
    path.write_text(json.dumps(entries))
    monkeypatch.setattr(rw, "_pod_registry_path", lambda: str(path))
    monkeypatch.setattr(rw, "_pod_health_ok", lambda *a, **k: True)
    return path


def _entry(image=IMG, gpu="NVIDIA A40", pool="m"):
    return {"pool": pool, "proc": "someone-else", "url": "https://h/v1",
            "key": "k", "hourly_usd": 0.59, "image": image, "gpu": gpu}


def test_a_matching_pod_is_adopted(monkeypatch, tmp_path):
    _registry(monkeypatch, tmp_path, {"p1": _entry()})
    pod_id, url, key, rate = rw.find_shared_pod("m", "/healthz", "mine",
                                                image=IMG, gpus=["NVIDIA A40"])
    assert (pod_id, url, rate) == ("p1", "https://h/v1", 0.59)


def test_a_different_image_is_refused(monkeypatch, tmp_path, capsys):
    _registry(monkeypatch, tmp_path,
              {"p1": _entry(image="ghcr.io/nextime/coderai-vllm:latest")})
    assert rw.find_shared_pod("m", "/healthz", "mine", image=IMG,
                              gpus=["NVIDIA A40"])[0] is None
    out = capsys.readouterr().out
    assert "not adopting pod p1" in out and ":latest" in out


def test_a_fallback_card_is_refused(monkeypatch, tmp_path, capsys):
    _registry(monkeypatch, tmp_path, {"p1": _entry(gpu="NVIDIA RTX A6000")})
    assert rw.find_shared_pod("m", "/healthz", "mine", image=IMG,
                              gpus=["NVIDIA A40"])[0] is None
    out = capsys.readouterr().out
    assert "RTX A6000" in out and "NVIDIA A40" in out


def test_any_of_several_preferred_cards_is_accepted(monkeypatch, tmp_path):
    _registry(monkeypatch, tmp_path, {"p1": _entry(gpu="NVIDIA L40S")})
    assert rw.find_shared_pod("m", "/healthz", "mine", image=IMG,
                              gpus=["NVIDIA A40", "NVIDIA L40S"])[0] == "p1"


def test_the_card_name_matches_with_or_without_the_vendor(monkeypatch, tmp_path):
    # RunPod spells the id and the display name differently in places.
    _registry(monkeypatch, tmp_path, {"p1": _entry(gpu="A40")})
    assert rw.find_shared_pod("m", "/healthz", "mine", image=IMG,
                              gpus=["NVIDIA A40"])[0] == "p1"


def test_an_entry_from_before_this_check_is_refused(monkeypatch, tmp_path, capsys):
    """Trusting it risks a silent answer from the wrong code; refusing costs one
    extra pod once, and the old one is reaped."""
    old = _entry()
    del old["image"]
    del old["gpu"]
    _registry(monkeypatch, tmp_path, {"p1": old})
    assert rw.find_shared_pod("m", "/healthz", "mine", image=IMG,
                              gpus=["NVIDIA A40"])[0] is None
    assert "unrecorded image" in capsys.readouterr().out


def test_an_unconstrained_caller_still_adopts(monkeypatch, tmp_path):
    """A pool with no preferred card and a caller that names no image keep the
    old behaviour: the checks are constraints, not new requirements."""
    old = _entry()
    del old["image"]
    del old["gpu"]
    _registry(monkeypatch, tmp_path, {"p1": old})
    assert rw.find_shared_pod("m", "/healthz", "mine")[0] == "p1"


def test_another_pools_pod_is_never_considered(monkeypatch, tmp_path):
    _registry(monkeypatch, tmp_path, {"p1": _entry(pool="other")})
    assert rw.find_shared_pod("m", "/healthz", "mine", image=IMG,
                              gpus=["NVIDIA A40"])[0] is None


def test_the_image_and_card_are_recorded_at_creation(monkeypatch, tmp_path):
    path = tmp_path / "runpod-pods.json"
    path.write_text("{}")
    monkeypatch.setattr(rw, "_pod_registry_path", lambda: str(path))
    rw.register_pod("p1", "m", "https://h/v1", "k", hourly_usd=0.59,
                    image=IMG, gpu="NVIDIA A40")
    info = json.loads(path.read_text())["p1"]
    assert info["image"] == IMG and info["gpu"] == "NVIDIA A40"


def test_a_re_registration_keeps_them(monkeypatch, tmp_path):
    """Adoption re-registers the pod under the new owner and does not know what
    it is running; erasing the fields would make it un-adoptable next time."""
    path = tmp_path / "runpod-pods.json"
    path.write_text(json.dumps({"p1": _entry()}))
    monkeypatch.setattr(rw, "_pod_registry_path", lambda: str(path))
    rw.register_pod("p1", "m", "https://h/v1", "k", hourly_usd=0.59)
    info = json.loads(path.read_text())["p1"]
    assert info["image"] == IMG and info["gpu"] == "NVIDIA A40"


def test_the_pool_passes_both_when_it_adopts():
    """Pinned against the source: the checks are only worth anything if the one
    caller actually uses them."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    blk = src.split("def _adopt_shared_pod")[1][:3500]
    assert "image=want_image, gpus=want_gpus" in blk
    assert 'want_image = str(plan.get("image") or "")' in blk
    assert "narrow_gpu_choice(self.mcfg" in blk
