"""One network volume per region, so multi-region renting keeps its storage.

RunPod ties a network volume to a single data centre. With one volume id the
pool had to pin every pod to that region and throw away the `data_centers`
fallback list — an install chose between renting wherever a card was free and
re-downloading the weights onto every cold pod. These tests pin the map's
behaviour: the region is chosen first, the volume follows it, and a region with
no volume is dropped loudly rather than rented without storage.
"""
import pytest

from codai.api import runpod_worker as rw
from codai.api.runpod_worker import (parse_model_runpod, volume_for,
                                     volume_map_for, _parse_volume_map)


class _Acct:
    def __init__(self, vol="", vmap=None, mount="", dcs=None, dc=""):
        self.network_volume_id = vol
        self.network_volumes = vmap or {}
        self.volume_mount_path = mount
        self.data_centers = dcs or []
        self.data_center = dc
        self.global_volume_id = ""


# --- the parser -------------------------------------------------------------

def test_map_from_dict_upper_cases_regions():
    # The feature turns on the map's key equalling the region a candidate was
    # ranked for, so a map written in lower case must still match.
    assert _parse_volume_map({"eu-se-1": "vol-a"}) == {"EU-SE-1": "vol-a"}


def test_map_from_string_form():
    assert _parse_volume_map("EU-SE-1=vol-a, EU-RO-1=vol-b") == {
        "EU-SE-1": "vol-a", "EU-RO-1": "vol-b"}


def test_map_drops_regions_with_no_volume():
    # A region with a blank id is the one outcome the map exists to prevent.
    assert _parse_volume_map({"EU-SE-1": "vol-a", "EU-RO-1": ""}) == {"EU-SE-1": "vol-a"}


def test_map_of_nothing_is_empty():
    assert _parse_volume_map(None) == {} and _parse_volume_map("") == {}


def test_parse_model_runpod_reads_the_map():
    cfg = parse_model_runpod({"network_volumes": {"EU-SE-1": "vol-a"}})
    assert cfg.network_volumes == {"EU-SE-1": "vol-a"}


def test_a_map_forces_secure_cloud_like_a_single_id():
    # RunPod serves network volumes on Secure Cloud only; COMMUNITY candidates
    # would just be rejected at attach time.
    assert parse_model_runpod(
        {"network_volumes": {"EU-SE-1": "vol-a"}}).cloud_types == ["SECURE"]


# --- resolution -------------------------------------------------------------

def test_volume_for_picks_the_region_s_volume():
    cfg = parse_model_runpod({"network_volumes": {"EU-SE-1": "vol-a",
                                                  "EU-RO-1": "vol-b"}})
    assert volume_for(cfg, _Acct(), "EU-RO-1") == ("vol-b", "/workspace")
    assert volume_for(cfg, _Acct(), "eu-se-1") == ("vol-a", "/workspace")


def test_volume_for_refuses_a_region_the_map_does_not_name():
    # Never silently attach another region's volume: the attach would fail, and
    # worse, it would place the data somewhere the operator did not name.
    cfg = parse_model_runpod({"network_volumes": {"EU-SE-1": "vol-a"}})
    assert volume_for(cfg, _Acct(), "US-TX-1") == ("", "")


def test_volume_for_without_a_region_answers_is_there_storage():
    # Callers that size the disk or validate venv_on_volume have no region in
    # hand and only use the truthiness and the mount, which are the same for
    # every region.
    cfg = parse_model_runpod({"network_volumes": {"EU-SE-1": "vol-a"}})
    vol, mount = volume_for(cfg, _Acct())
    assert vol == "vol-a" and mount == "/workspace"


def test_single_id_still_behaves_exactly_as_before():
    cfg = parse_model_runpod({"network_volume_id": "vol-single"})
    assert volume_for(cfg, _Acct()) == ("vol-single", "/workspace")
    # and it is region-agnostic: the single id is not keyed by anything
    assert volume_for(cfg, _Acct(), "EU-RO-1") == ("vol-single", "/workspace")


def test_the_pool_s_own_map_wins_whole_not_merged():
    # A model that lists its regions is making a complete statement about where
    # its data may live; inheriting an extra region from the account would put
    # weights in a region the model never named.
    cfg = parse_model_runpod({"network_volumes": {"EU-SE-1": "vol-model"}})
    acct = _Acct(vmap={"EU-SE-1": "vol-acct", "US-TX-1": "vol-acct-2"})
    assert volume_map_for(cfg, acct) == {"EU-SE-1": "vol-model"}


def test_account_map_is_used_when_the_model_has_none():
    acct = _Acct(vmap={"EU-SE-1": "vol-acct"})
    assert volume_for(parse_model_runpod({}), acct, "EU-SE-1")[0] == "vol-acct"


def test_a_map_supersedes_the_single_id():
    cfg = parse_model_runpod({"network_volume_id": "vol-old",
                              "network_volumes": {"EU-RO-1": "vol-new"}})
    assert volume_for(cfg, _Acct(), "EU-RO-1")[0] == "vol-new"


def test_mount_path_override_applies_to_every_region():
    cfg = parse_model_runpod({"network_volumes": {"EU-SE-1": "vol-a"},
                              "volume_mount_path": "/data"})
    assert volume_for(cfg, _Acct(), "EU-SE-1") == ("vol-a", "/data")


# --- placement --------------------------------------------------------------

class _Pool:
    """Just enough of RunpodPodPool to exercise its region resolution."""
    model_key = "m"
    _data_centers = rw.RunpodPodPool._data_centers
    _with_volumes = rw.RunpodPodPool._with_volumes

    def __init__(self, mcfg, account, volume_dc="", seed_dc=""):
        self.mcfg, self.account = mcfg, account
        self._volume_dc, self._seed_dc = volume_dc, seed_dc


def test_a_single_volume_still_collapses_the_region_list():
    # The behaviour that made the feature necessary, kept for the single id: a
    # pod outside the volume's region cannot attach it.
    cfg = parse_model_runpod({"network_volume_id": "vol-a",
                              "data_centers": ["EU-SE-1", "EU-RO-1"]})
    assert _Pool(cfg, _Acct(), volume_dc="EU-NL-1")._data_centers() == ["EU-NL-1"]


def test_a_map_keeps_the_whole_region_list():
    cfg = parse_model_runpod({"network_volumes": {"EU-SE-1": "a", "EU-RO-1": "b"},
                              "data_centers": ["EU-SE-1", "EU-RO-1"]})
    pool = _Pool(cfg, _Acct(), volume_dc="EU-SE-1")
    assert pool._data_centers() == ["EU-SE-1", "EU-RO-1"]


def test_order_is_preserved_because_order_is_the_fallback_order():
    cfg = parse_model_runpod({"network_volumes": {"EU-SE-1": "a", "EU-RO-1": "b"},
                              "data_centers": ["EU-RO-1", "EU-SE-1"]})
    assert _Pool(cfg, _Acct())._data_centers() == ["EU-RO-1", "EU-SE-1"]


def test_a_region_with_no_volume_is_dropped(capsys):
    cfg = parse_model_runpod({"network_volumes": {"EU-SE-1": "a"},
                              "data_centers": ["EU-SE-1", "EU-RO-1"]})
    assert _Pool(cfg, _Acct())._data_centers() == ["EU-SE-1"]
    out = capsys.readouterr().out
    # Dropping it silently would hide a slow boot nobody could explain.
    assert "EU-RO-1" in out and "no volume configured" in out


def test_no_region_left_is_a_refusal_naming_both_sets():
    from codai.api.runpod_client import RunpodError
    cfg = parse_model_runpod({"network_volumes": {"US-TX-1": "a"},
                              "data_centers": ["EU-SE-1"]})
    with pytest.raises(RunpodError) as e:
        _Pool(cfg, _Acct())._data_centers()
    assert "EU-SE-1" in str(e.value) and "US-TX-1" in str(e.value)


def test_a_map_with_no_region_list_is_its_own_region_list():
    # The operator put a volume in each of those regions, which is a statement
    # about where this model may run.
    cfg = parse_model_runpod({"network_volumes": {"EU-SE-1": "a", "EU-RO-1": "b"}})
    assert _Pool(cfg, _Acct())._data_centers() == ["EU-SE-1", "EU-RO-1"]


def test_no_map_and_no_list_still_means_runpod_chooses():
    assert _Pool(parse_model_runpod({}), _Acct())._data_centers() == [""]


def test_a_seed_run_names_one_region_and_ignores_the_filter():
    # The volume it is filling is empty by definition; that is the point.
    cfg = parse_model_runpod({"network_volumes": {"EU-SE-1": "a"},
                              "data_centers": ["EU-SE-1"]})
    assert _Pool(cfg, _Acct(), seed_dc="eu-ro-1")._data_centers() == ["EU-RO-1"]


# --- seeding ----------------------------------------------------------------

def test_seed_region_refuses_a_region_with_no_volume():
    cfg = parse_model_runpod({"network_volumes": {"EU-SE-1": "a"}})

    class _P(_Pool):
        seed_region = rw.RunpodPodPool.seed_region
        pods = []

    with pytest.raises(ValueError) as e:
        _P(cfg, _Acct()).seed_region("US-TX-1")
    assert "nothing to" in str(e.value) and "EU-SE-1" in str(e.value)


def test_seed_region_boots_one_pod_then_terminates_it():
    cfg = parse_model_runpod({"network_volumes": {"EU-RO-1": "vol-b"}})

    class _Handle:
        def __init__(self, pid):
            self.pod_id = pid

    class _P(_Pool):
        seed_region = rw.RunpodPodPool.seed_region

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.pods, self.killed, self.seen_dc = [], [], None

        def _provision_one(self):
            # The region must be pinned for the duration of the boot, or the
            # pod lands somewhere the volume is not.
            self.seen_dc = self._data_centers()
            self.pods.append(_Handle("pod-new"))

        def _terminate(self, pod, reason):
            self.killed.append((pod.pod_id, reason))
            self.pods.remove(pod)

    pool = _P(cfg, _Acct())
    out = pool.seed_region("eu-ro-1")
    assert pool.seen_dc == ["EU-RO-1"]
    assert out["region"] == "EU-RO-1" and out["volume_id"] == "vol-b"
    assert out["pods"] == ["pod-new"]
    # The seed's product is on the volume, not in the pod.
    assert pool.killed and pool.killed[0][0] == "pod-new"
    assert pool.pods == []
    # And the pin does not outlive the seed.
    assert pool._seed_dc == ""


def test_seed_region_clears_the_pin_even_when_the_boot_fails():
    cfg = parse_model_runpod({"network_volumes": {"EU-RO-1": "vol-b"}})

    class _P(_Pool):
        seed_region = rw.RunpodPodPool.seed_region
        pods = []

        def _provision_one(self):
            raise RuntimeError("no capacity")

    pool = _P(cfg, _Acct())
    with pytest.raises(RuntimeError):
        pool.seed_region("EU-RO-1")
    # Otherwise every later pod for this model would be pinned to the region of
    # a seed that failed hours ago.
    assert pool._seed_dc == ""
