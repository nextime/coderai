"""A global volume must not pin the pod to a region — that is the whole point.

RunPod's Global Volumes (beta, Sept 2026) are region-independent storage backed
by object storage. A NETWORK volume lives in one data centre and a pod elsewhere
cannot attach it, which is why coderai forces the pod into the volume's region.
Applying that same pinning to a global volume would throw away the only reason
to use one: store the weights once, rent a GPU in whichever region has one.

So the two are resolved separately, and these tests hold them apart.
"""
import pathlib

from codai.api import runpod_worker as rw

ROOT = pathlib.Path(rw.__file__).resolve().parents[2]


class _Acct:
    def __init__(self, net="", glob="", dc="", mount="/workspace"):
        self.network_volume_id = net
        self.global_volume_id = glob
        self.data_center = dc
        self.volume_mount_path = mount


def test_the_account_setting_applies_to_every_model():
    mcfg = rw.parse_model_runpod({})
    assert rw.global_volume_for(mcfg, _Acct(glob="gv-123")) == "gv-123"


def test_a_model_can_override_the_account_volume():
    mcfg = rw.parse_model_runpod({"global_volume_id": "gv-model"})
    assert rw.global_volume_for(mcfg, _Acct(glob="gv-account")) == "gv-model"


def test_no_volume_configured_is_empty_not_an_error():
    assert rw.global_volume_for(rw.parse_model_runpod({}), _Acct()) == ""


def test_a_global_volume_does_not_pin_the_data_centre():
    """The network volume's region wins over everything; a global volume must
    leave the choice alone so the scaler can rank cards across all regions."""
    pool = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    pool.account = _Acct(glob="gv-123")
    pool.mcfg = rw.parse_model_runpod({"global_volume_id": "gv-123"})
    pool._volume_dc = None
    assert pool._data_center() == ""      # free to go anywhere
    pool.mcfg.data_center = "EU-RO-1"
    assert pool._data_center() == "EU-RO-1"   # only because the model asked


def test_only_the_network_volume_triggers_the_region_lookup():
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    blk = src.split("vol_id, vol_mount = volume_for")[1].split("self._tls = False")[0]
    # The region lookup sits under the network-volume branch, and the global
    # branch that follows must not contain it.
    net, glob = blk.split("global_vol = global_volume_for")
    assert "_data_center_for_volume" in net
    assert "_data_center_for_volume" not in glob


def test_the_two_volumes_are_not_attached_together_by_accident():
    """With both attached RunPod moves the global one to /workspace-global, so
    pointing the pod env at /workspace would aim it at the wrong volume."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    blk = src.split("global_vol = global_volume_for")[1][:600]
    assert "if global_vol and not vol_id:" in blk
    assert "/workspace-global" in src


def test_the_client_sends_the_field_on_both_api_shapes():
    """create_pod has a GraphQL body and a REST body; a volume wired into only
    one of them attaches on some code paths and silently not on others."""
    src = (ROOT / "codai/api/runpod_client.py").read_text()
    assert src.count("GLOBAL_VOLUME_FIELD] = global_volume_id") == 2
    assert "global_volume_id: str = \"\"" in src


def test_the_field_name_is_overridable_because_it_is_undocumented():
    """RunPod documents global volumes as console-only and the v1 validator
    accepts unknown body keys silently, so a wrong name cannot be detected by
    probing — it yields a pod with no volume that re-downloads its weights."""
    from codai.api import runpod_client
    assert runpod_client.GLOBAL_VOLUME_FIELD
    src = (ROOT / "codai/api/runpod_client.py").read_text()
    assert "CODERAI_RUNPOD_GLOBAL_VOLUME_FIELD" in src
    assert "VERIFY" in src        # says out loud that it is unconfirmed


def test_the_config_documents_that_it_cannot_be_created_by_api():
    cfgsrc = (ROOT / "codai/config.py").read_text()
    blk = cfgsrc.split("global_volume_id")[0][-1400:]
    assert "console" in blk.lower()
    assert "STANDARD" in blk      # the enum the API actually allows


# ------------------------------------------- "anywhere in the EU", said safely

class _AcctDC(_Acct):
    def __init__(self, dcs=None, dc="", **kw):
        super().__init__(**kw)
        self.data_centers = dcs or []
        self.data_center = dc


def _pool(mcfg_block=None, acct=None):
    pool = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    pool.account = acct or _AcctDC()
    pool.mcfg = rw.parse_model_runpod(mcfg_block or {})
    pool._volume_dc = None
    return pool


def test_a_region_list_is_parsed_from_a_string_or_a_list():
    for value in ("EU-RO-1,EU-FR-1", "EU-RO-1 EU-FR-1", ["eu-ro-1", "EU-FR-1"]):
        assert rw._parse_dc_list(value) == ["EU-RO-1", "EU-FR-1"], value
    assert rw._parse_dc_list("") == []
    assert rw._parse_dc_list(None) == []


def test_the_order_is_the_fallback_order_and_is_preserved():
    """Not sorted: the first entry is the preferred region and the scaler only
    moves on when it cannot get a card there."""
    assert rw._parse_dc_list("EU-RO-1,EU-FR-1,EUR-NO-1")[0] == "EU-RO-1"
    assert rw._parse_dc_list("EU-FR-1,EU-RO-1")[0] == "EU-FR-1"


def test_duplicates_collapse():
    assert rw._parse_dc_list("EU-RO-1, eu-ro-1 ,EU-FR-1") == ["EU-RO-1", "EU-FR-1"]


def test_the_model_list_beats_the_account_list():
    pool = _pool({"data_centers": "EU-FR-1"}, _AcctDC(dcs=["EU-RO-1"]))
    assert pool._data_centers() == ["EU-FR-1"]


def test_a_single_data_center_still_works_exactly_as_before():
    pool = _pool({"data_center": "EU-SE-1"})
    assert pool._data_centers() == ["EU-SE-1"]
    assert pool._data_center() == "EU-SE-1"


def test_nothing_configured_still_means_runpod_chooses():
    pool = _pool()
    assert pool._data_centers() == [""]
    assert pool._data_center() == ""


def test_a_network_volume_overrides_every_region_setting():
    """It exists in ONE region and no pod elsewhere can attach it, so a
    multi-region list must not send a pod where the attach would fail."""
    pool = _pool({"data_centers": "EU-FR-1,EU-RO-1"}, _AcctDC(dcs=["EUR-NO-1"]))
    pool._volume_dc = "EU-NL-1"
    assert pool._data_centers() == ["EU-NL-1"]


def test_candidates_are_expanded_across_regions_for_the_capacity_fallback():
    """The existing retry walks `ranked` on a capacity miss; pairing each card
    with each region makes it walk regions too, with no new control flow."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    blk = src.split("ranked = _rank_gpus(client, self.mcfg, self.account)")[1][:700]
    assert 'dict(sel, _dc=dc) for sel in ranked for dc in dcs' in blk
    assert "if len(dcs) > 1:" in blk


def test_the_pod_is_created_in_and_records_the_region_it_got():
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    assert src.count('sel.get("_dc") or self._data_center()') >= 3   # create, log, handle
