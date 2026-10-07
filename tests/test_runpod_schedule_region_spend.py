"""Per-model region, the warm-pod schedule, and the admin-scoped spend API.

The schedule governs only the WARM FLOOR: outside its window the pool stops
paying to hold a pod ready, but a request arriving out of hours must still
cold-start one. Several tests pin that distinction, because getting it wrong
turns a cost setting into an outage.
"""
import datetime
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _rw():
    from codai.api import runpod_worker
    return runpod_worker


def _at(y, m, d, hh, mm):
    return datetime.datetime(y, m, d, hh, mm).timestamp()


# --------------------------------------------------------------- parsing
def test_days_accept_names_numbers_and_comma_strings():
    rw = _rw()
    assert rw._parse_days("mon,tue,wed") == [0, 1, 2]
    assert rw._parse_days(["monday", "SUN"]) == [0, 6]
    assert rw._parse_days("0,4,6") == [0, 4, 6]
    assert rw._parse_days([1, 1, 2]) == [1, 2]
    # Unrecognised tokens are dropped, never guessed into a real day.
    assert rw._parse_days("mon,nonsense,9") == [0]
    assert rw._parse_days("") == [] and rw._parse_days(None) == []


def test_times_are_normalised_and_rubbish_rejected():
    rw = _rw()
    assert rw._parse_hhmm("8:5") == "08:05"
    assert rw._parse_hhmm("0805") == "08:05"
    assert rw._parse_hhmm("23:59") == "23:59"
    for bad in ("", "25:00", "08:70", "noon", "8", None):
        assert rw._parse_hhmm(bad) == "", bad


def test_an_incomplete_schedule_is_no_schedule():
    """A half-filled window must not silently pin the warm floor to zero."""
    rw = _rw()
    cfg = rw.parse_model_runpod({"min_pods": 1, "schedule_enabled": True,
                                 "schedule_start": "08:00"})
    assert cfg.schedule_enabled is False
    assert rw.schedule_state(cfg)["in_window"] is True


def test_no_schedule_always_reports_in_window():
    rw = _rw()
    cfg = rw.parse_model_runpod({"min_pods": 1})
    st = rw.schedule_state(cfg)
    assert st["enabled"] is False and st["in_window"] is True


# --------------------------------------------------------------- windows
@pytest.mark.parametrize("when,inside", [
    (_at(2026, 10, 7, 7, 59), False),   # Wed, before
    (_at(2026, 10, 7, 8, 0), True),     # Wed, start is inclusive
    (_at(2026, 10, 7, 19, 59), True),
    (_at(2026, 10, 7, 20, 0), False),   # end is exclusive
    (_at(2026, 10, 10, 12, 0), False),  # Saturday is not selected
])
def test_weekday_window(when, inside):
    rw = _rw()
    cfg = rw.parse_model_runpod({"min_pods": 1, "schedule_enabled": True,
                                 "schedule_start": "08:00", "schedule_end": "20:00",
                                 "schedule_days": "mon,tue,wed,thu,fri"})
    assert rw.schedule_state(cfg, when)["in_window"] is inside


@pytest.mark.parametrize("when,inside", [
    (_at(2026, 10, 9, 21, 59), False),  # Fri before start
    (_at(2026, 10, 9, 22, 30), True),   # Fri inside
    (_at(2026, 10, 10, 5, 59), True),   # Sat morning still Friday's window
    (_at(2026, 10, 10, 6, 0), False),   # window closed
    (_at(2026, 10, 10, 23, 0), False),  # Sat night is NOT selected
])
def test_overnight_window_belongs_to_the_day_it_starts_on(when, inside):
    rw = _rw()
    cfg = rw.parse_model_runpod({"min_pods": 1, "schedule_enabled": True,
                                 "schedule_start": "22:00", "schedule_end": "06:00",
                                 "schedule_days": "fri"})
    assert rw.schedule_state(cfg, when)["in_window"] is inside


def test_empty_day_list_means_every_day():
    rw = _rw()
    cfg = rw.parse_model_runpod({"min_pods": 1, "schedule_enabled": True,
                                 "schedule_start": "08:00", "schedule_end": "20:00"})
    for day in range(5, 7):  # Sat, Sun
        when = _at(2026, 10, 10 + (day - 5), 12, 0)
        assert rw.schedule_state(cfg, when)["in_window"] is True


def test_next_change_points_at_the_coming_boundary():
    rw = _rw()
    cfg = rw.parse_model_runpod({"min_pods": 1, "schedule_enabled": True,
                                 "schedule_start": "08:00", "schedule_end": "20:00",
                                 "schedule_days": "mon,tue,wed,thu,fri"})
    assert rw.schedule_state(cfg, _at(2026, 10, 7, 9, 0))["next_change"].endswith("T20:00")
    # Friday evening rolls to Monday morning, not Saturday.
    nxt = rw.schedule_state(cfg, _at(2026, 10, 9, 21, 0))["next_change"]
    assert nxt.startswith("2026-10-12T08:00")


def test_an_unknown_timezone_falls_back_to_local_not_utc():
    """A typo in the zone must not silently shift the window by hours."""
    rw = _rw()
    assert rw._schedule_zone("Europe/Nowhere") is None
    assert rw._schedule_zone("") is None


# --------------------------------------------------------------- the floor
def test_effective_min_pods_drops_to_zero_outside_the_window(monkeypatch):
    rw = _rw()

    class _Pool:
        mcfg = rw.parse_model_runpod({"min_pods": 2, "schedule_enabled": True,
                                      "schedule_start": "08:00",
                                      "schedule_end": "20:00"})
        effective_min_pods = rw.RunpodPodPool.effective_min_pods

    pool = _Pool()
    monkeypatch.setattr(rw, "schedule_state", lambda cfg, now=None: {"in_window": True})
    assert pool.effective_min_pods() == 2
    monkeypatch.setattr(rw, "schedule_state", lambda cfg, now=None: {"in_window": False})
    assert pool.effective_min_pods() == 0


def test_the_schedule_never_blocks_a_request():
    """Outside the window the floor is 0 — which is the ordinary cold-start path,
    not a refusal. ensure_ready still provisions on demand."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    body = src.split("def effective_min_pods")[1].split("def maintain")[0]
    assert "return 0" not in body.replace("else 0", "")
    # ensure_ready's floor is max(min_pods, 1): it never consults the schedule.
    assert "want = max(self.mcfg.min_pods, 1)" in src


# --------------------------------------------------------------- region
def test_region_is_normalised_and_per_model():
    rw = _rw()
    assert rw.parse_model_runpod({"data_center": " eu-ro-1 "}).data_center == "EU-RO-1"
    assert rw.parse_model_runpod({}).data_center == ""


def test_volume_region_still_wins_over_the_model_setting():
    """A pod in another region cannot attach the volume, so the volume decides."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    assert 'dc = (self.mcfg.data_center' in src
    assert 'data_center_id=(getattr(self, "_volume_dc", "") or dc)' in src


# --------------------------------------------------- orchestration: admin API
def test_new_keys_survive_the_model_configure_whitelist():
    """The entry is rebuilt from scratch, so a key missing here is dropped."""
    src = (ROOT / "codai/admin/routes.py").read_text()
    block = src.split("# Per-model runpod block")[1].split("# Per-model `host` block")[0]
    for key in ("data_center", "schedule_start", "schedule_end", "schedule_tz",
                "schedule_enabled", "schedule_days"):
        assert f'"{key}"' in block, key


def test_schedule_status_is_reported_for_monitoring():
    worker = (ROOT / "codai/api/runpod_worker.py").read_text()
    assert "def pools_schedule_status" in worker
    for field in ("in_window", "next_change", "effective_min_pods", "healthy_pods"):
        assert field in worker.split("def pools_schedule_status")[1].split("def pods_status")[0]
    routes = (ROOT / "codai/admin/routes.py").read_text()
    assert '"schedules": schedules' in routes


# ------------------------------------------------------ orchestration: the UI
@pytest.mark.parametrize("field", [
    "cfg-rp-schedule-enabled", "cfg-rp-schedule-start", "cfg-rp-schedule-end",
    "cfg-rp-schedule-tz", "cfg-rp-schedule-days", "cfg-rp-data-center",
])
def test_every_new_setting_is_on_the_model_page(field):
    """Markup, loader and serialiser — a field missing from any one of the three
    is a setting that silently will not stick."""
    html = (ROOT / "codai/admin/templates/models.html").read_text()
    assert html.count(field) >= 3, f"{field}: only {html.count(field)} occurrences"


def test_the_serialiser_sends_the_new_keys():
    html = (ROOT / "codai/admin/templates/models.html").read_text()
    body = html.split("sticky_sessions:")[-1][:1200]
    for key in ("schedule_enabled:", "schedule_start:", "schedule_end:",
                "schedule_tz:", "schedule_days:", "data_center:"):
        assert key in body, key


# --------------------------------------------------------------- spend API
def test_spend_endpoint_requires_an_admin_scoped_key():
    src = (ROOT / "codai/api/app.py").read_text()
    assert '@app.get("/v1/runpod/spend"' in src
    body = src.split('@app.get("/v1/runpod/spend"')[1].split("@app.get")[0]
    assert "_admin_scoped_token(request)" in body
    assert "status_code=403" in body


def test_a_plain_token_is_not_admin_scoped(tmp_path, monkeypatch):
    from codai.admin.auth import SessionManager
    sm = SessionManager.__new__(SessionManager)
    data = {"users": [], "sessions": {}, "tokens": [
        {"id": 1, "name": "plain", "token": "sk-plain", "provider": "openai"},
        {"id": 2, "name": "privileged", "token": "sk-adm", "provider": "openai",
         "admin": True},
    ]}
    monkeypatch.setattr(sm, "_load_auth_data", lambda: data, raising=False)
    assert sm.token_is_admin("sk-adm") is True
    assert sm.token_is_admin("sk-plain") is False
    assert sm.token_is_admin("") is False
    assert sm.token_is_admin("sk-nope") is False
    # An ordinary key must still authenticate for the normal endpoints.
    assert sm.verify_token("sk-plain") is True


def test_admin_scope_is_opt_in_at_creation():
    src = (ROOT / "codai/frontproxy/admin_data.py").read_text()
    assert 'is_admin = bool(data.get("admin"))' in src
    assert '"admin": is_admin,' in src
    assert '@app.patch("/admin/api/tokens/{token_id}"' in src


def test_the_admin_scope_is_settable_from_the_tokens_page():
    """The API supports the flag; the page must actually expose it, or granting
    it means hand-editing auth.json."""
    html = (ROOT / "codai/admin/templates/tokens.html").read_text()
    assert 'id="t-admin"' in html                      # opt-in at creation
    assert "admin: document.getElementById('t-admin').checked" in html
    assert "function setTokenAdmin" in html            # toggle on an existing one
    assert "method:'PATCH'" in html
    assert "${t.admin ? 'checked' : ''}" in html       # reflects current state
