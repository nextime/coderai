"""coderai-ocr-workers: N processes that forward /v1/ocr pages, behind nginx.

One Python process forwards ~16-23 pages/s however many pods it has (one core
at 100%, the rest of the box idle, pod GPUs mostly idle); 128 pods need ~85.
So nginx sends exact /v1/ocr and /v1/ocr/batch to N worker processes and
everything else to the front, and the main engine keeps owning the fleet.

What must hold:
  * off by default, and "off" is nginx routing exactly as before;
  * a dead worker costs no page: retried on another only when the request never
    reached it (never on a timeout -- that would read the page twice, on two
    pods, billed twice), and the front is the backup when every worker is down;
  * a worker never demands the front's internal token (nginx cannot send it),
    registers no models, and never writes the shared config;
  * N workers together admit what the config says, not N times it.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai import ocr_fleet_workers as fw                           # noqa: E402


# ------------------------------------------------------------------ routing #

def test_off_writes_nothing_for_nginx_to_route():
    assert fw.render([]) == ("", "")


def test_the_upstream_balances_on_shared_counts_with_the_front_as_backup():
    up, _ = fw.render([18801, 18800])
    assert "least_conn;" in up
    assert "zone coderai_ocr" in up        # one set of counts for all nginx workers
    assert up.index("18800") < up.index("18801")
    assert f"server {fw.FRONT} backup;" in up
    assert "keepalive" in up


def test_only_the_two_page_paths_go_to_the_workers():
    _, loc = fw.render([18800])
    assert "location = /v1/ocr {" in loc
    assert "location = /v1/ocr/batch {" in loc
    assert loc.count("location") == 2      # status, admin, models stay on the front


def test_a_page_is_retried_only_when_it_never_reached_a_worker():
    _, loc = fw.render([18800, 18801])
    line = [ln for ln in loc.splitlines() if "proxy_next_upstream " in ln][0]
    assert line.strip() == "proxy_next_upstream error;"
    assert "timeout" not in line


def test_no_try_limit_starves_the_backup():
    """Measured: with `proxy_next_upstream_tries 3` and three dead workers the
    tries ran out before the front and the client got a 502."""
    _, loc = fw.render([18800, 18801, 18802])
    assert "proxy_next_upstream_tries" not in loc


def test_a_dead_worker_is_marked_down_on_its_first_failure():
    up, _ = fw.render([18800])
    assert "max_fails=1" in up


# --------------------------------------------------------------- the worker #

def test_a_worker_knows_its_slot_and_registers_no_models(monkeypatch):
    monkeypatch.setenv("CODERAI_INTERNAL_TOKEN", "front-secret")
    monkeypatch.setenv("CODERAI_ENGINE_PRIMARY", "1")
    env = fw.worker_env(2, 4)
    assert env["CODERAI_OCR_WORKER"] == "2/4"
    assert env["CODERAI_ENGINE_NAME"] == "ocrw2"
    assert env["CODERAI_ENGINE_MODELS"] == "[]"
    # nginx cannot present the front's token; the worker checks the bearer.
    assert "CODERAI_INTERNAL_TOKEN" not in env
    # Only the primary engine talks to the broker.
    assert "CODERAI_ENGINE_PRIMARY" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_the_worker_command_is_an_engine_on_its_own_port():
    cmd = fw.worker_cmd(3)
    assert "--engine-only" in cmd
    assert cmd[cmd.index("--internal-port") + 1] == str(fw.PORT_BASE + 3)


@pytest.mark.parametrize("env,cfg,want", [
    ("", None, 0),
    ("", {"ocr": {"workers": 6}}, 6),
    ("3", {"ocr": {"workers": 6}}, 3),          # the env wins
    ("999", None, fw.MAX_WORKERS),
    ("-2", None, 0),
    ("six", None, 0),
])
def test_how_many_workers(monkeypatch, tmp_path, env, cfg, want):
    monkeypatch.setenv("CODERAI_CONFIG_DIR", str(tmp_path))
    if cfg is not None:
        (tmp_path / "coderai").mkdir()
        (tmp_path / "coderai" / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("CODERAI_OCR_WORKERS", env)
    assert fw.wanted_workers() == want


# ------------------------------------------------- the worker's config share #

def _cfg_manager(monkeypatch, tmp_path, slot, ocr):
    from codai import config as cm
    (tmp_path / "config.json").write_text(json.dumps({"ocr": ocr}))
    if slot:
        monkeypatch.setenv("CODERAI_OCR_WORKER", slot)
    else:
        monkeypatch.delenv("CODERAI_OCR_WORKER", raising=False)
    m = cm.ConfigManager(str(tmp_path))
    m.load()
    return m


def test_n_workers_together_admit_what_the_config_says(monkeypatch, tmp_path):
    m = _cfg_manager(monkeypatch, tmp_path, "1/4",
                     {"max_concurrency": 16384, "max_queued_pages": 4096,
                      "surya_instances": 8, "workers": 4})
    assert m.config.ocr.max_concurrency == 4096
    assert m.config.ocr.max_queued_pages == 1024
    assert m.config.ocr.surya_instances == 2


def test_a_share_is_rounded_up_and_never_zero(monkeypatch, tmp_path):
    m = _cfg_manager(monkeypatch, tmp_path, "0/3",
                     {"max_queued_pages": 10, "surya_instances": 1})
    assert m.config.ocr.max_queued_pages == 4
    assert m.config.ocr.surya_instances == 1


def test_zero_still_means_unset(monkeypatch, tmp_path):
    m = _cfg_manager(monkeypatch, tmp_path, "0/3", {"max_queued_pages": 0})
    assert m.config.ocr.max_queued_pages == 0


def test_the_main_engine_keeps_the_configured_values(monkeypatch, tmp_path):
    m = _cfg_manager(monkeypatch, tmp_path, None, {"max_queued_pages": 4096})
    assert m.config.ocr.max_queued_pages == 4096


def test_a_worker_never_writes_its_share_back(monkeypatch, tmp_path):
    """Writing 1/N back would shrink the real config by N on every save."""
    m = _cfg_manager(monkeypatch, tmp_path, "0/4", {"max_queued_pages": 4096})
    before = (tmp_path / "config.json").read_text()
    m.save_config()
    m.save_models()
    assert (tmp_path / "config.json").read_text() == before


def test_the_setting_is_saved_by_the_main_process(monkeypatch, tmp_path):
    m = _cfg_manager(monkeypatch, tmp_path, None, {"workers": 6})
    assert m.config.ocr.workers == 6
    m.save_config()
    assert json.loads((tmp_path / "config.json").read_text())["ocr"]["workers"] == 6


# --------------------------------------------------------------- packaging #

def test_nginx_includes_the_files_and_can_hold_a_128_pod_fleet():
    conf = (ROOT / "packaging/linux/nginx.conf").read_text()
    assert "include /tmp/coderai-nginx/ocr-upstream*.conf;" in conf
    assert "include /tmp/coderai-nginx/ocr-locations*.conf;" in conf
    # The locations must be inside the server block, before `location /`.
    assert conf.index("ocr-locations") < conf.index("location / {")
    assert "worker_connections  16384;" in conf


def test_supervisord_runs_the_launcher():
    conf = (ROOT / "packaging/linux/supervisord.conf").read_text()
    blk = conf.split("[program:ocr_workers]")[1].split("[program:")[0]
    assert "codai/ocr_fleet_workers.py" in blk
    assert "autostart=true" in blk
    # Retiring workers may finish their pages before supervisord kills them.
    assert int(blk.split("stopwaitsecs=")[1].split()[0]) > fw.DRAIN_S


def test_the_launcher_imports_nothing_but_the_stdlib():
    """It runs beside the server and must not cost what it is there to save."""
    src = (ROOT / "codai/ocr_fleet_workers.py").read_text()
    imports = {ln.split()[1].split(".")[0] for ln in src.splitlines()
               if ln.startswith(("import ", "from "))}
    assert imports <= {"json", "os", "signal", "subprocess", "sys", "time", "urllib"}


@pytest.mark.skipif(not shutil.which("nginx") and not os.path.exists("/usr/sbin/nginx"),
                    reason="nginx not installed")
def test_real_nginx_accepts_the_rendered_files(tmp_path):
    nginx = shutil.which("nginx") or "/usr/sbin/nginx"
    inc = tmp_path / "inc"
    inc.mkdir()
    conf = (ROOT / "packaging/linux/nginx.conf").read_text()
    conf = (conf.replace("/tmp/coderai-nginx", str(inc))
                .replace("pid /tmp/nginx.pid;", f"pid {tmp_path}/nginx.pid;")
                .replace("/tmp/nginx-", f"{tmp_path}/nginx-")
                .replace("include       /etc/nginx/mime.types;", "")
                .replace("error_log /dev/stderr crit;", f"error_log {tmp_path}/e.log;"))
    (tmp_path / "nginx.conf").write_text(conf)
    up, loc = fw.render([18800, 18801])
    for files in (("", ""), (up, loc)):
        (inc / fw.UPSTREAM_FILE).write_text(files[0])
        (inc / fw.LOCATION_FILE).write_text(files[1])
        r = subprocess.run([nginx, "-t", "-q", "-c", str(tmp_path / "nginx.conf"),
                            "-p", str(tmp_path)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
