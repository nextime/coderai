"""The hourly version probe must not consume a podman lock per run.

``coderai:latest`` declares ``VOLUME /cache /config /models``. An ordinary
``podman run`` of it therefore creates three ANONYMOUS volumes, and ``--rm``
does not remove volumes -- that is a docker behaviour podman does not share.
podman allocates one lock per volume out of a fixed pool (``num_locks``,
default 2048) and reclaims it only when the volume is removed.

So the autoupgrade's once-an-hour ``grep the version out of the image`` probe
leaked three locks an hour. On 2026-11-05, 99 days and 1858 orphaned volumes
later, the pool ran dry on a production orchestrator and every ``podman run``
on the host that would create a volume began failing -- not just coderai's:
gitlab, nexus and the rest with it. The probe only ever wanted to read one file
out of the image, so it asks for the filesystem without the volumes.

And when a probe does fail, the script has to say what the engine said. For a
day this failure reported "wrong tag?", which is the one thing it was not.
"""

import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "packaging/linux/launcher/coderai-autoupgrade"


def _stub_engine(tmp_path, name, stdout="", stderr="", rc=0):
    """A fake container engine that records its argv and answers as told."""
    argv_log = tmp_path / f"{name}.argv"
    exe = tmp_path / name
    exe.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> "{argv_log}"\n'
        + (f'printf %s {stdout!r} \n' if stdout else "")
        + (f'printf %s {stderr!r} >&2\n' if stderr else "")
        + f"exit {rc}\n"
    )
    exe.chmod(0o755)
    return argv_log


def _run(tmp_path, engine, **stub):
    runner = tmp_path / "coderai-docker"
    runner.write_text("#!/bin/sh\nexit 0\n")
    runner.chmod(0o755)
    argv_log = _stub_engine(tmp_path, engine, **stub)
    env = dict(os.environ)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["CODERAI_AUTOUPGRADE_CONF"] = str(tmp_path / "no-such.conf")
    env["STATE_DIR"] = str(tmp_path)
    env["ENGINE"] = engine
    env["RUNNER"] = str(runner)
    r = subprocess.run([str(SCRIPT)], capture_output=True, text=True,
                       env=env, timeout=120)
    return r, (argv_log.read_text() if argv_log.exists() else "")


# ------------------------------------------------------------- the leak #
def test_the_podman_probe_ignores_the_images_volumes(tmp_path):
    _r, argv = _run(tmp_path, "podman", stderr="boom\n", rc=1)
    probe = [ln for ln in argv.splitlines() if "--entrypoint grep" in ln]
    assert probe, f"the probe never ran: {argv!r}"
    assert "--image-volume=ignore" in probe[0], probe[0]


def test_the_probe_still_reads_the_version_file(tmp_path):
    """Ignoring the volumes must not change WHAT is read."""
    _r, argv = _run(tmp_path, "podman", stderr="boom\n", rc=1)
    probe = [ln for ln in argv.splitlines() if "--entrypoint grep" in ln][0]
    assert "/opt/coderai/app/codai/__init__.py" in probe
    assert "__version__" in probe
    assert "--rm" in probe


def test_docker_is_not_given_a_podman_only_flag(tmp_path):
    """docker has no --image-volume and would refuse the whole invocation --
    and it has no leak either, since its --rm does take anonymous volumes."""
    _r, argv = _run(tmp_path, "docker", stderr="boom\n", rc=1)
    probe = [ln for ln in argv.splitlines() if "--entrypoint grep" in ln]
    assert probe, f"the probe never ran: {argv!r}"
    assert "--image-volume" not in probe[0], probe[0]


# -------------------------------------------------- saying what went wrong #
def test_a_failed_probe_relays_what_the_engine_said(tmp_path):
    """The operator needs the cause, not a guess at it."""
    msg = "Error: allocating lock for new volume: exceeded num_locks (2048)"
    r, _argv = _run(tmp_path, "podman", stderr=msg + "\n", rc=125)
    out = r.stdout + r.stderr
    assert "cannot read the version" in out, out[-800:]
    assert "num_locks" in out, out[-800:]


def test_a_silent_failure_still_suggests_the_tag_and_engine(tmp_path):
    """With nothing on stderr there is nothing to relay, so the old guess is
    still the best available -- but only then."""
    r, _argv = _run(tmp_path, "podman", rc=1)
    out = r.stdout + r.stderr
    assert "cannot read the version" in out, out[-800:]
    assert "wrong tag?" in out, out[-800:]


def test_the_probes_stderr_does_not_reach_the_operator_on_success(tmp_path):
    """A successful probe says nothing: the file is a diagnostic, not output."""
    r, _argv = _run(tmp_path, "podman", stdout='__version__ = "9.9.9"\n')
    out = r.stdout + r.stderr
    assert "cannot read the version" not in out, out[-800:]
