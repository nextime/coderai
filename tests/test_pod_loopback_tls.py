# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""A pod that terminates TLS must be able to reach ITSELF.

A pod on the direct path serves HTTPS on its only port. An OCR engine running
there is a SUBPROCESS: it attaches to the gateway by URL and goes through the
socket like any other client. It was handed `http://127.0.0.1:8000`, probed a
plain request at a TLS listener, and reported

    SURYA_INFERENCE_URL=http://127.0.0.1:8000/v1/ocr/attach/<lease>/v1
    is not reachable at /health. Start the server or unset the variable.

while the pod was answering that very address perfectly well over TLS.

The fix keeps verification ON end to end, which is the part worth pinning:
a name the certificate actually carries, resolved to loopback, and a trust
store holding the public roots AND our CA. No skipped hostname check and no
disabled verify anywhere on that path -- the orchestrator needs those to
reach a pod by a bare IP, but a pod reaching itself has no such excuse.
"""

import os
import pathlib
import re
import subprocess
import tempfile

import pytest

from codai.api import pod_tls
from codai.ocr import gateway


BOOT = pathlib.Path("packaging/runpod/boot.sh")


# ------------------------------------------------------- the certificate #
def test_the_cert_names_loopback_so_nothing_has_to_skip_the_check():
    leaf = pod_tls.issue_pod_cert("some-pod")
    with tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False) as f:
        f.write(leaf["cert"])
        path = f.name
    try:
        text = subprocess.run(["openssl", "x509", "-in", path, "-noout", "-text"],
                              capture_output=True, text=True).stdout
    finally:
        os.unlink(path)
    sans = " ".join(l for l in text.splitlines() if "DNS:" in l or "IP Address" in l)
    assert "DNS:coderai-pod" in sans
    assert "DNS:localhost" in sans
    assert "IP Address:127.0.0.1" in sans


def test_the_ca_travels_with_the_leaf():
    """Without it the pod holds a certificate it cannot verify."""
    env = pod_tls.pod_env("some-pod")
    assert pod_tls.ENV_CA in env
    assert env[pod_tls.ENV_CA].startswith("-----BEGIN")


def test_the_private_key_is_still_the_only_secret_sent():
    env = pod_tls.pod_env("some-pod")
    assert set(env) == {pod_tls.ENV_CERT, pod_tls.ENV_KEY, pod_tls.ENV_CA}
    # The CA's KEY must never be in there.
    assert "ca.key" not in " ".join(env.values())


# -------------------------------------------------------------- boot.sh #
def test_boot_sets_a_self_base_for_both_schemes():
    src = BOOT.read_text()
    assert 'export CODERAI_SELF_BASE="https://coderai-pod:8000"' in src
    assert 'export CODERAI_SELF_BASE="http://127.0.0.1:8000"' in src


def test_boot_maps_the_cert_name_to_loopback():
    assert "127.0.0.1 coderai-pod" in BOOT.read_text()


def test_boot_keeps_the_public_roots_in_the_bundle():
    """A bundle that lost them would break every model download on the pod --
    a worse failure than the one being fixed."""
    src = BOOT.read_text()
    assert "certifi" in src
    # ours is APPENDED to the public roots, not the other way round
    assert re.search(r'cat "\$BASE_CA" /run/coderai-tls/ca\.pem', src)


def test_boot_only_exports_a_bundle_it_actually_built():
    src = BOOT.read_text()
    block = src.split("BASE_CA=")[1].split("else\n    # Plain HTTP")[0]
    assert "-s /run/coderai-tls/bundle.pem" in block
    assert "export SSL_CERT_FILE" in block
    assert "export REQUESTS_CA_BUNDLE" in block


def test_boot_says_so_when_it_cannot():
    """Silence here is how the last two defects stayed hidden for days."""
    src = BOOT.read_text()
    assert src.count("cannot verify loopback") >= 2


def test_boot_is_valid_shell():
    assert subprocess.run(["bash", "-n", str(BOOT)]).returncode == 0


# ------------------------------------------------------------- the URL #
def test_self_base_follows_the_environment(monkeypatch):
    monkeypatch.setenv("CODERAI_SELF_BASE", "https://coderai-pod:8000")
    assert gateway.self_base() == "https://coderai-pod:8000"


def test_self_base_falls_back_to_loopback_http(monkeypatch):
    monkeypatch.delenv("CODERAI_SELF_BASE", raising=False)
    monkeypatch.delenv("CODERAI_PORT", raising=False)
    assert gateway.self_base() == "http://127.0.0.1:8000"


def test_the_engine_url_is_built_from_it(monkeypatch):
    monkeypatch.setenv("CODERAI_SELF_BASE", "https://coderai-pod:8000")
    url = gateway.engine_url("LEASE")
    assert url == "https://coderai-pod:8000/v1/ocr/attach/LEASE/v1"
    # surya derives /health by stripping the trailing /v1 -- see gateway.PREFIX
    assert url.endswith("/v1")
