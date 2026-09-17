# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""TLS for pods reached directly, so `direct_tcp` is not a choice between
"no 100 s cutoff" and "the token travels in clear".

RunPod's HTTP proxy terminates TLS itself and expects plain HTTP on the
container port, so TLS on the pod only makes sense on the direct TCP path.
There, nothing sits between this machine and the pod's public ip:port — so
the pod has to bring its own certificate, and this side has to know which
certificate to trust before the pod exists (the same problem as the bearer
token, solved the same way: decided here, sent as pod environment).

One CA per install, made once with the local openssl and kept beside the pod
registry. Each pod gets a leaf certificate signed by it at provision time,
sent as CODERAI_TLS_CERT / CODERAI_TLS_KEY; boot.sh writes them out and starts
uvicorn on them. This side verifies pod connections against the CA only —
no hostname check, because the pod's public IP is not known until after it
boots, and it adds nothing anyway: a certificate this CA did not sign fails
before a name is ever compared. See codai/api/pod_http.py for the client.
"""

import os
import subprocess
import tempfile
import threading

_lock = threading.Lock()

#: Environment variables the pod reads (boot.sh). PEM text, not paths.
ENV_CERT = "CODERAI_TLS_CERT"
ENV_KEY = "CODERAI_TLS_KEY"


def tls_dir() -> str:
    """Where the CA lives: the config dir, beside runpod-pods.json."""
    try:
        from codai.api.runpod_worker import _pod_registry_path
        base = os.path.dirname(_pod_registry_path())
    except Exception:
        base = os.path.expanduser("~/.coderai")
    return os.path.join(base, "runpod", "tls")


def ca_path() -> str:
    return os.path.join(tls_dir(), "ca.pem")


def _run(args, cwd=None) -> None:
    r = subprocess.run(["openssl", *args], cwd=cwd, capture_output=True, text=True,
                       timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"openssl {args[0]} failed: {(r.stderr or r.stdout).strip()[:300]}")


def ensure_ca() -> str:
    """Create the per-install CA if missing; return the path of ca.pem."""
    d = tls_dir()
    ca, key = os.path.join(d, "ca.pem"), os.path.join(d, "ca.key")
    with _lock:
        if os.path.isfile(ca) and os.path.isfile(key):
            return ca
        os.makedirs(d, exist_ok=True)
        os.chmod(d, 0o700)
        _run(["req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
              "-nodes", "-days", "3650", "-subj", "/CN=coderai pod CA",
              "-addext", "basicConstraints=critical,CA:TRUE",
              "-addext", "keyUsage=critical,keyCertSign,cRLSign",
              "-keyout", key, "-out", ca])
        os.chmod(key, 0o600)
        print(f"[runpod] created the pod TLS CA at {ca}", flush=True)
        return ca


def issue_pod_cert(pod_name: str, days: int = 30) -> dict:
    """A leaf certificate for one pod, signed by the CA: {cert, key} as PEM.

    Short-lived by design — a pod lives hours, and a leaked key from a pod
    that has since been reaped should not stay useful. The SAN is a fixed
    name, never checked (see the module docstring); it exists because some
    TLS stacks refuse a certificate with no SAN at all.
    """
    ca = ensure_ca()
    ca_key = os.path.join(tls_dir(), "ca.key")
    name = "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in pod_name)[:60] or "pod"
    with tempfile.TemporaryDirectory(prefix="coderai-pod-tls-") as tmp:
        key, csr, crt = (os.path.join(tmp, n) for n in ("pod.key", "pod.csr", "pod.pem"))
        ext = os.path.join(tmp, "ext.cnf")
        with open(ext, "w") as f:
            f.write("subjectAltName=DNS:coderai-pod\n"
                    "extendedKeyUsage=serverAuth\n"
                    "basicConstraints=CA:FALSE\n")
        _run(["req", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
              "-nodes", "-subj", f"/CN={name}", "-keyout", key, "-out", csr])
        _run(["x509", "-req", "-in", csr, "-CA", ca, "-CAkey", ca_key, "-CAcreateserial",
              "-days", str(int(days)), "-extfile", ext, "-out", crt])
        with open(crt) as f:
            cert_pem = f.read()
        with open(key) as f:
            key_pem = f.read()
    return {"cert": cert_pem, "key": key_pem}


def pod_env(pod_name: str) -> dict:
    """The environment that turns TLS on in a coderai pod."""
    leaf = issue_pod_cert(pod_name)
    return {ENV_CERT: leaf["cert"], ENV_KEY: leaf["key"]}
