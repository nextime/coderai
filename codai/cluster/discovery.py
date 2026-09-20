# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Finding the other coderai installs on the LAN — mDNS / DNS-SD.

Every install with ``cluster.discovery`` on announces itself as a
``_coderai._tcp.local.`` service (name, version, URL scheme, port, the
capabilities its engines offer) and browses for the others. Membership is
a **shared cluster token** (``cluster.token``): the announcement carries an
HMAC fingerprint of it — never the token — and only peers whose fingerprint
matches ours count as members. A head with ``cluster.auto_join`` on turns
every member it sees into a cluster node (``cluster.nodes`` entry in
memory, the shared token as its API key); a node accepts the shared token
on its ``/cluster/*`` endpoints. So one token typed on every box is the
whole setup: no URLs, no per-node API tokens.

mDNS is link-local multicast (224.0.0.251:5353): it crosses no router, and
a container sees it only with ``--network host`` (the launcher's default)
or a macvlan. Across subnets or VPNs, list the nodes by hand — both ways
coexist; a node named in ``cluster.nodes`` is never duplicated by discovery.

Needs the ``zeroconf`` package; without it discovery is simply off and the
Cluster page says so.
"""

import hmac
import hashlib
import socket
import threading
import time
import uuid
from typing import List, Optional

SERVICE = "_coderai._tcp.local."
_FP_MSG = b"coderai-cluster-membership-v1"


def token_fingerprint(token: str) -> str:
    """What an announcement carries instead of the token: HMAC-SHA256 keyed by
    the token over a fixed message, truncated. Proves 'I hold the same token'
    to anyone who also holds it, reveals nothing to anyone who doesn't."""
    if not token:
        return ""
    return hmac.new(token.encode("utf-8"), _FP_MSG, hashlib.sha256).hexdigest()[:24]


def available() -> bool:
    try:
        import zeroconf  # noqa: F401
        return True
    except Exception:
        return False


def _txt(props: dict) -> dict:
    return {str(k): str(v) for k, v in props.items() if v is not None}


class Discovery:
    """Announce this install and keep the list of the others seen.

    ``peers()`` is every coderai announcing on the link; ``members()`` is the
    subset holding our token (and never ourselves)."""

    def __init__(self, name: str, port: int, token: str = "", advertise_host: str = "",
                 scheme: str = "http", version: str = "", capabilities=None,
                 serve: bool = True, browse: bool = True, instance_id: str = ""):
        self.name = (name or socket.gethostname()).strip()
        self.port = int(port)
        self.token = token or ""
        self.advertise_host = advertise_host or ""
        self.scheme = scheme or "http"
        self.version = version or ""
        self.capabilities = capabilities      # callable or list; read at announce
        self.serve = bool(serve)
        self.browse = bool(browse)
        self.instance_id = instance_id or uuid.uuid4().hex[:12]
        self._zc = None
        self._info = None
        self._browser = None
        self._peers: dict = {}                # instance_id -> dict
        self._lock = threading.Lock()
        self.error = ""
        self.started_at = 0.0

    # ----------------------------------------------------------- lifecycle
    def _address(self) -> str:
        from codai.cluster.rpc import advertise_host
        return advertise_host(self.advertise_host)

    def _caps(self) -> List[str]:
        c = self.capabilities
        try:
            c = c() if callable(c) else c
        except Exception:
            c = []
        return sorted({str(x) for x in (c or [])})

    def _service_info(self):
        from zeroconf import ServiceInfo
        addr = self._address()
        props = _txt({
            "name": self.name, "id": self.instance_id, "ver": self.version,
            "scheme": self.scheme, "fp": token_fingerprint(self.token),
            "caps": ",".join(self._caps()), "serve": "1" if self.serve else "0",
            "path": "/cluster/state",
        })
        # The instance name must be unique on the link; the coderai node name
        # plus a short instance id keeps two boxes with the same hostname apart.
        inst = f"{self.name}-{self.instance_id}.{SERVICE}"
        return ServiceInfo(SERVICE, inst, addresses=[socket.inet_aton(addr)],
                           port=self.port, properties=props, server=f"{self.name}.local.")

    def start(self) -> bool:
        try:
            from zeroconf import Zeroconf, ServiceBrowser
        except Exception as exc:
            self.error = f"zeroconf not installed ({exc.__class__.__name__}); discovery off"
            print(f"[discovery] {self.error}", flush=True)
            return False
        try:
            self._zc = Zeroconf()
            self._info = self._service_info()
            self._zc.register_service(self._info)
            if self.browse:
                self._browser = ServiceBrowser(self._zc, SERVICE, handlers=[self._on_change])
            self.started_at = time.time()
            print(f"[discovery] announcing '{self.name}' as {SERVICE} on "
                  f"{self.scheme}://{self._address()}:{self.port}"
                  f"{' (token set)' if self.token else ' (NO cluster token: visible, joins nothing)'}",
                  flush=True)
            return True
        except Exception as exc:
            self.error = f"mDNS not started: {exc}"
            print(f"[discovery] {self.error}", flush=True)
            self.stop()
            return False

    def refresh(self) -> None:
        """Re-announce (capabilities changed, e.g. an engine came up)."""
        if self._zc is None or self._info is None:
            return
        try:
            new = self._service_info()
            self._zc.update_service(new)
            self._info = new
        except Exception as exc:
            print(f"[discovery] re-announce failed: {exc}", flush=True)

    def stop(self) -> None:
        zc, self._zc = self._zc, None
        if zc is None:
            return
        try:
            if self._info is not None:
                zc.unregister_service(self._info)
        except Exception:
            pass
        try:
            zc.close()
        except Exception:
            pass

    # ------------------------------------------------------------- browsing
    def _on_change(self, zeroconf, service_type, name, state_change):
        try:
            from zeroconf import ServiceStateChange
            if state_change == ServiceStateChange.Removed:
                with self._lock:
                    for k, p in list(self._peers.items()):
                        if p.get("service") == name:
                            self._peers.pop(k, None)
                return
            info = zeroconf.get_service_info(service_type, name, timeout=1500)
            if info is None:
                return
            self._record(name, info)
        except Exception as exc:
            print(f"[discovery] browse event error: {exc}", flush=True)

    def _record(self, service_name: str, info) -> None:
        props = {}
        for k, v in (info.properties or {}).items():
            try:
                props[k.decode() if isinstance(k, bytes) else str(k)] = \
                    v.decode() if isinstance(v, bytes) else ("" if v is None else str(v))
            except Exception:
                continue
        iid = props.get("id") or service_name
        addrs = []
        try:
            addrs = info.parsed_addresses()
        except Exception:
            try:
                addrs = [socket.inet_ntoa(a) for a in info.addresses]
            except Exception:
                addrs = []
        # Prefer an IPv4 address; mDNS may carry both.
        addr = next((a for a in addrs if ":" not in a), addrs[0] if addrs else "")
        if not addr:
            return
        scheme = props.get("scheme") or "http"
        peer = {
            "instance": iid, "service": service_name,
            "name": props.get("name") or service_name.split(".")[0],
            "url": f"{scheme}://{addr}:{info.port}",
            "version": props.get("ver", ""), "fingerprint": props.get("fp", ""),
            "capabilities": [c for c in (props.get("caps") or "").split(",") if c],
            "serve": props.get("serve", "1") != "0",
            "seen": time.time(),
        }
        with self._lock:
            self._peers[iid] = peer

    def peers(self, max_age_s: float = 0.0) -> List[dict]:
        """Everyone announcing (ourselves excluded)."""
        now = time.time()
        with self._lock:
            out = [dict(p) for k, p in self._peers.items() if k != self.instance_id]
        if max_age_s > 0:
            out = [p for p in out if now - p["seen"] <= max_age_s]
        for p in out:
            p["member"] = bool(self.token) and p.get("fingerprint") == token_fingerprint(self.token)
        return sorted(out, key=lambda p: p["name"].lower())

    def members(self) -> List[dict]:
        """Peers holding our cluster token, that answer as a node."""
        return [p for p in self.peers() if p["member"] and p.get("serve", True)]

    def status(self) -> dict:
        return {"available": available(), "running": self._zc is not None,
                "error": self.error, "name": self.name, "instance": self.instance_id,
                "token_set": bool(self.token), "port": self.port,
                "peers": self.peers()}


def member_node_specs(members: List[dict], token: str, known_names, timeout_s: float = 4.0):
    """Discovered members as NodeSpecs for the supervisor, skipping any name
    (or URL) that ``cluster.nodes`` already declares by hand."""
    from codai.cluster.nodes import NodeSpec
    known = {str(n).lower() for n in (known_names or [])}
    out = []
    for p in members:
        name = p["name"]
        if name.lower() in known or p["url"].lower() in known:
            continue
        # Two boxes announcing the same name: disambiguate with the instance id
        # so both get an engine rather than one silently shadowing the other.
        if any(o.name.lower() == name.lower() for o in out):
            name = f"{name}-{p['instance'][:6]}"
        out.append(NodeSpec(name=name, url=p["url"], api_key=token,
                            verify="off" if p["url"].startswith("https") else "system",
                            timeout_s=timeout_s))
    return out
