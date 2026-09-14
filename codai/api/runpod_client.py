# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Thin RunPod API client (GPU catalog + Pod lifecycle).

coderai talks to RunPod over its GraphQL API (https://graphql-spec.runpod.io) — the most
stable surface for GPU pricing and pod create/query/stop/terminate. Serverless invocation
uses the separate ``/v2/{endpoint}`` REST base and is handled in the worker, not here.

This module is pure transport: no scaling/policy logic (that lives in
:mod:`codai.api.runpod_worker`). The API key comes from the caller (RunpodConfig) and is
NEVER logged — only masked in error strings.
"""

import time
from typing import Optional


class RunpodError(RuntimeError):
    """A RunPod API call failed (network, auth, or GraphQL error)."""


def _mask(key: str) -> str:
    if not key:
        return "<none>"
    return (key[:4] + "…" + key[-4:]) if len(key) > 10 else "****"


# A pod's public HTTP proxy hostname is derived from its id + the container port.
def pod_proxy_url(pod_id: str, port: int) -> str:
    return f"https://{pod_id}-{int(port)}.proxy.runpod.net"


# Deep link to the pod in the RunPod web console (container + vLLM logs live there).
def pod_console_url(pod_id: str) -> str:
    return f"https://www.runpod.io/console/pods/{pod_id}"


class RunpodClient:
    def __init__(self, cfg):
        self._cfg = cfg
        self._api_key = (getattr(cfg, "api_key", "") or "").strip()
        self._gql_base = (getattr(cfg, "api_base", "") or "https://api.runpod.io/graphql").strip()
        if not self._api_key:
            raise RunpodError("RunPod API key not configured (config.runpod.api_key).")

    # ------------------------------------------------------------------ #
    # transport
    # ------------------------------------------------------------------ #
    def _gql(self, query: str, variables: Optional[dict] = None,
             timeout: float = 30.0) -> dict:
        import requests
        # RunPod authenticates GraphQL via the ?api_key= query param.
        url = self._gql_base + ("&" if "?" in self._gql_base else "?") + "api_key=" + self._api_key
        try:
            r = requests.post(url, json={"query": query, "variables": variables or {}},
                              timeout=timeout,
                              headers={"Content-Type": "application/json"})
        except requests.RequestException as exc:
            raise RunpodError(f"RunPod API unreachable ({_mask(self._api_key)}): {exc}")
        if r.status_code == 401:
            raise RunpodError(f"RunPod rejected the API key ({_mask(self._api_key)}) — 401.")
        if r.status_code >= 400:
            raise RunpodError(f"RunPod API HTTP {r.status_code}: {r.text[:300]}")
        try:
            body = r.json()
        except ValueError:
            raise RunpodError(f"RunPod API returned non-JSON: {r.text[:200]}")
        if body.get("errors"):
            msg = "; ".join(e.get("message", str(e)) for e in body["errors"])
            raise RunpodError(f"RunPod GraphQL error: {msg}")
        return body.get("data") or {}

    def ping(self) -> dict:
        """Cheap auth check — returns {"id","email"?} for the account or raises."""
        data = self._gql("query { myself { id } }", timeout=15.0)
        return data.get("myself") or {}

    # ------------------------------------------------------------------ #
    # GPU catalog + selection
    # ------------------------------------------------------------------ #
    def list_gpu_types(self) -> list:
        """Return the RunPod GPU catalog with pricing, normalized:

        [{id, display_name, memory_gb, secure_price, community_price,
          on_demand_price, spot_price}]  (prices are USD/hr for 1 GPU; None if N/A)
        """
        q = """
        query GpuTypes {
          gpuTypes {
            id
            displayName
            memoryInGb
            securePrice
            communityPrice
            secureSpotPrice
            communitySpotPrice
          }
        }"""
        data = self._gql(q)
        out = []
        for g in (data.get("gpuTypes") or []):
            secure = g.get("securePrice")
            community = g.get("communityPrice")
            on_demand = secure if secure is not None else community
            spot = g.get("secureSpotPrice")
            if spot is None:
                spot = g.get("communitySpotPrice")
            out.append({
                "id": g.get("id"),
                "display_name": g.get("displayName") or g.get("id"),
                "memory_gb": g.get("memoryInGb"),
                "secure_price": secure,
                "community_price": community,
                "on_demand_price": on_demand,
                "spot_price": spot,
            })
        return out

    def pick_gpu(self, *, min_vram_gb: float = 0.0, max_hourly_usd: float = 0.0,
                 allow_spot: bool = False, cloud_type: str = "SECURE",
                 explicit_gpu_type: str = "") -> dict:
        """Choose the cheapest GPU that meets the constraints.

        Returns {gpu_type_id, display_name, memory_gb, price, is_spot} or raises
        RunpodError if nothing qualifies under the price ceiling.
        """
        catalog = self.list_gpu_types()
        secure = (cloud_type or "SECURE").upper() == "SECURE"

        def _price(g, spot):
            if spot:
                return g.get("spot_price")
            return g.get("secure_price") if secure else g.get("community_price")

        # Explicit GPU type: honor it, still enforce the price ceiling.
        if explicit_gpu_type:
            for g in catalog:
                if g["id"] == explicit_gpu_type or g["display_name"] == explicit_gpu_type:
                    for spot in ((True, False) if allow_spot else (False,)):
                        p = _price(g, spot)
                        if p is not None and (max_hourly_usd <= 0 or p <= max_hourly_usd):
                            return {"gpu_type_id": g["id"], "display_name": g["display_name"],
                                    "memory_gb": g["memory_gb"], "price": p, "is_spot": spot}
                    raise RunpodError(
                        f"GPU {explicit_gpu_type!r} exceeds the ${max_hourly_usd}/hr ceiling "
                        f"(cheapest {cloud_type} price ${_price(g, allow_spot)}).")
            raise RunpodError(f"GPU type {explicit_gpu_type!r} not found in the RunPod catalog.")

        # Otherwise pick cheapest qualifying by VRAM + price. Prefer spot when allowed.
        candidates = []
        for g in catalog:
            mem = g.get("memory_gb") or 0
            if min_vram_gb and mem < min_vram_gb:
                continue
            for spot in ((True, False) if allow_spot else (False,)):
                p = _price(g, spot)
                if p is None or p <= 0:
                    continue
                if max_hourly_usd > 0 and p > max_hourly_usd:
                    continue
                candidates.append((p, spot, g))
        if not candidates:
            raise RunpodError(
                f"No {cloud_type} GPU with ≥{min_vram_gb}GB VRAM under "
                f"${max_hourly_usd}/hr (spot={'on' if allow_spot else 'off'}).")
        candidates.sort(key=lambda c: (c[0], not c[1]))  # cheapest, spot-first on tie
        price, spot, g = candidates[0]
        return {"gpu_type_id": g["id"], "display_name": g["display_name"],
                "memory_gb": g["memory_gb"], "price": price, "is_spot": spot}

    # ------------------------------------------------------------------ #
    # Pod lifecycle
    # ------------------------------------------------------------------ #
    def create_pod(self, *, name: str, image: str, gpu_type_id: str, port: int,
                   cloud_type: str = "SECURE", gpu_count: int = 1,
                   container_disk_gb: int = 40, volume_gb: int = 0,
                   volume_mount_path: str = "/workspace", env: Optional[dict] = None,
                   docker_args: str = "", is_spot: bool = False,
                   bid_per_gpu: float = 0.0, data_center_id: str = "",
                   registry_auth_id: str = "",
                   entrypoint: Optional[list] = None,
                   start_cmd: Optional[list] = None,
                   network_volume_id: str = "") -> str:
        """Provision a pod (on-demand or interruptible/spot). Returns the pod id.

        ``entrypoint``/``start_cmd`` override the image's ENTRYPOINT/CMD, which is
        what lets a pod fetch something before its server starts — an image whose
        ENTRYPOINT is the server itself (vllm/vllm-openai) otherwise treats every
        argument as the server's own. Only the REST API accepts those, so the
        request goes there when they are given and stays on GraphQL otherwise.
        """
        if entrypoint or start_cmd:
            return self._create_pod_rest(
                network_volume_id=network_volume_id,
                name=name, image=image, gpu_type_id=gpu_type_id, port=port,
                cloud_type=cloud_type, gpu_count=gpu_count,
                container_disk_gb=container_disk_gb, volume_gb=volume_gb,
                volume_mount_path=volume_mount_path, env=env,
                is_spot=is_spot, bid_per_gpu=bid_per_gpu,
                data_center_id=data_center_id, registry_auth_id=registry_auth_id,
                entrypoint=entrypoint, start_cmd=start_cmd)
        env_list = [{"key": str(k), "value": str(v)} for k, v in (env or {}).items()]
        ports = f"{int(port)}/http"
        common = {
            "cloudType": (cloud_type or "SECURE").upper(),
            "gpuCount": int(gpu_count),
            "gpuTypeId": gpu_type_id,
            "name": name,
            "imageName": image,
            "containerDiskInGb": int(container_disk_gb),
            "volumeInGb": int(volume_gb),
            "volumeMountPath": volume_mount_path,
            "ports": ports,
            "dockerArgs": docker_args or "",
            "env": env_list,
        }
        if data_center_id:
            common["dataCenterId"] = data_center_id
        # A private image needs credentials RunPod holds for you: create them under
        # Settings -> Container Registry Credentials and paste the id here. Public
        # images (vLLM, llama.cpp, a public coderai) need none.
        if registry_auth_id:
            common["containerRegistryAuthId"] = registry_auth_id
        # A network volume persists across pods: weights downloaded once are
        # there for every later pod, which is the difference between paying a
        # cold model download per pod and paying it once.
        if network_volume_id:
            common["networkVolumeId"] = network_volume_id
        if is_spot:
            common["bidPerGpu"] = float(bid_per_gpu)
            mutation = "podRentInterruptable"
            var_type = "PodRentInterruptableInput"
        else:
            mutation = "podFindAndDeployOnDemand"
            var_type = "PodFindAndDeployOnDemandInput"
        q = f"""
        mutation Deploy($input: {var_type}!) {{
          {mutation}(input: $input) {{ id imageName machineId }}
        }}"""
        data = self._gql(q, {"input": common}, timeout=90.0)
        node = data.get(mutation) or {}
        pod_id = node.get("id")
        if not pod_id:
            raise RunpodError(f"RunPod did not return a pod id for {name!r} (capacity? {data}).")
        return pod_id

    def _create_pod_rest(self, *, name, image, gpu_type_id, port, cloud_type,
                         gpu_count, container_disk_gb, volume_gb, volume_mount_path,
                         env, is_spot, bid_per_gpu, data_center_id,
                         registry_auth_id, entrypoint, start_cmd,
                         network_volume_id: str = "") -> str:
        """Create a pod through the REST API, which accepts entrypoint overrides."""
        import requests
        rest = (getattr(self._cfg, "rest_base", "") or "https://rest.runpod.io/v1").rstrip("/")
        body = {
            "name": name,
            "imageName": image,
            "gpuTypeIds": [gpu_type_id],
            "gpuCount": int(gpu_count),
            "cloudType": (cloud_type or "SECURE").upper(),
            "containerDiskInGb": int(container_disk_gb),
            "volumeInGb": int(volume_gb),
            "volumeMountPath": volume_mount_path,
            "ports": [f"{int(port)}/http"],
            "env": {str(k): str(v) for k, v in (env or {}).items()},
        }
        if entrypoint:
            body["dockerEntrypoint"] = list(entrypoint)
        if start_cmd:
            body["dockerStartCmd"] = list(start_cmd)
        if data_center_id:
            body["dataCenterIds"] = [data_center_id]
        if registry_auth_id:
            body["containerRegistryAuthId"] = registry_auth_id
        if network_volume_id:
            body["networkVolumeId"] = network_volume_id
        if is_spot:
            body["interruptible"] = True
            body["bidPerGpu"] = float(bid_per_gpu)
        r = requests.post(f"{rest}/pods", json=body,
                          headers={"Authorization": f"Bearer {self._api_key}",
                                   "Content-Type": "application/json"},
                          timeout=90)
        if r.status_code >= 400:
            raise RunpodError(f"RunPod REST create failed ({r.status_code}): "
                              f"{(r.text or '')[:400]}")
        data = r.json() if r.content else {}
        pod_id = (data.get("id") or (data.get("pod") or {}).get("id")
                  or (data.get("data") or {}).get("id"))
        if not pod_id:
            raise RunpodError(f"RunPod REST create returned no pod id: {str(data)[:300]}")
        return pod_id

    def list_network_volumes(self) -> list:
        """The account's network volumes: [{id, name, size, dataCenterId}, …].

        Needed because a pod must run in the volume's OWN data center — a volume
        in EU-RO-1 cannot be attached to a pod anywhere else, and the failure
        arrives as an unhelpful capacity error.
        """
        import requests
        rest = (getattr(self._cfg, "rest_base", "") or "https://rest.runpod.io/v1").rstrip("/")
        try:
            r = requests.get(f"{rest}/networkvolumes",
                             headers={"Authorization": f"Bearer {self._api_key}"},
                             timeout=30)
            if r.status_code != 200:
                return []
            data = r.json()
            return data if isinstance(data, list) else (data.get("data") or [])
        except Exception as exc:
            print(f"[runpod] could not list network volumes: {exc}", flush=True)
            return []

    def get_pod(self, pod_id: str) -> dict:
        """Return {id, status, uptime_s, cost_per_hr, ports:[...], ready:bool, url?}."""
        q = """
        query Pod($input: PodFilter!) {
          pod(input: $input) {
            id
            desiredStatus
            costPerHr
            runtime {
              uptimeInSeconds
              ports { ip isIpPublic privatePort publicPort type }
            }
          }
        }"""
        data = self._gql(q, {"input": {"podId": pod_id}}, timeout=20.0)
        p = data.get("pod") or {}
        rt = p.get("runtime") or {}
        ports = rt.get("ports") or []
        return {
            "id": p.get("id") or pod_id,
            "status": p.get("desiredStatus"),
            "uptime_s": rt.get("uptimeInSeconds") or 0,
            "cost_per_hr": p.get("costPerHr"),
            "ports": ports,
            # A pod is "ready" for us once its http proxy port is exposed.
            "ready": bool(ports),
        }

    def stop_pod(self, pod_id: str) -> None:
        q = "mutation($input: PodStopInput!) { podStop(input: $input) { id } }"
        try:
            self._gql(q, {"input": {"podId": pod_id}}, timeout=30.0)
        except RunpodError:
            raise

    def terminate_pod(self, pod_id: str) -> None:
        q = "mutation($input: PodTerminateInput!) { podTerminate(input: $input) }"
        self._gql(q, {"input": {"podId": pod_id}}, timeout=30.0)

    def list_pods(self) -> list:
        """List the account's pods: [{id, name, status, cost_per_hr, image}]."""
        q = """
        query Pods {
          myself {
            pods { id name desiredStatus costPerHr imageName }
          }
        }"""
        data = self._gql(q)
        me = data.get("myself") or {}
        return [{
            "id": p.get("id"), "name": p.get("name"),
            "status": p.get("desiredStatus"), "cost_per_hr": p.get("costPerHr"),
            "image": p.get("imageName"),
        } for p in (me.get("pods") or [])]

    def get_pod_logs(self, pod_id: str, tail: int = 200) -> dict:
        """Best-effort container/vLLM logs for a pod via the REST API.

        RunPod's log surface is the web console; this tries the REST logs route and
        always returns a console URL to fall back to. Returns
        {console_url, logs?: str, error?: str}."""
        import requests
        out = {"console_url": pod_console_url(pod_id)}
        rest = (getattr(self._cfg, "rest_base", "") or "https://rest.runpod.io/v1").rstrip("/")
        hdrs = {"Authorization": f"Bearer {self._api_key}"}
        # RunPod's log surface has moved around; try the plausible routes in order and
        # keep the first that answers. The console link is the always-available path.
        attempts = [
            (f"{rest}/pods/{pod_id}/logs", None),
            (f"{rest}/pods/{pod_id}/logs", {"limit": tail}),
            (f"{rest}/pods/{pod_id}/containerLogs", None),
        ]
        errors = []
        for url, params in attempts:
            try:
                r = requests.get(url, headers=hdrs, params=params, timeout=15)
            except requests.RequestException as exc:
                errors.append(f"{url.rsplit('/', 1)[-1]}: unreachable ({exc})")
                continue
            if r.status_code == 200:
                try:
                    body = r.json()
                    out["logs"] = body if isinstance(body, str) else \
                        (body.get("logs") or body.get("data") or str(body))
                except ValueError:
                    out["logs"] = r.text
                return out
            errors.append(f"HTTP {r.status_code}")
        out["error"] = ("logs API unavailable (" + "; ".join(errors[:3])
                        + ") — use the console link")
        return out

    def wait_ready(self, pod_id: str, port: int, ready_timeout: float = 900.0,
                   poll_every: float = 5.0) -> str:
        """Poll until the pod exposes its proxy port; return the proxy base URL.

        Only waits for the port to be exposed — the HTTP service inside (vLLM) is
        health-checked separately by the worker."""
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            info = self.get_pod(pod_id)
            if info.get("status") in ("TERMINATED", "FAILED"):
                raise RunpodError(f"Pod {pod_id} entered status {info.get('status')} before ready.")
            if info.get("ready"):
                return pod_proxy_url(pod_id, port)
            time.sleep(poll_every)
        raise RunpodError(f"Pod {pod_id} did not expose port {port} within {ready_timeout}s.")
