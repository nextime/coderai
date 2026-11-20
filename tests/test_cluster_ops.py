"""Zero-config membership, prefix-cache routing, metering and recovery:
mDNS discovery with a shared cluster token, the affinity map the front routes
conversations with, the Prometheus page, node log tails, and the re-push of a
node's models when it comes back.

No network: discovery is fed fake ServiceInfo records, the front is a
TestClient over an in-memory registry.
"""

import json
import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codai.frontproxy.registry import Engine, EngineRegistry


# -------------------------------------------------------------- discovery
def _info(name, addr, port, props):
    class _I:
        def __init__(self):
            self.port = port
            self.properties = {k.encode(): v.encode() for k, v in props.items()}

        def parsed_addresses(self):
            return [addr]
    return _I()


def test_token_fingerprint_hides_the_token_and_only_matches_itself():
    from codai.cluster.discovery import token_fingerprint
    a, b = token_fingerprint("secret-1"), token_fingerprint("secret-2")
    assert a and b and a != b and len(a) == 24
    assert "secret" not in a and token_fingerprint("secret-1") == a
    assert token_fingerprint("") == ""


def test_discovery_lists_peers_and_knows_its_members():
    from codai.cluster.discovery import Discovery, SERVICE, token_fingerprint
    d = Discovery(name="head", port=8776, token="tok", instance_id="me0000000000")
    fp = token_fingerprint("tok")
    d._record(f"box2-a.{SERVICE}", _info("box2", "10.0.0.2", 8776,
                                         {"id": "aaaa", "name": "box2", "ver": "0.2.21",
                                          "scheme": "http", "fp": fp, "caps": "gguf,transformers"}))
    d._record(f"box3-b.{SERVICE}", _info("box3", "10.0.0.3", 8776,
                                         {"id": "bbbb", "name": "box3", "fp": "other"}))
    d._record(f"quiet-c.{SERVICE}", _info("quiet", "10.0.0.4", 8776,
                                          {"id": "cccc", "name": "quiet", "fp": fp, "serve": "0"}))
    d._record(f"head-me.{SERVICE}", _info("head", "10.0.0.1", 8776,
                                          {"id": "me0000000000", "name": "head", "fp": fp}))
    peers = d.peers()
    assert [p["name"] for p in peers] == ["box2", "box3", "quiet"]      # never ourselves
    box2 = peers[0]
    assert box2["url"] == "http://10.0.0.2:8776" and box2["member"] and box2["version"] == "0.2.21"
    assert box2["capabilities"] == ["gguf", "transformers"]
    assert not peers[1]["member"]
    assert [m["name"] for m in d.members()] == ["box2"]             # quiet holds the token but does not serve
    st = d.status()
    assert st["token_set"] and not st["running"] and len(st["peers"]) == 3


def test_members_become_node_specs_unless_listed_by_hand():
    from codai.cluster.discovery import member_node_specs
    members = [{"name": "box2", "url": "http://10.0.0.2:8776", "instance": "aaaa"},
               {"name": "box3", "url": "https://10.0.0.3:8776", "instance": "bbbb"},
               {"name": "box3", "url": "http://10.0.0.5:8776", "instance": "cccc"},
               {"name": "listed", "url": "http://10.0.0.9:8776", "instance": "dddd"}]
    specs = member_node_specs(members, "tok", known_names=["listed"], timeout_s=2.0)
    assert [s.name for s in specs] == ["box2", "box3", "box3-cccc"]
    assert all(s.api_key == "tok" and s.timeout_s == 2.0 for s in specs)
    assert specs[1].verify == "off" and specs[0].verify == "system"


def _cfg(nodes=None, **cluster):
    server = types.SimpleNamespace(internal_port_base=18780, port=18776, engine_specs=None,
                                   default_engine=None, isolate_gguf_engine=False,
                                   proxy_status_timeout=1.0, max_parallel_requests=1,
                                   max_parallel_requests_overrides={},
                                   dpm_force_performance_level_overrides={},
                                   engine_env_overrides={}, https=False)
    cl = dict(enabled=True, nodes=nodes or [], poll_timeout_s=1.0, rpc_servers=[], rpc_bin="",
              advertise_host="", token="", discovery=False, auto_join=True, serve=True,
              node_name="")
    cl.update(cluster)
    return types.SimpleNamespace(
        server=server, models=types.SimpleNamespace(max_model_instances=1),
        offload=types.SimpleNamespace(gpu_split=False),
        thermal=types.SimpleNamespace(supervisor_enabled=False),
        cluster=types.SimpleNamespace(**cl),
        vllm=types.SimpleNamespace(enabled=False), ds4=None, colibri=None, k3=None,
        ktransformers=None)


def _sup(cfg):
    from codai.frontproxy.engine_supervisor import EngineSupervisor
    return EngineSupervisor(cfg, None, EngineRegistry(), models_path=None, internal_token="t")


def test_the_head_merges_discovered_members_and_lends_its_token_to_listed_nodes():
    cfg = _cfg(nodes=[{"name": "box2", "url": "http://box2:8776"},
                      {"name": "keyed", "url": "http://k:8776", "api_key": "own"}],
               token="tok", discovery=True)
    sup = _sup(cfg)
    sup.discovery = types.SimpleNamespace(members=lambda: [
        {"name": "box2", "url": "http://10.0.0.2:8776", "instance": "aaaa"},   # listed: skipped
        {"name": "box9", "url": "http://10.0.0.9:8776", "instance": "zzzz"}])
    specs = {s.name: s for s in sup._node_specs()}
    assert set(specs) == {"box2", "keyed", "box9"}
    assert specs["box2"].api_key == "tok"           # blank → the shared token
    assert specs["keyed"].api_key == "own"          # its own stays
    assert specs["box9"].api_key == "tok" and getattr(specs["box9"], "discovered", False)
    sup._sync_cluster_nodes()
    got = {e.name: e.url for e in sup.registry.remotes()}
    assert got["box9"] == "http://10.0.0.9:8776"
    assert next(e for e in sup.registry.remotes() if e.name == "box9").discovered
    # The member vanishes → its engine goes; auto_join off → never added.
    sup.discovery = types.SimpleNamespace(members=lambda: [])
    sup._sync_cluster_nodes()
    assert set(e.name for e in sup.registry.remotes()) == {"box2", "keyed"}
    cfg.cluster.auto_join = False
    sup.discovery = types.SimpleNamespace(members=lambda: [
        {"name": "box9", "url": "http://10.0.0.9:8776", "instance": "zzzz"}])
    assert "box9" not in {s.name for s in sup._node_specs()}


def test_discovery_follows_the_config_live(monkeypatch):
    started, stopped = [], []

    class _Fake:
        def __init__(self, **kw):
            self.kw = kw
            self.started_at = 0.0
            self.error = ""

        def start(self):
            started.append(self.kw)
            self.started_at = 1.0
            return True

        def stop(self):
            stopped.append(self.kw)

        def refresh(self):
            pass

        def members(self):
            return []
    import codai.cluster.discovery as disc
    monkeypatch.setattr(disc, "Discovery", _Fake)
    cfg = _cfg(token="tok", discovery=False)
    sup = _sup(cfg)
    sup.discovery = None
    sup._discovery_sig = None
    sup._sync_discovery()
    assert sup.discovery is None and not started
    cfg.cluster.discovery = True
    cfg.cluster.node_name = "head"
    sup._sync_discovery()
    assert sup.discovery is not None and started[-1]["name"] == "head" and started[-1]["token"] == "tok"
    assert started[-1]["port"] == 18776 and started[-1]["scheme"] == "http"
    sup._sync_discovery()                       # unchanged config: nothing happens
    assert len(started) == 1
    cfg.cluster.token = "tok2"
    sup._sync_discovery()                       # token changed: restarted with it
    assert len(started) == 2 and len(stopped) == 1 and started[-1]["token"] == "tok2"
    cfg.cluster.discovery = False
    sup._sync_discovery()
    assert sup.discovery is None and len(stopped) == 2


def test_a_node_that_comes_back_gets_its_models_pushed_again():
    sup = _sup(_cfg(nodes=[{"name": "box2", "url": "http://box2:8776"}]))
    sup._sync_cluster_nodes()
    node = sup.registry.remotes()[0]
    node.assigned_models = {"m1", "m2"}
    sup._pending_reload.pop(node.name, None)

    class _Up:
        status_code = 200

        def json(self):
            return {"healthy": True, "loaded_models": [], "capabilities": ["gguf"]}

    class _Down:
        status_code = 503

        def json(self):
            return {}
    node.http_sync = types.SimpleNamespace(get=lambda url: _Up(), close=lambda: None)
    sup._poll_remote(node)
    assert node.healthy and node.name not in sup._pending_reload   # first contact: the normal push path
    node.http_sync = types.SimpleNamespace(get=lambda url: _Down(), close=lambda: None)
    sup._poll_remote(node)
    assert not node.healthy
    node.http_sync = types.SimpleNamespace(get=lambda url: _Up(), close=lambda: None)
    sup._poll_remote(node)
    assert node.healthy and sup._pending_reload[node.name] == ["m1", "m2"]
    assert sup._node_recoveries == 1


# ------------------------------------------------------ prefix affinity
def test_conversation_key_is_stable_across_turns_and_distinct_between_chats():
    from codai.frontproxy.affinity import conversation_key
    t1 = json.dumps({"model": "m", "messages": [{"role": "system", "content": "S"},
                                                {"role": "user", "content": "hi"}]}).encode()
    t2 = json.dumps({"model": "m", "messages": [{"role": "system", "content": "S"},
                                                {"role": "user", "content": "hi"},
                                                {"role": "assistant", "content": "hello"},
                                                {"role": "user", "content": "more"}]}).encode()
    other = json.dumps({"model": "m", "messages": [{"role": "user", "content": "bye"}]}).encode()
    k1 = conversation_key("/v1/chat/completions", t1)
    assert k1 and k1.startswith("hash:") and conversation_key("/v1/chat/completions", t2) == k1
    assert conversation_key("/v1/chat/completions", other) != k1
    assert conversation_key("/v1/chat/completions", t1, {"x-session-id": "abc"}) == "sid:abc"
    assert conversation_key("/v1/chat/completions",
                            json.dumps({"user": "u1", "messages": []}).encode()) == "user:u1"
    assert conversation_key("/v1/completions", json.dumps({"prompt": "Once"}).encode()).startswith("hash:")
    assert conversation_key("/v1/images/generations", t1) is None
    assert conversation_key("/v1/chat/completions", b"") is None
    assert conversation_key("/v1/chat/completions", b"not json") is None


def test_the_router_goes_back_to_the_engine_that_served_the_conversation():
    from codai.frontproxy.affinity import PrefixAffinity
    from codai.frontproxy.router import pick_engine
    reg = EngineRegistry()
    a = Engine(id=0, gpu=None, port=1, name="nvidia", backend="nvidia", primary=True)
    b = Engine(id=500, gpu=None, port=0, name="box2", backend="node", remote=True,
               url="http://box2:8776", capabilities={"gguf", "transformers"})
    reg.add(a)
    reg.add(b)
    reg.update_state(0, healthy=True, loaded_models=["m"])
    reg.update_state(500, healthy=True, loaded_models=["m"])
    a.inflight = 3
    # No preference: the least busy holder (the node, idle) wins.
    assert pick_engine(reg, "/v1/chat/completions", "POST", "m", "transformers") is b
    aff = PrefixAffinity(ttl_s=60)
    aff.remember("hash:1", "nvidia")
    assert pick_engine(reg, "/v1/chat/completions", "POST", "m", "transformers",
                       prefer=aff.get("hash:1")) is a
    # The preferred engine no longer holds the model → ordinary routing.
    reg.update_state(0, healthy=True, loaded_models=[])
    assert pick_engine(reg, "/v1/chat/completions", "POST", "m", "transformers",
                       prefer="nvidia") is b
    # A pin still wins over the preference.
    reg.update_state(0, healthy=True, loaded_models=["m"])
    assert pick_engine(reg, "/v1/chat/completions", "POST", "m", "transformers",
                       pinned="box2", prefer="nvidia") is b
    # Expiry and forgetting.
    aff2 = PrefixAffinity(ttl_s=0.01)
    aff2.remember("k", "nvidia")
    time.sleep(0.02)
    assert aff2.get("k") is None
    aff.forget_engine("nvidia")
    assert aff.get("hash:1") is None


# ------------------------------------------------- front: metrics + logs
def test_the_front_meters_by_key_exports_prometheus_and_tails_logs(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from codai.config import ConfigManager
    cm = ConfigManager(str(tmp_path))
    cm.load()
    cm.config.server.port = 18776
    cm.config.cluster.node_name = "me"
    cm.config.cluster.token = "shared"
    cm.auth_data.setdefault("tokens", []).append({"id": 1, "name": "fighters", "token": "secret"})
    cm.save_auth()
    from codai.frontproxy import app as fapp
    app = fapp.build_app(cm.config, config_dir=str(tmp_path))
    front = app.state.front
    e = Engine(id=0, gpu=None, port=1, name="nvidia", backend="nvidia")
    front.registry.add(e)
    front.registry.update_state(0, healthy=True, loaded_models=["m"], vram={"total": 24, "free": 10})
    n = Engine(id=500, gpu=None, port=0, name="box2", backend="node", remote=True, url="http://box2:8776")
    front.registry.add(n)
    front.supervisor = types.SimpleNamespace(rpc_manager=None, discovery=None, _node_recoveries=2)
    c = TestClient(app)

    # The shared cluster token opens /cluster/* like an API token does.
    assert c.get("/cluster/state", headers={"Authorization": "Bearer shared"}).status_code == 200
    assert c.get("/cluster/state", headers={"Authorization": "Bearer nope"}).status_code == 401

    # Metering: attributed to the token's NAME, never its value.
    front._record_activity("m", "text", 200, time.time() - 1.5, engine=e,
                           headers={"authorization": "Bearer secret"})
    front._record_activity("m", "text", 500, time.time(), engine=e,
                           headers={"authorization": "Bearer secret"})
    front._record_activity("m", "image", 200, time.time(), engine="box2",
                           headers={"cookie": "session_18776=abc"})
    front._record_activity("m", "text", 200, time.time(), engine=None, headers=None)
    u = front.metrics.usage_tables()
    assert u["by_key"]["fighters"]["requests"] == 2 and u["by_key"]["fighters"]["errors"] == 1
    assert u["by_key"]["fighters"]["engines"] == {"nvidia": 2}
    assert u["by_key"]["session"]["kinds"] == {"image": 1}
    assert u["by_key"]["anonymous"]["requests"] == 1
    assert u["by_model"]["m"]["requests"] == 4
    assert "secret" not in json.dumps(u)

    assert c.get("/metrics").status_code == 401                     # not public by default
    r = c.get("/metrics", headers={"Authorization": "Bearer shared"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    body = r.text
    assert 'coderai_requests_total{key="fighters",model="m",kind="text",status="200",engine="nvidia"} 1' in body
    assert 'coderai_engine_up{engine="nvidia",backend="nvidia",remote="0"} 1' in body
    assert 'coderai_node_up{node="box2"} 0' in body
    assert 'coderai_engine_vram_free_bytes{engine="nvidia",backend="nvidia",remote="0"} 10737418240' in body
    assert "coderai_node_recoveries_total 2" in body
    assert "coderai_info{version=" in body and "secret" not in body
    cm.config.server.metrics_public = True
    assert c.get("/metrics").status_code == 200
    r = c.get("/admin/api/usage", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200 and r.json()["by_key"]["fighters"]["requests"] == 2

    # Log tail: the front's own ring, and a node's over its /cluster/logs.
    fapp._LOG_RING.clear()
    fapp._LOG_RING.extend(["[front] one", "[front] two"])
    r = c.get("/cluster/logs?lines=1", headers={"Authorization": "Bearer shared"})
    assert r.status_code == 200 and r.json() == {"node": "me", "lines": ["[front] two"]}
    r = c.get("/admin/api/cluster/logs", headers={"Authorization": "Bearer secret"})
    assert r.json()["lines"] == ["[front] one", "[front] two"]

    class _R:
        status_code = 200

        def json(self):
            return {"node": "box2", "lines": ["[front] on box2"]}

    async def _get(url, params=None):
        assert url == "http://box2:8776/cluster/logs" and params == {"lines": 50}
        return _R()
    n.http_short = types.SimpleNamespace(get=_get)
    r = c.get("/admin/api/cluster/logs?node=box2&lines=50", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200 and r.json()["lines"] == ["[front] on box2"]
    assert c.get("/admin/api/cluster/logs?node=nope", headers={"Authorization": "Bearer secret"}).status_code == 404

    # The overview carries discovery state and the usage tables.
    r = c.get("/admin/api/cluster", headers={"Authorization": "Bearer secret"})
    d = r.json()
    assert d["discovery"]["enabled"] is False and d["discovery"]["token_set"] is True
    assert d["usage"]["by_key"]["fighters"]["requests"] == 2


def test_the_stdout_wrapper_keeps_a_ring_of_complete_lines():
    import io
    from codai.frontproxy import app as fapp
    fapp._LOG_RING.clear()
    w = fapp._TimestampedStdout(io.StringIO())
    w.write("hello ")
    w.write("world\nsecond line\npart")
    assert [ln.split("]", 1)[1] for ln in fapp.recent_log_lines()] == ["hello world", "second line"]
    w.write("ial\n")
    assert fapp.recent_log_lines(1)[0].endswith("partial")
    w.write("\rprogress 10%\rprogress 90%\n")
    assert fapp.recent_log_lines(1) == ["progress 90%"]


# --------------------------------------------------- row split / config
def test_split_mode_and_cluster_fields_round_trip_and_are_saved(tmp_path, monkeypatch):
    from codai.config import ConfigManager
    cm = ConfigManager(str(tmp_path))
    cm.load()
    cm.config.cluster.token = "tok"
    cm.config.cluster.discovery = True
    cm.config.cluster.prefix_affinity_ttl_s = 60
    cm.config.server.metrics_public = True
    cm.save_config()
    cm2 = ConfigManager(str(tmp_path))
    cm2.load()
    assert cm2.config.cluster.token == "tok" and cm2.config.cluster.discovery
    assert cm2.config.cluster.auto_join and cm2.config.cluster.prefix_affinity
    assert cm2.config.cluster.prefix_affinity_ttl_s == 60 and cm2.config.server.metrics_public
    # split_mode is on the model-save whitelist.
    import codai.admin.routes as routes
    src = Path(routes.__file__).read_text()
    assert '"split_mode"' in src
