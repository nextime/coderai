"""Several coderai installs as one: nodes as engines of a head, one GGUF over
RPC servers on other machines, vLLM/SGLang over several nodes, pools of hosts
and remotes, and the engine ⇄ backend compatibility table the Models page
filters with.

Nothing here touches a network or a GPU: nodes are fake registries, RPC
servers are asserted absent from the local build (the shim must say so
plainly), commands are ``true``/``false``.
"""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codai.frontproxy.registry import Engine, EngineRegistry


# ------------------------------------------------------------------ nodes
def test_node_specs_are_parsed_and_narrowed():
    from codai.cluster.nodes import enabled_nodes, parse_nodes
    specs = parse_nodes([
        {"name": "box2", "url": "https://box2:8776/", "api_key": "k", "verify": "off",
         "capabilities": "gguf, whisper"},
        {"url": "http://box3:8776", "enabled": False},
        {"name": "box2", "url": "http://dup"},          # duplicate name: dropped
        {"name": "nourl"},
    ])
    assert [s.name for s in specs] == ["box2", "node2"]
    assert specs[0].url == "https://box2:8776"
    assert specs[0].capabilities == ["gguf", "whisper"]
    assert specs[0].headers() == {"Authorization": "Bearer k"}
    assert specs[0].verify_arg() is False
    cfg = types.SimpleNamespace(enabled=True, nodes=[s.__dict__ for s in specs],
                                poll_timeout_s=2.5)
    live = enabled_nodes(cfg)
    assert [s.name for s in live] == ["box2"] and live[0].timeout_s == 2.5
    assert enabled_nodes(types.SimpleNamespace(enabled=False, nodes=[])) == []


def test_a_node_reports_itself_as_one_engine():
    from codai.cluster.nodes import aggregate_state
    reg = EngineRegistry()
    a = Engine(id=0, gpu=None, port=1, name="nvidia", backend="nvidia")
    b = Engine(id=1, gpu=None, port=2, name="radeon", backend="vulkan")
    sysw = Engine(id=900, gpu=None, port=3, name="system", backend="none", role="system")
    for e in (a, b, sysw):
        reg.add(e)
    reg.update_state(0, healthy=True, loaded_models=["m1"], vram={"total": 24, "free": 10})
    reg.update_state(1, healthy=True, loaded_models=["m2"], vram={"total": 8, "free": 8})
    st = aggregate_state(reg, rpc_endpoints=[{"endpoint": "10.0.0.2:50052"}], node_name="box2")
    assert st["healthy"] and st["node"] == "box2"
    assert st["loaded_models"] == ["m1", "m2"]
    assert st["vram"] == {"total": 32, "free": 18, "used": 14}
    assert "transformers" in st["capabilities"] and "gguf" in st["capabilities"]
    assert "runpod" not in st["capabilities"]           # a node never rents on the head's behalf
    assert [e["name"] for e in st["engines"]] == ["nvidia", "radeon"]
    assert st["rpc_servers"][0]["endpoint"] == "10.0.0.2:50052"


def test_entries_travel_portable_and_localize_on_the_node(tmp_path):
    from codai.cluster.nodes import localize_entries, portable_entry
    here = tmp_path / "m.gguf"
    here.write_bytes(b"x")
    e = portable_entry({"path": "Org/Repo", "engine": "box2", "gpu_split": True,
                        "rpc_servers": "a:1", "tensor_split": "0.5,0.5"})
    assert e["cluster_source"] == "Org/Repo"
    payload, refused = localize_entries([
        {"path": str(here), "engine": "box2", "runpod": {"x": 1}},          # exists here
        {"path": "/nowhere/m.gguf", "cluster_source": "Org/Repo", "host": {}},
        {"path": "/nowhere/n.gguf", "node_paths": {"box2": str(here)}},
        {"path": "/nowhere/o.gguf", "alias": "o"},                          # nothing usable
        {"path": "/nowhere/p.gguf", "cluster_source": "Org/P", "rpc_servers": "a:1",
         "tensor_split": "0.5,0.5", "gpu_split": True},
    ], node_name="box2")
    assert [p["path"] for p in payload] == [str(here), "Org/Repo", str(here), "Org/P"]
    assert "engine" not in payload[0] and "runpod" not in payload[0]
    assert "host" not in payload[1]
    # The split travels only with a model that spreads over RPC peers.
    assert "tensor_split" in payload[3] and payload[3]["gpu_split"] is True
    assert len(refused) == 1 and "o" in refused[0]


# ---------------------------------------------------------- head registry
def _cluster_config(nodes, enabled=True, rpc=None):
    server = types.SimpleNamespace(internal_port_base=18780, port=18776, engine_specs=None,
                                   default_engine=None, isolate_gguf_engine=False,
                                   proxy_status_timeout=1.0, max_parallel_requests=1,
                                   max_parallel_requests_overrides={},
                                   dpm_force_performance_level_overrides={},
                                   engine_env_overrides={})
    return types.SimpleNamespace(
        server=server, models=types.SimpleNamespace(max_model_instances=1),
        offload=types.SimpleNamespace(gpu_split=False),
        thermal=types.SimpleNamespace(supervisor_enabled=False),
        cluster=types.SimpleNamespace(enabled=enabled, nodes=nodes, poll_timeout_s=1.0,
                                      rpc_servers=rpc or [], rpc_bin="", advertise_host=""),
        vllm=types.SimpleNamespace(enabled=False), ds4=None, colibri=None, k3=None,
        ktransformers=None)


def _supervisor(cfg, models_path=None):
    from codai.frontproxy.engine_supervisor import EngineSupervisor
    return EngineSupervisor(cfg, None, EngineRegistry(), models_path=models_path,
                            internal_token="t")


def test_remote_engines_are_built_from_cluster_nodes_and_never_spawned(monkeypatch):
    sup = _supervisor(_cluster_config([{"name": "box2", "url": "http://box2:8776",
                                        "api_key": "k", "capabilities": ["gguf"]}]))
    monkeypatch.setattr("codai.frontproxy.gpu_detect.nvidia_gpus", lambda: [])
    monkeypatch.setattr("codai.frontproxy.gpu_detect.gpu_vendors", lambda: set())
    engines = sup._build_engines()
    names = [e.name for e in engines]
    assert names == ["cpu", "box2"]
    node = engines[1]
    assert node.remote and node.url == "http://box2:8776" and node.state_path == "/cluster/state"
    assert node.capabilities == {"gguf"} and node.caps_fixed
    assert node.http_sync is not None and node.http_long is not None
    assert node.proc is None and not node.is_alive()      # no poll answered yet
    node.healthy = True
    assert node.is_alive()                                # alive = its poll answers
    sup._spawn(node)                                     # a no-op, not a crash
    assert node.proc is None
    assert sup.restart_engine(node.id) is False           # restarted from its own admin


def test_nodes_come_and_go_with_the_config():
    cfg = _cluster_config([{"name": "box2", "url": "http://box2:8776"}])
    sup = _supervisor(cfg)
    sup._sync_cluster_nodes()
    assert [e.name for e in sup.registry.remotes()] == ["box2"]
    first = sup.registry.remotes()[0]
    sup._sync_cluster_nodes()                             # unchanged: same object
    assert sup.registry.remotes()[0] is first
    cfg.cluster.nodes = [{"name": "box2", "url": "http://box2:9000"},
                         {"name": "box3", "url": "http://box3:8776"}]
    sup._sync_cluster_nodes()
    got = {e.name: e.url for e in sup.registry.remotes()}
    assert got == {"box2": "http://box2:9000", "box3": "http://box3:8776"}
    assert sup.registry.remotes()[0] is not first         # url changed → rebuilt
    cfg.cluster.enabled = False
    sup._sync_cluster_nodes()
    assert sup.registry.remotes() == []


def test_a_node_poll_takes_the_state_it_reports(monkeypatch):
    sup = _supervisor(_cluster_config([{"name": "box2", "url": "http://box2:8776"}]))
    sup._sync_cluster_nodes()
    node = sup.registry.remotes()[0]

    class _R:
        status_code = 200

        def json(self):
            return {"healthy": True, "loaded_models": ["m"], "vram": {"total": 32, "free": 1},
                    "capabilities": ["transformers", "gguf", "whisper"],
                    "engines": [{"name": "nvidia"}], "rpc_servers": [{"endpoint": "b:1"}]}

    node.http_sync = types.SimpleNamespace(get=lambda url: _R(), close=lambda: None)
    sup._poll_remote(node)
    assert node.healthy and node.loaded_models == {"m"}
    assert node.capabilities == {"transformers", "gguf", "whisper"}   # not fixed: believed
    assert node.rpc_servers == [{"endpoint": "b:1"}]

    class _Denied(_R):
        status_code = 401

    node.http_sync = types.SimpleNamespace(get=lambda url: _Denied(), close=lambda: None)
    sup._poll_remote(node)
    assert not node.healthy and "token" in node.last_error


def test_assignment_and_routing_reach_a_pinned_node(tmp_path):
    from codai.frontproxy.assignment import compute_assignment
    from codai.frontproxy.router import pick_engine
    models = tmp_path / "models.json"
    models.write_text(json.dumps({"text_models": [
        {"path": "/AI/a.gguf", "engine": "box2"},
        {"path": "/AI/b.gguf"},
    ]}))
    local = Engine(id=0, gpu=None, port=1, name="nvidia", backend="nvidia")
    node = Engine(id=500, gpu=None, port=0, name="box2", backend="node", remote=True,
                  url="http://box2:8776", capabilities={"gguf"})
    asg = compute_assignment([local, node], str(models))
    assert asg["box2"] == ["/AI/a.gguf"]
    reg = EngineRegistry()
    reg.add(local)
    reg.add(node)
    reg.update_state(0, healthy=True)
    reg.update_state(500, healthy=True)
    node.assigned_models = {"/AI/a.gguf"}
    picked = pick_engine(reg, "/v1/chat/completions", "POST", "a.gguf", "gguf", pinned="box2")
    assert picked is node
    # A hard pin to a node that is down fails rather than running elsewhere.
    reg.update_state(500, healthy=False)
    node.draining = True
    assert pick_engine(reg, "/v1/chat/completions", "POST", "a.gguf", "gguf", pinned="box2") is None


def test_the_head_pushes_entries_with_the_assignment(tmp_path):
    models = tmp_path / "models.json"
    models.write_text(json.dumps({"gguf_models": [
        {"path": "Org/Repo", "alias": "r", "engine": "box2", "rpc_servers": "x:1"}]}))
    sup = _supervisor(_cluster_config([{"name": "box2", "url": "http://box2:8776"}]),
                      models_path=str(models))
    sup._sync_cluster_nodes()
    node = sup.registry.remotes()[0]
    sent = {}

    class _R:
        status_code = 200

        def json(self):
            return {"refused": ["something"]}

    def _post(url, json=None):
        sent["url"], sent["json"] = url, json
        return _R()
    node.http_sync = types.SimpleNamespace(post=_post, close=lambda: None)
    sup._push_node_reload(node, ["r"])
    assert sent["url"].endswith("/cluster/reload-config")
    assert sent["json"]["assigned"] == ["r"]
    ent = sent["json"]["entries"][0]
    assert ent["path"] == "Org/Repo" and ent["model_type"] == "gguf_models"
    assert ent["cluster_source"] == "Org/Repo"


# ------------------------------------------------------------ front app
def test_the_front_answers_cluster_calls_with_a_valid_token(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from codai.config import ConfigManager
    cm = ConfigManager(str(tmp_path))
    cm.load()
    cm.config.server.port = 18776
    cm.config.cluster.node_name = "me"
    # A token in auth.json is what a head holds.
    cm.auth_data.setdefault("tokens", []).append({"id": 1, "name": "head", "token": "secret"})
    cm.save_auth()
    from codai.frontproxy.app import build_app
    app = build_app(cm.config, config_dir=str(tmp_path))
    front = app.state.front
    e = Engine(id=0, gpu=None, port=1, name="nvidia", backend="nvidia")
    front.registry.add(e)
    front.registry.update_state(0, healthy=True, loaded_models=["m"])
    front.supervisor = types.SimpleNamespace(rpc_manager=None)
    c = TestClient(app)
    assert c.get("/cluster/state").status_code == 401
    assert c.get("/cluster/state", headers={"Authorization": "Bearer nope"}).status_code == 401
    r = c.get("/cluster/state", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200
    d = r.json()
    assert d["node"] == "me" and d["loaded_models"] == ["m"] and "transformers" in d["capabilities"]

    # The reload push registers the localized entries on every engine here.
    calls = []

    class _Resp:
        status_code = 200

        def json(self):
            return {"added": ["Org/Repo"], "already_known": [], "refused": []}

    async def _post(url, json=None, headers=None):
        calls.append((url, json, headers))
        return _Resp()
    monkeypatch.setattr(front._long, "post", _post)
    r = c.post("/cluster/reload-config", headers={"Authorization": "Bearer secret"},
               json={"assigned": ["r"], "entries": [
                   {"path": "/not/here.gguf", "cluster_source": "Org/Repo", "engine": "x"},
                   {"path": "/not/here2.gguf"}]})
    assert r.status_code == 200
    d = r.json()
    assert d["added"] == ["Org/Repo"] and len(d["refused"]) == 1
    assert calls and calls[0][0].endswith("/v1/models/register")
    assert calls[0][1] == [{"path": "Org/Repo"}]
    assert calls[0][2]["authorization"] == "Bearer secret"

    # A head's load/unload arrive as token-guarded twins that run the front's
    # own admin action under a short-lived admin session of this node.
    seen = {}

    async def _fake_load(req):
        seen["path"] = req.url.path
        seen["cookie"] = req.headers.get("cookie", "")
        seen["auth"] = req.headers.get("authorization", "")
        seen["body"] = await req.body()
        from fastapi.responses import JSONResponse
        return JSONResponse({"success": True})
    front.model_load = _fake_load
    r = c.post("/cluster/model-load", headers={"Authorization": "Bearer secret"},
               json={"path": "/AI/m.gguf"})
    assert r.status_code == 200 and r.json() == {"success": True}
    assert seen["path"] == "/admin/api/model-load" and seen["cookie"].startswith("session=")
    assert seen["auth"] == "" and json.loads(seen["body"])["path"] == "/AI/m.gguf"
    from codai.admin.auth import SessionManager
    sm = SessionManager(tmp_path)
    assert sm.validate_session(seen["cookie"].split("=", 1)[1]) is None   # destroyed after use
    assert c.post("/cluster/model-load", json={"path": "x"}).status_code == 401

    # cluster.serve off → this install is never a node.
    cm.config.cluster.serve = False
    assert c.get("/cluster/state", headers={"Authorization": "Bearer secret"}).status_code == 401


def test_engines_list_marks_nodes():
    from codai.frontproxy.app import FrontProxy
    cfg = _cluster_config([])
    cfg.server.proxy_status_timeout = 1.0
    front = FrontProxy(cfg)
    front.registry.add(Engine(id=500, gpu=None, port=0, name="box2", backend="node",
                              remote=True, url="http://box2:8776", capabilities={"gguf"}))
    row = front.engines_list()[0]
    assert row["remote"] and row["url"] == "http://box2:8776" and row["capabilities"] == ["gguf"]
    # Node traffic drops the caller's credentials; local engines keep them.
    node = front.registry.get(500)
    assert front._eh(node, {"authorization": "Bearer user", "cookie": "s", "accept": "*/*"}) == {"accept": "*/*"}
    local = Engine(id=0, gpu=None, port=1)
    assert front._eh(local, {"authorization": "Bearer user"}) == {"authorization": "Bearer user"}
    assert front._lc(local) is front._long


# ------------------------------------------------------------- compat
def test_engine_backend_compatibility_table():
    from codai.cluster.compat import check, required_cap
    nv = ("nvidia", {"transformers", "gguf", "whisper", "vllm", "ds4"})
    vk = ("vulkan", {"gguf", "whisper"})
    assert required_cap("auto", "/x/m.gguf") == "gguf"
    assert required_cap("auto", "/x/m") == "transformers"
    assert check(*vk, "vllm")[0] is False
    assert check(*vk, "nvidia", "/x/m.safetensors")[0] is False   # transformers on a Vulkan card
    assert check(*vk, "auto", "/x/m.gguf") == (True, "")
    ok, why = check(*nv, "vulkan", "/x/m.gguf")
    assert ok and "CUDA" in why                                     # runs, differently
    assert check(*nv, "host")[0] is False
    assert check("node", {"gguf"}, "auto", "/x/m.gguf") == (True, "")


# -------------------------------------------------------- pools (D)
def test_capability_remotes_accept_lists(monkeypatch):
    from codai.api import remote_gateway as g
    cfg = types.SimpleNamespace(enabled=True, endpoints={
        "images": "http://a:1, http://b:2\nhttp://a:1/", "video": ["http://v:1"], "ocr": ""},
        api_key="")
    monkeypatch.setattr(g, "_remotes_config", lambda: cfg)
    lists = g.capability_endpoint_lists()
    assert lists["images"] == ["http://a:1", "http://b:2"] and lists["video"] == ["http://v:1"]
    assert "ocr" not in lists
    assert g.capability_endpoints()["images"] == "http://a:1"


def test_endpoint_pool_picks_a_healthy_url_and_fails_over(monkeypatch):
    from codai.api import remote_gateway as g
    up = {"http://a:1": False, "http://b:2": True}

    class _R:
        def __init__(self, ok):
            self.status_code = 200 if ok else 503

    monkeypatch.setattr(g.pod_http, "get", lambda url, **kw: _R(up[url.rsplit("/healthz", 1)[0]]))
    pool = g.EndpointPool("images", ["http://a:1", "http://b:2"])
    handle, url = pool.acquire()
    assert url == "http://b:2"
    assert pool.failover("http://b:2") is None            # the only healthy one just died
    up["http://a:1"] = True
    pool._down_until.clear()
    assert pool.failover("http://b:2") == "http://a:1"
    pool.release("http://a:1")
    up["http://a:1"] = False
    pool._healthy_until.clear()
    pool._down_until.clear()
    with pytest.raises(RuntimeError):
        g.EndpointPool("images", ["http://a:1"]).acquire()


def test_host_pool_spreads_over_machines_and_starts_one_only_when_none_is_up(monkeypatch):
    from codai.api import host_worker as hw
    up = {"http://a:1": True, "http://b:2": True}
    monkeypatch.setattr(hw, "_health_ok", lambda url, path, key, timeout=5.0: up[url])
    ran = []
    monkeypatch.setattr(hw.HostPool, "_run", lambda self, cmd: ran.append(cmd))
    pool = hw.HostPool("m", hw.parse_hosts({
        "url": "http://a:1", "api_key": "ka", "start_cmd": "start-a", "stop_cmd": "stop-a",
        "hosts": "http://b:2 | kb"}))
    h1, u1 = pool.acquire()
    h2, u2 = pool.acquire()
    assert {u1, u2} == {"http://a:1", "http://b:2"}      # least in-flight
    assert pool.api_key in ("ka", "kb")
    pool.release(h1)
    pool.release(h2)
    up["http://a:1"] = up["http://b:2"] = False
    pool._healthy_until.clear()
    # Nothing answers: the first host with a start_cmd is started (health
    # is patched to come back up right after the command).
    monkeypatch.setattr(hw.HostPool, "_run",
                        lambda self, cmd: (ran.append(cmd), up.__setitem__("http://a:1", True)))
    h, u = pool.acquire()
    assert u == "http://a:1" and ran[-1] == "start-a"
    assert pool.failover("http://a:1") in ("http://b:2", None)
    assert [s["url"] for s in pool.status()] == ["http://a:1", "http://b:2"]


# ---------------------------------------------------------------- RPC (B)
def test_rpc_endpoints_normalize_and_the_shim_refuses_plainly_without_the_backend():
    from codai.backends import ggml_rpc
    assert ggml_rpc.normalize_endpoints(" a:1, b:2\n a:1 ,nope") == ["a:1", "b:2"]
    assert ggml_rpc.normalize_endpoints(["x:5", "x:5"]) == ["x:5"]
    ok, why = ggml_rpc.available()
    if not ok:
        # The local build has no RPC backend: a model naming servers must
        # fail with the reason, never quietly load on local cards only.
        assert "GGML_RPC" in why or "libggml" in why
        with pytest.raises(RuntimeError):
            ggml_rpc.register_servers("10.0.0.2:50052")
    assert ggml_rpc.register_servers([]) == {}            # nothing asked, nothing done


def test_rpc_server_specs_and_command():
    from codai.cluster.rpc import RpcServerManager, build_cmd, parse_rpc_servers
    specs = parse_rpc_servers([
        {"name": "3090", "port": 50052, "device": "CUDA0", "mem_gb": 20, "threads": 4},
        {"port": 50052},                                   # duplicate port: dropped
        {"name": "x", "port": "bad"},
        {"port": 50053, "enabled": False},
    ])
    assert [s.name for s in specs] == ["3090", "rpc4"]
    assert build_cmd("/bin/rpc-server", specs[0]) == [
        "/bin/rpc-server", "-H", "0.0.0.0", "-p", "50052", "-d", "CUDA0", "-m", "20480", "-t", "4"]
    mgr = RpcServerManager(specs, binary="/nonexistent/rpc-server", advertise="10.0.0.9")
    assert [s.name for s in mgr.specs] == ["3090"]        # disabled one filtered
    mgr.start()                                          # no binary: says so, no crash
    eps = mgr.endpoints()
    assert eps[0]["endpoint"] == "10.0.0.9:50052" and eps[0]["alive"] is False


def test_rpc_servers_reach_the_gguf_backend_kwargs():
    src = Path(__file__).resolve().parents[1] / "codai" / "backends" / "vulkan.py"
    text = src.read_text()
    assert "_rpc.register_servers(_rpc_eps)" in text
    assert "with _rpc.load_with_devices(_rpc_eps):" in text
    mgr = (Path(__file__).resolve().parents[1] / "codai" / "models" / "manager.py").read_text()
    assert mgr.count("kwargs['rpc_servers'] = _rpc") == 2


# ---------------------------------------------------------- multi-node (C)
def test_multinode_commands_render_and_run():
    from codai.cluster.multinode import NodeSet, SglangNodes, parse_nodes, render, run_cmd
    assert render("ssh b ray start --address={ray_address} {x}", ray_address="h:1") == \
        "ssh b ray start --address=h:1 {x}"
    nodes = parse_nodes("box2 | true | true | 2\nbox3 | true\n\n")
    assert nodes == [{"name": "box2", "start_cmd": "true", "stop_cmd": "true", "gpus": "2"},
                     {"name": "box3", "start_cmd": "true"}]
    assert parse_nodes([{"start_cmd": "x"}]) == [{"start_cmd": "x", "name": "node1"}]
    ns = NodeSet(nodes)
    assert ns.gpus_expected() == 3
    ns.start(ray_address="h:1")
    assert len(ns._started) == 2
    ns.stop()
    assert ns._started == []
    with pytest.raises(RuntimeError):
        run_cmd("false")
    sg = SglangNodes([{"name": "b", "start_cmd": "true"}], 2, "10.0.0.1:20000")
    assert sg.rank0_args() == ["--nnodes", "2", "--node-rank", "0", "--dist-init-addr", "10.0.0.1:20000"]
    sg.start(model="/m")
    with pytest.raises(RuntimeError):
        SglangNodes([], 3, "h:1").start(model="/m")       # not enough node commands


def test_vllm_and_sglang_launch_commands_carry_the_parallelism():
    from codai.api import kt_worker, vllm_worker
    from codai.config import KtransformersConfig, VllmConfig
    c = VllmConfig(tensor_parallel_size=2, pipeline_parallel_size=2,
                   nodes=[{"name": "b", "start_cmd": "true"}])
    assert vllm_worker.needs_ray(c)
    cmd = vllm_worker._launch_cmd("py", c, "0.0.0.0", 1, "/m")
    assert cmd[cmd.index("--pipeline-parallel-size") + 1] == "2"
    assert cmd[cmd.index("--distributed-executor-backend") + 1] == "ray"
    assert not vllm_worker.needs_ray(VllmConfig())
    plain = vllm_worker._launch_cmd("py", VllmConfig(), "0.0.0.0", 1, "/m")
    assert "--distributed-executor-backend" not in plain and "--pipeline-parallel-size" not in plain
    k = KtransformersConfig(nnodes=2, dist_init_addr="10.0.0.1:20000", tp_size=4)
    kc = kt_worker._launch_cmd(k, "0.0.0.0", 1, "/m")
    assert kc[kc.index("--nnodes") + 1] == "2" and kc[kc.index("--tp-size") + 1] == "4"
    assert "--nnodes" not in kt_worker._launch_cmd(KtransformersConfig(), "0.0.0.0", 1, "/m")


def test_per_model_blocks_override_only_the_allowed_fields():
    from codai.backends.overrides import KT_FIELDS, VLLM_FIELDS, apply
    from codai.config import KtransformersConfig, VllmConfig
    v = apply(VllmConfig(), {"pipeline_parallel_size": "3", "venv": "/evil", "extra_args": "--x",
                             "gpu_memory_utilization": "0.5", "nodes": [{"name": "b"}]},
              VLLM_FIELDS, "vllm")
    assert v.pipeline_parallel_size == 3 and v.venv == "" and v.extra_args == "--x"
    assert v.gpu_memory_utilization == 0.5 and v.nodes == [{"name": "b"}]
    k = apply(KtransformersConfig(), {"nnodes": 2, "install_dir": "/evil", "tp_size": ""},
              KT_FIELDS, "kt")
    assert k.nnodes == 2 and k.install_dir is None and k.tp_size == 0
    assert apply(VllmConfig(), {}, VLLM_FIELDS) is not None


# ---------------------------------------------------------------- config
def test_cluster_config_round_trips(tmp_path):
    from codai.config import ConfigManager
    cm = ConfigManager(str(tmp_path))
    cm.load()
    cm.config.cluster.enabled = True
    cm.config.cluster.nodes = [{"name": "box2", "url": "http://box2:8776", "api_key": "k"}]
    cm.config.cluster.rpc_servers = [{"name": "3090", "port": 50052}]
    cm.config.vllm.pipeline_parallel_size = 2
    cm.config.vllm.nodes = [{"name": "b", "start_cmd": "true"}]
    cm.config.ktransformers.nnodes = 2
    cm.save_config()
    cm2 = ConfigManager(str(tmp_path))
    cm2.load()
    assert cm2.config.cluster.enabled and cm2.config.cluster.nodes[0]["name"] == "box2"
    assert cm2.config.cluster.rpc_servers[0]["port"] == 50052
    assert cm2.config.vllm.pipeline_parallel_size == 2 and cm2.config.vllm.nodes[0]["name"] == "b"
    assert cm2.config.ktransformers.nnodes == 2


def test_the_save_path_refuses_an_impossible_pin_and_knows_nodes(monkeypatch):
    from codai.admin import routes
    monkeypatch.setattr(routes, "_cluster_node_named",
                        lambda name: {"name": "box2", "capabilities": ["gguf"]} if name == "box2" else None)
    assert routes.validate_engine_pin("box2", "/x/m.gguf", None, model_backend="auto") == []
    warn = routes.validate_engine_pin("box2", "/x/m.safetensors", None, model_backend="nvidia")
    assert warn and "transformers" in warn[0]
    warn = routes.validate_engine_pin("radeon", "/x/m.gguf", None, model_backend="vllm")
    assert warn and "vllm" in warn[0]


# ------------------------------------------------------------ fan-out (work)
def test_fanout_splits_and_merges_json_requests():
    from codai.cluster import fanout as f
    assert f.kind_for("/v1/images/generations") == "count" and f.kind_for("/v1/chat/completions") is None
    parts = f.split_json("count", {"n": 5, "seed": 7, "prompt": "x"}, 3, 2)
    assert [p["n"] for p in parts] == [2, 2, 1] and [p["seed"] for p in parts] == [7, 9, 11]
    assert f.split_json("count", {"n": 1}, 3, 2) is None
    merged = f.merge_json("count", "/v1/images/generations", {"n": 5},
                          [{"created": 1, "data": [1, 2]}, {"data": [3, 4]}, {"data": [5]}])
    assert merged["data"] == [1, 2, 3, 4, 5]
    emb = f.split_json("list:input", {"input": ["a", "b", "c"], "model": "e"}, 2, 2)
    assert [p["input"] for p in emb] == [["a", "b"], ["c"]]
    m = f.merge_json("list:input", "/v1/embeddings", {"input": ["a", "b", "c"]}, [
        {"data": [{"index": 0, "embedding": [1]}, {"index": 1, "embedding": [2]}],
         "usage": {"prompt_tokens": 2, "total_tokens": 2}},
        {"data": [{"index": 0, "embedding": [3]}], "usage": {"prompt_tokens": 1, "total_tokens": 1}}])
    assert [e["index"] for e in m["data"]] == [0, 1, 2] and m["usage"]["total_tokens"] == 3
    rr = f.merge_json("list:documents", "/v1/rerank", {"documents": ["d0", "d1", "d2"], "top_n": 2}, [
        {"results": [{"index": 0, "relevance_score": 0.1}, {"index": 1, "relevance_score": 0.9}]},
        {"results": [{"index": 0, "relevance_score": 0.5}]}])
    assert [r["index"] for r in rr["results"]] == [1, 2]
    tts = f.split_json("text:input", {"input": "One. Two! Three? Four."}, 2, 2)
    assert [p["input"] for p in tts] == ["One. Two!", "Three? Four."]
    assert f.split_json("text:input", {"input": "Just one sentence."}, 2, 2) is None


def test_fanout_chooses_engines_that_can_serve_and_honours_the_list():
    from codai.cluster.fanout import choose_engines
    reg = EngineRegistry()
    a = Engine(id=0, gpu=None, port=1, name="nvidia", backend="nvidia")
    b = Engine(id=500, gpu=None, port=0, name="box2", backend="node", remote=True,
               url="http://box2", capabilities={"transformers"})
    c = Engine(id=501, gpu=None, port=0, name="box3", backend="node", remote=True,
               url="http://box3", capabilities={"gguf"})
    for e in (a, b, c):
        reg.add(e)
        reg.update_state(e.id, healthy=True)
    got = [e.name for e in choose_engines(reg, "sdxl", "transformers", [], 0)]
    assert got == ["nvidia", "box2"]                      # box3 cannot serve transformers
    got = [e.name for e in choose_engines(reg, "sdxl", "transformers", ["box2"], 0)]
    assert got == ["box2"]
    got = [e.name for e in choose_engines(reg, "sdxl", "transformers", ["local", "box2"], 1)]
    assert got == ["nvidia"]
    reg.update_state(500, healthy=False)
    assert [e.name for e in choose_engines(reg, "sdxl", "transformers", [], 0)] == ["nvidia"]


def test_transcription_windows_shift_and_join():
    from codai.cluster.fanout import merge_transcriptions
    parts = [{"text": "hello there", "segments": [{"start": 0.0, "end": 1.0, "text": "hello there"}]},
             {"text": "general", "segments": [{"start": 0.5, "end": 1.5, "text": "general"}],
              "words": [{"word": "general", "start": 0.5, "end": 1.5}]}]
    m = merge_transcriptions("verbose_json", parts, [(0.0, 10.0), (10.0, 20.0)])
    assert m["text"] == "hello there general"
    assert [s["start"] for s in m["segments"]] == [0.0, 10.5] and m["segments"][1]["id"] == 1
    assert m["words"][0]["start"] == 10.5 and m["duration"] == 20.0
    assert merge_transcriptions("json", parts, [(0, 1), (1, 2)]) == {"text": "hello there general"}


def test_the_front_fans_an_image_request_out_and_merges(tmp_path):
    """Two fake engines each generate their share of n=3; the client gets one
    answer with 3 images, and a node's file URLs are rewritten to the head."""
    from fastapi.testclient import TestClient
    from codai.config import ConfigManager
    cm = ConfigManager(str(tmp_path))
    cm.load()
    cm.config.server.port = 18777
    (tmp_path / "models.json").write_text(json.dumps({"image_models": [
        {"path": "sdxl", "distribute": {"enabled": True}}]}))
    from codai.frontproxy.app import build_app
    app = build_app(cm.config, config_dir=str(tmp_path))
    front = app.state.front
    a = Engine(id=0, gpu=None, port=1, name="nvidia", backend="nvidia")
    b = Engine(id=500, gpu=None, port=0, name="box2", backend="node", remote=True,
               url="http://box2:8776", capabilities={"transformers"})
    front.registry.add(a)
    front.registry.add(b)
    front.registry.update_state(0, healthy=True)
    front.registry.update_state(500, healthy=True)
    front.supervisor = types.SimpleNamespace(rpc_manager=None)
    seen = []

    class _Resp:
        def __init__(self, url, content):
            self.status_code, self.content, self.headers = 200, content, {"content-type": "application/json"}

    def _client(base):
        class _C:
            async def request(self, method, url, headers=None, content=None, **kw):
                body = json.loads(content)
                seen.append((base, body["n"], body.get("seed")))
                data = [{"url": f"{base}/v1/files/img{base[-1]}{i}.png"} for i in range(body["n"])]
                return _Resp(url, json.dumps({"created": 1, "data": data}).encode())
        return _C()
    front._long = _client("http://local")
    b.http_long = _client("http://box2")
    # The front now enforces the API bearer on /v1 (it used to fall open when
    # nothing was configured), so the token this test sends has to be a real one.
    import json as _json
    _auth = tmp_path / "auth.json"
    _d = _json.loads(_auth.read_text()) if _auth.exists() else {"users": [], "sessions": {}}
    _d.setdefault("tokens", []).append(
        {"id": 99, "name": "test", "token": "t", "provider": "openai"})
    _auth.write_text(_json.dumps(_d))
    c = TestClient(app)
    r = c.post("/v1/images/generations", json={"model": "sdxl", "prompt": "cat", "n": 3, "seed": 1},
               headers={"Authorization": "Bearer t", "Host": "head:18777"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["data"]) == 3
    assert sorted(seen) == [("http://box2", 1, 3), ("http://local", 2, 1)]
    # The node's file lives on the node; its URL now points at the head.
    node_urls = [x["url"] for x in d["data"] if "img2" in x["url"]]
    assert node_urls == ["http://head:18777/v1/files/img20.png"]
    assert front._node_files["img20.png"] is b
    # n=1 carries nothing to split: the ordinary single-engine route takes it.
    import asyncio as _aio

    class _Req:
        headers = {"content-type": "application/json"}
    body = json.dumps({"model": "sdxl", "prompt": "cat", "n": 1}).encode()
    out = _aio.new_event_loop().run_until_complete(
        front._maybe_fanout(_Req(), "/v1/images/generations", "sdxl", body))
    assert out is None


# ------------------------------------------------------- LoRA multi-node (DDP)
def _ddp_worker(rank, world, port, q):
    import torch
    from codai.cluster import ddp
    block = {"rank": rank, "world_size": world, "master_addr": "127.0.0.1",
             "master_port": port, "backend": "gloo", "timeout_s": 60}
    with ddp.context(block) as ctx:
        p = torch.nn.Parameter(torch.full((4,), float(rank + 1)))
        ddp.broadcast_params([p])                       # everyone starts from rank 0's 1.0
        start = p.data.clone()
        p.grad = torch.full((4,), float(rank + 1))      # rank 0: 1, rank 1: 2 → mean 1.5
        ddp.sync_grads([p])
        q.put((rank, start.tolist(), p.grad.tolist(), [ddp.index(s, 3) for s in range(4)],
               ddp.is_main(), ctx.scratch_dir() if not ctx.is_main else ""))


def test_ddp_ranks_share_init_and_average_gradients():
    import multiprocessing as mp
    from codai.cluster import ddp
    pytest.importorskip("torch.distributed")
    port = ddp.free_port()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_ddp_worker, args=(r, 2, port, q)) for r in range(2)]
    for p in procs:
        p.start()
    out = {}
    for _ in procs:
        r, start, grad, idx, main, scratch = q.get(timeout=120)
        out[r] = (start, grad, idx, main, scratch)
    for p in procs:
        p.join(timeout=30)
    assert out[0][0] == [1.0] * 4 and out[1][0] == [1.0] * 4     # broadcast from rank 0
    assert out[0][1] == [1.5] * 4 and out[1][1] == [1.5] * 4     # averaged gradient
    assert out[0][2] == [0, 2, 1, 0] and out[1][2] == [1, 0, 2, 1]  # interleaved samples
    assert out[0][3] is True and out[1][3] is False and out[1][4]


def test_ddp_helpers_are_no_ops_outside_a_context():
    from codai.cluster import ddp
    assert ddp.current() is None and ddp.is_main() and ddp.index(7, 3) == 1
    ddp.sync_grads([]); ddp.broadcast_params([])
    assert ddp.parse_nodes("box2, box3") == [{"name": "box2"}, {"name": "box3"}]
    assert ddp.parse_nodes([{"url": "http://x", "api_key": "k"}])[0]["url"] == "http://x"
    with pytest.raises(ValueError):
        ddp.resolve_peers([{"name": "nowhere"}])
    with ddp.context({"world_size": 1}) as c:
        assert c is None


def test_the_trainers_call_the_ddp_hooks():
    src = (Path(__file__).resolve().parents[1] / "codai" / "api" / "loras.py").read_text()
    assert src.count("_ddp.index(step, n)") == 5
    assert src.count("_ddp.sync_grads(lora_params)") == 5
    assert src.count("_ddp.broadcast_params(lora_params)") == 5
    assert "with _ddp.context(getattr(req, \"distributed\", None)):" in src


# --------------------------------------------- pipeline parts on other machines
def test_components_parse_and_tensor_roundtrip():
    from codai.cluster.components import b64_to_tensor, parse_components, tensor_to_b64
    torch = pytest.importorskip("torch")
    assert parse_components({"text_encoder": "box2", "low_noise": " here ", "vae": ""}) == {"text_encoder": "box2"}
    assert parse_components({"vae": {"url": "http://x", "api_key": "k"}})["vae"]["url"] == "http://x"
    t = torch.randn(1, 4, 3, 2, 2, dtype=torch.bfloat16)
    back = b64_to_tensor(tensor_to_b64(t))
    assert back.dtype == torch.bfloat16 and torch.equal(back, t)


def test_sliced_timesteps_resume_a_flow_match_scheduler_mid_way():
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    from codai.cluster.components import SlicedTimesteps, boundary_step
    sched = diffusers.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=5.0)
    sched.set_timesteps(10)
    full_ts, full_sig = sched.timesteps.clone(), sched.sigmas.clone()
    with SlicedTimesteps(sched, 6):
        sched.set_timesteps(10)
        assert torch.equal(sched.timesteps, full_ts[6:]) and torch.equal(sched.sigmas, full_sig[6:])
        # step() still finds each remaining timestep's sigma pair
        x = torch.zeros(1, 2)
        out = sched.step(torch.ones(1, 2), sched.timesteps[0], x, return_dict=False)[0]
        assert torch.isfinite(out).all()
    sched.set_timesteps(10)
    assert torch.equal(sched.timesteps, full_ts)          # restored after the block

    class _Cfg:
        boundary_ratio = 0.875

    class _Pipe:
        config = _Cfg()
        transformer_2 = object()
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=5.0)
    k = boundary_step(_Pipe(), {"num_inference_steps": 10})
    ts = _Pipe.scheduler
    ts.set_timesteps(10)
    assert k is not None and float(ts.timesteps[k]) < 875 <= float(ts.timesteps[k - 1])
    _Pipe.transformer_2 = None
    assert boundary_step(_Pipe(), {"num_inference_steps": 10}) is None


def test_handoff_ops_drive_a_fake_pipeline():
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from codai.api import video as v
    from codai.cluster.components import b64_to_tensor, tensor_to_b64
    import diffusers

    class _FakePipe:
        _execution_device = torch.device("cpu")
        transformer = types.SimpleNamespace(dtype=torch.float32)
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

        def __init__(self):
            self.calls = []

        def encode_prompt(self, prompt, negative_prompt, do_classifier_free_guidance,
                          num_videos_per_prompt, max_sequence_length, device):
            return torch.ones(1, 3, 4), torch.zeros(1, 3, 4)

        def __call__(self, **kw):
            self.scheduler.set_timesteps(kw["num_inference_steps"])
            self.calls.append({"steps_left": len(self.scheduler.timesteps),
                               "has_embeds": "prompt_embeds" in kw, "out": kw.get("output_type")})
            return types.SimpleNamespace(frames=kw["latents"] * 2)

    req = types.SimpleNamespace(model="wan")
    p = _FakePipe()
    enc = v._run_handoff(p, {"prompt": "a cat"}, req, {"op": "encode"})
    assert isinstance(enc, v._HandoffPayload) and b64_to_tensor(enc.payload["prompt_embeds"]).shape == (1, 3, 4)
    lat = torch.full((1, 2, 2), 3.0)
    den = v._run_handoff(p, {"num_inference_steps": 8, "prompt": "a cat"}, req,
                         {"op": "denoise", "start_step": 5, "latents": tensor_to_b64(lat),
                          "return": "latent", "prompt_embeds": enc.payload["prompt_embeds"],
                          "negative_prompt_embeds": enc.payload["negative_prompt_embeds"]})
    assert p.calls[-1] == {"steps_left": 3, "has_embeds": True, "out": "latent"}
    assert torch.equal(b64_to_tensor(den.payload["latents"]), lat * 2)
    dec = v._run_handoff(p, {"num_inference_steps": 8}, req,
                         {"op": "decode", "latents": tensor_to_b64(lat), "return": "video"})
    assert p.calls[-1]["steps_left"] == 0 and not isinstance(dec, v._HandoffPayload)
    with pytest.raises(ValueError):
        v._run_handoff(p, {}, req, {"op": "bogus", "latents": tensor_to_b64(lat)})
