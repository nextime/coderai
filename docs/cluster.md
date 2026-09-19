# Several machines as one coderai

Three things are possible once coderai runs on more than one machine, and
they answer three different needs:

| Need | Mechanism | Where it is set |
|---|---|---|
| More cards, one admin: a model runs *somewhere* on the network, requests are routed there by capability and load | **Cluster nodes** — other coderai installs used as engines of this one | Settings → Cluster; per model: Engine / card |
| One GGUF too big for any single machine | **llama.cpp RPC** — cards on other machines join this machine's layer split | Settings → Cluster → RPC servers; per model: *Cards on other machines* |
| One HF model too big for any single machine, served with continuous batching | **vLLM over Ray** (pipeline parallel across nodes) / **SGLang multi-node** | Settings → vLLM / ktransformers; per model: the vLLM / kt block |
| Several machines serving the same capability or the same `host` model, with failover | **Pools** of remotes and hosts | Settings → Remote capabilities; per model: *More machines* |

Everything is visible on **Admin → Cluster**: engines here, nodes, RPC
servers, hosts, capability remotes and pods, each with whether it answers now.

The Models page never lets you save a pair that cannot work: pick a backend
and the engines that cannot run it are greyed out; pin an engine and the
backends it cannot run are greyed out; an impossible pair already selected
shows a red note and Save refuses (`codai/cluster/compat.py` is the table,
`validate_engine_pin` applies it again server-side).

---

## 1. Cluster nodes — another coderai as an engine of this one

The front already runs N engine processes it cannot see inside: it polls
`/internal/engine-state`, hands each the models it owns, proxies by
capability and load. A **node** is that, one network hop away: a whole
coderai (its own front, cards, engines, thermal protection, admin) that this
front treats as one more engine.

### On the node

Nothing to install. Create an API token on its **Tokens** page — that is what
the head holds. Optionally, in Settings → Cluster:

* **This node's name** — what the head calls it (defaults to the hostname);
  also the key of a model's per-node path.
* **Let other heads use this install as a node** — on by default; off makes
  `/cluster/*` answer 401 to everyone.

If the node serves HTTPS with a self-signed certificate, copy its PEM: the
head pastes it in the node row (verify = *pem*).

### On the head

Settings → Cluster → *Use the nodes below as engines*, one row per node:
name, URL, the node's token, how to verify its certificate (*system* for a
public CA, *pem* with the pasted certificate, *off* for plain http / a LAN you
trust), and optionally a capability list that narrows what the head will send
it (blank = whatever the node reports).

```json
"cluster": {
  "enabled": true,
  "nodes": [
    {"name": "box2", "url": "https://box2:8776", "api_key": "…", "verify": "pem",
     "ca_pem": "-----BEGIN CERTIFICATE-----…", "capabilities": []}
  ]
}
```

The head then:

* polls `GET /cluster/state` on the node (its front answers from its own
  registry — cheap, never waits on a busy GPU there): loaded models, VRAM
  summed over its engines, tasks, and the **union of its engines'
  capabilities** (a node with an NVIDIA and a Radeon engine offers both
  `transformers` and `gguf`; the node's own front then picks the card);
* assigns models to it like to any engine — a per-model **Engine / card**
  pin by node name, or the default engine, or round-robin among compatible
  engines — and pushes the assignment with the models' entries
  (`POST /cluster/reload-config`); a model the node's catalogue lacks is
  registered in memory on every engine there, under a path that exists on
  the node, else the HuggingFace id / URL coderai can recover for it (the
  hub cache path encodes the repo). Set **Path on the node** on the model
  when neither applies;
* proxies requests there with the node's token, never the caller's
  credentials; the node applies its own queue, rate limits and thermal rules;
* forwards **Load / Unload** from the head's Models page to the node
  (`/cluster/model-load`, `/cluster/model-unload` — the node runs the action
  under a short-lived admin session of its own).

Nodes appear in the Engine / card select as `name (cluster node)`, on the
Tasks page tiles, and on the Cluster page with their engines, VRAM and last
error (`401: the node refused the token`, `404: not a coderai front`, a
connection error…). Changing the node list in Settings takes effect on the
next poll; no restart.

What a node is **not**: it never rents pods on the head's behalf and never
nests nodes of its own — the head sees the node's cards, not the node's
remotes.

## 2. One GGUF over several machines — llama.cpp RPC

llama.cpp's RPC backend makes a card on another machine one more ggml
device: register `host:port` where an `rpc-server` listens and the usual
layer split (`tensor_split`) spreads the model over local and remote cards
alike. Per token only the activations between the layers on each side cross
the wire, so decode over a LAN is workable; loading and prompt processing
are wire-bound. 1 GbE hurts (~100 MB/s: a 20 GB slice takes minutes to load
and long prompts crawl); 10 GbE is fine. The protocol has no authentication:
bind it to a LAN or WireGuard address, never the internet.

### The machine lending its cards

Settings → Cluster → *RPC servers this machine runs*: one row per process —
bind host/port, the ggml device (`CUDA0`, `Vulkan1` … blank = the server's
default), an optional memory cap, threads for a CPU server. The front starts
them, restarts them if they die (up to a crash-loop limit), and advertises
`advertise_host:port` on `/cluster/state`, so a head lists them on its
Models page. A machine that is also a cluster node does this from the same
Settings section; a machine that is only lending cards runs coderai the
same way and simply has no models.

The binary is built by `packaging/build-rpc-server.sh` from the **same
llama.cpp** the bundled llama-cpp-python vendors (the RPC protocol is
versioned; a client and a server from different commits refuse each other).
`build.sh` runs it after building llama-cpp-python; the OCI image and the
`coderai-llama` pod image ship it. The llama image also runs it beside the
API when started with `CODERAI_RPC_SERVER_PORT=50052`
(`CODERAI_RPC_SERVER_DEVICE`, `CODERAI_RPC_SERVER_MEM_MB`,
`CODERAI_RPC_SERVER_ONLY=1` for a card-only container) — a pod with a direct
TCP port, or a host you run the image on.

```json
"cluster": {"rpc_servers": [
  {"name": "3090", "host": "10.0.0.2", "port": 50052, "device": "CUDA0", "mem_gb": 0},
  {"name": "rx580", "host": "10.0.0.2", "port": 50053, "device": "Vulkan1"}
]}
```

### The model

On the model's page, **Cards on other machines**: `host:port, host2:port`.
Known servers (this machine's and every node's) are suggested. The model is
split by definition; the ratio in **Weight distribution** reads *local cards
first, then these servers in the listed order*, e.g. `0.6,0.4` for one local
card and one RPC server. Blank ratio = proportional to free memory, RPC
devices included (the VRAM and Speed strategies both see them).

```json
{"path": "/AI/guffcache/big-Q4_K_M.gguf", "rpc_servers": "10.0.0.2:50052, 10.0.0.3:50052",
 "tensor_split": "0.4,0.3,0.3"}
```

Under the hood (`codai/backends/ggml_rpc.py`): the RPC registration is
process-wide and permanent, so from the first registration on every load in
that engine gets an explicit device list — this machine's cards plus exactly
the servers the model asked for. A model that names servers the build cannot
use fails with the reason (`built without GGML_RPC`, `rpc-server at … did
not answer`) rather than quietly loading on local cards only. A model on a
cluster node with `rpc_servers` set spreads from that node; the servers are
addressed from there.

The bundled llama-cpp-python must have been built with `-DGGML_RPC=ON`
(`build.sh` and the images do; an older install: rebuild it).

## 3. One HF model over several machines — vLLM on Ray, SGLang multi-node

vLLM and SGLang spread a model over machines on their own; coderai does the
choreography: starts the peers, tells them where rank 0 is, waits for them,
launches, and tears everything down with the service.

**Keep tensor parallel inside a machine** (an all-reduce every layer wants
NVLink/PCIe) and **make pipeline parallel the number of machines** (one
activation transfer per layer boundary). Every machine must run the same
build — the `coderai-vllm` image on all of them is the easy way.

### vLLM

Settings → vLLM → *Several machines over Ray* sets the defaults; a model's
own **vLLM** block (visible when its backend is vLLM) overrides them:
tensor parallel, pipeline parallel, executor, an existing Ray cluster to
join, memory fraction, extra args, and the **nodes** — one per line,
`name | start command | stop command | gpus`. With nodes listed (or pipeline
> 1, or executor = ray) coderai:

1. starts a Ray head here (`ray start --head --port 6379
   --node-ip-address <advertise host>`), unless `ray_address` names an
   existing cluster;
2. runs each node's start command with `{ray_address}`, `{head}` and
   `{port}` filled in — e.g.
   `ssh box2 docker run -d --rm --name vllm-worker --gpus all --network host -e CODERAI_RAY_ADDRESS={ray_address} ghcr.io/nextime/coderai-vllm:latest`
   (the image joins as a Ray worker when `CODERAI_RAY_ADDRESS` is set);
3. waits until Ray reports tensor × pipeline GPUs (`nodes_ready_timeout_s`);
4. launches vLLM with `--pipeline-parallel-size N --distributed-executor-backend ray`.

Stopping the service runs the nodes' stop commands and `ray stop`.

```json
{"path": "Org/Huge-70B", "backend": "vllm",
 "vllm": {"tensor_parallel_size": 2, "pipeline_parallel_size": 2,
          "nodes": [{"name": "box2", "gpus": 2,
                     "start_cmd": "ssh box2 docker run -d --rm --name vllm-worker --gpus all --network host -e CODERAI_RAY_ADDRESS={ray_address} ghcr.io/nextime/coderai-vllm:latest",
                     "stop_cmd": "ssh box2 docker stop vllm-worker"}]}}
```

### SGLang (ktransformers)

Settings → ktransformers → *Several machines*, or the model's **kt** block:
`nnodes`, `tp_size` (GPUs across all nodes), the rank-0 address
(`dist_init_addr`, default `<advertise host>:20000`), and the node commands
for ranks 1… — `name | start command | stop command`, templated with
`{rank}`, `{nnodes}`, `{dist_init_addr}`, `{model}`. Ranks 1… are started
first; rank 0 launches here with `--nnodes N --node-rank 0 --dist-init-addr`.

## 4. Pools — several machines for one capability or one host model

* **Remote capabilities** (Settings → Remote capabilities): a capability's
  URL field takes several URLs, comma-separated. They form a pool: the
  first healthy, least-busy one takes each request; a URL that fails is
  marked down for 30 s and the request moves to the next one.
* **`backend: host` models**: *More machines*, one per line `url | token |
  start command | stop command` (blank fields inherit the ones above). Same
  pool: healthy + fewest in flight wins, a dead one is skipped, an on-demand
  one is started only when none is up, and one that dies mid-request hands
  the request to another.

## Reference

* Config: `cluster` section in `config.json` — `enabled`, `nodes`,
  `node_name`, `serve`, `poll_timeout_s`, `rpc_servers`, `rpc_bin`,
  `advertise_host`; `vllm.pipeline_parallel_size / distributed_executor_backend /
  ray_address / ray_port / nodes / nodes_ready_timeout_s`;
  `ktransformers.nnodes / dist_init_addr / tp_size / nodes`.
* Per model (models.json): `engine` (a node name), `node_path` /
  `node_paths`, `rpc_servers`, `vllm` block, `kt` block, `host.hosts`.
* Node API (token of the node): `GET /cluster/state`, `GET /cluster/rpc-servers`,
  `POST /cluster/reload-config`, `POST /cluster/model-load`, `POST /cluster/model-unload`.
* Head: `GET /admin/api/cluster` (what the Cluster page shows).
* Code: `codai/cluster/` (nodes, rpc, multinode, compat, overview),
  `codai/backends/ggml_rpc.py`, `codai/backends/overrides.py`,
  `codai/frontproxy/engine_supervisor.py` (remote engines),
  `codai/api/remote_gateway.EndpointPool`, `codai/api/host_worker.HostPool`.
