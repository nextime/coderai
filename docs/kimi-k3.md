# Kimi-K3 via kimi-k3-in-c (`k3` backend)

CoderAI can serve **Kimi-K3** (2.78T params, 16/896 experts active, 93 layers) through
[FareedKhan-dev/kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c), a
portable-C CPU engine that streams the always-on dense trunk + routed experts from disk,
running the full model in as little as ~8 GB RAM (more RAM = faster, not "more capable").

Upstream `kimi-k3-in-c` is a **one-shot batch CLI**. CoderAI applies
`packaging/patch-k3.py` to add a **resident serve loop** that speaks the same mux
stdin/stdout protocol as [colibri](glm-colibri.md), so the same in-process
`MuxEngine` client drives it — the model, packed trunk, expert cache and tokenizer stay
warm across requests (no ~108 GB cold-start trunk read per prompt).

## Requirements

- **CPU**: x86-64 with **AVX2 + FMA** (no AVX-512 required).
- **Storage**: ~1.7 TB fast local storage — the ~1.56 TB HF checkpoint **plus** a
  ~109 GB packed dense trunk produced by the repo's `scripts/pack-trunk.sh`.
- **RAM**: 8 GB minimum; pin more of the trunk for speed (see `trunk_gb` / presets).
- CoderAI never downloads the checkpoint — point the config at it.

## Enable & configure

Settings → **Kimi-K3 (kimi-k3-in-c)** card, or `config.json` `"k3"`:

| field | meaning |
|---|---|
| `enabled` | turn the engine on |
| `model_path` | the Kimi-K3 checkpoint directory (config.json + tokenizer + shards) |
| `trunk_dir` | packed dense trunk dir (`scripts/pack-trunk.sh`); blank = fully resident (~114 GB RAM) |
| `tok_dir` | tokenizer dir; blank = use the checkpoint dir |
| `model_id` | id/alias that routes here (default `kimi-k3`) |
| `preset` | `laptop`/`desktop`/`workstation`/`server`/`max` (blank = use `trunk_gb`/`cache_gb`) |
| `trunk_gb` / `cache_gb` | GiB pinned of the trunk / expert LRU cache |
| `ctx` | serve context capacity (engine `K3_MAXT`) |
| `extra_env` | `K3_EXPERT_GB`, `K3_BITS`, `K3_DIRECT`, `K3_PIPE`, … |
| `auto_build` | clone + patch + `make` the engine binary on first use |

First use clones the repo, applies the serve-loop patch, and builds `bin/k3`
(`build.sh --k3` prebuilds it for packaging).

## Routing

A request routes to `k3` when (highest precedence first):

1. the model's models.json entry has `backend: "k3"`, **or**
2. `k3` is enabled and the name equals its `model_id` alias, **or**
3. the name contains a Kimi-K3 marker (`kimi-k3`) **and** only one Kimi-capable engine
   is enabled.

If **both** `k3` and `colibri` (which also serves Kimi-K3) are enabled, a bare Kimi
name is **ambiguous** — set the model's `backend` to choose. See
[`resolve_engine_backend`](../codai/models/manager.py).

## Notes

- **CPU-only** (no CUDA/GPU tier here) — for GPU Kimi-K3, use colibri's `kimi_k3`
  engine (optional Vulkan tier) or ktransformers instead.
- Greedy incremental decode, single conversation at a time (single KV slot); KV /
  recurrent state is reset between requests (stateless HTTP turns).
- The Kimi-K3 chat template is rendered as an XTML string
  (`render_kimi_xtml`, `codai/backends/colibri_families.py`) and tokenized by the engine.
