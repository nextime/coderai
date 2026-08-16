# ktransformers via SGLang (`kt` backend)

CoderAI can serve large MoE models through
[ktransformers](https://github.com/kvcache-ai/ktransformers) (KVCache.AI, Apache-2.0), a
**CPU+GPU heterogeneous** inference engine — Intel AMX/AVX512/AVX2 CPU kernels for the
quantized experts + GPU for the dense trunk/attention, with NUMA-aware expert placement
and disk offload. It exposes an OpenAI-compatible HTTP server via **SGLang**, so — like
[ds4](deepseek-ds4.md) — CoderAI launches it as a managed subprocess and proxies to it.

One kt backend can serve many families: **DeepSeek-V3/R1/V4, Kimi-K2/K2.5, Qwen3,
GLM-5/5.2, MiniMax**. Because it overlaps every other engine, it is selected **only** by
an explicit `backend: "kt"` pin or its configured `model_id` alias — never by a broad
name marker (that would collide with ds4/colibri/k3).

## Requirements

- **SGLang + kt-kernel installed out of band** (kt-kernel is a native build). CoderAI
  does not compile it per-request; with `auto_build` it attempts `pip install sglang`,
  but kt-kernel must be provided. Best throughput needs **AMX / AVX-512** CPUs.
- The HF model directory + the KT quantized weights (not downloaded by CoderAI).

## Enable & configure

Settings → **ktransformers (SGLang)** card, or `config.json` `"ktransformers"`:

| field | meaning |
|---|---|
| `enabled` | turn the engine on |
| `model_path` | HF model directory (SGLang `--model`) |
| `kt_weight_path` | KT quantized weights directory (`--kt-weight-path`) |
| `model_id` | id/alias that routes here + SGLang `--served-model-name` (default `ktransformers`) |
| `host` / `port` | SGLang bind host / port (`0` = auto-pick a free port) |
| `ctx` | context length (`--context-length`) |
| `extra_args` | extra `sglang.launch_server` flags (e.g. `--tp-size 1`) |
| `extra_env` | free-form `KEY=VALUE` env for the subprocess |
| `auto_build` | attempt `pip install sglang` if missing (heavy; off by default) |

On first use CoderAI runs, roughly:

```
python -m sglang.launch_server --host <h> --port <p> --model <model_path> \
       --kt-weight-path <kt_weight_path> --served-model-name <model_id> \
       --context-length <ctx> [extra_args]
```

then health-checks `/v1/models` and proxies `/v1/chat/completions` to it.

## Routing

Route to `kt` only via an explicit `backend: "kt"` model entry or a request whose model
matches the kt `model_id`. To run, e.g., DeepSeek-V4 on ktransformers instead of ds4, add
a models.json entry `{ "path": "…", "backend": "kt", "alias": "deepseek-v4-kt" }`.

## Notes

- Heaviest to set up of the four engines, but a single maintained backend covering many
  families — a good long-term option where AMX/multi-GPU serving is available.
- Managed like ds4: the subprocess is torn down on model eviction / shutdown.
