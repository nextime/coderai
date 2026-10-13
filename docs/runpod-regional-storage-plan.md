# Per-region storage, so multi-region and a slim pod can coexist

Status: **built in v0.3.55** (2026-10-13), except where noted at the end.
The reference lives in `docs/runpod.md` §8; this file keeps the reasoning.

## The contradiction to remove

coderai already rents across regions. `data_centers` (account-wide, or per
model) is a list, and `_provision_one` expands the ranked cards into
(card, region) pairs so a capacity miss in one region falls through to the
next instead of failing the lease.

It also already knows how to keep weights and a venv off the container disk:
`network_volume_id` + `volume_mount_path` attach a volume, `volume_path` names
weights already on it, and `venv_on_volume` makes the first pod build the venv
there so every later pod starts in seconds from the dependency-free
`coderai-*-slim` image instead of pulling a 12-27 GB capability image.

The two cannot be used together. RunPod ties a network volume to one data
centre, and a pod elsewhere cannot attach it, so `_data_centers()` answers
with `[the volume's region]` and discards the whole list:

```python
pinned = getattr(self, "_volume_dc", "") or ""
if pinned:
    return [pinned]
```

That is correct for one volume — placing a pod where the attach would fail is
worse than not trying. But it means an install must choose: rent in several
regions and re-download everything onto every cold pod, or keep persistent
storage and accept a single region's capacity. On a card as contended as the
A40 that choice is the difference between a lease that fills and one that
waits.

The intended escape hatch was RunPod's **global volume** (region-independent,
beta Sept 2026), which coderai supports via `global_volume_id` and deliberately
does *not* let pin a region. It is not a usable answer: the type cannot be
created through the public API at all — `POST /v2/network-volumes` accepts only
`STANDARD|HIGH_PERFORMANCE` and demands a `dataCenter` — so it exists only for
an operator who made one by hand in the console, and the object-storage
semantics behind it (no locking, no atomic rename, eventual consistency) are
wrong for anything but read-mostly weights. Support for it stays; it stops
being the plan.

## What was built

**One volume per region, and the region chosen for the pod picks the volume.**

1. **Config.** A map alongside the single id, on both the account block and a
   model's `runpod` block:

   ```json
   {"runpod": {"network_volumes": {"EU-SE-1": "abc123", "EU-RO-1": "def456"},
               "volume_mount_path": "/workspace"}}
   ```

   `network_volume_id` keeps working and means what it means now. When the map
   is present it supersedes it. Per-model map overrides the account map, like
   every other runpod field.

2. **Placement.** `_data_centers()` stops collapsing to one region when a map is
   configured: the allowed regions are the intersection of the configured
   region list with the regions the map has a volume for — preserving the
   caller's order, so preference still ranks. A region with no volume is
   dropped and said so, rather than producing a pod that silently has no
   storage. With exactly one entry the behaviour is identical to today, which
   is how this ships without changing any existing install.

3. **Attachment.** The volume is resolved from the candidate's `_dc`, not once
   per pool. `volume_for(mcfg, account)` becomes `volume_for(mcfg, account,
   dc="")` and the one call inside `_create_with_fallback` passes
   `sel.get("_dc")`. `_data_center_for_volume` stays as the fallback path for a
   bare `network_volume_id`.

4. **Seeding.** A volume in a new region is empty, and the first pod there pays
   for everything — which is the cost the volume exists to remove. Two halves,
   and they are not the same problem:

   - **The venv.** `venv_bootstrap_script` already builds it under a lock with
     a `.ready` marker written last, so an empty regional volume self-heals:
     the first pod in that region builds it once and every pod after finds it.
     Nothing to add. The cost is one slow boot per region per profile change.
   - **The models.** These are large and should not be fetched N times. A seed
     operation per region, reusing what `models_transfer.py` already does
     (a HuggingFace repo id or a URL costs this install nothing and pulls at
     datacentre speed) — rent the cheapest *CPU-only* pod in the target region,
     attach that region's volume, pull the model onto it, terminate. Explicitly
     not a copy from region to region: egress is paid twice and the source is
     the registry either way.

   Exposed as an admin action and an endpoint, per (region, model), reporting
   what each region's volume already holds — because the honest answer to "why
   did this pod take 11 minutes" must be visible before the boot, not inferred
   after it.

5. **Reporting.** `pools_schedule_status` carries `network_volumes`, next to
   the `by_region` tally it already had, so a slow region is attributable.

## Where it landed

| Piece | Where |
|---|---|
| `network_volumes` on the account | `RunpodConfig.network_volumes` (`codai/config.py`) |
| …and per model | `RunpodModelConfig.network_volumes`, parsed by `_parse_volume_map` |
| Resolution | `volume_map_for()` / `volume_for(mcfg, account, dc="")` |
| Placement | `RunpodPodPool._data_centers()` + `_with_volumes()` |
| Attachment | `_create_with_fallback`, per candidate, from `sel["_dc"]` |
| Seeding | `RunpodPodPool.seed_region()`, `seed_region_volumes()`, `POST /admin/api/runpod/seed-volumes` |
| UI | account: *Per-region volumes* on the settings page; model: the same on its RunPod block |
| Tests | `tests/test_runpod_regional_volumes.py` (25) |

## Still open

- **Nothing verified against RunPod itself.** Every test is offline. The next
  multi-region install is the first real proof, and the first thing to read is
  whether `dataCenterId` and the map's key are spelled the same way.
- The seed rents a **GPU** pod, because that is the pod shape coderai knows how
  to boot and health-check. A CPU-only pod would be cheaper for a pure
  download; RunPod exposes those through a different create path, which is a
  separate piece of work.
- No per-region view of **what a volume already holds**, so "why was this boot
  slow" is still answered from the log rather than before the fact.

## What this does not change

- A lease's GPU narrowing (`gpu_types` / `allow_other_gpus`, v0.3.54) is
  independent of this and keeps composing with it: narrowing picks cards,
  this picks regions.
- No request-level control over regions or volumes. Where data may be stored is
  an operator's decision — several installs have a legal answer to it — and a
  caller must not be able to widen it. Same one-directional rule as the GPU
  narrowing, and for the same reason.
