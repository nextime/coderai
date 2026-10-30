# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Host-RAM admission control for model loads.

A load is admitted to VRAM by estimate: :meth:`ensure_vram_for` sizes the model,
evicts what is in the way, and the load proceeds. Host RAM had no equivalent, and
the asymmetry is not cosmetic -- it is the difference between a clean refusal and
a dead engine.

What it cost, measured on a 54 GB production host serving four engines
(2026-10-29/30). Six model cards configured their big weights unquantized:
``load_in_4bit=False`` with ``component_quantization={'text_encoder': '4bit'}``,
which quantizes the text encoder and leaves the transformer at bf16; two video
cards asked for ``offload_strategy`` ``model``/``balanced``, which parks weights
on the host by definition. Requesting any of them staged tens of GB into host
RAM, and the kernel resolved it:

    Oct 29 23:03  Killed coderai-nvidia-gguf   anon-rss 46.3 GB
    Oct 30 03:10  Killed coderai-nvidia        anon-rss 47.7 GB
    Oct 30 03:36  Killed coderai-nvidia        anon-rss 31.6 GB
    Oct 30 03:49  Killed coderai-nvidia        anon-rss 32.7 GB
    front: engine#0 exited (code -9); respawning (1/5 in 120s)

The RAM watcher saw it and could not help: it logged ``RSS 50.6/43.0 GB (118%)
-- mitigation [gc+trim+evict_idle] -> 53.7 GB``. Its ladder can only release
memory that is already held; nothing in it can decline the load that is at that
moment streaming bf16 weights into the host. So every client saw SIGKILL --
reported as "crashes the engine", "won't load", and, for whatever else was in
flight during the 120-second respawn, as an unrelated failure.

This module is the missing half: size the host-RAM cost of a load BEFORE it
starts, and if it cannot fit, refuse it by name with the remedy attached. One
model then fails with an explanation instead of every model failing with a
corpse.

Deliberately pure arithmetic -- no manager, no torch, no /proc. The caller
measures and evicts; these functions only decide. That keeps the decision
testable without a GPU, which is the half that was never exercised before.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

# Head-room for the load itself, on top of whatever the weights park on the host.
# A shard-streaming load holds a shard buffer, the safetensors read window and the
# dequantisation scratch at once; measured between 1.2 and 1.8 GB across the HF and
# diffusers paths, so 2.0 GB covers it without pretending to be precise.
STAGING_GB = 2.0

# Host RAM never promised to a model, so the OS, the front and the sibling engines
# keep breathing room even when a load is admitted right up to the line.
DEFAULT_FLOOR_GB = 4.0

# Strategies that put the weights on the HOST by design, rather than as overflow.
# For these the host cost is the model's whole footprint, not the part that missed
# the GPU: that is what the operator asked for.
HOST_RESIDENT_STRATEGIES = frozenset({"model", "sequential", "cpu"})

# Strategies that keep every weight on the GPU, so a model too big for the card
# fails on VRAM (loudly, recoverably) and never reaches host RAM at all.
GPU_ONLY_STRATEGIES = frozenset({"none"})


class Need(NamedTuple):
    """The estimated host-RAM cost of a pending load, and where it came from."""

    gb: float
    basis: str          # "measured" | "host-resident" | "gpu-only" | "spill" | "unknown"
    detail: str         # one human-readable line, for the log and the refusal


class Budget(NamedTuple):
    """How much host RAM a load may take right now, and what limits it."""

    gb: float
    binding: str        # "kernel" | "cap" | "floor"
    detail: str


class Verdict(NamedTuple):
    ok: bool
    shortfall_gb: float


class HostRamExhausted(RuntimeError):
    """A load was refused because its host-RAM cost does not fit.

    Carries the numbers so an API layer can report them without re-deriving
    anything, and so a test can assert on them rather than on prose.
    """

    def __init__(self, model: str, need: Need, budget: Budget, remedy: str = ""):
        self.model = model
        self.need = need
        self.budget = budget
        self.remedy = remedy
        self.shortfall_gb = round(max(0.0, need.gb - budget.gb), 2)
        super().__init__(self.message())

    def message(self) -> str:
        msg = (f"'{self.model}' needs about {self.need.gb:.1f} GB of host RAM to "
               f"load ({self.need.detail}), and only {self.budget.gb:.1f} GB is "
               f"available ({self.budget.detail}) — short by "
               f"{self.shortfall_gb:.1f} GB. Refusing the load: going ahead would "
               f"exhaust host memory and the kernel would kill the engine, which "
               f"fails every other request too.")
        if self.remedy:
            msg += " " + self.remedy
        return msg


def estimate_need(*, total_need_gb: float, free_vram_gb: float,
                  measured_ram_gb: Optional[float] = None,
                  offload_strategy: Optional[str] = None,
                  no_ram: bool = False,
                  staging_gb: float = STAGING_GB) -> Need:
    """Host RAM a pending load will want, in GB.

    ``total_need_gb`` is the model's FULL footprint -- the placement-independent
    figure ``measured_vram_gb`` already records -- not the GPU-resident slice.

    The estimate follows what actually determines the cost:

    * ``measured_ram_gb`` wins outright. The loader records what a load really
      parked on the host (``of which X GB was offloaded to host RAM this load``),
      so once a model has loaded once there is no need to guess.
    * A host-resident offload strategy means the whole footprint lands on the
      host, because that is the point of asking for it.
    * A GPU-only strategy means none of it does: too big a model fails on VRAM.
    * Otherwise the host carries the SPILL -- whatever did not fit on the card.
      This is the case that killed the engine: a bf16 transformer on a card with
      other models resident spills nearly all of itself.

    An unknown footprint yields the staging allowance alone, flagged ``unknown``:
    a load whose size cannot be determined is never refused on a number nobody
    has, it just loses the protection.
    """
    staging = max(0.0, float(staging_gb or 0.0))
    strategy = (offload_strategy or "").strip().lower()

    if measured_ram_gb is not None and float(measured_ram_gb) > 0:
        held = float(measured_ram_gb)
        return Need(round(held + staging, 2), "measured",
                    f"{held:.1f} GB measured on host RAM on a previous load, "
                    f"plus {staging:.1f} GB to stage it")

    if no_ram or strategy in GPU_ONLY_STRATEGIES:
        return Need(round(staging, 2), "gpu-only",
                    f"loads GPU-only, so only the {staging:.1f} GB load buffer "
                    f"touches host RAM")

    total = max(0.0, float(total_need_gb or 0.0))
    if total <= 0:
        return Need(round(staging, 2), "unknown",
                    "footprint unknown, so only the load buffer is accounted for")

    if strategy in HOST_RESIDENT_STRATEGIES:
        return Need(round(total + staging, 2), "host-resident",
                    f"offload_strategy='{strategy}' keeps the whole "
                    f"{total:.1f} GB footprint on the host, plus "
                    f"{staging:.1f} GB to stage it")

    spill = max(0.0, total - max(0.0, float(free_vram_gb or 0.0)))
    return Need(round(spill + staging, 2), "spill",
                f"{total:.1f} GB footprint against {max(0.0, float(free_vram_gb or 0.0)):.1f} GB "
                f"free VRAM spills {spill:.1f} GB to the host, plus "
                f"{staging:.1f} GB to stage it")


def budget(*, kernel_available_gb: float, cap_gb: Optional[float] = None,
           fleet_held_gb: float = 0.0,
           floor_gb: float = DEFAULT_FLOOR_GB) -> Budget:
    """Host RAM a load may take right now, in GB.

    Two limits apply and the tighter one binds:

    * what the kernel says can be allocated (``MemAvailable``, which already
      counts the page cache it is willing to drop), and
    * the configured server-wide ceiling minus what the fleet already holds.

    ``floor_gb`` is then held back from whichever bound won, so admitting a load
    at the limit still leaves the OS and the sibling engines room to run. The
    result is clamped at zero: a fleet already over its cap has no budget, which
    is the correct answer rather than a negative one.
    """
    floor = max(0.0, float(floor_gb or 0.0))
    kernel = max(0.0, float(kernel_available_gb or 0.0))
    limit, binding, detail = kernel, "kernel", f"{kernel:.1f} GB available on the host"

    if cap_gb and float(cap_gb) > 0:
        headroom = float(cap_gb) - max(0.0, float(fleet_held_gb or 0.0))
        if headroom < limit:
            limit, binding = headroom, "cap"
            detail = (f"{headroom:.1f} GB left under the {float(cap_gb):.0f} GB "
                      f"server RAM cap, with the fleet holding "
                      f"{max(0.0, float(fleet_held_gb or 0.0)):.1f} GB")

    usable = limit - floor
    if usable <= 0:
        return Budget(0.0, "floor",
                      detail + f", all of it inside the {floor:.1f} GB kept free "
                               f"for the OS and the other engines")
    return Budget(round(usable, 2), binding,
                  detail + f", less {floor:.1f} GB kept free for the OS and the "
                           f"other engines")


def verdict(need: Need, bud: Budget) -> Verdict:
    """Whether ``need`` fits in ``bud``."""
    if need.gb <= bud.gb:
        return Verdict(True, 0.0)
    return Verdict(False, round(need.gb - bud.gb, 2))


def remedy_for(cfg: Optional[dict], basis: str = "") -> str:
    """One actionable sentence about what to change on the card, or "".

    Derived from the card itself, because the fix differs: an unquantized
    transformer wants quantizing, a host-resident offload strategy wants
    reconsidering, and a model that simply does not fit wants a smaller build.
    """
    cfg = cfg or {}
    comp = cfg.get("component_quantization") or {}
    if isinstance(comp, dict) and comp and not any(
            k in comp for k in ("transformer", "transformer_2", "unet")):
        covered = ", ".join(sorted(comp)) or "nothing"
        return ("This card quantizes only " + covered + ", so its transformer "
                "loads at full precision — add 'transformer' to "
                "component_quantization (or request the pre-quantized build of "
                "this model) and it will fit.")
    strategy = str(cfg.get("offload_strategy") or "").strip().lower()
    if strategy in HOST_RESIDENT_STRATEGIES:
        return (f"This card sets offload_strategy='{strategy}', which holds the "
                f"weights in host RAM by design — quantize it or give it a "
                f"strategy that keeps them on the GPU.")
    if basis == "spill" and not (cfg.get("load_in_4bit") or cfg.get("load_in_8bit")):
        return ("This card has no quantization set, so its weights load at full "
                "precision — quantize it, or request a pre-quantized build.")
    return ""
