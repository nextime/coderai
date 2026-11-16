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

"""QLoRA for a causal LM: which weights are adapted, and the step itself.

Three things here, kept apart from the dataset module because they are the
parts that need torch, and apart from the job/route because they are the parts
worth testing numerically.

**Which modules get an adapter.** The bases are often multimodal, and a LoRA
meant to teach a text task must not touch the vision tower -- adapting it spends
rank on a tower that sees no images during training, and degrades the one thing
the base already did well. Selection returns FULL module names rather than the
usual suffix list (``["q_proj", "k_proj", ...]``), because a suffix list matches
the same names inside a vision encoder: on a model whose tower calls its
projections ``q_proj`` too, "text only" silently becomes "everything".

**The step.** Cross-entropy over the labels the dataset module produced, with
``ignore_index=-100``, scaled by the example's weight over a FIXED divisor. The
masked positions then contribute exactly nothing -- not approximately: the
gradient of an ignored position is zero, which is testable and is tested.

**Surviving the run.** An example that will not fit is skipped rather than
allowed to kill the process, and progress is checkpointed as the adapter plus a
step counter, so a pod that dies resumes instead of restarting.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: Module-name fragments that mean "this is the vision side". Matched on the
#: dotted path, so `visual.blocks.0.attn.q_proj` is excluded by `visual` while
#: `model.layers.0.self_attn.q_proj` is not.
VISION_MARKERS = (
    "visual", "vision_tower", "vision_model", "vision_encoder",
    "image_encoder", "multi_modal_projector", "mm_projector", "mmproj",
    "merger", "patch_embed", "image_newline", "audio_tower",
)

#: The projections a text LoRA adapts: attention and MLP. Deliberately not the
#: embeddings or the LM head -- adapting the head on a 250k vocab dwarfs the
#: adapter and changes the output distribution directly rather than through the
#: representation, which is not what rank-16 LoRA is for.
LANGUAGE_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    # Fused variants, for bases that pack qkv or the MLP gate+up together.
    "qkv_proj", "gate_up_proj", "qkv", "wqkv",
)


def is_vision_module(name: str) -> bool:
    """True when this dotted module path belongs to a vision/audio encoder."""
    parts = name.split(".")
    return any(any(m == p or m in p for m in VISION_MARKERS) for p in parts)


def select_target_modules(named_linears: Iterable[str], *,
                          suffixes: Sequence[str] = LANGUAGE_SUFFIXES,
                          ) -> List[str]:
    """Full names of the linear modules a text LoRA should adapt.

    Takes the names of the model's Linear modules (``n for n, m in
    model.named_modules() if isinstance(m, nn.Linear)``) and keeps the language
    projections. Full names, not suffixes: see the module docstring.
    """
    out = []
    for name in named_linears:
        if is_vision_module(name):
            continue
        tail = name.rsplit(".", 1)[-1]
        if tail in suffixes:
            out.append(name)
    return out


def linear_module_names(model) -> List[str]:
    """The model's Linear (and 4-bit Linear) module names, in declaration order."""
    names = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if cls in ("Linear", "Linear4bit", "Linear8bitLt", "LoraLinear4bit",
                   "NF4Linear", "Params4bitLinear"):
            names.append(name)
    return names


def describe_selection(all_linears: Sequence[str],
                       selected: Sequence[str]) -> str:
    """A line for the log: what was adapted and what was deliberately left out.

    Printed because "the vision tower was left alone" is a claim, and a claim
    about 700 modules should be countable rather than trusted.
    """
    vision = [n for n in all_linears if is_vision_module(n)]
    skipped_lang = [n for n in all_linears
                    if not is_vision_module(n) and n not in set(selected)]
    return (f"LoRA targets {len(selected)} linear modules; left alone: "
            f"{len(vision)} vision/audio, {len(skipped_lang)} other language "
            f"(embeddings, head, norms)")


# --------------------------------------------------------------------------- #
# the configs
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class QLoraSettings:
    """The knobs, with the defaults a 27B-class base wants on one card."""

    rank: int = 16
    alpha: int = 32                 # 2x rank: the usual pairing, scale = 2
    dropout: float = 0.05
    learning_rate: float = 1e-4
    max_len: int = 8192
    accumulation_steps: int = 8
    #: 4-bit NF4 with bf16 compute. nf4 over fp4 because it is the
    #: information-theoretically better fit for normally-distributed weights,
    #: and double quantization saves ~0.4 GB on a 27B at no measurable cost.
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"
    enable_thinking: bool = False
    seed: int = 42


def bnb_config(st: QLoraSettings):
    """The ``BitsAndBytesConfig`` for a 4-bit base."""
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=st.quant_type,
        bnb_4bit_use_double_quant=bool(st.double_quant),
        bnb_4bit_compute_dtype=getattr(torch, st.compute_dtype))


def lora_config(st: QLoraSettings, target_modules: Sequence[str]):
    """The ``LoraConfig``, with the targets already chosen by name."""
    from peft import LoraConfig
    return LoraConfig(
        r=int(st.rank), lora_alpha=int(st.alpha), lora_dropout=float(st.dropout),
        bias="none", task_type="CAUSAL_LM",
        target_modules=list(target_modules))


# --------------------------------------------------------------------------- #
# the step
# --------------------------------------------------------------------------- #

def masked_cross_entropy(logits, labels):
    """Next-token cross-entropy over the un-ignored positions.

    Shifted here rather than by the caller: ``logits[i]`` predicts
    ``labels[i+1]``, and an off-by-one in the shift trains the model to predict
    the token it was just given -- which lowers the loss beautifully and teaches
    nothing.

    ``reduction="mean"`` over the kept positions, so an example contributes its
    MEAN token loss: a 40-token answer and a 4000-token answer weigh the same
    before the example's own weight is applied. That is what a per-example
    weight means; summing would make length the weight.
    """
    import torch
    import torch.nn.functional as F
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1), ignore_index=-100, reduction="mean")


def scaled_backward(logits, labels, weight: float, accumulation_steps: int):
    """Loss for one example, scaled, backward-ed. Returns the unscaled loss.

    The unscaled value is what goes in the log: a reader comparing loss across
    examples of different weights needs the model's actual loss, not a number
    that moves with the curation.
    """
    from codai.api.text_lora_dataset import loss_scale
    loss = masked_cross_entropy(logits, labels)
    (loss * loss_scale(weight, accumulation_steps)).backward()
    return float(loss.detach())


class OomSkipped(Exception):
    """This example did not fit; the run continues without it."""


def step_with_oom_guard(fn, *, on_skip=None):
    """Run one example's step, turning an OOM into a skip rather than a death.

    A single long example must not end a run that has been going for hours --
    that is the failure that wastes a pod and, in a multi-pod round, keeps every
    other pod waiting for an average that will never come.
    """
    import torch
    try:
        return fn()
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        if on_skip is not None:
            on_skip(exc)
        raise OomSkipped(str(exc)[:200])


# --------------------------------------------------------------------------- #
# surviving a restart
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class Progress:
    """Where a run is, in a form that survives the process.

    The adapter is the state that matters; the counters say what has already
    been seen so a resume does not re-teach the same examples first.
    """

    step: int = 0
    examples_seen: int = 0
    round: int = 0
    skipped_oom: int = 0
    loss_ema: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def save_checkpoint(directory: str, model, progress: Progress) -> str:
    """Adapter + counters, written so an interrupted write cannot be read back.

    The JSON goes to a temp file and is renamed: a checkpoint half-written by a
    pod that died mid-save would otherwise be loaded on resume and fail to
    parse, turning a recoverable interruption into a lost run.
    """
    os.makedirs(directory, exist_ok=True)
    model.save_pretrained(directory)
    tmp = os.path.join(directory, ".progress.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(progress.as_dict(), fh)
    os.replace(tmp, os.path.join(directory, "progress.json"))
    return directory


def load_progress(directory: str) -> Progress:
    """The counters from a checkpoint, or a fresh zero state."""
    path = os.path.join(directory, "progress.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
    except Exception:
        return Progress()
    known = {f.name for f in dataclasses.fields(Progress)}
    return Progress(**{k: v for k, v in data.items() if k in known})


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

def load_base_and_adapter(base: str, settings: QLoraSettings):
    """The 4-bit base with a LoRA on its language projections.

    Returns ``(peft_model, selection_line)``. Separated from the loop so the
    loop can be exercised against a small real model, and so a failure to load
    says which half failed.
    """
    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb_config(settings),
        dtype=getattr(torch, settings.compute_dtype),
        device_map="auto", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    # Gradient checkpointing: a 27B base at 8192 tokens does not otherwise fit
    # alongside its activations on one card, and recomputation is cheaper than
    # the OOM it avoids.
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    except Exception:
        pass

    linears = linear_module_names(model)
    targets = select_target_modules(linears)
    if not targets:
        raise RuntimeError(
            f"no language projections found to adapt in {base!r}: "
            f"{len(linears)} linear modules, none matching "
            f"{', '.join(LANGUAGE_SUFFIXES[:4])}… — this base names its layers "
            "differently and needs an explicit target list")
    line = describe_selection(linears, targets)
    print(f"[text-lora] {line}", flush=True)
    peft_model = get_peft_model(model, lora_config(settings, targets))
    try:
        peft_model.print_trainable_parameters()
    except Exception:
        pass
    return peft_model, line


def run_training(*, base: str, encoded: Sequence[Any], settings: QLoraSettings,
                 out_dir: str, on_progress=None, check=None,
                 resume: bool = True, init_adapter: str = "") -> Dict[str, Any]:
    """Train the adapter and save it. Returns a summary.

    One example per forward: these sequences reach 8192 tokens and a 27B 4-bit
    base has no room for a padded batch, so the batch is made by ACCUMULATION
    rather than by padding. That also means no padding tokens exist to be
    masked wrongly, and an example's loss is exactly its own.
    """
    import random

    import torch

    random.seed(settings.seed)
    torch.manual_seed(settings.seed)

    model, _line = load_base_and_adapter(base, settings)
    if init_adapter:
        # A distributed round starts from the PREVIOUS round's average, not from
        # a fresh adapter: that is the whole mechanism. Loaded into the existing
        # LoRA modules by name, and a key that does not fit is fatal rather than
        # skipped -- silently starting from random weights would look exactly
        # like a round that trained badly.
        from safetensors.torch import load_file
        incoming = load_file(init_adapter)
        own = dict(model.state_dict())
        missing = [k for k in incoming if k not in own]
        if missing:
            raise RuntimeError(
                f"{init_adapter} has {len(missing)} key(s) this model has no "
                f"place for, e.g. {missing[:3]} -- it was trained against a "
                f"different base or target set")
        loaded = model.load_state_dict(incoming, strict=False)
        print(f"[text-lora] resumed from {os.path.basename(init_adapter)} "
              f"({len(incoming)} tensors)", flush=True)
        del incoming, own, loaded
    progress = load_progress(out_dir) if resume else Progress()
    if progress.step:
        print(f"[text-lora] resuming from step {progress.step} "
              f"({progress.examples_seen} examples already seen)", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=settings.learning_rate)

    order = list(range(len(encoded)))
    random.Random(settings.seed).shuffle(order)
    # Resume skips what was already seen rather than re-teaching it first: the
    # order is seeded, so "already seen" means the same examples on every run.
    order = order[progress.examples_seen:]

    accum = max(1, int(settings.accumulation_steps))
    losses: List[float] = []
    model.train()
    for i, idx in enumerate(order, 1):
        if check is not None:
            check()
        ex = encoded[idx]
        ids = torch.tensor([ex.input_ids], device=model.device)
        labels = torch.tensor([ex.labels], device=model.device)

        def _one():
            out = model(input_ids=ids)
            return scaled_backward(out.logits, labels, ex.weight, accum)

        try:
            loss = step_with_oom_guard(
                _one, on_skip=lambda e: print(
                    f"[text-lora] example (line {ex.line}, {len(ex.input_ids)} "
                    f"tokens) did not fit; skipped", flush=True))
        except OomSkipped:
            progress.skipped_oom += 1
            continue

        losses.append(loss)
        progress.examples_seen += 1
        progress.loss_ema = (loss if progress.loss_ema is None
                             else 0.98 * progress.loss_ema + 0.02 * loss)

        if i % accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            progress.step += 1
            if on_progress is not None and progress.step % 5 == 0:
                on_progress(step=progress.step, status="training",
                            message=(f"step {progress.step}, loss "
                                     f"{progress.loss_ema:.4f}"))
            # Checkpointed on a step boundary, so what is saved is a state the
            # optimizer agreed with rather than a half-accumulated one.
            if progress.step % 50 == 0:
                save_checkpoint(out_dir, model, progress)

    # Flush a partial accumulation: dropping it would silently discard the last
    # few examples of every run, and of every shard in a distributed one.
    if len(losses) % accum:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        progress.step += 1

    save_checkpoint(out_dir, model, progress)
    mean = sum(losses) / len(losses) if losses else None
    return {"steps": progress.step, "examples_trained": len(losses),
            "skipped_oom": progress.skipped_oom,
            "mean_loss": mean, "final_loss_ema": progress.loss_ema}
