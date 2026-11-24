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
    # spark2_5 (Spark-X2.5): fused QKV, attention output, head-wise attention
    # gate. Without these only the MLP would be adapted and attention left
    # untouched -- a materially weaker adapter, and `select_target_modules`
    # would not even see the self-attention block.
    "q_k_v_proj", "out_proj", "g_proj",
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
    #: ``"sft"`` (next-token supervised learning of the answer) or ``"decision"``
    #: (soft cross-entropy over the answer-letter logits at the final prompt
    #: position -- a calibrated classifier, no generation). The dataset must be
    #: encoded for the SAME objective; `text_lora_dataset.prepare` enforces it.
    objective: str = "sft"
    #: The candidate answer letters a decision model chooses between, for the
    #: record and for the serving contract. The per-example ``decision`` target
    #: is authoritative for training; this is the default/declared set.
    decision_letters: tuple = ()
    #: Which distribution the ``decision`` loss trains: ``"letters"`` (default --
    #: softmax over the answer-letter logits alone, the quantity ``/v1/decision``
    #: renormalises to and a gate thresholds on) or ``"vocab"`` (full-vocabulary
    #: softmax gathered at the letters, the pre-``"letters"`` behaviour). See
    #: :func:`decision_distillation_loss`.
    decision_loss_scope: str = "letters"
    #: An explicit override of which module-name tails to adapt, instead of
    #: :data:`LANGUAGE_SUFFIXES`. Empty means the default. Lets an operator pin a
    #: base's exact projection set (e.g. a spark2_5 adapter's target list).
    target_suffixes: tuple = ()


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
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    # THE KEPT POSITIONS ARE GATHERED FIRST, and this is a memory fix, not a
    # tidying one. The arithmetic is identical -- `ignore_index=-100` with
    # `reduction="mean"` is a mean over exactly these positions -- but the
    # cost is not. This used to `.contiguous()` the whole logits tensor and
    # hand all of it to cross_entropy, which upcasts to fp32 internally and
    # keeps that for the backward.
    #
    # On this workload the prompt is masked and the answer is not, so of a
    # ~5,000-token sequence about 392 positions are trained: the full-width
    # form paid for 5,000. At a 151,936-token vocabulary that is a 1.5 GB
    # copy, ~3 GB of fp32 log-softmax, ~3 GB saved for backward and ~3 GB of
    # gradient -- about 10 GB of transients per example, to produce a loss
    # over 7% of them. Measured consequence: 13 of 16 examples raised
    # OutOfMemoryError on a 48 GB card and the loop SKIPS an OOM, so the pod
    # reported `done` having trained 3, and the run called that 100%.
    #
    # Gathering with a boolean mask produces a compact [kept, vocab] tensor
    # directly, so nothing of full width is ever materialised or upcast.
    keep = shift_labels != -100
    if not bool(keep.any()):
        # No trainable position. Returning a plain zero would detach the
        # graph and make `backward()` raise; this keeps it and contributes
        # nothing. `encode` already refuses such an example, so this is a
        # guard rather than a path.
        return (shift_logits.sum() * 0.0).to(torch.float32)
    picked = shift_logits[keep]                  # [kept, vocab]
    target = shift_labels[keep]                  # [kept]
    return F.cross_entropy(picked.float(), target, reduction="mean")


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


def decision_distillation_loss(logits, letter_token_ids, target_probs,
                               scope: str = "letters"):
    """Soft cross-entropy over the answer-letter logits at the LAST position.

    ``logits`` is ``[1, seq, vocab]`` -- or ``[1, 1, vocab]`` when the forward
    was told to keep only the last position. The decision example is prompt-only
    and scored at the final prompt token, so ``logits[0, -1, :]`` is the model's
    distribution over the token it would generate first -- the answer letter.
    The loss is ``-sum(p_i * log q_i)``: the cross-entropy from the target ``p``
    to the model's letter distribution ``q``, equal to ``KL(p || q)`` up to the
    target's own (constant) entropy, so minimising it matches ``q`` to
    ``target_probs`` and bottoms out at ``H(target)`` when they coincide.

    ``scope`` chooses WHICH ``q`` is trained, and it has to be the one the
    serving side reads back, or the run optimises one quantity while the gate
    thresholds another:

    * ``"letters"`` (default) -- softmax over the letter logits ALONE. This is
      exactly what ``/v1/decision`` computes: it takes each letter's probability
      and renormalises over the letters that were found, and renormalising a
      softmax over a subset is identically the softmax restricted to that subset.
      So this trains the calibrated letter ratios a confidence gate uses, and a
      perfect fit reaches ``H(target)`` -- the clean "aligned" signal.
    * ``"vocab"`` -- softmax over the FULL vocabulary, gathered at the letters.
      That also drives the off-menu mass toward zero, so it never reaches
      ``H(target)`` even in theory and spends some of a low rank's capacity on
      "emit only letters" instead of the ratios. Kept so an adapter trained
      before ``"letters"`` existed can be reproduced unchanged.
    """
    import torch
    import torch.nn.functional as F
    last = logits[0, -1, :].float()                      # [vocab]
    ids = torch.as_tensor(letter_token_ids, device=last.device, dtype=torch.long)
    if str(scope) == "vocab":
        logp = F.log_softmax(last, dim=-1).index_select(0, ids)   # [n_letters]
    else:
        # The letters alone -- the quantity /v1/decision renormalises to.
        letter_logits = last.index_select(0, ids)                 # [n_letters]
        logp = F.log_softmax(letter_logits, dim=-1)               # [n_letters]
    p = torch.as_tensor(target_probs, device=last.device, dtype=logp.dtype)
    p = p / p.sum()
    return -(p * logp).sum()


def scaled_backward_decision(logits, letter_token_ids, target_probs,
                             weight: float, accumulation_steps: int,
                             scope: str = "letters"):
    """The decision counterpart of :func:`scaled_backward`.

    Same weight/accumulation scaling so a decision run mixes with the rest of the
    fleet machinery unchanged; returns the unscaled soft cross-entropy for the
    log. ``scope`` is forwarded to :func:`decision_distillation_loss`.
    """
    from codai.api.text_lora_dataset import loss_scale
    loss = decision_distillation_loss(logits, letter_token_ids, target_probs,
                                      scope=scope)
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

def describe_loaded_base(model) -> Dict[str, Any]:
    """What the card is actually holding, as data rather than a log line.

    STRUCTURED because the log is the one place this cannot be read. A
    training pod's stdout is not collected anywhere -- there is no public
    pod-log API and the orchestrator never sees it -- so a `print` here is
    written for nobody. The same mistake was made twice in one day: the
    reason a fleet trained 4 of every 16 examples could not be established
    from outside the pod, and the first attempt to fix that put the answer
    straight back into stdout.

    It matters because the training loop turns an OutOfMemoryError into a
    SKIPPED example. A base that silently failed to quantize, or that
    `device_map="auto"` spilled onto the CPU because it did not fit, does not
    announce itself: it looks exactly like a pod that did less work, and
    answers `done`.
    """
    import collections

    import torch
    dtypes: collections.Counter = collections.Counter()
    devices: collections.Counter = collections.Counter()
    n_bytes = 0
    for p in model.parameters():
        dtypes[str(p.dtype)] += 1
        devices[str(p.device)] += 1
        n_bytes += p.numel() * p.element_size()
    # bnb keeps 4-bit weights as uint8 blocks, so the dtype is the tell.
    quantized = any(("uint8" in d) or ("int8" in d) for d in dtypes)
    out: Dict[str, Any] = {
        "param_bytes": int(n_bytes),
        "param_gb": round(n_bytes / 1e9, 2),
        "dtypes": dict(dtypes),
        "devices": dict(devices),
        "quantized_4bit": bool(quantized),
        "cpu_params": int(sum(v for k, v in devices.items() if "cpu" in k)
                          if len(devices) > 1 else 0),
    }
    if torch.cuda.is_available():
        out["vram_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        out["vram_allocated_gb"] = round(
            torch.cuda.memory_allocated() / 1e9, 2)
        out["card"] = str(torch.cuda.get_device_properties(0).name)
    out["line"] = (
        f"base loaded: {out['param_gb']}GB of parameters across "
        f"{out['devices']}, dtypes {out['dtypes']}, 4-bit="
        + ("yes" if quantized else "NO - THE BASE IS NOT QUANTIZED")
        + (f", {out.get('card')} has {out.get('vram_total_gb')}GB with "
           f"{out.get('vram_allocated_gb')}GB allocated"
           if "vram_total_gb" in out else ""))
    return out


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
    # PUT THE BIG TENSORS BACK IN THE COMPUTE DTYPE. `prepare_model_for_kbit_
    # training` does, in its own words, "cast all non INT8 parameters to
    # fp32" -- every parameter that is not a `Params4bit`, whatever its size.
    # Upcasting the NORMS is the point of that and they are kept: 1-D, a few
    # MB, and fp32 is what makes layer norm stable in a 4-bit run. Upcasting
    # `embed_tokens` and `lm_head` is collateral, and at a 151,936-token
    # vocabulary it is ruinous twice over:
    #
    #   * the weights go from 3.1 GB to 6.2 GB -- measured, a 27B base that
    #     should occupy ~14 GB reported 22.68 GB with 867 fp32 tensors;
    #   * and an fp32 `lm_head` produces fp32 LOGITS. At 4,703 tokens that is
    #     [1, 4703, 151936] = 2.86 GB for one forward and as much again for
    #     its gradient, against 1.43 GB apiece in bf16, while an fp32
    #     residual stream doubles every checkpointed activation besides.
    #
    # On a 48 GB card that was the difference between fitting and not: with
    # 23.09 GB resident, 12 of every 16 examples raised OutOfMemoryError --
    # and the loop SKIPS an OOM, so the pod answered `done` having trained 4
    # and the run called it 100%. The loss is still computed in fp32 (the
    # gathered logits are cast in `masked_cross_entropy`), so this costs no
    # precision where precision is load-bearing.
    _compute = getattr(torch, settings.compute_dtype)
    _recast = 0
    for _name, _p in model.named_parameters():
        if _p.dtype == torch.float32 and _p.ndim > 1:
            _p.data = _p.data.to(_compute)
            _recast += 1
    if _recast:
        print(f"[text-lora] put {_recast} parameter tensor(s) back in "
              f"{settings.compute_dtype} after the kbit prepare upcast them "
              f"to fp32 (norms left in fp32)", flush=True)
    # Gradient checkpointing: a 27B base at 8192 tokens does not otherwise fit
    # alongside its activations on one card, and recomputation is cheaper than
    # the OOM it avoids.
    model.config.use_cache = False
    # NOT SWALLOWED. Without checkpointing a 27B base at the 8192-token cap
    # does not fit its own activations on one 48GB card, so this failing
    # quietly turns every long example into an OOM -- which the loop then
    # SKIPS, so the run reports "done" having trained a handful of the
    # examples it was asked for. And without `enable_input_require_grads`
    # the checkpointed segments take no grad at their input, which costs no
    # memory and simply trains nothing. Either way the run is worthless and
    # the operator must be told at setup, not left to infer it from the
    # averaging weights.
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    except Exception as exc:
        raise RuntimeError(
            f"could not enable gradient checkpointing on {base!r}: "
            f"{type(exc).__name__}: {exc}. Training without it OOMs on every "
            f"long example and those are skipped, not raised, so the run "
            f"would look like it worked") from exc

    linears = linear_module_names(model)
    # Only pass `suffixes` when the request pinned an override: the default call
    # uses select_target_modules' own LANGUAGE_SUFFIXES default, which is the
    # historical signature.
    if settings.target_suffixes:
        suffixes = tuple(settings.target_suffixes)
        targets = select_target_modules(linears, suffixes=suffixes)
    else:
        suffixes = LANGUAGE_SUFFIXES
        targets = select_target_modules(linears)
    if not targets:
        raise RuntimeError(
            f"no language projections found to adapt in {base!r}: "
            f"{len(linears)} linear modules, none matching "
            f"{', '.join(suffixes[:4])}… — this base names its layers "
            "differently and needs an explicit target list (set "
            "`target_modules` on the request)")
    line = describe_selection(linears, targets)
    print(f"[text-lora] {line}", flush=True)

    try:
        _fp = describe_loaded_base(model)
        print(f"[text-lora] {_fp['line']}", flush=True)
        if _fp.get("cpu_params"):
            print(f"[text-lora] WARNING: {_fp['cpu_params']} parameter "
                  f"tensor(s) are on the CPU, so the base did not fit on the "
                  f"card. Every long example will OOM and the loop SKIPS an "
                  f"OOM, so the round reports success having trained almost "
                  f"nothing", flush=True)
    except Exception as exc:
        # Never fatal: this is instrumentation, and refusing to train because
        # the report failed would be worse than training without it.
        print(f"[text-lora] (could not describe the loaded base: {exc})",
              flush=True)
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
    # REPORTED BEFORE THE FIRST STEP, and through the progress callback rather
    # than only in the return value: if every example OOMs the run still ends
    # normally, but if the pod wedges or is killed the return value never
    # arrives, and this is exactly the case where the footprint is what one
    # wants to know. The pod puts it in `/status`, which is the only channel
    # out of a training pod.
    _footprint: Dict[str, Any] = {}
    try:
        _footprint = describe_loaded_base(model)
        if on_progress is not None:
            on_progress(step=0, status="loaded", base=_footprint,
                        message=_footprint.get("line", ""))
    except Exception as exc:
        print(f"[text-lora] (could not report the loaded base: {exc})",
              flush=True)
    if init_adapter:
        # A distributed round starts from the PREVIOUS round's average, not from
        # a fresh adapter: that is the whole mechanism. Loaded into the existing
        # LoRA modules by name, and a key that does not fit is fatal rather than
        # skipped -- silently starting from random weights would look exactly
        # like a round that trained badly.
        #
        # THROUGH PEFT, not torch's load_state_dict, because the two naming
        # conventions are not the same one. An adapter ON DISK has no adapter
        # name in its keys -- `...lora_A.weight`, which is what
        # `save_pretrained` writes -- while the LIVE module's state dict does:
        # `...lora_A.default.weight`. `set_peft_model_state_dict` inserts that
        # name on the way in; comparing the file's keys against
        # `model.state_dict()` cannot ever match, so the check below reported
        # EVERY key missing and raised. Measured in production: 512 of 512 keys
        # "this model has no place for", on an adapter this very function had
        # just saved, which killed all four surviving pods at round 1 of a
        # 16-round run while round 0 had looked perfect. A distributed run
        # could not pass its first exchange at all.
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        incoming = load_file(init_adapter)
        res = set_peft_model_state_dict(model, incoming)
        unexpected = sorted(getattr(res, "unexpected_keys", None) or [])
        if unexpected:
            # DELIBERATELY WORDED DIFFERENTLY from the check this replaced.
            # That one said "has N key(s) this model has no place for" and so
            # did this one, which made the two indistinguishable in a log --
            # so when every pod in a fleet failed with the old message, there
            # was no way to tell a genuine mismatch from a pod running the
            # PRE-FIX IMAGE. That is exactly what happened: an explicit image
            # tag in the config kept serving the old code for five hours
            # while the message looked like a real key disagreement, and the
            # time went on a dataset theory instead of on the config.
            raise RuntimeError(
                f"PEFT refused {len(unexpected)} of {len(incoming)} tensors "
                f"from {os.path.basename(init_adapter)}, e.g. "
                f"{unexpected[:3]} -- this adapter was trained against a "
                f"different base or a different target_modules/rank. "
                f"(A failure here reading 'no place for' instead is the "
                f"pre-0.6.69 loader, which means this pod is running an "
                f"old trainer image.)")
        print(f"[text-lora] resumed from {os.path.basename(init_adapter)} "
              f"({len(incoming)} tensors)", flush=True)
        del incoming, res
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

        if getattr(ex, "objective", "sft") == "decision":
            # Prompt-only: no labels tensor, scored at the final position.
            _letters = ex.letter_token_ids
            _probs = ex.target_probs
            _scope = getattr(settings, "decision_loss_scope", "letters")

            def _one():
                # Only the last position's logits matter for a decision, so ask
                # the head to compute that one row instead of seq x vocab (a
                # 3.4k-token prompt is 3.4k x 131k logits otherwise). A base
                # whose forward does not accept the hint raises TypeError or
                # ignores it; either way the slice below stays correct.
                try:
                    out = model(input_ids=ids, logits_to_keep=1)
                except TypeError:
                    out = model(input_ids=ids)
                return scaled_backward_decision(
                    out.logits, _letters, _probs, ex.weight, accum, _scope)
        else:
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
            "asked": len(encoded),
            "base": _footprint,
            "mean_loss": mean, "final_loss_ema": progress.loss_ema}
