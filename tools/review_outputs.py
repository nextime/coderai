#!/usr/bin/env python3
"""
Township Fighters — Output Review & LoRA Training UI

A lightweight web UI for reviewing generated characters, environments, and
videos, collecting good/bad ratings, and exporting approved images as a LoRA
training dataset.

Usage:
    python tools/review_outputs.py [--out-dir ./township_output] [--port 7860]

Then open http://localhost:7860 in your browser.

LoRA export creates:
    <out-dir>/lora_dataset/
        images/          ← approved images
        metadata.jsonl   ← caption per image (for dreambooth-style training)
        train_lora.sh    ← ready-to-run training command

Requirements:
    pip install diffusers accelerate peft  (for training)
    pip install Pillow                      (for thumbnail generation, usually present)
"""

import argparse
import base64
import http.server
import io
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Feedback persistence
# ─────────────────────────────────────────────────────────────────────────────

FEEDBACK_FILE = "feedback.json"


def _feedback_path(out_dir: Path) -> Path:
    return out_dir / FEEDBACK_FILE


def load_feedback(out_dir: Path) -> dict:
    p = _feedback_path(out_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"version": 1, "items": {}}


def save_feedback(out_dir: Path, data: dict):
    _feedback_path(out_dir).write_text(json.dumps(data, indent=2))


def set_rating(out_dir: Path, rel_path: str, rating: str, note: str = ""):
    data = load_feedback(out_dir)
    data["items"][rel_path] = {
        "rating": rating,   # "good" | "bad" | "skip"
        "note": note,
        "timestamp": int(time.time()),
    }
    save_feedback(out_dir, data)


# ─────────────────────────────────────────────────────────────────────────────
# Output discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_outputs(out_dir: Path) -> dict:
    """Return structured inventory of everything in the output directory."""
    inv = {"characters": {}, "environments": {}, "videos": []}

    chars_dir = out_dir / "characters"
    if chars_dir.exists():
        for char in sorted(chars_dir.iterdir()):
            if not char.is_dir():
                continue
            meta_file = char / "meta.json"
            meta = {}
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                except Exception:
                    pass
            images = sorted(char.glob("ref_*.png")) + sorted(char.glob("ref_*.jpg"))
            inv["characters"][char.name] = {
                "meta": meta,
                "images": [str(p.relative_to(out_dir)) for p in images],
            }

    envs_dir = out_dir / "environments"
    if envs_dir.exists():
        for env in sorted(envs_dir.iterdir()):
            if not env.is_dir():
                continue
            meta_file = env / "meta.json"
            meta = {}
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                except Exception:
                    pass
            images = sorted(env.glob("ref_*.png")) + sorted(env.glob("ref_*.jpg"))
            inv["environments"][env.name] = {
                "meta": meta,
                "images": [str(p.relative_to(out_dir)) for p in images],
            }

    videos_dir = out_dir / "videos"
    if videos_dir.exists():
        clips = sorted(videos_dir.glob("*_clip*.mp4"))
        finals = [p for p in sorted(videos_dir.glob("*.mp4")) if p not in clips]
        inv["videos"] = [str(p.relative_to(out_dir)) for p in finals + clips]

    return inv


# ─────────────────────────────────────────────────────────────────────────────
# LoRA training export
# ─────────────────────────────────────────────────────────────────────────────

def export_lora_dataset(out_dir: Path, base_model: Optional[str] = None,
                        steps: int = 500, lr: str = "1e-4") -> dict:
    """
    Collect all "good"-rated images + their prompts and write a
    dreambooth-compatible dataset under <out-dir>/lora_dataset/.
    Returns a summary dict.
    """
    feedback = load_feedback(out_dir)
    good = {k: v for k, v in feedback["items"].items()
            if v.get("rating") == "good"}

    lora_dir = out_dir / "lora_dataset"
    imgs_dir = lora_dir / "images"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    meta_lines = []
    copied = 0
    skipped = 0

    for rel_path, fb in sorted(good.items()):
        src = out_dir / rel_path
        if not src.exists() or not rel_path.lower().endswith((".png", ".jpg", ".jpeg")):
            skipped += 1
            continue

        # Build a caption from the meta.json stored alongside the image
        parts = Path(rel_path).parts   # e.g. ("characters", "khumalo", "ref_00.png")
        caption = fb.get("note", "").strip()
        if not caption and len(parts) >= 2:
            category = parts[0]        # "characters" or "environments"
            name = parts[1]
            # Look for meta.json
            meta_path = out_dir / category / name / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    caption = meta.get("prompt", "") or meta.get("description", "")
                except Exception:
                    pass
            if not caption:
                caption = f"{name.replace('_', ' ')}, African township fighter, cinematic"

        dest_name = rel_path.replace("/", "_").replace("\\", "_")
        dest = imgs_dir / dest_name
        shutil.copy2(src, dest)
        meta_lines.append(json.dumps({
            "file_name": f"images/{dest_name}",
            "text": caption,
        }))
        copied += 1

    if not meta_lines:
        return {"ok": False, "error": "No good-rated images found. Rate some images first."}

    (lora_dir / "metadata.jsonl").write_text("\n".join(meta_lines) + "\n")

    # Detect any existing LoRA to extend
    existing_lora = _find_existing_lora(lora_dir)

    # Write a ready-to-run training script
    model = base_model or "stabilityai/stable-diffusion-xl-base-1.0"
    _write_train_script(lora_dir, model, existing_lora, steps=steps, lr=lr)

    result = {
        "ok": True,
        "dataset_dir": str(lora_dir),
        "images": copied,
        "skipped": skipped,
        "train_script": str(lora_dir / "train_lora.sh"),
        "metadata": str(lora_dir / "metadata.jsonl"),
    }
    if existing_lora:
        result["extending"] = str(existing_lora)
    return result


def _find_existing_lora(lora_dir: Path) -> Optional[Path]:
    """
    Return the most recent LoRA weights to extend from, checking in order:
    1. Latest checkpoint-NNNN/ subdirectory inside lora_weights/
    2. Most recently modified .safetensors file inside lora_weights/
    """
    weights_dir = lora_dir / "lora_weights"
    if not weights_dir.exists():
        return None

    # Prefer the highest-numbered checkpoint directory (resumable mid-training)
    checkpoints = sorted(
        [d for d in weights_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1])
    )
    if checkpoints:
        return checkpoints[-1]

    # Fall back to the most recently modified .safetensors file
    safetensors = sorted(
        weights_dir.glob("*.safetensors"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return safetensors[0] if safetensors else None


def _write_train_script(lora_dir: Path, base_model: str,
                        existing_lora: Optional[Path] = None,
                        steps: int = 500, lr: str = "1e-4"):
    """
    Write train_lora.sh.
    - If existing_lora is a checkpoint-NNNN/ dir  → resume via --resume_from_checkpoint
    - If existing_lora is a .safetensors file      → initialize LoRA from it, continue training
    - If None                                       → fresh LoRA from scratch
    """
    weights_dir = lora_dir / "lora_weights"

    if existing_lora and existing_lora.is_dir():
        # Mid-training checkpoint: trainer can resume exactly
        resume_flag = f'--resume_from_checkpoint="{existing_lora}"'
        extend_note = f"# Resuming from checkpoint: {existing_lora}"
        init_flag = ""
    elif existing_lora and existing_lora.suffix == ".safetensors":
        # Completed LoRA: load its adapter weights as starting point.
        # The diffusers trainer supports --lora_model_name_or_path for this.
        resume_flag = ""
        init_flag = f'--lora_model_name_or_path="{existing_lora}" \\'
        extend_note = f"# Extending existing LoRA: {existing_lora}"
    else:
        resume_flag = ""
        init_flag = ""
        extend_note = "# Fresh LoRA training from scratch"

    resume_line = f"  {resume_flag} \\\n" if resume_flag else ""
    init_line   = f"  {init_flag}\n" if init_flag else ""

    train_sh = f"""#!/bin/bash
# LoRA training script — generated by review_outputs.py
# Requires: pip install diffusers accelerate peft transformers
{extend_note}

DATASET_DIR="{lora_dir}"
OUTPUT_DIR="{weights_dir}"
BASE_MODEL="{base_model}"

mkdir -p "$OUTPUT_DIR"

accelerate launch --mixed_precision="fp16" \\
  -m diffusers.scripts.train_dreambooth_lora_sdxl \\
  --pretrained_model_name_or_path="$BASE_MODEL" \\
  --dataset_name="$DATASET_DIR" \\
  --output_dir="$OUTPUT_DIR" \\
  --mixed_precision="fp16" \\
  --resolution=1024 \\
  --train_batch_size=1 \\
  --gradient_accumulation_steps=4 \\
  --learning_rate={lr} \\
  --lr_scheduler="constant" \\
  --lr_warmup_steps=0 \\
  --max_train_steps={steps} \\
  --checkpointing_steps=100 \\
  --seed=42 \\
  --report_to="none" \\
{resume_line}{init_line}
echo ""
echo "LoRA weights saved to: $OUTPUT_DIR"
echo "To use: add the .safetensors file to your CoderAI model config as a LoRA."
"""
    script = lora_dir / "train_lora.sh"
    script.write_text(train_sh)
    script.chmod(0o755)


def run_lora_training(out_dir: Path, steps: int = 500, lr: str = "1e-4") -> dict:
    """Launch the generated train_lora.sh in the background."""
    lora_dir = out_dir / "lora_dataset"
    script = lora_dir / "train_lora.sh"
    if not script.exists():
        return {"ok": False, "error": "No training script found. Export dataset first."}
    # Re-generate script with latest steps/lr in case user changed them
    existing = _find_existing_lora(lora_dir)
    meta = lora_dir / "metadata.jsonl"
    if meta.exists():
        # Read base model from the existing script if we have one
        base_model = "stabilityai/stable-diffusion-xl-base-1.0"
        try:
            for line in script.read_text().splitlines():
                if line.strip().startswith("BASE_MODEL="):
                    base_model = line.split("=", 1)[1].strip().strip('"')
                    break
        except Exception:
            pass
        _write_train_script(lora_dir, base_model, existing, steps=steps, lr=lr)
    log_path = out_dir / "lora_dataset" / "training.log"
    proc = subprocess.Popen(
        ["bash", str(script)],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        cwd=str(out_dir),
    )
    return {
        "ok": True,
        "pid": proc.pid,
        "log": str(log_path),
        "message": f"Training started (PID {proc.pid}). Watch {log_path}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Embedded HTML/JS UI
# ─────────────────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Township Fighters — Review</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#111;color:#e0e0e0;min-height:100vh}
header{background:#1a1a1a;border-bottom:1px solid #333;padding:.75rem 1.25rem;display:flex;align-items:center;gap:1rem;position:sticky;top:0;z-index:100}
header h1{font-size:15px;font-weight:700;color:#fff}
.tabs{display:flex;gap:.25rem}
.tab{padding:.35rem .85rem;border-radius:5px;cursor:pointer;font-size:12px;font-weight:600;color:#999;background:transparent;border:1px solid transparent;transition:all .15s}
.tab.active,.tab:hover{background:#2a2a2a;color:#fff;border-color:#444}
.tab.active{background:#6366f1;border-color:#6366f1;color:#fff}
.stats{margin-left:auto;font-size:11px;color:#666}
.stats span{color:#888}
main{padding:1.25rem;max-width:1400px;margin:0 auto}
.section{display:none}.section.active{display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:1rem}
.card{background:#1c1c1c;border:1px solid #2a2a2a;border-radius:8px;overflow:hidden;transition:border-color .15s}
.card:hover{border-color:#444}
.card.good{border-color:#22c55e}
.card.bad{border-color:#ef4444}
.card.skip{border-color:#f59e0b}
.thumb{width:100%;aspect-ratio:1;object-fit:cover;display:block;cursor:pointer;background:#0a0a0a}
.card-body{padding:.5rem}
.card-name{font-size:11px;color:#aaa;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:.4rem}
.card-actions{display:flex;gap:.3rem}
.btn{flex:1;padding:.3rem;border-radius:4px;border:none;cursor:pointer;font-size:11px;font-weight:600;transition:opacity .1s}
.btn:hover{opacity:.8}
.btn-good{background:#16a34a;color:#fff}
.btn-bad{background:#dc2626;color:#fff}
.btn-skip{background:#d97706;color:#fff}
.btn-clear{background:#2a2a2a;color:#aaa;font-size:10px}
.note-input{width:100%;margin-top:.35rem;padding:.25rem .4rem;background:#111;border:1px solid #333;border-radius:3px;color:#ccc;font-size:10px;resize:none}
.group-header{font-size:13px;font-weight:700;color:#ddd;margin:1.25rem 0 .6rem;padding-bottom:.3rem;border-bottom:1px solid #2a2a2a}
.video-card{background:#1c1c1c;border:1px solid #2a2a2a;border-radius:8px;overflow:hidden;transition:border-color .15s}
.video-card:hover{border-color:#444}
.video-card.good{border-color:#22c55e}.video-card.bad{border-color:#ef4444}.video-card.skip{border-color:#f59e0b}
.video-card video{width:100%;display:block;max-height:200px;background:#000}
.video-body{padding:.5rem}
.video-name{font-size:11px;color:#aaa;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:.4rem}
.video-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:1rem}
.train-panel{background:#1c1c1c;border:1px solid #2a2a2a;border-radius:8px;padding:1.25rem;max-width:680px}
.train-panel h2{font-size:14px;font-weight:700;margin-bottom:.75rem}
.train-row{display:flex;gap:.75rem;align-items:center;margin-bottom:.75rem}
.train-label{font-size:12px;color:#aaa;width:130px;flex-shrink:0}
.train-input{flex:1;padding:.4rem .6rem;background:#111;border:1px solid #333;border-radius:4px;color:#ddd;font-size:12px}
.action-btn{padding:.55rem 1.2rem;background:#6366f1;color:#fff;border:none;border-radius:5px;cursor:pointer;font-size:13px;font-weight:600}
.action-btn:hover{background:#4f46e5}
.action-btn.danger{background:#dc2626}
.result-box{background:#111;border:1px solid #333;border-radius:5px;padding:.75rem;font-size:12px;color:#aaa;margin-top:.75rem;white-space:pre-wrap;display:none}
.result-box.visible{display:block}
.counter-badge{display:inline-block;padding:.1rem .4rem;border-radius:3px;font-size:10px;font-weight:700;margin-left:.3rem}
.good-badge{background:#16a34a22;color:#4ade80}
.bad-badge{background:#dc262622;color:#f87171}
.skip-badge{background:#d9770622;color:#fbbf24}
.lightbox{display:none;position:fixed;inset:0;background:rgba(0,0,0,.9);z-index:9999;align-items:center;justify-content:center;cursor:zoom-out}
.lightbox.visible{display:flex}
.lightbox img{max-width:90vw;max-height:90vh;object-fit:contain;border-radius:6px}
</style>
</head>
<body>
<div id="lightbox" class="lightbox" onclick="closeLightbox()"><img id="lb-img" src=""></div>
<header>
  <h1>Township Fighters — Review</h1>
  <div class="tabs">
    <div class="tab active" onclick="switchTab('characters',this)">Characters <span id="t-chars" class="counter-badge good-badge"></span></div>
    <div class="tab" onclick="switchTab('environments',this)">Environments <span id="t-envs" class="counter-badge good-badge"></span></div>
    <div class="tab" onclick="switchTab('videos',this)">Videos <span id="t-vids" class="counter-badge good-badge"></span></div>
    <div class="tab" onclick="switchTab('training',this)">Training</div>
  </div>
  <div class="stats"><span id="stat-good">0</span> good · <span id="stat-bad">0</span> bad · <span id="stat-skip">0</span> maybe</div>
</header>
<main>
  <div id="sec-characters" class="section active"></div>
  <div id="sec-environments" class="section"></div>
  <div id="sec-videos" class="section"></div>
  <div id="sec-training" class="section">
    <div class="train-panel">
      <h2>LoRA Training from Approved Images</h2>

      <div id="extend-notice" style="display:none;background:#1a2a1a;border:1px solid #2d4a2d;border-radius:5px;padding:.6rem .8rem;margin-bottom:.8rem;font-size:12px;color:#86efac;line-height:1.5"></div>

      <div class="train-row">
        <span class="train-label">Base image model</span>
        <input id="base-model" class="train-input" value="" placeholder="e.g. John6666/pornmaster-pro-pony-asianponyv3vae-sdxl">
      </div>
      <div class="train-row">
        <span class="train-label">Extra steps</span>
        <input id="extra-steps" class="train-input" type="number" value="500" min="50" max="5000"
               title="Total new steps to run (added on top of any existing training)">
      </div>
      <div class="train-row">
        <span class="train-label">Learning rate</span>
        <input id="lr" class="train-input" value="1e-4" placeholder="1e-4">
      </div>

      <div style="display:flex;gap:.75rem;flex-wrap:wrap;margin-top:.25rem">
        <button class="action-btn" onclick="exportDataset()">1. Export / refresh dataset</button>
        <button class="action-btn" onclick="trainLora()">2. Train / extend LoRA</button>
      </div>
      <div id="train-result" class="result-box"></div>

      <div style="margin-top:1rem;font-size:11px;color:#555;line-height:1.7">
        <b style="color:#888">First run (fresh LoRA):</b><br>
        &nbsp;1. Rate images 👍 Good in Characters / Environments tabs.<br>
        &nbsp;2. Click <b>Export dataset</b> → copies approved images + prompts to <code>lora_dataset/</code>.<br>
        &nbsp;3. Click <b>Train LoRA</b> → runs training in background, saves <code>lora_weights/*.safetensors</code>.<br>
        &nbsp;4. Add the <code>.safetensors</code> to CoderAI's image model LoRA settings.<br>
        <br>
        <b style="color:#888">Subsequent runs (extending an existing LoRA):</b><br>
        &nbsp;1. Generate more content, rate new images as 👍 Good.<br>
        &nbsp;2. Click <b>Export dataset</b> → adds new images alongside old ones.<br>
        &nbsp;3. Click <b>Train LoRA</b> → auto-detects existing weights and continues from them.<br>
        &nbsp;&nbsp;&nbsp;&nbsp;If a <code>checkpoint-NNNN/</code> dir exists → resumes exactly from that point.<br>
        &nbsp;&nbsp;&nbsp;&nbsp;If only a <code>.safetensors</code> exists → initializes adapter from it, then trains.<br>
        <br>
        <b style="color:#888">Requirements:</b> <code>pip install peft accelerate</code>
      </div>
    </div>
  </div>
</main>

<script>
let _inv = {}, _fb = {};

async function api(path, body) {
  const opts = body
    ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)}
    : {method:'GET'};
  const r = await fetch(path, opts);
  return r.json();
}

async function init() {
  const data = await api('/api/data');
  _inv = data.inventory;
  _fb  = data.feedback;
  renderCharacters();
  renderEnvironments();
  renderVideos();
  updateStats();
}

function switchTab(name, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('sec-' + name).classList.add('active');
  if (name === 'training') checkExistingLora();
}

function rating(relPath) { return (_fb[relPath] || {}).rating || ''; }
function note(relPath)   { return (_fb[relPath] || {}).note   || ''; }

function updateStats() {
  const vals = Object.values(_fb);
  document.getElementById('stat-good').textContent = vals.filter(v=>v.rating==='good').length;
  document.getElementById('stat-bad').textContent  = vals.filter(v=>v.rating==='bad').length;
  document.getElementById('stat-skip').textContent = vals.filter(v=>v.rating==='skip').length;

  // Update tab badges
  const charGood = Object.keys(_fb).filter(k=>k.startsWith('characters/')&&_fb[k].rating==='good').length;
  const envGood  = Object.keys(_fb).filter(k=>k.startsWith('environments/')&&_fb[k].rating==='good').length;
  const vidGood  = Object.keys(_fb).filter(k=>k.startsWith('videos/')&&_fb[k].rating==='good').length;
  document.getElementById('t-chars').textContent = charGood || '';
  document.getElementById('t-envs').textContent  = envGood  || '';
  document.getElementById('t-vids').textContent  = vidGood  || '';
}

async function rate(relPath, rating, noteEl) {
  const n = noteEl ? noteEl.value : (note(relPath) || '');
  _fb[relPath] = {rating, note: n, timestamp: Date.now()/1000|0};
  await api('/api/rate', {path: relPath, rating, note: n});
  // Update card border
  document.querySelectorAll(`[data-path="${CSS.escape(relPath)}"]`).forEach(el => {
    el.classList.remove('good','bad','skip');
    if (rating) el.classList.add(rating);
  });
  updateStats();
}

async function clearRate(relPath) {
  delete _fb[relPath];
  await api('/api/rate', {path: relPath, rating: '', note: ''});
  document.querySelectorAll(`[data-path="${CSS.escape(relPath)}"]`).forEach(el => {
    el.classList.remove('good','bad','skip');
  });
  updateStats();
}

function imageCard(relPath) {
  const r = rating(relPath);
  const n = note(relPath);
  const fname = relPath.split('/').pop();
  const id = relPath.replace(/[^a-z0-9]/gi,'_');
  return `
  <div class="card ${r}" data-path="${relPath}">
    <img class="thumb" src="/file/${relPath}" alt="${fname}"
         title="${relPath}" onclick="openLightbox('/file/${relPath}')">
    <div class="card-body">
      <div class="card-name" title="${relPath}">${fname}</div>
      <div class="card-actions">
        <button class="btn btn-good"  onclick="rate('${relPath}','good', document.getElementById('n_${id}'))">👍</button>
        <button class="btn btn-bad"   onclick="rate('${relPath}','bad',  document.getElementById('n_${id}'))">👎</button>
        <button class="btn btn-skip"  onclick="rate('${relPath}','skip', document.getElementById('n_${id}'))">🤔</button>
        <button class="btn btn-clear" onclick="clearRate('${relPath}')">✕</button>
      </div>
      <textarea id="n_${id}" class="note-input" rows="2"
                placeholder="Optional note…"
                onblur="if(_fb['${relPath}'])rate('${relPath}',_fb['${relPath}'].rating,this)"
      >${n}</textarea>
    </div>
  </div>`;
}

function videoCard(relPath) {
  const r = rating(relPath);
  const n = note(relPath);
  const fname = relPath.split('/').pop();
  const id = relPath.replace(/[^a-z0-9]/gi,'_');
  return `
  <div class="video-card ${r}" data-path="${relPath}">
    <video controls preload="metadata" src="/file/${relPath}"></video>
    <div class="video-body">
      <div class="video-name" title="${relPath}">${fname}</div>
      <div class="card-actions">
        <button class="btn btn-good"  onclick="rate('${relPath}','good', document.getElementById('n_${id}'))">👍 Good</button>
        <button class="btn btn-bad"   onclick="rate('${relPath}','bad',  document.getElementById('n_${id}'))">👎 Bad</button>
        <button class="btn btn-skip"  onclick="rate('${relPath}','skip', document.getElementById('n_${id}'))">🤔 Maybe</button>
        <button class="btn btn-clear" onclick="clearRate('${relPath}')">✕</button>
      </div>
      <textarea id="n_${id}" class="note-input" rows="2"
                placeholder="Optional note…"
                onblur="if(_fb['${relPath}'])rate('${relPath}',_fb['${relPath}'].rating,this)"
      >${n}</textarea>
    </div>
  </div>`;
}

function renderCharacters() {
  let html = '';
  for (const [name, data] of Object.entries(_inv.characters || {})) {
    const meta = data.meta || {};
    const desc = meta.description || '';
    html += `<div class="group-header">${name}<span style="font-weight:400;color:#666;font-size:11px;margin-left:.5rem">${desc}</span></div>`;
    html += '<div class="grid">';
    for (const img of data.images) html += imageCard(img);
    html += '</div>';
  }
  document.getElementById('sec-characters').innerHTML = html || '<div style="color:#555;padding:2rem">No characters found in output directory.</div>';
}

function renderEnvironments() {
  let html = '';
  for (const [name, data] of Object.entries(_inv.environments || {})) {
    const meta = data.meta || {};
    const desc = meta.description || '';
    html += `<div class="group-header">${name}<span style="font-weight:400;color:#666;font-size:11px;margin-left:.5rem">${desc}</span></div>`;
    html += '<div class="grid">';
    for (const img of data.images) html += imageCard(img);
    html += '</div>';
  }
  document.getElementById('sec-environments').innerHTML = html || '<div style="color:#555;padding:2rem">No environments found in output directory.</div>';
}

function renderVideos() {
  let html = '<div class="video-grid">';
  for (const v of (_inv.videos || [])) html += videoCard(v);
  html += '</div>';
  document.getElementById('sec-videos').innerHTML =
    (_inv.videos || []).length ? html : '<div style="color:#555;padding:2rem">No videos found in output directory.</div>';
}

async function exportDataset() {
  const baseModel = document.getElementById('base-model').value.trim();
  const steps = parseInt(document.getElementById('extra-steps').value) || 500;
  const lr = document.getElementById('lr').value.trim() || '1e-4';
  const box = document.getElementById('train-result');
  box.textContent = 'Exporting dataset…';
  box.classList.add('visible');
  const r = await api('/api/export', {base_model: baseModel, steps, lr});
  if (r.ok) {
    let msg = `✓ Dataset ready:\n  ${r.images} image(s) → ${r.dataset_dir}\n  Script: ${r.train_script}`;
    if (r.extending) msg += `\n\n  ↪ Will extend existing LoRA:\n    ${r.extending}`;
    box.textContent = msg;
    // Update the notice banner
    const notice = document.getElementById('extend-notice');
    if (r.extending) {
      notice.style.display = '';
      notice.innerHTML = `↪ Extending existing LoRA: <code>${r.extending}</code><br>
        New approved images will be added on top — previous learning is preserved.`;
    } else {
      notice.style.display = 'none';
    }
  } else {
    box.textContent = '✗ ' + (r.error || 'Export failed');
  }
}

async function trainLora() {
  const steps = parseInt(document.getElementById('extra-steps').value) || 500;
  const lr = document.getElementById('lr').value.trim() || '1e-4';
  const box = document.getElementById('train-result');
  box.textContent = 'Starting training…';
  box.classList.add('visible');
  const r = await api('/api/train', {steps, lr});
  box.textContent = r.ok ? `✓ ${r.message}\n\nLog file: ${r.log}` : '✗ ' + (r.error || 'Training failed');
}

// Check for existing LoRA on tab switch to training
async function checkExistingLora() {
  const r = await api('/api/lora-status');
  const notice = document.getElementById('extend-notice');
  if (r.existing_lora) {
    notice.style.display = '';
    notice.innerHTML = `↪ Existing LoRA detected: <code>${r.existing_lora}</code><br>
      Exporting will continue from this checkpoint — previous learning is preserved.`;
  }
}

function openLightbox(src) {
  document.getElementById('lb-img').src = src;
  document.getElementById('lightbox').classList.add('visible');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('visible');
}
document.addEventListener('keydown', e => { if (e.key==='Escape') closeLightbox(); });

init();
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# HTTP server
# ─────────────────────────────────────────────────────────────────────────────

class ReviewHandler(http.server.BaseHTTPRequestHandler):
    out_dir: Path = None

    def log_message(self, fmt, *args):
        pass   # suppress per-request access log

    def _send(self, code: int, content_type: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: dict, code: int = 200):
        body = json.dumps(data).encode()
        self._send(code, "application/json", body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "":
            self._send(200, "text/html; charset=utf-8", _HTML.encode())
            return

        if path == "/api/data":
            inv = discover_outputs(self.out_dir)
            fb = load_feedback(self.out_dir)
            self._json({"inventory": inv, "feedback": fb["items"]})
            return

        if path == "/api/lora-status":
            existing = _find_existing_lora(self.out_dir / "lora_dataset")
            self._json({"existing_lora": str(existing) if existing else None})
            return

        if path.startswith("/file/"):
            rel = urllib.parse.unquote(path[6:])
            abs_path = self.out_dir / rel
            # Security: resolve and ensure it's inside out_dir
            try:
                abs_path = abs_path.resolve()
                self.out_dir.resolve()
                abs_path.relative_to(self.out_dir.resolve())
            except (ValueError, Exception):
                self._send(403, "text/plain", b"Forbidden")
                return
            if not abs_path.exists():
                self._send(404, "text/plain", b"Not found")
                return
            mime = mimetypes.guess_type(str(abs_path))[0] or "application/octet-stream"
            self._send(200, mime, abs_path.read_bytes())
            return

        self._send(404, "text/plain", b"Not found")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        path = urllib.parse.urlparse(self.path).path

        if path == "/api/rate":
            rel = body.get("path", "")
            rating = body.get("rating", "")
            note = body.get("note", "")
            if rating:
                set_rating(self.out_dir, rel, rating, note)
            else:
                # Clear
                data = load_feedback(self.out_dir)
                data["items"].pop(rel, None)
                save_feedback(self.out_dir, data)
            self._json({"ok": True})
            return

        if path == "/api/export":
            result = export_lora_dataset(
                self.out_dir,
                base_model=body.get("base_model") or None,
                steps=int(body.get("steps") or 500),
                lr=body.get("lr") or "1e-4",
            )
            self._json(result)
            return

        if path == "/api/train":
            result = run_lora_training(
                self.out_dir,
                steps=int(body.get("steps") or 500),
                lr=body.get("lr") or "1e-4",
            )
            self._json(result)
            return

        self._send(404, "text/plain", b"Not found")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Township Fighters — Output Review & LoRA Training UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WHAT THIS TOOL DOES
───────────────────
  Serves a web UI (no external dependencies beyond Python stdlib) where you
  review everything gen_township_fighters.py produced, rate each item, and
  progressively improve generation quality through LoRA fine-tuning.

  Characters tab   — grid of all fighter reference images, one group per
                     fighter. Click any image to enlarge it. Rate each one:
                       👍 Good  — include in training data
                       👎 Bad   — exclude (wrong face, bad lighting, etc.)
                       🤔 Maybe — revisit later
                     Add an optional text note per image.

  Environments tab — same for environment reference images.

  Videos tab       — inline video player for every generated clip (short,
                     long, and outcome videos). Rate to track which prompts
                     and settings produced the best results.

  Training tab     — build and launch LoRA training from approved images:
                       Step 1: Export dataset  → copies all 👍-rated images +
                               their generation prompts into lora_dataset/
                               (dreambooth format with metadata.jsonl).
                       Step 2: Train LoRA      → runs lora_dataset/train_lora.sh
                               via accelerate in a background process; saves
                               .safetensors weights to lora_dataset/lora_weights/.
                     You can also tune Steps (how long to train) and Learning
                     rate directly in the tab before clicking either button.

FEEDBACK STORAGE
────────────────
  Ratings are saved instantly to <out-dir>/feedback.json — no submit button.
  The file is plain JSON and can be committed to version control to track
  which outputs were acceptable across multiple generation runs.

OUTPUT DIRECTORY STRUCTURE
──────────────────────────
  <out-dir>/
    characters/<name>/ref_NN.png   ← fighter reference images
    environments/<name>/ref_NN.png ← location reference images
    videos/match_*.mp4             ← fight clips
    feedback.json                  ← your ratings (written by this tool)
    lora_dataset/
      images/                      ← approved images copied here
      metadata.jsonl               ← one caption per image (generation prompt)
      train_lora.sh                ← ready-to-run training command
      lora_weights/
        pytorch_lora_weights.safetensors  ← final LoRA weights
        checkpoint-100/            ← mid-training checkpoints
        checkpoint-200/

LORA TRAINING — FIRST RUN (fresh LoRA from scratch)
────────────────────────────────────────────────────
  Requirements:  pip install peft accelerate
  (diffusers and transformers are already installed by CoderAI)

  1. Generate content with gen_township_fighters.py.
  2. Open this UI and rate images:  👍 for ones with good likeness/style.
     Aim for at least 10-20 good images per subject for decent results.
  3. In the Training tab, set the base model to the image model you used
     (e.g. John6666/pornmaster-pro-pony-asianponyv3vae-sdxl).
  4. Click "Export dataset" — verifies the dataset and shows a summary.
  5. Click "Train LoRA" — training runs in the background (~5-15 min on
     an RTX 3090 for 500 steps). Watch lora_dataset/training.log.
  6. When done, add the .safetensors file to CoderAI:
       Admin → Models → configure your image model → LoRA path

LORA TRAINING — SUBSEQUENT RUNS (extending an existing LoRA)
─────────────────────────────────────────────────────────────
  After generating more fighters or environments and rating the new images:

  1. Click "Export dataset" again — new approved images are added to the
     dataset alongside the old ones.
  2. The tool auto-detects any existing weights:
       • If lora_weights/checkpoint-NNN/ exists  → uses --resume_from_checkpoint
         (full optimizer state restored; smoothest convergence).
       • If only a .safetensors file exists       → uses --lora_model_name_or_path
         (adapter weights loaded as starting point; optimizer restarts).
  3. A green banner in the Training tab confirms what will be extended.
  4. Click "Train LoRA" — runs the additional steps on top of what was
     already learned. Use a lower learning rate (e.g. 5e-5) for refinement.
  5. The updated .safetensors replaces the old one in lora_weights/.

  Each round of generate → rate → train improves quality incrementally.
  There is no limit to how many rounds you can do.

TUNING TIPS
───────────
  Steps        500  = quick first pass; good enough to see improvement
               1000 = more thorough; use for final production LoRA
  Learning rate
    1e-4  = default for a fresh LoRA or large dataset additions
    5e-5  = gentler refinement when extending an existing LoRA
    1e-5  = very fine correction; use only with a mature LoRA

  If generated characters look too generic after training:
    → Add more diverse good images (different angles, lighting)
    → Lower the learning rate on the next extension run

  If generated characters over-fit (all look the same):
    → Fewer steps or higher learning rate on the next run
    → Drop some near-duplicate images from the dataset

EXAMPLES
────────
  # Open the UI for the default output directory:
  python tools/review_outputs.py

  # Custom output directory:
  python tools/review_outputs.py --out-dir /data/township_project

  # Different port (useful if 7860 is taken by another tool):
  python tools/review_outputs.py --port 8888

  # Headless server — don't auto-open the browser:
  python tools/review_outputs.py --no-open --port 7860

  # Full workflow in one session:
  python tools/gen_township_fighters.py --out-dir ./fights          # generate
  python tools/review_outputs.py --out-dir ./fights                 # review & train
  python tools/gen_township_fighters.py --out-dir ./fights \\        # generate more
    --reuse-fighters --reuse-environments --skip-characters \\
    --skip-environments
  python tools/review_outputs.py --out-dir ./fights                 # extend LoRA
""",
    )
    parser.add_argument("--out-dir", default="./township_output", metavar="DIR",
                        help="Output directory from gen_township_fighters.py (default: ./township_output)")
    parser.add_argument("--port", type=int, default=7860, metavar="PORT",
                        help="Port to serve the UI on (default: 7860)")
    parser.add_argument("--no-open", action="store_true",
                        help="Do not automatically open the browser")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if not out_dir.exists():
        print(f"✗ Output directory not found: {out_dir}")
        print("  Run gen_township_fighters.py first to generate content.")
        sys.exit(1)

    ReviewHandler.out_dir = out_dir

    server = http.server.HTTPServer(("0.0.0.0", args.port), ReviewHandler)

    inv = discover_outputs(out_dir)
    n_chars = sum(len(v["images"]) for v in inv["characters"].values())
    n_envs  = sum(len(v["images"]) for v in inv["environments"].values())
    n_vids  = len(inv["videos"])
    fb = load_feedback(out_dir)
    n_rated = len(fb["items"])

    print(f"""
╔══════════════════════════════════════════════════════════╗
║       Township Fighters — Output Review UI               ║
╚══════════════════════════════════════════════════════════╝
  Output dir : {out_dir}
  Content    : {n_chars} character images · {n_envs} environment images · {n_vids} videos
  Feedback   : {n_rated} item(s) already rated
  URL        : http://localhost:{args.port}
""")

    if not args.no_open:
        def _open():
            time.sleep(0.4)
            import webbrowser
            webbrowser.open(f"http://localhost:{args.port}")
        threading.Thread(target=_open, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
