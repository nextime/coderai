#!/usr/bin/env python3
"""
Township Fighters Content Generator
Generates characters, environments, and fight videos via the CoderAI API.

Run with --help for full usage.
"""

import argparse
import base64
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import requests

# Force line-buffered output so every print appears immediately even when
# stdout is piped to a file or another process.
sys.stdout.reconfigure(line_buffering=True)


def _log(*args, **kwargs):
    """print() that always flushes immediately."""
    print(*args, **kwargs, flush=True)


def _run_with_spinner(label: str, fn, *args, **kwargs):
    """
    Run fn(*args, **kwargs) in a background thread while printing a live
    elapsed-time ticker on stdout so the user knows the script is alive.
    Returns the function's return value, or re-raises any exception it threw.
    """
    result_box = [None]
    exc_box    = [None]

    def _worker():
        try:
            result_box[0] = fn(*args, **kwargs)
        except Exception as e:
            exc_box[0] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    start = time.monotonic()
    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    idx = 0
    while t.is_alive():
        elapsed = time.monotonic() - start
        sys.stdout.write(f"\r    {spinner[idx % len(spinner)]} {label}  {elapsed:.0f}s elapsed…")
        sys.stdout.flush()
        idx += 1
        t.join(timeout=1.0)
    # Clear the spinner line
    sys.stdout.write("\r" + " " * 72 + "\r")
    sys.stdout.flush()

    if exc_box[0] is not None:
        raise exc_box[0]
    return result_box[0]

# ─────────────────────────────────────────────────────────────────────────────
# Fighter pool — South Africa + Uganda/Kampala
# ─────────────────────────────────────────────────────────────────────────────

FIGHTER_POOL = [
    # ── South Africa ──────────────────────────────────────────────────────────
    {
        "name": "khumalo",
        "gender": "male",
        "region": "soweto",
        "description": "Heavyweight from Soweto, shaved head, powerful build, tribal scar on left cheek",
        "prompt": (
            "Portrait of a muscular South African man from Soweto township, shaved head, "
            "tribal scar on left cheek, intense gaze, boxing singlet, realistic, studio lighting, "
            "detailed skin texture, African township fighter"
        ),
    },
    {
        "name": "dlamini",
        "gender": "male",
        "region": "khayelitsha",
        "description": "Lightweight from Khayelitsha, lean and fast, dreadlocks, sharp eyes",
        "prompt": (
            "Portrait of a lean young South African man from Khayelitsha, short dreadlocks, "
            "sharp focused eyes, athletic build, worn boxing shorts, realistic portrait, "
            "dramatic side lighting, African urban fighter"
        ),
    },
    {
        "name": "nkosi",
        "gender": "male",
        "region": "alexandra",
        "description": "Middleweight from Alexandra, muscular arms, close-cropped hair, determined expression",
        "prompt": (
            "Portrait of a muscular South African man from Alexandra township, close-cropped hair, "
            "broad shoulders, determined expression, street fighter, realistic portrait, "
            "natural light, African urban setting background"
        ),
    },
    {
        "name": "sibanda",
        "gender": "male",
        "region": "harare",
        "description": "Welterweight from Harare township, tall and wiry, high cheekbones, confident stance",
        "prompt": (
            "Portrait of a tall wiry Zimbabwean man from Harare township, high cheekbones, "
            "confident stance, MMA shorts, athletic, realistic portrait, warm African afternoon light"
        ),
    },
    {
        "name": "okonkwo",
        "gender": "male",
        "region": "ajegunle",
        "description": "Super heavyweight from Lagos Ajegunle, massive frame, bald, tattoo on neck",
        "prompt": (
            "Portrait of a massive Nigerian man from Ajegunle Lagos, bald head, "
            "tribal tattoo on neck, imposing build, free fighting attire, "
            "realistic portrait, dramatic low-angle lighting, powerful stance"
        ),
    },
    {
        "name": "mutombo",
        "gender": "male",
        "region": "matonge",
        "description": "Middleweight from Kinshasa Matonge, fast hands, short natural hair, focused",
        "prompt": (
            "Portrait of a focused Congolese fighter from Matonge Kinshasa, short natural hair, "
            "quick hands visible, athletic middleweight build, street fight attire, "
            "realistic portrait, urban African backdrop"
        ),
    },
    # ── Uganda / Kampala ──────────────────────────────────────────────────────
    {
        "name": "ssebuliba",
        "gender": "male",
        "region": "kampala_kisenyi",
        "description": "Welterweight from Kisenyi Kampala, compact and explosive, shaved temples, tribal marks",
        "prompt": (
            "Portrait of a compact muscular Ugandan man from Kisenyi Kampala slum, shaved temples, "
            "traditional tribal marks on cheeks, explosive athletic build, MMA shorts, "
            "realistic portrait, warm equatorial afternoon light, focused expression"
        ),
    },
    {
        "name": "kato",
        "gender": "male",
        "region": "kampala_katwe",
        "description": "Lightweight from Katwe Kampala, lightning quick, lean, close-cropped hair, scar on chin",
        "prompt": (
            "Portrait of a lean fast Ugandan fighter from Katwe Kampala, very short hair, "
            "small scar on chin, bright intense eyes, boxer's stance, street fighting attire, "
            "realistic portrait, urban Kampala background, dramatic lighting"
        ),
    },
    {
        "name": "okello",
        "gender": "male",
        "region": "kampala_bwaise",
        "description": "Heavyweight from Bwaise Kampala, massive chest, braided hair, battle-hardened",
        "prompt": (
            "Portrait of a heavyweight Ugandan fighter from Bwaise Kampala, massive broad chest, "
            "short cornrow braids, battle-hardened face, traditional beaded necklace, "
            "realistic portrait, gritty urban background, powerful confident stance"
        ),
    },
    {
        "name": "mugisha",
        "gender": "male",
        "region": "jinja",
        "description": "Middleweight from Jinja, tall and technical, lean face, long arms",
        "prompt": (
            "Portrait of a tall technical fighter from Jinja Uganda, lean angular face, "
            "unusually long arms, athletic middleweight build, boxing wraps on hands, "
            "realistic portrait, riverside industrial background, focused gaze"
        ),
    },
    {
        "name": "byarugaba",
        "gender": "male",
        "region": "kampala_makindye",
        "description": "Super middleweight from Makindye Kampala, stocky and powerful, close beard, gold tooth",
        "prompt": (
            "Portrait of a stocky powerful Ugandan fighter from Makindye Kampala, "
            "neat close beard, single gold tooth visible in slight grin, "
            "thick neck and shoulders, street fighter attire, realistic portrait, "
            "evening light, confident intimidating expression"
        ),
    },
    {
        "name": "namutebi",
        "gender": "female",
        "region": "kampala_nansana",
        "description": "Female welterweight from Nansana Kampala, fierce and technical, natural hair, arm tattoos",
        "prompt": (
            "Portrait of a fierce athletic Ugandan woman fighter from Nansana Kampala, "
            "natural afro hair pulled back, arm tattoos, determined expression, "
            "welterweight build, sports bra and MMA shorts, realistic portrait, "
            "dramatic side lighting, strong and capable stance"
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Environment pool — South Africa + Uganda/Kampala
# ─────────────────────────────────────────────────────────────────────────────

ENVIRONMENT_POOL = [
    # ── South Africa ──────────────────────────────────────────────────────────
    {
        "name": "soweto_gym",
        "region": "soweto",
        "description": "Rundown boxing gym in Soweto, cracked walls, old heavy bags, dim fluorescent light",
        "prompt": (
            "Interior of a worn-down boxing gym in Soweto South Africa, cracked concrete walls, "
            "old heavy punching bags, dim flickering fluorescent lights, dusty floor, "
            "posters of fighters on wall, cinematic, realistic, gritty atmosphere"
        ),
    },
    {
        "name": "township_ring",
        "region": "soweto",
        "description": "Makeshift outdoor boxing ring in African township, rope ring, crowd around, sunset",
        "prompt": (
            "Outdoor makeshift boxing ring in an African township, weathered rope and posts, "
            "crowd of spectators surrounding the ring, golden hour sunset, dust in the air, "
            "corrugated iron shacks in background, cinematic wide shot, realistic"
        ),
    },
    {
        "name": "dusty_road",
        "region": "south_africa",
        "description": "Dirt road in African township at dusk, shacks on both sides, streetlight",
        "prompt": (
            "Dirt road in an African township at dusk, corrugated iron shacks on both sides, "
            "single orange streetlight casting long shadows, red dust on the ground, "
            "cinematic dramatic lighting, wide shot, gritty urban Africa"
        ),
    },
    {
        "name": "township_square",
        "region": "johannesburg",
        "description": "Open square in Johannesburg township, concrete floor, graffiti walls, crowd gathering",
        "prompt": (
            "Open square in an African township, cracked concrete ground, "
            "graffiti-covered walls, gathered crowd watching, evening light, "
            "cinematic establishing shot, Johannesburg township atmosphere, realistic"
        ),
    },
    {
        "name": "khayelitsha_gym",
        "region": "khayelitsha",
        "description": "Small improvised gym in Khayelitsha Cape Town, corrugated walls, sand floor",
        "prompt": (
            "Small improvised fighting gym in Khayelitsha Cape Town, corrugated iron walls, "
            "sand and dirt floor, hand-painted equipment, single bare bulb, "
            "posters and photos on wall, cinematic, realistic, South African township setting"
        ),
    },
    {
        "name": "ajegunle_street",
        "region": "ajegunle",
        "description": "Narrow street in Ajegunle Lagos, market stalls pushed aside, night scene",
        "prompt": (
            "Narrow street in Ajegunle Lagos Nigeria at night, market stalls pushed to the sides, "
            "electric generator lights, crowd gathered, puddles reflecting light, "
            "cinematic night photography, gritty West African urban atmosphere"
        ),
    },
    # ── Uganda / Kampala ──────────────────────────────────────────────────────
    {
        "name": "katwe_backyard",
        "region": "katwe",
        "description": "Cramped backyard in Katwe Kampala, chain-link fence, mud walls, improvised ring",
        "prompt": (
            "Cramped backyard fighting area in Katwe Kampala Uganda, chain-link fence perimeter, "
            "mud brick walls, makeshift rope ring, crowd packed tight, "
            "single floodlight on a pole, realistic cinematic, equatorial night atmosphere"
        ),
    },
    {
        "name": "kisenyi_alley",
        "region": "kisenyi",
        "description": "Narrow alley in Kisenyi Kampala at dusk, vendors clearing space, spectators on walls",
        "prompt": (
            "Narrow alley in Kisenyi slum Kampala Uganda at dusk, market vendors moving aside, "
            "spectators climbing walls and rooftops to watch, warm orange light from setting sun, "
            "red laterite dust on ground, cinematic wide angle, realistic gritty atmosphere"
        ),
    },
    {
        "name": "bwaise_ring",
        "region": "bwaise",
        "description": "Flooded-area outdoor ring in Bwaise Kampala, wooden platform above mud, crowd",
        "prompt": (
            "Outdoor fighting platform built above muddy ground in Bwaise Kampala Uganda, "
            "wooden planks as floor, rusted corrugated iron walls surrounding area, "
            "dense crowd of spectators, overhead generator light, cinematic dramatic shot, realistic"
        ),
    },
    {
        "name": "kampala_rooftop",
        "region": "kampala_central",
        "description": "Rooftop in central Kampala at sunset, city skyline behind, improvised fighting space",
        "prompt": (
            "Rooftop fighting arena in central Kampala Uganda at sunset, "
            "Kampala city skyline in background, improvised rope boundary, "
            "crowd gathered around, warm golden light, cinematic establishing shot, realistic"
        ),
    },
    {
        "name": "jinja_warehouse",
        "region": "jinja",
        "description": "Abandoned warehouse near Jinja waterfront, industrial, dramatic light shafts",
        "prompt": (
            "Interior of abandoned industrial warehouse near Jinja Uganda waterfront, "
            "dramatic shafts of light through broken roof panels, concrete floor, "
            "rusted machinery in corners, spectators in shadows, cinematic moody atmosphere"
        ),
    },
    {
        "name": "makindye_market",
        "region": "makindye",
        "description": "Cleared market in Makindye Kampala after closing, stalls pushed aside, evening",
        "prompt": (
            "Market area in Makindye Kampala Uganda cleared for a fight after closing time, "
            "wooden market stalls pushed to edges, oil lamp and generator light, "
            "tight crowd of local spectators, red laterite ground, cinematic realistic"
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Static fallback shot prompts (used when LLM is not available)
# ─────────────────────────────────────────────────────────────────────────────

FIGHT_SHOT_TEMPLATES = [
    "exchanging heavy blows at close range, both fighters connecting, crowd erupting",
    "delivering a powerful uppercut, opponent's head snapping back on impact",
    "grappling and clinching against the ropes, sweat flying, crowd pushing forward",
    "dodging a hook and countering with a fast body shot, fluid athletic movement",
    "thrown back against the ropes, covering up desperately, absorbing punishment",
    "landing an explosive four-punch combination, each blow landing clean",
    "circling cautiously, both fighters reading each other, tension building",
    "throwing a spinning heel kick, opponent barely ducking under it",
    "ground-and-pound sequence, dominant position, crowd on their feet",
    "catching an overhand right, knees buckling, clutching for a clinch to survive",
    "breaking from a clinch, both fighters throwing wild hooks simultaneously",
    "landing a clean liver shot, opponent visibly hurt, doubling over",
    "referee warning both fighters, tempers flaring, crowd booing and cheering",
    "slipping inside a jab and returning a sharp elbow, street-fight style",
    "both fighters bloodied and exhausted, still throwing hard in the final seconds",
]

WIN_SHOT_TEMPLATES = {
    "win": [
        "raising both fists to the sky in victory, crowd surging forward, referee holding hand up",
        "falling to knees in triumph, tears of joy, cornerman rushing in to celebrate",
        "pointing to the crowd with a wide grin, sweat and blood on face, victorious",
    ],
    "ko_win": [
        "standing over knocked-out opponent, arms raised, crowd going absolutely wild",
        "walking to neutral corner after the finish, calm and dominant, referee waving it off",
        "roaring at the crowd with fist raised as opponent lies motionless behind them",
    ],
    "retire": [
        "sitting slumped on corner stool, trainer applying ice pack, head bowed in defeat",
        "being helped to corner by trainer, unable to continue, crowd respectfully quiet",
        "shaking head slowly as trainer calls it off, emotional moment, cornerman embracing",
    ],
    "draw": [
        "both fighters standing exhausted side by side, referee raising both hands simultaneously",
        "two fighters embracing grudgingly, both bloodied, respect after a brutal even contest",
        "both men looking at each other with grudging respect as announcer reads split decision",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Global prompt configuration (editable from the web "Prompts" page).
# Overrides are persisted to <out-dir>/prompts_config.json and re-applied to the
# module-level templates so both the LLM prompts and the static fallbacks change.
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT_OUTCOMES = ("win", "ko_win", "retire", "draw")


def prompts_config_snapshot() -> dict:
    """Return the current global prompt templates as a plain dict."""
    import sys as _sys
    m = _sys.modules[__name__]
    return {
        "llm_system": m._LLM_SYSTEM,
        "llm_outcome_system": m._LLM_OUTCOME_SYSTEM,
        "fight_shot_templates": list(m.FIGHT_SHOT_TEMPLATES),
        "win_shot_templates": {k: list(m.WIN_SHOT_TEMPLATES.get(k, []))
                               for k in _PROMPT_OUTCOMES},
    }


def apply_prompts_config(cfg: dict) -> None:
    """Apply a (partial) prompt-config dict to the live module globals."""
    if not cfg:
        return
    import sys as _sys
    m = _sys.modules[__name__]
    if cfg.get("llm_system"):
        m._LLM_SYSTEM = str(cfg["llm_system"])
    if cfg.get("llm_outcome_system"):
        m._LLM_OUTCOME_SYSTEM = str(cfg["llm_outcome_system"])
    fst = cfg.get("fight_shot_templates")
    if isinstance(fst, list):
        cleaned = [str(s).strip() for s in fst if str(s).strip()]
        if cleaned:
            m.FIGHT_SHOT_TEMPLATES = cleaned
    wst = cfg.get("win_shot_templates")
    if isinstance(wst, dict):
        merged = {k: list(v) for k, v in m.WIN_SHOT_TEMPLATES.items()}
        for k in _PROMPT_OUTCOMES:
            if isinstance(wst.get(k), list):
                cleaned = [str(s).strip() for s in wst[k] if str(s).strip()]
                if cleaned:
                    merged[k] = cleaned
        m.WIN_SHOT_TEMPLATES = merged


def _prompts_config_path(out_dir) -> Path:
    return Path(out_dir) / "prompts_config.json"


def load_prompts_config(out_dir) -> dict:
    p = _prompts_config_path(out_dir)
    if p.exists():
        try:
            return json.loads(p.read_text()) or {}
        except Exception:
            return {}
    return {}


def save_prompts_config(out_dir, cfg: dict) -> None:
    p = _prompts_config_path(out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2))

# ─────────────────────────────────────────────────────────────────────────────
# Local output structure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _char_dir(out_dir: Path, name: str) -> Path:
    return out_dir / "characters" / name

def _env_dir(out_dir: Path, name: str) -> Path:
    return out_dir / "environments" / name


def _save_profile_locally(out_dir: Path, kind: str, name: str, meta: dict,
                           images_b64: list) -> Path:
    """Save character or environment reference images + metadata to local disk."""
    d = (out_dir / (kind + "s") / name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    for i, raw in enumerate(images_b64):
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        try:
            (d / f"ref_{i:02d}.png").write_bytes(base64.b64decode(raw))
        except Exception:
            pass
    _log(f"    saved {len(images_b64)} image(s) → {d}")
    return d


def _load_local_profiles(out_dir: Path, kind: str) -> list:
    """Return list of names that have a complete local save (meta.json + at least one ref image)."""
    base = out_dir / (kind + "s")
    if not base.exists():
        return []
    result = []
    for d in sorted(base.iterdir()):
        if (d / "meta.json").exists() and list(d.glob("ref_*.png")):
            result.append(d.name)
    return result


def _ensure_in_coderai(client, kind: str, name: str, out_dir: Path) -> bool:
    """
    Upload a locally saved profile to CoderAI if it's not already there.
    Returns True on success.
    """
    # Check if already present
    try:
        if kind == "character":
            existing = [c["name"] for c in client.list_characters()]
        else:
            existing = [e["name"] for e in client.list_environments()]
        if name in existing:
            return True
    except Exception:
        pass

    # Load local images
    d = out_dir / (kind + "s") / name
    imgs = sorted(d.glob("ref_*.png"))
    if not imgs:
        _log(f"    ✗ No local images for {kind} '{name}'")
        return False

    b64_list = []
    for p in imgs:
        b64_list.append("data:image/png;base64," + base64.b64encode(p.read_bytes()).decode())

    meta = {}
    try:
        meta = json.loads((d / "meta.json").read_text())
    except Exception:
        pass

    try:
        endpoint = "/v1/characters" if kind == "character" else "/v1/environments"
        client._post(endpoint, {
            "name": name,
            "description": meta.get("description", ""),
            "images": b64_list,
        })
        _log(f"    ↑ uploaded '{name}' to CoderAI ({len(b64_list)} images)")
        return True
    except Exception as e:
        _log(f"    ✗ Upload failed for '{name}': {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Fatal-error detection
# ─────────────────────────────────────────────────────────────────────────────

def _is_fatal(err: Exception) -> bool:
    msg = str(err).lower()
    return any(k in msg for k in (
        "404", "entry not found", "not found", "failed to load",
        "no such file", "does not exist", "invalid model",
    ))


# ─────────────────────────────────────────────────────────────────────────────
# CoderAI API client
# ─────────────────────────────────────────────────────────────────────────────

class CoderAIClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 7200):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def _post(self, path: str, body: dict) -> dict:
        r = self.session.post(f"{self.base}{path}", json=body, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(f"POST {path} → {r.status_code}: {r.text[:400]}")
        return r.json()

    def _get(self, path: str) -> dict:
        r = self.session.get(f"{self.base}{path}", timeout=30)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str) -> dict:
        r = self.session.delete(f"{self.base}{path}", timeout=30)
        r.raise_for_status()
        return r.json()

    def _patch(self, path: str, body: dict) -> dict:
        r = self.session.patch(f"{self.base}{path}", json=body, timeout=60)
        if not r.ok:
            raise RuntimeError(f"PATCH {path} → {r.status_code}: {r.text[:400]}")
        return r.json()

    def delete_profile(self, kind: str, name: str) -> dict:
        plural = "characters" if kind == "character" else "environments"
        return self._delete(f"/v1/{plural}/{name}")

    def patch_profile(self, kind: str, name: str, description: str = None,
                      remove_indices: list = None, add_images: list = None) -> dict:
        plural = "characters" if kind == "character" else "environments"
        body = {}
        if description is not None:
            body["description"] = description
        if remove_indices:
            body["remove_indices"] = remove_indices
        if add_images:
            # Each entry may be a data-uri/base64 str or a {data,label} dict.
            imgs = []
            for j, im in enumerate(add_images):
                if isinstance(im, dict):
                    imgs.append(im)
                else:
                    imgs.append({"data": im, "label": f"regen_{j:02d}"})
            body["add_images"] = imgs
        return self._patch(f"/v1/{plural}/{name}", body)

    def list_models(self) -> list:
        return self._get("/v1/models").get("data", [])

    def list_image_models(self) -> list:
        return [m for m in self.list_models()
                if "image_generation" in (m.get("capabilities") or [])]

    def list_video_models(self) -> list:
        return [m for m in self.list_models()
                if "video_generation" in (m.get("capabilities") or [])]

    def list_text_models(self) -> list:
        return [m for m in self.list_models()
                if "text_generation" in (m.get("capabilities") or [])]

    def list_characters(self) -> list:
        try:
            return self._get("/v1/characters").get("characters", [])
        except Exception:
            return []

    def list_environments(self) -> list:
        try:
            return self._get("/v1/environments").get("environments", [])
        except Exception:
            return []

    def fetch_profile_images(self, kind: str, name: str) -> list:
        """Fetch base64 image list for a character or environment from CoderAI."""
        # Use the public /v1/ endpoints (work with Bearer auth).
        # The admin endpoints require a session cookie which the script doesn't have.
        kind_plural = "characters" if kind == "character" else "environments"
        try:
            d = self._get(f"/v1/{kind_plural}/{name}")
            return [img.get("data", "") for img in d.get("images", [])]
        except Exception:
            return []

    def generate_character(self, name: str, prompt: str, description: str,
                           model: str, n: int = 4, size: str = "512x512") -> dict:
        return self._post("/v1/characters/generate", {
            "name": name, "prompt": prompt, "description": description,
            "model": model, "n": n,
            "width": int(size.split("x")[0]), "height": int(size.split("x")[1]),
        })

    def generate_environment(self, name: str, prompt: str, description: str,
                             model: str, n: int = 3, size: str = "768x512") -> dict:
        return self._post("/v1/environments/generate", {
            "name": name, "prompt": prompt, "description": description,
            "model": model, "n": n,
            "width": int(size.split("x")[0]), "height": int(size.split("x")[1]),
        })

    def generate_image(self, prompt: str, model: str,
                       character_profiles: list = None,
                       loras: list = None, character_strength: float = 0.7,
                       size: str = "512x512", steps: int = 28,
                       seed: int = None, environment_profiles: list = None) -> bytes:
        """Generate a single still image (used for keyframes). Returns PNG bytes."""
        w, h = size.split("x")
        body = {
            "model": model, "prompt": prompt, "n": 1,
            "size": f"{int(w)}x{int(h)}", "steps": int(steps),
            "response_format": "b64_json",
        }
        if character_profiles:
            body["character_profiles"] = list(character_profiles)
            body["character_strength"] = character_strength
        if environment_profiles:
            body["environment_profiles"] = list(environment_profiles)
            body["character_strength"] = character_strength
        if loras:
            body["loras"] = loras
        if seed is not None:
            body["seed"] = seed
        d = self._post("/v1/images/generations", body)
        item = (d.get("data") or [{}])[0]
        raw = item.get("b64_json") or item.get("data") or item.get("url", "")
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        if raw.startswith("http"):
            import urllib.request
            with urllib.request.urlopen(raw, timeout=600) as resp:
                return resp.read()
        return base64.b64decode(raw)

    def train_lora(self, name: str, base_model: str, character: str = None,
                   environment: str = None, images: list = None,
                   steps: int = 800, rank: int = 16,
                   resolution: int = 512, train_base_model: str = None) -> dict:
        """Train a per-character or per-environment LoRA on the server.
        Blocks until complete."""
        body = {"name": name, "base_model": base_model,
                "steps": int(steps), "rank": int(rank), "resolution": int(resolution)}
        if train_base_model:
            body["train_base_model"] = train_base_model
        if character:
            body["character"] = character
        if environment:
            body["environment"] = environment
        if images:
            body["images"] = images
        return self._post("/v1/loras/train", body)

    def list_loras(self) -> list:
        try:
            return self._get("/v1/loras").get("loras", [])
        except Exception:
            return []

    def lora_progress(self) -> dict:
        try:
            return self._get("/v1/loras/progress")
        except Exception:
            return {}

    def chat_complete(self, model: str, system: str, user: str,
                      max_tokens: int = 400) -> str:
        import re as _re
        d = self._post("/v1/chat/completions", {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.95,
        })
        text = d["choices"][0]["message"]["content"] or ""
        # Strip <think>...</think> reasoning blocks (Qwen3/DeepSeek thinking mode).
        # Also handle incomplete blocks cut off by max_tokens.
        text = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL | _re.IGNORECASE)
        text = _re.sub(r'<think>.*$', '', text, flags=_re.DOTALL | _re.IGNORECASE)
        # The model sometimes runs past its turn and hallucinates extra
        # conversation turns (e.g. "...jab.\n\nAssistant:\n\nLow-angle...").
        # Keep only the FIRST turn: cut at any role marker or chat-template token.
        text = _re.split(
            r'(?im)^\s*(?:assistant|user|system|human)\s*:|<\|im_(?:start|end)\|>|<\|endoftext\|>|</s>',
            text, maxsplit=1)[0]
        # We ask for ONE sentence; if several leaked through, keep the first.
        text = text.strip()
        # Collapse internal blank lines that a leaked turn boundary left behind.
        text = _re.split(r'\n\s*\n', text, maxsplit=1)[0].strip()
        # Strip a chat-template role token glued onto the END of the content with
        # no separator (e.g. "...streetlights.user", "...platform.assistant").
        text = _re.sub(r'(?i)\s*(?:<\|im_end\|>|<\|endoftext\|>|</s>)\s*$', '', text)
        text = _re.sub(r'(?i)(?<=[.!?\'"”’)])\s*(?:user|assistant|system|human)\s*$', '', text)
        return text.strip()

    def upscale_video(self, video_bytes: bytes, factor: int = 2,
                      model: str = None) -> bytes:
        """POST to /v1/video/upscale and return the upscaled video bytes."""
        body = {
            "video": "data:video/mp4;base64," + base64.b64encode(video_bytes).decode(),
            "upscale_factor": factor,
            "response_format": "b64_mp4",
        }
        if model:
            body["model"] = model
        d = self._post("/v1/video/upscale", body)
        raw = d["data"][0].get("b64_mp4") or d["data"][0].get("url", "")
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        if raw.startswith("http"):
            import urllib.request
            with urllib.request.urlopen(raw, timeout=600) as resp:
                return resp.read()
        return base64.b64decode(raw)

    def interpolate_video(self, video_bytes: bytes, fps_multiplier: int = 2,
                          model: str = None) -> bytes:
        """POST to /v1/video/interpolate and return the interpolated video bytes."""
        body = {
            "video": "data:video/mp4;base64," + base64.b64encode(video_bytes).decode(),
            "fps_multiplier": fps_multiplier,
            "response_format": "b64_mp4",
        }
        if model:
            body["model"] = model
        d = self._post("/v1/video/interpolate", body)
        raw = d["data"][0].get("b64_mp4") or d["data"][0].get("url", "")
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        if raw.startswith("http"):
            import urllib.request
            with urllib.request.urlopen(raw, timeout=600) as resp:
                return resp.read()
        return base64.b64decode(raw)

    def generate_video_clip(self, prompt: str, model: str,
                            character_profiles: list = None,
                            environment_name: str = None,
                            num_frames: int = 49, fps: int = 8,
                            width: int = 512, height: int = 512,
                            seed: int = None,
                            init_image: bytes = None,
                            loras: list = None) -> bytes:
        # When a keyframe image is supplied, drive the model as text+image→video
        # (ti2v) so the first frame already shows the right fighters.
        mode = "ti2v" if init_image else "t2v"
        body = {
            "model": model, "prompt": prompt,
            "num_frames": num_frames, "fps": fps,
            "width": width, "height": height,
            "response_format": "b64_mp4", "mode": mode,
        }
        if character_profiles:
            body["character_profiles"] = character_profiles
        if environment_name:
            body["prompt"] = f"[{environment_name} location] " + body["prompt"]
        if seed is not None:
            body["seed"] = seed
        if init_image:
            body["init_image"] = "data:image/png;base64," + base64.b64encode(init_image).decode()
        if loras:
            body["loras"] = loras

        d = self._post("/v1/video/generations", body)
        raw = d["data"][0].get("b64_mp4") or d["data"][0].get("url", "")
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        if raw.startswith("http"):
            import urllib.request
            with urllib.request.urlopen(raw, timeout=120) as resp:
                return resp.read()
        return base64.b64decode(raw)


# ─────────────────────────────────────────────────────────────────────────────
# LLM prompt generation
# ─────────────────────────────────────────────────────────────────────────────

_LLM_SYSTEM = """\
You are a creative director writing vivid video-generation prompts for African street fighting scenes.
Each prompt must be ONE sentence, 15-35 words, cinematic and specific.
Vary camera angles (close-up, wide, low angle, over-shoulder), lighting (dusk, generator light, \
noon sun, spotlight), and action (strikes, clinch, footwork, takedown, ground work, crowd reaction).
Do NOT use generic phrases like "high quality" or "realistic". Return ONLY the prompt, no quotes."""

_LLM_OUTCOME_SYSTEM = """\
You are a creative director writing vivid 15-25 word video-generation prompts for fight outcome moments.
Be specific about body language, expression, lighting, and atmosphere.
Return ONLY the prompt, no quotes or explanation."""

# Snapshot the built-in defaults now that every template/system prompt is
# defined (before any saved override is applied), so the web "Prompts" page can
# offer a reset-to-defaults.
_PROMPT_DEFAULTS = prompts_config_snapshot()


class PromptGenerator:
    """Generates varied prompts, falling back to static templates if LLM unavailable."""

    def __init__(self, client: CoderAIClient, text_model: Optional[str] = None,
                 char_descriptions: dict = None):
        self.client = client
        self.model = text_model
        self.char_descriptions: dict = char_descriptions or {}
        self._used_fight: list[str] = []
        self._used_outcome: dict[str, list[str]] = {}

    def fight_shot(self, f1: str, f2: str, env_desc: str, match_context: str = "",
                   avoid: list = None) -> str:
        """Generate a unique fight shot prompt.

        `avoid` is an explicit list of actions to steer away from — used to pass
        the current match's already-generated shots so every clip in a match is
        distinct (the global recent-window only covers the last few).
        """
        # Cap the avoid list to a SMALL recent window.  A long avoid list bloats
        # the prompt and pushes the model into rambling/think output that cleans
        # to nothing — keep it short and focused (caller's recent + global recent).
        avoid_set = list(avoid or [])[-5:]
        for a in self._used_fight[-3:]:
            if a not in avoid_set:
                avoid_set.append(a)
        avoid_set = avoid_set[-6:]

        if self.model:
            # Try the LLM up to twice; on an empty/too-short reply, retry once with
            # a shorter avoid hint before giving up to the template.
            for attempt in range(2):
                try:
                    used_hint = (f"\nAvoid repeating these actions: {'; '.join(avoid_set)}."
                                 if avoid_set else "")
                    f1_desc = self.char_descriptions.get(f1, f1)
                    f2_desc = self.char_descriptions.get(f2, f2)
                    prompt = self.client.chat_complete(
                        model=self.model,
                        system=_LLM_SYSTEM,
                        user=(
                            f"Fighter 1: {f1_desc}. Fighter 2: {f2_desc}. "
                            f"Location: {env_desc}. "
                            f"{match_context}{used_hint}\n"
                            "Write one fight action shot prompt."
                        ),
                        max_tokens=120,
                    ).strip()
                    if len(prompt) < 8:
                        raise ValueError(f"LLM returned too-short response: {prompt!r}")
                    self._used_fight.append(prompt[:60])
                    return prompt
                except Exception as e:
                    if attempt == 0:
                        avoid_set = avoid_set[-2:]  # shorten the prompt and retry once
                        continue
                    _log(f"    (LLM prompt failed after retry: {e} — using template)")

        # Static fallback — avoid both the global recent window and `avoid`.
        _avoid_lower = {a.lower() for a in avoid_set}
        available = [p for p in FIGHT_SHOT_TEMPLATES if p.lower() not in _avoid_lower]
        if not available:
            available = FIGHT_SHOT_TEMPLATES
        choice = random.choice(available)
        self._used_fight.append(choice)
        return choice

    def outcome_shot(self, fighter: str, outcome: str, env_desc: str) -> str:
        """Generate a unique outcome shot prompt."""
        templates = WIN_SHOT_TEMPLATES.get(outcome, WIN_SHOT_TEMPLATES["win"])
        used = self._used_outcome.setdefault(outcome, [])

        if self.model:
            outcome_labels = {
                "win": "winning by decision — arm raised by referee",
                "ko_win": "winning by knockout — opponent is down",
                "retire": "losing — corner retiring them from the bout",
                "draw": "both fighters in a draw — referee raises both hands",
            }
            _avoid = used[-2:]
            for attempt in range(2):
                try:
                    used_hint = f" Avoid: {'; '.join(_avoid)}." if _avoid else ""
                    f_desc = self.char_descriptions.get(fighter, fighter)
                    prompt = self.client.chat_complete(
                        model=self.model,
                        system=_LLM_OUTCOME_SYSTEM,
                        user=(
                            f"Fighter: {f_desc}. Outcome: {outcome_labels.get(outcome, outcome)}. "
                            f"Location: {env_desc}.{used_hint} Write one outcome moment prompt."
                        ),
                        max_tokens=100,
                    ).strip()
                    if len(prompt) < 8:
                        raise ValueError(f"LLM returned too-short response: {prompt!r}")
                    used.append(prompt[:60])
                    return prompt
                except Exception as e:
                    if attempt == 0:
                        _avoid = []  # drop the hint and retry once
                        continue
                    _log(f"    (LLM prompt failed after retry: {e} — using template)")

        available = [t for t in templates if t not in used]
        if not available:
            available = templates
        choice = random.choice(available)
        used.append(choice)
        return choice


# ─────────────────────────────────────────────────────────────────────────────
# Video utilities
# ─────────────────────────────────────────────────────────────────────────────

def concat_videos(clip_paths: list, out_path: str):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
        list_path = f.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", list_path, "-c", "copy", out_path],
            check=True, capture_output=True,
        )
    finally:
        os.unlink(list_path)


def get_video_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def frames_for_seconds(seconds: float, fps: int = 8) -> int:
    return max(8, int(seconds * fps))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Characters
# ─────────────────────────────────────────────────────────────────────────────

def resolve_fighters(client: CoderAIClient, args, out_dir: Path) -> list:
    """
    Return list of fighter names to use for videos.
    Priority: --fighters > --reuse-fighters (local dir then CoderAI) > generate.
    Ensures every name is present in CoderAI before returning.
    """
    if args.fighters:
        names = [n.strip() for n in args.fighters.split(",")]
        _log(f"  Using specified fighters: {', '.join(names)}")
        for n in names:
            _ensure_in_coderai(client, "character", n, out_dir)
        return names

    if args.reuse_fighters:
        # Prefer local output directory; fall back to CoderAI-only profiles
        local = _load_local_profiles(out_dir, "character")
        if local:
            _log(f"  Reusing {len(local)} locally saved fighter(s): {', '.join(local)}")
            for n in local:
                _ensure_in_coderai(client, "character", n, out_dir)
            return local
        existing = [c["name"] for c in client.list_characters()]
        if existing:
            _log(f"  Reusing {len(existing)} CoderAI character(s): {', '.join(existing)}")
            return existing
        _log("  No saved fighters found — will generate from pool.")

    return None   # signal to stage_characters to generate


def stage_characters(client: CoderAIClient, image_model: str, out_dir: Path,
                     region_filter: Optional[str] = None,
                     include_female: bool = False) -> list:
    _log("\n" + "═" * 60)
    _log("  STAGE 1 — Characters")
    _log("═" * 60)

    pool = FIGHTER_POOL
    if not include_female:
        pool = [f for f in pool if f.get("gender", "male") == "male"]
    if region_filter:
        filtered = [f for f in pool if region_filter in f["region"]]
        if filtered:
            pool = filtered
        else:
            _log(f"  No fighters match region filter '{region_filter}', using full pool")

    _log(f"  Pool: {len(pool)} fighter(s)"
         + ("" if include_female else "  (male only — use --include-female to add female fighters)"))

    done, failed = [], []
    for i, fighter in enumerate(pool, 1):
        name = fighter["name"]
        _log(f"\n  [{i}/{len(pool)}] {name}  ({fighter['region']}, {fighter.get('gender','male')})")
        _log(f"    description: {fighter['description']}")
        _log(f"    prompt: {fighter['prompt']}")
        try:
            d = _run_with_spinner(
                f"generating character '{name}'",
                client.generate_character,
                name=name, prompt=fighter["prompt"],
                description=fighter["description"], model=image_model, n=4,
            )
            _log(f"    ✓ {d.get('image_count', '?')} reference images saved in CoderAI")
            # Fetch and save locally
            images = client.fetch_profile_images("character", name)
            if images:
                _save_profile_locally(out_dir, "character", name, {
                    "name": name,
                    "gender": fighter.get("gender", "male"),
                    "region": fighter["region"],
                    "description": fighter["description"],
                    "prompt": fighter["prompt"],
                }, images)
            done.append(name)
        except Exception as e:
            _log(f"    ✗ FAILED: {e}")
            failed.append(name)

    _log(f"\n  Characters: {len(done)} ok, {len(failed)} failed"
         + (f" ({', '.join(failed)})" if failed else ""))
    return done


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Environments
# ─────────────────────────────────────────────────────────────────────────────

def resolve_environments(client: CoderAIClient, args, out_dir: Path) -> list:
    if args.environments:
        names = [n.strip() for n in args.environments.split(",")]
        _log(f"  Using specified environments: {', '.join(names)}")
        for n in names:
            _ensure_in_coderai(client, "environment", n, out_dir)
        return names

    if args.reuse_environments:
        local = _load_local_profiles(out_dir, "environment")
        if local:
            _log(f"  Reusing {len(local)} locally saved environment(s): {', '.join(local)}")
            for n in local:
                _ensure_in_coderai(client, "environment", n, out_dir)
            return local
        existing = [e["name"] for e in client.list_environments()]
        if existing:
            _log(f"  Reusing {len(existing)} CoderAI environment(s): {', '.join(existing)}")
            return existing
        _log("  No saved environments found — will generate from pool.")

    return None


def stage_environments(client: CoderAIClient, image_model: str, out_dir: Path,
                       region_filter: Optional[str] = None) -> list:
    _log("\n" + "═" * 60)
    _log("  STAGE 2 — Environments")
    _log("═" * 60)

    pool = ENVIRONMENT_POOL
    if region_filter:
        filtered = [e for e in pool if region_filter in e["region"]]
        if filtered:
            pool = filtered
        else:
            _log(f"  No environments match region filter '{region_filter}', using full pool")

    done, failed = [], []
    for i, env in enumerate(pool, 1):
        name = env["name"]
        _log(f"\n  [{i}/{len(pool)}] {name}  ({env['region']})")
        _log(f"    description: {env['description']}")
        _log(f"    prompt: {env['prompt']}")
        try:
            d = _run_with_spinner(
                f"generating environment '{name}'",
                client.generate_environment,
                name=name, prompt=env["prompt"],
                description=env["description"], model=image_model,
                n=3, size="768x512",
            )
            _log(f"    ✓ {d.get('image_count', '?')} reference images saved in CoderAI")
            images = client.fetch_profile_images("environment", name)
            if images:
                _save_profile_locally(out_dir, "environment", name, {
                    "name": name,
                    "region": env["region"],
                    "description": env["description"],
                    "prompt": env["prompt"],
                }, images)
            done.append(name)
        except Exception as e:
            _log(f"    ✗ FAILED: {e}")
            failed.append(name)

    _log(f"\n  Environments: {len(done)} ok, {len(failed)} failed"
         + (f" ({', '.join(failed)})" if failed else ""))
    return done


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Videos
# ─────────────────────────────────────────────────────────────────────────────

def _env_description(env_name: str) -> str:
    """Return a human description for an environment name (for LLM context)."""
    for e in ENVIRONMENT_POOL:
        if e["name"] == env_name:
            return e["description"]
    return env_name.replace("_", " ")


def _build_char_descriptions(out_dir: Path) -> dict:
    """Return {name: description} merging FIGHTER_POOL + locally saved meta.json files."""
    desc = {f["name"]: f["description"] for f in FIGHTER_POOL}
    chars_dir = out_dir / "characters"
    if chars_dir.exists():
        for d in chars_dir.iterdir():
            meta_path = d / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    if meta.get("description"):
                        desc[d.name] = meta["description"]
                except Exception:
                    pass
    return desc


_VALID_CONSISTENCY = {"prompt", "ipadapter", "keyframe", "lora"}


def parse_consistency(spec: str) -> set:
    """Normalize a --consistency string into a set of strategy flags.

    'prompt' is always implied (descriptions are cheap and always help).
    Unknown tokens are ignored with no error so the pipeline stays robust.
    """
    out = {"prompt"}
    for tok in (spec or "").replace(";", ",").split(","):
        tok = tok.strip().lower()
        if tok in _VALID_CONSISTENCY:
            out.add(tok)
    return out


# Option keys persisted in a saved configuration file. These mirror the
# argparse dests for the generation options (web_port included so the web UI
# port is preserved). --save / --config / --cli-mode are deliberately excluded.
CONFIG_FIELDS = [
    "base_url", "api_key", "image_model", "video_model", "text_model",
    "no_llm", "out_dir", "fps", "clip_delay", "region", "include_female",
    "skip_characters", "reuse_fighters", "fighters",
    "skip_environments", "reuse_environments", "environments",
    "skip_videos", "only_outcomes", "matches",
    "only_characters", "only_environments", "only_assets",
    "only_prompts", "only_videos",
    "consistency", "keyframe_steps", "keyframe_size",
    "character_strength", "lora_steps", "lora_rank", "lora_weight",
    "lora_train_base_model",
    "no_env_loras", "env_lora_steps", "env_lora_rank", "env_lora_weight",
    "web_port",
]


def config_from_args(args) -> dict:
    """Extract the persistable generation options from an args namespace."""
    return {k: getattr(args, k) for k in CONFIG_FIELDS if hasattr(args, k)}


def save_config(path: str, args) -> dict:
    """Write the selected generation options to a JSON config file."""
    data = config_from_args(args)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    return data


def load_config(path: str) -> dict:
    """Load generation options from a saved JSON config file.

    Only recognised keys are returned; unknown keys are ignored so a config
    saved by a newer/older version stays usable.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("config file must contain a JSON object")
    return {k: v for k, v in data.items() if k in CONFIG_FIELDS}


def _fighter_desc_hint(name: str, char_descriptions: dict) -> str:
    """Return a compact visual hint for embedding in a video prompt."""
    desc = char_descriptions.get(name, "")
    if not desc:
        return name
    # Strip weight-class prefix ("Heavyweight from Soweto, " → keep the rest)
    # but keep it short — drop anything after the 3rd comma.
    parts = [p.strip() for p in desc.split(",")]
    # Keep up to 3 key visual traits (skip pure location/weight-class words)
    visual = ", ".join(parts[:3])
    return f"{visual}"


def _write_concat(clips: list, out_path: str, label: str):
    paths = [p for p, _ in clips]
    if not paths:
        return
    if len(paths) == 1:
        import shutil
        shutil.copy(paths[0], out_path)
    else:
        try:
            concat_videos(paths, out_path)
        except Exception as e:
            _log(f"    ✗ Concat failed ({label}): {e}")
            return
    _log(f"    ✓ {label} → {Path(out_path).name}  ({get_video_duration(out_path):.1f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# Character-consistency helpers (keyframe bridge + LoRA)
# ─────────────────────────────────────────────────────────────────────────────

def _reassemble_finals(video_dir: Path, match_name: str,
                       short_target: float = 45.0, long_target: float = 70.0) -> int:
    """Rebuild a match's short/long videos from its existing clip files on disk
    (ffmpeg concat only — no model). Returns the number of clips used."""
    clip_files = sorted(video_dir.glob(f"{match_name}_clip*.mp4"))
    clips = []
    for p in clip_files:
        dur = get_video_duration(str(p)) or 0.0
        clips.append((str(p), dur if dur > 0 else 5.0))
    if not clips:
        return 0
    short_clips, accum, pos = [], 0.0, 0
    while accum < short_target and clips:
        path, dur = clips[pos % len(clips)]
        short_clips.append((path, dur))
        accum += dur
        pos += 1
    _write_concat(short_clips, str(video_dir / f"{match_name}_short.mp4"),
                  f"short (~{short_target:.0f}s)")
    _write_concat(clips, str(video_dir / f"{match_name}_long.mp4"),
                  f"long  (~{long_target:.0f}s)")
    return len(clips)


def _clip_stem_fight(match_name: str, idx: int) -> str:
    return f"{match_name}_clip{idx:02d}"


def _clip_stem_outcome(fighter: str, outcome: str, match_name: str = None) -> str:
    # Per-match outcomes are named "<match>_<fighter>_<outcome>" so each match
    # has its own (different) outcome scenes. Legacy per-fighter outcomes (no
    # match_name) keep the old "<fighter>_<outcome>" name.
    if match_name:
        return f"{match_name}_{fighter}_{outcome}"
    return f"{fighter}_{outcome}"


def _lora_specs_for(fighters: list, lora_map: dict, weight: float) -> list:
    """Build the `loras` request list for the fighters appearing in a clip."""
    specs = []
    for f in fighters:
        path = (lora_map or {}).get(f)
        if path:
            specs.append({"model": path, "weight": float(weight), "name": f})
    return specs


def _env_lora_specs_for(env: str, env_lora_map: dict, weight: float) -> list:
    """Build the `loras` request entry for the environment used in a clip."""
    if not env:
        return []
    path = (env_lora_map or {}).get(env)
    if path:
        return [{"model": path, "weight": float(weight), "name": f"env_{env}"}]
    return []


# Per-kind LoRA training parameters: server name prefix, local cache file,
# the train_lora keyword used to pull reference images, and a friendly label.
_LORA_KINDS = {
    "character":   {"prefix": "fighter_", "file": "loras.json",     "label": "Character"},
    "environment": {"prefix": "env_",     "file": "env_loras.json", "label": "Environment"},
}


def _train_profile_loras(client: CoderAIClient, image_model: str, out_dir: Path,
                         names: list, kind: str,
                         lora_steps: int = 800, lora_rank: int = 16,
                         train_base_model: str = None) -> dict:
    """Train one identity LoRA per profile of `kind` (server-side).

    Returns {name: lora_path}. Resumable: skips profiles whose LoRA already
    exists locally (<kind>_loras.json) or on the server. All training is grouped
    here so the image base model is touched once for the whole batch.

    `train_base_model` overrides the model the LoRA is *trained* against (must be
    a UNet-based SD1.x/SDXL model); `image_model` stays the generation model.
    """
    spec = _LORA_KINDS[kind]
    _log("\n" + "═" * 60)
    _log(f"  STAGE — {spec['label']} LoRA training")
    _log("═" * 60)
    lora_file = out_dir / spec["file"]
    lora_map = {}
    if lora_file.exists():
        try:
            lora_map = json.loads(lora_file.read_text()) or {}
        except Exception:
            lora_map = {}

    try:
        existing = {l.get("name"): l.get("path") for l in client.list_loras()}
    except Exception:
        existing = {}

    def _save():
        try:
            lora_file.write_text(json.dumps(lora_map, indent=2))
        except Exception as e:
            _log(f"    ⚠ could not save {spec['file']}: {e}")

    for i, name in enumerate(names, 1):
        lora_name = f"{spec['prefix']}{name}"
        # Already trained and recorded?
        cur = lora_map.get(name)
        if cur and Path(cur).exists():
            _log(f"  [{i}/{len(names)}] {name}: reusing trained LoRA")
            continue
        if lora_name in existing and existing[lora_name]:
            lora_map[name] = existing[lora_name]
            _save()
            _log(f"  [{i}/{len(names)}] {name}: found existing LoRA on server")
            continue
        _log(f"  [{i}/{len(names)}] {name}: training LoRA "
             f"({lora_steps} steps, rank {lora_rank}) — this can take a while…")
        train_kwargs = dict(name=lora_name, base_model=image_model,
                            steps=lora_steps, rank=lora_rank)
        if train_base_model:
            train_kwargs["train_base_model"] = train_base_model
        train_kwargs[kind] = name  # character=name OR environment=name
        try:
            res = _run_with_spinner(
                f"training {kind} LoRA '{name}'",
                client.train_lora, **train_kwargs,
            )
            path = res.get("path")
            if path:
                lora_map[name] = path
                _save()
                _log(f"    ✓ LoRA saved → {path}")
            else:
                _log(f"    ✗ training returned no path: {res}")
        except Exception as e:
            _log(f"    ✗ LoRA training failed for {name}: {e}")

    _log(f"\n  {spec['label']} LoRAs ready: {len(lora_map)}/{len(names)}")
    return lora_map


def stage_loras(client: CoderAIClient, image_model: str, out_dir: Path,
                char_names: list, lora_steps: int = 800, lora_rank: int = 16,
                train_base_model: str = None) -> dict:
    """Train one identity LoRA per fighter. Returns {fighter: lora_path}."""
    return _train_profile_loras(client, image_model, out_dir, char_names,
                                "character", lora_steps, lora_rank, train_base_model)


def stage_env_loras(client: CoderAIClient, image_model: str, out_dir: Path,
                    env_names: list, lora_steps: int = 800, lora_rank: int = 16,
                    train_base_model: str = None) -> dict:
    """Train one identity LoRA per environment. Returns {environment: lora_path}."""
    return _train_profile_loras(client, image_model, out_dir, env_names,
                                "environment", lora_steps, lora_rank, train_base_model)


def _generate_keyframes(client: CoderAIClient, image_model: str, keyframe_dir: Path,
                        fight_plan: list, outcome_plan: list, consistency: set,
                        lora_map: dict, char_strength: float, keyframe_steps: int,
                        keyframe_size: str, lora_weight: float,
                        env_lora_map: dict = None, env_lora_weight: float = 0.8):
    """Generate one keyframe still per clip (image model). Saved as PNG keyed by
    the clip's output stem so the render phase can pick them up as init images.
    Resumable: existing PNGs are kept."""
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    use_ip = "ipadapter" in consistency or "keyframe" in consistency
    use_lora = "lora" in consistency

    # Flatten all clips into (stem, prompt, fighters, env) jobs.
    jobs = []
    for m in fight_plan:
        for c in m["clips"]:
            jobs.append((_clip_stem_fight(m["match_name"], c["idx"]),
                         c["prompt"], [m["f1"], m["f2"]], m.get("env")))
    for o in outcome_plan:
        jobs.append((_clip_stem_outcome(o["fighter"], o["outcome"], o.get("match_name")),
                     o["prompt"], [o["fighter"]], o.get("env")))

    _log(f"\n  ── Keyframe phase — {len(jobs)} keyframe image(s) (image model) ──")
    made, skipped, failed = 0, 0, 0
    for k, (stem, prompt, fighters, env) in enumerate(jobs, 1):
        out_png = keyframe_dir / f"{stem}.png"
        if out_png.exists() and out_png.stat().st_size > 0:
            skipped += 1
            continue
        profiles = list(fighters) if use_ip else None
        loras = None
        if use_lora:
            loras = (_lora_specs_for(fighters, lora_map, lora_weight)
                     + _env_lora_specs_for(env, env_lora_map, env_lora_weight)) or None
        kf_prompt = prompt
        if env:
            kf_prompt = f"[{env} location] " + kf_prompt
        try:
            img = _run_with_spinner(
                f"keyframe {k}/{len(jobs)} — {stem}",
                client.generate_image,
                prompt=kf_prompt, model=image_model,
                character_profiles=profiles, loras=loras,
                character_strength=char_strength, size=keyframe_size,
                steps=keyframe_steps,
            )
            out_png.write_bytes(img)
            made += 1
        except Exception as e:
            failed += 1
            _log(f"    ✗ keyframe {stem} failed: {e}")
    _log(f"  ── Keyframes: {made} new, {skipped} reused, {failed} failed ──")


def stage_videos(client: CoderAIClient, video_model: str, out_dir: Path,
                 char_names: list, env_names: list,
                 fps: int = 8, clip_delay: float = 5.0,
                 prompter: "PromptGenerator" = None,
                 prompts_only: bool = False, videos_only: bool = False,
                 num_matches: int = 6, only_outcomes: bool = False,
                 char_descriptions: dict = None,
                 consistency: set = None, image_model: str = None,
                 lora_map: dict = None, char_strength: float = 0.7,
                 keyframe_steps: int = 28, keyframe_size: str = "512x512",
                 lora_weight: float = 0.85, keyframes_only: bool = False,
                 env_lora_map: dict = None, env_lora_weight: float = 0.8):
    _log("\n" + "═" * 60)
    _log("  STAGE 3 — Videos")
    _log("═" * 60)
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    prompts_file = video_dir / "prompts.json"
    keyframe_dir = video_dir / "keyframes"

    # Keyframes-only step: load saved prompts, generate keyframes, stop.
    if keyframes_only:
        if not prompts_file.exists():
            _log(f"  ✗ Keyframe step requires saved prompts, but {prompts_file} not found.\n"
                 f"    Run the Prompts step first.")
            return
        with open(prompts_file) as f:
            saved = json.load(f)
        if not image_model:
            _log("  ✗ Keyframe step requires an image model.")
            return
        _generate_keyframes(client, image_model, keyframe_dir,
                            saved.get("fight_plan", []), saved.get("outcome_plan", []),
                            consistency or {"prompt", "keyframe"}, lora_map or {},
                            char_strength, keyframe_steps, keyframe_size, lora_weight,
                            env_lora_map=env_lora_map or {}, env_lora_weight=env_lora_weight)
        return

    consistency = consistency or {"prompt"}
    lora_map = lora_map or {}
    use_keyframe = "keyframe" in consistency

    if prompter is None:
        prompter = PromptGenerator(client)

    if char_descriptions is None:
        char_descriptions = _build_char_descriptions(out_dir)

    # =========================================================================
    # VIDEOS-ONLY — reuse the prompts written in a previous run.
    # =========================================================================
    if videos_only:
        if not prompts_file.exists():
            _log(f"  ✗ --only-videos requires saved prompts, but {prompts_file} not found.\n"
                 f"    Run with --only-prompts (or a full run) first.")
            return
        with open(prompts_file) as f:
            saved = json.load(f)
        fight_plan = saved.get("fight_plan", [])
        outcome_plan = saved.get("outcome_plan", [])
        if only_outcomes:
            # Render only the outcome clips; the matches were rendered already.
            fight_plan = []
            _log("  --only-outcomes: rendering outcome clips only (skipping matches).")
        total_matches = len(fight_plan)
        total_fight_clips = sum(len(m["clips"]) for m in fight_plan)
        total_outcomes = len(outcome_plan)
        _log(f"\n  Loaded saved prompts from {prompts_file.name}: "
             f"{total_matches} matches ({total_fight_clips} clips) + {total_outcomes} outcome clips")
        # Keyframe phase (image model) BEFORE the video render, so the image
        # model loads once for all keyframes, then the video model loads once.
        if use_keyframe and image_model:
            _generate_keyframes(client, image_model, keyframe_dir,
                                fight_plan, outcome_plan, consistency, lora_map,
                                char_strength, keyframe_steps, keyframe_size, lora_weight,
                                env_lora_map=env_lora_map or {}, env_lora_weight=env_lora_weight)
        # Jump straight to Phase 3 (rendering) below.
        return _stage_videos_render(
            client, video_model, video_dir, fight_plan, outcome_plan,
            total_matches, total_outcomes, fps, clip_delay,
            consistency=consistency, lora_map=lora_map,
            keyframe_dir=keyframe_dir if use_keyframe else None,
            lora_weight=lora_weight,
            env_lora_map=env_lora_map or {}, env_lora_weight=env_lora_weight)

    # =========================================================================
    # PHASE 1 — PLAN every clip up front (no API calls).
    # Swapping between the text model (prompt writing) and the video model is
    # extremely expensive (evict + reload each time).  So we plan all clips,
    # then do ALL text prompts together (Phase 2, text model stays resident),
    # then ALL video renders together (Phase 3, video model stays resident).
    # =========================================================================
    # Build exactly `num_matches` fight pairings (0 when only generating
    # outcomes). Fighters are paired without repeats while the pool lasts; if
    # more matches than unique pairs are requested, fighters are reused with
    # random distinct opponents.
    fighters_pool = list(char_names)
    random.shuffle(fighters_pool)
    pairs = []
    _want_matches = 0 if only_outcomes else max(0, num_matches)
    for _ in range(_want_matches):
        if len(fighters_pool) >= 2:
            pairs.append((fighters_pool.pop(), fighters_pool.pop()))
        elif char_names:
            a = random.choice(char_names)
            b = random.choice([c for c in char_names if c != a] or char_names)
            pairs.append((a, b))
        else:
            break

    # Existing match names on disk so re-runs don't collide with saved matches.
    _used_names = set()
    if prompts_file.exists():
        try:
            for _m in json.loads(prompts_file.read_text()).get("fight_plan", []):
                if _m.get("match_name"):
                    _used_names.add(_m["match_name"])
        except Exception:
            pass

    def _unique_match_name(f1, f2):
        # The same pair (+ environment) can have multiple matches, so names get a
        # numeric suffix when the base name is already taken.
        base = f"match_{f1}_vs_{f2}"
        name, n = base, 2
        while name in _used_names:
            name = f"{base}_{n}"
            n += 1
        _used_names.add(name)
        return name

    # Fight-match plan: each match holds a list of clip specs.
    fight_plan = []
    for f1, f2 in pairs:
        short_target = random.uniform(40, 50)
        long_target  = random.uniform(65, 75)
        env = random.choice(env_names) if env_names else None
        env_desc = _env_description(env) if env else "African township"
        clips_spec, planned, ci = [], 0.0, 0
        while planned < long_target:
            clip_seconds = min(long_target - planned, random.uniform(4, 8))
            round_num = ci // 3 + 1
            intensity = ("early exchanges" if round_num == 1
                         else "midpoint battle" if round_num == 2
                         else "climactic final exchange")
            clips_spec.append({
                "idx": ci, "clip_seconds": clip_seconds,
                "nf": frames_for_seconds(clip_seconds, fps),
                "intensity": intensity, "shot": None, "prompt": None,
            })
            planned += clip_seconds
            ci += 1
        fight_plan.append({
            "f1": f1, "f2": f2, "env": env, "env_desc": env_desc,
            "match_name": _unique_match_name(f1, f2),
            "short_target": short_target, "long_target": long_target,
            "clips": clips_spec,
        })

    # Outcome-clip plan: per MATCH, one clip per participating fighter per
    # outcome — so each match has its own (different) outcome scenes.
    outcomes = ["win", "ko_win", "retire", "draw"]
    outcome_plan = []
    for m in fight_plan:
        for fighter in (m["f1"], m["f2"]):
            for outcome in outcomes:
                target_s = random.uniform(10, 15)
                outcome_plan.append({
                    "match_name": m["match_name"],
                    "fighter": fighter, "outcome": outcome,
                    "env": m["env"], "env_desc": m["env_desc"],
                    "target_s": target_s, "nf": frames_for_seconds(target_s, fps),
                    "shot": None, "prompt": None,
                })

    total_matches = len(fight_plan)
    total_fight_clips = sum(len(m["clips"]) for m in fight_plan)
    total_outcomes = len(outcome_plan)
    _log(f"\n  Planned: {total_matches} matches ({total_fight_clips} fight clips) "
         f"+ {total_outcomes} outcome clips = {total_fight_clips + total_outcomes} videos")

    # =========================================================================
    # PHASE 2 — Generate ALL text prompts first (text model stays loaded).
    # =========================================================================
    _log("\n  ── Phase A — writing all prompts (text model) ──")
    _pidx, _ptot = 0, total_fight_clips + total_outcomes
    for m in fight_plan:
        # Per-match avoid list: every clip in this match steers away from the
        # shots already written for the SAME match, so a 12-clip fight stays
        # varied throughout (not just within the global recent-5 window).
        match_avoid = []
        for c in m["clips"]:
            _pidx += 1
            shot = prompter.fight_shot(
                m["f1"], m["f2"], m["env_desc"],
                match_context=f"Match stage: {c['intensity']}. ",
                avoid=match_avoid)
            c["shot"] = shot
            f1_hint = _fighter_desc_hint(m['f1'], char_descriptions)
            f2_hint = _fighter_desc_hint(m['f2'], char_descriptions)
            c["prompt"] = (
                f"{f1_hint} vs {f2_hint} — {shot} — African township free fight, cinematic"
            )
            match_avoid.append(shot[:60])
            _log(f"  │  [{_pidx}/{_ptot}] {m['f1']} vs {m['f2']} clip{c['idx']:02d}: {shot}")
    for o in outcome_plan:
        _pidx += 1
        shot = prompter.outcome_shot(o["fighter"], o["outcome"], o["env_desc"])
        o["shot"] = shot
        f_hint = _fighter_desc_hint(o["fighter"], char_descriptions)
        o["prompt"] = f"{f_hint} — {shot} — African township fight, cinematic"
        _log(f"  │  [{_pidx}/{_ptot}] {o['fighter']} {o['outcome']}: {shot}")
    _log("  ── Phase A complete — all prompts written ──")

    # Persist the plan + prompts so a later run can render without re-prompting
    # (--only-videos), or so --only-prompts can be inspected / committed.
    # When only generating outcomes, keep any match prompts already on disk so
    # the saved file stays complete (matches + new outcomes).
    save_fight_plan = fight_plan
    if only_outcomes and prompts_file.exists():
        try:
            with open(prompts_file) as f:
                save_fight_plan = json.load(f).get("fight_plan", []) or []
        except Exception:
            save_fight_plan = []
    try:
        with open(prompts_file, "w") as f:
            json.dump({"fight_plan": save_fight_plan, "outcome_plan": outcome_plan,
                       "fps": fps}, f, indent=2)
        _log(f"  Saved prompts → {prompts_file}")
    except Exception as e:
        _log(f"  ⚠ Could not save prompts: {e}")

    if prompts_only:
        _log("  --only-prompts: stopping after prompt generation (no videos rendered).")
        return

    # Keyframe phase (image model) BEFORE rendering, grouped so the image model
    # loads once for all keyframes, then the video model loads once for all clips.
    if use_keyframe and image_model:
        _generate_keyframes(client, image_model, keyframe_dir,
                            fight_plan, outcome_plan, consistency, lora_map,
                            char_strength, keyframe_steps, keyframe_size, lora_weight,
                            env_lora_map=env_lora_map or {}, env_lora_weight=env_lora_weight)

    return _stage_videos_render(
        client, video_model, video_dir, fight_plan, outcome_plan,
        total_matches, total_outcomes, fps, clip_delay,
        consistency=consistency, lora_map=lora_map,
        keyframe_dir=keyframe_dir if use_keyframe else None,
        lora_weight=lora_weight,
        env_lora_map=env_lora_map or {}, env_lora_weight=env_lora_weight)


def _stage_videos_render(client, video_model, video_dir, fight_plan, outcome_plan,
                         total_matches, total_outcomes, fps, clip_delay,
                         consistency=None, lora_map=None, keyframe_dir=None,
                         lora_weight=0.85, env_lora_map=None, env_lora_weight=0.8,
                         progress_cb=None, clip_cb=None):
    """PHASE 3 — render ALL videos from pre-written prompts (video model stays loaded).

    progress_cb(done, total, label) — optional; called after each clip finishes so
    callers (e.g. the web match-render job) can surface per-clip advancement.
    clip_cb(gidx, phase, ok) — optional; phase is "start" (clip gidx begins) or
    "end" (finished, ok=True/False). gidx is a 0-based index over the combined
    sequence of fight clips (in plan order) followed by outcome clips.
    """
    _log("\n  ── Phase B — rendering all videos (video model) ──")
    render_start = time.monotonic()
    consistency = consistency or {"prompt"}
    lora_map = lora_map or {}
    env_lora_map = env_lora_map or {}
    use_lora = "lora" in consistency

    _total_clips = sum(len(m.get("clips", [])) for m in fight_plan) + len(outcome_plan)
    _done_clips = 0
    _gidx = 0  # running index over the combined clip sequence

    def _tick(label=""):
        nonlocal _done_clips
        _done_clips += 1
        if progress_cb:
            try:
                progress_cb(_done_clips, _total_clips, label)
            except Exception:
                pass

    def _clip(phase, ok=None):
        if clip_cb:
            try:
                clip_cb(_gidx, phase, ok)
            except Exception:
                pass

    def _keyframe_bytes(stem: str):
        if not keyframe_dir:
            return None
        p = Path(keyframe_dir) / f"{stem}.png"
        if p.exists() and p.stat().st_size > 0:
            try:
                return p.read_bytes()
            except Exception:
                return None
        return None

    def _render(label, prompt, profiles, env, nf, out_path, stem=None, fighters=None):
        """Render one clip; returns (ok, duration_or_None, fatal)."""
        init_image = _keyframe_bytes(stem) if stem else None
        loras = None
        if use_lora:
            loras = (_lora_specs_for(fighters or profiles or [], lora_map, lora_weight)
                     + _env_lora_specs_for(env, env_lora_map, env_lora_weight)) or None
        try:
            mp4 = _run_with_spinner(
                label, client.generate_video_clip,
                prompt=prompt, model=video_model,
                character_profiles=profiles, environment_name=env,
                num_frames=nf, fps=fps, seed=random.randint(0, 2**31),
                init_image=init_image, loras=loras,
            )
            Path(out_path).write_bytes(mp4)
            return True, (get_video_duration(out_path) or None), False
        except Exception as e:
            if _is_fatal(e):
                _log(f"    ✗ Fatal: {e}")
                return False, None, True
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate limit" in err_str.lower()
            backoff = clip_delay * (4 if is_rate_limit else 2)
            _log(f"    ✗ failed: {e}  (waiting {backoff:.0f}s)")
            time.sleep(backoff)
            return False, None, False

    fatal = False
    rendered_clips = 0

    # 3a. Fight matches
    for i, m in enumerate(fight_plan):
        if fatal:
            break
        elapsed = time.monotonic() - render_start
        _log(f"\n  ┌─ Match {i+1}/{total_matches}: {m['f1']} vs {m['f2']}  "
             f"env={m['env']}  (rendering {elapsed:.0f}s)")
        clips = []
        consecutive_failures = 0
        for c in m["clips"]:
            if fatal:
                break
            if rendered_clips > 0:
                time.sleep(clip_delay)
            clip_stem = _clip_stem_fight(m['match_name'], c['idx'])
            clip_path = video_dir / f"{clip_stem}.mp4"
            _log(f"  │  clip {c['idx']:02d}  {c['clip_seconds']:.1f}s / {c['nf']}f")
            _clip("start")
            ok, dur, is_fatal = _render(
                f"clip {c['idx']:02d} — {m['f1']} vs {m['f2']}",
                c["prompt"], [m["f1"], m["f2"]], m["env"], c["nf"], str(clip_path),
                stem=clip_stem, fighters=[m["f1"], m["f2"]])
            if is_fatal:
                fatal = True
                _clip("end", False)
                break
            if ok:
                clips.append((str(clip_path), dur or c["clip_seconds"]))
                rendered_clips += 1
                consecutive_failures = 0
                _log(f"  │    ✓ {clip_path.name}  ({dur or c['clip_seconds']:.1f}s)")
                _clip("end", True)
                _tick(f"clip {c['idx']:02d} of {m['match_name']}")
            else:
                consecutive_failures += 1
                _clip("end", False)
                _tick(f"clip {c['idx']:02d} failed")
                if consecutive_failures >= 3:
                    _log("  │    ✗ 3 consecutive failures — skipping rest of match")
                    _gidx += 1
                    break
            _gidx += 1

        if not clips:
            _log("  └─ ✗ No clips generated for this match")
            continue

        # Assemble short + long concats from the actual rendered clips.
        short_clips, short_accum, pos = [], 0.0, 0
        while short_accum < m["short_target"] and clips:
            path, dur = clips[pos % len(clips)]
            short_clips.append((path, dur))
            short_accum += dur
            pos += 1
        _write_concat(short_clips, str(video_dir / f"{m['match_name']}_short.mp4"),
                      f"short (~{m['short_target']:.0f}s)")
        _write_concat(clips, str(video_dir / f"{m['match_name']}_long.mp4"),
                      f"long  (~{m['long_target']:.0f}s)")
        _log(f"  └─ match {i+1}/{total_matches} done  ({len(clips)} clips)")

    # 3b. Per-fighter outcome clips
    _log(f"\n  Outcome clips — {total_outcomes} total")
    for oi, o in enumerate(outcome_plan):
        if fatal:
            _log("  ✗ Aborting remaining outcome clips (fatal error)")
            break
        if rendered_clips > 0:
            time.sleep(clip_delay)
        clip_name = _clip_stem_outcome(o['fighter'], o['outcome'], o.get('match_name'))
        out_path = str(video_dir / f"{clip_name}.mp4")
        _log(f"\n  [{oi+1}/{total_outcomes}] {clip_name}  ({o['target_s']:.0f}s, env={o['env']})")
        _clip("start")
        ok, dur, is_fatal = _render(
            f"{clip_name} outcome clip",
            o["prompt"], [o["fighter"]], o["env"], o["nf"], out_path,
            stem=clip_name, fighters=[o["fighter"]])
        if is_fatal:
            fatal = True
            _clip("end", False)
            _gidx += 1
            continue
        if ok:
            rendered_clips += 1
            _log(f"    ✓ {dur or o['target_s']:.1f}s → {clip_name}.mp4")
        _clip("end", bool(ok))
        _tick(f"output {oi+1}/{total_outcomes}")
        _gidx += 1

    _log(f"\n  Videos saved to: {video_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Model auto-selection
# ─────────────────────────────────────────────────────────────────────────────

def pick_model(client: CoderAIClient, kind: str, override: str = None) -> str:
    if override:
        return override
    models = {
        "image": client.list_image_models,
        "video": client.list_video_models,
        "text":  client.list_text_models,
    }[kind]()
    if not models:
        if kind == "text":
            return None   # optional
        raise RuntimeError(
            f"No {kind} generation model found. "
            f"Configure one in CoderAI or pass --{kind}-model MODEL_ID"
        )
    chosen = models[0]["id"]
    _log(f"  Auto-selected {kind} model: {chosen}")
    return chosen


# ─────────────────────────────────────────────────────────────────────────────
# Web UI
# ─────────────────────────────────────────────────────────────────────────────

def launch_web_ui(default_args):
    """Launch a local web interface for Township Fighters content generation.

    Serves on http://localhost:<port> using only the stdlib. The UI has:
      /            — settings form + Start button + live log
      /gallery     — media browser (all produced images/videos)
      /media/<...> — raw file serving for images and videos
      /stream      — SSE endpoint for live log output
      /status      — JSON: {"running": bool, "done": bool}
      /start       — POST: start the generation with submitted options
      /stop        — POST: request graceful abort
    """
    import http.server
    import socketserver
    import urllib.parse
    import mimetypes
    import queue as _queue_mod

    port = getattr(default_args, 'web_port', 7788)
    out_dir = Path(default_args.out_dir)
    # Apply any saved global-prompt overrides so the UI + runs use them.
    apply_prompts_config(load_prompts_config(out_dir))

    # Shared state
    _state = {
        "running": False,
        "done": False,
        "current": "",            # label of the run currently/last executing
        "log_lines": [],          # all lines so far (for late-joining SSE clients)
        "abort": threading.Event(),
        "jobs": {},               # job_id -> {status, progress, output, error}
    }
    _log_q: "_queue_mod.Queue[Optional[str]]" = _queue_mod.Queue()
    _sse_clients: list = []
    _sse_lock = threading.Lock()
    _jobs_lock = threading.Lock()

    def _ffmpeg_exe():
        """Return an ffmpeg binary path, preferring the system one."""
        import shutil as _sh
        fb = _sh.which("ffmpeg")
        if fb:
            return fb
        try:
            import imageio_ffmpeg as _iiff
            return _iiff.get_ffmpeg_exe()
        except Exception:
            raise RuntimeError("ffmpeg not found. Install it or pip install imageio-ffmpeg.")

    def _probe_video(fpath: Path) -> dict:
        """Return {width, height, fps, duration} for a video file via ffprobe."""
        import subprocess as _sp, json as _j, shutil as _sh
        ffprobe = _sh.which("ffprobe")
        if not ffprobe:
            return {}
        try:
            r = _sp.run([ffprobe, "-v", "quiet", "-print_format", "json",
                         "-show_streams", str(fpath)],
                        capture_output=True, text=True, timeout=10)
            data = _j.loads(r.stdout)
            for s in data.get("streams", []):
                if s.get("codec_type") == "video":
                    w = int(s.get("width", 0))
                    h = int(s.get("height", 0))
                    fps_str = s.get("r_frame_rate", "0/1")
                    try:
                        num, den = fps_str.split("/")
                        fps = round(int(num) / int(den), 2) if int(den) else 0
                    except Exception:
                        fps = 0
                    dur = float(s.get("duration", 0) or data.get("format", {}).get("duration", 0) or 0)
                    return {"width": w, "height": h, "fps": fps, "duration": dur}
        except Exception:
            pass
        return {}

    def _run_process_job(job_id: str, fpath: Path, op: str, param):
        """Send a video to CoderAI for upscaling / FPS interpolation."""
        with _jobs_lock:
            _state["jobs"][job_id] = {"status": "running", "progress": 5,
                                      "output": None, "error": None}

        stem   = fpath.stem
        suffix = fpath.suffix
        client = CoderAIClient(default_args.base_url,
                               getattr(default_args, 'api_key', None),
                               timeout=7200)

        def _set_progress(pct, msg=""):
            with _jobs_lock:
                _state["jobs"][job_id]["progress"] = pct
                if msg:
                    _state["jobs"][job_id]["_msg"] = msg

        def _finish(out_path: Path):
            rel = str(out_path.relative_to(out_dir)).replace("\\", "/")
            with _jobs_lock:
                _state["jobs"][job_id].update({
                    "status": "done", "progress": 100,
                    "output": rel, "output_name": out_path.name,
                })

        def _fail(msg: str):
            with _jobs_lock:
                _state["jobs"][job_id].update({"status": "error", "error": msg})

        try:
            video_bytes = fpath.read_bytes()

            if op == "upscale":
                scale = int(param)
                out_path = fpath.parent / f"{stem}_x{scale}{suffix}"
                _set_progress(10, f"Sending to CoderAI for {scale}× upscale…")
                result = client.upscale_video(video_bytes, factor=scale)
                out_path.write_bytes(result)
                _finish(out_path)

            elif op == "fps":
                probe  = _probe_video(fpath)
                src_fps = probe.get("fps", 8) or 8
                target  = int(param)
                # fps_multiplier is the ratio; CoderAI expects an integer multiplier
                mult = max(2, round(target / src_fps))
                out_path = fpath.parent / f"{stem}_fps{target}{suffix}"
                _set_progress(10, f"Sending to CoderAI for FPS interpolation (×{mult})…")
                result = client.interpolate_video(video_bytes, fps_multiplier=mult)
                out_path.write_bytes(result)
                _finish(out_path)

            elif op == "upscale_fps":
                scale, target_fps = int(param[0]), int(param[1])
                # Step 1: upscale
                out_up = fpath.parent / f"{stem}_x{scale}_fps{target_fps}{suffix}"
                _set_progress(10, f"Step 1/2 — {scale}× upscale via CoderAI…")
                upscaled = client.upscale_video(video_bytes, factor=scale)
                _set_progress(55, f"Step 2/2 — FPS interpolation to {target_fps}fps via CoderAI…")
                # compute multiplier from the source clip's FPS
                probe  = _probe_video(fpath)
                src_fps = probe.get("fps", 8) or 8
                mult = max(2, round(target_fps / src_fps))
                result = client.interpolate_video(upscaled, fps_multiplier=mult)
                out_up.write_bytes(result)
                _finish(out_up)

            else:
                _fail(f"Unknown operation: {op}")

        except Exception as exc:
            _fail(str(exc))

    def _next_ref_path(base: Path, ext: str = ".png") -> Path:
        """Return the next free ref_NN path in a profile folder. The index is
        unique across extensions so ref_00.png and ref_00.jpg can't coexist."""
        i = 0
        while True:
            if not any((base / f"ref_{i:02d}{e}").exists()
                       for e in (".png", ".jpg", ".jpeg", ".webp", ".gif")):
                return base / f"ref_{i:02d}{ext}"
            i += 1

    def _run_regen_job(job_id: str, kind: str, name: str, count: int, guide: bool):
        """Generate `count` NEW reference images for a profile and APPEND them,
        preserving every existing (non-deleted) image. Runs server-side image
        generation; updates job progress for the profile page to poll."""
        with _jobs_lock:
            _state["jobs"][job_id] = {"status": "running", "progress": 3,
                                      "output": None, "error": None,
                                      "_msg": "starting…", "added": 0,
                                      "kind": kind, "name": name, "jtype": "regen"}

        def _prog(pct, msg=""):
            with _jobs_lock:
                _state["jobs"][job_id]["progress"] = pct
                if msg:
                    _state["jobs"][job_id]["_msg"] = msg
            if msg:
                print(f"  [regen {name}] {msg}", flush=True)

        def _fail(msg):
            with _jobs_lock:
                _state["jobs"][job_id].update({"status": "error", "error": msg})

        try:
            base = out_dir / (kind + "s") / name
            meta = {}
            try:
                meta = json.loads((base / "meta.json").read_text())
            except Exception:
                pass
            # Build a generation prompt from the saved profile.
            prompt = (meta.get("prompt") or meta.get("description") or name).strip()
            if kind == "environment":
                size = "768x512"
            else:
                size = "512x512"

            client = CoderAIClient(default_args.base_url,
                                   getattr(default_args, "api_key", None))
            _prog(8, "selecting image model…")
            model = getattr(default_args, "image_model", None)
            if not model:
                try:
                    model = pick_model(client, "image", None)
                except Exception as e:
                    _fail(f"no image model available: {e}")
                    return

            # Guide new images with the surviving references via IP-Adapter so
            # regenerated refs match the ones the user kept. Characters use
            # character_profiles; environments use environment_profiles.
            char_p = [name] if (guide and kind == "character") else None
            env_p = [name] if (guide and kind == "environment") else None

            base.mkdir(parents=True, exist_ok=True)
            added_uris = []
            for k in range(count):
                _prog(int(10 + 80 * k / max(1, count)),
                      f"generating image {k+1}/{count}…")
                try:
                    img = client.generate_image(
                        prompt=prompt, model=model,
                        character_profiles=char_p, environment_profiles=env_p,
                        character_strength=0.7,
                        size=size, steps=28, seed=random.randint(0, 2**31),
                    )
                except Exception as e:
                    _web_log(f"  ✗ regen image {k+1}/{count} for {name} failed: {e}")
                    continue
                out_png = _next_ref_path(base)
                out_png.write_bytes(img)
                added_uris.append("data:image/png;base64," +
                                  base64.b64encode(img).decode())

            if not added_uris:
                _fail("no images were generated")
                return

            # Append to the CoderAI server profile too (best-effort), so video
            # and keyframe generation that resolves this profile sees them.
            _prog(94, "syncing new images to CoderAI…")
            synced = True
            try:
                client.patch_profile(kind, name, add_images=added_uris)
            except Exception:
                synced = False

            with _jobs_lock:
                _state["jobs"][job_id].update({
                    "status": "done", "progress": 100,
                    "added": len(added_uris), "synced": synced,
                    "_msg": f"added {len(added_uris)} image(s)",
                })
        except Exception as exc:
            _fail(str(exc))

    def _run_train_lora_job(job_id: str, kind: str, name: str, steps: int, rank: int):
        """Train one profile's identity LoRA (server-side, blocking) while
        polling the server's progress so the profile page shows live step
        counts. On success records the path in loras.json / env_loras.json."""
        with _jobs_lock:
            _state["jobs"][job_id] = {"status": "running", "progress": 2,
                                      "output": None, "error": None,
                                      "_msg": "starting…",
                                      "kind": kind, "name": name, "jtype": "train"}

        def _prog(pct, msg=""):
            with _jobs_lock:
                _state["jobs"][job_id]["progress"] = max(2, min(99, int(pct)))
                if msg:
                    _state["jobs"][job_id]["_msg"] = msg
            if msg:
                print(f"  [train {name}] {msg}", flush=True)

        def _fail(msg):
            with _jobs_lock:
                _state["jobs"][job_id].update({"status": "error", "error": msg})

        try:
            client = CoderAIClient(default_args.base_url,
                                   getattr(default_args, "api_key", None))
            _prog(4, "selecting image model…")
            model = getattr(default_args, "image_model", None)
            if not model:
                try:
                    model = pick_model(client, "image", None)
                except Exception as e:
                    _fail(f"no image model available: {e}")
                    return

            # Server-side training pulls reference images from the CoderAI copy
            # of the profile — make sure the local profile is uploaded first, or
            # training would fail instantly with "no training images".
            _prog(5, "ensuring profile is on CoderAI…")
            try:
                _ensure_in_coderai(client, kind, name, out_dir)
            except Exception:
                pass

            prefix = "fighter_" if kind == "character" else "env_"
            lora_name = f"{prefix}{name}"
            _web_log(f"  🧠 Training {kind} LoRA '{lora_name}' "
                     f"({steps} steps, rank {rank})…")

            # Run the blocking train call in an inner thread; poll progress here.
            result, err = {}, {}
            def _do():
                try:
                    kwargs = dict(name=lora_name, base_model=model,
                                  steps=int(steps), rank=int(rank))
                    _tbm = getattr(default_args, "lora_train_base_model", None) or None
                    if _tbm:
                        kwargs["train_base_model"] = _tbm
                    kwargs[kind] = name
                    result["res"] = client.train_lora(**kwargs)
                except Exception as e:
                    err["e"] = str(e)
            t = threading.Thread(target=_do, daemon=True)
            t.start()
            _prog(6, "training…")
            while t.is_alive():
                time.sleep(1.5)
                try:
                    p = client.lora_progress()
                except Exception:
                    continue
                if p.get("name") and p.get("name") != lora_name:
                    continue
                total = p.get("total") or steps
                step = p.get("step") or 0
                pct = 6 + int(90 * step / max(1, total))
                _prog(pct, p.get("message") or p.get("status") or "training")
            t.join()

            if err:
                _web_log(f"  ✗ LoRA training failed for {name}: {err['e']}")
                _fail(err["e"])
                return
            res = result.get("res") or {}
            path = res.get("path")
            if not path:
                _fail(f"training returned no path: {res}")
                return

            # Record the trained LoRA in the on-disk map so video/keyframe runs reuse it.
            map_file = out_dir / ("loras.json" if kind == "character" else "env_loras.json")
            try:
                lmap = json.loads(map_file.read_text()) if map_file.exists() else {}
            except Exception:
                lmap = {}
            lmap[name] = path
            try:
                map_file.write_text(json.dumps(lmap, indent=2))
            except Exception:
                pass
            _web_log(f"  ✓ LoRA trained → {path}")
            with _jobs_lock:
                _state["jobs"][job_id].update({
                    "status": "done", "progress": 100, "path": path,
                    "_msg": "training complete",
                })
        except Exception as exc:
            _fail(str(exc))

    def _run_match_job(job_id: str, scope: str, params: dict):
        """Regenerate part of a match: re-render clips (video model), re-render
        outcome outputs, or reassemble the final short/long videos from existing
        clips (no model). Detailed progress streams to the Run-page log."""
        with _jobs_lock:
            _state["jobs"][job_id] = {"status": "running", "progress": 3,
                                      "output": None, "error": None,
                                      "_msg": "starting…", "jtype": "match",
                                      "scope": scope, "match": params.get("match")}

        def _prog(pct, msg=""):
            with _jobs_lock:
                _state["jobs"][job_id]["progress"] = max(2, min(99, int(pct)))
                if msg:
                    _state["jobs"][job_id]["_msg"] = msg
            if msg:
                print(f"  [match] {msg}", flush=True)

        def _fail(msg):
            with _jobs_lock:
                _state["jobs"][job_id].update({"status": "error", "error": msg})

        def _done(msg):
            with _jobs_lock:
                j = _state["jobs"][job_id]
                # Any item still left pending/rendering is now finished.
                for it in j.get("items", []):
                    if it.get("status") in ("pending", "rendering"):
                        it["status"] = "done"
                j.update({"status": "done", "progress": 100, "_msg": msg})

        def _set_items(labels):
            with _jobs_lock:
                _state["jobs"][job_id]["items"] = [
                    {"label": lbl, "status": "pending"} for lbl in labels]

        def _item(gidx, phase, ok=None):
            st = "rendering" if phase == "start" else ("done" if ok else "failed")
            with _jobs_lock:
                items = _state["jobs"][job_id].get("items") or []
                if 0 <= gidx < len(items):
                    items[gidx]["status"] = st

        def _load_map(fname):
            fp = out_dir / fname
            if fp.exists():
                try:
                    return json.loads(fp.read_text()) or {}
                except Exception:
                    return {}
            return {}

        try:
            import sys as _sys
            _sys.modules[__name__]._log = _patched_log  # stream detail to web log
            vdir = out_dir / "videos"
            pf = vdir / "prompts.json"
            data = {}
            if pf.exists():
                try:
                    data = json.loads(pf.read_text())
                except Exception:
                    data = {}
            fight_plan = data.get("fight_plan", [])
            outcome_plan = data.get("outcome_plan", [])
            fps = int(data.get("fps") or getattr(default_args, "fps", 8))
            match_name = params.get("match")

            # ── Reassemble only: no model needed ───────────────────────────────
            if scope == "reassemble":
                m = next((x for x in fight_plan if x.get("match_name") == match_name), {})
                st = float(m.get("short_target", 45)); lt = float(m.get("long_target", 70))
                _prog(25, "reassembling final videos…")
                n = _reassemble_finals(vdir, match_name, st, lt)
                if n == 0:
                    _fail("no clips found to reassemble")
                    return
                _done(f"reassembled from {n} clip(s)")
                return

            # ── Render scopes: need the video model + consistency settings ─────
            client = CoderAIClient(default_args.base_url,
                                   getattr(default_args, "api_key", None))
            _prog(5, "selecting video model…")
            video_model = getattr(default_args, "video_model", None)
            if not video_model:
                try:
                    video_model = pick_model(client, "video", None)
                except Exception as e:
                    _fail(f"no video model available: {e}")
                    return
            consistency = parse_consistency(getattr(default_args, "consistency", "keyframe"))
            lora_map = _load_map("loras.json")
            env_lora_map = _load_map("env_loras.json")
            keyframe_dir = vdir / "keyframes" if "keyframe" in consistency else None
            clip_delay = float(getattr(default_args, "clip_delay", 5.0))
            lw = float(getattr(default_args, "lora_weight", 0.85))
            elw = float(getattr(default_args, "env_lora_weight", 0.8))

            if scope in ("match-clips", "clip"):
                m = next((x for x in fight_plan if x.get("match_name") == match_name), None)
                if not m:
                    _fail("match not found in prompts.json — render it from the Run page first")
                    return
                mm = dict(m)
                if scope == "clip":
                    idx = int(params.get("idx"))
                    mm["clips"] = [c for c in m["clips"] if int(c["idx"]) == idx]
                    if not mm["clips"]:
                        _fail("clip not found in prompts.json")
                        return
                _set_items([f"clip {int(c['idx']):02d}" for c in mm["clips"]])
                _prog(8, f"rendering {len(mm['clips'])} clip(s)…")

                def _cb(done, total, label):
                    pct = 8 + int(88 * done / max(1, total))
                    _prog(pct, f"clip {done}/{total} done"
                               + (f" — {label}" if label else ""))

                _stage_videos_render(
                    client, video_model, vdir, [mm], [], 1, 0, fps, clip_delay,
                    consistency=consistency, lora_map=lora_map,
                    keyframe_dir=keyframe_dir, lora_weight=lw,
                    env_lora_map=env_lora_map, env_lora_weight=elw,
                    progress_cb=_cb, clip_cb=_item)
                _done(f"re-rendered {len(mm['clips'])} clip(s)")
                return

            if scope in ("outcomes", "outcome"):
                fighter = params.get("fighter")
                outcome = params.get("outcome")
                if scope == "outcome":
                    sel = [o for o in outcome_plan
                           if o.get("fighter") == fighter and o.get("outcome") == outcome
                           and (not match_name or o.get("match_name") == match_name)]
                elif match_name:
                    # All outcomes of this match.
                    sel = [o for o in outcome_plan if o.get("match_name") == match_name]
                elif fighter:
                    sel = [o for o in outcome_plan if o.get("fighter") == fighter]
                else:
                    sel = list(outcome_plan)
                if not sel:
                    _fail("no matching outputs in prompts.json")
                    return
                _set_items([_clip_stem_outcome(o['fighter'], o['outcome'],
                                                o.get('match_name')) for o in sel])
                _prog(8, f"rendering {len(sel)} output clip(s)…")

                def _cb(done, total, label):
                    pct = 8 + int(88 * done / max(1, total))
                    _prog(pct, f"output {done}/{total} done"
                               + (f" — {label}" if label else ""))

                _stage_videos_render(
                    client, video_model, vdir, [], sel, 0, len(sel), fps, clip_delay,
                    consistency=consistency, lora_map=lora_map,
                    keyframe_dir=keyframe_dir, lora_weight=lw,
                    env_lora_map=env_lora_map, env_lora_weight=elw,
                    progress_cb=_cb, clip_cb=_item)
                _done(f"re-rendered {len(sel)} output(s)")
                return

            _fail(f"unknown scope: {scope}")
        except Exception as exc:
            _fail(str(exc))

    def _web_log(msg: str):
        """Override _log so output goes to both stdout and the web log queue."""
        print(msg, flush=True)
        _state["log_lines"].append(msg)
        _log_q.put(msg)
        with _sse_lock:
            for q in list(_sse_clients):
                try:
                    q.put(msg)
                except Exception:
                    pass

    # Patch the module-level _log to also feed the web UI
    import sys as _sys
    _self_mod = _sys.modules[__name__]
    _orig_log = _self_mod._log

    def _patched_log(*args, **kwargs):
        import io
        buf = io.StringIO()
        print(*args, **kwargs, file=buf)
        _web_log(buf.getvalue().rstrip("\n"))

    # ── HTML pages ──────────────────────────────────────────────────────────

    _CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#0f0f0f;color:#e0e0e0;min-height:100vh}
a{color:#7eb8f7;text-decoration:none}a:hover{text-decoration:underline}
h1{font-size:1.4rem;font-weight:700;margin-bottom:.75rem}
h2{font-size:1.1rem;font-weight:600;margin-bottom:.5rem}
.nav{background:#1a1a1a;border-bottom:1px solid #333;padding:.6rem 1.2rem;display:flex;gap:1.2rem;align-items:center}
.nav span{font-weight:700;color:#f5a623;margin-right:.5rem}
.container{max-width:960px;margin:0 auto;padding:1.2rem}
.card{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:1rem;margin-bottom:1rem}
label{display:block;font-size:.8rem;color:#aaa;margin-bottom:.2rem;margin-top:.6rem}
label:first-child{margin-top:0}
input[type=text],input[type=number],input[type=url],select{
  background:#111;border:1px solid #333;color:#e0e0e0;padding:.35rem .5rem;
  border-radius:4px;width:100%;font-size:.85rem}
input[type=checkbox]{width:auto;accent-color:#f5a623;margin-right:.3rem}
.row{display:grid;grid-template-columns:1fr 1fr;gap:.75rem}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.75rem}
.hint{font-size:.72rem;color:#666;margin-top:.15rem}
.btn{display:inline-block;padding:.45rem 1.1rem;border-radius:5px;border:none;
     cursor:pointer;font-size:.9rem;font-weight:600}
.btn-primary{background:#f5a623;color:#000}.btn-primary:hover{background:#e8960f}
.btn-danger{background:#c0392b;color:#fff}.btn-danger:hover{background:#a93226}
.btn-secondary{background:#2a2a2a;color:#ccc;border:1px solid #444}
.btn-secondary:hover{background:#333}
#log-box{background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:.75rem;
         height:340px;overflow-y:auto;font-family:monospace;font-size:.78rem;
         line-height:1.55;white-space:pre-wrap;word-break:break-all}
#log-box .info{color:#9ad89a}#log-box .warn{color:#f5c842}#log-box .err{color:#e07070}
#log-box .head{color:#f5a623;font-weight:700}
.status-pill{display:inline-block;padding:.2rem .55rem;border-radius:10px;
             font-size:.72rem;font-weight:700}
.status-idle{background:#333;color:#888}
.status-run{background:#1a4a1a;color:#7ed87e}
.status-done{background:#1a1a4a;color:#7ea8f7}
.gallery-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:.75rem}
.media-card{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:6px;overflow:hidden}
.media-card img{width:100%;display:block;height:160px;object-fit:cover;background:#111}
.media-card video{width:100%;display:block;height:160px;object-fit:cover;background:#111}
.media-card .mc-label{padding:.4rem .5rem;font-size:.72rem;color:#999;
                       white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.section-title{font-size:.85rem;font-weight:700;color:#aaa;
               text-transform:uppercase;letter-spacing:.05em;margin:1rem 0 .4rem}
.mc-actions{display:flex;gap:.3rem;padding:.3rem .4rem;border-top:1px solid #222;flex-wrap:wrap}
.mc-actions button{font-size:.68rem;padding:.18rem .45rem;border-radius:3px;border:none;
                   cursor:pointer;background:#2a2a2a;color:#ccc;font-weight:500}
.mc-actions button:hover{background:#3a3a3a;color:#fff}
.mc-actions button.active{background:#1a3a1a;color:#7ed87e}
.mc-info{font-size:.67rem;color:#555;padding:.15rem .45rem .3rem;font-family:monospace}
/* modal */
.modal-bg{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:100;
          align-items:center;justify-content:center}
.modal-bg.open{display:flex}
.modal{background:#1a1a1a;border:1px solid #333;border-radius:10px;padding:1.2rem;
       min-width:340px;max-width:480px;width:90%}
.modal h3{font-size:1rem;font-weight:700;margin-bottom:.75rem}
.modal .row2{display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin:.5rem 0}
.modal label{font-size:.8rem;color:#aaa;display:block;margin-bottom:.2rem}
.modal select,.modal input{background:#111;border:1px solid #333;color:#e0e0e0;
                            padding:.3rem .45rem;border-radius:4px;width:100%;font-size:.82rem}
.progress-bar{background:#222;border-radius:4px;height:8px;margin:.6rem 0;overflow:hidden}
.progress-fill{height:100%;background:#f5a623;border-radius:4px;transition:width .4s}
.progress-fill.ok{background:#3a9d3a}
.progress-fill.fail{background:#c0392b}
.progress-fill.striped{background:repeating-linear-gradient(45deg,#f5a623,#f5a623 8px,#d98e0f 8px,#d98e0f 16px);animation:prgmove 1s linear infinite}
@keyframes prgmove{from{background-position:0 0}to{background-position:32px 0}}
#match-progress{margin:.6rem 0}
#match-progress.hidden{display:none}
.prg-global .prg-label{font-size:.78rem;color:#bbb;margin-bottom:.1rem;font-weight:600}
.prg-items{display:flex;flex-direction:column;gap:.15rem;margin-top:.5rem}
.prg-item .prg-ilabel{font-size:.72rem;color:#999}
.prg-item .progress-bar{height:6px;margin:.12rem 0}
.job-status{font-size:.78rem;margin-top:.4rem;min-height:1.2rem}
.job-status.done{color:#7ed87e}.job-status.error{color:#e07070}
/* profile editor */
textarea{background:#111;border:1px solid #333;color:#e0e0e0;padding:.35rem .5rem;
         border-radius:4px;width:100%;font-size:.85rem;font-family:inherit;
         resize:vertical;min-height:3rem}
.pf-head{display:flex;justify-content:space-between;align-items:center;gap:.6rem}
.pf-name{font-weight:700;color:#f5a623;font-size:1.05rem}
.pf-thumbs{display:flex;gap:.4rem;flex-wrap:wrap;margin:.5rem 0}
.pf-thumb{position:relative;width:92px;height:92px}
.pf-thumb img{width:92px;height:92px;object-fit:cover;border-radius:4px;background:#111;cursor:zoom-in}
/* image lightbox (click a thumbnail to enlarge) */
.lightbox-bg{display:none;position:fixed;inset:0;background:rgba(0,0,0,.9);z-index:200;
             align-items:center;justify-content:center;cursor:zoom-out;padding:1.5rem}
.lightbox-bg.open{display:flex}
.lightbox-bg img{max-width:95vw;max-height:95vh;object-fit:contain;border-radius:8px;
                 box-shadow:0 0 40px rgba(0,0,0,.8)}
.pf-thumb-del{position:absolute;top:2px;right:2px;background:rgba(192,57,43,.92);color:#fff;
              border:none;border-radius:3px;cursor:pointer;font-size:.7rem;
              width:18px;height:18px;line-height:1;padding:0}
.pf-thumb-del:hover{background:#c0392b}
.pf-status{font-size:.76rem;color:#7ed87e;min-height:1.1rem;margin-left:.5rem}
.pf-actions{display:flex;gap:.5rem;align-items:center;margin-top:.7rem}
"""

    # Shared modal dialog + promise-based helpers (uiConfirm/uiAlert/uiPrompt),
    # injected into every page so we never use the browser's alert/confirm/prompt.
    # Plain string (not an f-string): its braces are literal JS/HTML.
    _modal_block = """
<div class=modal-bg id=ui-modal>
  <div class=modal>
    <h3 id=ui-modal-title></h3>
    <div id=ui-modal-body style="font-size:.86rem;color:#cfcfcf;white-space:pre-wrap"></div>
    <div id=ui-modal-input-wrap style="display:none;margin-top:.7rem">
      <input type=text id=ui-modal-input autocomplete=off>
    </div>
    <div style="display:flex;gap:.5rem;margin-top:1.1rem;justify-content:flex-end">
      <button class="btn btn-secondary" id=ui-modal-cancel type=button>Cancel</button>
      <button class="btn btn-primary" id=ui-modal-ok type=button>OK</button>
    </div>
  </div>
</div>
<script>
(function(){
  const bg=document.getElementById('ui-modal');
  const titleEl=document.getElementById('ui-modal-title');
  const bodyEl=document.getElementById('ui-modal-body');
  const inWrap=document.getElementById('ui-modal-input-wrap');
  const inEl=document.getElementById('ui-modal-input');
  const okBtn=document.getElementById('ui-modal-ok');
  const cancelBtn=document.getElementById('ui-modal-cancel');
  let _res=null, _hasInput=false;
  function done(val){ bg.classList.remove('open'); const r=_res; _res=null; if(r) r(val); }
  function open(o){
    return new Promise(res=>{
      _res=res; _hasInput=(o.input!==undefined);
      titleEl.textContent=o.title||'';
      bodyEl.textContent=o.message||'';
      okBtn.textContent=o.okText||'OK';
      okBtn.className='btn '+(o.danger?'btn-danger':'btn-primary');
      cancelBtn.style.display=(o.cancel===false)?'none':'';
      if(_hasInput){ inWrap.style.display='block'; inEl.value=o.input||'';
        setTimeout(()=>{inEl.focus();inEl.select();},40); }
      else { inWrap.style.display='none'; }
      bg.classList.add('open');
    });
  }
  okBtn.onclick=()=>done(_hasInput?inEl.value:true);
  cancelBtn.onclick=()=>done(_hasInput?null:false);
  bg.onclick=e=>{ if(e.target===bg) done(_hasInput?null:false); };
  inEl.addEventListener('keydown',e=>{ if(e.key==='Enter'){e.preventDefault();okBtn.click();} });
  document.addEventListener('keydown',e=>{ if(bg.classList.contains('open')&&e.key==='Escape') cancelBtn.click(); });
  window.uiConfirm=(message,o)=>open(Object.assign({title:'Confirm',message:message,okText:'OK'},o||{}));
  window.uiAlert=(message,o)=>open(Object.assign({title:'Notice',message:message,okText:'OK',cancel:false},o||{}));
  window.uiPrompt=(message,def,o)=>open(Object.assign({title:'Input',message:message,input:(def||''),okText:'Save'},o||{}));
})();
</script>"""

    # Shared image lightbox: call showImg(src, alt) to enlarge any image; click
    # the backdrop or press Escape to close. Injected into every page.
    _lightbox_block = """
<div class=lightbox-bg id=img-lightbox onclick="this.classList.remove('open')">
  <img id=img-lightbox-img src="" alt="">
</div>
<script>
(function(){
  window.showImg=function(src,alt){
    var bg=document.getElementById('img-lightbox');
    var im=document.getElementById('img-lightbox-img');
    if(!bg||!im) return;
    im.src=src; im.alt=alt||'';
    bg.classList.add('open');
  };
  document.addEventListener('keydown',function(e){
    if(e.key==='Escape'){var bg=document.getElementById('img-lightbox'); if(bg) bg.classList.remove('open');}
  });
})();
</script>"""

    def _page(title, body, active="run"):
        nav_items = [
            ("run", "/", "▶ Run"),
            ("characters", "/characters", "👤 Characters"),
            ("environments", "/environments", "🏞 Environments"),
            ("matches", "/matches", "🥊 Matches"),
            ("prompts", "/prompts", "✍ Prompts"),
            ("gallery", "/gallery", "🎬 Gallery"),
        ]
        nav = "".join(
            f'<a href="{href}" style="{"color:#f5a623;font-weight:700" if k==active else ""}">{label}</a>'
            for k, href, label in nav_items
        )
        return f"""<!doctype html><html lang=en><head>
<meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>{title} — Township Fighters</title>
<style>{_CSS}</style></head><body>
<div class=nav><span>⚔ Township Fighters</span>{nav}</div>
<div class=container>{body}</div>
{_modal_block}
{_lightbox_block}
</body></html>"""

    def _run_page_html(args_ns):
        import json as _json
        def _v(attr, default=""): return getattr(args_ns, attr, default)
        def _c(attr): return " checked" if getattr(args_ns, attr, False) else ""

        # If the script was launched with -c/--config, the Save button defaults
        # to that same path so saving overwrites the loaded config file.
        _cfg_arg = getattr(args_ns, "config", None)
        _save_default = os.path.abspath(_cfg_arg) if _cfg_arg else "township_config.json"
        _save_default_js = _json.dumps(_save_default)

        char_mode = ("reuse" if _v("reuse_fighters") else
                     "fighters" if _v("fighters") else
                     "skip" if _v("skip_characters") else "generate")
        env_mode  = ("reuse" if _v("reuse_environments") else
                     "environments" if _v("environments") else
                     "skip" if _v("skip_environments") else "generate")
        stage3_mode = ("only_prompts" if _v("only_prompts") else
                       "only_videos" if _v("only_videos") else "full")
        only_stage = ("characters" if _v("only_characters") else
                      "environments" if _v("only_environments") else
                      "assets" if _v("only_assets") else "all")

        return f"""
<h1>Run settings</h1>
<form id=run-form method=post action=/start>
<div class=card>
  <h2>Connection</h2>
  <div class=row>
    <div><label>CoderAI base URL</label>
         <input name=base_url type=url value="{_v('base_url','http://127.0.0.1:8776')}"></div>
    <div><label>API key <span class=hint>(leave blank if auth disabled)</span></label>
         <input name=api_key type=text value="{_v('api_key') or ''}"></div>
  </div>
</div>

<div class=card>
  <h2>Models <span class=hint>(blank = auto-select)</span></h2>
  <div class=row3>
    <div><label>Image model</label><input name=image_model type=text value="{_v('image_model') or ''}"></div>
    <div><label>Video model</label><input name=video_model type=text value="{_v('video_model') or ''}"></div>
    <div><label>Text / LLM model</label><input name=text_model type=text value="{_v('text_model') or ''}"></div>
  </div>
  <div style="margin-top:.6rem">
    <label><input type=checkbox name=no_llm{_c('no_llm')}> Disable LLM prompt generation</label>
  </div>
</div>

<div class=card>
  <h2>Output</h2>
  <div class=row>
    <div><label>Output directory</label>
         <input name=out_dir type=text value="{_v('out_dir','./township_output')}"></div>
    <div><label>Region filter <span class=hint>(kampala, soweto, jinja… or blank)</span></label>
         <input name=region type=text value="{_v('region') or ''}"></div>
  </div>
  <div style="margin-top:.6rem">
    <label><input type=checkbox name=include_female{_c('include_female')}> Include female fighters</label>
  </div>
</div>

<div class=card>
  <h2>Stage 1 — Characters</h2>
  <label>Mode</label>
  <select name=char_mode id=char_mode onchange="toggleCharFields()">
    <option value=generate{"selected" if char_mode=="generate" else ""}>Generate new characters</option>
    <option value=reuse{"selected" if char_mode=="reuse" else ""}>Reuse all existing in CoderAI</option>
    <option value=skip{"selected" if char_mode=="skip" else ""}>Skip (use saved / pool names)</option>
    <option value=fighters{"selected" if char_mode=="fighters" else ""}>Use specific names</option>
  </select>
  <div id=char_names_row style="display:{"block" if char_mode=="fighters" else "none"};margin-top:.4rem">
    <label>Fighter names <span class=hint>(comma-separated)</span></label>
    <input name=fighters type=text value="{_v('fighters') or ''}">
  </div>
</div>

<div class=card>
  <h2>Stage 2 — Environments</h2>
  <label>Mode</label>
  <select name=env_mode id=env_mode onchange="toggleEnvFields()">
    <option value=generate{"selected" if env_mode=="generate" else ""}>Generate new environments</option>
    <option value=reuse{"selected" if env_mode=="reuse" else ""}>Reuse all existing in CoderAI</option>
    <option value=skip{"selected" if env_mode=="skip" else ""}>Skip (use saved / pool names)</option>
    <option value=environments{"selected" if env_mode=="environments" else ""}>Use specific names</option>
  </select>
  <div id=env_names_row style="display:{"block" if env_mode=="environments" else "none"};margin-top:.4rem">
    <label>Environment names <span class=hint>(comma-separated)</span></label>
    <input name=environments type=text value="{_v('environments') or ''}">
  </div>
</div>

<div class=card>
  <h2>Stage 3 — Videos</h2>
  <div class=row>
    <div><label>Number of fight matches</label>
         <input name=matches type=number min=0 max=50 value="{_v('matches', 6)}"></div>
    <div><label>FPS</label>
         <input name=fps type=number min=1 max=30 value="{_v('fps', 8)}"></div>
  </div>
  <div class=row style="margin-top:.4rem">
    <div><label>Clip delay between requests (seconds)</label>
         <input name=clip_delay type=number min=0 step=0.5 value="{_v('clip_delay', 5.0)}"></div>
  </div>
  <div style="margin-top:.6rem">
    <label><input type=checkbox name=skip_videos{_c('skip_videos')}> Skip Stage 3 entirely</label><br>
    <label><input type=checkbox name=only_outcomes{_c('only_outcomes')}> Outcomes only (skip fight matches)</label>
  </div>
  <label style="margin-top:.75rem">Video stage mode</label>
  <select name=stage3_mode>
    <option value=full{"selected" if stage3_mode=="full" else ""}>Full (prompts + render)</option>
    <option value=only_prompts{"selected" if stage3_mode=="only_prompts" else ""}>Prompts only (no render)</option>
    <option value=only_videos{"selected" if stage3_mode=="only_videos" else ""}>Videos only (use saved prompts)</option>
  </select>
  <label style="margin-top:.75rem">Run scope</label>
  <select name=only_stage>
    <option value=all{"selected" if only_stage=="all" else ""}>All stages</option>
    <option value=characters{"selected" if only_stage=="characters" else ""}>Characters only</option>
    <option value=environments{"selected" if only_stage=="environments" else ""}>Environments only</option>
    <option value=assets{"selected" if only_stage=="assets" else ""}>Assets only (chars + envs, no video)</option>
  </select>
</div>

<div class=card>
  <h2>Character Consistency</h2>
  <p class=hint style="margin-bottom:.5rem">How fighters are kept looking the same across clips and matching their portraits. Stackable.</p>
  <label>Strategy</label>
  <select name=consistency id=consistency onchange="toggleConsFields()">
    <option value=prompt{"selected" if _v('consistency','keyframe')=="prompt" else ""}>prompt — descriptions only (fastest)</option>
    <option value=keyframe{"selected" if _v('consistency','keyframe')=="keyframe" else ""}>keyframe — image→video bridge (balanced, default)</option>
    <option value="keyframe,ipadapter"{"selected" if _v('consistency','keyframe')=="keyframe,ipadapter" else ""}>keyframe + IP-Adapter — portrait-guided keyframes</option>
    <option value="keyframe,lora"{"selected" if _v('consistency','keyframe')=="keyframe,lora" else ""}>keyframe + LoRA — trained identity (strongest, slowest)</option>
    <option value=lora{"selected" if _v('consistency','keyframe')=="lora" else ""}>lora only — trained identity, no keyframe</option>
  </select>
  <div id=keyframe_fields style="margin-top:.6rem">
    <div class=row>
      <div><label>Keyframe steps</label>
           <input name=keyframe_steps type=number min=8 max=80 value="{_v('keyframe_steps', 28)}"></div>
      <div><label>Keyframe size <span class=hint>(WxH)</span></label>
           <input name=keyframe_size type=text value="{_v('keyframe_size','512x512')}"></div>
      <div><label>Character strength <span class=hint>(IP-Adapter 0-1)</span></label>
           <input name=character_strength type=number min=0 max=1 step=0.05 value="{_v('character_strength', 0.7)}"></div>
    </div>
  </div>
  <div id=lora_fields style="margin-top:.6rem">
    <div><label>LoRA training base model <span class=hint>(SD1.x/SDXL — leave empty to train on the image model)</span></label>
         <input name=lora_train_base_model type=text style="width:100%"
                placeholder="e.g. stabilityai/stable-diffusion-xl-base-1.0"
                value="{_v('lora_train_base_model','')}"></div>
    <p class=hint style="margin:.2rem 0 .6rem">Z-Image / Flux / SD3 image models can't be LoRA-trained directly. Set a UNet-based SD1.x or SDXL model here to train identity LoRAs while still generating with the image model above.</p>
    <label style="margin-top:0">Character LoRAs <span class=hint>(per-fighter identity)</span></label>
    <div class=row3>
      <div><label>LoRA train steps</label>
           <input name=lora_steps type=number min=100 max=3000 step=50 value="{_v('lora_steps', 800)}"></div>
      <div><label>LoRA rank</label>
           <input name=lora_rank type=number min=2 max=128 value="{_v('lora_rank', 16)}"></div>
      <div><label>LoRA weight <span class=hint>(at generation)</span></label>
           <input name=lora_weight type=number min=0 max=2 step=0.05 value="{_v('lora_weight', 0.85)}"></div>
    </div>
    <div style="margin-top:.6rem">
      <label><input type=checkbox name=env_loras{"" if _v('no_env_loras') else " checked"}> Also train per-environment LoRAs <span class=hint>(lock each location’s look)</span></label>
    </div>
    <div class=row3 style="margin-top:.4rem">
      <div><label>Env LoRA train steps</label>
           <input name=env_lora_steps type=number min=100 max=3000 step=50 value="{_v('env_lora_steps', 800)}"></div>
      <div><label>Env LoRA rank</label>
           <input name=env_lora_rank type=number min=2 max=128 value="{_v('env_lora_rank', 16)}"></div>
      <div><label>Env LoRA weight <span class=hint>(at generation)</span></label>
           <input name=env_lora_weight type=number min=0 max=2 step=0.05 value="{_v('env_lora_weight', 0.8)}"></div>
    </div>
  </div>
</div>

<input type=hidden name=step id=step_field value="">
<div style="display:flex;gap:.75rem;align-items:center;margin-top:.25rem;flex-wrap:wrap">
  <button class="btn btn-primary" type=submit id=start-btn>▶ Full run</button>
  <button class="btn btn-danger" type=button id=stop-btn onclick="stopRun()" style="display:none">■ Stop</button>
  <button class="btn btn-secondary" type=button onclick="saveConfig()" title="Download the current options as a JSON config file you can reuse with --config / -c">💾 Save config</button>
  <span id=status-pill class="status-pill status-idle">Idle</span>
</div>
<div style="margin-top:.6rem">
  <span class=hint>Run individual steps (each picks up where the last left off):</span><br>
  <div style="display:flex;gap:.4rem;margin-top:.35rem;flex-wrap:wrap">
    <button class="btn btn-secondary" type=button onclick="runStep('characters')">1 · Characters</button>
    <button class="btn btn-secondary" type=button onclick="runStep('environments')">2 · Environments</button>
    <button class="btn btn-secondary" type=button onclick="runStep('prompts')">3 · Prompts</button>
    <button class="btn btn-secondary" type=button onclick="runStep('loras')">4 · Train LoRAs</button>
    <button class="btn btn-secondary" type=button onclick="runStep('keyframes')">5 · Keyframes</button>
    <button class="btn btn-secondary" type=button onclick="runStep('videos')">6 · Render videos</button>
  </div>
</div>
</form>

<div style="margin-top:1.2rem">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.3rem">
    <h2>Log</h2>
    <button class="btn btn-secondary" style="font-size:.75rem;padding:.2rem .6rem" onclick="clearLog()">Clear</button>
  </div>
  <div id=log-box></div>
</div>

<script>
function toggleCharFields(){{
  document.getElementById('char_names_row').style.display =
    document.getElementById('char_mode').value === 'fighters' ? 'block' : 'none';
}}
function toggleEnvFields(){{
  document.getElementById('env_names_row').style.display =
    document.getElementById('env_mode').value === 'environments' ? 'block' : 'none';
}}
function toggleConsFields(){{
  const c = document.getElementById('consistency').value;
  document.getElementById('keyframe_fields').style.display = c.includes('keyframe') ? 'block' : 'none';
  document.getElementById('lora_fields').style.display = c.includes('lora') ? 'block' : 'none';
}}
async function runStep(step){{
  const form = document.getElementById('run-form');
  document.getElementById('step_field').value = step;
  const fd = new FormData(form);
  document.getElementById('step_field').value = '';  // reset for next full run
  const r = await fetch('/start',{{method:'POST',body:fd}});
  const j = await r.json();
  if(j.error){{ appendLog('✗ '+j.error); return; }}
  setStatus(true,false);
  startSSE();
  setTimeout(refreshStatus, 500);
}}
async function saveConfig(){{
  // Save the current options to a file ON THE SERVER (where the script runs).
  // Relative paths are written inside the output directory. When launched with
  // -c/--config, the default below is that same config path (overwrite-in-place).
  const def = {_save_default_js};
  const path = await uiPrompt('Save configuration to file (relative paths go inside the output dir):', def,
    {{title:'Save configuration', okText:'Save'}});
  if(path === null) return;  // cancelled
  const fd = new FormData(document.getElementById('run-form'));
  fd.set('path', path.trim() || def);
  try {{
    const r = await fetch('/save-config',{{method:'POST',body:fd}});
    const j = await r.json();
    if(j.error){{ appendLog('✗ Save failed: '+j.error); return; }}
    appendLog('✓ Saved configuration to '+j.path+'  (reuse with --config '+j.path+')');
  }} catch(e) {{
    appendLog('✗ Save failed: '+e);
  }}
}}
function clearLog(){{ document.getElementById('log-box').innerHTML=''; }}

let _es = null;
function colorLine(t){{
  const low = t.toLowerCase();
  if(t.indexOf('▶')!==-1 || t.trim().startsWith('━'))
    return '<span class=head>'+escHtml(t)+'</span>';
  if(low.includes('✗')||low.includes('error')||low.includes('oom')||low.includes('fatal'))
    return '<span class=err>'+escHtml(t)+'</span>';
  if(low.includes('✓')||low.includes('loaded')||low.includes('saved')||low.includes('done'))
    return '<span class=info>'+escHtml(t)+'</span>';
  if(low.includes('⚠')||low.includes('warn')||low.includes('retry'))
    return '<span class=warn>'+escHtml(t)+'</span>';
  return escHtml(t);
}}
function escHtml(s){{
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}
function appendLog(t){{
  const box=document.getElementById('log-box');
  box.innerHTML += colorLine(t)+'\\n';
  box.scrollTop=box.scrollHeight;
}}
function setStatus(running, done, label){{
  const pill=document.getElementById('status-pill');
  const startBtn=document.getElementById('start-btn');
  const stopBtn=document.getElementById('stop-btn');
  const lbl = label ? (' — '+label) : '';
  if(running){{ pill.className='status-pill status-run'; pill.textContent='Running…'+lbl; }}
  else if(done){{ pill.className='status-pill status-done'; pill.textContent='Done'+lbl; }}
  else{{ pill.className='status-pill status-idle'; pill.textContent='Idle'; }}
  startBtn.style.display = running ? 'none' : '';
  stopBtn.style.display  = running ? '' : 'none';
  if(!running && _es){{ _es.close(); _es=null; }}
}}
function refreshStatus(){{
  fetch('/status').then(r=>r.json()).then(d=>setStatus(d.running,d.done,d.label)).catch(()=>{{}});
}}
function startSSE(){{
  if(_es){{ _es.close(); }}
  _es = new EventSource('/stream');
  _es.onmessage = e => appendLog(e.data);
  _es.onerror = () => {{
    setTimeout(()=>fetch('/status').then(r=>r.json()).then(d=>setStatus(d.running,d.done,d.label)),1000);
  }};
}}
function pollStatus(){{
  fetch('/status').then(r=>r.json()).then(d=>{{
    setStatus(d.running,d.done,d.label);
    if(d.running) setTimeout(pollStatus,3000);
  }}).catch(()=>setTimeout(pollStatus,5000));
}}
document.getElementById('run-form').onsubmit = async function(e){{
  e.preventDefault();
  const fd = new FormData(this);
  const r = await fetch('/start',{{method:'POST',body:fd}});
  const j = await r.json();
  if(j.error){{ appendLog('✗ '+j.error); return; }}
  setStatus(true,false);
  startSSE();
  setTimeout(refreshStatus, 500);
}};
async function stopRun(){{
  await fetch('/stop',{{method:'POST'}});
}}
// Restore state on page load
toggleConsFields();
fetch('/status').then(r=>r.json()).then(d=>{{
  setStatus(d.running,d.done,d.label);
  if(d.running) startSSE();
  d.log.forEach(l=>appendLog(l));
}});
</script>"""

    def _list_profiles(kind: str) -> list:
        """Return locally-saved profiles of a kind with their meta + image files."""
        base = out_dir / (kind + "s")
        out = []
        if not base.exists():
            return out
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            mp = d / "meta.json"
            if not mp.exists():
                continue
            try:
                meta = json.loads(mp.read_text())
            except Exception:
                meta = {}
            imgs = sorted(
                p.name for p in d.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            )
            out.append({"name": d.name, "meta": meta, "images": imgs})
        return out

    def _profiles_html(kind: str):
        import html as _html
        label = "Characters" if kind == "character" else "Environments"
        profiles = _list_profiles(kind)
        # Which profiles already have a trained LoRA (from the on-disk map).
        _lora_file = out_dir / ("loras.json" if kind == "character" else "env_loras.json")
        try:
            _lora_map = json.loads(_lora_file.read_text()) if _lora_file.exists() else {}
        except Exception:
            _lora_map = {}

        def esc(v):
            return _html.escape(str(v if v is not None else ""), quote=True)

        cards = []
        for p in profiles:
            name = p["name"]
            meta = p["meta"]
            thumbs = "".join(
                f'<div class=pf-thumb>'
                f'<img src="/media/{kind}s/{esc(name)}/{esc(img)}" loading=lazy alt="{esc(img)}" '
                f'title="Click to enlarge" onclick="showImg(this.src, this.alt)">'
                f'<button class=pf-thumb-del title="Delete this image" '
                f'onclick="delImg(\'{kind}\',\'{esc(name)}\',\'{esc(img)}\')">✕</button>'
                f'</div>'
                for img in p["images"]
            ) or '<span class=hint>No reference images.</span>'

            gender_html = ""
            if kind == "character":
                gender_html = (
                    f'<div><label>Gender</label>'
                    f'<input type=text data-field=gender value="{esc(meta.get("gender",""))}"></div>'
                )
            # Both kinds can guide regeneration with their kept references via
            # IP-Adapter (characters → character_profiles, envs → environment_profiles).
            guide_html = (
                '<label style="margin:0;font-size:.78rem;display:inline-flex;align-items:center;gap:.25rem">'
                '<input type=checkbox data-regen=guide checked style="width:auto"> match kept refs</label>'
            )

            # Any OTHER scalar fields present in meta.json become editable inputs
            # too, so every field of a profile can be edited (not just the fixed
            # set). Bookkeeping / already-rendered keys are excluded.
            _shown = {"name", "region", "gender", "description", "prompt",
                      "images", "image_count", "created_at", "created"}
            extra_rows = []
            for fk, fval in meta.items():
                if fk in _shown or isinstance(fval, (dict, list)):
                    continue
                extra_rows.append(
                    f'<div><label>{esc(fk)}</label>'
                    f'<input type=text data-field="{esc(fk)}" value="{esc(fval)}"></div>'
                )
            extra_html = (f'<div class=row3 style="margin-top:.4rem">{"".join(extra_rows)}</div>'
                          if extra_rows else "")

            cards.append(
                f'<div class=card id="pf-{kind}-{esc(name)}">'
                f'  <div class=pf-head>'
                f'    <span class=pf-name>{esc(name)}</span>'
                f'    <span class=hint>{len(p["images"])} image(s)</span>'
                f'  </div>'
                f'  <div class=pf-thumbs>{thumbs}</div>'
                f'  <div class=row>'
                f'    <div><label>Region</label>'
                f'<input type=text data-field=region value="{esc(meta.get("region",""))}"></div>'
                f'    {gender_html}'
                f'  </div>'
                f'  <label>Description <span class=hint>(synced to CoderAI)</span></label>'
                f'  <textarea data-field=description rows=2>{esc(meta.get("description",""))}</textarea>'
                f'  <label>Prompt</label>'
                f'  <textarea data-field=prompt rows=3>{esc(meta.get("prompt",""))}</textarea>'
                f'  {extra_html}'
                f'  <div class=pf-actions>'
                f'    <button class="btn btn-primary" style="font-size:.82rem;padding:.35rem .9rem" '
                f'onclick="saveProfile(\'{kind}\',\'{esc(name)}\')">💾 Save</button>'
                f'    <button class="btn btn-danger" style="font-size:.82rem;padding:.35rem .9rem" '
                f'onclick="delProfile(\'{kind}\',\'{esc(name)}\')">🗑 Remove</button>'
                f'    <span class=pf-status></span>'
                f'  </div>'
                f'  <div class=pf-actions style="border-top:1px solid #222;padding-top:.6rem;margin-top:.6rem">'
                f'    <label style="margin:0;font-size:.78rem">Add <input type=number data-regen=count '
                f'value=4 min=1 max=8 style="width:54px;display:inline-block"> new ref(s)</label>'
                f'    {guide_html}'
                f'    <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
                f'onclick="regenProfile(\'{kind}\',\'{esc(name)}\')">♻ Regenerate references</button>'
                f'    <span class=pf-regen-status style="font-size:.76rem;color:#7ea8f7"></span>'
                f'  </div>'
                f'  <div class=pf-actions style="padding-top:.5rem">'
                f'    <label style="margin:0;font-size:.78rem">Or upload your own:</label>'
                f'    <input type=file data-upload=files accept="image/*" multiple '
                f'style="font-size:.76rem;width:auto;flex:1;min-width:160px">'
                f'    <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
                f'onclick="uploadRefs(\'{kind}\',\'{esc(name)}\')">⬆ Upload references</button>'
                f'    <span class=pf-upload-status style="font-size:.76rem;color:#7ea8f7"></span>'
                f'  </div>'
                f'  <div class=pf-actions style="border-top:1px solid #222;padding-top:.6rem;margin-top:.6rem">'
                f'    <span style="font-size:.78rem;color:{"#7ed87e" if (_lora_map.get(name)) else "#888"}">'
                f'Identity LoRA: {"trained ✓" if (_lora_map.get(name)) else "not trained"}</span>'
                f'    <label style="margin:0;font-size:.78rem">steps <input type=number data-lora=steps '
                f'value=800 min=50 max=5000 step=50 style="width:66px;display:inline-block"></label>'
                f'    <label style="margin:0;font-size:.78rem">rank <input type=number data-lora=rank '
                f'value=16 min=2 max=128 style="width:54px;display:inline-block"></label>'
                f'    <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
                f'onclick="trainLora(\'{kind}\',\'{esc(name)}\')">🧠 {"Retrain" if (_lora_map.get(name)) else "Train"} LoRA</button>'
                f'    <span class=pf-lora-status style="font-size:.76rem;color:#7ea8f7"></span>'
                f'  </div>'
                f'</div>'
            )

        if cards:
            inner = "".join(cards)
        else:
            inner = (f'<div class=card style="color:#666">No {label.lower()} found in '
                     f'<code>{esc(str(out_dir))}</code> yet. Generate some from the Run page first.</div>')

        script = """
<script>
async function saveProfile(kind,name){
  const root=document.getElementById('pf-'+kind+'-'+name);
  const st=root.querySelector('.pf-status');
  st.style.color='#aaa'; st.textContent='Saving…';
  const fd=new FormData();
  fd.append('kind',kind); fd.append('name',name);
  root.querySelectorAll('[data-field]').forEach(el=>fd.append(el.getAttribute('data-field'),el.value));
  try{
    const r=await fetch('/profile/save',{method:'POST',body:fd});
    const j=await r.json();
    if(j.error){st.style.color='#e07070'; st.textContent='✗ '+j.error; return;}
    st.style.color='#7ed87e'; st.textContent='✓ Saved'+(j.synced?' (synced to CoderAI)':'');
  }catch(e){st.style.color='#e07070'; st.textContent='✗ '+e;}
}
async function delProfile(kind,name){
  if(!(await uiConfirm('Remove "'+name+'" and all its images? This deletes the local profile'
    +' and removes it from CoderAI. This cannot be undone.',
    {title:'Remove profile', okText:'Remove', danger:true})))return;
  const fd=new FormData(); fd.append('kind',kind); fd.append('name',name);
  const r=await fetch('/profile/delete',{method:'POST',body:fd});
  const j=await r.json();
  if(j.error){await uiAlert('Delete failed: '+j.error,{title:'Error'}); return;}
  const el=document.getElementById('pf-'+kind+'-'+name);
  if(el) el.remove();
}
async function delImg(kind,name,file){
  if(!(await uiConfirm('Delete image "'+file+'"?',
    {title:'Delete image', okText:'Delete', danger:true})))return;
  const fd=new FormData(); fd.append('kind',kind); fd.append('name',name); fd.append('file',file);
  const r=await fetch('/profile/delete-image',{method:'POST',body:fd});
  const j=await r.json();
  if(j.error){await uiAlert('Delete failed: '+j.error,{title:'Error'}); return;}
  location.reload();
}
async function regenProfile(kind,name){
  const root=document.getElementById('pf-'+kind+'-'+name);
  const st=root.querySelector('.pf-regen-status');
  const cnt=root.querySelector('[data-regen=count]');
  const guideEl=root.querySelector('[data-regen=guide]');
  const count=Math.max(1,Math.min(8,parseInt(cnt&&cnt.value||'4',10)||4));
  const fd=new FormData();
  fd.append('kind',kind); fd.append('name',name); fd.append('count',count);
  fd.append('guide', (guideEl && guideEl.checked) ? '1' : '0');
  st.style.color='#aaa'; st.textContent='Starting…';
  let j;
  try{
    const r=await fetch('/profile/regenerate',{method:'POST',body:fd});
    j=await r.json();
  }catch(e){ st.style.color='#e07070'; st.textContent='✗ '+e; return; }
  if(j.error){ st.style.color='#e07070'; st.textContent='✗ '+j.error; return; }
  pollRegen(j.job_id, st);
}
function pollRegen(jobId, st){
  st.style.color='#7ea8f7';
  const poll=async()=>{
    let d;
    try{ d=await (await fetch('/job/'+jobId)).json(); }
    catch(e){ setTimeout(poll,1500); return; }
    const pct=d.progress||0;
    if(d.status==='running'){ st.textContent='⏳ '+(d._msg||('working… '+pct+'%')); setTimeout(poll,1200); }
    else if(d.status==='done'){
      st.style.color='#7ed87e';
      st.textContent='✓ added '+(d.added||0)+' image(s)'+(d.synced===false?' (local only)':'')+' — reloading…';
      setTimeout(()=>location.reload(),900);
    } else {
      st.style.color='#e07070'; st.textContent='✗ '+(d.error||'failed');
    }
  };
  setTimeout(poll,800);
}
async function trainLora(kind,name){
  const root=document.getElementById('pf-'+kind+'-'+name);
  const st=root.querySelector('.pf-lora-status');
  const steps=parseInt(root.querySelector('[data-lora=steps]').value||'800',10);
  const rank=parseInt(root.querySelector('[data-lora=rank]').value||'16',10);
  if(!(await uiConfirm('Train identity LoRA for "'+name+'" ('+steps+' steps)? '
    +'This evicts loaded models and can take several minutes.',
    {title:'Train LoRA', okText:'Train'})))return;
  const fd=new FormData();
  fd.append('kind',kind); fd.append('name',name);
  fd.append('steps',steps); fd.append('rank',rank);
  st.style.color='#aaa'; st.textContent='Starting…';
  let j;
  try{ j=await (await fetch('/profile/train-lora',{method:'POST',body:fd})).json(); }
  catch(e){ st.style.color='#e07070'; st.textContent='✗ '+e; return; }
  if(j.error){ st.style.color='#e07070'; st.textContent='✗ '+j.error; return; }
  pollTrain(j.job_id, st);
}
function pollTrain(jobId, st){
  st.style.color='#7ea8f7';
  const poll=async()=>{
    let d;
    try{ d=await (await fetch('/job/'+jobId)).json(); }
    catch(e){ setTimeout(poll,2000); return; }
    const pct=d.progress||0;
    if(d.status==='running'){ st.textContent='⏳ '+(d._msg||('training… '+pct+'%'))+' ('+pct+'%)'; setTimeout(poll,1500); }
    else if(d.status==='done'){
      st.style.color='#7ed87e'; st.textContent='✓ LoRA trained — reloading…';
      setTimeout(()=>location.reload(),1000);
    } else { st.style.color='#e07070'; st.textContent='✗ '+(d.error||'failed'); }
  };
  setTimeout(poll,900);
}
// On page load, re-attach progress to any regen/train job still running so a
// reload doesn't lose the live progress display.
async function resumeActiveJobs(){
  let data;
  try{ data=await (await fetch('/active-jobs')).json(); }
  catch(e){ return; }
  (data.jobs||[]).forEach(j=>{
    const root=document.getElementById('pf-'+j.kind+'-'+j.name);
    if(!root) return;
    if(j.jtype==='regen'){
      const st=root.querySelector('.pf-regen-status');
      if(st){ st.textContent='⏳ '+(j._msg||'working…'); pollRegen(j.job_id, st); }
    } else if(j.jtype==='train'){
      const st=root.querySelector('.pf-lora-status');
      if(st){ st.textContent='⏳ '+(j._msg||'training…'); pollTrain(j.job_id, st); }
    }
  });
}
document.addEventListener('DOMContentLoaded', resumeActiveJobs);
async function uploadRefs(kind,name){
  const root=document.getElementById('pf-'+kind+'-'+name);
  const inp=root.querySelector('[data-upload=files]');
  const st=root.querySelector('.pf-upload-status');
  if(!inp||!inp.files||!inp.files.length){
    st.style.color='#e07070'; st.textContent='Choose image file(s) first'; return;
  }
  const fd=new FormData();
  fd.append('kind',kind); fd.append('name',name);
  for(const f of inp.files) fd.append('files',f);
  st.style.color='#aaa'; st.textContent='Uploading '+inp.files.length+' file(s)…';
  try{
    const r=await fetch('/profile/upload-image',{method:'POST',body:fd});
    const j=await r.json();
    if(j.error){ st.style.color='#e07070'; st.textContent='✗ '+j.error; return; }
    st.style.color='#7ed87e';
    st.textContent='✓ added '+j.added+(j.rejected?(' ('+j.rejected+' skipped)'):'')
      +(j.synced===false?' (local only)':'')+' — reloading…';
    setTimeout(()=>location.reload(),800);
  }catch(e){ st.style.color='#e07070'; st.textContent='✗ '+e; }
}
</script>"""
        return (f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<h1>{label}</h1>'
                f'<a href="/{kind}s" class="btn btn-secondary" style="font-size:.8rem">↻ Refresh</a></div>'
                f'<p class=hint style="margin-bottom:.8rem">Edit a profile’s fields and Save, or '
                f'Remove it entirely. Changes apply to the local output folder and are synced to CoderAI.</p>'
                f'{inner}{script}')

    def _scan_matches():
        """Return (vdir, plan, fight_by_name, matches, legacy_outcomes).

        Each matches[mn] may hold: finals{short,long}, clips[], outcomes[(fighter,
        outcome, path)]. Per-match outcomes are files '<match>_<fighter>_<outcome>';
        legacy per-fighter outcomes ('<fighter>_<outcome>') go to legacy_outcomes.
        """
        # Longest outcome first so 'ko_win' isn't matched as 'win'.
        _oc = sorted(_PROMPT_OUTCOMES, key=len, reverse=True)
        vdir = out_dir / "videos"
        plan = {}
        pf = vdir / "prompts.json"
        if pf.exists():
            try:
                plan = json.loads(pf.read_text())
            except Exception:
                plan = {}
        fight_by_name = {m.get("match_name"): m for m in plan.get("fight_plan", [])}
        matches, leftovers = {}, []
        if vdir.exists():
            for v in sorted(vdir.glob("*.mp4")):
                stem = v.stem
                if stem.endswith("_short") or stem.endswith("_long"):
                    mn, kind = stem.rsplit("_", 1)
                    matches.setdefault(mn, {}).setdefault("finals", {})[kind] = v
                elif "_clip" in stem:
                    mn = stem.split("_clip")[0]
                    matches.setdefault(mn, {}).setdefault("clips", []).append(v)
                else:
                    leftovers.append(v)
        for mn in fight_by_name:
            matches.setdefault(mn, {})

        # Resolve leftovers into per-match outcomes (longest matching match name
        # prefix) or legacy per-fighter outcomes.
        known = sorted(matches.keys(), key=len, reverse=True)
        legacy_outcomes = []
        for v in leftovers:
            stem = v.stem
            outcome = next((o for o in _oc if stem.endswith("_" + o)), None)
            if not outcome:
                continue
            core = stem[: -(len(outcome) + 1)]   # "<match>_<fighter>" or "<fighter>"
            mn = next((k for k in known if k and core.startswith(k + "_")), None)
            if mn:
                fighter = core[len(mn) + 1:]
                matches[mn].setdefault("outcomes", []).append((fighter, outcome, v))
            else:
                legacy_outcomes.append((core, outcome, v))
        return vdir, plan, fight_by_name, matches, legacy_outcomes

    def _dur_str(p: Path) -> str:
        d = get_video_duration(str(p)) or 0
        return f"{d:.0f}s" if d else "?"

    def _esc(v):
        import html as _html
        return _html.escape(str(v if v is not None else ""), quote=True)

    def _vid_tag(p: Path, h=180):
        url = "/media/" + str(p.relative_to(out_dir)).replace("\\", "/")
        return (f'<video src="{_esc(url)}" controls preload=none '
                f'style="width:100%;height:{h}px;object-fit:cover;'
                f'border-radius:6px;background:#111"></video>')

    # Shared JS for the Matches list + detail pages (regenerate / save / remove).
    _match_js = """
<script>
function _findStatus(ev){
  const card = ev && ev.target ? ev.target.closest('.card') : null;
  return card ? card.querySelector('.match-status') : document.getElementById('detail-status');
}
function _pollJob(jobId, setSt){
  const poll=async()=>{
    let d;
    try{ d=await (await fetch('/job/'+jobId)).json(); }
    catch(e){ setTimeout(poll,2000); return; }
    const pct=d.progress||0;
    if(d.status==='running'){ setSt('#7ea8f7','⏳ '+(d._msg||('working… '+pct+'%'))+' ('+pct+'%)'); setTimeout(poll,1500); }
    else if(d.status==='done'){ setSt('#7ed87e','✓ '+(d._msg||'done')+' — reloading…'); setTimeout(()=>location.reload(),1200); }
    else { setSt('#e07070','✗ '+(d.error||'failed')); }
  };
  setTimeout(poll,900);
}
function _esch(s){ return String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function _renderMatchBars(wrap, d){
  if(!wrap) return;
  const pct=d.progress||0;
  let h='<div class=prg-global><div class=prg-label>Overall — '+pct+'%</div>'
       +'<div class=progress-bar><div class=progress-fill style="width:'+pct+'%"></div></div></div>';
  const items=d.items||[];
  if(items.length){
    h+='<div class=prg-items>';
    for(const it of items){
      const s=it.status||'pending';
      let w='0%', cls='';
      if(s==='rendering'){ w='100%'; cls=' striped'; }
      else if(s==='done'){ w='100%'; cls=' ok'; }
      else if(s==='failed'){ w='100%'; cls=' fail'; }
      const icon=s==='done'?'✓':(s==='failed'?'✗':(s==='rendering'?'⏳':'·'));
      h+='<div class=prg-item><div class=prg-ilabel>'+icon+' '+_esch(it.label)+' — '+s+'</div>'
        +'<div class=progress-bar><div class="progress-fill'+cls+'" style="width:'+w+'"></div></div></div>';
    }
    h+='</div>';
  }
  wrap.innerHTML=h;
  wrap.classList.remove('hidden');
}
// Detail-page poller: drives the text status AND the visual progress bars.
function _pollMatchBars(jobId, setSt, wrap){
  const poll=async()=>{
    let d;
    try{ d=await (await fetch('/job/'+jobId)).json(); }
    catch(e){ setTimeout(poll,2000); return; }
    _renderMatchBars(wrap, d);
    const pct=d.progress||0;
    if(d.status==='running'){ setSt('#7ea8f7','⏳ '+(d._msg||'working…')+' ('+pct+'%)'); setTimeout(poll,1200); }
    else if(d.status==='done'){ setSt('#7ed87e','✓ '+(d._msg||'done')+' — reloading…'); setTimeout(()=>location.reload(),1600); }
    else { setSt('#e07070','✗ '+(d.error||'failed')); }
  };
  setTimeout(poll,500);
}
async function reMatch(ev, scope, params){
  if(ev) ev.preventDefault();
  const labels={'match-clips':'Re-render ALL clips of this match (uses the video model, can take a while)?',
                'clip':'Re-render this single clip?',
                'reassemble':'Reassemble the final short/long videos from the existing clips? (fast, no model)',
                'outcomes':'Re-render all output clips for this fighter (uses the video model)?',
                'outcome':'Re-render this output clip?'};
  if(!(await uiConfirm(labels[scope]||'Proceed?',
       {title:'Regenerate', okText:(scope==='reassemble'?'Reassemble':'Re-render'),
        danger:(scope!=='reassemble')})))return;
  const stEl=_findStatus(ev);
  const setSt=(c,t)=>{ if(stEl){ stEl.style.color=c; stEl.textContent=t; } };
  const fd=new FormData(); fd.append('scope',scope);
  for(const k in params) fd.append(k, params[k]);
  setSt('#aaa','Starting…');
  let j;
  try{ j=await (await fetch('/matches/render',{method:'POST',body:fd})).json(); }
  catch(e){ setSt('#e07070','✗ '+e); return; }
  if(j.error){ setSt('#e07070','✗ '+j.error); return; }
  const wrap=document.getElementById('match-progress');
  if(wrap && scope!=='reassemble'){ wrap.innerHTML=''; wrap.classList.remove('hidden'); _pollMatchBars(j.job_id, setSt, wrap); }
  else { _pollJob(j.job_id, setSt); }
}
async function delVid(ev, scope, params){
  if(ev) ev.preventDefault();
  const labels={'clip':'Delete this clip video file?',
                'final':'Delete this assembled video file?',
                'match':'Delete ALL video files for this match (clips + finals)? The plan/prompts are kept so you can re-render.',
                'output':'Delete this output video file?',
                'outputs':'Delete ALL output video files for this fighter?'};
  if(!(await uiConfirm(labels[scope]||'Delete?',{title:'Remove videos', okText:'Delete', danger:true})))return;
  const stEl=_findStatus(ev);
  const setSt=(c,t)=>{ if(stEl){ stEl.style.color=c; stEl.textContent=t; } };
  const fd=new FormData(); fd.append('scope',scope);
  for(const k in params) fd.append(k, params[k]);
  let j;
  try{ j=await (await fetch('/matches/delete',{method:'POST',body:fd})).json(); }
  catch(e){ setSt('#e07070','✗ '+e); return; }
  if(j.error){ setSt('#e07070','✗ '+j.error); return; }
  setSt('#7ed87e','✓ removed '+(j.removed||0)+' file(s) — reloading…');
  setTimeout(()=>location.reload(),700);
}
async function saveMatch(ev, name){
  const root=document.getElementById('detail');
  const st=document.getElementById('detail-status');
  const setSt=(c,t)=>{ st.style.color=c; st.textContent=t; };
  const fd=new FormData(); fd.append('mode','match'); fd.append('name',name);
  ['f1','f2','env','short_target','long_target'].forEach(k=>{
    const el=root.querySelector('[data-field="'+k+'"]'); if(el) fd.append(k, el.value);
  });
  root.querySelectorAll('[data-clip]').forEach(el=>fd.append('clip_'+el.getAttribute('data-clip'), el.value));
  root.querySelectorAll('[data-outc]').forEach(el=>fd.append('outc_'+el.getAttribute('data-outc'), el.value));
  setSt('#aaa','Saving…');
  try{
    const j=await (await fetch('/matches/save',{method:'POST',body:fd})).json();
    if(j.error){ setSt('#e07070','✗ '+j.error); return; }
    setSt('#7ed87e','✓ Saved');
  }catch(e){ setSt('#e07070','✗ '+e); }
}
async function saveOutputs(ev, fighter){
  const root=document.getElementById('detail');
  const st=document.getElementById('detail-status');
  const setSt=(c,t)=>{ st.style.color=c; st.textContent=t; };
  const fd=new FormData(); fd.append('mode','outputs'); fd.append('fighter',fighter);
  root.querySelectorAll('[data-out]').forEach(el=>fd.append('out_'+el.getAttribute('data-out'), el.value));
  setSt('#aaa','Saving…');
  try{
    const j=await (await fetch('/matches/save',{method:'POST',body:fd})).json();
    if(j.error){ setSt('#e07070','✗ '+j.error); return; }
    setSt('#7ed87e','✓ Saved');
  }catch(e){ setSt('#e07070','✗ '+e); }
}
// After a page reload, re-attach the progress display to an in-flight render
// for the match being viewed (detail page) or a visible match card.
async function resumeMatchJobs(){
  let data;
  try{ data=await (await fetch('/active-jobs')).json(); }
  catch(e){ return; }
  for(const j of (data.jobs||[])){
    if(j.jtype!=='match') continue;
    const det=document.getElementById('detail');
    if(det && det.getAttribute('data-match')===j.match){
      const stEl=document.getElementById('detail-status');
      if(!stEl) continue;
      const setSt=(c,t)=>{ stEl.style.color=c; stEl.textContent=t; };
      setSt('#7ea8f7','⏳ '+(j._msg||'rendering…')+' ('+(j.progress||0)+'%)');
      const wrap=document.getElementById('match-progress');
      if(wrap && j.scope!=='reassemble'){ _pollMatchBars(j.job_id, setSt, wrap); }
      else { _pollJob(j.job_id, setSt); }
    } else {
      const card=document.querySelector('.match-card[data-match="'+(j.match||'')+'"]');
      if(!card) continue;
      const stEl=card.querySelector('.match-status');
      if(!stEl) continue;
      const setSt=(c,t)=>{ stEl.style.color=c; stEl.textContent=t; };
      setSt('#7ea8f7','⏳ '+(j._msg||'rendering…')+' ('+(j.progress||0)+'%)');
      _pollJob(j.job_id, setSt);
    }
  }
}
document.addEventListener('DOMContentLoaded', resumeMatchJobs);
</script>"""

    def _match_preview(mn, info):
        """Lightweight preview thumbnail: clip00 keyframe image if present, else
        a metadata-only poster of the short/first video. No full video load."""
        kf = out_dir / "videos" / "keyframes" / f"{mn}_clip00.png"
        box = "width:128px;height:72px;object-fit:cover;border-radius:5px;background:#111;flex:none"
        if kf.exists():
            url = "/media/" + str(kf.relative_to(out_dir)).replace("\\", "/")
            return f'<img src="{_esc(url)}" loading=lazy style="{box}">'
        finals = info.get("finals", {})
        clips = info.get("clips", [])
        vp = finals.get("short") or finals.get("long") or (sorted(clips, key=lambda p: p.name)[0] if clips else None)
        if vp:
            url = "/media/" + str(vp.relative_to(out_dir)).replace("\\", "/")
            return f'<video src="{_esc(url)}" preload=metadata muted style="{box}"></video>'
        return (f'<div style="{box};display:flex;align-items:center;justify-content:center;'
                f'color:#555;font-size:.7rem">no preview</div>')

    def _matches_html():
        """Lightweight LIST of matches (with preview) — videos load on detail."""
        vdir, plan, fight_by_name, matches, legacy_outcomes = _scan_matches()

        def _row(mn):
            info = matches[mn]
            meta = fight_by_name.get(mn, {})
            clips = info.get("clips", [])
            finals = info.get("finals", {})
            n_out = len(info.get("outcomes", []))
            f1, f2 = meta.get("f1", ""), meta.get("f2", "")
            env = meta.get("env", "")
            title = f"{f1} vs {f2}" if f1 else mn.replace("match_", "").replace("_", " ")
            fin = ", ".join(k for k in ("short", "long") if k in finals) or "none"
            return (
                f'<div class="card match-card" id="row-{_esc(mn)}" data-match="{_esc(mn)}" '
                f'style="display:flex;align-items:center;gap:.8rem;padding:.7rem 1rem">'
                f'  {_match_preview(mn, info)}'
                f'  <div style="flex:1;min-width:0">'
                f'    <div class=pf-name style="font-size:.98rem">🥊 {_esc(title)}</div>'
                f'    <div class=hint>{_esc(env) or "no env"} · {len(clips)} clip(s) · '
                f'{n_out} outcome(s) · finals: {_esc(fin)}</div>'
                f'    <div class=hint style="opacity:.6">{_esc(mn)}</div>'
                f'  </div>'
                f'  <span class=match-status style="font-size:.74rem;color:#7ea8f7"></span>'
                f'  <a class="btn btn-primary" style="font-size:.8rem;padding:.35rem .9rem" '
                f'href="/match?name={_esc(mn)}">Open ▸</a>'
                f'  <button class="btn btn-danger" style="font-size:.8rem;padding:.35rem .8rem" '
                f'onclick="delVid(event,\'match\',{{match:\'{_esc(mn)}\'}})">🗑 Remove videos</button>'
                f'</div>'
            )

        match_rows = "".join(_row(mn) for mn in sorted(matches))

        # Legacy per-fighter outcome files (from older runs) — simple list.
        leg_rows = ""
        if legacy_outcomes:
            cells = "".join(
                f'<div class=card style="display:flex;align-items:center;gap:.6rem;padding:.5rem .8rem">'
                f'  <div style="flex:1"><span class=pf-name style="font-size:.9rem">{_esc(core)}</span>'
                f'  <span class=hint> · {_esc(outcome)}</span></div>'
                f'  <a class="btn btn-secondary" style="font-size:.78rem;padding:.25rem .7rem" '
                f'href="/media/videos/{_esc(p.name)}" target=_blank>▶ View</a>'
                f'  <button class="btn btn-danger" style="font-size:.78rem;padding:.25rem .7rem" '
                f'onclick="delVid(event,\'file\',{{file:\'{_esc(p.name)}\'}})">🗑</button>'
                f'</div>'
                for core, outcome, p in sorted(legacy_outcomes, key=lambda t: t[2].name)
            )
            leg_rows = ('<div class=section-title style="margin:1.1rem 0 .4rem">'
                        'Legacy per-fighter outputs</div>' + cells)

        body = ""
        if match_rows:
            body += '<div class=section-title style="margin:.3rem 0 .4rem">Matches</div>' + match_rows
        body += leg_rows
        if not body:
            body = ('<div class=card style="color:#666">No matches found yet. Render '
                    'videos from the Run page first (or run the Videos step).</div>')

        return (f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<h1>Matches</h1>'
                f'<a href="/matches" class="btn btn-secondary" style="font-size:.8rem">↻ Refresh</a></div>'
                f'<p class=hint style="margin-bottom:.8rem">Select a match to view, edit and '
                f'regenerate its clips, finals and outcomes. Videos load on the detail page.</p>'
                f'{body}{_match_js}')

    def _match_detail_html(name=None, fighter=None):
        """Detail view for a single match: embeds that match's videos only, with
        edit + regenerate + remove for its clips, finals and outcomes."""
        vdir, plan, fight_by_name, matches, legacy_outcomes = _scan_matches()
        back = ('<a href="/matches" style="font-size:.85rem">‹ Back to matches</a>')

        # ── Match detail ───────────────────────────────────────────────────────
        if not name or name not in matches:
            return f'<div class=card style="color:#666">Match not found. {back}</div>'
        meta = fight_by_name.get(name, {})
        info = matches[name]
        finals = info.get("finals", {})
        clip_files = {}
        for p in info.get("clips", []):
            suf = p.stem.split("_clip")[-1]
            if suf.isdigit():
                clip_files[int(suf)] = p
        plan_clips = meta.get("clips", [])
        f1, f2 = meta.get("f1", ""), meta.get("f2", "")
        env = meta.get("env", "")
        title = f"{f1} vs {f2}" if f1 else name.replace("match_", "").replace("_", " ")

        finals_html = ""
        for k in ("short", "long"):
            if k in finals:
                finals_html += (
                    f'<div style="flex:1;min-width:220px">'
                    f'<div class=hint style="margin-bottom:.2rem">{k} ({_dur_str(finals[k])})</div>'
                    f'{_vid_tag(finals[k])}'
                    f'<div style="margin-top:.3rem"><button class="btn btn-danger" '
                    f'style="font-size:.76rem;padding:.25rem .7rem" '
                    f'onclick="delVid(event,\'final\',{{match:\'{_esc(name)}\',which:\'{k}\'}})">🗑 Remove {k}</button></div>'
                    f'</div>')
        if not finals_html:
            finals_html = '<span class=hint>No assembled videos yet.</span>'

        # Clip tiles: prefer the saved plan order; show video + editable prompt.
        clip_tiles = []
        idxs = [c["idx"] for c in plan_clips] if plan_clips else sorted(clip_files)
        for c in (plan_clips or [{"idx": i, "prompt": ""} for i in idxs]):
            idx = c["idx"]
            vp = clip_files.get(idx)
            vid_html = _vid_tag(vp, 120) if vp else '<div class=hint>not rendered</div>'
            rm_html = (f'<button class="btn btn-danger" style="font-size:.72rem;padding:.2rem .55rem" '
                       f'onclick="delVid(event,\'clip\',{{match:\'{_esc(name)}\',idx:\'{idx}\'}})">🗑</button>'
                       if vp else '')
            clip_tiles.append(
                f'<div class=card style="width:230px">'
                f'  <div class=hint style="display:flex;justify-content:space-between;align-items:center">'
                f'<span>clip {idx:02d}</span>'
                f'<span><a href="#" style="color:#7eb8f7" '
                f'onclick="reMatch(event,\'clip\',{{match:\'{_esc(name)}\',idx:\'{idx}\'}})">re-render</a> {rm_html}</span></div>'
                f'  {vid_html}'
                f'  <textarea data-clip="{idx}" rows=2 style="margin-top:.3rem">{_esc(c.get("prompt",""))}</textarea>'
                f'</div>'
            )
        clip_tiles_html = "".join(clip_tiles) or '<span class=hint>No clips planned.</span>'

        # ── Outcomes for this match (per participating fighter) ────────────────
        rendered_out = {(fr, oc): p for (fr, oc, p) in info.get("outcomes", [])}
        plan_out = {(o["fighter"], o["outcome"]): o for o in plan.get("outcome_plan", [])
                    if o.get("match_name") == name}
        out_fighters = [x for x in (f1, f2) if x]
        # Include any fighters that appear in rendered/planned outcomes but not in meta.
        for (fr, _oc) in list(rendered_out) + list(plan_out):
            if fr not in out_fighters:
                out_fighters.append(fr)
        outcome_groups = []
        for fr in out_fighters:
            tiles = []
            for oc in _PROMPT_OUTCOMES:
                p = rendered_out.get((fr, oc))
                o = plan_out.get((fr, oc), {})
                vid = _vid_tag(p, 110) if p else '<div class=hint>not rendered</div>'
                rm = (f'<button class="btn btn-danger" style="font-size:.72rem;padding:.2rem .55rem" '
                      f'onclick="delVid(event,\'output\',{{match:\'{_esc(name)}\',fighter:\'{_esc(fr)}\',outcome:\'{_esc(oc)}\'}})">🗑</button>'
                      if p else '')
                act = "re-render" if p else "render"
                tiles.append(
                    f'<div class=card style="width:215px">'
                    f'  <div class=hint style="display:flex;justify-content:space-between;align-items:center">'
                    f'<span>{_esc(oc)}</span>'
                    f'<span><a href="#" style="color:#7eb8f7" '
                    f'onclick="reMatch(event,\'outcome\',{{match:\'{_esc(name)}\',fighter:\'{_esc(fr)}\',outcome:\'{_esc(oc)}\'}})">{act}</a> {rm}</span></div>'
                    f'  {vid}'
                    f'  <textarea data-outc="{_esc(fr)}|{_esc(oc)}" rows=2 style="margin-top:.3rem">{_esc(o.get("prompt",""))}</textarea>'
                    f'</div>'
                )
            outcome_groups.append(
                f'<div style="margin-top:.5rem"><div class=hint style="font-weight:700;color:#bbb">'
                f'{_esc(fr)}</div>'
                f'<div style="display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.25rem">{"".join(tiles)}</div></div>'
            )
        outcomes_html = "".join(outcome_groups) or '<span class=hint>No outcomes planned for this match.</span>'

        return (
            f'<div id=detail data-match="{_esc(name or "")}">{back}'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin:.4rem 0">'
            f'<h1>🥊 {_esc(title)}</h1>'
            f'<span id=detail-status style="font-size:.8rem;color:#7ea8f7"></span></div>'
            # edit meta
            f'<div class=card>'
            f'  <div class=row3>'
            f'    <div><label>Fighter 1</label><input type=text data-field=f1 value="{_esc(f1)}"></div>'
            f'    <div><label>Fighter 2</label><input type=text data-field=f2 value="{_esc(f2)}"></div>'
            f'    <div><label>Environment</label><input type=text data-field=env value="{_esc(env)}"></div>'
            f'  </div>'
            f'  <div class=row style="margin-top:.4rem">'
            f'    <div><label>Short target (s)</label><input type=number data-field=short_target '
            f'value="{_esc(meta.get("short_target",45))}"></div>'
            f'    <div><label>Long target (s)</label><input type=number data-field=long_target '
            f'value="{_esc(meta.get("long_target",70))}"></div>'
            f'  </div>'
            f'  <div class=pf-actions style="margin-top:.6rem">'
            f'    <button class="btn btn-primary" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="saveMatch(event,\'{_esc(name)}\')">💾 Save match</button>'
            f'    <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="reMatch(event,\'match-clips\',{{match:\'{_esc(name)}\'}})">♻ Re-render all clips</button>'
            f'    <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="reMatch(event,\'reassemble\',{{match:\'{_esc(name)}\'}})">🎞 Reassemble finals</button>'
            f'    <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="reMatch(event,\'outcomes\',{{match:\'{_esc(name)}\'}})">♻ Re-render all outcomes</button>'
            f'    <button class="btn btn-danger" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="delVid(event,\'match\',{{match:\'{_esc(name)}\'}})">🗑 Remove all videos</button>'
            f'  </div>'
            f'</div>'
            f'<div id=match-progress class=hidden></div>'
            f'<div class=section-title style="margin:.7rem 0 .3rem">Final videos</div>'
            f'<div style="display:flex;gap:.6rem;flex-wrap:wrap">{finals_html}</div>'
            f'<div class=section-title style="margin:.9rem 0 .3rem">Single clips '
            f'<span class=hint>(edit prompt, then Save match + Re-render)</span></div>'
            f'<div style="display:flex;gap:.6rem;flex-wrap:wrap">{clip_tiles_html}</div>'
            f'<div class=section-title style="margin:.9rem 0 .3rem">Outcomes '
            f'<span class=hint>(per fighter — edit prompt, Save match, then Re-render)</span></div>'
            f'{outcomes_html}'
            f'</div>{_match_js}'
        )

    def _prompts_html():
        """Edit the global prompt templates used by the script (LLM system
        prompts + static fallback shot/outcome templates)."""
        cfg = prompts_config_snapshot()  # live values (defaults + any overrides)

        def ta(field, value, rows=3):
            return (f'<textarea data-pf="{_esc(field)}" rows={rows}>'
                    f'{_esc(value)}</textarea>')

        wst = cfg["win_shot_templates"]
        outcome_blocks = "".join(
            f'<label>{_esc(k)} outcome templates <span class=hint>(one per line)</span></label>'
            f'{ta("win::"+k, chr(10).join(wst.get(k, [])), 4)}'
            for k in _PROMPT_OUTCOMES
        )

        body = (
            f'<div id=pf-root>'
            f'<div class=card>'
            f'  <h2>Master prompts (LLM)</h2>'
            f'  <p class=hint style="margin-bottom:.4rem">These are the <b>master prompts</b> the text '
            f'model follows to write a <b>unique</b> prompt for every clip and every outcome of every '
            f'match (so no two matches/outcomes are identical). A text model must be enabled.</p>'
            f'  <label>Fight-shot master prompt</label>{ta("llm_system", cfg["llm_system"], 5)}'
            f'  <label>Outcome master prompt</label>{ta("llm_outcome_system", cfg["llm_outcome_system"], 4)}'
            f'</div>'
            f'<div class=card>'
            f'  <h2>Static fallback templates</h2>'
            f'  <p class=hint style="margin-bottom:.4rem">Only used when <b>no text model</b> is '
            f'available — these are identical across matches, so enable a text model for per-match '
            f'variety. One template per line.</p>'
            f'  <label>Fight-shot templates <span class=hint>(one per line)</span></label>'
            f'  {ta("fight_shot_templates", chr(10).join(cfg["fight_shot_templates"]), 8)}'
            f'  {outcome_blocks}'
            f'</div>'
            f'<div class=pf-actions>'
            f'  <button class="btn btn-primary" style="font-size:.85rem;padding:.4rem 1rem" '
            f'onclick="savePrompts(event)">💾 Save prompts</button>'
            f'  <button class="btn btn-secondary" style="font-size:.85rem;padding:.4rem 1rem" '
            f'onclick="resetPrompts(event)">↺ Reset to defaults</button>'
            f'  <span id=pf-status style="font-size:.8rem;color:#7ea8f7"></span>'
            f'</div>'
            f'</div>'
        )

        script = """
<script>
function _gatherPrompts(){
  const root=document.getElementById('pf-root');
  const cfg={llm_system:'',llm_outcome_system:'',fight_shot_templates:[],win_shot_templates:{}};
  root.querySelectorAll('[data-pf]').forEach(el=>{
    const f=el.getAttribute('data-pf'); const v=el.value;
    if(f==='llm_system'||f==='llm_outcome_system'){ cfg[f]=v; }
    else if(f==='fight_shot_templates'){ cfg.fight_shot_templates=v.split('\\n').map(s=>s.trim()).filter(Boolean); }
    else if(f.startsWith('win::')){ cfg.win_shot_templates[f.slice(5)]=v.split('\\n').map(s=>s.trim()).filter(Boolean); }
  });
  return cfg;
}
async function savePrompts(ev){
  const st=document.getElementById('pf-status');
  const set=(c,t)=>{st.style.color=c;st.textContent=t;};
  set('#aaa','Saving…');
  const fd=new FormData(); fd.append('config', JSON.stringify(_gatherPrompts()));
  try{
    const j=await (await fetch('/prompts/save',{method:'POST',body:fd})).json();
    if(j.error){ set('#e07070','✗ '+j.error); return; }
    set('#7ed87e','✓ Saved — applied to future runs');
  }catch(e){ set('#e07070','✗ '+e); }
}
async function resetPrompts(ev){
  if(!(await uiConfirm('Reset all global prompts to the built-in defaults?',
      {title:'Reset prompts', okText:'Reset', danger:true})))return;
  const st=document.getElementById('pf-status');
  const fd=new FormData(); fd.append('reset','1');
  try{
    const j=await (await fetch('/prompts/save',{method:'POST',body:fd})).json();
    if(j.error){ st.style.color='#e07070'; st.textContent='✗ '+j.error; return; }
    location.reload();
  }catch(e){ st.style.color='#e07070'; st.textContent='✗ '+e; }
}
</script>"""

        return (f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<h1>Global prompts</h1></div>'
                f'<p class=hint style="margin-bottom:.8rem">These templates drive how clip and '
                f'outcome prompts are written for every match. Per-match prompts are edited on the '
                f'Matches page. Changes apply to future runs/regenerations.</p>'
                f'{body}{script}')

    def _gallery_html(out_path: Path):
        sections = []

        def _video_card(f):
            rel  = f.relative_to(out_path)
            url  = "/media/" + str(rel).replace("\\", "/")
            name = f.name
            rel_str = str(rel).replace("\\", "/")
            probe = _probe_video(f)
            info = ""
            if probe:
                w,h,fps = probe.get("width",0),probe.get("height",0),probe.get("fps",0)
                dur = probe.get("duration",0)
                info = f'{w}×{h}  {fps}fps  {dur:.1f}s'
            escaped = rel_str.replace("'", "\\'")
            return (
                f'<div class=media-card>'
                f'<video src="{url}" controls preload=metadata></video>'
                f'<div class=mc-label title="{name}">{name}</div>'
                f'{"<div class=mc-info>"+info+"</div>" if info else ""}'
                f'<div class=mc-actions>'
                f'  <button onclick="openProc(\'{escaped}\',\'upscale\',2)">⬆ 2×</button>'
                f'  <button onclick="openProc(\'{escaped}\',\'upscale\',4)">⬆ 4×</button>'
                f'  <button onclick="openProc(\'{escaped}\',\'fps\',null)">🎞 FPS</button>'
                f'  <button onclick="openProc(\'{escaped}\',\'upscale_fps\',null)">⬆+🎞 Both</button>'
                f'</div>'
                f'</div>'
            )

        def _image_card(f):
            rel = f.relative_to(out_path)
            url = "/media/" + str(rel).replace("\\","/")
            name = f.name
            return (f'<div class=media-card>'
                    f'<img src="{url}" loading=lazy alt="{name}">'
                    f'<div class=mc-label title="{name}">{name}</div>'
                    f'</div>')

        def _vid_cards(files):
            return "".join(_video_card(f) for f in sorted(files))

        def _img_cards(files):
            return "".join(_image_card(f) for f in sorted(files))

        # Characters
        char_dir = out_path / "characters"
        if char_dir.exists():
            imgs = list(char_dir.rglob("*.png")) + list(char_dir.rglob("*.jpg")) + list(char_dir.rglob("*.webp"))
            if imgs:
                sections.append(f'<div class=section-title>Characters ({len(imgs)} images)</div>'
                                 f'<div class=gallery-grid>{_img_cards(imgs)}</div>')

        # Environments
        env_dir = out_path / "environments"
        if env_dir.exists():
            imgs = list(env_dir.rglob("*.png")) + list(env_dir.rglob("*.jpg")) + list(env_dir.rglob("*.webp"))
            if imgs:
                sections.append(f'<div class=section-title>Environments ({len(imgs)} images)</div>'
                                 f'<div class=gallery-grid>{_img_cards(imgs)}</div>')

        # Videos — concatenated first, then individual clips, then outcomes
        vid_dir = out_path / "videos"
        if vid_dir.exists():
            all_vids = list(vid_dir.glob("*.mp4"))
            concat   = [v for v in all_vids if v.stem.endswith(("_short","_long"))]
            clips    = [v for v in all_vids if "_clip" in v.stem]
            outcomes = [v for v in all_vids
                        if not v.stem.endswith(("_short","_long")) and "_clip" not in v.stem]
            if concat:
                sections.append(f'<div class=section-title>Assembled matches ({len(concat)})</div>'
                                 f'<div class=gallery-grid>{_vid_cards(concat)}</div>')
            if outcomes:
                sections.append(f'<div class=section-title>Outcome clips ({len(outcomes)})</div>'
                                 f'<div class=gallery-grid>{_vid_cards(outcomes)}</div>')
            if clips:
                sections.append(f'<div class=section-title>Fight clips ({len(clips)})</div>'
                                 f'<div class=gallery-grid>{_vid_cards(clips)}</div>')

        if not sections:
            body_inner = '<div class=card style="color:#666">No media found yet. Run the generator to produce content.</div>'
        else:
            body_inner = "".join(sections)

        modal = """
<div class=modal-bg id=proc-modal onclick="if(event.target===this)closeProc()">
  <div class=modal>
    <h3 id=modal-title>Process video</h3>
    <div id=modal-body></div>
    <div style="display:flex;gap:.5rem;margin-top:.9rem">
      <button class="btn btn-primary" id=modal-go onclick="submitProc()">Process</button>
      <button class="btn btn-secondary" onclick="closeProc()">Cancel</button>
    </div>
    <div class=progress-bar id=modal-pbar style="display:none"><div class=progress-fill id=modal-pfill style="width:0%"></div></div>
    <div class=job-status id=modal-status></div>
  </div>
</div>
<script>
let _curFile=null,_curOp=null,_curJobId=null,_pollTimer=null;
const FPS_OPTIONS=[12,16,24,30,60];
function openProc(file,op,param){
  _curFile=file; _curOp=op;
  const title=document.getElementById('modal-title');
  const body=document.getElementById('modal-body');
  document.getElementById('modal-status').textContent='';
  document.getElementById('modal-status').className='job-status';
  document.getElementById('modal-pbar').style.display='none';
  document.getElementById('modal-pfill').style.width='0%';
  document.getElementById('modal-go').disabled=false;
  if(op==='upscale'){
    title.textContent=`Upscale ${param}×`;
    body.innerHTML=`<p style="font-size:.82rem;color:#aaa">Scale video to ${param}× resolution using CoderAI's video super-resolution endpoint. Output saved as a new file alongside the original.</p>`;
    _curOp='upscale'; _curJobId=null;
    document.getElementById('modal-go').onclick=()=>submitProc(param);
  } else if(op==='fps'){
    title.textContent='Raise FPS';
    body.innerHTML=`<label>Target FPS</label><select id=fps-sel>${FPS_OPTIONS.map(f=>'<option value='+f+(f===24?' selected':'')+'>'+f+' fps</option>').join('')}</select><p style="font-size:.75rem;color:#666;margin-top:.4rem">Uses CoderAI's frame interpolation endpoint (RIFE when available, ffmpeg minterpolate fallback). Longer videos take a few minutes.</p>`;
    document.getElementById('modal-go').onclick=()=>submitProc(parseInt(document.getElementById('fps-sel').value));
  } else if(op==='upscale_fps'){
    title.textContent='Upscale + Raise FPS';
    body.innerHTML=`<div class=row2>
      <div><label>Upscale</label><select id=us-sel><option value=2>2×</option><option value=4>4×</option></select></div>
      <div><label>Target FPS</label><select id=fps-sel2>${FPS_OPTIONS.map(f=>'<option value='+f+(f===24?' selected':'')+'>'+f+' fps</option>').join('')}</select></div>
    </div><p style="font-size:.75rem;color:#666;margin-top:.4rem">Upscales via CoderAI, then interpolates FPS via CoderAI in two sequential requests. This takes longer than either step alone.</p>`;
    document.getElementById('modal-go').onclick=()=>submitProc([
      parseInt(document.getElementById('us-sel').value),
      parseInt(document.getElementById('fps-sel2').value)
    ]);
  }
  document.getElementById('proc-modal').classList.add('open');
}
function closeProc(){
  document.getElementById('proc-modal').classList.remove('open');
  if(_pollTimer){clearInterval(_pollTimer);_pollTimer=null;}
}
async function submitProc(param){
  document.getElementById('modal-go').disabled=true;
  document.getElementById('modal-status').textContent='Starting…';
  document.getElementById('modal-pbar').style.display='block';
  const fd=new FormData();
  fd.append('file',_curFile);
  fd.append('op',_curOp);
  fd.append('param',JSON.stringify(param));
  const r=await fetch('/process',{method:'POST',body:fd});
  const j=await r.json();
  if(j.error){
    document.getElementById('modal-status').textContent='✗ '+j.error;
    document.getElementById('modal-status').className='job-status error';
    document.getElementById('modal-go').disabled=false;
    return;
  }
  _curJobId=j.job_id;
  _pollTimer=setInterval(pollJob,1200);
}
async function pollJob(){
  if(!_curJobId)return;
  const r=await fetch('/job/'+_curJobId);
  const j=await r.json();
  const pct=j.progress||0;
  document.getElementById('modal-pfill').style.width=pct+'%';
  document.getElementById('modal-status').textContent=
    j.status==='running'?(j._msg||`Sending to CoderAI… ${pct}%`):
    j.status==='done'?`✓ Done → ${j.output_name}`:
    `✗ ${j.error||'failed'}`;
  document.getElementById('modal-status').className='job-status'+(j.status==='done'?' done':j.status==='error'?' error':'');
  if(j.status!=='running'){
    clearInterval(_pollTimer);_pollTimer=null;
    document.getElementById('modal-go').disabled=false;
    if(j.status==='done'){
      // add a download/view link
      const st=document.getElementById('modal-status');
      st.innerHTML+=` <a href="/media/${j.output}" target=_blank style="color:#7eb8f7">▶ View</a>`;
    }
  }
}
</script>"""
        return (f'<h1>Gallery</h1>'
                f'<div style="margin-bottom:.5rem;display:flex;justify-content:flex-end">'
                f'<a href=/gallery class="btn btn-secondary" style="font-size:.8rem">↻ Refresh</a></div>'
                f'{body_inner}{modal}')

    # ── HTTP handler ────────────────────────────────────────────────────────

    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args): pass  # suppress access log

        def handle_one_request(self):
            # Browsers routinely abort media (video seek/scrub) and SSE
            # connections mid-response; that surfaces as BrokenPipe/ConnReset.
            # Swallow those so they don't spam the terminal with tracebacks.
            try:
                super().handle_one_request()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                self.close_connection = True

        def finish(self):
            try:
                super().finish()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def _send(self, code, ctype, body):
            if isinstance(body, str): body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"

            if path in ("/", ""):
                html = _page("Run", _run_page_html(default_args), "run")
                self._send(200, "text/html; charset=utf-8", html)

            elif path == "/characters":
                html = _page("Characters", _profiles_html("character"), "characters")
                self._send(200, "text/html; charset=utf-8", html)

            elif path == "/environments":
                html = _page("Environments", _profiles_html("environment"), "environments")
                self._send(200, "text/html; charset=utf-8", html)

            elif path == "/matches":
                html = _page("Matches", _matches_html(), "matches")
                self._send(200, "text/html; charset=utf-8", html)

            elif path == "/match":
                qs = urllib.parse.parse_qs(parsed.query)
                nm = qs.get("name", [None])[0]
                fr = qs.get("fighter", [None])[0]
                html = _page("Match", _match_detail_html(nm, fr), "matches")
                self._send(200, "text/html; charset=utf-8", html)

            elif path == "/prompts":
                html = _page("Prompts", _prompts_html(), "prompts")
                self._send(200, "text/html; charset=utf-8", html)

            elif path == "/gallery":
                html = _page("Gallery", _gallery_html(out_dir), "gallery")
                self._send(200, "text/html; charset=utf-8", html)

            elif path == "/status":
                import json as _j
                payload = _j.dumps({
                    "running": _state["running"],
                    "done": _state["done"],
                    "label": _state.get("current", ""),
                    "log": _state["log_lines"][-200:],
                })
                self._send(200, "application/json", payload)

            elif path == "/stream":
                # Server-Sent Events
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                q: "_queue_mod.Queue" = _queue_mod.Queue()
                # Replay existing log
                for line in list(_state["log_lines"]):
                    try:
                        self.wfile.write(f"data: {line}\n\n".encode())
                    except Exception:
                        return
                with _sse_lock:
                    _sse_clients.append(q)
                try:
                    while True:
                        try:
                            msg = q.get(timeout=15)
                            if msg is None:
                                break
                            self.wfile.write(f"data: {msg}\n\n".encode())
                            self.wfile.flush()
                        except _queue_mod.Empty:
                            # Heartbeat
                            try:
                                self.wfile.write(b": keep-alive\n\n")
                                self.wfile.flush()
                            except Exception:
                                break
                except Exception:
                    pass
                finally:
                    with _sse_lock:
                        try: _sse_clients.remove(q)
                        except ValueError: pass

            elif path.startswith("/job/"):
                import json as _j
                job_id = path[5:]
                with _jobs_lock:
                    job = dict(_state["jobs"].get(job_id, {"status": "not_found"}))
                self._send(200, "application/json", _j.dumps(job))

            elif path == "/active-jobs":
                # Running regen/train jobs, so a reloaded Characters/Environments
                # page can re-attach its progress display to in-flight work.
                import json as _j
                active = []
                with _jobs_lock:
                    for jid, j in _state["jobs"].items():
                        if j.get("status") == "running" and j.get("jtype") in ("regen", "train", "match"):
                            active.append({
                                "job_id": jid,
                                "kind": j.get("kind"),
                                "name": j.get("name"),
                                "jtype": j.get("jtype"),
                                "scope": j.get("scope"),
                                "match": j.get("match"),
                                "progress": j.get("progress", 0),
                                "_msg": j.get("_msg", ""),
                            })
                self._send(200, "application/json", _j.dumps({"jobs": active}))

            elif path.startswith("/media/"):
                rel = path[len("/media/"):]
                fpath = out_dir / rel
                if not fpath.exists() or not fpath.is_file():
                    self._send(404, "text/plain", "Not found")
                    return
                # Serve with range support for video
                ctype = mimetypes.guess_type(str(fpath))[0] or "application/octet-stream"
                size = fpath.stat().st_size
                rng = self.headers.get("Range")
                if rng and rng.startswith("bytes="):
                    try:
                        parts = rng[6:].split("-")
                        start = int(parts[0]) if parts[0] else 0
                        end = int(parts[1]) if len(parts) > 1 and parts[1] else size - 1
                        end = min(end, size - 1)
                        length = end - start + 1
                        self.send_response(206)
                        self.send_header("Content-Type", ctype)
                        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                        self.send_header("Content-Length", str(length))
                        self.send_header("Accept-Ranges", "bytes")
                        self.end_headers()
                        with open(fpath, "rb") as fh:
                            fh.seek(start)
                            remaining = length
                            while remaining:
                                chunk = fh.read(min(65536, remaining))
                                if not chunk: break
                                self.wfile.write(chunk)
                                remaining -= len(chunk)
                        return
                    except Exception:
                        pass
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with open(fpath, "rb") as fh:
                    while True:
                        chunk = fh.read(65536)
                        if not chunk: break
                        self.wfile.write(chunk)
            else:
                self._send(404, "text/plain", "Not found")

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path == "/stop":
                _state["abort"].set()
                import json as _j
                self._send(200, "application/json", _j.dumps({"ok": True}))
                return

            if path == "/profile/regenerate":
                import json as _j, uuid as _u
                clen = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(clen)
                ctype = self.headers.get("Content-Type", "")
                if "multipart/form-data" in ctype:
                    boundary = ctype.split("boundary=")[-1].strip().encode()
                    form = _parse_multipart(raw, boundary)
                else:
                    form = dict(urllib.parse.parse_qsl(raw.decode(errors="replace")))

                def _fv(k, default=""):
                    v = form.get(k)
                    if v is None: return default
                    return v if isinstance(v, str) else v.decode(errors="replace")

                kind = _fv("kind"); name = _fv("name")
                if (kind not in ("character", "environment") or not name
                        or "/" in name or "\\" in name or ".." in name):
                    self._send(400, "application/json",
                               _j.dumps({"error": "invalid kind/name"}))
                    return
                try:
                    count = max(1, min(8, int(_fv("count", "4") or 4)))
                except ValueError:
                    count = 4
                guide = _fv("guide", "1") not in ("0", "false", "no", "")
                job_id = _u.uuid4().hex[:12]
                threading.Thread(target=_run_regen_job,
                                 args=(job_id, kind, name, count, guide),
                                 daemon=True).start()
                self._send(200, "application/json", _j.dumps({"job_id": job_id}))
                return

            if path == "/profile/train-lora":
                import json as _j, uuid as _u
                clen = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(clen)
                ctype = self.headers.get("Content-Type", "")
                if "multipart/form-data" in ctype:
                    boundary = ctype.split("boundary=")[-1].strip().encode()
                    form = _parse_multipart(raw, boundary)
                else:
                    form = dict(urllib.parse.parse_qsl(raw.decode(errors="replace")))

                def _fv(k, default=""):
                    v = form.get(k)
                    if v is None: return default
                    return v if isinstance(v, str) else v.decode(errors="replace")

                kind = _fv("kind"); name = _fv("name")
                if (kind not in ("character", "environment") or not name
                        or "/" in name or "\\" in name or ".." in name):
                    self._send(400, "application/json",
                               _j.dumps({"error": "invalid kind/name"}))
                    return
                try:
                    steps = max(50, min(5000, int(_fv("steps", "800") or 800)))
                except ValueError:
                    steps = 800
                try:
                    rank = max(2, min(128, int(_fv("rank", "16") or 16)))
                except ValueError:
                    rank = 16
                job_id = _u.uuid4().hex[:12]
                threading.Thread(target=_run_train_lora_job,
                                 args=(job_id, kind, name, steps, rank),
                                 daemon=True).start()
                self._send(200, "application/json", _j.dumps({"job_id": job_id}))
                return

            if path == "/matches/render":
                import json as _j, uuid as _u
                clen = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(clen)
                ctype = self.headers.get("Content-Type", "")
                if "multipart/form-data" in ctype:
                    boundary = ctype.split("boundary=")[-1].strip().encode()
                    form = _parse_multipart(raw, boundary)
                else:
                    form = dict(urllib.parse.parse_qsl(raw.decode(errors="replace")))

                def _fv(k, default=""):
                    v = form.get(k)
                    if v is None: return default
                    return v if isinstance(v, str) else v.decode(errors="replace")

                scope = _fv("scope")
                if scope not in ("match-clips", "clip", "reassemble", "outcomes", "outcome"):
                    self._send(400, "application/json",
                               _j.dumps({"error": "invalid scope"}))
                    return
                params = {}
                for k in ("match", "idx", "fighter", "outcome"):
                    val = _fv(k)
                    if val:
                        # Guard path-like fields against traversal.
                        if k in ("match", "fighter") and ("/" in val or "\\" in val or ".." in val):
                            self._send(400, "application/json",
                                       _j.dumps({"error": f"invalid {k}"}))
                            return
                        params[k] = val
                job_id = _u.uuid4().hex[:12]
                threading.Thread(target=_run_match_job,
                                 args=(job_id, scope, params),
                                 daemon=True).start()
                self._send(200, "application/json", _j.dumps({"job_id": job_id}))
                return

            if path == "/prompts/save":
                import json as _j
                clen = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(clen)
                ctype = self.headers.get("Content-Type", "")
                if "multipart/form-data" in ctype:
                    boundary = ctype.split("boundary=")[-1].strip().encode()
                    form = _parse_multipart(raw, boundary)
                else:
                    form = dict(urllib.parse.parse_qsl(raw.decode(errors="replace")))

                def _fv(k, default=""):
                    v = form.get(k)
                    if v is None: return default
                    return v if isinstance(v, str) else v.decode(errors="replace")

                if _fv("reset"):
                    apply_prompts_config(_PROMPT_DEFAULTS)
                    try:
                        p = _prompts_config_path(out_dir)
                        if p.exists():
                            p.unlink()
                    except Exception:
                        pass
                    self._send(200, "application/json", _j.dumps({"ok": True, "reset": True}))
                    return
                try:
                    cfg = _j.loads(_fv("config", "{}"))
                except Exception as e:
                    self._send(400, "application/json", _j.dumps({"error": f"bad config: {e}"}))
                    return
                if not isinstance(cfg, dict):
                    self._send(400, "application/json", _j.dumps({"error": "config must be an object"}))
                    return
                apply_prompts_config(cfg)
                try:
                    save_prompts_config(out_dir, cfg)
                except Exception as e:
                    self._send(500, "application/json", _j.dumps({"error": f"cannot save: {e}"}))
                    return
                self._send(200, "application/json", _j.dumps({"ok": True}))
                return

            if path in ("/matches/save", "/matches/delete"):
                import json as _j
                clen = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(clen)
                ctype = self.headers.get("Content-Type", "")
                if "multipart/form-data" in ctype:
                    boundary = ctype.split("boundary=")[-1].strip().encode()
                    form = _parse_multipart(raw, boundary)
                else:
                    form = dict(urllib.parse.parse_qsl(raw.decode(errors="replace")))

                def _fv(k, default=""):
                    v = form.get(k)
                    if v is None: return default
                    return v if isinstance(v, str) else v.decode(errors="replace")

                def _safe(s):
                    return s and "/" not in s and "\\" not in s and ".." not in s

                vdir = out_dir / "videos"
                prompts_file = vdir / "prompts.json"

                if path == "/matches/save":
                    try:
                        data = json.loads(prompts_file.read_text()) if prompts_file.exists() else {}
                    except Exception:
                        data = {}
                    mode = _fv("mode")
                    if mode == "match":
                        nm = _fv("name")
                        if not _safe(nm):
                            self._send(400, "application/json", _j.dumps({"error": "invalid name"})); return
                        m = next((x for x in data.get("fight_plan", []) if x.get("match_name") == nm), None)
                        if not m:
                            self._send(404, "application/json", _j.dumps({"error": "match not in prompts.json"})); return
                        if "f1" in form: m["f1"] = _fv("f1") or m.get("f1")
                        if "f2" in form: m["f2"] = _fv("f2") or m.get("f2")
                        if "env" in form: m["env"] = _fv("env") or None
                        for _tk in ("short_target", "long_target"):
                            if _tk in form:
                                try: m[_tk] = float(_fv(_tk))
                                except ValueError: pass
                        for c in m.get("clips", []):
                            key = f"clip_{c['idx']}"
                            if key in form:
                                c["prompt"] = _fv(key)
                        # Outcome prompts for this match: keys "outc_<fighter>|<outcome>".
                        for fk in list(form.keys()):
                            if not fk.startswith("outc_"):
                                continue
                            spec = fk[5:]
                            fr_o, _, oc_o = spec.rpartition("|")
                            if not fr_o or not oc_o:
                                continue
                            o = next((x for x in data.get("outcome_plan", [])
                                      if x.get("match_name") == nm
                                      and x.get("fighter") == fr_o
                                      and x.get("outcome") == oc_o), None)
                            if o is None:
                                # Create the entry so the prompt is kept even if the
                                # plan didn't have it yet.
                                o = {"match_name": nm, "fighter": fr_o, "outcome": oc_o,
                                     "env": m.get("env"), "env_desc": _env_description(m.get("env"))
                                     if m.get("env") else "African township"}
                                data.setdefault("outcome_plan", []).append(o)
                            o["prompt"] = _fv(fk)
                    elif mode == "outputs":
                        fr = _fv("fighter")
                        if not _safe(fr):
                            self._send(400, "application/json", _j.dumps({"error": "invalid fighter"})); return
                        for o in data.get("outcome_plan", []):
                            if o.get("fighter") == fr:
                                key = f"out_{o['outcome']}"
                                if key in form:
                                    o["prompt"] = _fv(key)
                    else:
                        self._send(400, "application/json", _j.dumps({"error": "invalid mode"})); return
                    try:
                        vdir.mkdir(parents=True, exist_ok=True)
                        prompts_file.write_text(json.dumps(data, indent=2))
                    except Exception as e:
                        self._send(500, "application/json", _j.dumps({"error": f"cannot save: {e}"})); return
                    self._send(200, "application/json", _j.dumps({"ok": True}))
                    return

                # /matches/delete — remove rendered video files (keep plan/keyframes)
                scope = _fv("scope")
                removed = 0

                def _rm(p: Path):
                    nonlocal removed
                    try:
                        if p.exists() and p.is_file():
                            p.unlink(); removed += 1
                    except Exception:
                        pass

                if scope == "clip":
                    mn, idx = _fv("match"), _fv("idx")
                    if not _safe(mn):
                        self._send(400, "application/json", _j.dumps({"error": "invalid match"})); return
                    try:
                        _rm(vdir / f"{mn}_clip{int(idx):02d}.mp4")
                    except ValueError:
                        pass
                elif scope == "final":
                    mn, which = _fv("match"), _fv("which")
                    if not _safe(mn) or which not in ("short", "long"):
                        self._send(400, "application/json", _j.dumps({"error": "invalid args"})); return
                    _rm(vdir / f"{mn}_{which}.mp4")
                elif scope == "match":
                    mn = _fv("match")
                    if not _safe(mn):
                        self._send(400, "application/json", _j.dumps({"error": "invalid match"})); return
                    # Use the scanner so a sibling match (e.g. "<mn>_2") is not
                    # caught by a naive "<mn>_*" glob.
                    _, _, _, _matches_map, _ = _scan_matches()
                    _info = _matches_map.get(mn, {})
                    for _p in list(_info.get("finals", {}).values()):
                        _rm(_p)
                    for _p in _info.get("clips", []):
                        _rm(_p)
                    for (_f, _o, _p) in _info.get("outcomes", []):
                        _rm(_p)
                elif scope == "output":
                    mn, fr, oc = _fv("match"), _fv("fighter"), _fv("outcome")
                    if not _safe(fr) or not _safe(oc):
                        self._send(400, "application/json", _j.dumps({"error": "invalid args"})); return
                    if mn and _safe(mn):
                        _rm(vdir / f"{mn}_{fr}_{oc}.mp4")        # per-match outcome
                    else:
                        _rm(vdir / f"{fr}_{oc}.mp4")             # legacy per-fighter
                elif scope == "outputs":
                    mn, fr = _fv("match"), _fv("fighter")
                    if mn and _safe(mn):
                        _, _, _, _matches_map, _ = _scan_matches()
                        for (_f, _o, _p) in _matches_map.get(mn, {}).get("outcomes", []):
                            _rm(_p)
                    elif fr and _safe(fr):
                        for p in vdir.glob(f"{fr}_*.mp4"):
                            if "_clip" not in p.stem and not p.stem.startswith("match_"):
                                _rm(p)
                    else:
                        self._send(400, "application/json", _j.dumps({"error": "invalid args"})); return
                elif scope == "file":
                    fn = _fv("file")
                    if not _safe(fn) or not fn.endswith(".mp4"):
                        self._send(400, "application/json", _j.dumps({"error": "invalid file"})); return
                    _rm(vdir / fn)
                else:
                    self._send(400, "application/json", _j.dumps({"error": "invalid scope"})); return
                self._send(200, "application/json", _j.dumps({"ok": True, "removed": removed}))
                return

            if path == "/profile/upload-image":
                # Append user-uploaded image files as new references, preserving
                # all existing ones.
                import json as _j
                clen = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(clen)
                ctype = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in ctype:
                    self._send(400, "application/json",
                               _j.dumps({"error": "expected multipart/form-data"}))
                    return
                boundary = ctype.split("boundary=")[-1].strip().encode()
                fields, files = _parse_multipart_full(raw, boundary)
                kind = fields.get("kind", "")
                name = fields.get("name", "")
                if (kind not in ("character", "environment") or not name
                        or "/" in name or "\\" in name or ".." in name):
                    self._send(400, "application/json",
                               _j.dumps({"error": "invalid kind/name"}))
                    return

                def _img_ext(data: bytes):
                    if data[:4] == b"\x89PNG": return ".png", "image/png"
                    if data[:2] == b"\xff\xd8": return ".jpg", "image/jpeg"
                    if data[:4] == b"RIFF" and data[8:12] == b"WEBP": return ".webp", "image/webp"
                    if data[:6] in (b"GIF87a", b"GIF89a"): return ".gif", "image/gif"
                    return None, None

                base = out_dir / (kind + "s") / name
                base.mkdir(parents=True, exist_ok=True)
                added_uris, rejected = [], 0
                for f in files:
                    data = f.get("data") or b""
                    ext, mime = _img_ext(data)
                    if not ext:
                        rejected += 1
                        continue
                    out_p = _next_ref_path(base, ext)
                    try:
                        out_p.write_bytes(data)
                    except Exception:
                        rejected += 1
                        continue
                    added_uris.append("data:%s;base64,%s" % (
                        mime, __import__("base64").b64encode(data).decode()))
                if not added_uris:
                    self._send(400, "application/json",
                               _j.dumps({"error": "no valid image files uploaded"}))
                    return
                synced = True
                try:
                    client = CoderAIClient(default_args.base_url,
                                           getattr(default_args, "api_key", None))
                    client.patch_profile(kind, name, add_images=added_uris)
                except Exception:
                    synced = False
                self._send(200, "application/json",
                           _j.dumps({"ok": True, "added": len(added_uris),
                                     "rejected": rejected, "synced": synced}))
                return

            if path in ("/profile/save", "/profile/delete", "/profile/delete-image"):
                import json as _j
                import shutil as _shutil
                clen = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(clen)
                ctype = self.headers.get("Content-Type", "")
                if "multipart/form-data" in ctype:
                    boundary = ctype.split("boundary=")[-1].strip().encode()
                    form = _parse_multipart(raw, boundary)
                else:
                    form = dict(urllib.parse.parse_qsl(raw.decode(errors="replace")))

                def _fv(k, default=""):
                    v = form.get(k)
                    if v is None: return default
                    return v if isinstance(v, str) else v.decode(errors="replace")

                kind = _fv("kind")
                name = _fv("name")
                # Reject anything that could escape the profile directory.
                if (kind not in ("character", "environment") or not name
                        or "/" in name or "\\" in name or ".." in name):
                    self._send(400, "application/json",
                               _j.dumps({"error": "invalid kind/name"}))
                    return

                base = out_dir / (kind + "s") / name
                client = CoderAIClient(default_args.base_url,
                                       getattr(default_args, "api_key", None))

                if path == "/profile/delete":
                    try:
                        if base.exists():
                            _shutil.rmtree(base)
                    except Exception as e:
                        self._send(500, "application/json",
                                   _j.dumps({"error": f"cannot delete local: {e}"}))
                        return
                    synced = True
                    try:
                        client.delete_profile(kind, name)
                    except Exception:
                        synced = False
                    self._send(200, "application/json",
                               _j.dumps({"ok": True, "synced": synced}))
                    return

                if path == "/profile/delete-image":
                    file = _fv("file")
                    if not file or "/" in file or "\\" in file or ".." in file:
                        self._send(400, "application/json",
                                   _j.dumps({"error": "invalid file"}))
                        return
                    imgs = sorted(
                        p.name for p in base.iterdir()
                        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                    ) if base.exists() else []
                    idx = imgs.index(file) if file in imgs else None
                    fp = base / file
                    try:
                        if fp.exists():
                            fp.unlink()
                    except Exception as e:
                        self._send(500, "application/json",
                                   _j.dumps({"error": f"cannot delete image: {e}"}))
                        return
                    if idx is not None:
                        try:
                            client.patch_profile(kind, name, remove_indices=[idx])
                        except Exception:
                            pass
                    self._send(200, "application/json", _j.dumps({"ok": True}))
                    return

                # /profile/save — update local meta.json (+ sync description to server)
                meta_path = base / "meta.json"
                meta = {}
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    pass
                # Write back every submitted field (the editor exposes all of
                # them), skipping control + bookkeeping keys.
                _reserved = {"kind", "name", "images", "image_count",
                             "created_at", "created"}
                for fk in form.keys():
                    if fk in _reserved:
                        continue
                    meta[fk] = _fv(fk)
                meta["name"] = name
                try:
                    base.mkdir(parents=True, exist_ok=True)
                    meta_path.write_text(json.dumps(meta, indent=2))
                except Exception as e:
                    self._send(500, "application/json",
                               _j.dumps({"error": f"cannot save: {e}"}))
                    return
                synced = False
                try:
                    client.patch_profile(kind, name, description=meta.get("description", ""))
                    synced = True
                except Exception:
                    pass
                self._send(200, "application/json",
                           _j.dumps({"ok": True, "synced": synced}))
                return

            if path == "/save-config":
                # Write the submitted options to a config file ON THE SERVER
                # (the machine running this script), reusable later via --config.
                import json as _j
                clen = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(clen)
                ctype = self.headers.get("Content-Type", "")
                if "multipart/form-data" in ctype:
                    boundary = ctype.split("boundary=")[-1].strip().encode()
                    form = _parse_multipart(raw, boundary)
                else:
                    form = dict(urllib.parse.parse_qsl(raw.decode(errors="replace")))

                def _fv(k, default=""):
                    v = form.get(k)
                    if v is None: return default
                    return v if isinstance(v, str) else v.decode(errors="replace")

                def _s(v):
                    v = (v or "").strip()
                    return v or None

                cm = _fv("char_mode", "generate")
                em = _fv("env_mode", "generate")
                s3 = _fv("stage3_mode", "full")
                sc = _fv("only_stage", "all")
                cfg = {
                    "base_url": _fv("base_url", "http://127.0.0.1:8776"),
                    "api_key": _s(_fv("api_key")),
                    "image_model": _s(_fv("image_model")),
                    "video_model": _s(_fv("video_model")),
                    "text_model": _s(_fv("text_model")),
                    "no_llm": "no_llm" in form,
                    "out_dir": _fv("out_dir", "./township_output"),
                    "region": _s(_fv("region")),
                    "include_female": "include_female" in form,
                    "fps": int(_fv("fps", "8") or 8),
                    "clip_delay": float(_fv("clip_delay", "5") or 5),
                    "matches": int(_fv("matches", "6") or 6),
                    "skip_videos": "skip_videos" in form,
                    "only_outcomes": "only_outcomes" in form,
                    "consistency": _fv("consistency", "keyframe"),
                    "keyframe_steps": int(_fv("keyframe_steps", "28") or 28),
                    "keyframe_size": _fv("keyframe_size", "512x512"),
                    "character_strength": float(_fv("character_strength", "0.7") or 0.7),
                    "lora_steps": int(_fv("lora_steps", "800") or 800),
                    "lora_rank": int(_fv("lora_rank", "16") or 16),
                    "lora_weight": float(_fv("lora_weight", "0.85") or 0.85),
                    "lora_train_base_model": _s(_fv("lora_train_base_model")) or "",
                    "no_env_loras": "env_loras" not in form,
                    "env_lora_steps": int(_fv("env_lora_steps", "800") or 800),
                    "env_lora_rank": int(_fv("env_lora_rank", "16") or 16),
                    "env_lora_weight": float(_fv("env_lora_weight", "0.8") or 0.8),
                    "skip_characters": cm == "skip",
                    "reuse_fighters": cm == "reuse",
                    "fighters": _s(_fv("fighters")) if cm == "fighters" else None,
                    "skip_environments": em == "skip",
                    "reuse_environments": em == "reuse",
                    "environments": _s(_fv("environments")) if em == "environments" else None,
                    "only_prompts": s3 == "only_prompts",
                    "only_videos": s3 == "only_videos",
                    "only_characters": sc == "characters",
                    "only_environments": sc == "environments",
                    "only_assets": sc == "assets",
                    "web_port": port,
                }
                # Apply the changed connection/model settings to the live session
                # immediately, so subsequent per-profile jobs (regenerate, train
                # LoRA) and runs use them — not the values from script launch.
                for _k in ("base_url", "api_key", "image_model",
                           "video_model", "text_model", "lora_train_base_model"):
                    setattr(default_args, _k, cfg.get(_k))
                _web_log(f"  ⚙ Settings applied (image model: "
                         f"{cfg.get('image_model') or 'auto'})")
                # Resolve the target path. Relative paths land inside out_dir;
                # the filename is sanitised to its basename for relative saves to
                # avoid writing outside the output tree from the browser.
                req_path = _fv("path", "").strip() or "township_config.json"
                p = Path(req_path)
                if p.is_absolute():
                    target = p
                else:
                    target = Path(default_args.out_dir) / os.path.basename(req_path)
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with open(target, "w", encoding="utf-8") as f:
                        _j.dump(cfg, f, indent=2, sort_keys=True)
                        f.write("\n")
                    self._send(200, "application/json",
                               _j.dumps({"ok": True, "path": str(target.resolve())}))
                except Exception as e:
                    self._send(500, "application/json",
                               _j.dumps({"error": f"cannot save: {e}"}))
                return

            if path == "/process":
                import json as _j, uuid as _u
                clen = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(clen)
                ctype = self.headers.get("Content-Type", "")
                if "multipart/form-data" in ctype:
                    boundary = ctype.split("boundary=")[-1].strip().encode()
                    form = _parse_multipart(raw, boundary)
                else:
                    form = dict(urllib.parse.parse_qsl(raw.decode(errors="replace")))
                rel = form.get("file", "")
                op  = form.get("op", "")
                param_raw = form.get("param", "null")
                try:
                    param = _j.loads(param_raw)
                except Exception:
                    param = param_raw
                fpath = out_dir / rel
                if not fpath.exists() or fpath.suffix.lower() not in (".mp4",".webm",".mov"):
                    self._send(400, "application/json", _j.dumps({"error": "Invalid file"}))
                    return
                if op not in ("upscale", "fps", "upscale_fps"):
                    self._send(400, "application/json", _j.dumps({"error": f"Unknown op: {op}"}))
                    return
                job_id = _u.uuid4().hex[:12]
                threading.Thread(
                    target=_run_process_job,
                    args=(job_id, fpath, op, param),
                    daemon=True
                ).start()
                self._send(200, "application/json", _j.dumps({"job_id": job_id}))
                return

            if path != "/start":
                self._send(404, "text/plain", "Not found")
                return

            if _state["running"]:
                import json as _j
                self._send(409, "application/json", _j.dumps({"error": "Already running"}))
                return

            # Parse multipart/form-data
            import json as _j
            clen = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(clen)
            ctype = self.headers.get("Content-Type", "")
            if "multipart/form-data" in ctype:
                boundary = ctype.split("boundary=")[-1].strip().encode()
                form = _parse_multipart(raw, boundary)
            else:
                form = dict(urllib.parse.parse_qsl(raw.decode(errors="replace")))

            def _fv(k, default=""):
                v = form.get(k)
                if v is None: return default
                return v if isinstance(v, str) else v.decode(errors="replace")

            # Build a namespace from form values
            import argparse as _ap
            ns = _ap.Namespace()
            ns.base_url     = _fv("base_url", "http://127.0.0.1:8776")
            ns.api_key      = _fv("api_key") or None
            ns.image_model  = _fv("image_model") or None
            ns.video_model  = _fv("video_model") or None
            ns.text_model   = _fv("text_model") or None
            ns.no_llm       = "no_llm" in form
            ns.out_dir      = _fv("out_dir", "./township_output")
            ns.region       = _fv("region") or None
            ns.include_female = "include_female" in form
            ns.fps          = int(_fv("fps", "8"))
            ns.clip_delay   = float(_fv("clip_delay", "5.0"))
            ns.matches      = int(_fv("matches", "6"))
            ns.skip_videos  = "skip_videos" in form
            ns.only_outcomes = "only_outcomes" in form
            # consistency config
            ns.consistency       = _fv("consistency", "keyframe")
            ns.keyframe_steps    = int(_fv("keyframe_steps", "28"))
            ns.keyframe_size     = _fv("keyframe_size", "512x512")
            ns.character_strength = float(_fv("character_strength", "0.7"))
            ns.lora_steps        = int(_fv("lora_steps", "800"))
            ns.lora_rank         = int(_fv("lora_rank", "16"))
            ns.lora_weight       = float(_fv("lora_weight", "0.85"))
            ns.lora_train_base_model = (_fv("lora_train_base_model", "") or "").strip()
            # Environment LoRAs: checkbox "env_loras" present ⇒ train them.
            ns.no_env_loras      = ("env_loras" not in form)
            ns.env_lora_steps    = int(_fv("env_lora_steps", "800"))
            ns.env_lora_rank     = int(_fv("env_lora_rank", "16"))
            ns.env_lora_weight   = float(_fv("env_lora_weight", "0.8"))
            # char mode
            cm = _fv("char_mode", "generate")
            ns.skip_characters  = (cm == "skip")
            ns.reuse_fighters   = (cm == "reuse")
            ns.fighters         = _fv("fighters") or None if cm == "fighters" else None
            # env mode
            em = _fv("env_mode", "generate")
            ns.skip_environments  = (em == "skip")
            ns.reuse_environments = (em == "reuse")
            ns.environments       = _fv("environments") or None if em == "environments" else None
            # stage3 mode
            s3 = _fv("stage3_mode", "full")
            ns.only_prompts = (s3 == "only_prompts")
            ns.only_videos  = (s3 == "only_videos")
            # only-stage
            os_mode = _fv("only_stage", "all")
            ns.only_characters  = (os_mode == "characters")
            ns.only_environments= (os_mode == "environments")
            ns.only_assets      = (os_mode == "assets")
            ns.cli_mode = True
            ns.web_port = port

            # Apply the submitted connection/model settings to the live session
            # so later per-profile jobs (regenerate, train LoRA) use them too.
            for _k in ("base_url", "api_key", "image_model",
                       "video_model", "text_model", "lora_train_base_model"):
                setattr(default_args, _k, getattr(ns, _k, None))

            # Apply the same shortcut logic as CLI
            if ns.only_characters:
                ns.skip_environments = True; ns.skip_videos = True
            if ns.only_environments:
                ns.skip_characters = True;   ns.skip_videos = True
            if ns.only_assets:
                ns.skip_videos = True
            if ns.only_videos:
                ns.skip_characters = True;   ns.skip_environments = True

            # ── Step buttons: map a single step to the right flags ──────────────
            # Each step assumes the previous ones already produced their output
            # (stages are resumable / pick up saved state on disk).
            ns.only_loras = False
            ns.only_keyframes = False
            step = _fv("step", "").strip()
            if step:
                ns.only_characters = ns.only_environments = ns.only_assets = False
                ns.only_prompts = ns.only_videos = False
                ns.skip_characters = ns.skip_environments = ns.skip_videos = False
                if step == "characters":
                    ns.only_characters = True; ns.skip_environments = True; ns.skip_videos = True
                elif step == "environments":
                    ns.only_environments = True; ns.skip_characters = True; ns.skip_videos = True
                elif step == "prompts":
                    ns.skip_characters = True; ns.skip_environments = True; ns.only_prompts = True
                elif step == "loras":
                    ns.skip_characters = True; ns.skip_environments = True
                    ns.skip_videos = True; ns.only_loras = True
                elif step == "keyframes":
                    ns.skip_characters = True; ns.skip_environments = True
                    ns.only_keyframes = True
                elif step == "videos":
                    ns.skip_characters = True; ns.skip_environments = True; ns.only_videos = True

            # Human-readable label for what's being executed, shown as a banner
            # at the top of the (freshly cleared) log so it's always obvious
            # which run/step is currently in progress.
            _STEP_LABELS = {
                "characters":   "Step 1 · Generate Characters",
                "environments": "Step 2 · Generate Environments",
                "prompts":      "Step 3 · Write Video Prompts",
                "loras":        "Step 4 · Train Character LoRAs",
                "keyframes":    "Step 5 · Generate Keyframes",
                "videos":       "Step 6 · Render Videos",
            }
            if step:
                run_label = _STEP_LABELS.get(step, f"Step · {step}")
            elif ns.only_characters:
                run_label = "Full Run · Characters only"
            elif ns.only_environments:
                run_label = "Full Run · Environments only"
            elif ns.only_assets:
                run_label = "Full Run · Assets only (characters + environments)"
            elif ns.only_prompts:
                run_label = "Full Run · Prompts only"
            elif ns.only_videos:
                run_label = "Full Run · Videos only (render from saved prompts)"
            else:
                run_label = "Full Run · All stages"

            _state["abort"].clear()
            _state["log_lines"].clear()
            _state["done"] = False
            _state["running"] = True
            _state["current"] = run_label

            _bar = "━" * 58
            _web_log(_bar)
            _web_log(f"▶ {run_label}    [{time.strftime('%H:%M:%S')}]")
            _web_log(f"   consistency: {ns.consistency}    output: {ns.out_dir}")
            _web_log(_bar)

            def _run_thread():
                import sys as _sys
                _m = _sys.modules[__name__]
                _m._log = _patched_log
                try:
                    _run_main_with_args(ns)
                except Exception as exc:
                    _web_log(f"✗ Fatal error: {exc}")
                finally:
                    _state["running"] = False
                    _state["done"] = True
                    _web_log(f"✓ {run_label} — complete.")
                    with _sse_lock:
                        for q in list(_sse_clients):
                            try: q.put(None)
                            except Exception: pass

            threading.Thread(target=_run_thread, daemon=True).start()
            self._send(200, "application/json", _j.dumps({"ok": True}))

    def _parse_multipart(body: bytes, boundary: bytes) -> dict:
        """Minimal multipart/form-data parser."""
        result = {}
        delimiter = b"--" + boundary
        parts = body.split(delimiter)
        for part in parts[1:]:
            if part.strip() in (b"", b"--", b"--\r\n"):
                continue
            if b"\r\n\r\n" not in part:
                continue
            header_raw, _, value = part.partition(b"\r\n\r\n")
            value = value.rstrip(b"\r\n")
            headers_text = header_raw.decode(errors="replace")
            name = None
            for hdr_line in headers_text.splitlines():
                if "Content-Disposition" in hdr_line and 'name="' in hdr_line:
                    name = hdr_line.split('name="')[1].split('"')[0]
            if name:
                result[name] = value.decode(errors="replace")
        return result

    def _parse_multipart_full(body: bytes, boundary: bytes):
        """Parse multipart/form-data preserving raw bytes for file parts.

        Returns (fields, files) where fields maps name→str and files is a list
        of {'name', 'filename', 'data': bytes} for parts with a filename.
        """
        fields, files = {}, []
        delimiter = b"--" + boundary
        for part in body.split(delimiter)[1:]:
            if part[:2] == b"\r\n":
                part = part[2:]
            if part.strip() in (b"", b"--"):
                continue
            if b"\r\n\r\n" not in part:
                continue
            header_raw, _, value = part.partition(b"\r\n\r\n")
            if value.endswith(b"\r\n"):
                value = value[:-2]
            headers_text = header_raw.decode(errors="replace")
            name = filename = None
            for hdr_line in headers_text.splitlines():
                if "Content-Disposition" in hdr_line:
                    if 'name="' in hdr_line:
                        name = hdr_line.split('name="')[1].split('"')[0]
                    if 'filename="' in hdr_line:
                        filename = hdr_line.split('filename="')[1].split('"')[0]
            if name is None:
                continue
            if filename is not None:
                files.append({"name": name, "filename": filename, "data": value})
            else:
                fields[name] = value.decode(errors="replace")
        return fields, files

    def _run_main_with_args(args):
        """Run the full generation pipeline with a pre-built args Namespace."""
        # Identical logic to main() after parse_args(), driven by args.
        out_dir_r = Path(args.out_dir)
        out_dir_r.mkdir(parents=True, exist_ok=True)
        apply_prompts_config(load_prompts_config(out_dir_r))

        _web_log("╔══════════════════════════════════════════════════════════╗")
        _web_log("║       Township Fighters — Content Generator              ║")
        _web_log("╚══════════════════════════════════════════════════════════╝")
        _web_log(f"  CoderAI: {args.base_url}")
        _web_log(f"  Output:  {out_dir_r.resolve()}")
        if args.region:
            _web_log(f"  Region filter: {args.region}")

        client = CoderAIClient(args.base_url, args.api_key)
        try:
            models = client.list_models()
            _web_log(f"  Connected — {len(models)} model(s) available")
        except Exception as e:
            _web_log(f"\n✗ Cannot reach CoderAI at {args.base_url}: {e}")
            return

        _cons = parse_consistency(getattr(args, "consistency", "keyframe"))
        # The image model is needed to generate characters/environments, AND for
        # keyframe/LoRA consistency whenever we'll render, train, or make keyframes.
        _cons_needs_image = (("keyframe" in _cons) or ("lora" in _cons)) and (
            not args.skip_videos or getattr(args, "only_loras", False)
            or getattr(args, "only_keyframes", False))
        need_image = (not (args.skip_characters or args.reuse_fighters or args.fighters)
                      or not (args.skip_environments or args.reuse_environments or args.environments)
                      or _cons_needs_image)
        image_model = pick_model(client, "image", args.image_model) if need_image else None

        need_video = (not args.skip_videos and not args.only_prompts
                      and not getattr(args, "only_loras", False)
                      and not getattr(args, "only_keyframes", False))
        video_model = pick_model(client, "video", args.video_model) if need_video else None

        text_model = None
        need_text = not args.skip_videos and not args.no_llm and not args.only_videos
        if need_text:
            try:
                text_model = pick_model(client, "text", args.text_model)
                if text_model:
                    _web_log(f"  LLM prompts: enabled ({text_model})")
            except Exception:
                _web_log("  LLM prompts: disabled (no text model available)")

        char_descriptions = _build_char_descriptions(out_dir_r)
        prompter = PromptGenerator(client, text_model, char_descriptions=char_descriptions)
        consistency = parse_consistency(getattr(args, "consistency", "keyframe"))
        _web_log(f"  Consistency strategy: {', '.join(sorted(consistency))}")

        char_names = None
        if args.fighters or args.reuse_fighters:
            char_names = resolve_fighters(client, args, out_dir_r)
        elif args.skip_characters:
            char_names = _load_local_profiles(out_dir_r, "character")
            if char_names:
                _web_log(f"  Skipping generation, using {len(char_names)} locally saved fighter(s): {', '.join(char_names)}")
                for n in char_names:
                    _ensure_in_coderai(client, "character", n, out_dir_r)
            else:
                char_names = [f["name"] for f in FIGHTER_POOL if f.get("gender","male") == "male" or args.include_female]
                _web_log(f"  Skipping generation, assuming pool names: {', '.join(char_names)}")
        else:
            char_names = stage_characters(client, image_model, out_dir_r,
                                          region_filter=args.region,
                                          include_female=args.include_female)

        env_names = None
        if args.environments or args.reuse_environments:
            env_names = resolve_environments(client, args, out_dir_r)
        elif args.skip_environments:
            env_names = _load_local_profiles(out_dir_r, "environment")
            if env_names:
                _web_log(f"  Skipping generation, using {len(env_names)} locally saved environment(s): {', '.join(env_names)}")
                for n in env_names:
                    _ensure_in_coderai(client, "environment", n, out_dir_r)
            else:
                env_names = [e["name"] for e in ENVIRONMENT_POOL]
                _web_log(f"  Skipping generation, assuming pool names: {', '.join(env_names)}")
        else:
            env_names = stage_environments(client, image_model, out_dir_r, region_filter=args.region)

        only_loras = getattr(args, "only_loras", False)
        only_keyframes = getattr(args, "only_keyframes", False)

        # Load any previously-trained LoRA maps from disk so keyframe/video steps
        # can reuse them without retraining (characters + environments).
        def _load_map(fname):
            fp = out_dir_r / fname
            if fp.exists():
                try:
                    return json.loads(fp.read_text()) or {}
                except Exception:
                    return {}
            return {}
        lora_map = _load_map("loras.json")
        env_lora_map = _load_map("env_loras.json")
        _no_env_loras = getattr(args, "no_env_loras", False)
        _env_lora_weight = getattr(args, "env_lora_weight", 0.8)

        # Train LoRAs when requested. The explicit "Train LoRAs" step (only_loras)
        # always trains regardless of the consistency strategy — the user asked
        # for it directly. A full run trains only when 'lora' is in consistency.
        _want_lora = only_loras or ("lora" in consistency and not args.skip_videos)
        if _want_lora:
            if char_names:
                _web_log(f"  Training character LoRAs for {len(char_names)} fighter(s): "
                         f"{', '.join(char_names)}")
                lora_map = stage_loras(client, image_model, out_dir_r, char_names or [],
                                       lora_steps=getattr(args, "lora_steps", 800),
                                       lora_rank=getattr(args, "lora_rank", 16),
                                       train_base_model=getattr(args, "lora_train_base_model", None) or None)
            else:
                _web_log("  ⚠ No characters found to train LoRAs for. Generate or "
                         "select fighters first (Characters page).")
            if not _no_env_loras:
                if env_names:
                    _web_log(f"  Training environment LoRAs for {len(env_names)} location(s): "
                             f"{', '.join(env_names)}")
                    env_lora_map = stage_env_loras(client, image_model, out_dir_r, env_names or [],
                                                   lora_steps=getattr(args, "env_lora_steps", 800),
                                                   lora_rank=getattr(args, "env_lora_rank", 16),
                                                   train_base_model=getattr(args, "lora_train_base_model", None) or None)
                else:
                    _web_log("  ⚠ No environments found to train LoRAs for.")

        if only_loras:
            _web_log(f"\n✓ LoRA step complete. "
                     f"Characters: {len(lora_map)} | Environments: {len(env_lora_map)}")
        elif only_keyframes:
            stage_videos(
                client, video_model, out_dir_r,
                char_names=char_names or [], env_names=env_names or [],
                fps=args.fps, clip_delay=args.clip_delay, prompter=prompter,
                num_matches=args.matches, only_outcomes=args.only_outcomes,
                char_descriptions=char_descriptions,
                consistency=consistency, image_model=image_model, lora_map=lora_map,
                char_strength=getattr(args, "character_strength", 0.7),
                keyframe_steps=getattr(args, "keyframe_steps", 28),
                keyframe_size=getattr(args, "keyframe_size", "512x512"),
                lora_weight=getattr(args, "lora_weight", 0.85),
                env_lora_map=env_lora_map, env_lora_weight=_env_lora_weight,
                keyframes_only=True,
            )
            _web_log("\n✓ Keyframe step complete.")
        elif not args.skip_videos:
            stage_videos(
                client, video_model, out_dir_r,
                char_names=char_names or [],
                env_names=env_names or [],
                fps=args.fps,
                clip_delay=args.clip_delay,
                prompter=prompter,
                prompts_only=args.only_prompts,
                videos_only=args.only_videos,
                num_matches=args.matches,
                only_outcomes=args.only_outcomes,
                char_descriptions=char_descriptions,
                consistency=consistency, image_model=image_model,
                lora_map=lora_map,
                char_strength=getattr(args, "character_strength", 0.7),
                keyframe_steps=getattr(args, "keyframe_steps", 28),
                keyframe_size=getattr(args, "keyframe_size", "512x512"),
                lora_weight=getattr(args, "lora_weight", 0.85),
                env_lora_map=env_lora_map, env_lora_weight=_env_lora_weight,
            )

        _web_log("\n✓ Done.")

    # ── Start the server ────────────────────────────────────────────────────
    class _ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = _ThreadedServer(("", port), _Handler)
    url = f"http://localhost:{port}"
    print(f"\n⚔  Township Fighters — Web UI")
    print(f"   Open: {url}")
    print(f"   Press Ctrl+C to stop\n", flush=True)
    # Only auto-open a browser when explicitly asked (--browser); on a headless
    # server webbrowser.open can otherwise spawn a terminal text browser.
    if getattr(default_args, "browser", False):
        try:
            import webbrowser
            threading.Timer(0.4, lambda: webbrowser.open(url)).start()
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nWeb UI stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    all_fighter_names = ", ".join(f["name"] for f in FIGHTER_POOL)
    all_env_names     = ", ".join(e["name"] for e in ENVIRONMENT_POOL)
    sa_fighters   = ", ".join(f["name"] for f in FIGHTER_POOL if "kampala" not in f["region"] and f["region"] not in ("jinja",))
    ug_fighters   = ", ".join(f["name"] for f in FIGHTER_POOL if "kampala" in f["region"] or f["region"] in ("jinja",))
    sa_envs = ", ".join(e["name"] for e in ENVIRONMENT_POOL if e["region"] not in ("katwe","kisenyi","bwaise","kampala_central","jinja","makindye"))
    ug_envs = ", ".join(e["name"] for e in ENVIRONMENT_POOL if e["region"] in ("katwe","kisenyi","bwaise","kampala_central","jinja","makindye"))

    parser = argparse.ArgumentParser(
        description="Township Fighters — Content Generator for CoderAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
WHAT THIS SCRIPT DOES
─────────────────────
  Stage 1 – Characters
    Generates fighter portrait profiles via CoderAI's image model (4 reference
    images each). Fighters span South Africa and Uganda/Kampala.

  Stage 2 – Environments
    Generates township location profiles (3 reference images each).
    Locations span South Africa and Uganda/Kampala.

  Stage 3 – Videos
    a) Fight matches — one short (40-50s) and one long (65-75s) version per
       pair, composed from multiple 4-8s clips. Same clips are reused for both.
    b) Outcome clips — 10-15s per fighter: win, ko_win, retire, draw.
    Prompts are generated by an LLM (--text-model) for maximum variety, or
    fall back to a static template pool if no text model is configured.

FIGHTER POOL
────────────
  South Africa: {sa_fighters}
  Uganda/Kampala: {ug_fighters}

ENVIRONMENT POOL
────────────────
  South Africa: {sa_envs}
  Uganda/Kampala: {ug_envs}

REQUIREMENTS
────────────
  • CoderAI running (default: http://127.0.0.1:8776)
  • Image generation model (for characters + environments)
  • Video generation model (for fight videos)
  • Text/LLM model (optional, for varied prompts — recommended)
  • ffmpeg + ffprobe on PATH (for clip concatenation)

EXAMPLES
────────
  # Full run — auto-selects models, generates everything:
  python tools/gen_township_fighters.py

  # Specify models explicitly:
  python tools/gen_township_fighters.py \\
    --image-model "owner/my-image-model" \\
    --video-model "owner/my-video-model" \\
    --text-model  "owner/my-llm-model"

  # Only generate Uganda/Kampala content:
  python tools/gen_township_fighters.py --region kampala

  # Reuse fighters/environments from the local output directory (re-uploads to
  # CoderAI if needed), then generate new fight videos:
  python tools/gen_township_fighters.py \\
    --reuse-fighters --reuse-environments --video-model "owner/video-model"

  # Pick specific saved profiles by name (loads from local dir, uploads if needed):
  python tools/gen_township_fighters.py \\
    --fighters khumalo,ssebuliba \\
    --environments katwe_backyard,township_ring \\
    --skip-characters --skip-environments

  # Include female fighters (namutebi):
  python tools/gen_township_fighters.py --include-female

  # Generate only videos (characters + environments already done):
  python tools/gen_township_fighters.py \\
    --skip-characters --skip-environments \\
    --video-model "owner/my-video-model"

  # Generate ONLY characters (no environments, no videos):
  python tools/gen_township_fighters.py --only-characters

  # Generate ONLY environments:
  python tools/gen_township_fighters.py --only-environments

  # Generate BOTH characters and environments, but no videos:
  python tools/gen_township_fighters.py --only-assets

  # Stage 3 split — write all video prompts now (saved to videos/prompts.json):
  python tools/gen_township_fighters.py --reuse-fighters --reuse-environments --only-prompts

  # …then render the videos later from those saved prompts (no text model load):
  python tools/gen_township_fighters.py --only-videos

  # Remote CoderAI with API key, custom output, slower FPS:
  python tools/gen_township_fighters.py \\
    --base-url http://192.168.1.10:8776 --api-key sk-abc123 \\
    --out-dir /data/fights --fps 24

  # Increase delay between requests if hitting rate limits:
  python tools/gen_township_fighters.py --clip-delay 15

OUTPUT LAYOUT
─────────────
  <out-dir>/
    characters/
      khumalo/
        meta.json          ← name, gender, region, description, prompt
        ref_00.png         ← reference images fetched from CoderAI
        ref_01.png
        ...
      ssebuliba/
        meta.json
        ref_00.png
        ...
    environments/
      katwe_backyard/
        meta.json
        ref_00.png
        ...
    videos/
      match_khumalo_vs_ssebuliba_short.mp4   ← 40-50s
      match_khumalo_vs_ssebuliba_long.mp4    ← 65-75s
      match_khumalo_vs_ssebuliba_clip00.mp4  ← individual shots
      ...
      khumalo_win.mp4
      khumalo_ko_win.mp4
      khumalo_retire.mp4
      khumalo_draw.mp4
      ...
""",
    )
    parser.add_argument("-c", "--config", default=None, metavar="FILE",
                        help="Load default options from a previously saved JSON config "
                             "file (see --save). Command-line arguments still override "
                             "values from the file.")
    parser.add_argument("-s", "--save", default=None, metavar="FILE",
                        help="Save the selected generation options to a JSON config file "
                             "and exit (no generation is run). Combine with other flags to "
                             "capture them, then reuse later with --config.")
    parser.add_argument("--base-url",  default="http://127.0.0.1:8776", metavar="URL",
                        help="CoderAI base URL (default: http://127.0.0.1:8776)")
    parser.add_argument("--api-key",   default=None, metavar="KEY",
                        help="Bearer token if CoderAI auth is enabled")
    parser.add_argument("--image-model", default=None, metavar="MODEL_ID",
                        help="Image generation model. Auto-selected if omitted.")
    parser.add_argument("--video-model", default=None, metavar="MODEL_ID",
                        help="Video generation model. Auto-selected if omitted.")
    parser.add_argument("--text-model",  default=None, metavar="MODEL_ID",
                        help="LLM for prompt generation (recommended for variety). Auto-selected if omitted.")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM prompt generation even if a text model is available.")
    parser.add_argument("--out-dir",   default="./township_output", metavar="DIR",
                        help="Output directory (default: ./township_output)")
    parser.add_argument("--fps",       type=int, default=8, metavar="N",
                        help="Video FPS (default: 8). Higher = smoother, much slower.")
    parser.add_argument("--clip-delay", type=float, default=5.0, metavar="SECONDS",
                        help="Seconds between video clip requests (default: 5). Raise if rate-limited.")
    parser.add_argument("--region",    default=None, metavar="REGION",
                        help="Filter characters/environments by region keyword, e.g. kampala, soweto, jinja.")

    # Character control
    char_grp = parser.add_mutually_exclusive_group()
    char_grp.add_argument("--skip-characters", action="store_true",
                          help="Skip Stage 1 — do not generate any character profiles.")
    char_grp.add_argument("--reuse-fighters", action="store_true",
                          help="Skip Stage 1 and reuse ALL existing character profiles already in CoderAI.")
    char_grp.add_argument("--fighters", default=None, metavar="NAME,NAME,...",
                          help="Comma-separated fighter profile names to use (skip generation).")

    # Environment control
    env_grp = parser.add_mutually_exclusive_group()
    env_grp.add_argument("--skip-environments", action="store_true",
                         help="Skip Stage 2 — do not generate any environment profiles.")
    env_grp.add_argument("--reuse-environments", action="store_true",
                         help="Skip Stage 2 and reuse ALL existing environment profiles already in CoderAI.")
    env_grp.add_argument("--environments", default=None, metavar="NAME,NAME,...",
                         help="Comma-separated environment profile names to use (skip generation).")

    parser.add_argument("--skip-videos", action="store_true",
                        help="Skip Stage 3 — only generate characters and/or environments.")
    parser.add_argument("--include-female", action="store_true",
                        help="Include female fighters (namutebi). Default is male-only.")

    # Selective generation — convenience shortcuts for running only part of the
    # pipeline.  These just set the underlying --skip-* flags below.
    only_grp = parser.add_mutually_exclusive_group()
    only_grp.add_argument("--only-characters", action="store_true",
                          help="Generate ONLY characters (implies --skip-environments --skip-videos).")
    only_grp.add_argument("--only-environments", action="store_true",
                          help="Generate ONLY environments (implies --skip-characters --skip-videos).")
    only_grp.add_argument("--only-assets", action="store_true",
                          help="Generate ONLY characters AND environments, no videos (implies --skip-videos).")

    # Stage 3 split: prompts vs videos.  Prompts are saved to
    # <out-dir>/videos/prompts.json so a later run can render from them.
    stage3_grp = parser.add_mutually_exclusive_group()
    stage3_grp.add_argument("--only-prompts", action="store_true",
                            help="Stage 3: write & save all video prompts, but render NO videos.")
    stage3_grp.add_argument("--only-videos", action="store_true",
                            help="Stage 3: render videos from previously-saved prompts (skip prompt writing).")

    parser.add_argument("--matches", type=int, default=6, metavar="N",
                        help="Number of fight matches to prepare in Stage 3 (default: 6).")
    parser.add_argument("--only-outcomes", action="store_true",
                        help="Stage 3: prepare ONLY the per-fighter outcome clips, skipping fight "
                             "matches (use when the match videos already exist). Composes with "
                             "--only-prompts / --only-videos.")
    # ── Character consistency ──────────────────────────────────────────────
    cons_grp = parser.add_argument_group("character consistency")
    cons_grp.add_argument(
        "--consistency", default="keyframe", metavar="STRATEGY",
        help="Comma-separated consistency strategies to keep fighters looking the "
             "same across clips (and matching their portraits). Options: "
             "'prompt' (descriptions in prompt, cheapest), "
             "'ipadapter' (portrait images via IP-Adapter where supported), "
             "'keyframe' (generate a keyframe image then animate it via i2v — "
             "balanced default), "
             "'lora' (train a per-fighter identity LoRA, applied to image+video — "
             "strongest, slowest). Stackable, e.g. 'keyframe,lora'. "
             "Default: keyframe")
    cons_grp.add_argument("--keyframe-steps", type=int, default=28, metavar="N",
                          help="Inference steps for keyframe image generation (default: 28).")
    cons_grp.add_argument("--keyframe-size", default="512x512", metavar="WxH",
                          help="Keyframe image resolution (default: 512x512).")
    cons_grp.add_argument("--character-strength", type=float, default=0.7, metavar="F",
                          help="IP-Adapter character reference strength 0-1 (default: 0.7).")
    cons_grp.add_argument("--lora-steps", type=int, default=800, metavar="N",
                          help="Training steps per character LoRA (default: 800).")
    cons_grp.add_argument("--lora-rank", type=int, default=16, metavar="N",
                          help="LoRA rank (default: 16).")
    cons_grp.add_argument("--lora-weight", type=float, default=0.85, metavar="F",
                          help="Weight applied to each character LoRA at generation (default: 0.85).")
    cons_grp.add_argument("--lora-train-base-model", default="", metavar="MODEL",
                          help="Separate UNet-based SD1.x/SDXL model (models.json key or HF id/path) "
                               "to TRAIN LoRAs against, when the generation image model is a "
                               "transformer/DiT (Z-Image, Flux, SD3) this trainer can't target. "
                               "Generation still uses --image-model. Empty = train on --image-model.")
    cons_grp.add_argument("--no-env-loras", action="store_true",
                          help="Do not train/apply per-environment identity LoRAs when the "
                               "'lora' strategy is active (by default environments get LoRAs too).")
    cons_grp.add_argument("--env-lora-steps", type=int, default=800, metavar="N",
                          help="Training steps per environment LoRA (default: 800).")
    cons_grp.add_argument("--env-lora-rank", type=int, default=16, metavar="N",
                          help="Environment LoRA rank (default: 16).")
    cons_grp.add_argument("--env-lora-weight", type=float, default=0.8, metavar="F",
                          help="Weight applied to each environment LoRA at generation (default: 0.8).")

    parser.add_argument("--cli-mode", action="store_true",
                        help="Run in CLI mode (default when --cli-mode is present). "
                             "Without this flag the script launches a web UI instead of processing.")
    parser.add_argument("--web-port", type=int, default=7788, metavar="PORT",
                        help="Port for the web UI (default: 7788, only used without --cli-mode).")
    parser.add_argument("--browser", action="store_true",
                        help="Auto-open a web browser at the UI URL on startup. Off by default "
                             "(avoids spawning a terminal text browser on headless servers).")

    # Two-phase parse: pre-scan for -c/--config so the saved values become
    # parser defaults that explicit command-line arguments can still override.
    pre, _ = parser.parse_known_args()
    if pre.config:
        try:
            cfg = load_config(pre.config)
        except Exception as e:
            parser.error(f"cannot load config {pre.config}: {e}")
        parser.set_defaults(**cfg)
        _log(f"  Loaded {len(cfg)} option(s) from config: {pre.config}")

    args = parser.parse_args()

    # --save: capture the selected options to a config file and exit.
    if args.save:
        try:
            saved = save_config(args.save, args)
        except Exception as e:
            parser.error(f"cannot save config {args.save}: {e}")
        _log(f"✓ Saved configuration ({len(saved)} options) to {args.save}")
        return

    if not args.cli_mode:
        launch_web_ui(args)
        return

    # Apply --only-* shortcuts by enabling the relevant skip flags.
    if args.only_characters:
        args.skip_environments = True
        args.skip_videos = True
    if args.only_environments:
        args.skip_characters = True
        args.skip_videos = True
    if args.only_assets:
        args.skip_videos = True
    # --only-videos renders from saved prompts.json (which already embeds the
    # fighter/environment names), so skip the asset stages — they exist already.
    if args.only_videos:
        args.skip_characters = True
        args.skip_environments = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_prompts_config(load_prompts_config(out_dir))

    _log("╔══════════════════════════════════════════════════════════╗")
    _log("║       Township Fighters — Content Generator              ║")
    _log("╚══════════════════════════════════════════════════════════╝")
    _log(f"  CoderAI: {args.base_url}")
    _log(f"  Output:  {out_dir.resolve()}")
    if args.region:
        _log(f"  Region filter: {args.region}")

    client = CoderAIClient(args.base_url, args.api_key)

    try:
        models = client.list_models()
        _log(f"  Connected — {len(models)} model(s) available")
    except Exception as e:
        _log(f"\n✗ Cannot reach CoderAI at {args.base_url}: {e}")
        sys.exit(1)

    # ── Resolve which fighters/environments to use ─────────────────────────────
    _cons = parse_consistency(getattr(args, "consistency", "keyframe"))
    # Image model also needed for keyframe/LoRA consistency when we'll render.
    _cons_needs_image = (("keyframe" in _cons) or ("lora" in _cons)) and not args.skip_videos
    need_image = (not (args.skip_characters or args.reuse_fighters or args.fighters)
                  or not (args.skip_environments or args.reuse_environments or args.environments)
                  or _cons_needs_image)

    image_model = pick_model(client, "image", args.image_model) if need_image else None
    # Video model is only needed to RENDER (not for --only-prompts).
    need_video = not args.skip_videos and not args.only_prompts
    video_model = pick_model(client, "video", args.video_model) if need_video else None

    # Text model is only needed to WRITE prompts (not for --only-videos, which
    # reuses prompts saved to disk).
    text_model = None
    need_text = not args.skip_videos and not args.no_llm and not args.only_videos
    if need_text:
        try:
            text_model = pick_model(client, "text", args.text_model)
            if text_model:
                _log(f"  LLM prompts: enabled ({text_model})")
        except Exception:
            _log("  LLM prompts: disabled (no text model available)")

    char_descriptions = _build_char_descriptions(out_dir)
    prompter = PromptGenerator(client, text_model, char_descriptions=char_descriptions)
    consistency = parse_consistency(getattr(args, "consistency", "keyframe"))
    _log(f"  Consistency strategy: {', '.join(sorted(consistency))}")

    # ── Stage 1: Characters ────────────────────────────────────────────────────
    char_names = None
    if args.fighters or args.reuse_fighters:
        char_names = resolve_fighters(client, args, out_dir)
    elif args.skip_characters:
        char_names = _load_local_profiles(out_dir, "character")
        if char_names:
            _log(f"  Skipping generation, using {len(char_names)} locally saved fighter(s): {', '.join(char_names)}")
            for n in char_names:
                _ensure_in_coderai(client, "character", n, out_dir)
        else:
            char_names = [f["name"] for f in FIGHTER_POOL if f.get("gender","male") == "male" or args.include_female]
            _log(f"  Skipping generation, assuming pool names: {', '.join(char_names)}")
    else:
        char_names = stage_characters(client, image_model, out_dir,
                                      region_filter=args.region,
                                      include_female=args.include_female)

    # ── Stage 2: Environments ──────────────────────────────────────────────────
    env_names = None
    if args.environments or args.reuse_environments:
        env_names = resolve_environments(client, args, out_dir)
    elif args.skip_environments:
        env_names = _load_local_profiles(out_dir, "environment")
        if env_names:
            _log(f"  Skipping generation, using {len(env_names)} locally saved environment(s): {', '.join(env_names)}")
            for n in env_names:
                _ensure_in_coderai(client, "environment", n, out_dir)
        else:
            env_names = [e["name"] for e in ENVIRONMENT_POOL]
            _log(f"  Skipping generation, assuming pool names: {', '.join(env_names)}")
    else:
        env_names = stage_environments(client, image_model, out_dir, region_filter=args.region)

    # ── Stage 2.5: LoRA training (image base model) — characters + environments ─
    lora_map = {}
    env_lora_map = {}
    if "lora" in consistency and not args.skip_videos and (char_names or []):
        lora_map = stage_loras(client, image_model, out_dir, char_names or [],
                               lora_steps=getattr(args, "lora_steps", 800),
                               lora_rank=getattr(args, "lora_rank", 16),
                               train_base_model=getattr(args, "lora_train_base_model", None) or None)
    if ("lora" in consistency and not args.skip_videos
            and not getattr(args, "no_env_loras", False) and (env_names or [])):
        env_lora_map = stage_env_loras(client, image_model, out_dir, env_names or [],
                                       lora_steps=getattr(args, "env_lora_steps", 800),
                                       lora_rank=getattr(args, "env_lora_rank", 16),
                                       train_base_model=getattr(args, "lora_train_base_model", None) or None)

    # ── Stage 3: Videos ────────────────────────────────────────────────────────
    if not args.skip_videos:
        stage_videos(
            client, video_model, out_dir,
            char_names=char_names or [],
            env_names=env_names or [],
            fps=args.fps,
            clip_delay=args.clip_delay,
            prompter=prompter,
            prompts_only=args.only_prompts,
            videos_only=args.only_videos,
            num_matches=args.matches,
            only_outcomes=args.only_outcomes,
            char_descriptions=char_descriptions,
            consistency=consistency, image_model=image_model,
            lora_map=lora_map,
            char_strength=getattr(args, "character_strength", 0.7),
            keyframe_steps=getattr(args, "keyframe_steps", 28),
            keyframe_size=getattr(args, "keyframe_size", "512x512"),
            lora_weight=getattr(args, "lora_weight", 0.85),
            env_lora_map=env_lora_map,
            env_lora_weight=getattr(args, "env_lora_weight", 0.8),
        )

    _log("\n✓ Done.")


if __name__ == "__main__":
    main()
