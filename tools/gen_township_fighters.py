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
import urllib.parse
from pathlib import Path
from typing import Optional

import requests

# Force line-buffered output so every print appears immediately even when
# stdout is piped to a file or another process.
sys.stdout.reconfigure(line_buffering=True)


def _log(*args, **kwargs):
    """print() that always flushes immediately."""
    print(*args, **kwargs, flush=True)


def _run_with_spinner(label: str, fn, *args, poll_fn=None, step_cb=None, **kwargs):
    """
    Run fn(*args, **kwargs) in a background thread while printing a live
    elapsed-time ticker on stdout so the user knows the script is alive.
    Returns the function's return value, or re-raises any exception it threw.

    ``poll_fn`` (optional): a zero-arg callable returning the server's live
    generation-progress dict (``current``/``total``/``pct``/``it_per_s``/``phase``)
    — e.g. ``client.video_progress``. When given, the spinner shows the real
    diffusion step ("step 12/25 (48%) 1.3it/s") and each sample is forwarded to
    ``step_cb(progress_dict)`` so a web caller can drive a per-clip step bar.
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
        step_txt = ""
        if poll_fn is not None:
            try:
                p = poll_fn() or {}
            except Exception:
                p = {}
            if p.get("active") and (p.get("total") or 0) > 0:
                cur, tot = int(p.get("current") or 0), int(p.get("total") or 0)
                pct = int(p.get("pct") or 0)
                its = p.get("it_per_s") or 0
                phase = p.get("phase") or ""
                step_txt = (f" · {phase} {cur}/{tot} ({pct}%)"
                            + (f" {its:.2f}it/s" if its else ""))
            elif p.get("phase") == "loading":
                step_txt = " · loading model…"
            if step_cb is not None:
                try:
                    step_cb(p)
                except Exception:
                    pass
        sys.stdout.write(f"\r    {spinner[idx % len(spinner)]} {label}  "
                         f"{elapsed:.0f}s{step_txt}…    ")
        sys.stdout.flush()
        idx += 1
        t.join(timeout=1.0)
    # Clear the spinner line
    sys.stdout.write("\r" + " " * 96 + "\r")
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

# Trailing style/motion cue appended to every fight-clip prompt. The motion words
# ("fast-paced, rapid explosive motion, dynamic action") push the I2V model toward
# more movement per clip — it tends to produce gentle, slow motion otherwise,
# especially when anchored to a keyframe. Kept here so the planner and the per-match
# Re-plan stay in sync.
FIGHT_PROMPT_SUFFIX = ("African township free fight, fast-paced, rapid explosive "
                       "motion, dynamic action, continuous forward motion that never "
                       "reverses, rewinds or loops back, cinematic, consistent "
                       "characters, wardrobe and setting")

# Pre-fight intro clips: the OPENING of a match — two solo fighter entrances
# followed by a stare-down where the referee signals the start. The ENTRANCE is
# explosive and threatening (the fighter charges in galvanized, shadow-boxing the
# air, ready to fight) but there is NO opponent and no blow LANDS; the FACE-OFF is
# a tense stare-down where the referee launches the bout, no strikes landed yet.
ENTRANCE_PROMPT_SUFFIX = ("African township fight night entrance, confident focused "
                          "athlete walking out, composed and determined, in character, "
                          "natural human body language, cinematic, consistent character, "
                          "wardrobe and setting, warming up alone with no opponent present")

FACEOFF_PROMPT_SUFFIX = ("African township fight night, cinematic, dramatic, intense "
                         "menacing stare-down, the referee signalling the START, "
                         "consistent characters, wardrobe and setting, tension before "
                         "the fight, no strikes landed yet")

# Back-compat alias (older saved data / external callers may reference it).
INTRO_PROMPT_SUFFIX = ENTRANCE_PROMPT_SUFFIX

ENTRANCE_SHOT_TEMPLATES = [
    "walking out into the arena rolling their shoulders and loosening their neck, calm and focused, giving the crowd a confident nod, slow low-angle tracking push-in through dust and spotlight haze",
    "striding in and shadow-boxing a few crisp jab-cross combinations to warm up, breathing steady and eyes locked ahead, smooth whip-pan circling the entrance",
    "bouncing lightly on their toes throwing a couple of relaxed test kicks into the air, settling into their stance with quiet determination, dynamic orbiting camera",
    "stepping over the barrier into the fighting ground, raising a fist to the cheering crowd with a composed smile, dramatic dropping crane shot",
    "walking in through smoke and fire-barrel glow, calmly cracking their knuckles and squaring their stance, focused expression, tight handheld push-in",
    "jogging in and throwing a few smooth combination drills, nodding to their corner with steady confidence, sweeping kinetic crane shot over the packed yard",
]

FACEOFF_SHOT_TEMPLATES = [
    "standing face to face foreheads almost touching in a tense menacing stare-down, the referee stepping between them and chopping a hand down to START the fight, crowd electric, slow orbit snapping to a push-in",
    "squaring off chest to chest trading furious words, the referee splitting them apart then sweeping an arm to signal FIGHT, dramatic low-angle shot, crowd roaring",
    "locked in a furious eye-to-eye stare-down with fists clenched, the referee raising a hand between them and dropping it to launch the bout, tight handheld push-in",
]


def _continuity_clause(env_name: str) -> str:
    """Deterministic wardrobe + environment continuity phrase appended to EVERY
    clip and outcome prompt of a match. Because each clip (and each chained
    sub-render within it) and every outcome carries the same clause, the whole
    match keeps the same outfits and location — the strongest lever for cross-clip
    consistency since it doesn't rely on the LLM remembering."""
    loc = (env_name or "").replace("_", " ").strip()
    where = f"the same {loc}" if loc else "the same township location"
    return ("each fighter keeps the EXACT SAME outfit, colours, hair and "
            f"accessories throughout, fighting in {where} with the same "
            "background, surfaces and crowd in every shot")

FIGHT_SHOT_TEMPLATES = [
    # Grounded, realistic human MMA — the majority of the rotation.
    "exploding forward with a hard low leg kick that buckles the opponent's lead leg, camera tracking the impact as the crowd reacts",
    "ripping a tight body-head hook combination that drops the opponent to a knee, sweat flying, orbiting camera",
    "snapping a crisp jab-cross-hook combo with sharp head movement, slipping the counter, fast push-in on the cleanest shots",
    "shooting in for a double-leg and driving the opponent down into the dirt, dust kicking up, camera dropping with the takedown",
    "raining down ground-and-pound from full mount, the opponent covering up and turtling, tight overhead close-up",
    "ripping a short elbow in the clinch that splits the brow, blood sheeting down, hard cut to the wound",
    "landing a thudding body kick that folds the opponent over, then a knee up the middle, low tracking angle",
    "uncorking a clean uppercut, the opponent's head snapping back, sweat spraying, super-slow impact frame",
    "catching a kick and crashing the opponent down into side control, crowd surging, sweeping pan to the mount",
    "cinching a tight guillotine as the opponent thrashes to escape, camera circling the struggle",
    "stuffing a takedown and spiking a knee to the body, opponent collapsing back, low tracking shot",
    "pummeling for underhooks in a grinding clinch against the fence, then a trip to the ground, tight handheld coverage",
    "stalking forward behind a stiff push kick that backs the opponent into the wall, wide establishing move",
    "trading punches at close range, both fighters connecting and bleeding, crowd roaring, shaky handheld close-up",
    "snapping a head kick that whips across the ducking opponent's skull, sweat flying, wide low-angle framing the arc",
    "slipping a wild swing and firing back a sharp counter cross down the middle, reverse-angle reveal of the counter",
    "working a takedown into half-guard and grinding out short elbows, gritty overhead and over-shoulder coverage",
    # Spectacular / acrobatic accents — used sparingly, the minority.
    "launching a spinning back-kick deep into the body, opponent folding over it, slow whip-pan following the spin",
    "detonating a flying knee flush on the jaw, sweat spraying, low-angle shot punching up at the action",
    "running two steps up a wall and springing off into a cartwheel kick, heel cracking across the opponent's jaw, sweeping crane shot",
    "back-flipping clear of a wild swing then landing into a spinning hook kick, orbiting low camera",
    "shoving the opponent crashing into a market stall, debris scattering, dust bursting up, wide scenic establishing shot",
    "kipping up off the ground straight into a spinning elbow, sweat spraying through the firelight, dynamic orbit",
]

# Rotating technique focus passed one-per-clip to the prompt writer so a match
# doesn't collapse into "all punches". The planner cycles through these (shuffled)
# so consecutive clips emphasise different MMA disciplines, ACROBATICS and camera
# language — the strongest lever against boxing-only, static-shot monotony.
FIGHT_ACTION_FOCUS = [
    "a realistic kicking exchange (low leg kicks, body kicks, head kicks, push kicks) — shoot it with a low tracking angle following the strikes",
    "knees and elbows in a tight clinch against the wall or fence — tight handheld close-up on the impacts",
    "a grounded takedown, slam or throw driving the fight to the mat — drop the camera with the slam",
    "ground-and-pound or a scramble on the floor — overhead and over-shoulder coverage",
    "a fight-ending submission attempt (choke, armbar, guillotine) and the escape — slow orbit around the lock",
    "blistering boxing combinations with head movement and counters — fast push-in on the cleanest shots",
    "a measured striking exchange at range, both fighters feinting, slipping and countering — clean side tracking shot",
    "dirty clinch work — pummeling for underhooks, short knees and a trip to the ground — tight handheld coverage",
    "a defensive sequence — slipping, blocking and ducking — then a sharp counter back to offence — reverse-angle reveal of the counter",
    "an explosive spinning or flying technique (spinning back-kick, spinning elbow, flying knee) — whip-pan that follows the rotation",
    "ONE acrobatic accent move — a cartwheel kick, backflip evasion or springboard — used sparingly, captured with a sweeping crane or orbit",
    "occasionally using the SCENERY — shoving the opponent into a fence, a car or a market stall — wide cinematic establishing shot",
]

# Bold MOVING-CAMERA directives. A static keyframe tends to lock the I2V camera,
# so for a subset of clips one of these is stated PROMINENTLY at the FRONT of the
# prompt (where the model weights it most) to force real camera movement through
# the environment instead of a fixed, locked-off frame.
CAMERA_MOVES = [
    "The camera sweeps in a fast 360-degree orbit around the fighters",
    "A sweeping crane shot rises and booms up high over the action",
    "The camera rushes in with a rapid dolly push-in toward the fighters",
    "A fast tracking shot races alongside the fighters as they move",
    "The camera whip-pans hard across the scene following the action",
    "A low Steadicam glides and weaves through the environment around the fighters",
    "The camera arcs around the fighters in a dramatic circular dolly move",
    "An aggressive handheld camera pushes, shakes and weaves through the brawl",
    "A drone-style camera swoops down through the environment toward the fighters",
    "The camera cranes down low then pulls back wide revealing the whole arena",
]

# The decisive FINISH that ends the bout — the first clip of every outcome video.
# Keyed by outcome; the victory celebration is in WIN_SHOT_TEMPLATES.
FINISH_SHOT_TEMPLATES = {
    "win": [
        "the final bell ringing as both fighters, exhausted and still on their feet, lower their fists — the bout going the full distance to a points decision, no knockout",
        "trading one last measured exchange as the horn sounds with both still standing and spent, the contest headed to the judges' scorecards",
        "the closing seconds ticking down, both fighters upright, bloodied and breathing hard, the fight decided on points rather than a finish",
    ],
    "ko_win": [
        "detonating one final brutal knockout blow, the opponent going stiff and crashing face-first to the ground",
        "catching the opponent flush with a head kick that drops them unconscious, body folding to the dirt",
        "finishing with a thunderous flying knee, the opponent collapsing in a heap as the crowd gasps",
    ],
    "retire": [
        "the beaten opponent turning their back and walking away toward their corner to quit, refusing to keep fighting",
        "the opponent waving it off, shaking their head and trudging out of the fight to retire as the winner stands tall",
        "the opponent raising a hand in surrender and walking away, done, unable to continue",
    ],
    "draw": [
        "both fighters trading their last savage blows to the final second, bloodied, exhausted, neither giving an inch",
        "a furious final exchange of hooks and knees, both men staggering but still swinging at the horn",
        "two spent warriors emptying the tank in a brutal last flurry as the bell ends an even war",
    ],
}

# The VICTORY / decision moment — the SECOND clip of every outcome video:
# the winning fighter and the referee raising their arm.
WIN_SHOT_TEMPLATES = {
    "win": [
        "the referee checking the scorecard then stepping in and raising the winner's arm as the points-decision victor, crowd surging",
        "the referee announcing the judges' decision and thrusting the winner's wrist to the sky as the points winner, sweat and blood glistening",
        "the referee lifting the winner's arm high after the points decision as the victor nods to the roaring crowd",
    ],
    "ko_win": [
        "the winner roaring with both fists raised as the referee lifts their arm in victory, crowd going wild",
        "the victor climbing the fence in triumph then dropping down as the referee raises their arm",
        "the winner pacing with arms thrown up before the referee grabs their wrist and hoists it skyward",
    ],
    "retire": [
        "the winner calmly raising a bloodied fist as the referee lifts their arm and the crowd applauds",
        "the victor bowing to the crowd before the referee hoists their arm high in triumph",
        "the winner standing tall as the referee grabs their wrist and thrusts their arm to the sky",
    ],
    "draw": [
        "all THREE in frame — both bloodied fighters side by side with the referee standing between them, the referee gripping a wrist of each and thrusting BOTH fighters' fists high to the sky at once, declaring a draw",
        "both exhausted fighters and the referee together in shot, the referee raising BOTH fighters' arms simultaneously to signal a draw, the two warriors spent, the crowd applauding the war",
        "the referee between the two fighters lifting BOTH their fists overhead at the same time as the crowd roars, all three figures clearly in frame, an honourable draw",
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
        "finish_shot_templates": {k: list(m.FINISH_SHOT_TEMPLATES.get(k, []))
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
    fnt = cfg.get("finish_shot_templates")
    if isinstance(fnt, dict):
        merged = {k: list(v) for k, v in m.FINISH_SHOT_TEMPLATES.items()}
        for k in _PROMPT_OUTCOMES:
            if isinstance(fnt.get(k), list):
                cleaned = [str(s).strip() for s in fnt[k] if str(s).strip()]
                if cleaned:
                    merged[k] = cleaned
        m.FINISH_SHOT_TEMPLATES = merged


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

    def _post(self, path: str, body: dict, timeout: int = None) -> dict:
        r = self.session.post(f"{self.base}{path}", json=body,
                              timeout=timeout if timeout is not None else self.timeout)
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
                           model: str, n: int = 4, size: str = "512x512",
                           steps: int = None) -> dict:
        body = {
            "name": name, "prompt": prompt, "description": description,
            "model": model, "n": n,
            "width": int(size.split("x")[0]), "height": int(size.split("x")[1]),
        }
        if steps:
            body["steps"] = int(steps)
        return self._post("/v1/characters/generate", body)

    def generate_environment(self, name: str, prompt: str, description: str,
                             model: str, n: int = 3, size: str = "768x512",
                             steps: int = None) -> dict:
        body = {
            "name": name, "prompt": prompt, "description": description,
            "model": model, "n": n,
            "width": int(size.split("x")[0]), "height": int(size.split("x")[1]),
        }
        if steps:
            body["steps"] = int(steps)
        return self._post("/v1/environments/generate", body)

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
                   resolution: int = 512, train_base_model: str = None,
                   target: str = "image", quantize_4bit: bool = True,
                   wait: bool = True, session: str = None) -> dict:
        """Train a per-character or per-environment LoRA on the server.
        Blocks until complete.

        target="image" trains an SD1.x/SDXL UNet LoRA (for keyframes); target=
        "video" trains a Wan video-DiT LoRA against `base_model` (the video model),
        so it loads directly on the video pipeline."""
        body = {"name": name, "base_model": base_model,
                "steps": int(steps), "rank": int(rank), "resolution": int(resolution),
                "target": target, "wait": bool(wait)}
        if session:
            body["session"] = session
        if target == "video":
            body["quantize_4bit"] = bool(quantize_4bit)
        if train_base_model:
            body["train_base_model"] = train_base_model
        if character:
            body["character"] = character
        if environment:
            body["environment"] = environment
        if images:
            body["images"] = images
        # Video DiT training (e.g. Wan A14B) can take many hours including the
        # one-off model load; image LoRA is quicker. Allow a long ceiling so the
        # blocking POST doesn't read-timeout while the server is still training.
        train_timeout = 24 * 3600 if target == "video" else 4 * 3600
        return self._post("/v1/loras/train", body, timeout=train_timeout)

    def list_loras(self) -> list:
        try:
            return self._get("/v1/loras").get("loras", [])
        except Exception:
            return []

    def lora_progress(self, job: str = None, session: str = None) -> dict:
        try:
            q = ""
            if job:
                q = f"?job={urllib.parse.quote(job)}"
            elif session:
                q = f"?session={urllib.parse.quote(session)}"
            return self._get(f"/v1/loras/progress{q}")
        except Exception:
            return {}

    def image_progress(self) -> dict:
        """Live diffusion-step progress of the in-flight image generation
        (keyframes / character / environment refs). Best-effort; {} on error."""
        try:
            return self._get("/v1/images/progress") or {}
        except Exception:
            return {}

    def video_progress(self) -> dict:
        """Live diffusion-step progress of the in-flight video clip. {} on error."""
        try:
            return self._get("/v1/video/progress") or {}
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
        """POST to /v1/video/upscale and return the upscaled video bytes.

        Requests `response_format=url` so the (large, super-resolved) result is
        saved server-side and streamed back over a plain HTTP GET instead of being
        base64-inflated into the JSON body. The server falls back to b64 when it
        has no file path configured; _video_bytes handles both."""
        body = {
            "video": "data:video/mp4;base64," + base64.b64encode(video_bytes).decode(),
            "upscale_factor": factor,
            "response_format": "url",
        }
        if model:
            body["model"] = model
        return self._video_bytes(self._post("/v1/video/upscale", body))

    def interpolate_video(self, video_bytes: bytes, fps_multiplier: int = 2,
                          model: str = None, output_fps: int = None) -> bytes:
        """POST to /v1/video/interpolate and return the interpolated video bytes.

        Uses `response_format=url` (streamed GET) for the large result; see
        upscale_video. Falls back to b64 transparently. `output_fps` overrides the
        final encode fps (None = source × multiplier, preserves duration)."""
        body = {
            "video": "data:video/mp4;base64," + base64.b64encode(video_bytes).decode(),
            "fps_multiplier": fps_multiplier,
            "response_format": "url",
        }
        if model:
            body["model"] = model
        if output_fps:
            body["output_fps"] = int(output_fps)
        return self._video_bytes(self._post("/v1/video/interpolate", body))

    def generate_video_clip(self, prompt: str, model: str,
                            character_profiles: list = None,
                            environment_name: str = None,
                            num_frames: int = 49, fps: int = 8,
                            width: int = 832, height: int = 480,
                            seed: int = None,
                            init_image: bytes = None,
                            loras: list = None,
                            cond_frames: list = None) -> bytes:
        # `cond_frames` (a list of PNG byte tails from the previous chained part)
        # drives VACE 'extend' continuation: the model sees real motion and carries
        # it FORWARD, fixing the forward-then-rewind boomerang of single-frame
        # seeding. Otherwise, a keyframe image drives text+image→video (ti2v).
        if cond_frames:
            mode = "extend"
        elif init_image:
            mode = "ti2v"
        else:
            mode = "t2v"
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
        if cond_frames:
            body["cond_frames"] = [
                "data:image/png;base64," + base64.b64encode(f).decode()
                for f in cond_frames
            ]
        if loras:
            body["loras"] = loras

        d = self._post("/v1/video/generations", body)
        return self._video_bytes(d)

    def _video_bytes(self, d: dict) -> bytes:
        """Extract mp4 bytes from a /v1/video/* response (b64 or URL form)."""
        item = (d.get("data") or [{}])[0]
        raw = item.get("b64_mp4") or item.get("url", "")
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        if raw.startswith("http"):
            import urllib.request
            with urllib.request.urlopen(raw, timeout=600) as resp:
                return resp.read()
        return base64.b64decode(raw)


# ─────────────────────────────────────────────────────────────────────────────
# LLM prompt generation
# ─────────────────────────────────────────────────────────────────────────────

_LLM_SYSTEM = """\
You are a creative director writing vivid, BRUTAL video-generation prompts for African street fighting scenes.
Each prompt must be ONE sentence, 18-38 words, cinematic, specific and full of action.
Emphasize FAST, continuous, explosive, savage motion — describe action mid-movement, never static, posed, or \
slow-motion. The motion must PROGRESS FORWARD in one direction through the clip: no reversing, no \
rewinding, no looping back to the starting pose, no boomerang or back-and-forth motion.
This is a full MMA / no-rules fight between two REAL HUMAN athletes — grounded, believable and brutal, \
NOT superhero or monster choreography. The MAJORITY of clips should be normal, realistic MMA exchanges: \
boxing combinations with head movement, jabs, crosses, hooks and uppercuts, low leg kicks, body kicks, \
head kicks, push kicks, knees in the clinch, elbows, takedowns and slams, sprawls, ground-and-pound, \
mount and guard scrambles, clinch work against the wall, chokes and submission attempts. Do NOT default \
to only punches — vary the technique across clips, and keep the strikes physically plausible for trained \
human fighters. Each prompt should center on a DIFFERENT technique than the recent ones.
OCCASIONALLY (a MINORITY of clips, roughly one in three or four — never most of them) make a shot more \
spectacular and acrobatic for variety: a spinning back-kick, spinning elbow, flying knee, a cartwheel or \
tornado kick, a backflip evasion, kip-up or a flashy aerial takedown. Use these as accents, not the norm. \
Most of the fight stays on the ground as a hard, technical human brawl.
The township environment can OCCASIONALLY come into play — a fighter shoved into a fence or wall, dust \
kicking up, the pressing crowd — and now and then (rarely) a bigger scenic moment like crashing into a \
market stall or going over a parked car. Keep scenery use believable and sparing, not a constant \
parkour playground.
Make it gritty, visceral and BRUTAL: it is encouraged to OFTEN (but not every clip) show the damage — a \
bloodied nose or lip, a split brow, blood spray, sweat-and-blood on the face, a stagger or knockdown — \
when a hard blow lands.
ALWAYS specify CAMERA WORK in the prompt: describe a kinetic camera move and angle — a low tracking shot, \
a fast whip-pan following a spin, a sweeping crane/orbit around an aerial move, a tight handheld close-up \
shaking on impact, an overhead ground shot, an over-the-shoulder angle, a wide scenic establishing move, \
or a dramatic push-in on the cleanest strike. The camera should feel kinetic and dynamic, never locked off.
Make every shot feel SCENIC and cinematic — rich depth, dramatic lighting and atmosphere (dust, smoke, \
sparks, swirling crowd). Vary lighting (dusk, generator light, noon sun, harsh spotlight, headlights, \
fire-barrel glow).
Always refer to each fighter by their NAME (given in the user message), not only by their description.
WARDROBE CONTINUITY (critical): every clip of a match — and every chained part within a clip, plus the \
outcome clips — must show each fighter in the IDENTICAL outfit: the same garments, exact same colours, \
same hair and accessories given in their description. Restate the key clothing details so they stay \
constant; NEVER change, add, remove, or restyle clothing, and never switch a fighter to different \
shorts/gloves/colours between shots.
ENVIRONMENT CONTINUITY (critical): stay in the ONE given location for the whole match — the same \
ring/yard/street, the same surfaces, walls, structures, lighting and crowd. Describe the same place \
every time; never move the fight to a different setting between clips or outcomes.
Do NOT use generic phrases like "high quality" or "realistic". Return ONLY the prompt, no quotes."""

_LLM_OUTCOME_SYSTEM = """\
You are a creative director writing vivid 18-30 word video-generation prompts for the END of a brutal fight.
An outcome video is made of TWO consecutive shots and you write ONE of them at a time (the user tells you which):
  • the FINISH — the last decisive action that ENDS the fight: the knockout blow landing and the loser \
crashing down, the fight-ending submission, the final overwhelming flurry, or the beaten fighter / their \
corner signalling they cannot continue. Describe FORWARD, explosive, savage motion, with dynamic camera \
work (low angle, whip-pan, handheld push-in) — this is the climax, make it hit hard.
  • the VICTORY — immediately after: the WINNING fighter and the REFEREE raising the winner's arm in \
victory (for a draw, the referee raises BOTH fighters' arms). Show triumph, exhaustion, the roaring crowd, \
and the referee's raised-arm gesture explicitly, with a celebratory camera move (push-in, crane, orbit).
Be specific about body language, expression, lighting, atmosphere and CAMERA movement.
Refer to fighters by their NAME (given in the user message).
WARDROBE + LOCATION CONTINUITY (critical): the outcome happens in the SAME match — keep each fighter in \
the IDENTICAL outfit (same garments, exact colours, hair, accessories) described, and in the SAME \
location as the fight, with consistent background, surfaces and lighting. Never change clothing or move \
to a different place.
Return ONLY the prompt, no quotes or explanation."""

_LLM_INTRO_SYSTEM = """\
You are a creative director writing vivid 18-32 word video-generation prompts for the DRAMATIC OPENING of an African township fight night — BEFORE any fighting starts.
You write ONE of these shot types at a time (the user says which):
  • an ENTRANCE — ONE fighter ALONE walking out into the arena: a confident, composed, in-character entrance that suits THAT fighter's personality and description. They are a real human athlete, not a monster — keep it grounded and believable: walking or jogging in, loosening up, rolling the shoulders, a few crisp warm-up shadow-boxing punches or test kicks AT THE AIR, bouncing lightly on the toes, a focused or determined look, a confident nod or raised fist to the crowd. Show quiet confidence and readiness, NOT cartoonish rage — avoid roaring, snarling, baring teeth, chest-pounding and savage glaring. There is NO opponent in frame and no blow LANDS on anyone. Use a dynamic camera move (low-angle tracking push-in, whip-pan, sweeping crane, orbit) and rich atmosphere (dust, smoke, fire-barrel glow, spotlights).
  • a FACE-OFF — BOTH fighters squaring off face to face in a tense menacing stare-down, with the REFEREE between them giving the gesture to START the fight (chopping or dropping a hand, sweeping an arm). Show the tension and the referee's start signal explicitly. Still NO punches or kicks thrown yet.
Always refer to each fighter (and the referee) by their NAME given in the user message.
WARDROBE + LOCATION CONTINUITY (critical): keep each fighter in the IDENTICAL outfit (same garments, exact colours, hair, accessories) described, in the SAME township location with consistent background, surfaces, lighting and crowd.
Do NOT use generic phrases like "high quality" or "realistic". Return ONLY the prompt, no quotes."""

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
        self._used_intro: dict[str, list[str]] = {}

    def fight_shot(self, f1: str, f2: str, env_desc: str, match_context: str = "",
                   avoid: list = None, action_focus: str = "") -> str:
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
                    focus_hint = (f"\nThis clip should focus on {action_focus}."
                                  if action_focus else "")
                    _d1 = self.char_descriptions.get(f1, "")
                    _d2 = self.char_descriptions.get(f2, "")
                    f1_desc = f"{f1} ({_d1})" if _d1 else f1
                    f2_desc = f"{f2} ({_d2})" if _d2 else f2
                    prompt = self.client.chat_complete(
                        model=self.model,
                        system=_LLM_SYSTEM,
                        user=(
                            f"Fighter 1: {f1_desc}. Fighter 2: {f2_desc}. "
                            f"Location: {env_desc}. "
                            f"{match_context}{used_hint}{focus_hint}\n"
                            "Write one fight action shot prompt. Refer to each "
                            "fighter by their NAME (not just their description)."
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

    def outcome_shot(self, fighter: str, outcome: str, env_desc: str,
                     role: str = "finish", opponent: str = None) -> str:
        """Generate a unique outcome shot prompt for one clip of an outcome video.

        `role` selects which of the two outcome clips this is:
          • "finish" / "final_exchange" → the decisive fight-ending action.
          • "victory" / "draw_decision" → the winner + referee raising the arm.
        `opponent` is the other fighter's name, woven into the description so the
        finish/celebration reads as a real two-fighter moment.
        """
        is_finish = role in ("finish", "final_exchange")
        src = FINISH_SHOT_TEMPLATES if is_finish else WIN_SHOT_TEMPLATES
        templates = src.get(outcome, src["win"])
        used = self._used_outcome.setdefault(f"{outcome}:{role}", [])
        opp = opponent or "the opponent"

        if self.model:
            finish_labels = {
                "win": f"the FINISH (a POINTS decision, NOT a knockout) — the final bell rings with both {fighter} and {opp} still on their feet, exhausted and spent, the bout having gone the full distance to be decided on the judges' scorecards",
                "ko_win": f"the FINISH — {fighter} landing the brutal knockout blow that drops {opp} unconscious to the ground",
                "retire": f"the FINISH — {opp} backing away, turning their back and walking off to retire, refusing to continue, while {fighter} stands tall",
                "draw": f"the FINISH — a final all-out exchange where {fighter} and {opp} trade brutal blows to the last second",
            }
            # Victory clip features ONLY the winner + referee (no opponent) — except
            # a draw, which raises both fighters' arms.
            victory_labels = {
                "win": f"the VICTORY (a POINTS decision) — the referee checks the scorecard then raises {fighter}'s arm as the points-decision winner, {fighter} alone in frame, crowd roaring",
                "ko_win": f"the VICTORY — {fighter} celebrating with fists raised as the referee raises {fighter}'s arm, crowd going wild",
                "retire": f"the VICTORY — the referee raising {fighter}'s arm in victory as {fighter} salutes the crowd",
                "draw": f"the VICTORY (a DRAW) — ALL THREE in frame: {fighter} and {opp} side by side with the REFEREE standing between them, the referee gripping a wrist of each and thrusting BOTH {fighter}'s and {opp}'s fists high to the sky at once to declare the draw",
            }
            labels = finish_labels if is_finish else victory_labels
            _avoid = used[-2:]
            for attempt in range(2):
                try:
                    used_hint = f" Avoid: {'; '.join(_avoid)}." if _avoid else ""
                    _df = self.char_descriptions.get(fighter, "")
                    f_desc = f"{fighter} ({_df})" if _df else fighter
                    _do = self.char_descriptions.get(opponent or "", "")
                    o_desc = f"{opponent} ({_do})" if (opponent and _do) else (opponent or "")
                    # The victory clip features only the winner (except a draw), so
                    # don't feed the opponent into a non-draw victory shot.
                    _include_opp = is_finish or outcome == "draw"
                    opp_line = (f"Opponent: {o_desc}. " if (o_desc and _include_opp) else "")
                    prompt = self.client.chat_complete(
                        model=self.model,
                        system=_LLM_OUTCOME_SYSTEM,
                        user=(
                            f"Fighter: {f_desc}. {opp_line}Shot to write: {labels.get(outcome, outcome)}. "
                            f"Location: {env_desc}.{used_hint} Write this ONE outcome shot prompt. "
                            "Refer to the fighters by their NAME and include dynamic camera work."
                        ),
                        max_tokens=110,
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

    def intro_shot(self, role: str, f1: str, f2: str = None, env_desc: str = "",
                   referee: str = None) -> str:
        """Generate a pre-fight INTRO shot prompt for one opening clip of a match.

        `role`:
          • "entrance" → ONE fighter (f1) alone making a bold, scenic solo entrance.
          • "faceoff"  → f1 vs f2 stare-down with the referee signalling START.
        No fighting happens in these clips — they set the scene before the bout.
        """
        is_face = role == "faceoff"
        src = FACEOFF_SHOT_TEMPLATES if is_face else ENTRANCE_SHOT_TEMPLATES
        used = self._used_intro.setdefault(role, [])

        if self.model:
            if is_face:
                _d1 = self.char_descriptions.get(f1, "")
                _d2 = self.char_descriptions.get(f2, "")
                f1d = f"{f1} ({_d1})" if _d1 else f1
                f2d = f"{f2} ({_d2})" if _d2 else f2
                ref_line = f" Referee: {referee}." if referee else " A referee."
                label = (f"a FACE-OFF stare-down between {f1d} and {f2d} with the "
                         "referee giving the signal to START the fight")
                user_extra = f"Fighter 1: {f1d}. Fighter 2: {f2d}.{ref_line} "
            else:
                _d1 = self.char_descriptions.get(f1, "")
                f1d = f"{f1} ({_d1})" if _d1 else f1
                label = f"a bold, dramatic, scenic solo ENTRANCE of {f1d}"
                user_extra = f"Fighter: {f1d}. "
            _avoid = used[-2:]
            for attempt in range(2):
                try:
                    used_hint = f" Avoid: {'; '.join(_avoid)}." if _avoid else ""
                    prompt = self.client.chat_complete(
                        model=self.model,
                        system=_LLM_INTRO_SYSTEM,
                        user=(
                            f"{user_extra}Shot to write: {label}. Location: {env_desc}."
                            f"{used_hint} Write this ONE intro shot prompt. Refer to the "
                            "fighters (and referee) by their NAME."
                        ),
                        max_tokens=110,
                    ).strip()
                    if len(prompt) < 8:
                        raise ValueError(f"LLM returned too-short response: {prompt!r}")
                    used.append(prompt[:60])
                    return prompt
                except Exception as e:
                    if attempt == 0:
                        _avoid = []  # drop the hint and retry once
                        continue
                    _log(f"    (LLM intro prompt failed after retry: {e} — using template)")

        available = [t for t in src if t not in used]
        if not available:
            available = src
        choice = random.choice(available)
        used.append(choice)
        return choice


# ─────────────────────────────────────────────────────────────────────────────
# Form auto-generation — invent a fresh character / referee / environment (or a
# match pairing) with the text model so the create forms can be one-click filled
# and then reviewed/edited before submitting.
# ─────────────────────────────────────────────────────────────────────────────

_AUTOGEN_CHAR_SYS = (
    "You invent original characters for an African-township street-fighting video "
    "generator. Reply with ONE compact JSON object and nothing else — no prose, no "
    "code fence.")
_AUTOGEN_ENV_SYS = (
    "You invent original locations for an African-township street-fighting video "
    "generator. Reply with ONE compact JSON object and nothing else — no prose, no "
    "code fence.")

_NAME_SYL = ["kha", "lo", "ndlo", "vu", "dla", "mini", "nko", "si", "mbe", "ki",
             "zu", "lu", "tha", "bo", "jek", "sa", "nta", "ba", "we", "do", "ma",
             "se", "thi", "za", "nge", "qa"]


def _gen_name_token() -> str:
    return "".join(random.choice(_NAME_SYL) for _ in range(random.randint(2, 3)))


def _slugify_name(s: str) -> str:
    import re as _re
    s = (s or "").strip().lower().replace(" ", "_")
    s = _re.sub(r"[^a-z0-9_]+", "_", s).strip("_")
    return s


def _unique_name(base: str, taken: set) -> str:
    base = base or "new"
    if base not in taken:
        return base
    i = 2
    while f"{base}_{i}" in taken:
        i += 1
    return f"{base}_{i}"


def _extract_json_obj(text: str) -> dict:
    """Best-effort parse of a JSON object from an LLM reply (handles ``` fences and
    surrounding prose)."""
    import re as _re
    if not text:
        return {}
    t = text.strip()
    t = _re.sub(r"^```(?:json)?\s*", "", t)
    t = _re.sub(r"\s*```$", "", t)
    m = _re.search(r"\{.*\}", t, _re.DOTALL)
    if m:
        t = m.group(0)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _autogen_profile_payload(client, text_model, kind: str, role: str,
                             existing_names) -> dict:
    """Invent a fresh profile's form fields. Uses the text model when available,
    falling back to template synthesis so the button always returns something
    editable. Returns {name, region, gender?, description, prompt, role?}."""
    role = (role or "fighter").strip().lower()
    taken = set(existing_names or [])
    avoid = ", ".join(sorted(taken)[:40]) or "(none yet)"
    payload = {}
    if client and text_model:
        try:
            if kind == "environment":
                user = (
                    "Invent a NEW gritty African-township fight LOCATION. "
                    f"Do NOT reuse these names: {avoid}. JSON keys: "
                    "name (short lowercase slug, words joined by _), "
                    "region (a township or city), "
                    "description (one vivid sentence on the place + mood + light), "
                    "prompt (a detailed image-generation prompt for the empty "
                    "location, cinematic, realistic, gritty). Return ONLY the JSON.")
                payload = _extract_json_obj(client.chat_complete(
                    model=text_model, system=_AUTOGEN_ENV_SYS, user=user,
                    max_tokens=320))
            elif role == "referee":
                user = (
                    "Invent a NEW fight REFEREE / official (NOT a fighter). "
                    f"Do NOT reuse these names: {avoid}. JSON keys: "
                    "name (short lowercase slug), region (a township or city), "
                    "gender (male/female), "
                    "description (one sentence: build, age, demeanour — an official, "
                    "not a competitor), "
                    "prompt (a detailed portrait image-generation prompt of the "
                    "referee wearing a black-and-white striped referee shirt, "
                    "authoritative, realistic; NOT in fight kit). Return ONLY JSON.")
                payload = _extract_json_obj(client.chat_complete(
                    model=text_model, system=_AUTOGEN_CHAR_SYS, user=user,
                    max_tokens=320))
            else:
                user = (
                    "Invent a NEW African-township street FIGHTER. "
                    f"Do NOT reuse these names: {avoid}. JSON keys: "
                    "name (short lowercase surname-like slug), "
                    "region (a township or city), gender (male/female), "
                    "description (one vivid sentence: weight class, build, hair, a "
                    "distinguishing mark), "
                    "prompt (a detailed portrait image-generation prompt ending with "
                    "'African township fighter', realistic, dramatic lighting). "
                    "Return ONLY the JSON.")
                payload = _extract_json_obj(client.chat_complete(
                    model=text_model, system=_AUTOGEN_CHAR_SYS, user=user,
                    max_tokens=320))
        except Exception:
            payload = {}

    out = {}
    # Name — slugified, made unique against what already exists.
    _nm = _slugify_name(payload.get("name") or "")
    if not _nm:
        if kind == "environment":
            _nm = f"{_gen_name_token()}_spot"
        elif role == "referee":
            _nm = f"ref_{_gen_name_token()}"
        else:
            _nm = _gen_name_token()
    out["name"] = _unique_name(_nm, taken)

    if kind == "environment":
        _src = random.choice(ENVIRONMENT_POOL)
        out["region"] = str(payload.get("region") or _src["region"]).strip()
        out["description"] = str(payload.get("description") or _src["description"]).strip()
        out["prompt"] = str(payload.get("prompt") or _src["prompt"]).strip()
    else:
        _src = random.choice(FIGHTER_POOL)
        out["region"] = str(payload.get("region") or _src["region"]).strip()
        out["gender"] = str(payload.get("gender") or _src.get("gender", "male")).strip() or "male"
        if role == "referee":
            out["role"] = "referee"
            out["description"] = str(
                payload.get("description")
                or "Veteran fight referee, calm and authoritative.").strip()
            out["prompt"] = str(
                payload.get("prompt")
                or (f"Portrait of an African fight referee wearing {REFEREE_WARDROBE}, "
                    "authoritative stance, realistic, dramatic lighting, NOT a fighter")
            ).strip()
        else:
            out["description"] = str(payload.get("description") or _src["description"]).strip()
            out["prompt"] = str(payload.get("prompt") or _src["prompt"]).strip()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Video utilities
# ─────────────────────────────────────────────────────────────────────────────

def concat_videos(clip_paths: list, out_path: str, reencode: bool = False,
                  fps: int = None):
    """Concatenate clips with the ffmpeg concat demuxer.

    ``reencode=False`` stream-copies (fast) — fine for the final short/long
    assembly where each segment is a separate shot and a cut between them is
    expected.

    ``reencode=True`` RE-ENCODES with a constant frame rate. Use it to join the
    sub-renders of ONE chained shot: stream-copying mp4s that carry B-frames /
    an edit list / reset PTS makes players FREEZE on a segment's first frame for
    its whole duration (the "first half is a static image" bug). Re-encoding
    regenerates clean, monotonic, constant-rate timestamps so the join is truly
    seamless. Falls back to stream copy if the encoder is unavailable."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
        list_path = f.name
    try:
        if reencode:
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                   "-pix_fmt", "yuv420p", "-vsync", "cfr"]
            if fps:
                cmd += ["-r", str(int(fps))]
            cmd += ["-movflags", "+faststart", out_path]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                return
            except subprocess.CalledProcessError as e:
                _log(f"    ⚠ seamless re-encode concat failed ({e.stderr.decode(errors='replace')[:120] if e.stderr else e}); "
                     f"falling back to stream copy")
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


def _split_frame_budget(total: int, chunk_max: int) -> list:
    """Split a clip's total frame budget into chained sub-renders, each
    ≤ chunk_max, distributed as EVENLY as possible (so there's no tiny trailing
    part). Returns [total] when it already fits in one render."""
    total = max(1, int(total))
    chunk_max = max(1, int(chunk_max))
    if total <= chunk_max:
        return [total]
    import math as _math
    n = _math.ceil(total / chunk_max)
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def _last_frame_png(mp4_path: str) -> Optional[bytes]:
    """Extract the final frame of a clip as PNG bytes (used to seed the next
    chained sub-render so the join is seamless). Returns None on failure."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp = tf.name
        # -sseof -1 reads the last ~1s; -update 1 keeps overwriting so the file
        # left on disk is the LAST decoded frame — exact and cheap for short clips.
        subprocess.run(
            ["ffmpeg", "-y", "-sseof", "-1", "-i", mp4_path,
             "-update", "1", "-q:v", "2", tmp],
            check=True, capture_output=True,
        )
        data = Path(tmp).read_bytes()
        return data or None
    except Exception:
        return None
    finally:
        if tmp:
            try: os.unlink(tmp)
            except Exception: pass


def _last_frames_png(mp4_path: str, k: int) -> list:
    """Extract the LAST `k` frames of a clip as a list of PNG bytes, in order
    (oldest → newest). Used to seed a VACE 'extend' continuation with real motion
    so the join carries velocity forward instead of boomeranging. Returns [] on
    failure (caller falls back to single-frame seeding)."""
    if k <= 1:
        one = _last_frame_png(mp4_path)
        return [one] if one else []
    tmpd = None
    try:
        tmpd = tempfile.mkdtemp(prefix="twtail_")
        # Pull the last ~2s, write each decoded frame, then keep the final k.
        subprocess.run(
            ["ffmpeg", "-y", "-sseof", "-2", "-i", mp4_path,
             "-q:v", "2", os.path.join(tmpd, "f_%04d.png")],
            check=True, capture_output=True,
        )
        files = sorted(Path(tmpd).glob("f_*.png"))
        files = files[-k:]
        frames = [p.read_bytes() for p in files]
        return [f for f in frames if f]
    except Exception:
        return []
    finally:
        if tmpd:
            import shutil as _sh
            _sh.rmtree(tmpd, ignore_errors=True)


# Wan2.2-A14B is trained for clips up to ~81 frames; beyond that temporal
# coherence breaks down (the frames visibly "jump") in a SINGLE generation.
MODEL_MAX_FRAMES = 81

# A planned clip can be LONGER than one model call: when it exceeds the
# single-render cap (SINGLE_CLIP_MAX_FRAMES) it is rendered as several chained
# sub-renders (each ≤ the cap) and concatenated into one continuous shot — the
# last frame of each part seeds the next. So the planned clip length may run up
# to MAX_PLANNED_FRAMES even though no single model call exceeds the cap.
SINGLE_CLIP_MAX_FRAMES = 50      # max frames in ONE model generation (≤ MODEL_MAX_FRAMES)
MAX_PLANNED_FRAMES = 480         # ceiling for a whole (possibly chained) clip

# Number of trailing frames of the previous chained part fed to a VACE model as
# 'extend' conditioning. More frames = stronger motion-continuity (kills the
# forward/rewind boomerang) but more conditioning cost; ~5 carries velocity well.
VACE_TAIL_FRAMES = 5

# Per fight-clip frame budget. Frame count (not seconds) is the real control: it's
# the model's motion budget and is fps-independent, so a clip is CLIP_*_FRAMES
# frames played at the chosen fps → duration = frames / fps (e.g. 50-70 frames at
# 16 fps ≈ 3.1-4.4 s).
CLIP_MIN_FRAMES = 50
CLIP_MAX_FRAMES = 70


def _clip_frame_range(lo, hi):
    """Normalize a (min, max) fight-clip frame budget from config: clamp to
    [8, MAX_PLANNED_FRAMES] and ensure lo <= hi, so bad UI input can't break the
    planner (random.randint requires lo <= hi). A clip longer than one model call
    is split + chained at render time, so the ceiling is MAX_PLANNED_FRAMES, not
    the per-render MODEL_MAX_FRAMES."""
    try:
        lo = int(lo); hi = int(hi)
    except (TypeError, ValueError):
        lo, hi = CLIP_MIN_FRAMES, CLIP_MAX_FRAMES
    lo = max(8, min(lo, MAX_PLANNED_FRAMES))
    hi = max(8, min(hi, MAX_PLANNED_FRAMES))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


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
                     include_female: bool = False,
                     max_count: int = 0, n_refs: int = 4) -> list:
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
    # Cap how many fighters to generate (0 = whole filtered pool).
    if max_count and max_count > 0 and len(pool) > max_count:
        _log(f"  Limiting to first {max_count} of {len(pool)} pooled fighter(s)")
        pool = pool[:max_count]

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
                description=fighter["description"], model=image_model,
                n=max(1, int(n_refs or 4)),
                poll_fn=client.image_progress,
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
                       region_filter: Optional[str] = None,
                       max_count: int = 0, n_refs: int = 3) -> list:
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
    # Cap how many environments to generate (0 = whole filtered pool).
    if max_count and max_count > 0 and len(pool) > max_count:
        _log(f"  Limiting to first {max_count} of {len(pool)} pooled environment(s)")
        pool = pool[:max_count]

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
                n=max(1, int(n_refs or 3)), size="768x512",
                poll_fn=client.image_progress,
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


# Clothing nouns to lift the fighter's fixed outfit out of their profile prompt,
# so every keyframe can state it explicitly (the LoRA/IP-Adapter alone drift).
_CLOTHING_WORDS = (
    "shorts", "trunks", "singlet", "vest", "gloves", "wraps", "hand wraps",
    "sports bra", "bra", "tank top", "tank", "t-shirt", "shirt", "hoodie",
    "boots", "headgear", "headband", "bandana", "jersey", "tracksuit", "kit",
    "trousers", "pants", "leggings", "belt", "gi", "kimono", "sneakers",
)


def _extract_outfit(prompt_text: str) -> str:
    """Pull the wardrobe phrase(s) out of a profile prompt (e.g. 'worn boxing
    shorts', 'boxing singlet', 'sports bra and MMA shorts'). Returns '' when none
    are found. Comma/semicolon/period clauses that mention a clothing noun are
    kept (up to two), so the colour/material adjectives stay attached."""
    import re as _re
    out, seen = [], set()
    for part in _re.split(r"[,.;]", prompt_text or ""):
        p = part.strip()
        pl = p.lower()
        if not p:
            continue
        if any(w in pl for w in _CLOTHING_WORDS):
            key = pl
            if key not in seen:
                seen.add(key)
                out.append(p)
    return ", ".join(out[:2])


def _build_char_outfits(out_dir: Path) -> dict:
    """Return {name: outfit_phrase} from each fighter's profile PROMPT (the
    `prompt` field carries clothing; `description` usually doesn't). Merges
    FIGHTER_POOL + locally saved meta.json so user-created fighters are covered."""
    outfits = {}
    for f in FIGHTER_POOL:
        o = _extract_outfit(f.get("prompt", ""))
        if o:
            outfits[f["name"]] = o
    chars_dir = out_dir / "characters"
    if chars_dir.exists():
        for d in chars_dir.iterdir():
            meta_path = d / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    o = _extract_outfit(meta.get("prompt", "") or meta.get("description", ""))
                    if o:
                        outfits[d.name] = o
                except Exception:
                    pass
    return outfits


# Fighter profiles name a garment but rarely a COLOUR ("boxing singlet", "MMA
# shorts"), and some name no recognisable garment at all ("free fighting attire").
# Without an explicit, fixed colour the image model invents a new one for every
# keyframe, so a fighter's kit changes shot to shot. We therefore lock a
# deterministic colour per fighter (stable hash of the name) and a default
# garment, persisted to wardrobe.json so it stays identical across a whole match
# AND is editable by the user.
_OUTFIT_COLORS = [
    "crimson red", "royal blue", "emerald green", "golden yellow",
    "black and white", "burnt orange", "deep purple", "teal",
    "scarlet and black", "navy blue", "maroon", "forest green",
    "bright red", "electric blue", "lime green", "charcoal grey",
]
_COLOR_WORDS = {
    "red", "blue", "green", "yellow", "orange", "purple", "black", "white",
    "teal", "maroon", "navy", "crimson", "scarlet", "gold", "golden", "silver",
    "grey", "gray", "pink", "brown", "violet", "indigo", "turquoise", "cyan",
    "magenta", "lime", "olive", "beige", "tan", "khaki",
}
_DEFAULT_GARMENT = "fight shorts"  # when the profile names no recognisable garment
# Every word used in a palette colour, so _strip_colors removes the whole phrase
# (incl. modifiers like 'royal'/'emerald'/'charcoal' that aren't standalone colour
# words) and a reshuffle can't leave a stray modifier behind.
_PALETTE_TOKENS = {t for c in _OUTFIT_COLORS for t in c.lower().split()}


def _has_color(text: str) -> bool:
    tl = (text or "").lower()
    return any(w in tl.split() or w in tl for w in _COLOR_WORDS)


def _color_for(name: str) -> str:
    """Deterministic colour for a fighter so an un-coloured garment is rendered
    in the SAME colour in every keyframe (no shot-to-shot drift)."""
    import hashlib
    h = int(hashlib.md5((name or "").encode("utf-8")).hexdigest(), 16)
    return _OUTFIT_COLORS[h % len(_OUTFIT_COLORS)]


def _distinct_color(name: str, used: set) -> str:
    """A fighter's locked colour, preferring its deterministic hash colour but
    walking to the next FREE palette colour when that one is already taken — so no
    two fighters share a colour (up to the palette size). Distinct colours matter
    because two fighters share one keyframe; identical colours invite the image
    model to swap/blend their kits."""
    pref = _color_for(name)
    start = _OUTFIT_COLORS.index(pref) if pref in _OUTFIT_COLORS else 0
    n = len(_OUTFIT_COLORS)
    for k in range(n):
        c = _OUTFIT_COLORS[(start + k) % n]
        if c not in used:
            return c
    return pref  # palette exhausted — accept a repeat


def _canonical_outfit(name: str, profile_text: str, used: set = None) -> str:
    """A fighter's locked outfit phrase, always with a colour. Uses the garment
    extracted from the profile (or a default), and prepends a colour when none is
    stated — a distinct one when `used` (already-assigned colours) is given."""
    garment = _extract_outfit(profile_text) or _DEFAULT_GARMENT
    if not _has_color(garment):
        color = _distinct_color(name, used) if used is not None else _color_for(name)
        garment = f"{color} {garment}"
    return garment


# A referee is a non-fighter official who appears in the VICTORY clip of an outcome
# (raising the winner's arm / handing over a cup). Refs are stored as ordinary
# `character` profiles so the server's image + LoRA pipeline works on them, but are
# tagged meta["role"]=="referee" so they are kept OUT of the fighter pools and
# dressed in an official's uniform instead of coloured fight kit.
REFEREE_WARDROBE = ("a black-and-white vertically striped referee shirt, "
                    "black trousers and an official's whistle")


def _is_referee_meta(meta: dict) -> bool:
    return str((meta or {}).get("role", "")).strip().lower() == "referee"


def _list_character_roles(out_dir: Path) -> dict:
    """{name: role} for locally-saved characters; role defaults to 'fighter'."""
    roles = {}
    base = Path(out_dir) / "characters"
    if base.exists():
        for d in base.iterdir():
            mp = d / "meta.json"
            if not mp.exists():
                continue
            try:
                meta = json.loads(mp.read_text())
            except Exception:
                meta = {}
            roles[d.name] = "referee" if _is_referee_meta(meta) else "fighter"
    return roles


def _referee_names(out_dir: Path) -> list:
    """Names of locally-saved referee profiles (sorted, deterministic)."""
    return sorted(n for n, r in _list_character_roles(out_dir).items()
                  if r == "referee")


def _referee_for(out_dir: Path, key: str = "") -> Optional[str]:
    """Pick a referee profile to officiate a scene. Deterministic per `key` (e.g.
    the match name) so the same match always gets the same ref; None if none exist."""
    refs = _referee_names(out_dir)
    if not refs:
        return None
    if not key:
        return refs[0]
    return refs[hash(key) % len(refs)]


def _load_wardrobe(out_dir: Path) -> dict:
    """Canonical {name: outfit} with colours locked, persisted to wardrobe.json so
    every keyframe of a match dresses each fighter identically — and the user can
    edit it. Built from fighter profiles on first use; newly-seen fighters are
    merged in without overwriting existing (possibly user-edited) entries. New
    fighters are given a colour distinct from those already assigned."""
    wf = out_dir / "wardrobe.json"
    saved = {}
    if wf.exists():
        try:
            saved = json.loads(wf.read_text())
        except Exception:
            saved = {}
    merged = dict(saved)
    # Colours already in use (from existing/edited entries) so freshly-added
    # fighters pick a different one.
    used = {c for c in _OUTFIT_COLORS
            for v in merged.values() if c in str(v or "").lower()}
    # All known fighters: built-in pool + locally saved characters.
    sources = {f["name"]: f.get("prompt", "") for f in FIGHTER_POOL}
    referees = set()
    chars_dir = out_dir / "characters"
    if chars_dir.exists():
        for d in chars_dir.iterdir():
            mp = d / "meta.json"
            if mp.exists():
                try:
                    meta = json.loads(mp.read_text())
                    sources[d.name] = meta.get("prompt", "") or meta.get("description", "")
                    if _is_referee_meta(meta):
                        referees.add(d.name)
                except Exception:
                    pass
    # Deterministic order so colour assignment is stable across runs.
    for nm in sorted(sources):
        if nm not in merged or not str(merged.get(nm) or "").strip():
            # Referees wear a fixed official's uniform (no coloured fight kit, no
            # palette colour reserved for them).
            if nm in referees:
                merged[nm] = REFEREE_WARDROBE
                continue
            outfit = _canonical_outfit(nm, sources[nm], used)
            merged[nm] = outfit
            for c in _OUTFIT_COLORS:
                if c in outfit.lower():
                    used.add(c)
                    break
    if merged != saved:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            wf.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
        except Exception:
            pass
    return merged


def _save_wardrobe(out_dir: Path, wardrobe: dict) -> None:
    """Persist a {name: outfit} wardrobe map to wardrobe.json (UI editor save)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = {str(k): str(v).strip() for k, v in (wardrobe or {}).items()
             if str(v).strip()}
    (out_dir / "wardrobe.json").write_text(
        json.dumps(clean, indent=2, sort_keys=True) + "\n")


def _strip_colors(text: str) -> str:
    """Drop colour words (and the connector 'and') from an outfit phrase, leaving
    just the garment, e.g. 'black and white fight shorts' → 'fight shorts'. Strips
    every token used in the palette too (e.g. 'royal', 'emerald', 'charcoal') so a
    reshuffle doesn't leave a stray colour modifier behind."""
    drop = _COLOR_WORDS | _PALETTE_TOKENS | {"and"}
    toks = [t for t in str(text or "").split() if t.lower() not in drop]
    return " ".join(toks).strip()


def _reshuffle_wardrobe(out_dir: Path) -> dict:
    """Reassign a DISTINCT colour to every fighter while keeping each one's garment.
    Fixes colour clashes (two fighters sharing a colour) in one click; overwrites
    manually-set colours but preserves garments."""
    w = _load_wardrobe(out_dir)
    used, new = set(), {}
    for nm in sorted(w):
        garment = _strip_colors(w[nm]) or _DEFAULT_GARMENT
        color = _distinct_color(nm, used)
        used.add(color)
        new[nm] = f"{color} {garment}".strip()
    _save_wardrobe(out_dir, new)
    return new


def _compose_kf_prompt(base_prompt: str, fighters: list, env: str,
                       env_desc: str, outfits: dict) -> str:
    """Build the keyframe image prompt from a clip/outcome's base prompt:
    a wardrobe lead with locked colours (so clothing stays consistent) plus an
    explicit environment clause using the FULL env description (not just the bare
    name), so the still actually lands in the right location.

    With TWO fighters in one frame the image model is prone to attribute leakage —
    painting fighter A's colour onto fighter B. We therefore (a) state each
    fighter's outfit up front, (b) restate the colour→fighter binding at the end,
    and (c) explicitly say the two kits are DIFFERENT colours and must not be
    swapped. We deliberately avoid the phrase "same colours" on a single still:
    it reads as "both fighters the same colour" and causes the very bleed we want
    to prevent (per-keyframe colour stability comes from reusing the SAME locked
    wardrobe string everywhere, not from telling the model "same")."""
    kf = base_prompt or ""
    # A camera-motion clip's prompt leads with a moving-camera directive ("Bold
    # moving camera — …. ") that is meaningless for a STILL keyframe and can induce
    # motion blur — strip it so the keyframe composes a clean, sharp anchor frame.
    if kf.startswith("Bold moving camera —"):
        _dot = kf.find(". ")
        if _dot != -1:
            kf = kf[_dot + 2:]
    worn = [(n, outfits[n]) for n in fighters if outfits.get(n)]
    if worn:
        wardrobe = "; ".join(f"{n} wearing {o}" for n, o in worn)
        if len(worn) >= 2:
            binding = ("; ".join(f"{n}'s outfit is {o}" for n, o in worn[:2]))
            tail = f" — {binding}; "
            # Only claim "different colours" when they ACTUALLY differ (two fighters
            # sharing a locked colour is a wardrobe.json clash the user can fix).
            if worn[0][1].strip().lower() != worn[1][1].strip().lower():
                tail += ("the two fighters wear clearly different colours, ")
            tail += ("keep each fighter's outfit colour bound to that fighter, "
                     "do not swap, mix or blend their colours")
        else:
            tail = f" — keep {worn[0][0]}'s exact outfit and colour ({worn[0][1]})"
        kf = f"{wardrobe}. {kf}{tail}"
    loc = (env_desc or "").strip() or (f"{env} location" if env else "")
    if loc:
        kf = f"[location: {loc}] " + kf
    return kf


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
    "upscale_model", "upscale_model_2x", "upscale_model_4x", "interpolation_model",
    "no_llm", "out_dir", "fps", "playback_fps", "clip_delay", "region", "include_female",
    "skip_characters", "reuse_fighters", "fighters", "num_fighters", "char_refs",
    "skip_environments", "reuse_environments", "environments", "num_environments", "env_refs",
    "skip_videos", "only_outcomes", "matches",
    "only_characters", "only_environments", "only_assets",
    "only_prompts", "only_videos",
    "consistency", "keyframe_steps", "keyframe_size",
    "character_strength", "lora_steps", "lora_rank", "lora_weight",
    "lora_train_base_model",
    "no_env_loras", "env_lora_steps", "env_lora_rank", "env_lora_weight",
    "video_loras", "video_lora_scale", "video_size",
    "clip_min_frames", "clip_max_frames", "single_clip_max_frames",
    "outcome_min_frames", "outcome_max_frames",
    "short_min", "short_max", "long_min", "long_max",
    "upscale_factor", "fps_multiplier",
    "web_port",
    "upload_endpoint", "upload_token", "upload_fixture_id", "upload_after_render",
    "odds_ranges",
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


def _ref_gen_res_steps(args):
    """(size, steps) for reference / keyframe image generation, honouring the
    resolution + steps configured on the Run page (keyframe_size / keyframe_steps).
    Re-reads the saved config file when available so Run-page edits apply without a
    restart; falls back to the launch args, then sane defaults. This is what makes
    the characters/environments/matches reference generators use the same resolution
    and steps as keyframes instead of hardcoded 512/768 + 28 steps."""
    size = getattr(args, "keyframe_size", None) or "832x480"
    steps = getattr(args, "keyframe_steps", None) or 28
    try:
        cf = getattr(args, "config", None)
        if cf and os.path.isfile(cf):
            c = load_config(cf)
            size = c.get("keyframe_size") or size
            steps = c.get("keyframe_steps") or steps
    except Exception:
        pass
    return str(size), int(steps)


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


# ===========================================================================
# Township Combat League upload — odds generation, anti-arbitrage, ZIP packing
# and chunked upload to the league server (mbetterd 3-step fixture-source API).
#
# Payout matrix (mirrored from the server's SureBetAnalyzer):
#   First extraction:  under / over (exactly one wins).
#   Second extraction: win1, win2, ko1, ko2, ret1, ret2, draw where
#     ko1 → fighter 2 wins by KO        (pays ko1 and win2)
#     ko2 → fighter 1 wins by KO        (pays ko2 and win1)
#     ret1 → fighter 2 wins by retire   (pays ret1 and win2)
#     ret2 → fighter 1 wins by retire   (pays ret2 and win1)
#     win1 → fighter 1 wins on points;  win2 → fighter 2;  draw → draw.
# ===========================================================================

# Odds ranges (inclusive); values are always rounded to two decimals.
ODDS_RANGES = {
    "under": (1.00, 2.00),
    "over": (1.00, 2.00),
    "win1": (1.00, 3.50),
    "win2": (1.00, 3.50),
    "ko1": (2.50, 7.00),
    "ko2": (2.50, 7.00),
    "ret1": (2.50, 7.00),
    "ret2": (2.50, 7.00),
    "draw": (1.50, 5.50),
}

OUTCOME_COLUMNS = list(ODDS_RANGES.keys())

# Anti-arbitrage ("sure bet") thresholds — mirror of app/utils/sure_bet_analyzer.py
MIN_SURE_BET_ODDS_FIRST = 2.0    # under/over: sure bet if BOTH > this
MIN_SURE_BET_ODDS_SECOND = 3.0   # win1/win2/draw: sure bet if ALL > this
MIN_SURE_BET_PRODUCT = 4.0       # 2-outcome scenarios: sure bet if product > this


def default_ranges() -> dict:
    """A fresh copy of the built-in odds ranges."""
    return {k: (lo, hi) for k, (lo, hi) in ODDS_RANGES.items()}


def _range_dict_to_json(ranges: dict) -> dict:
    """Convert a range map of tuples to a JSON-friendly map of ``[min, max]``
    lists (so it round-trips cleanly through config files)."""
    return {col: [round(float(lo), 2), round(float(hi), 2)]
            for col, (lo, hi) in ranges.items()}


def merge_ranges(overrides: Optional[dict]) -> dict:
    """Merge user-supplied range overrides onto the defaults.

    ``overrides`` may be a mapping of ``column -> [min, max]`` (or ``(min, max)``);
    unknown columns and malformed pairs are ignored so a partial/old config stays
    usable. Returns a complete, validated range map (min <= max, all columns).
    """
    ranges = default_ranges()
    if isinstance(overrides, dict):
        for col, pair in overrides.items():
            if col not in ranges:
                continue
            try:
                lo, hi = float(pair[0]), float(pair[1])
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            if lo > hi:
                lo, hi = hi, lo
            ranges[col] = (round(lo, 2), round(hi, 2))
    return ranges


def check_arbitrage(odds: dict) -> tuple:
    """Return ``(ok, reason)``. ``ok`` is True when the odds are SAFE (no sure
    bet). Replicates every check the server's ``SureBetAnalyzer.analyze_match``
    performs, with the same strict ``>`` comparisons, so anything we accept the
    server will too."""
    under = odds.get("under")
    over = odds.get("over")
    win1 = odds.get("win1")
    win2 = odds.get("win2")
    ko1 = odds.get("ko1")
    ko2 = odds.get("ko2")
    ret1 = odds.get("ret1")
    ret2 = odds.get("ret2")
    draw = odds.get("draw")

    # First extraction: under/over.
    if under and over and under > MIN_SURE_BET_ODDS_FIRST and over > MIN_SURE_BET_ODDS_FIRST:
        return False, f"under ({under}) and over ({over}) both > {MIN_SURE_BET_ODDS_FIRST}"

    # Second extraction: win1/win2/draw all above threshold (one always wins).
    if (win1 and win2 and draw and win1 > MIN_SURE_BET_ODDS_SECOND
            and win2 > MIN_SURE_BET_ODDS_SECOND and draw > MIN_SURE_BET_ODDS_SECOND):
        return False, (f"win1 ({win1}), win2 ({win2}) and draw ({draw}) all "
                       f"> {MIN_SURE_BET_ODDS_SECOND}")

    # Two-outcome scenarios: product must not exceed the threshold.
    pairs = [
        ("ko1", ko1, "win2", win2),
        ("ko2", ko2, "win1", win1),
        ("ret1", ret1, "win2", win2),
        ("ret2", ret2, "win1", win1),
    ]
    for an, a, bn, b in pairs:
        if a and b and a * b > MIN_SURE_BET_PRODUCT:
            return False, f"{an} ({a}) * {bn} ({b}) = {a * b:.2f} > {MIN_SURE_BET_PRODUCT}"

    # Matrix safety margin: must be > 1.0.
    if all(v for v in (win1, win2, ko1, ko2, ret1, ret2, draw)):
        margin = (1.0 / draw
                  + 1.0 / win1 + 1.0 / ko2 + 1.0 / ret2
                  + 1.0 / win2 + 1.0 / ko1 + 1.0 / ret1)
        if margin <= 1.0:
            return False, f"matrix safety margin {margin:.4f} <= 1.0"

    return True, "ok"


def _r2(lo: float, hi: float, rng) -> float:
    """Uniform sample in [lo, hi] rounded to two decimals (clamped into range)."""
    if hi < lo:
        hi = lo
    v = round(rng.uniform(lo, hi), 2)
    return min(max(v, round(lo, 2)), round(hi, 2))


def generate_odds(seed: int = None, max_tries: int = 10, ranges: dict = None) -> dict:
    """Generate a full, arbitrage-safe set of odds.

    Sampling is constraint-aware so a valid set is usually found on the first try:
    the win odds are drawn first (low enough that a KO/RET at its floor still
    fits), then each KO/RET is bounded by the product cap. Because the win minimum
    is ~1.0, a coupled KO/RET at the range maximum (e.g. 7.0) can NEVER avoid a
    sure bet, so the usable KO/RET ceiling is driven by the chosen win odd. The
    rounded result is verified with :func:`check_arbitrage` exactly as the server
    would, re-rolling up to ``max_tries`` times. Raises ``RuntimeError`` if no
    safe set is found."""
    rng = random.Random(seed)
    last_reason = "no attempts made"
    rg = merge_ranges(ranges)

    safe_product = MIN_SURE_BET_PRODUCT - 0.05
    ko_ret_lo = min(rg["ko1"][0], rg["ko2"][0], rg["ret1"][0], rg["ret2"][0])
    win_cap1 = min(rg["win1"][1], safe_product / ko_ret_lo)
    win_cap2 = min(rg["win2"][1], safe_product / ko_ret_lo)

    for _ in range(max_tries):
        win1 = _r2(rg["win1"][0], win_cap1, rng)
        win2 = _r2(rg["win2"][0], win_cap2, rng)

        ko1 = _r2(rg["ko1"][0], min(rg["ko1"][1], safe_product / win2), rng)
        ret1 = _r2(rg["ret1"][0], min(rg["ret1"][1], safe_product / win2), rng)
        ko2 = _r2(rg["ko2"][0], min(rg["ko2"][1], safe_product / win1), rng)
        ret2 = _r2(rg["ret2"][0], min(rg["ret2"][1], safe_product / win1), rng)

        draw = _r2(*rg["draw"], rng)
        under = _r2(*rg["under"], rng)
        over = _r2(*rg["over"], rng)

        odds = {
            "under": under, "over": over,
            "win1": win1, "win2": win2,
            "ko1": ko1, "ko2": ko2,
            "ret1": ret1, "ret2": ret2,
            "draw": draw,
        }
        ok, reason = check_arbitrage(odds)
        if ok:
            return odds
        last_reason = reason

    raise RuntimeError(
        f"could not generate arbitrage-safe odds in {max_tries} tries "
        f"(last failure: {last_reason})")


def _best_variant(base: Path):
    """Highest-quality existing variant for a base video path, or None if the
    base file does not exist. Reuses ``_video_variants`` (weakest→strongest)."""
    if not base.exists():
        return None
    variants = _video_variants(base)
    return variants[-1][1] if variants else base


def resolve_match_videos(out_dir, match_name: str, f1: str, f2: str) -> tuple:
    """Map each server ZIP filename to the best local video for the match.

    Returns ``(found, missing)`` where ``found`` maps ``"OVER.mp4" -> Path`` and
    ``missing`` lists the ZIP names with no source video. KO1/RET1 mean fighter 2
    wins; KO2/RET2 mean fighter 1 wins (server payout matrix)."""
    vdir = Path(out_dir) / "videos"

    # The draw is stored once per match under one fighter's name; find whichever.
    draw_src = None
    for cand in sorted(vdir.glob(f"{match_name}_*_draw.mp4")):
        suf = cand.stem[len(match_name) + 1:]
        if suf.endswith("_draw"):
            draw_src = cand
            break

    spec = {
        "OVER.mp4": vdir / f"{match_name}_long.mp4",
        "UNDER.mp4": vdir / f"{match_name}_short.mp4",
        "WIN1.mp4": vdir / f"{match_name}_{f1}_win.mp4",
        "WIN2.mp4": vdir / f"{match_name}_{f2}_win.mp4",
        "KO1.mp4": vdir / f"{match_name}_{f2}_ko_win.mp4",
        "KO2.mp4": vdir / f"{match_name}_{f1}_ko_win.mp4",
        "RET1.mp4": vdir / f"{match_name}_{f2}_retire.mp4",
        "RET2.mp4": vdir / f"{match_name}_{f1}_retire.mp4",
        "DRAW.mp4": draw_src,
    }

    found, missing = {}, []
    for name, base in spec.items():
        best = _best_variant(base) if base else None
        if best is not None and best.exists():
            found[name] = best
        else:
            missing.append(name)
    return found, missing


def match_video_signature(out_dir, match_name: str, f1: str, f2: str) -> str:
    """A stable signature over the match's source videos (size + mtime of the
    best variant for each ZIP slot). Changes whenever any video is re-rendered or
    re-enhanced, so a stored 'uploaded' state can be invalidated."""
    import hashlib
    found, missing = resolve_match_videos(out_dir, match_name, f1, f2)
    h = hashlib.sha1()
    for name in sorted(set(list(found.keys()) + missing)):
        p = found.get(name)
        if p is not None:
            st = p.stat()
            h.update(f"{name}:{p.name}:{st.st_size}:{int(st.st_mtime)}".encode())
        else:
            h.update(f"{name}:MISSING".encode())
    return h.hexdigest()


def build_match_zip(out_dir, match_name: str, f1: str, f2: str, dest_zip) -> tuple:
    """Pack the match's nine renamed videos into ``dest_zip``.

    Returns ``(ok, missing)``. ``ok`` is True only when all nine outcome videos
    were found and written. When some are missing the ZIP is still written with
    whatever exists so partial review is possible, but ``ok`` is False."""
    import zipfile
    found, missing = resolve_match_videos(out_dir, match_name, f1, f2)
    dest_zip = Path(dest_zip)
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_zip.with_suffix(dest_zip.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, path in found.items():
            zf.write(path, arcname=name)
    tmp.replace(dest_zip)
    return (len(missing) == 0), missing


def _json_or_raise(resp, what: str) -> dict:
    try:
        data = resp.json()
    except ValueError:
        data = {}
    if resp.status_code >= 400 or (isinstance(data, dict) and data.get("success") is False):
        msg = ((data.get("error") or data.get("details") or f"HTTP {resp.status_code}")
               if isinstance(data, dict) else f"HTTP {resp.status_code}")
        raise RuntimeError(f"{what} failed: {msg}")
    return data if isinstance(data, dict) else {}


def upload_match(endpoint: str, token: str, fixture_id: str,
                 meta: dict, odds: dict, zip_path,
                 progress_cb=None, chunk_size: int = 4 * 1024 * 1024,
                 timeout: int = 600) -> dict:
    """Upload one prepared match to the Township Combat League server.

    ``meta`` must contain ``fighter1_township``, ``fighter2_township`` and
    ``venue_kampala_township`` (optionally ``start_time``/``end_time`` ISO
    strings). The ZIP is uploaded in chunks so it survives request-size limits on
    reverse proxies. Returns the server's finalize JSON (includes ``match_id`` /
    ``match_number``); raises ``RuntimeError`` on any failure."""
    import uuid as _uuid

    def _emit(frac, label):
        if progress_cb:
            try:
                progress_cb(frac, label)
            except Exception:
                pass

    base = endpoint.rstrip("/")
    headers = {"Authorization": f"Bearer {token}"}
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise RuntimeError(f"ZIP not found: {zip_path}")

    # 1) Create the match (carries the odds).
    _emit(0.0, "creating match…")
    body = {
        "fighter1_township": meta["fighter1_township"],
        "fighter2_township": meta["fighter2_township"],
        "venue_kampala_township": meta["venue_kampala_township"],
        "outcomes": {k: float(v) for k, v in odds.items()},
    }
    for opt in ("start_time", "end_time", "result"):
        if meta.get(opt):
            body[opt] = meta[opt]

    r = requests.post(f"{base}/api/fixture/{fixture_id}/match",
                      json=body, headers=headers, timeout=timeout)
    data = _json_or_raise(r, "create match")
    match_id = data.get("match_id")
    if not match_id:
        raise RuntimeError(f"create match: no match_id in response ({data})")

    # 2) Upload the ZIP in chunks.
    upload_id = _uuid.uuid4().hex
    file_name = zip_path.name
    total = zip_path.stat().st_size
    total_chunks = max(1, (total + chunk_size - 1) // chunk_size)
    sent = 0
    with open(zip_path, "rb") as fh:
        for idx in range(total_chunks):
            blob = fh.read(chunk_size)
            files = {"chunk": (f"chunk_{idx}", blob, "application/octet-stream")}
            form = {
                "chunkIndex": str(idx),
                "totalChunks": str(total_chunks),
                "uploadId": upload_id,
                "fileName": file_name,
            }
            cr = requests.post(
                f"{base}/api/fixture/match/{match_id}/zip/chunk",
                data=form, files=files, headers=headers, timeout=timeout)
            _json_or_raise(cr, f"chunk {idx + 1}/{total_chunks}")
            sent += len(blob)
            _emit(0.05 + 0.85 * (sent / total if total else 1.0),
                  f"uploading {sent // (1024 * 1024)}/{total // (1024 * 1024)} MB")

    # 3) Finalize → assemble + go live.
    _emit(0.95, "finalizing…")
    fr = requests.post(
        f"{base}/api/fixture/match/{match_id}/zip/finalize",
        json={"uploadId": upload_id, "fileName": file_name},
        headers=headers, timeout=timeout)
    result = _json_or_raise(fr, "finalize")
    _emit(1.0, "done")
    result.setdefault("match_id", match_id)
    return result


# ---------------------------------------------------------------------------
# Township upload — config resolution + per-match upload state
# ---------------------------------------------------------------------------
def _resolve_odds_ranges(args) -> dict:
    """Return the configured odds-range overrides as a plain dict (or {}).

    ``args.odds_ranges`` may be a dict (loaded from config), a JSON string
    (rare CLI use), or None. Always returns something safe for
    ``generate_odds(ranges=...)``.
    """
    raw = getattr(args, "odds_ranges", None)
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except Exception:
            raw = None
    return raw if isinstance(raw, dict) else {}


def _collect_odds_ranges(fv) -> dict:
    """Assemble an odds-range dict from a form-value getter ``fv(key)``.

    Reads ``odds_<col>_min`` / ``odds_<col>_max`` for each outcome column. Falls
    back to the built-in default for any field left blank or unparseable, so the
    stored map is always complete and valid.
    """
    defaults = default_ranges()
    out = {}
    for col, (dlo, dhi) in defaults.items():
        def _num(suffix, fallback):
            try:
                return float(fv(f"odds_{col}_{suffix}", "") or fallback)
            except (TypeError, ValueError):
                return fallback
        out[col] = [_num("min", dlo), _num("max", dhi)]
    return _range_dict_to_json(merge_ranges(out))


def _upload_state_path(out_dir) -> Path:
    return Path(out_dir) / "videos" / "upload_state.json"


def _load_upload_state(out_dir) -> dict:
    p = _upload_state_path(out_dir)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _save_upload_state(out_dir, state: dict) -> None:
    p = _upload_state_path(out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(p)


def _get_match_upload(out_dir, mn: str) -> dict:
    return _load_upload_state(out_dir).get(mn, {}) or {}


def _set_match_upload(out_dir, mn: str, entry: dict) -> None:
    state = _load_upload_state(out_dir)
    if entry:
        state[mn] = entry
    else:
        state.pop(mn, None)
    _save_upload_state(out_dir, state)


def _clear_match_upload(out_dir, mn: str) -> None:
    _set_match_upload(out_dir, mn, {})


def _match_meta(out_dir, mn: str):
    """Return (f1, f2, env) for a match from its saved plan, or ("","","")."""
    pf = Path(out_dir) / "videos" / "prompts.json"
    if pf.exists():
        try:
            plan = json.loads(pf.read_text())
        except Exception:
            plan = {}
        for m in plan.get("fight_plan", []):
            if m.get("match_name") == mn:
                return m.get("f1", ""), m.get("f2", ""), m.get("env", "")
    return "", "", ""


def _upload_config_ready(args):
    """Return (ok, missing_list) for the server upload settings on ``args``."""
    missing = [k for k in ("upload_endpoint", "upload_token", "upload_fixture_id")
               if not (getattr(args, k, "") or "").strip()]
    return (not missing), missing


def prepare_match_odds_zip(out_dir, mn: str, args, log=None,
                           odds: dict = None, max_tries: int = 10) -> dict:
    """Generate arbitrage-safe odds (unless ``odds`` is supplied) and pack the
    match's renamed ZIP, persisting both into the upload state.

    Returns a dict: ``{ok, odds, missing, zip, arbitrage_ok, error}``. ``ok`` is
    True only when odds passed the anti-arbitrage check AND all nine videos were
    packed. When odds fail after ``max_tries`` the ZIP is still built so the
    operator can see what's missing, but ``ok`` is False.
    """
    log = log or _log
    f1, f2, _env = _match_meta(out_dir, mn)
    result = {"ok": False, "odds": None, "missing": [], "zip": None,
              "arbitrage_ok": False, "error": None}

    # 1) Odds (generate or validate the supplied set).
    ranges = _resolve_odds_ranges(args)
    if odds is None:
        try:
            odds = generate_odds(max_tries=max_tries, ranges=ranges)
        except RuntimeError as e:
            result["error"] = str(e)
            log(f"  ✗ odds: {e}")
            return result
    arb_ok, arb_reason = check_arbitrage(odds)
    result["odds"] = odds
    result["arbitrage_ok"] = arb_ok
    if not arb_ok:
        result["error"] = f"arbitrage check failed: {arb_reason}"
        log(f"  ✗ odds rejected: {arb_reason}")

    # 2) ZIP (best-quality variant of each of the nine videos).
    zip_path = Path(out_dir) / "videos" / "uploads" / f"{mn}.zip"
    zip_ok, missing = build_match_zip(out_dir, mn, f1, f2, zip_path)
    result["missing"] = missing
    result["zip"] = str(zip_path)
    if missing:
        log(f"  ⚠ ZIP missing {len(missing)} video(s): {', '.join(missing)}")

    # 3) Persist state (preserve any prior uploaded_at only if nothing changed —
    #    a fresh prepare invalidates a previous upload because content/odds moved).
    entry = {
        "odds": odds,
        "zip": str(zip_path),
        "missing": missing,
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sig": match_video_signature(out_dir, mn, f1, f2),
        "arbitrage_ok": arb_ok,
    }
    _set_match_upload(out_dir, mn, entry)

    result["ok"] = arb_ok and zip_ok
    if result["ok"]:
        log(f"  ✓ odds + ZIP ready for {mn}")
    return result


def upload_prepared_match(out_dir, mn: str, args, log=None, progress_cb=None) -> dict:
    """Upload a previously prepared match. Refuses (without contacting the server)
    when the upload config is incomplete, no ZIP/odds were prepared, the prepared
    videos changed since preparation, or the odds fail the anti-arbitrage re-check.

    Returns ``{ok, error, match_id, match_number}``.
    """
    log = log or _log
    out = {"ok": False, "error": None, "match_id": None, "match_number": None}

    cfg_ok, missing_cfg = _upload_config_ready(args)
    if not cfg_ok:
        out["error"] = f"upload not configured (missing: {', '.join(missing_cfg)})"
        return out

    entry = _get_match_upload(out_dir, mn)
    odds = entry.get("odds")
    zip_path = entry.get("zip")
    if not odds or not zip_path or not Path(zip_path).exists():
        out["error"] = "no prepared ZIP/odds — generate odds & ZIP first"
        return out
    if entry.get("missing"):
        out["error"] = f"ZIP incomplete — missing {', '.join(entry['missing'])}"
        return out

    f1, f2, env = _match_meta(out_dir, mn)
    # Re-validate the content signature: a re-render/enhance since prep invalidates.
    if entry.get("sig") and entry["sig"] != match_video_signature(out_dir, mn, f1, f2):
        out["error"] = "videos changed since the ZIP was prepared — regenerate the ZIP"
        return out
    # Re-run the anti-arbitrage check (never upload a sure bet).
    arb_ok, arb_reason = check_arbitrage(odds)
    if not arb_ok:
        out["error"] = f"arbitrage check failed: {arb_reason} — regenerate odds"
        return out

    meta = {
        "fighter1_township": f1 or mn,
        "fighter2_township": f2 or mn,
        "venue_kampala_township": env or "Township Arena",
    }
    try:
        res = upload_match(
            args.upload_endpoint, args.upload_token, args.upload_fixture_id,
            meta, odds, zip_path, progress_cb=progress_cb)
    except Exception as e:
        out["error"] = str(e)
        log(f"  ✗ upload failed: {e}")
        return out

    entry.update({
        "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "match_id": res.get("match_id"),
        "match_number": res.get("match_number"),
        "fixture_id": args.upload_fixture_id,
        "sig": match_video_signature(out_dir, mn, f1, f2),
    })
    _set_match_upload(out_dir, mn, entry)
    out.update({"ok": True, "match_id": res.get("match_id"),
                "match_number": res.get("match_number")})
    log(f"  ✓ uploaded {mn} → match #{res.get('match_number')}")
    return out


def _auto_upload_all_matches(out_dir, args, log=None) -> None:
    """After a full render, prepare odds+ZIP and upload every match that has its
    short/long finals on disk. Failures are logged and skipped (the match stays
    rendered locally for a later manual retry)."""
    log = log or _log
    cfg_ok, missing_cfg = _upload_config_ready(args)
    if not cfg_ok:
        log(f"  ⚠ skipping auto-upload — upload not configured (missing: {', '.join(missing_cfg)})")
        return
    pf = Path(out_dir) / "videos" / "prompts.json"
    if not pf.exists():
        return
    try:
        plan = json.loads(pf.read_text())
    except Exception:
        return
    vdir = Path(out_dir) / "videos"
    log("\n  ── Township upload: preparing odds + ZIPs and uploading ──")
    for m in plan.get("fight_plan", []):
        mn = m.get("match_name")
        if not mn:
            continue
        if not (vdir / f"{mn}_long.mp4").exists() and not (vdir / f"{mn}_short.mp4").exists():
            continue  # nothing rendered for this match
        log(f"\n  • {mn}")
        prep = prepare_match_odds_zip(out_dir, mn, args, log=log)
        if not prep["ok"]:
            log(f"    ⏭ skipping upload for {mn} ({prep.get('error') or 'incomplete'})")
            continue
        upload_prepared_match(out_dir, mn, args, log=log)


def _fighter_desc_hint(name: str, char_descriptions: dict) -> str:
    """Return a compact "name (visual traits)" hint for embedding in a prompt.

    The NAME is always kept: it identifies the fighter in the clip prompt and is
    the token the per-fighter LoRA + character_profiles anchor identity to. The
    short description (first few traits) just guides appearance.
    """
    desc = char_descriptions.get(name, "")
    if not desc:
        return name
    # Keep up to 3 key visual traits (drop anything after the 3rd comma) and
    # prefix the name so the prompt still says WHO this is.
    parts = [p.strip() for p in desc.split(",")]
    visual = ", ".join(parts[:3])
    return f"{name} ({visual})"


def _build_match_clip_specs(fps: int, cf_lo: int, cf_hi: int,
                            long_target: float, f1: str, f2: str) -> list:
    """Build a match's ordered clip specs: the pre-fight INTRO (a solo entrance for
    each fighter, then a referee-officiated face-off), followed by fight clips until
    `long_target` playback seconds is reached. The intro clips are extra and do NOT
    count toward `long_target`. Prompts are filled later by `_fill_clip_prompt`."""
    specs, ci = [], 0
    for _role, _who in (("entrance", [f1]), ("entrance", [f2]), ("faceoff", [f1, f2])):
        nf = random.randint(cf_lo, cf_hi)
        specs.append({"idx": ci, "clip_seconds": round(nf / max(1, fps), 2),
                      "nf": nf, "intensity": "introduction", "role": _role,
                      "fighters": list(_who), "shot": None, "prompt": None})
        ci += 1
    planned = 0.0
    _fight_ord = 0
    while planned < long_target:
        round_num = (ci - 3) // 3 + 1
        intensity = ("early exchanges" if round_num == 1
                     else "midpoint battle" if round_num == 2
                     else "climactic final exchange")
        nf = random.randint(cf_lo, cf_hi)
        # Every other fight clip is a CAMERA-MOTION clip (a bold moving-camera shot
        # through the environment) so a match isn't all locked-off frames.
        specs.append({"idx": ci, "clip_seconds": round(nf / max(1, fps), 2),
                      "nf": nf, "role": "fight", "intensity": intensity,
                      "camera": (_fight_ord % 2 == 1),
                      "shot": None, "prompt": None})
        planned += nf / max(1, fps)
        _fight_ord += 1
        ci += 1
    return specs


def _clip_role_fighters(c: dict, f1: str, f2: str):
    """Resolve the (role, fighters) for a fight-match clip.

    Honour an explicit `role` (+ `fighters`) when present — that's how new-style
    clips built by `_build_match_clip_specs` carry their identity. Otherwise INFER
    from position so the first three clips of ANY match (including legacy matches
    created before the intro feature, whose clips have no role) are the pre-fight
    intro: clip 0 = fighter-1 entrance, clip 1 = fighter-2 entrance, clip 2 = the
    referee-officiated face-off, and everything after = the fight."""
    role = c.get("role")
    if role:
        if role == "entrance":
            return role, list(c.get("fighters") or [f1])
        if role == "faceoff":
            return role, list(c.get("fighters") or [f1, f2])
        return "fight", [f1, f2]
    idx = int(c.get("idx", 0) or 0)
    if idx == 0:
        return "entrance", [f1]
    if idx == 1:
        return "entrance", [f2]
    if idx == 2:
        return "faceoff", [f1, f2]
    return "fight", [f1, f2]


def _fill_clip_prompt(prompter, c: dict, f1: str, f2: str, env, env_desc: str,
                      char_descriptions: dict, referee: str = None,
                      match_avoid: list = None, focus: str = "") -> str:
    """Populate `c['shot']` and `c['prompt']` for one clip according to its role.

    role "entrance" → a solo intro for the clip's single fighter; "faceoff" → a
    stare-down between both fighters with the referee signalling START; anything
    else → a normal fight shot (appended to `match_avoid` so the match stays varied).
    The resolved role/fighters are PERSISTED onto the clip so the keyframe + render
    use the same identity, and any stale `kf_prompt` override is cleared so this
    freshly written prompt drives keyframe generation."""
    role, fighters = _clip_role_fighters(c, f1, f2)
    # Persist the resolved identity (so legacy clips upgraded here keep their role),
    # and drop any prior keyframe-prompt override so the new prompt is authoritative.
    c["role"] = role
    if role in ("entrance", "faceoff"):
        c["fighters"] = list(fighters)
    c.pop("kf_prompt", None)
    cont = _continuity_clause(env or None)
    if role == "entrance":
        who = fighters[0]
        shot = prompter.intro_shot("entrance", who, env_desc=env_desc)
        hint = _fighter_desc_hint(who, char_descriptions)
        c["shot"] = shot
        c["prompt"] = f"{hint} — {shot} — {cont} — {ENTRANCE_PROMPT_SUFFIX}"
    elif role == "faceoff":
        shot = prompter.intro_shot("faceoff", f1, f2, env_desc=env_desc, referee=referee)
        h1 = _fighter_desc_hint(f1, char_descriptions)
        h2 = _fighter_desc_hint(f2, char_descriptions)
        c["shot"] = shot
        c["prompt"] = f"{h1} vs {h2} — {shot} — {cont} — {FACEOFF_PROMPT_SUFFIX}"
    else:
        shot = prompter.fight_shot(
            f1, f2, env_desc,
            match_context=f"Match stage: {c.get('intensity', '')}. ",
            avoid=match_avoid, action_focus=focus)
        h1 = _fighter_desc_hint(f1, char_descriptions)
        h2 = _fighter_desc_hint(f2, char_descriptions)
        c["shot"] = shot
        # Camera-motion clips lead with a bold moving-camera directive so the I2V
        # model actually moves the camera through the scene instead of locking off.
        # Legacy/un-flagged clips get a camera decision assigned here (and persisted)
        # so regenerating their prompt can add motion to an older match too.
        is_cam = c.get("camera")
        if is_cam is None:
            is_cam = (random.random() < 0.5)
            c["camera"] = is_cam
        if is_cam:
            move = random.choice(CAMERA_MOVES)
            c["prompt"] = (f"Bold moving camera — {move}. {h1} vs {h2} — {shot} "
                           f"— {cont} — {FIGHT_PROMPT_SUFFIX}")
        else:
            c["prompt"] = f"{h1} vs {h2} — {shot} — {cont} — {FIGHT_PROMPT_SUFFIX}"
        if match_avoid is not None:
            match_avoid.append(shot[:60])
    return c["shot"]


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
    # Long = the WHOLE match in order; if the rendered clips don't sum to the
    # long target (clips can render a touch short), pad by repeating from the
    # start so the final still covers the target length.
    long_clips = list(clips)
    accum, pos = sum(d for _, d in long_clips), 0
    while accum < long_target and clips:
        path, dur = clips[pos % len(clips)]
        long_clips.append((path, dur))
        accum += dur
        pos += 1
    _write_concat(long_clips, str(video_dir / f"{match_name}_long.mp4"),
                  f"long  (~{long_target:.0f}s)")
    return len(clips)


def _upscale_model_for(obj, factor) -> str:
    """Pick the configured upscale model for a given factor (2 or 4).

    Looks up the factor-specific field (`upscale_model_2x` / `upscale_model_4x`),
    falling back to the generic `upscale_model`, then to '' (→ the server
    auto-selects a configured upscaler). `obj` may be an argparse Namespace or a
    dict (config)."""
    def _g(name):
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)
    try:
        f = int(factor)
    except Exception:
        f = 0
    specific = _g(f"upscale_model_{f}x") if f in (2, 4) else None
    return ((specific or _g("upscale_model") or "")).strip()


def _enhance_suffix(upscale: int, fps_mult: int) -> str:
    """Filename suffix describing the enhancement, e.g. '_2x', '_3xfps', '_2x_2xfps'."""
    parts = []
    if upscale in (2, 4):
        parts.append(f"{upscale}x")
    if fps_mult and fps_mult > 1:
        parts.append(f"{fps_mult}xfps")
    return ("_" + "_".join(parts)) if parts else ""


def _video_variants(p: Path) -> list:
    """Return [(label, Path)] for a base video: the original first, then any
    enhanced siblings (`<stem>_2x.mp4`, `<stem>_2x_2xfps.mp4`, `<stem>_3xfps.mp4`,
    …) produced by the post-process enhancer. Labels are human-readable, e.g.
    'original', '2× upscaled', '2× + 2×fps'. Non-enhancement siblings are
    ignored. Sorted by an 'enhancement weight' so richer variants come later."""
    import re as _re
    out = [("original", p, 0)]
    stem = p.stem
    for sib in p.parent.glob(f"{stem}_*{p.suffix}"):
        if sib == p:
            continue
        suf = sib.stem[len(stem) + 1:]
        toks = suf.split("_")
        if not toks or not all(_re.fullmatch(r"\d+x(fps)?", t) for t in toks):
            continue
        weight, labels = 0, []
        for t in toks:
            n = int(_re.match(r"\d+", t).group())
            if t.endswith("xfps"):
                labels.append(f"{n}×fps"); weight += n
            else:
                labels.append(f"{n}× upscaled"); weight += n * 10
        out.append((" + ".join(labels), sib, weight))
    out.sort(key=lambda t: t[2])
    return [(lbl, pth) for (lbl, pth, _w) in out]


def _op_with_progress(client, phases, blocking_call, progress_cb=None) -> bytes:
    """Run `blocking_call()` while polling CoderAI's /v1/video/progress in a
    background thread, feeding `progress_cb(cur, total, it_per_s, phase)` live
    frame progress whenever the server's phase is one of `phases`. The callback's
    4th arg (phase) is optional for back-compat with 3-arg callbacks."""
    if not progress_cb:
        return blocking_call()
    import threading, inspect
    try:
        _nargs = len(inspect.signature(progress_cb).parameters)
    except (TypeError, ValueError):
        _nargs = 3
    stop = threading.Event()

    def _emit(cur, total, ips, phase):
        try:
            if _nargs >= 4:
                progress_cb(cur, total, ips, phase)
            else:
                progress_cb(cur, total, ips)
        except Exception:
            pass

    def _poll():
        last = -1
        while not stop.is_set():
            try:
                pr = client.video_progress()
            except Exception:
                pr = {}
            if pr.get("phase") in phases and pr.get("total"):
                cur = pr.get("current", 0)
                if cur != last:
                    last = cur
                    _emit(cur, pr.get("total", 0), pr.get("it_per_s", 0), pr.get("phase"))
            stop.wait(0.4)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    try:
        return blocking_call()
    finally:
        stop.set()
        t.join(timeout=1)


def _upscale_with_progress(client, data: bytes, factor: int, model,
                           progress_cb=None) -> bytes:
    """Upscale while streaming per-frame progress (phase 'upscaling')."""
    return _op_with_progress(client, ("upscaling",),
                             lambda: client.upscale_video(data, factor, model),
                             progress_cb)


def _interpolate_with_progress(client, data: bytes, fps_mult: int,
                               model=None, progress_cb=None, output_fps=None) -> bytes:
    """Raise FPS while streaming progress (phase 'interpolating'). `model` is the
    in-process interpolation model (RIFE/FILM); blank → CoderAI auto-selects a
    configured one. `output_fps` overrides the final play rate."""
    return _op_with_progress(client, ("interpolating",),
                             lambda: client.interpolate_video(data, fps_mult, model,
                                                              output_fps=output_fps),
                             progress_cb)


def _enhance_video_file(client, upscale_model: str, src: Path,
                        upscale: int = 0, fps_mult: int = 0,
                        force: bool = False, progress_cb=None,
                        interpolation_model: str = None,
                        final_fps: int = None) -> Optional[Path]:
    """Upscale (2x/4x) and/or raise FPS of one video, writing a NEW file alongside
    the original (e.g. match_short_2x_2xfps.mp4). Returns the new path, or None if
    nothing to do. Skips re-doing an already-enhanced output that is newer —
    unless `force` is set, which always re-enhances from the ORIGINAL `src`
    (overwriting the existing enhanced file).

    Upscaling is a real AI super-resolution op on CoderAI. `upscale_model` is
    optional: when blank, CoderAI auto-selects a configured AI upscaler (e.g.
    Real-ESRGAN). There is no CPU/ffmpeg fallback — if nothing is configured the
    server returns an error. Frame interpolation uses CoderAI's RIFE neural
    interpolator."""
    suffix = _enhance_suffix(upscale, fps_mult)
    if not suffix:
        return None
    out = src.with_name(src.stem + suffix + src.suffix)
    if not force and out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        _log(f"    ↻ already enhanced: {out.name}")
        return out
    # Forced re-enhance: delete the stale output first so it's a clean recreate
    # (and a mid-way failure can't leave an old file masquerading as fresh).
    if force and out.exists():
        try:
            out.unlink()
            _log(f"    🗑 removed old {out.name} (force re-enhance)")
        except Exception:
            pass
    data = src.read_bytes()
    if upscale in (2, 4):
        um = (upscale_model or "").strip() or None
        _log(f"    ⬆ upscaling {src.name} ×{upscale}"
             f"{(' via ' + um) if um else ' (CoderAI auto-select)'}…")
        data = _upscale_with_progress(client, data, upscale, um, progress_cb)
    if fps_mult and fps_mult > 1:
        _im = (interpolation_model or "").strip() or None
        _of = int(final_fps) if final_fps else None
        _log(f"    ⏩ raising FPS of {src.name} ×{fps_mult}"
             f"{(' via ' + _im) if _im else ' (CoderAI auto-select)'}"
             f"{(' → ' + str(_of) + 'fps') if _of else ''}…")
        data = _interpolate_with_progress(client, data, fps_mult, _im, progress_cb,
                                          output_fps=_of)
    out.write_bytes(data)
    _log(f"    ✓ enhanced → {out.name}  ({get_video_duration(str(out)):.1f}s)")
    return out


def _clip_stem_fight(match_name: str, idx: int) -> str:
    return f"{match_name}_clip{idx:02d}"


def _clip_stem_outcome(fighter: str, outcome: str, match_name: str = None) -> str:
    # Per-match outcomes are named "<match>_<fighter>_<outcome>" so each match
    # has its own (different) outcome scenes. Legacy per-fighter outcomes (no
    # match_name) keep the old "<fighter>_<outcome>" name.
    if match_name:
        return f"{match_name}_{fighter}_{outcome}"
    return f"{fighter}_{outcome}"


def _outcome_seg_stem(base: str, i: int) -> str:
    """Per-segment keyframe stem for an outcome video. Segment 0 keeps the legacy
    base stem (back-compat with the old single-keyframe outcomes); later segments
    append _s<i>. Each outcome clip (finish, then victory) gets its OWN keyframe so
    the victory shot is anchored to a real referee-raising-the-arm image instead of
    drifting off the finish frame."""
    return base if i == 0 else f"{base}_s{i}"


def _outcome_kf_stems(o: dict) -> list:
    """All keyframe stems an outcome produces — one per shot/segment (finish, then
    victory). Legacy outcomes without `shots` yield the single base stem."""
    base = _clip_stem_outcome(o["fighter"], o["outcome"], o.get("match_name"))
    n = len(o.get("shots") or []) or 1
    return [_outcome_seg_stem(base, i) for i in range(n)]


def _dedupe_match_draws(outcome_plan: list, match_name: str, mf: set) -> list:
    """Return outcome_plan with at MOST ONE draw for the given match.

    A draw means both fighters are declared joint winners, so there is exactly one
    draw per match — never one per fighter. Older plans (or plans built before this
    rule) can carry a draw for each fighter; this removes the extras in place,
    keeping the first. Non-draw outcomes and other matches are untouched."""
    seen_draw = False
    kept = []
    for o in outcome_plan:
        belongs = ((o.get("match_name") == match_name) if o.get("match_name")
                   else (o.get("fighter") in mf))
        if belongs and o.get("outcome") == "draw":
            if seen_draw:
                continue  # drop the duplicate draw
            seen_draw = True
        kept.append(o)
    return kept


def _outcome_segments_spec(outcome: str):
    """The clips that compose an outcome VIDEO and how the frame budget splits.

    Every outcome is a two-shot sequence: the decisive finish first, then the
    referee raising the winner's arm. Returns [(role, frame_fraction), ...]."""
    if outcome == "draw":
        return [("final_exchange", 0.55), ("draw_decision", 0.45)]
    return [("finish", 0.6), ("victory", 0.4)]


def _plan_outcome_shots(prompter, o: dict, char_descriptions: dict,
                        opponent: str = None) -> None:
    """(Re)generate the multi-clip `shots` for one outcome entry in place.

    Each outcome video is assembled from a decisive FINISH clip (how the match
    ends — KO, retirement, the last action) followed by a VICTORY clip (the
    winner + referee raising their arm; both arms for a draw). The total frame
    budget (`o['nf']`) is split across the clips per _outcome_segments_spec.

    For back-compat, o['shot'] / o['prompt'] mirror the FIRST (finish) clip —
    they still feed the outcome keyframe and the editable prompt shown in the UI.
    """
    opponent = opponent or o.get("opponent")
    spec = _outcome_segments_spec(o.get("outcome", "win"))
    total = int(o.get("nf") or 0) or 48
    f_hint = _fighter_desc_hint(o.get("fighter", ""), char_descriptions)
    cont = _continuity_clause(o.get("env"))
    shots, allocated = [], 0
    for i, (role, frac) in enumerate(spec):
        nf = (total - allocated) if i == len(spec) - 1 else max(8, int(round(total * frac)))
        allocated += nf
        shot = prompter.outcome_shot(o["fighter"], o["outcome"],
                                     o.get("env_desc") or "", role=role,
                                     opponent=opponent)
        prompt = (f"{f_hint} — {shot} — {cont} "
                  "— African township fight, cinematic, dynamic camera, brutal")
        shots.append({"role": role, "shot": shot, "prompt": prompt, "nf": int(nf)})
    o["shots"] = shots
    if opponent:
        o["opponent"] = opponent
    o["shot"] = shots[0]["shot"]
    o["prompt"] = shots[0]["prompt"]
    # Freshly written shots carry no per-shot keyframe-prompt override, so the
    # outcome keyframes recompose from these new prompts. Also drop any stale
    # entry-level override (legacy no-shots path) for the same reason.
    o.pop("kf_prompt", None)


def _model_slug(model_id: str) -> str:
    """Short filesystem-safe slug for a model id, used to tag video LoRAs with the
    exact model they were trained for (e.g. Wan-AI/Wan2.2-T2V-A14B-Diffusers →
    'wan-ai_wan2.2-t2v-a14b-diffusers')."""
    import re as _re
    s = (model_id or "").strip().lower()
    s = s.replace("/", "_").replace("\\", "_").replace(" ", "-")
    s = _re.sub(r"[^a-z0-9._-]+", "-", s).strip("-_.")
    return s or "model"


def _load_json_map(path: Path) -> dict:
    """Load a JSON dict map from disk, or {} if missing/unreadable."""
    try:
        if Path(path).exists():
            return json.loads(Path(path).read_text()) or {}
    except Exception:
        pass
    return {}


# ── Training session token + per-LoRA server job tracking ─────────────────────
# A stable per-output session token lets the server tag this client's training
# jobs so we (and only we) can recover them after a restart — no spillover into
# another user's concurrent run. Per-LoRA server job_ids are persisted so that
# after a township restart we re-attach to a still-running server job instead of
# launching a duplicate (the server keeps training regardless of this client).
def _session_token(out_dir: Path) -> str:
    p = Path(out_dir) / ".train_session"
    try:
        if p.exists():
            tok = p.read_text().strip()
            if tok:
                return tok
    except Exception:
        pass
    import uuid as _uuid
    tok = f"township-{_uuid.uuid4().hex[:12]}"
    try:
        p.write_text(tok)
    except Exception:
        pass
    return tok


def _train_jobs_file(out_dir: Path) -> Path:
    return Path(out_dir) / "lora_train_jobs.json"


def _get_lora_job_id(out_dir: Path, lora_name: str):
    return _load_json_map(_train_jobs_file(out_dir)).get(lora_name)


def _set_lora_job_id(out_dir: Path, lora_name: str, job_id):
    f = _train_jobs_file(out_dir)
    m = _load_json_map(f)
    if job_id:
        m[lora_name] = job_id
    else:
        m.pop(lora_name, None)
    try:
        f.write_text(json.dumps(m, indent=2))
    except Exception:
        pass


def _lora_name_ref(path: str):
    """Derive the server-side registered LoRA name from a trained-LoRA path.

    LoRAs are trained on the coderai server, which saves them under
    <loras_dir>/<registered_name>/pytorch_lora_weights.safetensors and returns
    that path. The *registered name* (the directory holding the weights) is a
    filesystem-path-independent handle the server can resolve no matter where it
    runs — so we reference LoRAs by `id: "name:<registered>"` instead of a raw
    path that's only meaningful if client and server share a disk. Returns None
    when the path doesn't look like a trained-LoRA layout."""
    if not path:
        return None
    p = str(path).rstrip("/\\")
    base = os.path.basename(p)
    if base.lower().endswith((".safetensors", ".bin", ".pt", ".ckpt")):
        reg = os.path.basename(os.path.dirname(p))
    else:
        reg = base  # path is the LoRA directory itself
    return f"name:{reg}" if reg else None


def _lora_spec(path: str, weight: float, name: str) -> dict:
    """One `loras` request entry. References the server-registered LoRA by name
    (works even when client and server are on different machines) and keeps the
    raw path as a legacy fallback for co-located setups."""
    spec = {"weight": float(weight), "name": name}
    ref = _lora_name_ref(path)
    if ref:
        spec["id"] = ref
    spec["model"] = path  # fallback: used only if the id can't be resolved
    return spec


def _lora_specs_for(fighters: list, lora_map: dict, weight: float) -> list:
    """Build the `loras` request list for the fighters appearing in a clip."""
    specs = []
    for f in fighters:
        path = (lora_map or {}).get(f)
        if path:
            specs.append(_lora_spec(path, weight, f))
    return specs


def _env_lora_specs_for(env: str, env_lora_map: dict, weight: float) -> list:
    """Build the `loras` request entry for the environment used in a clip."""
    if not env:
        return []
    path = (env_lora_map or {}).get(env)
    if path:
        return [_lora_spec(path, weight, f"env_{env}")]
    return []


def _video_lora_path(entry, slug: str):
    """Resolve a video-LoRA map entry to the path for the current model slug.

    Video maps are nested: name -> {slug: path} (a fighter can have a LoRA per
    video model). Tolerates a legacy flat string entry."""
    if isinstance(entry, dict):
        return entry.get(slug)
    if isinstance(entry, str):
        return entry
    return None


def _video_lora_specs_for(fighters: list, vmap: dict, slug: str, weight: float) -> list:
    """`loras` specs from the per-model video-LoRA map for the current video model."""
    specs = []
    for f in fighters:
        path = _video_lora_path((vmap or {}).get(f), slug)
        if path:
            specs.append(_lora_spec(path, weight, f))
    return specs


def _env_video_lora_specs_for(env: str, env_vmap: dict, slug: str, weight: float) -> list:
    if not env:
        return []
    path = _video_lora_path((env_vmap or {}).get(env), slug)
    if path:
        return [_lora_spec(path, weight, f"env_{env}")]
    return []


# Per-kind LoRA training parameters: server name prefix, local cache file,
# the train_lora keyword used to pull reference images, and a friendly label.
_LORA_KINDS = {
    "character":   {"prefix": "fighter_", "file": "loras.json",     "label": "Character"},
    "environment": {"prefix": "env_",     "file": "env_loras.json", "label": "Environment"},
}

# Video (Wan DiT) LoRAs are kept ALONGSIDE the image LoRAs above — separate maps,
# separate on-disk names tagged with the video model they were trained for.
_VIDEO_LORA_KINDS = {
    "character":   {"prefix": "vfighter_", "file": "video_loras.json",     "label": "Character video"},
    "environment": {"prefix": "venv_",     "file": "env_video_loras.json", "label": "Environment video"},
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


def _train_profile_video_loras(client: CoderAIClient, video_model: str, out_dir: Path,
                               names: list, kind: str,
                               lora_steps: int = 800, lora_rank: int = 16,
                               quantize_4bit: bool = True) -> dict:
    """Train one Wan video-DiT LoRA per profile of `kind`, against `video_model`.

    Returns the full nested map {name: {slug: path}}. Resumable: skips a profile
    that already has a LoRA for THIS video model's slug. The image LoRAs are left
    untouched — these are stored separately and tagged with the model slug."""
    spec = _VIDEO_LORA_KINDS[kind]
    slug = _model_slug(video_model)
    _log("\n" + "═" * 60)
    _log(f"  STAGE — {spec['label']} LoRA training  (model: {video_model})")
    _log("═" * 60)
    lora_file = out_dir / spec["file"]
    vmap = {}
    if lora_file.exists():
        try:
            vmap = json.loads(lora_file.read_text()) or {}
        except Exception:
            vmap = {}

    def _save():
        try:
            lora_file.write_text(json.dumps(vmap, indent=2))
        except Exception as e:
            _log(f"    ⚠ could not save {spec['file']}: {e}")

    for i, name in enumerate(names, 1):
        entry = vmap.get(name)
        cur = _video_lora_path(entry, slug)
        if cur and Path(cur).exists():
            _log(f"  [{i}/{len(names)}] {name}: reusing video LoRA for this model")
            continue
        lora_name = f"{spec['prefix']}{name}__{slug}"
        _log(f"  [{i}/{len(names)}] {name}: training video LoRA "
             f"({lora_steps} steps, rank {lora_rank}) — slow on large models…")
        try:
            res = _run_with_spinner(
                f"training {kind} video LoRA '{name}'",
                client.train_lora, name=lora_name, base_model=video_model,
                target="video", quantize_4bit=quantize_4bit,
                steps=lora_steps, rank=lora_rank, **{kind: name},
            )
            path = res.get("path")
            if path:
                if not isinstance(vmap.get(name), dict):
                    vmap[name] = {}
                vmap[name][slug] = path
                _save()
                _log(f"    ✓ video LoRA saved → {path}")
            else:
                _log(f"    ✗ training returned no path: {res}")
        except Exception as e:
            _log(f"    ✗ video LoRA training failed for {name}: {e}")

    _log(f"\n  {spec['label']} LoRAs ready for {slug}")
    return vmap


def stage_video_loras(client: CoderAIClient, video_model: str, out_dir: Path,
                      char_names: list, lora_steps: int = 800, lora_rank: int = 16,
                      quantize_4bit: bool = True) -> dict:
    """Train one Wan video LoRA per fighter against the video model."""
    return _train_profile_video_loras(client, video_model, out_dir, char_names,
                                      "character", lora_steps, lora_rank, quantize_4bit)


def stage_env_video_loras(client: CoderAIClient, video_model: str, out_dir: Path,
                          env_names: list, lora_steps: int = 800, lora_rank: int = 16,
                          quantize_4bit: bool = True) -> dict:
    """Train one Wan video LoRA per environment against the video model."""
    return _train_profile_video_loras(client, video_model, out_dir, env_names,
                                      "environment", lora_steps, lora_rank, quantize_4bit)


def _generate_keyframes(client: CoderAIClient, image_model: str, keyframe_dir: Path,
                        fight_plan: list, outcome_plan: list, consistency: set,
                        lora_map: dict, char_strength: float, keyframe_steps: int,
                        keyframe_size: str, lora_weight: float,
                        env_lora_map: dict = None, env_lora_weight: float = 0.8,
                        kf_cb=None, cancel_check=None):
    """Generate one keyframe still per clip (image model). Saved as PNG keyed by
    the clip's output stem so the render phase can pick them up as init images.
    Resumable: existing PNGs are kept.

    kf_cb(stem, phase, ok) — optional; fired so callers (the web match-render job)
    can show per-image progress. phase is "start" (this keyframe begins) or "end"
    (finished, ok=True/False); a reused/existing PNG fires "end" with ok=True.

    cancel_check() — optional; polled before each keyframe. When it returns true the
    run stops gracefully (keyframes already written are kept)."""
    def _kf(stem, phase, ok=None):
        if kf_cb:
            try:
                kf_cb(stem, phase, ok)
            except Exception:
                pass
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    use_ip = "ipadapter" in consistency or "keyframe" in consistency
    use_lora = "lora" in consistency
    # Each fighter's fixed outfit (lifted from their profile prompt). Stated
    # EXPLICITLY in every keyframe prompt so the image model paints the right
    # clothes instead of drifting — the keyframe then anchors the I2V clip, so
    # this is the strongest lever for wardrobe consistency.
    _out_dir = keyframe_dir.parent.parent
    _outfits = _load_wardrobe(_out_dir)

    # Map match_name -> [f1, f2] so an outcome keyframe attaches BOTH match
    # fighters' LoRAs (+ env), like the clips do — not just the single fighter
    # the scene is named after. Read the saved plan too, so a single-outcome
    # regen (fight_plan == []) can still resolve the pair.
    # Also map match_name -> (env, env_desc) so an outcome keyframe uses the
    # MATCH's current location, not a stale snapshot stored on the outcome entry
    # (the two diverge when the match env is changed in the UI — the match is
    # updated but the outcome entries are not). The match is authoritative.
    _mf_map, _menv_map, _mref_map = {}, {}, {}
    for m in fight_plan:
        _mf_map[m.get("match_name")] = [x for x in (m.get("f1"), m.get("f2")) if x]
        _menv_map[m.get("match_name")] = (m.get("env"), m.get("env_desc"))
        _mref_map[m.get("match_name")] = m.get("referee")
    try:
        _saved = json.loads((Path(keyframe_dir).parent / "prompts.json").read_text())
        for m in _saved.get("fight_plan", []):
            _mf_map.setdefault(m.get("match_name"),
                               [x for x in (m.get("f1"), m.get("f2")) if x])
            _menv_map.setdefault(m.get("match_name"),
                                 (m.get("env"), m.get("env_desc")))
            _mref_map.setdefault(m.get("match_name"), m.get("referee"))
    except Exception:
        pass

    # Flatten all clips into (stem, prompt, fighters, env, env_desc, kf_override)
    # jobs. kf_override is a user-edited keyframe prompt stored on the entry; when
    # present it is used VERBATIM (otherwise the prompt is composed from the base
    # prompt + locked wardrobe + environment).
    jobs = []
    for m in fight_plan:
        for c in m["clips"]:
            # Intro clips appear with only their own participants: a solo entrance
            # shows ONE fighter; the face-off shows both fighters + the referee.
            # Fight clips (and legacy clips with no role) show both fighters.
            _role = c.get("role", "fight")
            if _role in ("entrance", "faceoff"):
                _cf = list(c.get("fighters") or [m["f1"], m["f2"]])
                if _role == "faceoff":
                    _cref = (m.get("referee")
                             or _referee_for(_out_dir, m.get("match_name") or m.get("f1", "")))
                    if _cref and _cref not in _cf:
                        _cf.append(_cref)
            else:
                _cf = [m["f1"], m["f2"]]
            jobs.append((_clip_stem_fight(m["match_name"], c["idx"]),
                         c["prompt"], _cf, m.get("env"),
                         m.get("env_desc"), c.get("kf_prompt")))
    for o in outcome_plan:
        _of = _mf_map.get(o.get("match_name")) or [o["fighter"]]
        if o["fighter"] not in _of:
            _of = [o["fighter"]] + _of
        # Prefer the match's current env over the outcome's stored snapshot.
        _menv, _menvd = _menv_map.get(o.get("match_name"), (None, None))
        _oenv = _menv if _menv is not None else o.get("env")
        _oenvd = _menvd if _menv is not None else o.get("env_desc")
        _base = _clip_stem_outcome(o["fighter"], o["outcome"], o.get("match_name"))
        # The VICTORY shot is officiated by a referee — attach the referee profile
        # (IP-adapter + LoRA, like a fighter) so the official appears consistently
        # raising the winner's arm. The FINISH shot stays fighters-only. Prefer the
        # referee chosen for this match; fall back to any available referee profile.
        _ref = (_mref_map.get(o.get("match_name"))
                or _referee_for(_out_dir, o.get("match_name") or o.get("fighter", "")))
        _victory_roles = {"victory", "draw_decision"}
        # One keyframe PER outcome shot (finish, then victory) so each clip of the
        # outcome video anchors to its own image. Legacy outcomes without `shots`
        # produce the single base keyframe as before.
        _oshots = o.get("shots") or None
        _is_draw = o.get("outcome") == "draw"
        if _oshots:
            for si, s in enumerate(_oshots):
                if s.get("role") in _victory_roles:
                    # Victory clip: only the WINNER appears for a decisive result
                    # (the loser is out of frame); a draw raises BOTH fighters' arms.
                    _jf = list(_of) if _is_draw else [o["fighter"]]
                    if _ref and _ref not in _jf:
                        _jf.append(_ref)
                else:
                    _jf = list(_of)  # finish clip = both fighters
                jobs.append((_outcome_seg_stem(_base, si),
                             s.get("prompt") or o["prompt"], _jf, _oenv, _oenvd,
                             s.get("kf_prompt")))
        else:
            jobs.append((_base, o["prompt"], _of, _oenv, _oenvd, o.get("kf_prompt")))

    _log(f"\n  ── Keyframe phase — {len(jobs)} keyframe image(s) (image model) ──")
    made, skipped, failed = 0, 0, 0
    for k, (stem, prompt, fighters, env, env_desc, kf_override) in enumerate(jobs, 1):
        if cancel_check and cancel_check():
            _log(f"  ⏹ Keyframe generation cancelled by user "
                 f"({made} made, {skipped} reused so far)")
            break
        out_png = keyframe_dir / f"{stem}.png"
        if out_png.exists() and out_png.stat().st_size > 0:
            skipped += 1
            _kf(stem, "end", True)   # already present — show it as done
            continue
        _kf(stem, "start")
        profiles = list(fighters) if use_ip else None
        loras = None
        if use_lora:
            loras = (_lora_specs_for(fighters, lora_map, lora_weight)
                     + _env_lora_specs_for(env, env_lora_map, env_lora_weight)) or None
        # A user-edited keyframe prompt (stored on the entry) wins verbatim;
        # otherwise compose from the base prompt + locked wardrobe (each fighter's
        # fixed, coloured outfit) + the full environment description.
        if kf_override and str(kf_override).strip():
            kf_prompt = str(kf_override)
        else:
            kf_prompt = _compose_kf_prompt(prompt, fighters, env, env_desc, _outfits)
        try:
            img = _run_with_spinner(
                f"keyframe {k}/{len(jobs)} — {stem}",
                client.generate_image,
                prompt=kf_prompt, model=image_model,
                character_profiles=profiles, loras=loras,
                character_strength=char_strength, size=keyframe_size,
                steps=keyframe_steps,
                poll_fn=client.image_progress,
                step_cb=(lambda prog, _s=stem: _kf(_s, "step", prog)),
            )
            out_png.write_bytes(img)
            made += 1
            _kf(stem, "end", True)
        except Exception as e:
            failed += 1
            _kf(stem, "end", False)
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
                 keyframe_steps: int = 28, keyframe_size: str = "832x480",
                 lora_weight: float = 0.85, keyframes_only: bool = False,
                 env_lora_map: dict = None, env_lora_weight: float = 0.8,
                 upscale_factor: int = 0, fps_multiplier: int = 0,
                 video_lora_scale: float = 1.0,
                 clip_min_frames: int = CLIP_MIN_FRAMES,
                 clip_max_frames: int = CLIP_MAX_FRAMES,
                 video_size: str = "832x480",
                 short_min: float = 40.0, short_max: float = 50.0,
                 long_min: float = 65.0, long_max: float = 75.0,
                 single_clip_max_frames: int = SINGLE_CLIP_MAX_FRAMES,
                 outcome_min_frames: int = 96, outcome_max_frames: int = 150,
                 playback_fps: int = 0, upscale_model: str = None,
                 interpolation_model: str = None):
    # PLAYBACK fps decouples the encode/play rate from the model's frame budget:
    # Wan generates a fixed number of frames regardless of fps, so encoding the
    # same frames at a HIGHER rate plays them faster (less slow-motion). Used for
    # the mp4 encode AND the clip-count math, so the finals reach the target
    # length at the real playback speed. 0 = keep the generation fps.
    if playback_fps and int(playback_fps) > 0:
        fps = int(playback_fps)
    _log("\n" + "═" * 60)
    _log("  STAGE 3 — Videos")
    _log("═" * 60)
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    prompts_file = video_dir / "prompts.json"
    keyframe_dir = video_dir / "keyframes"

    # Per-model video LoRA maps (attached to the VIDEO request when present).
    video_lora_map = _load_json_map(out_dir / "video_loras.json")
    env_video_lora_map = _load_json_map(out_dir / "env_video_loras.json")

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
        _stage_videos_render(
            client, video_model, video_dir, fight_plan, outcome_plan,
            total_matches, total_outcomes, fps, clip_delay,
            consistency=consistency, lora_map=lora_map,
            keyframe_dir=keyframe_dir if use_keyframe else None,
            lora_weight=lora_weight,
            env_lora_map=env_lora_map or {}, env_lora_weight=env_lora_weight,
            video_lora_map=video_lora_map, env_video_lora_map=env_video_lora_map,
            video_lora_scale=video_lora_scale, video_size=video_size,
            single_clip_max_frames=single_clip_max_frames)
        _stage_enhance_videos(client, upscale_model, video_dir, fight_plan,
                              outcome_plan, upscale_factor, fps_multiplier,
                              interpolation_model=interpolation_model)
        return

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
    _cf_lo, _cf_hi = _clip_frame_range(clip_min_frames, clip_max_frames)
    # Final-assembly duration targets (seconds). The LONG target drives how many
    # clips the planner produces: it keeps adding clips (each nf/fps seconds of
    # playback) until their summed duration reaches the long target, so the long
    # cut is always filled. The SHORT target is the highlight length.
    _slo, _shi = sorted((float(short_min), float(short_max)))
    _llo, _lhi = sorted((float(long_min), float(long_max)))
    for f1, f2 in pairs:
        short_target = random.uniform(_slo, _shi)
        long_target  = random.uniform(_llo, _lhi)
        env = random.choice(env_names) if env_names else None
        env_desc = _env_description(env) if env else "African township"
        clips_spec, planned, ci = [], 0.0, 0
        # Pre-fight INTRO clips first: a bold solo entrance for EACH fighter, then a
        # stare-down face-off where the referee signals the start. The real fight
        # begins only after these. `fighters` records who appears in the clip so the
        # keyframe + render attach the right profiles (the face-off also gets the
        # referee, resolved at prompt/render time). These clips are extra — they do
        # NOT count against the long_target so the fight still fills the cut.
        for _intro_role, _intro_who in (("entrance", [f1]),
                                        ("entrance", [f2]),
                                        ("faceoff",  [f1, f2])):
            _nf = random.randint(_cf_lo, _cf_hi)
            clips_spec.append({
                "idx": ci, "clip_seconds": round(_nf / max(1, fps), 2),
                "nf": _nf, "intensity": "introduction",
                "role": _intro_role, "fighters": list(_intro_who),
                "shot": None, "prompt": None,
            })
            ci += 1
        while planned < long_target:
            round_num = (ci - 3) // 3 + 1
            intensity = ("early exchanges" if round_num == 1
                         else "midpoint battle" if round_num == 2
                         else "climactic final exchange")
            # Budget frames directly (the model's motion budget, fps-independent),
            # within the configured range. Duration = frames / playback fps.
            _nf = random.randint(_cf_lo, _cf_hi)
            clip_seconds = round(_nf / max(1, fps), 2)
            clips_spec.append({
                "idx": ci, "clip_seconds": clip_seconds,
                "nf": _nf, "role": "fight",
                "intensity": intensity, "shot": None, "prompt": None,
            })
            # Accumulate playback duration so the match reaches long_target.
            planned += _nf / max(1, fps)
            ci += 1
        fight_plan.append({
            "f1": f1, "f2": f2, "env": env, "env_desc": env_desc,
            "match_name": _unique_match_name(f1, f2),
            "short_target": short_target, "long_target": long_target,
            "clips": clips_spec,
        })

    # Outcome-clip plan: per MATCH, the DECISIVE outcomes (win / ko_win / retire)
    # are per-fighter (each fighter can win or lose), but a DRAW concerns BOTH
    # fighters so there is exactly ONE draw per match (not one per fighter).
    # Each outcome is a two-clip video (finish → victory) assembled at render time.
    decisive_outcomes = ["win", "ko_win", "retire"]
    outcome_plan = []
    # Outcome clips budget frames directly (like fight clips) within their own
    # configurable range; the total is split across the finish + victory clips and
    # each is chained the same way at render time. Duration = frames / playback fps.
    _of_lo, _of_hi = _clip_frame_range(outcome_min_frames, outcome_max_frames)

    def _new_outcome(m, fighter, outcome, opponent):
        _onf = random.randint(_of_lo, _of_hi)
        return {
            "match_name": m["match_name"],
            "fighter": fighter, "outcome": outcome, "opponent": opponent,
            "env": m["env"], "env_desc": m["env_desc"],
            "target_s": round(_onf / max(1, fps), 2),
            "nf": _onf,
            "shot": None, "prompt": None, "shots": None,
        }

    for m in fight_plan:
        for fighter in (m["f1"], m["f2"]):
            _opp = m["f2"] if fighter == m["f1"] else m["f1"]
            for outcome in decisive_outcomes:
                outcome_plan.append(_new_outcome(m, fighter, outcome, _opp))
        # Exactly one draw per match (represents both fighters), named after f1.
        outcome_plan.append(_new_outcome(m, m["f1"], "draw", m["f2"]))

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
        # Shuffled technique cycle so consecutive clips emphasise different
        # disciplines (kicks, clinch, ground, submissions…) instead of all punches.
        _focus_cycle = list(FIGHT_ACTION_FOCUS)
        random.shuffle(_focus_cycle)
        _ref = m.get("referee") or _referee_for(out_dir, m.get("match_name") or m["f1"])
        _ffi = 0  # fight-clip index (drives the technique-focus cycle, skips intros)
        for c in m["clips"]:
            _pidx += 1
            _role = c.get("role", "fight")
            _focus = ("" if _role != "fight"
                      else _focus_cycle[_ffi % len(_focus_cycle)])
            if _role == "fight":
                _ffi += 1
            shot = _fill_clip_prompt(prompter, c, m["f1"], m["f2"], m.get("env"),
                                     m["env_desc"], char_descriptions, referee=_ref,
                                     match_avoid=match_avoid, focus=_focus)
            _who = (f"{(c.get('fighters') or [m['f1']])[0]} entrance" if _role == "entrance"
                    else f"{m['f1']} vs {m['f2']} face-off" if _role == "faceoff"
                    else f"{m['f1']} vs {m['f2']}")
            _log(f"  │  [{_pidx}/{_ptot}] {_who} clip{c['idx']:02d}: {shot}")
    for o in outcome_plan:
        _pidx += 1
        _plan_outcome_shots(prompter, o, char_descriptions, o.get("opponent"))
        _roles = " → ".join(s["role"] for s in o.get("shots", []))
        _log(f"  │  [{_pidx}/{_ptot}] {o['fighter']} {o['outcome']} ({_roles}): {o['shot']}")
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

    _stage_videos_render(
        client, video_model, video_dir, fight_plan, outcome_plan,
        total_matches, total_outcomes, fps, clip_delay,
        consistency=consistency, lora_map=lora_map,
        keyframe_dir=keyframe_dir if use_keyframe else None,
        lora_weight=lora_weight,
        env_lora_map=env_lora_map or {}, env_lora_weight=env_lora_weight,
        video_lora_map=video_lora_map, env_video_lora_map=env_video_lora_map,
        video_lora_scale=video_lora_scale, video_size=video_size,
        single_clip_max_frames=single_clip_max_frames)
    _stage_enhance_videos(client, upscale_model, video_dir, fight_plan,
                          outcome_plan, upscale_factor, fps_multiplier,
                          interpolation_model=interpolation_model)


def _stage_videos_render(client, video_model, video_dir, fight_plan, outcome_plan,
                         total_matches, total_outcomes, fps, clip_delay,
                         consistency=None, lora_map=None, keyframe_dir=None,
                         lora_weight=0.85, env_lora_map=None, env_lora_weight=0.8,
                         progress_cb=None, clip_cb=None,
                         video_lora_map=None, env_video_lora_map=None,
                         assemble_finals=True, video_lora_scale=1.0,
                         video_size="832x480",
                         single_clip_max_frames=SINGLE_CLIP_MAX_FRAMES,
                         playback_fps=0, cancel_check=None):
    """PHASE 3 — render ALL videos from pre-written prompts (video model stays loaded).

    progress_cb(done, total, label) — optional; called after each clip finishes so
    callers (e.g. the web match-render job) can surface per-clip advancement.
    clip_cb(gidx, phase, ok) — optional; phase is "start" (clip gidx begins) or
    "end" (finished, ok=True/False). gidx is a 0-based index over the combined
    sequence of fight clips (in plan order) followed by outcome clips.

    LoRAs on the VIDEO request come from the per-model video LoRA maps (matched to
    this video model's slug) — image LoRAs don't apply to a Wan video DiT, so they
    are used only for keyframes, not here.
    """
    # Playback fps override (see stage_videos): encode + duration use the play
    # rate, the model's frame budget is unchanged. 0 = keep the generation fps.
    if playback_fps and int(playback_fps) > 0:
        fps = int(playback_fps)
    _log("\n  ── Phase B — rendering all videos (video model) ──")
    render_start = time.monotonic()
    consistency = consistency or {"prompt"}
    lora_map = lora_map or {}
    env_lora_map = env_lora_map or {}
    video_lora_map = video_lora_map or {}
    env_video_lora_map = env_video_lora_map or {}
    video_slug = _model_slug(video_model)
    use_lora = "lora" in consistency
    # Wan2.2 is trained on 16:9 (canonical 832×480 / 1280×720); square 512 is
    # off-distribution and worsens motion + colour drift. Render at the native
    # aspect; the keyframe (init_image) is generated at the same size so the I2V
    # anchor isn't stretched/letterboxed.
    try:
        _vw, _vh = (int(x) for x in str(video_size).lower().split("x", 1))
    except Exception:
        _vw, _vh = 832, 480

    # Map match_name -> [f1, f2] so an outcome clip (which belongs to a match)
    # can attach BOTH fighters' LoRAs + the environment, not just the single
    # fighter the scene is named after. Read from the saved plan too, so this
    # still resolves when re-rendering outcomes alone (fight_plan == []).
    _mf_map = {}
    _mref_map = {}
    _out_dir = Path(video_dir).parent.parent
    for _m in fight_plan:
        _mf_map[_m.get("match_name")] = [x for x in (_m.get("f1"), _m.get("f2")) if x]
        _mref_map[_m.get("match_name")] = _m.get("referee")
    try:
        _saved = json.loads((Path(video_dir) / "prompts.json").read_text())
        for _m in _saved.get("fight_plan", []):
            _mf_map.setdefault(_m.get("match_name"),
                               [x for x in (_m.get("f1"), _m.get("f2")) if x])
            _mref_map.setdefault(_m.get("match_name"), _m.get("referee"))
    except Exception:
        pass

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

    def _step(prog):
        """Forward a live diffusion-step progress dict for the CURRENT clip to the
        web UI (per-clip step bar). No-op without a clip_cb."""
        if clip_cb and prog:
            try:
                clip_cb(_gidx, "step", prog)
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

    # Max frames per SINGLE model generation. A clip whose budget exceeds this is
    # rendered as several chained sub-renders and concatenated into ONE shot.
    _chunk_max = max(8, min(int(single_clip_max_frames or SINGLE_CLIP_MAX_FRAMES),
                            MODEL_MAX_FRAMES))

    def _render_once(label, prompt, profiles, env, nf, out_path,
                     fighters=None, init_override=None, step_cb=None,
                     cond_frames=None):
        """One model generation → out_path. `init_override` (PNG bytes) wins over
        any keyframe; pass it to chain a sub-render onto the previous one's last
        frame. `cond_frames` (list of PNG byte tails) instead drives VACE 'extend'
        continuation. Returns (ok, duration_or_None, fatal)."""
        init_image = None if cond_frames else init_override
        loras = None
        if use_lora:
            # Video-DiT LoRAs trained for THIS video model (image LoRAs can't apply
            # to a Wan video transformer — they live on the keyframe path instead).
            # `video_lora_scale` dials the character+env LoRA influence DOWN at
            # video time only (keyframe LoRA weight is untouched): stacking several
            # base-trained LoRAs at full weight on a distilled Wan2.2 expert can
            # desaturate/over-smooth the clip, so a scale < 1 trades identity
            # strength for cleaner colour/motion.
            _cw = lora_weight * video_lora_scale
            _ew = env_lora_weight * video_lora_scale
            loras = (_video_lora_specs_for(fighters or profiles or [],
                                           video_lora_map, video_slug, _cw)
                     + _env_video_lora_specs_for(env, env_video_lora_map,
                                                 video_slug, _ew)) or None
        # Rate-limit (429) is transient — the server is just busy — so back off and
        # RETRY the same render instead of abandoning the clip. Only a genuine
        # error (or too many 429s) marks the clip failed.
        _rl_attempts = 0
        _RL_MAX = 40
        while True:
            try:
                mp4 = _run_with_spinner(
                    label, client.generate_video_clip,
                    prompt=prompt, model=video_model,
                    character_profiles=profiles, environment_name=env,
                    num_frames=nf, fps=fps, seed=random.randint(0, 2**31),
                    width=_vw, height=_vh,
                    init_image=init_image, loras=loras, cond_frames=cond_frames,
                    poll_fn=client.video_progress, step_cb=step_cb,
                )
                Path(out_path).write_bytes(mp4)
                return True, (get_video_duration(out_path) or None), False
            except Exception as e:
                if _is_fatal(e):
                    _log(f"    ✗ Fatal: {e}")
                    return False, None, True
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate limit" in err_str.lower()
                if is_rate_limit:
                    _rl_attempts += 1
                    if _rl_attempts > _RL_MAX:
                        _log(f"    ✗ still rate-limited after {_RL_MAX} retries — giving up on this clip")
                        return False, None, False
                    backoff = min(clip_delay * 4, 60)
                    _log(f"    ⏳ rate limited (429) — backing off {backoff:.0f}s and "
                         f"retrying (attempt {_rl_attempts}/{_RL_MAX})")
                    time.sleep(backoff)
                    continue  # retry — do NOT fail the clip on a 429
                backoff = clip_delay * 2
                _log(f"    ✗ failed: {e}  (waiting {backoff:.0f}s)")
                time.sleep(backoff)
                return False, None, False

    def _render(label, prompt, profiles, env, nf, out_path, stem=None, fighters=None,
                step_cb=None, segments=None):
        """Render one CLIP, splitting into chained sub-renders when the budget
        exceeds the single-render cap. The parts are concatenated into out_path as
        one continuous shot and discarded, so callers (and the Matches page) still
        see exactly one file per planned clip. Returns (ok, duration, fatal).

        `segments` (optional) is a list of (prompt, frames[, seg_stem[, seg_fighters]])
        describing a SEQUENCE of distinct shots to render back-to-back into one
        continuous video — e.g. an outcome's finish clip then its victory clip. Each
        segment uses its OWN prompt verbatim (a deliberate new action). When a segment
        carries its own keyframe stem (seg_stem), that segment is ANCHORED to its own
        keyframe image (a clean cut into a new, correctly-composed shot — e.g. the
        referee raising the winner's arm); otherwise it seeds from the previous shot's
        last frame / VACE tail for visual continuity. When a segment carries its own
        `seg_fighters` list, that segment applies exactly those character profiles +
        LoRAs (e.g. the FINISH shows both fighters, while the VICTORY shows only the
        winner + the referee). When omitted, the single (prompt, nf) is rendered as
        before."""
        keyframe = _keyframe_bytes(stem) if stem else None
        # Build a flat list of parts: (segment_prompt, frames, is_segment_start,
        # segment_keyframe_bytes, segment_fighters). The segment keyframe (if any)
        # anchors that segment's first part to its own image instead of chaining from
        # the prior; the segment fighters (if any) override the clip-level profiles +
        # LoRAs for that segment.
        if segments:
            seglist = []
            for seg in segments:
                sp, sn = seg[0], seg[1]
                sstem = seg[2] if len(seg) > 2 else None
                sfighters = seg[3] if len(seg) > 3 else None
                if int(sn) > 0:
                    seglist.append((str(sp), int(sn), sstem, sfighters))
        else:
            seglist = [(prompt, int(nf), stem, None)]
        parts_plan = []
        for sp, sn, sstem, sfighters in seglist:
            seg_kf = _keyframe_bytes(sstem) if sstem else None
            for bi, bn in enumerate(_split_frame_budget(sn, _chunk_max)):
                parts_plan.append((sp, bn, bi == 0, seg_kf, sfighters))
        if len(parts_plan) == 1:
            sp, bn, _, seg_kf, sfighters = parts_plan[0]
            _pf = sfighters if sfighters is not None else profiles
            _ff = sfighters if sfighters is not None else fighters
            return _render_once(label, sp, _pf, env, bn, out_path,
                                fighters=_ff, init_override=(seg_kf or keyframe),
                                step_cb=step_cb)
        # Chained multi-part shot: part 0 starts from the clip keyframe; each later
        # part is seeded by the previous part's last frame → seamless single take.
        # Parts live in a throwaway temp dir (NOT video_dir) so a crash can't leave
        # stray files for _scan_matches to mis-parse; only the concatenated result
        # lands at out_path.
        import shutil as _sh
        # A VACE model continues a chained part from the previous part's FRAME TAIL
        # (real motion → carries velocity forward), the proper fix for the
        # single-frame "boomerang". Non-VACE models fall back to single last-frame
        # seeding + the forward-motion prompt nudge.
        _vace = "vace" in (video_model or "").lower()
        _nparts = len(parts_plan)
        _budget = [p[1] for p in parts_plan]
        _log(f"    ↪ chaining {_nparts} parts {_budget} into one shot"
             + ("  [multi-shot sequence]" if segments and len(seglist) > 1 else "")
             + ("  [VACE frame-tail extend]" if _vace else ""))
        tmpd = tempfile.mkdtemp(prefix="twshot_")
        parts, prev_last, prev_tail = [], None, None
        try:
            for pi, (seg_prompt, pn, seg_start, seg_kf, seg_fighters) in enumerate(parts_plan):
                part_path = os.path.join(tmpd, f"part{pi:02d}.mp4")
                # Tag each part's step updates with part N/total so the UI can show
                # "concatenating shot — part 2/3" alongside the diffusion step.
                _pcb = ((lambda prog, _p=pi + 1, _n=_nparts:
                         step_cb({**(prog or {}), "part": _p, "parts": _n}))
                        if step_cb else None)
                # Seeding for this part:
                #   • segment start WITH its own keyframe → anchor to that keyframe
                #     (a clean cut into a new, correctly-composed shot).
                #   • part 0 (no segment keyframe) → the clip keyframe.
                #   • VACE    → the previous part's frame tail (motion continuation).
                #   • else    → the previous part's last frame (single image) + the
                #               forward-motion prompt nudge to discourage rewinding.
                anchored = bool(seg_start and seg_kf is not None)
                if anchored:
                    seed_img = seg_kf
                    cond_frames = None
                else:
                    seed_img = keyframe if pi == 0 else (prev_last or keyframe)
                    cond_frames = prev_tail if (_vace and pi > 0) else None
                    if cond_frames:
                        seed_img = None  # VACE conditions via the tail, not an init frame
                if pi == 0 or anchored:
                    # First part, or a segment anchored to its own keyframe → render
                    # the shot's own prompt verbatim (a deliberate, freshly-composed
                    # shot, not a continuation).
                    part_prompt = seg_prompt
                elif seg_start:
                    # A new deliberate shot in the sequence — use its own prompt
                    # verbatim, but flow on from the previous shot's last frame.
                    part_prompt = ("Continuing in the same unbroken shot from the "
                                   "previous moment, the scene now moves into: "
                                   + seg_prompt)
                else:
                    part_prompt = (
                        "Continuing seamlessly from the previous moment, the action keeps "
                        "moving FORWARD — new movement that advances the moment. " + seg_prompt)
                # Per-segment profiles + LoRAs (e.g. victory shot = winner + referee)
                # override the clip-level ones for this part when provided.
                _pf = seg_fighters if seg_fighters is not None else profiles
                _ff = seg_fighters if seg_fighters is not None else fighters
                ok, _dur, is_fatal = _render_once(
                    f"{label} [part {pi+1}/{_nparts}, {pn}f]",
                    part_prompt, _pf, env, pn, part_path,
                    fighters=_ff, init_override=seed_img, step_cb=_pcb,
                    cond_frames=cond_frames)
                if not ok:
                    return False, None, is_fatal
                parts.append(part_path)
                # Prepare seeds for the NEXT part.
                prev_last = _last_frame_png(part_path)
                prev_tail = _last_frames_png(part_path, VACE_TAIL_FRAMES) if _vace else None
                if pi < _nparts - 1 and not prev_last and not prev_tail:
                    _log("    ⚠ could not read part's tail — next part falls "
                         "back to the clip keyframe (possible visible seam)")
            # Re-encode the join: stream-copying the parts makes players freeze on
            # each part's first frame for its duration (static-first-half bug). The
            # parts are one continuous shot, so a clean CFR re-encode is correct.
            concat_videos(parts, out_path, reencode=True, fps=fps)
            return True, (get_video_duration(out_path) or None), False
        finally:
            _sh.rmtree(tmpd, ignore_errors=True)

    fatal = False
    cancelled = False
    rendered_clips = 0

    def _is_cancelled():
        try:
            return bool(cancel_check and cancel_check())
        except Exception:
            return False

    # 3a. Fight matches
    for i, m in enumerate(fight_plan):
        if fatal or cancelled:
            break
        elapsed = time.monotonic() - render_start
        _log(f"\n  ┌─ Match {i+1}/{total_matches}: {m['f1']} vs {m['f2']}  "
             f"env={m['env']}  (rendering {elapsed:.0f}s)")
        clips = []
        consecutive_failures = 0
        for c in m["clips"]:
            if fatal:
                break
            if _is_cancelled():
                cancelled = True
                _log("  ⏹ Clip rendering cancelled by user")
                break
            if rendered_clips > 0:
                time.sleep(clip_delay)
            clip_stem = _clip_stem_fight(m['match_name'], c['idx'])
            clip_path = video_dir / f"{clip_stem}.mp4"
            # Frame COUNT is the model's motion budget — it must NOT scale with the
            # playback fps. If it did, a higher fps just spreads the same motion over
            # proportionally more frames: identical playback seconds (still slow) and
            # often past the model's safe length (>~81 → temporal "jumps"). Use the
            # budget baked at prompt time, capped to the model max, and play it at the
            # (higher) encode fps so the SAME motion renders in LESS time = faster,
            # natural speed. Duration = nf / fps.
            # The full planned budget reaches _render, which splits + chains it
            # into ≤single-render-cap parts when it exceeds one model call.
            _nf = int(c.get("nf") or frames_for_seconds(c["clip_seconds"], 8))
            _nf = min(_nf, MAX_PLANNED_FRAMES)
            # Intro clips appear with only their own participants (solo entrance =
            # one fighter; face-off = both fighters + referee) so the right profiles
            # are attached; fight/legacy clips use both fighters.
            _role = c.get("role", "fight")
            if _role in ("entrance", "faceoff"):
                _cf = list(c.get("fighters") or [m["f1"], m["f2"]])
                if _role == "faceoff":
                    _cref = (_mref_map.get(m.get("match_name"))
                             or _referee_for(_out_dir, m.get("match_name") or m.get("f1", "")))
                    if _cref and _cref not in _cf:
                        _cf.append(_cref)
                _clabel = f"{_role} clip {c['idx']:02d} — {'/'.join(_cf)}"
            else:
                _cf = [m["f1"], m["f2"]]
                _clabel = f"clip {c['idx']:02d} — {m['f1']} vs {m['f2']}"
            _log(f"  │  clip {c['idx']:02d}  {c['clip_seconds']:.1f}s → {_nf}f @ {fps}fps "
                 f"= {_nf/max(1,fps):.1f}s")
            _clip("start")
            ok, dur, is_fatal = _render(
                _clabel,
                c["prompt"], _cf, m["env"], _nf, str(clip_path),
                stem=clip_stem, fighters=_cf, step_cb=_step)
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
        # Skipped when only a subset was rendered (e.g. regenerating ONE clip):
        # the `clips` accumulator would hold just that clip and clobber the
        # existing full short/long videos. Use "Reassemble finals" to rebuild
        # them from all clips on disk afterwards.
        if assemble_finals and not cancelled:
            short_clips, short_accum, pos = [], 0.0, 0
            while short_accum < m["short_target"] and clips:
                path, dur = clips[pos % len(clips)]
                short_clips.append((path, dur))
                short_accum += dur
                pos += 1
            _write_concat(short_clips, str(video_dir / f"{m['match_name']}_short.mp4"),
                          f"short (~{m['short_target']:.0f}s)")
            # Long = the whole match in order, padded by repeating from the start
            # if the rendered clips don't sum to the long target, so the final
            # always covers the configured length.
            long_clips = list(clips)
            long_accum, lpos = sum(d for _, d in long_clips), 0
            while long_accum < m["long_target"] and clips:
                path, dur = clips[lpos % len(clips)]
                long_clips.append((path, dur))
                long_accum += dur
                lpos += 1
            _write_concat(long_clips, str(video_dir / f"{m['match_name']}_long.mp4"),
                          f"long  (~{m['long_target']:.0f}s)")
        else:
            _log(f"  └─ (skipping short/long reassembly — subset render)")
        _log(f"  └─ match {i+1}/{total_matches} done  ({len(clips)} clips)")

    # 3b. Per-fighter outcome clips
    _log(f"\n  Outcome clips — {total_outcomes} total")
    for oi, o in enumerate(outcome_plan):
        if fatal:
            _log("  ✗ Aborting remaining outcome clips (fatal error)")
            break
        if cancelled or _is_cancelled():
            cancelled = True
            _log("  ⏹ Outcome rendering cancelled by user")
            break
        if rendered_clips > 0:
            time.sleep(clip_delay)
        clip_name = _clip_stem_outcome(o['fighter'], o['outcome'], o.get('match_name'))
        out_path = str(video_dir / f"{clip_name}.mp4")
        _log(f"\n  [{oi+1}/{total_outcomes}] {clip_name}  ({o['target_s']:.0f}s, env={o['env']})")
        _clip("start")
        # Attach BOTH match fighters' LoRAs (+ env) — an outcome scene still
        # belongs to the match. Fall back to the single fighter for legacy
        # outcomes whose match can't be resolved.
        _ofighters = _mf_map.get(o.get("match_name")) or [o["fighter"]]
        if o["fighter"] not in _ofighters:
            _ofighters = [o["fighter"]] + _ofighters
        _onf = int(o.get("nf") or frames_for_seconds(o["target_s"], 8))
        _onf = min(_onf, MAX_PLANNED_FRAMES)
        # An outcome video is a SEQUENCE of shots (finish → victory). Render them
        # back-to-back into one continuous clip via `segments`; legacy single-prompt
        # outcomes (no shots) fall back to the one-prompt path.
        _oshots = o.get("shots") or None
        _segments = None
        # Referee for this match's victory shot (prefer the chosen one; fall back to
        # any available referee profile), and whether this is the (both-arms) draw.
        _oref = (_mref_map.get(o.get("match_name"))
                 or _referee_for(_out_dir, o.get("match_name") or o.get("fighter", "")))
        _o_is_draw = o.get("outcome") == "draw"
        _victory_roles = {"victory", "draw_decision"}
        if _oshots:
            # Each segment carries its OWN keyframe stem (finish, then victory) so
            # the render anchors every outcome clip to its dedicated image, AND its
            # OWN fighters so the right profiles + LoRAs apply: the FINISH shows both
            # fighters; the VICTORY shows only the winner (+ referee) — or both
            # fighters (+ referee) for a draw.
            _segments = []
            for si, s in enumerate(_oshots):
                _snf = int(s.get("nf") or 0)
                if _snf <= 0:
                    continue
                if s.get("role") in _victory_roles:
                    _sf = list(_ofighters) if _o_is_draw else [o["fighter"]]
                    if _oref and _oref not in _sf:
                        _sf.append(_oref)
                else:
                    _sf = list(_ofighters)
                _segments.append((s.get("prompt") or o["prompt"], _snf,
                                  _outcome_seg_stem(clip_name, si), _sf))
            _onf = sum(n for _, n, _, _ in _segments) or _onf
            _roles = " → ".join(s.get("role", "?") for s in _oshots)
            _log(f"      ({len(_segments)}-shot sequence: {_roles})")
        ok, dur, is_fatal = _render(
            f"{clip_name} outcome clip",
            o["prompt"], [o["fighter"]], o["env"], _onf, out_path,
            stem=clip_name, fighters=_ofighters, step_cb=_step, segments=_segments)
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


def _enhance_targets(video_dir: Path, fight_plan: list, outcome_plan: list) -> list:
    """Existing final (short/long) + outcome video files to post-process."""
    targets = []
    for m in fight_plan:
        mn = m.get("match_name")
        for kind in ("short", "long"):
            p = video_dir / f"{mn}_{kind}.mp4"
            if p.exists():
                targets.append(p)
    for o in outcome_plan:
        p = video_dir / (_clip_stem_outcome(o["fighter"], o["outcome"],
                                             o.get("match_name")) + ".mp4")
        if p.exists() and p not in targets:
            targets.append(p)
    return targets


def _stage_enhance_videos(client, upscale_model, video_dir, fight_plan, outcome_plan,
                          upscale: int = 0, fps_mult: int = 0,
                          progress_cb=None, clip_cb=None,
                          interpolation_model: str = None,
                          final_fps: int = None) -> int:
    """PHASE C — upscale / raise-FPS the final + outcome videos (new files alongside).
    Returns the number of videos enhanced. Callbacks mirror _stage_videos_render."""
    if upscale not in (2, 4) and (not fps_mult or fps_mult <= 1):
        return 0
    targets = _enhance_targets(Path(video_dir), fight_plan, outcome_plan)
    if not targets:
        _log("  (no final/outcome videos found to enhance)")
        return 0
    label = []
    if upscale in (2, 4):
        label.append(f"upscale ×{upscale}")
    if fps_mult and fps_mult > 1:
        label.append(f"FPS ×{fps_mult}")
    _log(f"\n  ── Phase C — enhancing {len(targets)} video(s): {', '.join(label)} ──")
    done = 0
    for i, src in enumerate(targets):
        if clip_cb:
            try: clip_cb(i, "start")
            except Exception: pass
        ok = False
        try:
            def _frame_cb(cur, tot, ips, phase="upscaling", _name=src.name):
                _ips = f" ({ips}/s)" if ips else ""
                _ph = "interpolating" if str(phase).startswith("interp") else "upscaling"
                print(f"\r      🔍 {_name}: {_ph} frame {cur}/{tot}{_ips}    ",
                      end="", flush=True)
            _enhance_video_file(client, upscale_model, src, upscale, fps_mult,
                                progress_cb=_frame_cb,
                                interpolation_model=interpolation_model,
                                final_fps=final_fps)
            print()  # finish the \r progress line
            ok = True
            done += 1
        except Exception as e:
            _log(f"    ✗ enhance failed for {src.name}: {e}")
        if clip_cb:
            try: clip_cb(i, "end", ok)
            except Exception: pass
        if progress_cb:
            try: progress_cb(i + 1, len(targets), src.name)
            except Exception: pass
    _log(f"  └─ enhanced {done}/{len(targets)} video(s)")
    return done


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

# App route roots that appear as server-rendered URLs and JS fetch targets. Used
# to make the UI work behind a reverse-proxy sub-path mount (e.g. /township/).
_MOUNT_ROUTES = ("media", "api", "matches", "match", "characters",
                 "environments", "wardrobe", "prompts", "stream", "stop",
                 "job", "favicon.ico")


def _mount_html(html: str, prefix: str) -> str:
    """Rewrite a server-rendered page so it works under reverse-proxy sub-path
    ``prefix`` (e.g. '/township'). Prepends the prefix to app-route URLs in HTML
    attributes and injects a fetch/EventSource shim so JS calls are prefixed too.
    Idempotent: already-prefixed URLs are not matched again."""
    import re as _re
    if not prefix:
        return html
    routes = "|".join(_MOUNT_ROUTES)
    # 1) Attribute URLs: href/src/action/poster/value/data-* pointing at a route.
    attr_re = _re.compile(
        r'((?:href|src|action|poster|value|data-src|data-url)\s*=\s*["\'])'
        r'(/(?:' + routes + r')\b)')
    html = attr_re.sub(lambda m: m.group(1) + prefix + m.group(2), html)
    # 2) Home/nav link to bare root: href="/" -> href="<prefix>/".
    html = _re.sub(r'(href\s*=\s*(["\']))/\2',
                   lambda m: m.group(1) + prefix + '/' + m.group(2), html)
    # 3) JS shim: prefix root-absolute fetch()/EventSource() URLs at call time.
    if "/*coderai-mount*/" in html:
        return html
    shim = (
        "<script>/*coderai-mount*/(function(){var P=" + repr(prefix) + ";if(!P)return;"
        "window.ROOT_PATH=P;"
        "function fix(u){return (typeof u==='string'&&u.charAt(0)==='/'"
        "&&u.charAt(1)!=='/'&&u.indexOf(P+'/')!==0&&u!==P)?P+u:u;}"
        "var of=window.fetch.bind(window);window.fetch=function(u,o){return of(fix(u),o);};"
        "var OE=window.EventSource;if(OE){var NE=function(u,o){return new OE(fix(u),o);};"
        "NE.prototype=OE.prototype;window.EventSource=NE;}})();</script>"
    )
    if "</head>" in html:
        html = html.replace("</head>", shim + "</head>", 1)
    else:
        html = shim + html
    return html


def launch_web_ui(default_args):
    """Launch a local web interface for Township Fighters content generation.

    Serves on http://localhost:<port> using only the stdlib. The UI has:
      /            — settings form + Start button + live log
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

    # ── Single-worker generation queue ────────────────────────────────────────
    # Every coderai-bound generation job (profile create/regen, LoRA train, match
    # keyframes/clips/outcomes, new match) is funnelled through ONE worker thread so
    # the requests reach the server ONE AT A TIME instead of all at once. coderai
    # loads models on demand, so firing several concurrently makes it thrash AND
    # makes the UI progress flip between jobs. A job is registered as "queued" the
    # instant it is submitted (so the UI immediately shows it was accepted) and the
    # worker flips it to "running" when it actually starts.
    _gen_q: "_queue_mod.Queue" = _queue_mod.Queue()

    def _enqueue_gen(job_id: str, fn, *args, jtype="gen", kind=None, name=None,
                     match=None, scope=None, cancellable=False, **kwargs):
        """Register a job as queued and hand it to the single generation worker.
        The target fn is responsible for (re)initialising the job entry to a
        running state when it actually executes."""
        ahead = _gen_q.unfinished_tasks  # queued + in-progress, before this one
        with _jobs_lock:
            _state["jobs"][job_id] = {
                "status": "queued", "progress": 0, "output": None, "error": None,
                "jtype": jtype, "kind": kind, "name": name, "match": match,
                "scope": scope, "cancellable": cancellable, "cancel": False,
                "_msg": ("⏳ queued — starting…" if ahead <= 0
                         else f"⏳ queued — {ahead} generation(s) ahead…"),
            }
        _gen_q.put((job_id, fn, args, kwargs))
        return job_id

    def _gen_worker():
        while True:
            job_id, fn, args, kwargs = _gen_q.get()
            try:
                # Skip a job cancelled while it was still waiting in the queue.
                with _jobs_lock:
                    j = _state["jobs"].get(job_id)
                    if j and j.get("cancel"):
                        j.update({"status": "done", "cancelled": True,
                                  "progress": 100,
                                  "_msg": "⏹ cancelled before it started"})
                        _gen_q.task_done()
                        continue
                fn(*args, **kwargs)
            except Exception as e:
                with _jobs_lock:
                    if job_id in _state["jobs"]:
                        _state["jobs"][job_id].update(
                            {"status": "error", "error": f"worker error: {e}"})
            finally:
                _gen_q.task_done()

    threading.Thread(target=_gen_worker, daemon=True, name="gen-worker").start()

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

        # Pick the configured upscale model for whichever factor this op uses.
        try:
            _scale = int(param[0]) if op == "upscale_fps" else int(param)
        except Exception:
            _scale = 0
        _um = _upscale_model_for(default_args, _scale) or None
        _im = (getattr(default_args, "interpolation_model", None) or None)

        try:
            video_bytes = fpath.read_bytes()

            if op == "upscale":
                scale = int(param)
                out_path = fpath.parent / f"{stem}_x{scale}{suffix}"
                _set_progress(10, f"Sending to CoderAI for {scale}× upscale…")
                result = client.upscale_video(video_bytes, factor=scale, model=_um)
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
                result = client.interpolate_video(video_bytes, fps_multiplier=mult, model=_im)
                out_path.write_bytes(result)
                _finish(out_path)

            elif op == "upscale_fps":
                scale, target_fps = int(param[0]), int(param[1])
                # Step 1: upscale
                out_up = fpath.parent / f"{stem}_x{scale}_fps{target_fps}{suffix}"
                _set_progress(10, f"Step 1/2 — {scale}× upscale via CoderAI…")
                upscaled = client.upscale_video(video_bytes, factor=scale, model=_um)
                _set_progress(55, f"Step 2/2 — FPS interpolation to {target_fps}fps via CoderAI…")
                # compute multiplier from the source clip's FPS
                probe  = _probe_video(fpath)
                src_fps = probe.get("fps", 8) or 8
                mult = max(2, round(target_fps / src_fps))
                result = client.interpolate_video(upscaled, fps_multiplier=mult, model=_im)
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
            # Honour the Run-page resolution + steps (keyframe_size/keyframe_steps)
            # instead of a hardcoded size/28 steps.
            size, _ref_steps = _ref_gen_res_steps(default_args)

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
                        size=size, steps=_ref_steps, seed=random.randint(0, 2**31),
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

    def _run_create_profile_job(job_id: str, kind: str, name: str, meta: dict,
                                count: int):
        """Create a BRAND-NEW character/environment profile: generate reference
        images server-side, fetch them and save the profile locally (+ CoderAI)."""
        with _jobs_lock:
            _state["jobs"][job_id] = {"status": "running", "progress": 3,
                                      "output": None, "error": None,
                                      "_msg": "starting…", "added": 0,
                                      "kind": kind, "name": name, "jtype": "create"}

        def _prog(pct, msg=""):
            with _jobs_lock:
                _state["jobs"][job_id]["progress"] = pct
                if msg:
                    _state["jobs"][job_id]["_msg"] = msg
            if msg:
                print(f"  [create {name}] {msg}", flush=True)

        def _fail(msg):
            with _jobs_lock:
                _state["jobs"][job_id].update({"status": "error", "error": msg})

        try:
            base = out_dir / (kind + "s") / name
            if base.exists():
                _fail("a profile with that name already exists")
                return
            prompt = (meta.get("prompt") or meta.get("description") or name).strip()
            description = (meta.get("description") or "").strip()
            # A referee is generated as a character but explicitly dressed as a fight
            # official (striped shirt) rather than in fight kit, so the reference
            # images — and every keyframe that uses this profile — read as a referee.
            if _is_referee_meta(meta):
                _ref_look = (f"a fight referee / official wearing {REFEREE_WARDROBE}, "
                             "authoritative posture, NOT a fighter, no fight kit")
                prompt = f"{prompt}, {_ref_look}" if prompt and prompt != name else _ref_look
                if not description:
                    description = "Fight referee / official."
            # Honour the Run-page resolution + steps (keyframe_size/keyframe_steps).
            size, _ref_steps = _ref_gen_res_steps(default_args)
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
            _prog(15, f"generating {count} reference image(s)…")
            try:
                if kind == "character":
                    client.generate_character(name=name, prompt=prompt,
                                              description=description, model=model,
                                              n=count, size=size, steps=_ref_steps)
                else:
                    client.generate_environment(name=name, prompt=prompt,
                                                description=description, model=model,
                                                n=count, size=size, steps=_ref_steps)
            except Exception as e:
                _fail(f"generation failed: {e}")
                return
            _prog(85, "fetching reference images…")
            try:
                images = client.fetch_profile_images(kind, name)
            except Exception as e:
                _fail(f"could not fetch images: {e}")
                return
            if not images:
                _fail("no images were generated")
                return
            full_meta = dict(meta)
            full_meta.update({"name": name, "prompt": prompt,
                              "description": description})
            _save_profile_locally(out_dir, kind, name, full_meta, images)
            with _jobs_lock:
                _state["jobs"][job_id].update({
                    "status": "done", "progress": 100,
                    "added": len(images),
                    "_msg": f"created '{name}' with {len(images)} image(s)",
                })
        except Exception as exc:
            _fail(str(exc))

    def _run_new_match_job(job_id: str, f1: str, f2: str, env: str,
                           prompts_only: bool, referee: str = ""):
        """Create a NEW match (fresh pairing) in prompts.json. prompts_only writes
        the fight-clip + outcome prompts and stops; otherwise it hands off to the
        full end-to-end render (text→image→video) for the new match."""
        with _jobs_lock:
            _state["jobs"][job_id] = {"status": "running", "progress": 3,
                                      "output": None, "error": None,
                                      "_msg": "starting…", "jtype": "match",
                                      "scope": ("prompts" if prompts_only else "full"),
                                      "match": None, "cancel": False,
                                      "cancellable": True}

        def _cancelled():
            with _jobs_lock:
                return bool(_state["jobs"].get(job_id, {}).get("cancel"))

        def _prog(pct, msg=""):
            with _jobs_lock:
                _state["jobs"][job_id]["progress"] = max(2, min(99, int(pct)))
                if msg:
                    _state["jobs"][job_id]["_msg"] = msg
            if msg:
                print(f"  [new-match] {msg}", flush=True)

        def _fail(msg):
            with _jobs_lock:
                _state["jobs"][job_id].update({"status": "error", "error": msg})

        def _done(msg):
            with _jobs_lock:
                _state["jobs"][job_id].update({"status": "done", "progress": 100,
                                               "_msg": msg})

        try:
            import sys as _sys
            _sys.modules[__name__]._log = _patched_log  # stream detail to web log
            vdir = out_dir / "videos"
            vdir.mkdir(parents=True, exist_ok=True)
            pf = vdir / "prompts.json"
            data = {}
            if pf.exists():
                try:
                    data = json.loads(pf.read_text()) or {}
                except Exception:
                    data = {}
            fight_plan = data.get("fight_plan", [])
            outcome_plan = data.get("outcome_plan", [])
            fps = int(getattr(default_args, "fps", 0) or data.get("fps") or 8)

            # Unique match name (same scheme the full run uses).
            used = {m.get("match_name") for m in fight_plan}
            mbase = f"match_{f1}_vs_{f2}"
            name, n = mbase, 2
            while name in used:
                name = f"{mbase}_{n}"
                n += 1

            env_desc = _env_description(env) if env else "African township"
            _slo, _shi = sorted((float(getattr(default_args, "short_min", 40.0)),
                                 float(getattr(default_args, "short_max", 50.0))))
            _llo, _lhi = sorted((float(getattr(default_args, "long_min", 65.0)),
                                 float(getattr(default_args, "long_max", 75.0))))
            m = {"f1": f1, "f2": f2, "env": env or None, "env_desc": env_desc,
                 "match_name": name,
                 "referee": referee or None,
                 "short_target": random.uniform(_slo, _shi),
                 "long_target": random.uniform(_llo, _lhi),
                 "clips": []}
            fight_plan.append(m)

            _of_lo, _of_hi = _clip_frame_range(
                int(getattr(default_args, "outcome_min_frames", 96) or 96),
                int(getattr(default_args, "outcome_max_frames", 150) or 150))

            def _new_outcome(fighter, outcome, opp):
                _onf = random.randint(_of_lo, _of_hi)
                return {"match_name": name, "fighter": fighter, "outcome": outcome,
                        "opponent": opp, "env": m["env"], "env_desc": env_desc,
                        "target_s": round(_onf / max(1, fps), 2), "nf": _onf,
                        "shot": None, "prompt": None, "shots": None}

            for fighter in (f1, f2):
                _opp = f2 if fighter == f1 else f1
                for outcome in ("win", "ko_win", "retire"):
                    outcome_plan.append(_new_outcome(fighter, outcome, _opp))
            outcome_plan.append(_new_outcome(f1, "draw", f2))

            try:
                pf.write_text(json.dumps(
                    {"fight_plan": fight_plan, "outcome_plan": outcome_plan,
                     "fps": data.get("fps") or fps}, indent=2))
            except Exception as e:
                _fail(f"could not save prompts.json: {e}")
                return
            with _jobs_lock:
                _state["jobs"][job_id]["match"] = name

            if not prompts_only:
                # Hand the new (skeleton) match to the end-to-end pipeline, reusing
                # the SAME job id so the UI keeps polling one job. scope "full"
                # re-plans the prompts, regenerates keyframes, renders every clip +
                # outcome and assembles the finals.
                _prog(6, f"created {name} — running full match render…")
                _run_match_job(job_id, "full", {"match": name})
                return

            # ── Prompts-only: write the fight-clip + outcome prompts and stop. ────
            _prog(8, "preparing text model…")
            client = CoderAIClient(default_args.base_url,
                                   getattr(default_args, "api_key", None))
            text_model = None
            if not getattr(default_args, "no_llm", False):
                try:
                    text_model = pick_model(client, "text",
                                            getattr(default_args, "text_model", None))
                except Exception as e:
                    _log(f"  [new-match] no text model ({e}); using template prompts")
                    text_model = getattr(default_args, "text_model", None)
            char_descriptions = _build_char_descriptions(out_dir)
            prompter = PromptGenerator(client, text_model,
                                       char_descriptions=char_descriptions)
            _cf_lo, _cf_hi = _clip_frame_range(
                getattr(default_args, "clip_min_frames", CLIP_MIN_FRAMES),
                getattr(default_args, "clip_max_frames", CLIP_MAX_FRAMES))
            long_target = float(m["long_target"])
            new_clips = _build_match_clip_specs(fps, _cf_lo, _cf_hi, long_target, f1, f2)
            _ref = m.get("referee") or _referee_for(out_dir, name)
            match_avoid = []
            _focus_cycle = list(FIGHT_ACTION_FOCUS)
            random.shuffle(_focus_cycle)
            _ffi = 0
            for i, c in enumerate(new_clips):
                if _cancelled():
                    _fail("cancelled before prompts were complete")
                    return
                _focus = ("" if c.get("role", "fight") != "fight"
                          else _focus_cycle[_ffi % len(_focus_cycle)])
                if c.get("role", "fight") == "fight":
                    _ffi += 1
                _fill_clip_prompt(prompter, c, f1, f2, env, env_desc,
                                  char_descriptions, referee=_ref,
                                  match_avoid=match_avoid, focus=_focus)
                _prog(8 + int(60 * (i + 1) / max(1, len(new_clips))),
                      f"clip {i+1}/{len(new_clips)} prompt written")
            m["clips"] = new_clips
            # Outcome prompts for the new match.
            match_outcomes = [o for o in outcome_plan if o.get("match_name") == name]
            for j, o in enumerate(match_outcomes):
                if _cancelled():
                    _fail("cancelled before prompts were complete")
                    return
                _opp = f2 if o.get("fighter") == f1 else f1
                try:
                    _plan_outcome_shots(prompter, o, char_descriptions, _opp)
                except Exception:
                    pass
                _prog(68 + int(28 * (j + 1) / max(1, len(match_outcomes))),
                      f"outcome {j+1}/{len(match_outcomes)} prompt written")
            try:
                pf.write_text(json.dumps(
                    {"fight_plan": fight_plan, "outcome_plan": outcome_plan,
                     "fps": data.get("fps") or fps}, indent=2))
            except Exception as e:
                _fail(f"could not save prompts.json: {e}")
                return
            _done(f"created match {name}: {len(new_clips)} clip prompt(s) + "
                  f"{len(match_outcomes)} outcome prompt(s) written — now generate "
                  f"keyframes and render from the match page")
        except Exception as exc:
            _fail(str(exc))

    def _run_train_lora_job(job_id: str, kind: str, name: str, steps: int, rank: int,
                            target: str = "image"):
        """Train one profile's identity LoRA (server-side, blocking) while
        polling the server's progress so the profile page shows live step
        counts. target="image" records in loras.json/env_loras.json; target=
        "video" trains a Wan LoRA against the video model and records it (tagged
        with the model slug) in video_loras.json/env_video_loras.json."""
        is_video = (target == "video")
        with _jobs_lock:
            _state["jobs"][job_id] = {"status": "running", "progress": 2,
                                      "output": None, "error": None,
                                      "_msg": "starting…",
                                      "kind": kind, "name": name, "jtype": "train",
                                      "target": target}

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
            _prog(4, f"selecting {'video' if is_video else 'image'} model…")
            if is_video:
                model = getattr(default_args, "video_model", None)
                if not model:
                    try:
                        model = pick_model(client, "video", None)
                    except Exception as e:
                        _fail(f"no video model available: {e}")
                        return
            else:
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

            slug = _model_slug(model)
            if is_video:
                vprefix = "vfighter_" if kind == "character" else "venv_"
                lora_name = f"{vprefix}{name}__{slug}"
            else:
                prefix = "fighter_" if kind == "character" else "env_"
                lora_name = f"{prefix}{name}"
            _web_log(f"  🧠 Training {kind} {'VIDEO ' if is_video else ''}LoRA "
                     f"'{lora_name}' ({steps} steps, rank {rank})"
                     + (f" against {model}" if is_video else "") + "…")

            # Kick the training off ASYNCHRONOUSLY (wait=False): the server runs
            # it independently and we poll its job record. This (a) avoids HTTP
            # read-timeouts on multi-hour video trainings, and (b) survives a
            # township restart — the server keeps training and we re-attach by
            # job_id rather than launching a duplicate. The session token tags the
            # job as ours so recovery never picks up another client's job.
            session = _session_token(out_dir)
            _ACTIVE = ("queued", "preparing", "training", "saving")

            def _kickoff():
                kwargs = dict(name=lora_name, base_model=model,
                              steps=int(steps), rank=int(rank),
                              wait=False, session=session)
                if is_video:
                    kwargs["target"] = "video"
                else:
                    _tbm = getattr(default_args, "lora_train_base_model", None) or None
                    if _tbm:
                        kwargs["train_base_model"] = _tbm
                kwargs[kind] = name
                resp = client.train_lora(**kwargs)
                jid = resp.get("job_id")
                _set_lora_job_id(out_dir, lora_name, jid)
                return jid

            # Re-attach to an existing server job if we have one recorded and the
            # server still knows it (running OR finished while we were away).
            job = _get_lora_job_id(out_dir, lora_name)
            if job:
                pj = client.lora_progress(job=job)
                jstatus = (pj.get("status") or "").strip()
                if jstatus == "done" and pj.get("path"):
                    _web_log(f"  ↻ Re-attached: '{lora_name}' already trained.")
                elif jstatus in _ACTIVE:
                    _web_log(f"  ↻ Re-attached to running job for '{lora_name}'.")
                else:
                    # interrupted / error / unknown → resubmit (server resumes
                    # from its last on-disk checkpoint if one exists).
                    _web_log(f"  ↻ Previous job for '{lora_name}' was "
                             f"'{jstatus or 'lost'}' — resubmitting (resumes from "
                             f"checkpoint)…")
                    job = _kickoff()
            else:
                job = _kickoff()

            _start_ts = time.time()
            _prog(6, "preparing…")
            path = None
            err_msg = None
            _resubmits = 0
            while True:
                time.sleep(1.5)
                _elapsed = int(time.time() - _start_ts)
                _mm, _ss = divmod(_elapsed, 60)
                _et = f"{_mm}m{_ss:02d}s" if _mm else f"{_ss}s"
                try:
                    p = client.lora_progress(job=job)
                except Exception:
                    # Server busy (large video models can load for many minutes,
                    # starving the progress endpoint) — keep the UI visibly alive.
                    _prog(6, f"preparing — loading model… ({_et})")
                    continue
                status = (p.get("status") or "").strip()
                if status == "done":
                    path = p.get("path")
                    break
                if status == "error":
                    err_msg = p.get("message") or "training failed"
                    break
                if status in ("interrupted", "unknown"):
                    # Server restarted (or forgot the job) mid-train. Resubmit
                    # once to resume from checkpoint; give up if it keeps dying.
                    if _resubmits < 2:
                        _resubmits += 1
                        _web_log(f"  ↻ Server job '{status}' — resubmitting "
                                 f"'{lora_name}' (resume #{_resubmits})…")
                        try:
                            job = _kickoff()
                        except Exception as e:
                            err_msg = f"resubmit failed: {e}"
                            break
                        _prog(6, f"resuming after interruption… ({_et})")
                        continue
                    err_msg = f"training {status} and could not be resumed"
                    break
                total = p.get("total") or steps
                step = p.get("step") or 0
                if status == "queued":
                    # Our job is admitted but another training holds the GPU.
                    _prog(4, f"queued — waiting for GPU… ({_et})")
                elif status in ("preparing", "saving") or not step:
                    _prog(6, (p.get("message") or status or "preparing")
                             + f" ({_et})")
                else:
                    pct = 6 + int(90 * step / max(1, total))
                    _prog(pct, p.get("message") or status or "training")

            if err_msg:
                _web_log(f"  ✗ LoRA training failed for {name}: {err_msg}")
                _set_lora_job_id(out_dir, lora_name, None)
                _fail(err_msg)
                return
            if not path:
                _set_lora_job_id(out_dir, lora_name, None)
                _fail("training returned no path")
                return
            # Done — clear the recorded job so a future retrain starts clean.
            _set_lora_job_id(out_dir, lora_name, None)

            # Record the trained LoRA in the on-disk map so video/keyframe runs
            # reuse it. Image LoRAs → loras.json/env_loras.json (flat name→path).
            # Video LoRAs → video_loras.json/env_video_loras.json (nested
            # name→{model_slug: path}), keeping the image maps untouched.
            if is_video:
                map_file = out_dir / ("video_loras.json" if kind == "character"
                                      else "env_video_loras.json")
            else:
                map_file = out_dir / ("loras.json" if kind == "character"
                                      else "env_loras.json")
            try:
                lmap = json.loads(map_file.read_text()) if map_file.exists() else {}
            except Exception:
                lmap = {}
            if is_video:
                if not isinstance(lmap.get(name), dict):
                    lmap[name] = {}
                lmap[name][slug] = path
            else:
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
                                      "scope": scope, "match": params.get("match"),
                                      "cancel": False, "cancellable": True}

        def _cancelled():
            with _jobs_lock:
                return bool(_state["jobs"].get(job_id, {}).get("cancel"))

        def _cancel_done(msg):
            with _jobs_lock:
                j = _state["jobs"][job_id]
                for it in j.get("items", []):
                    if it.get("status") in ("pending", "rendering"):
                        it["status"] = "skipped"
                j.update({"status": "done", "cancelled": True,
                          "progress": 100, "_msg": msg})

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
            with _jobs_lock:
                items = _state["jobs"][job_id].get("items") or []
                if not (0 <= gidx < len(items)):
                    return
                if phase == "step":
                    # `ok` carries the live progress dict (current/total/pct/
                    # it_per_s, plus part/parts when the clip is a chained shot).
                    prog = ok or {}
                    if (prog.get("total") or 0) > 0:
                        items[gidx]["step"] = {
                            "cur": int(prog.get("current") or 0),
                            "tot": int(prog.get("total") or 0),
                            "pct": int(prog.get("pct") or 0),
                            "its": round(float(prog.get("it_per_s") or 0), 2),
                            "phase": prog.get("phase") or "",
                            "part": int(prog.get("part") or 0),
                            "parts": int(prog.get("parts") or 0),
                        }
                    return
                items[gidx]["status"] = (
                    "rendering" if phase == "start" else ("done" if ok else "failed"))
                if phase != "start":
                    items[gidx].pop("step", None)  # clear step bar on completion

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
            # fps is a playback/render choice, not baked content: let the LIVE
            # config win so changing it (e.g. 8→16 to fix slow-motion) applies to
            # re-renders without regenerating prompts. Falls back to the value
            # stored in prompts.json, then 8. The per-clip frame count is
            # recomputed from each clip's stored seconds at this fps in
            # _stage_videos_render, so duration stays constant and motion is
            # rendered at the model's native rate.
            fps = int(getattr(default_args, "fps", 0) or data.get("fps") or 8)
            # Playback-fps override: a higher play rate speeds up the (slow) Wan
            # motion and drives the clip count so finals reach the target length.
            _pfps = int(getattr(default_args, "playback_fps", 0) or 0)
            if _pfps > 0:
                fps = _pfps
            match_name = params.get("match")
            # Per-match playback-fps override (set in the match interface) wins over
            # the global rate, so each match can target its own final play rate.
            _mobj = next((x for x in fight_plan
                          if x.get("match_name") == match_name), None)
            if _mobj and int(_mobj.get("playback_fps") or 0) > 0:
                fps = int(_mobj["playback_fps"])

            # Auto-fix legacy matches: older plans stored a DRAW per fighter, but a
            # draw concerns BOTH fighters so there must be exactly ONE canonical draw
            # per match — owned by f1 with f2 as opponent (it represents both). On ANY
            # operation, collapse duplicates to that single canonical entry and ensure
            # it references both fighters, then persist, so the UI + every regen scope
            # see one draw for both fighters and you can regenerate just that draw.
            if match_name and _mobj:
                _f1, _f2 = _mobj.get("f1"), _mobj.get("f2")
                _mfset = {_f1, _f2} - {None}

                def _is_match_draw(o):
                    return (o.get("outcome") == "draw"
                            and ((o.get("match_name") == match_name) if o.get("match_name")
                                 else (o.get("fighter") in _mfset)))
                _draws = [o for o in outcome_plan if _is_match_draw(o)]
                if _draws:
                    # Prefer an already-f1-owned draw so its rendered files are kept;
                    # only force the owner to f1 when collapsing duplicates (a lone
                    # draw keeps its owner so its files aren't orphaned). Either way
                    # the opponent is the OTHER fighter, so the draw references both.
                    _keep = next((o for o in _draws if o.get("fighter") == _f1), _draws[0])
                    _multi = len(_draws) > 1
                    _owner = _f1 if _multi else _keep.get("fighter")
                    _opp = _f2 if _owner == _f1 else _f1
                    _changed = (_multi
                                or _keep.get("fighter") != _owner
                                or _keep.get("opponent") != _opp
                                or _keep.get("match_name") != match_name)
                    if _changed:
                        _keep["fighter"] = _owner
                        _keep["opponent"] = _opp
                        _keep["match_name"] = match_name
                        outcome_plan = [o for o in outcome_plan
                                        if not (_is_match_draw(o) and o is not _keep)]
                        try:
                            pf.write_text(json.dumps(
                                {"fight_plan": fight_plan, "outcome_plan": outcome_plan,
                                 "fps": data.get("fps") or fps}, indent=2))
                            _log(f"  [match] normalized draw → one draw for BOTH "
                                 f"fighters ({_owner} vs {_opp})"
                                 + (f" — collapsed {len(_draws)} per-fighter draws"
                                    if _multi else ""))
                        except Exception as _e:
                            _log(f"  [match] could not persist draw normalization: {_e}")

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

            # ── Re-plan a match's prompts (text model only; no video model) ────
            # Rebuilds JUST this match's fight-clip list — the count scales with the
            # current playback fps (faster fps → more, shorter clips) — and rewrites
            # each clip's prompt. Other matches, the outcome clips, and any already-
            # rendered clip files are left untouched (hit Re-render afterwards). This
            # is the per-match equivalent of a "Prompts only" run.
            if scope in ("replan", "prompts"):
                m = next((x for x in fight_plan if x.get("match_name") == match_name), None)
                if not m:
                    _fail("match not found in prompts.json")
                    return
                _prog(8, "preparing text model…")
                client = CoderAIClient(default_args.base_url,
                                       getattr(default_args, "api_key", None))
                # Auto-select a text model like a fresh run (config text_model is
                # usually null) so prompts are LLM-generated, not static templates.
                text_model = None
                if not getattr(default_args, "no_llm", False):
                    try:
                        text_model = pick_model(client, "text",
                                                getattr(default_args, "text_model", None))
                    except Exception as e:
                        _log(f"  [replan] no text model ({e}); using template prompts")
                        text_model = getattr(default_args, "text_model", None)
                char_descriptions = _build_char_descriptions(out_dir)
                prompter = PromptGenerator(client, text_model,
                                           char_descriptions=char_descriptions)
                long_target = float(m.get("long_target", 70))
                _cf_lo, _cf_hi = _clip_frame_range(
                    getattr(default_args, "clip_min_frames", CLIP_MIN_FRAMES),
                    getattr(default_args, "clip_max_frames", CLIP_MAX_FRAMES))
                _prog(12, f"re-planning clips for {match_name} @ {fps}fps "
                          f"({_cf_lo}-{_cf_hi}f/clip)…")
                # Rebuild the clip list with the same planner the full run uses:
                # intro clips (entrances + face-off) then fight clips, frame budget
                # within the configured range, match length counted in PLAYBACK
                # seconds (nf/fps).
                new_clips = _build_match_clip_specs(
                    fps, _cf_lo, _cf_hi, long_target, m["f1"], m["f2"])
                _ref = m.get("referee") or _referee_for(out_dir, match_name)
                # Write a fresh, varied prompt for each new clip.
                match_avoid = []
                _focus_cycle = list(FIGHT_ACTION_FOCUS)
                random.shuffle(_focus_cycle)
                _ffi = 0
                for i, c in enumerate(new_clips):
                    if _cancelled():
                        _cancel_done("⏹ cancelled — re-plan stopped (prompts unchanged)")
                        return
                    _focus = ("" if c.get("role", "fight") != "fight"
                              else _focus_cycle[_ffi % len(_focus_cycle)])
                    if c.get("role", "fight") == "fight":
                        _ffi += 1
                    _fill_clip_prompt(prompter, c, m["f1"], m["f2"], m.get("env"),
                                      m["env_desc"], char_descriptions, referee=_ref,
                                      match_avoid=match_avoid, focus=_focus)
                    _prog(12 + int(84 * (i + 1) / max(1, len(new_clips))),
                          f"clip {i+1}/{len(new_clips)} prompt written")
                m["clips"] = new_clips
                # Persist: this match's entry is updated in-place inside fight_plan;
                # everything else (other matches, outcomes, fps) is preserved.
                try:
                    pf.write_text(json.dumps(
                        {"fight_plan": fight_plan, "outcome_plan": outcome_plan,
                         "fps": data.get("fps") or fps}, indent=2))
                except Exception as e:
                    _fail(f"could not save prompts.json: {e}")
                    return
                _done(f"re-planned {len(new_clips)} clip(s) for {match_name} "
                      f"@ {fps}fps — now REGENERATE KEYFRAMES (the old ones match the "
                      f"previous prompts), THEN Re-render")
                return

            # ── Re-write ONLY one clip's prompt (text model) ───────────────────
            # Rewrites a single clip's prompt in place (role-aware: entrance /
            # face-off / fight), preserving the clip's frame budget and position.
            # Nothing is rendered — afterwards regenerate that clip's keyframe and
            # re-render it.
            if scope == "clip-prompt":
                m = next((x for x in fight_plan if x.get("match_name") == match_name), None)
                if not m:
                    _fail("match not found in prompts.json")
                    return
                try:
                    idx = int(params.get("idx"))
                except (TypeError, ValueError):
                    _fail("clip index required")
                    return
                c = next((x for x in m.get("clips", []) if int(x["idx"]) == idx), None)
                if not c:
                    _fail("clip not found in prompts.json")
                    return
                _prog(8, "preparing text model…")
                client = CoderAIClient(default_args.base_url,
                                       getattr(default_args, "api_key", None))
                text_model = None
                if not getattr(default_args, "no_llm", False):
                    try:
                        text_model = pick_model(client, "text",
                                                getattr(default_args, "text_model", None))
                    except Exception as e:
                        _log(f"  [clip-prompt] no text model ({e}); using template prompts")
                        text_model = getattr(default_args, "text_model", None)
                char_descriptions = _build_char_descriptions(out_dir)
                prompter = PromptGenerator(client, text_model,
                                           char_descriptions=char_descriptions)
                _ref = m.get("referee") or _referee_for(out_dir, match_name)
                # Steer a fight-clip rewrite away from this match's OTHER clip shots
                # so the new prompt stays distinct from its neighbours.
                _avoid = [str(x.get("shot"))[:60] for x in m.get("clips", [])
                          if x is not c and x.get("shot")]
                _focus = (random.choice(FIGHT_ACTION_FOCUS)
                          if c.get("role", "fight") == "fight" else "")
                _prog(40, f"rewriting clip {idx:02d} prompt…")
                if _cancelled():
                    _cancel_done("⏹ cancelled — clip prompt unchanged")
                    return
                _fill_clip_prompt(prompter, c, m["f1"], m["f2"], m.get("env"),
                                  m["env_desc"], char_descriptions, referee=_ref,
                                  match_avoid=_avoid, focus=_focus)
                try:
                    pf.write_text(json.dumps(
                        {"fight_plan": fight_plan, "outcome_plan": outcome_plan,
                         "fps": data.get("fps") or fps}, indent=2))
                except Exception as e:
                    _fail(f"could not save prompts.json: {e}")
                    return
                _done(f"rewrote clip {idx:02d} prompt for {match_name} — now "
                      f"regenerate its keyframe (kf↻), then re-render it")
                return

            # ── Re-plan ONLY this match's outcome prompts (text model) ──────────
            # Rewrites the finish + victory shots for every outcome of the match,
            # leaving the fight-clip prompts untouched.
            if scope == "replan-outcomes":
                m = next((x for x in fight_plan if x.get("match_name") == match_name), None)
                if not m:
                    _fail("match not found in prompts.json")
                    return
                _prog(8, "preparing text model…")
                client = CoderAIClient(default_args.base_url,
                                       getattr(default_args, "api_key", None))
                text_model = None
                if not getattr(default_args, "no_llm", False):
                    try:
                        text_model = pick_model(client, "text",
                                                getattr(default_args, "text_model", None))
                    except Exception as e:
                        _log(f"  [replan-outcomes] no text model ({e}); using templates")
                        text_model = getattr(default_args, "text_model", None)
                char_descriptions = _build_char_descriptions(out_dir)
                prompter = PromptGenerator(client, text_model,
                                           char_descriptions=char_descriptions)
                _mf = {m.get("f1"), m.get("f2")} - {None}
                # A draw concerns both fighters → keep exactly one (fixes legacy
                # plans that stored a draw per fighter).
                outcome_plan = _dedupe_match_draws(outcome_plan, match_name, _mf)

                def _belongs(o):
                    if o.get("match_name"):
                        return o.get("match_name") == match_name
                    return o.get("fighter") in _mf
                match_outcomes = [o for o in outcome_plan if _belongs(o)]
                if not match_outcomes:
                    _fail("no outcomes found for this match")
                    return
                for j, o in enumerate(match_outcomes):
                    if _cancelled():
                        _cancel_done("⏹ cancelled — outcome prompts unchanged")
                        return
                    # Force the match's current location so outcome keyframes + clips
                    # stay in one consistent setting.
                    if m.get("env"):
                        o["env"] = m.get("env")
                        o["env_desc"] = m.get("env_desc", o.get("env_desc"))
                    _opp = m.get("f2") if o.get("fighter") == m.get("f1") else m.get("f1")
                    try:
                        _plan_outcome_shots(prompter, o, char_descriptions, _opp)
                    except Exception as e:
                        _log(f"  [replan-outcomes] {o.get('fighter')} {o.get('outcome')}: {e}")
                    _prog(8 + int(88 * (j + 1) / max(1, len(match_outcomes))),
                          f"outcome {j+1}/{len(match_outcomes)} prompt written")
                try:
                    pf.write_text(json.dumps(
                        {"fight_plan": fight_plan, "outcome_plan": outcome_plan,
                         "fps": data.get("fps") or fps}, indent=2))
                except Exception as e:
                    _fail(f"could not save prompts.json: {e}")
                    return
                _done(f"re-planned {len(match_outcomes)} outcome prompt(s) for "
                      f"{match_name} — now REGENERATE the outcome KEYFRAMES, THEN "
                      f"re-render the outcomes")
                return

            # ── Re-write ONLY one outcome's prompts (text model) ───────────────
            # Rewrites a single outcome's finish + victory shots in place, leaving
            # the other outcomes and the fight clips untouched. Nothing is rendered —
            # afterwards regenerate that outcome's keyframes and re-render it.
            if scope == "outcome-prompt":
                m = next((x for x in fight_plan if x.get("match_name") == match_name), None)
                if not m:
                    _fail("match not found in prompts.json")
                    return
                fr, oc = params.get("fighter"), params.get("outcome")
                if not fr or not oc:
                    _fail("fighter and outcome required")
                    return
                o = next((x for x in outcome_plan
                          if x.get("fighter") == fr and x.get("outcome") == oc
                          and (x.get("match_name") in (None, match_name))), None)
                if not o:
                    _fail("outcome not found in prompts.json")
                    return
                _prog(8, "preparing text model…")
                client = CoderAIClient(default_args.base_url,
                                       getattr(default_args, "api_key", None))
                text_model = None
                if not getattr(default_args, "no_llm", False):
                    try:
                        text_model = pick_model(client, "text",
                                                getattr(default_args, "text_model", None))
                    except Exception as e:
                        _log(f"  [outcome-prompt] no text model ({e}); using templates")
                        text_model = getattr(default_args, "text_model", None)
                char_descriptions = _build_char_descriptions(out_dir)
                prompter = PromptGenerator(client, text_model,
                                           char_descriptions=char_descriptions)
                if _cancelled():
                    _cancel_done("⏹ cancelled — outcome prompt unchanged")
                    return
                # Keep the outcome in the match's current location.
                if m.get("env"):
                    o["env"] = m.get("env")
                    o["env_desc"] = m.get("env_desc", o.get("env_desc"))
                _opp = m.get("f2") if o.get("fighter") == m.get("f1") else m.get("f1")
                # A draw belongs to BOTH fighters — describe it that way in the log.
                _olabel = (f"{m.get('f1')} vs {m.get('f2')} draw" if oc == "draw"
                           else f"{fr} {oc}")
                _prog(40, f"rewriting {_olabel} prompts…")
                try:
                    _plan_outcome_shots(prompter, o, char_descriptions, _opp)
                except Exception as e:
                    _fail(f"could not rewrite outcome prompts: {e}")
                    return
                try:
                    pf.write_text(json.dumps(
                        {"fight_plan": fight_plan, "outcome_plan": outcome_plan,
                         "fps": data.get("fps") or fps}, indent=2))
                except Exception as e:
                    _fail(f"could not save prompts.json: {e}")
                    return
                _done(f"rewrote {_olabel} outcome prompts for {match_name} — now "
                      f"regenerate its keyframes (kf↻), then re-render it")
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
            video_lora_map = _load_map("video_loras.json")
            env_video_lora_map = _load_map("env_video_loras.json")
            keyframe_dir = vdir / "keyframes" if "keyframe" in consistency else None
            clip_delay = float(getattr(default_args, "clip_delay", 5.0))
            lw = float(getattr(default_args, "lora_weight", 0.85))
            elw = float(getattr(default_args, "env_lora_weight", 0.8))
            vls = float(getattr(default_args, "video_lora_scale", 1.0))
            vsz = str(getattr(default_args, "video_size", "832x480") or "832x480")
            scm = int(getattr(default_args, "single_clip_max_frames", SINGLE_CLIP_MAX_FRAMES)
                      or SINGLE_CLIP_MAX_FRAMES)

            # ── Full match regeneration (text → image → video, end to end) ─────
            # One click rebuilds EVERYTHING for this match in order: re-plan the
            # fight-clip + outcome prompts, regenerate all keyframes, then re-render
            # all clips + outcomes and reassemble the finals. Other matches are left
            # untouched. Uses the text, image AND video models, so it's the slowest
            # action — but it's the "just redo this whole match" button.
            if scope == "full":
                m = next((x for x in fight_plan if x.get("match_name") == match_name), None)
                if not m:
                    _fail("match not found in prompts.json — render it from the Run page first")
                    return
                image_model = getattr(default_args, "image_model", None)
                if not image_model:
                    try:
                        image_model = pick_model(client, "image", None)
                    except Exception as e:
                        _fail(f"no image model available: {e}")
                        return
                # Auto-select a text model exactly like a fresh run (config's
                # text_model is usually null) so prompts are regenerated by the LLM,
                # not the static fallback templates. Honour the no_llm toggle.
                text_model = None
                if not getattr(default_args, "no_llm", False):
                    try:
                        text_model = pick_model(client, "text",
                                                getattr(default_args, "text_model", None))
                    except Exception as e:
                        _log(f"  [full] no text model ({e}); using template prompts")
                        text_model = getattr(default_args, "text_model", None)
                char_descriptions = _build_char_descriptions(out_dir)
                _mf = {m.get("f1"), m.get("f2")} - {None}
                # A draw concerns both fighters → keep exactly one (fixes legacy
                # plans that stored a draw per fighter).
                outcome_plan = _dedupe_match_draws(outcome_plan, match_name, _mf)

                def _belongs(o):
                    if o.get("match_name"):
                        return o.get("match_name") == match_name
                    return o.get("fighter") in _mf  # legacy per-fighter outcome

                # Which outcomes belong to this match (stems are stable across the
                # re-plan, so this is computed once, up front).
                match_outcomes = [o for o in outcome_plan if _belongs(o)]

                # ── Clean slate — remove this match's existing keyframes + clips ──
                # A whole-match regen starts fresh: delete this match's keyframes,
                # clip videos, outcome videos and assembled finals first, so a
                # changed clip count can't leave orphaned files behind (e.g. a stale
                # clip that _reassemble_finals would glob into the finals).
                _prog(2, "removing this match's existing keyframes + clips…")
                _kfdir = vdir / "keyframes"
                _removed = 0
                _victims = list(vdir.glob(f"{match_name}_clip*.mp4"))
                _victims += list(_kfdir.glob(f"{match_name}_clip*.png"))
                _victims += [vdir / f"{match_name}_short.mp4",
                             vdir / f"{match_name}_long.mp4"]
                for o in match_outcomes:
                    _stem = _clip_stem_outcome(o["fighter"], o["outcome"], o.get("match_name"))
                    _victims += [vdir / f"{_stem}.mp4", _kfdir / f"{_stem}.png"]
                for _p in _victims:
                    try:
                        if _p.exists():
                            _p.unlink()
                            _removed += 1
                    except Exception:
                        pass
                _log(f"  [full] cleared {_removed} existing file(s) for {match_name}")

                # ── Phase 1/4 — re-plan prompts (fight clips + this match's outcomes)
                _prog(4, "phase 1/4 — re-planning prompts…")
                prompter = PromptGenerator(client, text_model,
                                           char_descriptions=char_descriptions)
                long_target = float(m.get("long_target", 70))
                _cf_lo, _cf_hi = _clip_frame_range(
                    getattr(default_args, "clip_min_frames", CLIP_MIN_FRAMES),
                    getattr(default_args, "clip_max_frames", CLIP_MAX_FRAMES))
                new_clips = _build_match_clip_specs(
                    fps, _cf_lo, _cf_hi, long_target, m["f1"], m["f2"])
                _ref = m.get("referee") or _referee_for(out_dir, match_name)
                match_avoid = []
                _focus_cycle = list(FIGHT_ACTION_FOCUS)
                random.shuffle(_focus_cycle)
                _ffi = 0
                for i, c in enumerate(new_clips):
                    if _cancelled():
                        _cancel_done("⏹ cancelled during prompt planning — nothing rendered")
                        return
                    _focus = ("" if c.get("role", "fight") != "fight"
                              else _focus_cycle[_ffi % len(_focus_cycle)])
                    if c.get("role", "fight") == "fight":
                        _ffi += 1
                    _fill_clip_prompt(prompter, c, m["f1"], m["f2"], m.get("env"),
                                      m["env_desc"], char_descriptions, referee=_ref,
                                      match_avoid=match_avoid, focus=_focus)
                    _prog(4 + int(14 * (i + 1) / max(1, len(new_clips))),
                          f"clip {i+1}/{len(new_clips)} prompt written")
                m["clips"] = new_clips
                # Re-plan this match's outcome prompts too, forcing the match's
                # location so keyframes + clips stay in one consistent setting.
                for o in match_outcomes:
                    if m.get("env"):
                        o["env"] = m.get("env")
                        o["env_desc"] = m.get("env_desc", o.get("env_desc"))
                    _opp = m.get("f2") if o.get("fighter") == m.get("f1") else m.get("f1")
                    try:
                        _plan_outcome_shots(prompter, o, char_descriptions, _opp)
                    except Exception:
                        pass
                try:
                    pf.write_text(json.dumps(
                        {"fight_plan": fight_plan, "outcome_plan": outcome_plan,
                         "fps": data.get("fps") or fps}, indent=2))
                except Exception as e:
                    _fail(f"could not save prompts.json: {e}")
                    return

                # ── Phase 2/4 — regenerate every keyframe (clips + outcomes) ──────
                kdir = vdir / "keyframes"
                mm = dict(m)
                kf_stems = [_clip_stem_fight(match_name, c["idx"]) for c in mm["clips"]]
                kf_stems += [s for o in match_outcomes for s in _outcome_kf_stems(o)]
                _set_items([f"keyframe {s}" for s in kf_stems])
                _kf_idx = {s: i for i, s in enumerate(kf_stems)}
                _kf_done = [0]
                for s in kf_stems:
                    try:
                        (kdir / f"{s}.png").unlink()
                    except Exception:
                        pass

                def _kf_cb(stem, phase, ok=None):
                    i = _kf_idx.get(stem)
                    if i is None:
                        return
                    _item(i, phase, ok)
                    if phase == "end":
                        _kf_done[0] += 1
                        _prog(20 + int(28 * _kf_done[0] / max(1, len(kf_stems))),
                              f"keyframe {_kf_done[0]}/{len(kf_stems)} done")

                _prog(20, f"phase 2/4 — regenerating {len(kf_stems)} keyframe(s)…")
                try:
                    _generate_keyframes(
                        client, image_model, kdir, [mm], match_outcomes,
                        consistency | {"keyframe"}, lora_map,
                        float(getattr(default_args, "character_strength", 0.7)),
                        int(getattr(default_args, "keyframe_steps", 28)),
                        getattr(default_args, "keyframe_size", "832x480"), lw,
                        env_lora_map=env_lora_map, env_lora_weight=elw,
                        kf_cb=_kf_cb, cancel_check=_cancelled)
                except Exception as e:
                    _fail(f"keyframe regeneration failed: {e}")
                    return
                for i, s in enumerate(kf_stems):
                    _item(i, "end", (kdir / f"{s}.png").exists())
                if _cancelled():
                    _cancel_done("⏹ cancelled after keyframes — clips not rendered")
                    return

                # ── Phase 3/4 — render all clips + outcomes (NO assembly yet) ────
                # assemble_finals=False so the short/long finals are NOT built here:
                # the user wants assembly as the explicit LAST step, after outcomes.
                # (_stage_videos_render renders clips then outcomes in this one call.)
                _set_items([f"clip {int(c['idx']):02d}" for c in mm["clips"]]
                           + [_clip_stem_outcome(o["fighter"], o["outcome"], o.get("match_name"))
                              for o in match_outcomes])

                def _full_cb(done, total, label):
                    _prog(52 + int(40 * done / max(1, total)),
                          f"render {done}/{total} done" + (f" — {label}" if label else ""))

                _prog(52, f"phase 3/4 — rendering {len(mm['clips'])} clip(s) "
                          f"+ {len(match_outcomes)} outcome(s)…")
                _stage_videos_render(
                    client, video_model, vdir, [mm], match_outcomes,
                    1, len(match_outcomes), fps, clip_delay,
                    consistency=consistency, lora_map=lora_map,
                    keyframe_dir=keyframe_dir, lora_weight=lw,
                    env_lora_map=env_lora_map, env_lora_weight=elw,
                    progress_cb=_full_cb, clip_cb=_item,
                    video_lora_map=video_lora_map, env_video_lora_map=env_video_lora_map,
                    assemble_finals=False, video_lora_scale=vls, video_size=vsz,
                    single_clip_max_frames=scm, cancel_check=_cancelled)
                if _cancelled():
                    _cancel_done("⏹ cancelled during render — finals not assembled")
                    return

                # ── Optional finalize/package flags (driven by the Run/match UI) ──
                # "package"  → enhance (2× upscale + 2× fps) + arbitrage-safe odds +
                #              renamed upload ZIP (videos picked from their highest-
                #              quality variant).  "upload" → also push to the
                #              configured Township endpoint (implies package).
                # When the caller doesn't pass "package" (e.g. CLI/legacy), default
                # to True so the historical always-enhance behaviour is preserved.
                def _truthy(v):
                    return str(v).strip().lower() in ("1", "true", "on", "yes")
                _want_upload = _truthy(params.get("upload", ""))
                _pkg_raw = params.get("package", None)
                _want_pkg = _want_upload or (
                    _truthy(_pkg_raw) if _pkg_raw is not None else True)

                # ── Phase 4 — assemble the final short/long videos (always) ──────
                _prog(86, "phase 4 — assembling final videos…")
                st = float(m.get("short_target", 45))
                lt = float(m.get("long_target", 70))
                n_assembled = _reassemble_finals(vdir, match_name, st, lt)

                # ── Phase 5 — quality pass: 2× AI upscale + 2× frame interpolation
                # of the final short/long assemblies and the outcome videos (the
                # per-clip fight videos are left at base res).  Gated by "package".
                _enh = 0
                if _want_pkg:
                    try:
                        _prog(90, "phase 5 — upscaling ×2 + interpolating ×2 finals + outcomes…")
                        _um = _upscale_model_for(default_args, 2) or None
                        _im = (getattr(default_args, "interpolation_model", None) or None)
                        _enh = _stage_enhance_videos(
                            client, _um, vdir, [mm], match_outcomes,
                            upscale=2, fps_mult=2, interpolation_model=_im)
                    except Exception as e:
                        _log(f"  ⚠ enhance pass failed (finals left at base res): {e}")

                # ── Phase 6 — odds + renamed upload ZIP (gated by "package") ──────
                _pkg_msg = ""
                if _want_pkg:
                    try:
                        _prog(96, "phase 6 — generating odds + packing upload ZIP…")
                        _pz = prepare_match_odds_zip(out_dir, match_name, default_args,
                                                     log=_log)
                        if _pz.get("ok"):
                            _pkg_msg = ", odds + ZIP ready"
                        elif _pz.get("missing"):
                            _pkg_msg = (", ZIP missing "
                                        f"{len(_pz['missing'])} video(s)")
                        elif _pz.get("error"):
                            _pkg_msg = f", packaging issue: {_pz['error']}"
                    except Exception as e:
                        _log(f"  ⚠ odds/ZIP packaging failed: {e}")
                        _pkg_msg = f", packaging failed: {e}"

                # ── Phase 7 — upload to the configured endpoint (gated by "upload")
                _up_msg = ""
                if _want_upload:
                    try:
                        _prog(98, "phase 7 — uploading to Township endpoint…")

                        def _ucb(frac, label, _p=_prog):
                            try:
                                _p(98 + int(2 * max(0.0, min(1.0, frac))), label)
                            except Exception:
                                pass

                        _ur = upload_prepared_match(out_dir, match_name, default_args,
                                                    log=_log, progress_cb=_ucb)
                        if _ur.get("ok"):
                            _up_msg = f", uploaded → match #{_ur.get('match_number')}"
                        else:
                            _up_msg = f", upload skipped: {_ur.get('error')}"
                    except Exception as e:
                        _log(f"  ⚠ upload failed: {e}")
                        _up_msg = f", upload failed: {e}"

                _done(f"regenerated match {match_name} end to end — "
                      f"{len(mm['clips'])} clip(s), {len(match_outcomes)} outcome(s), "
                      f"finals assembled from {n_assembled} clip(s)"
                      + (f", {_enh} video(s) upscaled ×2 + interpolated ×2"
                         if _want_pkg else "")
                      + _pkg_msg + _up_msg)
                return

            # ── Regenerate keyframes (image model) ─────────────────────────────
            # Deletes the targeted keyframe PNG(s) then regenerates them so a
            # subsequent clip re-render uses fresh keyframes (e.g. after a LoRA
            # retrain or profile edit). Does NOT re-render the video itself —
            # click "Re-render" afterwards to rebuild the clip from the new
            # keyframe. scope "keyframes" = whole match; "keyframe" = one clip
            # (idx) or one outcome (fighter+outcome).
            if scope in ("keyframes", "keyframes-missing", "keyframe"):
                # "keyframes-missing" only fills in keyframes that don't exist yet
                # (no delete); "keyframes"/"keyframe" delete then regenerate.
                missing_only = (scope == "keyframes-missing")
                image_model = getattr(default_args, "image_model", None)
                if not image_model:
                    try:
                        image_model = pick_model(client, "image", None)
                    except Exception as e:
                        _fail(f"no image model available: {e}")
                        return
                kdir = vdir / "keyframes"
                m = next((x for x in fight_plan
                          if x.get("match_name") == match_name), None)
                # Build the (filtered) fight/outcome plans + the list of stems to
                # (re)make. _generate_keyframes keeps existing PNGs, so deleting a
                # stem first forces a remake; leaving it forces a fill-in.
                fp, op, stems = [], [], []
                if scope == "keyframe" and params.get("fighter") and params.get("outcome"):
                    fr, oc = params.get("fighter"), params.get("outcome")
                    o = next((x for x in outcome_plan
                              if x.get("fighter") == fr and x.get("outcome") == oc
                              and (x.get("match_name") in (None, match_name))), None)
                    if not o:
                        _fail("outcome not found in prompts.json")
                        return
                    op = [o]
                    # Regenerate every segment keyframe for the outcome (finish +
                    # victory), or just one when a `seg` index is given. _generate_
                    # keyframes skips PNGs that already exist, so deleting only the
                    # targeted stem(s) below limits the remake to them.
                    _all_stems = _outcome_kf_stems(o)
                    _seg = params.get("seg")
                    if _seg is not None and str(_seg) != "":
                        try:
                            _si = int(_seg)
                        except (TypeError, ValueError):
                            _si = -1
                        stems = ([_all_stems[_si]] if 0 <= _si < len(_all_stems)
                                 else _all_stems)
                    else:
                        stems = _all_stems
                else:
                    if not m:
                        _fail("match not found in prompts.json — render it first")
                        return
                    mm = dict(m)
                    if scope == "keyframe":
                        idx = int(params.get("idx"))
                        mm["clips"] = [c for c in m["clips"] if int(c["idx"]) == idx]
                        if not mm["clips"]:
                            _fail("clip not found in prompts.json")
                            return
                    fp = [mm]
                    stems = [_clip_stem_fight(match_name, c["idx"]) for c in mm["clips"]]
                    # Whole-match scopes also cover this match's OUTCOME keyframes —
                    # not just the clips. Per-clip regen (scope == "keyframe" with
                    # idx) stays clip-only.
                    if scope in ("keyframes", "keyframes-missing"):
                        _mf = {mm.get("f1"), mm.get("f2")} - {None}
                        for o in outcome_plan:
                            if o.get("match_name"):
                                if o.get("match_name") != match_name:
                                    continue
                            elif o.get("fighter") not in _mf:
                                continue
                            op.append(o)
                            stems.extend(_outcome_kf_stems(o))
                # In missing-only mode, narrow progress items to the absent ones.
                work = ([s for s in stems if not (kdir / f"{s}.png").exists()]
                        if missing_only else stems)
                if missing_only and not work:
                    _done("no missing keyframes — all present")
                    return
                _set_items([f"keyframe {s}" for s in work])
                _kf_idx = {s: i for i, s in enumerate(work)}
                # Delete the targeted PNGs first so they're actually regenerated
                # (missing-only keeps existing ones → reported done by the callback).
                if not missing_only:
                    for s in work:
                        try:
                            (kdir / f"{s}.png").unlink()
                        except Exception:
                            pass
                # Per-image progress: _generate_keyframes fires kf_cb as each
                # keyframe starts/finishes, so the bars advance image-by-image
                # instead of all flipping at the end.
                _kf_done = [0]

                def _kf_cb(stem, phase, ok=None):
                    i = _kf_idx.get(stem)
                    if i is None:
                        return
                    _item(i, phase, ok)
                    if phase == "end":
                        _kf_done[0] += 1
                        _prog(10 + int(88 * _kf_done[0] / max(1, len(work))),
                              f"keyframe {_kf_done[0]}/{len(work)} done")

                _prog(10, ("filling in {n} missing keyframe(s)…" if missing_only
                           else "regenerating {n} keyframe(s)…").format(n=len(work)))
                try:
                    _generate_keyframes(
                        client, image_model, kdir, fp, op,
                        consistency | {"keyframe"}, lora_map,
                        float(getattr(default_args, "character_strength", 0.7)),
                        int(getattr(default_args, "keyframe_steps", 28)),
                        getattr(default_args, "keyframe_size", "832x480"), lw,
                        env_lora_map=env_lora_map, env_lora_weight=elw,
                        kf_cb=_kf_cb, cancel_check=_cancelled)
                except Exception as e:
                    _fail(f"keyframe regeneration failed: {e}")
                    return
                # Safety net: resolve any item the callback didn't (e.g. a stem
                # _generate_keyframes never visited) by whether its PNG now exists.
                for i, s in enumerate(work):
                    _item(i, "end", (kdir / f"{s}.png").exists())
                made = sum(1 for s in work if (kdir / f"{s}.png").exists())
                if _cancelled():
                    _cancel_done(f"⏹ cancelled — {made}/{len(work)} keyframe(s) done "
                                 f"before stopping")
                    return
                _done(f"{'generated' if missing_only else 'regenerated'} "
                      f"{made}/{len(work)} keyframe(s) — "
                      f"now click Re-render to rebuild the video(s)")
                return

            # ── Enhance: upscale / raise-FPS existing finals + outcome videos ──
            if scope == "enhance":
                try:
                    upscale = int(params.get("upscale") or 0)
                except Exception:
                    upscale = 0
                try:
                    fps_mult = int(params.get("fps") or 0)
                except Exception:
                    fps_mult = 0
                if upscale not in (2, 4) and (not fps_mult or fps_mult <= 1):
                    _fail("nothing selected — choose Upscale 2x/4x and/or a FPS multiplier")
                    return
                upscale_model = _upscale_model_for(default_args, upscale) or None
                interpolation_model = (getattr(default_args, "interpolation_model", None) or None)
                force = str(params.get("force") or "").lower() in ("1", "true", "yes", "on")
                target = params.get("target") or "all"
                m = next((x for x in fight_plan if x.get("match_name") == match_name), {})
                # Final playtime fps override from the enhance form (blank/0 =
                # source × multiplier, which preserves duration).
                try:
                    final_fps = int(params.get("final_fps") or 0) or None
                except Exception:
                    final_fps = None
                mf = {m.get("f1"), m.get("f2")} - {None}
                srcs = []
                if target in ("finals", "all"):
                    for kind in ("short", "long"):
                        p = vdir / f"{match_name}_{kind}.mp4"
                        if p.exists():
                            srcs.append(p)
                if target in ("outcomes", "all"):
                    for o in outcome_plan:
                        if o.get("match_name"):
                            if o.get("match_name") != match_name:
                                continue
                        elif o.get("fighter") not in mf:
                            continue
                        p = vdir / (_clip_stem_outcome(o["fighter"], o["outcome"],
                                                       o.get("match_name")) + ".mp4")
                        if p.exists() and p not in srcs:
                            srcs.append(p)
                if not srcs:
                    _fail("no matching videos found to enhance")
                    return
                _set_items([s.name for s in srcs])
                lbl = []
                if upscale in (2, 4):
                    lbl.append(f"×{upscale}")
                if fps_mult and fps_mult > 1:
                    lbl.append(f"{fps_mult}×fps")
                _prog(8, f"enhancing {len(srcs)} video(s) ({', '.join(lbl)})…")
                for i, src in enumerate(srcs):
                    _item(i, "start")
                    ok = False
                    try:
                        def _frame_cb(cur, tot, ips, phase="upscaling", _i=i):
                            _item(_i, "step", {
                                "current": cur, "total": tot,
                                "pct": int(cur / tot * 100) if tot else 0,
                                "it_per_s": ips, "phase": phase})
                        _enhance_video_file(client, upscale_model, src, upscale,
                                            fps_mult, force=force,
                                            progress_cb=_frame_cb,
                                            interpolation_model=interpolation_model,
                                            final_fps=final_fps)
                        ok = True
                    except Exception as e:
                        _dbg = str(e)
                        print(f"  [enhance] failed {src.name}: {_dbg}", flush=True)
                    _item(i, "end", ok)
                    _prog(8 + int(88 * (i + 1) / len(srcs)),
                          f"{i+1}/{len(srcs)} — {src.name}")
                _done(f"enhanced {len(srcs)} video(s)")
                return

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

                # A single-clip regenerate must NOT touch the assembled finals
                # (it would rebuild short/long from just the one clip). Re-rendering
                # ALL of a match's clips ("match-clips") does reassemble.
                _assemble = (scope != "clip")
                _stage_videos_render(
                    client, video_model, vdir, [mm], [], 1, 0, fps, clip_delay,
                    consistency=consistency, lora_map=lora_map,
                    keyframe_dir=keyframe_dir, lora_weight=lw,
                    env_lora_map=env_lora_map, env_lora_weight=elw,
                    progress_cb=_cb, clip_cb=_item,
                    video_lora_map=video_lora_map, env_video_lora_map=env_video_lora_map,
                    assemble_finals=_assemble, video_lora_scale=vls, video_size=vsz,
                    single_clip_max_frames=scm, cancel_check=_cancelled)
                if _cancelled():
                    _cancel_done("⏹ cancelled — stopped after the current clip")
                    return
                msg = f"re-rendered {len(mm['clips'])} clip(s)"
                if scope == "clip":
                    msg += " — click “Reassemble finals” to rebuild short/long"
                _done(msg)
                return

            if scope in ("outcomes", "outcome"):
                fighter = params.get("fighter")
                outcome = params.get("outcome")
                # Fighters of this match (to resolve LEGACY per-fighter outcomes,
                # which have no match_name — they belong to any match the fighter
                # appears in).
                _m = next((x for x in fight_plan
                           if x.get("match_name") == match_name), {}) if match_name else {}
                _match_fighters = {_m.get("f1"), _m.get("f2")} - {None}

                def _belongs(o):
                    if o.get("match_name"):
                        return (not match_name) or o.get("match_name") == match_name
                    # Legacy entry (no match_name): tie it to the match by fighter.
                    return (not match_name) or o.get("fighter") in _match_fighters

                if scope == "outcome":
                    sel = [o for o in outcome_plan
                           if o.get("fighter") == fighter and o.get("outcome") == outcome
                           and _belongs(o)]
                elif match_name:
                    # All outcomes of this match (per-match + legacy per-fighter).
                    sel = [o for o in outcome_plan if _belongs(o)]
                elif fighter:
                    sel = [o for o in outcome_plan if o.get("fighter") == fighter]
                else:
                    sel = list(outcome_plan)
                if not sel:
                    _fail("no matching outputs in prompts.json")
                    return
                # Keep a match's outcomes in the SAME environment as the match.
                # Legacy per-fighter outcomes carry their own (often different) env,
                # so override env/env_desc with the match's when rendering in a
                # match context.
                if match_name and _m.get("env"):
                    sel = [{**o, "env": _m.get("env"),
                            "env_desc": _m.get("env_desc", o.get("env_desc"))}
                           for o in sel]
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
                    progress_cb=_cb, clip_cb=_item,
                    video_lora_map=video_lora_map, env_video_lora_map=env_video_lora_map,
                    video_lora_scale=vls, video_size=vsz,
                    single_clip_max_frames=scm, cancel_check=_cancelled)
                if _cancelled():
                    _cancel_done("⏹ cancelled — stopped after the current output")
                    return
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
.section-title{font-size:.85rem;font-weight:700;color:#aaa;
               text-transform:uppercase;letter-spacing:.05em;margin:1rem 0 .4rem}
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
/* video lightbox (a preview enlarges + centers when you press play) */
.vlightbox-bg{display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:210;
              align-items:flex-start;justify-content:center;padding:3vh 1.5rem}
.vlightbox-bg.open{display:flex}
.vlightbox-bg video{max-width:92vw;max-height:90vh;width:auto;border-radius:10px;background:#000;
                    box-shadow:0 12px 48px rgba(0,0,0,.7)}
.vlightbox-close{position:fixed;top:.7rem;right:1.3rem;color:#fff;font-size:1.7rem;line-height:1;
                 cursor:pointer;z-index:211;font-weight:700}
.vlightbox-close:hover{color:#f5a623}
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
<div class=vlightbox-bg id=vid-lightbox onclick="if(event.target===this)window.closeVid&&closeVid()">
  <span class=vlightbox-close onclick="window.closeVid&&closeVid()">✕</span>
  <video id=vid-lightbox-vid controls playsinline></video>
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
  // Enlarge + center a video when it starts playing; pause the small inline one
  // and continue playback from the same spot in the big centered player.
  window.showVid=function(el){
    var bg=document.getElementById('vid-lightbox');
    var v=document.getElementById('vid-lightbox-vid');
    if(!bg||!v||!el) return;
    var t=0; try{ t=el.currentTime||0; }catch(e){}
    try{ el.pause(); }catch(e){}
    v.src=el.currentSrc||el.getAttribute('src')||el.src;
    bg.classList.add('open');
    v.onloadedmetadata=function(){ if(t>0){ try{ v.currentTime=t; }catch(e){} } };
    var p=v.play(); if(p&&p.catch) p.catch(function(){});
  };
  // Swap which variant (original / upscaled / higher-fps) the inline player uses.
  window.swapVid=function(sel){
    if(!sel) return;
    var wrap=sel.closest('.vplayer'); if(!wrap) return;
    var v=wrap.querySelector('video'); if(!v) return;
    var was=!v.paused; var t=0; try{ t=v.currentTime||0; }catch(e){}
    v.src=sel.value;
    try{ v.load(); }catch(e){}
    v.onloadedmetadata=function(){ if(t>0){ try{ v.currentTime=t; }catch(e){} } };
    if(was){ var p=v.play(); if(p&&p.catch) p.catch(function(){}); }
  };
  window.closeVid=function(){
    var bg=document.getElementById('vid-lightbox');
    var v=document.getElementById('vid-lightbox-vid');
    if(!bg||!v) return;
    try{ v.pause(); }catch(e){}
    bg.classList.remove('open');
    v.removeAttribute('src'); try{ v.load(); }catch(e){}
  };
  document.addEventListener('keydown',function(e){
    if(e.key==='Escape'){
      var bg=document.getElementById('img-lightbox'); if(bg) bg.classList.remove('open');
      if(window.closeVid) closeVid();
    }
  });
})();
</script>"""

    def _page(title, body, active="run"):
        nav_items = [
            ("run", "/", "▶ Run"),
            ("characters", "/characters", "👤 Characters"),
            ("environments", "/environments", "🏞 Environments"),
            ("matches", "/matches", "🥊 Matches"),
            ("wardrobe", "/wardrobe", "👕 Wardrobe"),
            ("prompts", "/prompts", "✍ Prompts"),
        ]
        nav = "".join(
            f'<a href="{href}" style="{"color:#f5a623;font-weight:700" if k==active else ""}">{label}</a>'
            for k, href, label in nav_items
        )
        return f"""<!doctype html><html lang=en><head>
<meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<link rel=icon type=image/x-icon href="/favicon.ico">
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
        def _sel(attr, val):
            try:
                return " selected" if int(getattr(args_ns, attr, 0) or 0) == int(val) else ""
            except Exception:
                return ""

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

        # Autocomplete the "specific names" fields with the characters/environments
        # that actually exist (locally saved profiles).
        import html as _html
        _char_dl = "".join(
            f'<option value="{_html.escape(p["name"], quote=True)}">'
            for p in _list_profiles("character") if not _is_referee_meta(p["meta"]))
        _env_dl = "".join(
            f'<option value="{_html.escape(p["name"], quote=True)}">'
            for p in _list_profiles("environment"))

        # Township upload: odds-range inputs, prefilled from the configured (or
        # default) ranges. Each column gets a min/max number input named
        # odds_<col>_min / odds_<col>_max so /save-config can reassemble them.
        _or = merge_ranges(_resolve_odds_ranges(args_ns))
        _odds_labels = {
            "under": "Under", "over": "Over", "win1": "Win 1", "win2": "Win 2",
            "ko1": "KO 1 (f2 by KO)", "ko2": "KO 2 (f1 by KO)",
            "ret1": "Ret. 1 (f2 by ret.)", "ret2": "Ret. 2 (f1 by ret.)",
            "draw": "Draw",
        }
        _odds_rows = "".join(
            f'<div><label>{_odds_labels[col]} <span class=hint>(min–max)</span></label>'
            f'<div style="display:flex;gap:.4rem">'
            f'<input name=odds_{col}_min type=number min=1 max=50 step=0.01 value="{_or[col][0]:.2f}">'
            f'<input name=odds_{col}_max type=number min=1 max=50 step=0.01 value="{_or[col][1]:.2f}">'
            f'</div></div>'
            for col in OUTCOME_COLUMNS)

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
  <div class=row3 style="margin-top:.6rem">
    <div><label>2× upscale model <span class=hint>(AI super-res used when 2× is picked; blank = use generic/auto)</span></label><input name=upscale_model_2x type=text value="{_v('upscale_model_2x') or ''}"></div>
    <div><label>4× upscale model <span class=hint>(used when 4× is picked; blank = use generic/auto)</span></label><input name=upscale_model_4x type=text value="{_v('upscale_model_4x') or ''}"></div>
    <div><label>Upscale model <span class=hint>(generic fallback; blank = CoderAI auto-selects)</span></label><input name=upscale_model type=text value="{_v('upscale_model') or ''}"></div>
  </div>
  <div class=row3 style="margin-top:.6rem">
    <div><label>Interpolation model <span class=hint>(AI FPS interpolation, RIFE/FILM; blank = CoderAI auto-selects)</span></label><input name=interpolation_model type=text value="{_v('interpolation_model') or ''}"></div>
    <div></div><div></div>
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
    <label>Fighter names <span class=hint>(comma-separated — type to autocomplete existing)</span></label>
    <input name=fighters type=text value="{_v('fighters') or ''}" list=existing_chars>
    <datalist id=existing_chars>{_char_dl}</datalist>
  </div>
  <div class=row style="margin-top:.6rem">
    <div><label>How many fighters <span class=hint>(0 = whole pool)</span></label>
         <input name=num_fighters type=number min=0 max=99 value="{_v('num_fighters', 0)}"></div>
    <div><label>Reference images / character <span class=hint>(more = healthier LoRA)</span></label>
         <input name=char_refs type=number min=1 max=40 value="{_v('char_refs', 4)}"></div>
    <div style="display:flex;align-items:flex-end">
      <label><input type=checkbox name=include_female{_c('include_female')}> Include female fighters</label>
    </div>
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
    <label>Environment names <span class=hint>(comma-separated — type to autocomplete existing)</span></label>
    <input name=environments type=text value="{_v('environments') or ''}" list=existing_envs>
    <datalist id=existing_envs>{_env_dl}</datalist>
  </div>
  <div class=row style="margin-top:.6rem">
    <div><label>How many environments <span class=hint>(0 = whole pool)</span></label>
         <input name=num_environments type=number min=0 max=99 value="{_v('num_environments', 0)}"></div>
    <div><label>Reference images / environment</label>
         <input name=env_refs type=number min=1 max=40 value="{_v('env_refs', 3)}"></div>
  </div>
</div>

<div class=card>
  <h2>Stage 3 — Videos</h2>
  <div class=row>
    <div><label>Number of fight matches</label>
         <input name=matches type=number min=0 max=50 value="{_v('matches', 6)}"></div>
    <div><label>Generation FPS <span class=hint>(base)</span></label>
         <input name=fps type=number min=1 max=60 value="{_v('fps', 8)}"></div>
    <div><label>Playback FPS <span class=hint>(0 = same; higher = faster motion)</span></label>
         <input name=playback_fps type=number min=0 max=60 value="{_v('playback_fps', 0)}"></div>
  </div>
  <div class=row style="margin-top:.4rem">
    <div><label>Clip delay between requests (seconds)</label>
         <input name=clip_delay type=number min=0 step=0.5 value="{_v('clip_delay', 5.0)}"></div>
  </div>
  <div class=row style="margin-top:.4rem">
    <div><label>Post-process upscale (finals + outcomes)</label>
         <select name=upscale_factor>
           <option value=0{_sel('upscale_factor', 0)}>none</option>
           <option value=2{_sel('upscale_factor', 2)}>2× (super-res)</option>
           <option value=4{_sel('upscale_factor', 4)}>4× (super-res)</option>
         </select></div>
    <div><label>Post-process raise FPS (finals + outcomes)</label>
         <select name=fps_multiplier>
           <option value=0{_sel('fps_multiplier', 0)}>none</option>
           <option value=2{_sel('fps_multiplier', 2)}>2×</option>
           <option value=3{_sel('fps_multiplier', 3)}>3×</option>
           <option value=4{_sel('fps_multiplier', 4)}>4×</option>
         </select></div>
  </div>
  <p class=hint style="margin-top:.15rem">Enhancement runs after rendering and writes new
     <code>*_2x</code>/<code>*_NxfpS</code> files alongside the originals.</p>
  <div style="margin-top:.6rem">
    <label><input type=checkbox name=skip_videos{_c('skip_videos')}> Skip Stage 3 entirely</label><br>
    <label><input type=checkbox name=only_outcomes{_c('only_outcomes')}> Outcomes only (skip fight matches)</label>
  </div>
  <label style="margin-top:.75rem">Video stage mode</label>
  <select name=stage3_mode>
    <option value=full{"selected" if stage3_mode=="full" else ""}>Full (prompts + render)</option>
    <option value=only_prompts{"selected" if stage3_mode=="only_prompts" else ""}>Matches prompts only (no render)</option>
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
           <input name=keyframe_size type=text value="{_v('keyframe_size','832x480')}"></div>
      <div><label>Character strength <span class=hint>(IP-Adapter 0-1)</span></label>
           <input name=character_strength type=number min=0 max=1 step=0.05 value="{_v('character_strength', 0.7)}"></div>
    </div>
    <div class=row>
      <div><label>Video size <span class=hint>(WxH — Wan native 832x480 / 1280x720)</span></label>
           <input name=video_size type=text value="{_v('video_size','832x480')}"></div>
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
    <details style="margin:.5rem 0 .2rem">
      <summary class=hint style="cursor:pointer">📐 Suggested training values (match steps to image count)</summary>
      <table style="width:100%;border-collapse:collapse;font-size:.74rem;margin-top:.4rem">
        <thead><tr style="color:#aaa;text-align:left">
          <th style="padding:.25rem .4rem;border-bottom:1px solid #333">Ref images</th>
          <th style="padding:.25rem .4rem;border-bottom:1px solid #333">Train steps</th>
          <th style="padding:.25rem .4rem;border-bottom:1px solid #333">LoRA weight</th>
        </tr></thead>
        <tbody style="color:#cfcfcf">
          <tr><td style="padding:.22rem .4rem">4</td><td style="padding:.22rem .4rem">300–450</td><td style="padding:.22rem .4rem">0.70–0.85</td></tr>
          <tr><td style="padding:.22rem .4rem">10–15</td><td style="padding:.22rem .4rem">700–900</td><td style="padding:.22rem .4rem">0.80–1.00</td></tr>
          <tr><td style="padding:.22rem .4rem">20+</td><td style="padding:.22rem .4rem">1000–1500</td><td style="padding:.22rem .4rem">0.90–1.00</td></tr>
        </tbody>
      </table>
      <p class=hint style="margin:.3rem 0 0">Aim for ~60–120 steps per image. Too many steps for few
      images <b>overfits</b> — fix it by lowering steps or adding reference images, not by training
      harder. The <b>Video LoRA scale</b> (~0.45) is a <i>separate</i> lever: it tames colour/motion
      from stacking LoRAs on the distilled Wan expert and is needed regardless of training amount.</p>
    </details>
    <div style="margin-top:.6rem">
      <label><input type=checkbox name=env_loras{"" if _v('no_env_loras') else " checked"}> Also train per-environment LoRAs <span class=hint>(lock each location’s look)</span></label>
      <label><input type=checkbox name=video_loras{" checked" if _v('video_loras') else ""}> Also train Wan VIDEO LoRAs <span class=hint>(per fighter/env, trained against the video model — heavy)</span></label>
    </div>
    <div class=row3 style="margin-top:.4rem">
      <div><label>Env LoRA train steps</label>
           <input name=env_lora_steps type=number min=100 max=3000 step=50 value="{_v('env_lora_steps', 800)}"></div>
      <div><label>Env LoRA rank</label>
           <input name=env_lora_rank type=number min=2 max=128 value="{_v('env_lora_rank', 16)}"></div>
      <div><label>Env LoRA weight <span class=hint>(at generation)</span></label>
           <input name=env_lora_weight type=number min=0 max=2 step=0.05 value="{_v('env_lora_weight', 0.8)}"></div>
      <div><label>Video LoRA scale <span class=hint>(×char+env weight, video only)</span></label>
           <input name=video_lora_scale type=number min=0 max=2 step=0.05 value="{_v('video_lora_scale', 1.0)}"></div>
    </div>
    <div class=row3 style="margin-top:.4rem">
      <div><label>Clip min frames <span class=hint>(per fight clip)</span></label>
           <input name=clip_min_frames type=number min=8 max=480 value="{_v('clip_min_frames', 50)}"></div>
      <div><label>Clip max frames <span class=hint>(dur = frames÷fps; >cap splits into one shot)</span></label>
           <input name=clip_max_frames type=number min=8 max=480 value="{_v('clip_max_frames', 70)}"></div>
      <div><label>Single-render cap <span class=hint>(≤81; longer = chained parts)</span></label>
           <input name=single_clip_max_frames type=number min=8 max=81 value="{_v('single_clip_max_frames', 50)}"></div>
    </div>
    <div class=row3 style="margin-top:.4rem">
      <div><label>Outcome min frames <span class=hint>(total: finish + victory)</span></label>
           <input name=outcome_min_frames type=number min=8 max=480 value="{_v('outcome_min_frames', 96)}"></div>
      <div><label>Outcome max frames <span class=hint>(split across the 2 outcome clips, then chained)</span></label>
           <input name=outcome_max_frames type=number min=8 max=480 value="{_v('outcome_max_frames', 150)}"></div>
      <div></div>
    </div>
    <div class=row style="margin-top:.4rem">
      <div><label>Short final assembly <span class=hint>(seconds, min–max)</span></label>
        <div style="display:flex;gap:.4rem">
          <input name=short_min type=number min=5 max=600 step=1 value="{_v('short_min', 40)}">
          <input name=short_max type=number min=5 max=600 step=1 value="{_v('short_max', 50)}">
        </div></div>
      <div><label>Long final assembly <span class=hint>(seconds, min–max — sets clip count)</span></label>
        <div style="display:flex;gap:.4rem">
          <input name=long_min type=number min=5 max=1200 step=1 value="{_v('long_min', 65)}">
          <input name=long_max type=number min=5 max=1200 step=1 value="{_v('long_max', 75)}">
        </div></div>
    </div>
  </div>
</div>

<div class=card>
  <h2>Township Combat League upload</h2>
  <div class=row>
    <div><label>Server endpoint <span class=hint>(base URL, e.g. https://townshipcombatleague.com)</span></label>
         <input name=upload_endpoint type=url value="{_v('upload_endpoint') or ''}"></div>
    <div><label>API token <span class=hint>(fixture-source token)</span></label>
         <input name=upload_token type=text value="{_v('upload_token') or ''}"></div>
  </div>
  <div class=row style="margin-top:.4rem">
    <div><label>Fixture ID <span class=hint>(must already exist on the server)</span></label>
         <input name=upload_fixture_id type=text value="{_v('upload_fixture_id') or ''}"></div>
    <div style="display:flex;align-items:flex-end">
      <label style="display:flex;align-items:center;gap:.4rem;margin:0">
        <input type=checkbox name=upload_after_render{_c('upload_after_render')}>
        Also generate odds, pack ZIP &amp; upload after a full render</label></div>
  </div>
  <details style="margin-top:.6rem">
    <summary style="cursor:pointer;font-weight:600">Odds ranges <span class=hint>(2-decimal odds; the anti-arbitrage check still applies)</span></summary>
    <p class=hint style="margin:.4rem 0">KO/Ret. odds are capped in practice by the no-arbitrage rule (a high KO paired with a win ≥1.0 is always a sure bet), so very high maxima may never be reached.</p>
    <div class=row3 style="margin-top:.4rem">
      {_odds_rows}
    </div>
  </details>
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
    <button class="btn btn-secondary" type=button onclick="runStep('prompts')" title="Build the match plan + write all clip/outcome prompts (no keyframes, no video render). Matches appear on the Matches page.">3 · Generate matches prompts</button>
    <button class="btn btn-secondary" type=button onclick="runStep('loras')">4 · Train LoRAs</button>
    <button class="btn btn-secondary" type=button onclick="runStep('video-loras')">4b · Train Video LoRAs</button>
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

    _video_model_cache = {"id": None}

    def _current_video_model() -> str:
        """The video model the UI/generation will actually use: the configured
        --video-model if set, else the server's auto-picked one (cached). Training
        tags video LoRAs with this model's slug, so the profile page must resolve
        it the same way to detect them."""
        vm = getattr(default_args, "video_model", None)
        if vm:
            return vm
        if _video_model_cache["id"]:
            return _video_model_cache["id"]
        try:
            client = CoderAIClient(default_args.base_url,
                                   getattr(default_args, "api_key", None))
            vm = pick_model(client, "video", None)
        except Exception:
            vm = ""
        _video_model_cache["id"] = vm or ""
        return _video_model_cache["id"]

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
        # Per-model video LoRA map + the current video model's slug, to show which
        # fighters already have a Wan video LoRA for the configured video model.
        _vlora_file = out_dir / ("video_loras.json" if kind == "character"
                                 else "env_video_loras.json")
        _vlora_map = _load_json_map(_vlora_file)
        _vslug = _model_slug(_current_video_model())

        def esc(v):
            return _html.escape(str(v if v is not None else ""), quote=True)

        cards = []
        for p in profiles:
            name = p["name"]
            meta = p["meta"]
            # Video LoRA status: prefer a match for the current video model, but
            # still report a LoRA trained for a DIFFERENT model so it never looks
            # untrained just because the active model can't be resolved.
            _ventry = _vlora_map.get(name)
            _vcur = _video_lora_path(_ventry, _vslug)
            _vslugs = (sorted(k for k, v in _ventry.items() if v)
                       if isinstance(_ventry, dict) else
                       (["(legacy)"] if isinstance(_ventry, str) and _ventry else []))
            if _vcur:
                _vstate = ("#7ed87e", f"trained ✓ ({esc(_vslug)})", "Retrain")
            elif _vslugs:
                _vstate = ("#d8b84a",
                           "trained for: " + esc(", ".join(_vslugs))
                           + (f" — not for {esc(_vslug)}" if _vslug else ""),
                           "Train")
            else:
                _vstate = ("#888",
                           f"not trained ({esc(_vslug or 'no video model')})", "Train")
            _vcolor, _vlabel, _vbtn = _vstate
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

            # LoRA train defaults mirror the Run-page configuration so a profile
            # created by a run trains with the same steps/rank you set there.
            # Characters use lora_steps/rank; environments use env_lora_steps/rank.
            if kind == "character":
                _def_lora_steps = int(getattr(default_args, "lora_steps", 800) or 800)
                _def_lora_rank  = int(getattr(default_args, "lora_rank", 16) or 16)
            else:
                _def_lora_steps = int(getattr(default_args, "env_lora_steps", 800) or 800)
                _def_lora_rank  = int(getattr(default_args, "env_lora_rank", 16) or 16)

            cards.append(
                f'<div class=card id="pf-{kind}-{esc(name)}">'
                f'  <div class=pf-head>'
                f'    <span class=pf-name>{esc(name)}'
                + (' <span style="font-size:.66rem;font-weight:700;color:#1b1b1b;'
                   'background:#d8b84a;border-radius:4px;padding:.05rem .4rem;'
                   'vertical-align:middle">REFEREE</span>'
                   if (kind == "character" and _is_referee_meta(meta)) else '')
                + f'</span>'
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
                f'value={_def_lora_steps} min=50 max=5000 step=50 style="width:66px;display:inline-block"></label>'
                f'    <label style="margin:0;font-size:.78rem">rank <input type=number data-lora=rank '
                f'value={_def_lora_rank} min=2 max=128 style="width:54px;display:inline-block"></label>'
                f'    <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
                f'onclick="trainLora(\'{kind}\',\'{esc(name)}\')">🧠 {"Retrain" if (_lora_map.get(name)) else "Train"} image LoRA</button>'
                f'    <span class=pf-lora-status style="font-size:.76rem;color:#7ea8f7"></span>'
                f'  </div>'
                # Video (Wan) LoRA — separate, tagged with the current video model.
                f'  <div class=pf-actions style="border-top:1px solid #222;padding-top:.6rem;margin-top:.6rem">'
                f'    <span style="font-size:.78rem;color:{_vcolor}">Video LoRA: {_vlabel}</span>'
                f'    <label style="margin:0;font-size:.78rem">steps <input type=number data-vlora=steps '
                f'value={_def_lora_steps} min=50 max=5000 step=50 style="width:66px;display:inline-block"></label>'
                f'    <label style="margin:0;font-size:.78rem">rank <input type=number data-vlora=rank '
                f'value={_def_lora_rank} min=2 max=128 style="width:54px;display:inline-block"></label>'
                f'    <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
                f'onclick="trainLora(\'{kind}\',\'{esc(name)}\',\'video\')">🎬 '
                f'{_vbtn} video LoRA</button>'
                f'    <span class=pf-vlora-status style="font-size:.76rem;color:#7ea8f7"></span>'
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
    if(d.status==='queued'){ st.textContent=d._msg||'⏳ queued…'; setTimeout(poll,1200); }
    else if(d.status==='running'){ st.textContent='⏳ '+(d._msg||('working… '+pct+'%')); setTimeout(poll,1200); }
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
async function trainLora(kind,name,target){
  target = target||'image';
  const isV = target==='video';
  const root=document.getElementById('pf-'+kind+'-'+name);
  const st=root.querySelector(isV?'.pf-vlora-status':'.pf-lora-status');
  const sel=isV?'[data-vlora=steps]':'[data-lora=steps]';
  const rsel=isV?'[data-vlora=rank]':'[data-lora=rank]';
  const steps=parseInt((root.querySelector(sel)||{}).value||'800',10);
  const rank=parseInt((root.querySelector(rsel)||{}).value||'16',10);
  const msg = isV
    ? 'Train a VIDEO (Wan) LoRA for "'+name+'" ('+steps+' steps) against the configured video model? '
      +'This is heavy — it evicts loaded models and can take a long time (large video models may need 4-bit).'
    : 'Train image identity LoRA for "'+name+'" ('+steps+' steps)? '
      +'This evicts loaded models and can take several minutes.';
  if(!(await uiConfirm(msg,{title:(isV?'Train video LoRA':'Train LoRA'), okText:'Train'})))return;
  const fd=new FormData();
  fd.append('kind',kind); fd.append('name',name);
  fd.append('steps',steps); fd.append('rank',rank);
  fd.append('target',target);
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
    if(d.status==='queued'){ st.textContent=d._msg||'⏳ queued…'; setTimeout(poll,1200); }
    else if(d.status==='running'){ st.textContent='⏳ '+(d._msg||('training… '+pct+'%'))+' ('+pct+'%)'; setTimeout(poll,1500); }
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
      const st=root.querySelector(j.target==='video'?'.pf-vlora-status':'.pf-lora-status');
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
async function autogenProfile(kind){
  const root=document.getElementById('new-'+kind);
  const st=root.querySelector('.nf-status');
  const roleEl=root.querySelector('[data-new=role]');
  const role=roleEl?roleEl.value:'fighter';
  const fd=new FormData(); fd.append('kind',kind); fd.append('role',role);
  st.style.color='#aaa'; st.textContent='✨ Inventing '+(role==='referee'?'referee':kind)+'…';
  let j;
  try{ const r=await fetch('/profile/autogen',{method:'POST',body:fd}); j=await r.json(); }
  catch(e){ st.style.color='#e07070'; st.textContent='✗ '+e; return; }
  if(j.error){ st.style.color='#e07070'; st.textContent='✗ '+j.error; return; }
  const f=j.fields||{};
  Object.keys(f).forEach(k=>{
    const el=root.querySelector('[data-new='+k+']');
    if(el && f[k]!=null && f[k]!=='') el.value=f[k];
  });
  st.style.color='#7ed87e'; st.textContent='✓ Filled — review/edit, then Create';
}
async function createProfile(kind){
  const root=document.getElementById('new-'+kind);
  const st=root.querySelector('.nf-status');
  const name=(root.querySelector('[data-new=name]').value||'').trim();
  if(!name){ st.style.color='#e07070'; st.textContent='Enter a name first'; return; }
  const fd=new FormData();
  fd.append('kind',kind); fd.append('name',name);
  root.querySelectorAll('[data-new]').forEach(el=>{
    const k=el.getAttribute('data-new'); if(k!=='name') fd.append(k,el.value);
  });
  st.style.color='#aaa'; st.textContent='Starting…';
  let j;
  try{
    const r=await fetch('/profile/create',{method:'POST',body:fd});
    j=await r.json();
  }catch(e){ st.style.color='#e07070'; st.textContent='✗ '+e; return; }
  if(j.error){ st.style.color='#e07070'; st.textContent='✗ '+j.error; return; }
  pollRegen(j.job_id, st);
}
</script>"""
        # Form to create a brand-new profile (generates reference images server-side).
        _gender_row = (
            '<div><label>Gender</label>'
            '<input type=text data-new=gender value="male"></div>'
            '<div><label>Role <span class=hint>(referee = official, not a fighter)</span></label>'
            '<select data-new=role>'
            '<option value="fighter">fighter</option>'
            '<option value="referee">referee</option>'
            '</select></div>'
        ) if kind == "character" else ""
        new_form = (
            f'<details class=card style="margin-bottom:1rem" id="new-{kind}">'
            f'  <summary style="cursor:pointer;font-weight:600">➕ New {label[:-1].lower()}</summary>'
            f'  <div class=row style="margin-top:.6rem">'
            f'    <div><label>Name</label><input type=text data-new=name '
            f'placeholder="unique name"></div>'
            f'    <div><label>Region</label><input type=text data-new=region></div>'
            f'    {_gender_row}'
            f'  </div>'
            f'  <label>Description</label>'
            f'  <textarea data-new=description rows=2></textarea>'
            f'  <label>Prompt <span class=hint>(drives image generation)</span></label>'
            f'  <textarea data-new=prompt rows=3></textarea>'
            f'  <div class=pf-actions style="margin-top:.5rem">'
            f'    <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
            + (f'title="Pick fighter/referee above first, then auto-fill the form with '
               f'the text model" '
               if kind == "character" else 'title="Auto-fill the form with the text model" ')
            + f'onclick="autogenProfile(\'{kind}\')">✨ Autogenerate</button>'
            f'    <label style="margin:0;font-size:.78rem">Generate <input type=number '
            f'data-new=count value=4 min=1 max=8 style="width:54px;display:inline-block"> '
            f'reference image(s)</label>'
            f'    <button class="btn btn-primary" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="createProfile(\'{kind}\')">✅ Create {label[:-1].lower()}</button>'
            f'    <span class="nf-status pf-regen-status" style="font-size:.76rem;color:#7ea8f7"></span>'
            f'  </div>'
            f'  <p class=hint style="margin:.4rem 0 0">“Autogenerate” fills the fields '
            f'above with the text model so you can review and edit them before creating.</p>'
            f'</details>'
        )
        return (f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<h1>{label}</h1>'
                f'<a href="/{kind}s" class="btn btn-secondary" style="font-size:.8rem">↻ Refresh</a></div>'
                f'<p class=hint style="margin-bottom:.8rem">Edit a profile’s fields and Save, or '
                f'Remove it entirely. Changes apply to the local output folder and are synced to CoderAI.</p>'
                f'{new_form}{inner}{script}')

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

    def _vid_tag(p: Path, h=180, poster: Path = None):
        # Default to the highest enhanced variant when one exists (variants are
        # sorted weakest→strongest, original first), so the player shows the best
        # available quality up front; the dropdown still lets you pick others.
        variants = _video_variants(p)
        default = variants[-1][1] if len(variants) > 1 else p
        url = "/media/" + str(default.relative_to(out_dir)).replace("\\", "/")
        pa = ""
        if poster is not None and poster.exists() and poster.stat().st_size > 0:
            purl = "/media/" + str(poster.relative_to(out_dir)).replace("\\", "/")
            pa = f' poster="{_esc(purl)}?t={int(poster.stat().st_mtime)}"'
        video = (f'<video src="{_esc(url)}" controls preload=none{pa} '
                 f'onplay="window.showVid&&showVid(this)" '
                 f'title="Press play to enlarge" '
                 f'style="width:100%;height:{h}px;object-fit:cover;cursor:zoom-in;'
                 f'border-radius:6px;background:#111"></video>')
        # When enhanced variants exist, offer a selector to switch the source.
        sel = ""
        if len(variants) > 1:
            opts = "".join(
                f'<option value="{_esc("/media/" + str(vp.relative_to(out_dir)).replace(chr(92), "/"))}"'
                f'{" selected" if vp == default else ""}>{_esc(lbl)}</option>'
                for lbl, vp in variants)
            sel = (f'<select onchange="window.swapVid&&swapVid(this)" '
                   f'title="Choose which version to play" '
                   f'style="width:100%;margin-bottom:.25rem;font-size:.74rem;'
                   f'padding:.15rem .3rem">{opts}</select>')
        return f'<div class=vplayer>{sel}{video}</div>'

    def _kf_img_tag(kf: Path, h=120):
        """A keyframe still used as a preview when a clip/outcome isn't rendered."""
        if not (kf and kf.exists() and kf.stat().st_size > 0):
            return None
        url = "/media/" + str(kf.relative_to(out_dir)).replace("\\", "/")
        return (f'<img src="{_esc(url)}?t={int(kf.stat().st_mtime)}" loading=lazy '
                f'title="Keyframe preview (not yet rendered)" '
                f'style="width:100%;height:{h}px;object-fit:cover;border-radius:6px;'
                f'background:#111;opacity:.85">')

    # Keyframes page: save the per-keyframe prompt overrides (empty box clears the
    # override → the prompt is auto-composed from wardrobe + environment again).
    _kf_prompt_js = """
<script>
async function saveKfPrompts(ev, name){
  const root=document.getElementById('detail');
  const st=document.getElementById('detail-status');
  const setSt=(c,t)=>{ st.style.color=c; st.textContent=t; };
  const fd=new FormData(); fd.append('mode','kfprompts'); fd.append('name',name);
  root.querySelectorAll('[data-kfclip]').forEach(el=>fd.append('kfclip_'+el.getAttribute('data-kfclip'), el.value));
  root.querySelectorAll('[data-kfoutc]').forEach(el=>fd.append('kfoutc_'+el.getAttribute('data-kfoutc'), el.value));
  setSt('#aaa','Saving…');
  try{
    const j=await (await fetch('/matches/save',{method:'POST',body:fd})).json();
    if(j.error){ setSt('#e07070','✗ '+j.error); return; }
    setSt('#7ed87e','✓ Saved — now Regenerate the keyframe(s)');
  }catch(e){ setSt('#e07070','✗ '+e); }
}
</script>"""

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
    if(d.status==='queued'){ setSt('#7ea8f7', d._msg||'⏳ queued…'); setTimeout(poll,1200); }
    else if(d.status==='running'){ setSt('#7ea8f7','⏳ '+(d._msg||('working… '+pct+'%'))+' ('+pct+'%)'); setTimeout(poll,1500); }
    else if(d.status==='done'){ setSt('#7ed87e','✓ '+(d._msg||'done')+' — reloading…'); setTimeout(()=>location.reload(),1200); }
    else { setSt('#e07070','✗ '+(d.error||'failed')); }
  };
  setTimeout(poll,900);
}
function _esch(s){ return String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
// Short, friendly name for a queued/running match job (by scope) so the queue
// list reads "Whole match", "All clips", etc. instead of raw scope tokens.
function _jobScopeLabel(j){
  const m={'full':'Whole match','match-clips':'All clips','clip':'One clip',
    'clip-prompt':'Clip prompt','outcomes':'All outcomes','outcome':'One outcome',
    'outcome-prompt':'Outcome prompt',
    'keyframes':'All keyframes','keyframes-missing':'Missing keyframes',
    'keyframe':'One keyframe','replan':'Re-plan prompts',
    'replan-outcomes':'Outcome prompts','reassemble':'Reassemble','enhance':'Enhance'};
  return m[(j&&j.scope)||'']||((j&&j.scope)||'task');
}
function _renderMatchBars(wrap, d, jobId, queued){
  if(!wrap) return;
  const pct=d.progress||0;
  const qn=(queued&&queued.length)||0;
  const total=1+qn;
  // The cancel button stops the WHOLE queue (running job + everything waiting),
  // so allow it whether this job is running or just queued.
  const canCancel=((d.status==='running'||d.status==='queued') && d.cancellable && jobId);
  const cancelLabel=total>1?('⏹ Cancel all ('+total+')'):'⏹ Cancel';
  // No inline onclick (its nested quotes are mangled by the server-side string
  // emit) — the handler is attached after innerHTML below.
  const cancelBtn=canCancel
    ? '<button class="btn btn-danger" id="cancel-'+jobId+'" '
      +'style="font-size:.72rem;padding:.15rem .6rem;float:right">'+cancelLabel+'</button>'
    : '';
  const head=(d.status==='queued')?_esch(d._msg||'⏳ queued…'):('Overall — '+pct+'%');
  let h='<div class=prg-global><div class=prg-label>'+cancelBtn+head+'</div>'
       +'<div class=progress-bar><div class="progress-fill'+(d.status==='queued'?' striped':'')+'" style="width:'+(d.status==='queued'?100:pct)+'%"></div></div></div>';
  const items=d.items||[];
  if(items.length){
    h+='<div class=prg-items>';
    for(const it of items){
      const s=it.status||'pending';
      let w='0%', cls='', sub=s;
      if(s==='rendering'){
        const st=it.step;
        if(st && st.tot>0){
          // Real diffusion-step progress (and part X/Y when this clip is a
          // chained single shot made of several concatenated generations).
          w=st.pct+'%';
          const partTxt=(st.parts>1)?('shot part '+st.part+'/'+st.parts+' · '):'';
          const itsTxt=st.its?(' · '+st.its+'it/s'):'';
          sub=partTxt+(st.phase||'step')+' '+st.cur+'/'+st.tot+' ('+st.pct+'%)'+itsTxt;
        } else { w='100%'; cls=' striped'; sub='starting…'; }
      }
      else if(s==='done'){ w='100%'; cls=' ok'; }
      else if(s==='failed'){ w='100%'; cls=' fail'; }
      else if(s==='skipped'){ w='0%'; sub='skipped'; }
      const icon=s==='done'?'✓':(s==='failed'?'✗':(s==='rendering'?'⏳':(s==='skipped'?'⏹':'·')));
      h+='<div class=prg-item><div class=prg-ilabel>'+icon+' '+_esch(it.label)+' — '+_esch(sub)+'</div>'
        +'<div class=progress-bar><div class="progress-fill'+cls+'" style="width:'+w+'"></div></div></div>';
    }
    h+='</div>';
  }
  // Show what is waiting behind the running job (serialised server-side) so the
  // queue is visible, not just a count.
  if(queued && queued.length){
    h+='<div class=prg-items style="margin-top:.5rem">'
      +'<div class=prg-label style="opacity:.85">Queued ('+queued.length+')</div>';
    for(const q of queued){
      h+='<div class=prg-item><div class=prg-ilabel>⏳ '+_esch(_jobScopeLabel(q))
        +' — '+_esch(q._msg||'queued…')+'</div>'
        +'<div class=progress-bar><div class="progress-fill striped" style="width:100%"></div></div></div>';
    }
    h+='</div>';
  }
  wrap.innerHTML=h;
  wrap.classList.remove('hidden');
  if(canCancel){
    const cb=document.getElementById('cancel-'+jobId);
    if(cb) cb.onclick=()=>cancelAll();
  }
}
// Detail-page poller: drives the text status AND the visual progress bars.
function _pollMatchBars(jobId, setSt, wrap){
  const poll=async()=>{
    let d;
    try{ d=await (await fetch('/job/'+jobId)).json(); }
    catch(e){ setTimeout(poll,2000); return; }
    _renderMatchBars(wrap, d, jobId);
    const pct=d.progress||0;
    if(d.status==='running'){ setSt('#7ea8f7','⏳ '+(d._msg||'working…')+' ('+pct+'%)'); setTimeout(poll,1200); }
    else if(d.status==='done'){
      if(d.cancelled){ setSt('#e0a060', (d._msg||'⏹ cancelled')+' — reloading…'); }
      else { setSt('#7ed87e','✓ '+(d._msg||'done')+' — reloading…'); }
      setTimeout(()=>location.reload(),1600);
    }
    else { setSt('#e07070','✗ '+(d.error||'failed')); }
  };
  setTimeout(poll,500);
}
// Single persistent monitor for the match-detail progress area. Instead of one
// poller per job (which makes two queued jobs fight over the shared bars and the
// status text — the "blinking" bug), ONE monitor polls /active-jobs, renders the
// currently-RUNNING job for this match and notes how many are queued behind it.
// Jobs are serialised server-side so only one runs at a time; this just reflects
// that single source of truth.
let _matchMonOn=false;
const _cardMon=new Set();
// Matches-LIST page: ONE monitor per match card (instead of one poller per job,
// which made a card's status blink between the running job and its queued ones).
// Reads /active-jobs, shows the running job's progress + how many are queued for
// that match, and reloads the list when the match's jobs all finish.
function startMatchCardMonitor(match){
  if(!match || _cardMon.has(match)) return;
  _cardMon.add(match);
  const card=document.querySelector('.match-card[data-match="'+match+'"]');
  const stEl=card?card.querySelector('.match-status'):null;
  if(!stEl){ _cardMon.delete(match); return; }
  const setSt=(c,t)=>{ stEl.style.color=c; stEl.textContent=t; };
  let sawActive=false;
  const tick=async()=>{
    let data;
    try{ data=await (await fetch('/active-jobs')).json(); }
    catch(e){ setTimeout(tick,2500); return; }
    const mine=(data.jobs||[]).filter(j=>j.jtype==='match' && (j.match||'')===match);
    const running=mine.find(j=>j.status==='running');
    const queued=mine.filter(j=>j.status==='queued');
    if(running){
      sawActive=true;
      let t='⏳ '+(running._msg||'rendering…')+' ('+(running.progress||0)+'%)';
      if(queued.length) t+=' · '+queued.length+' queued';
      setSt('#7ea8f7',t);
      setTimeout(tick,1500);
    } else if(queued.length){
      sawActive=true;
      setSt('#7ea8f7','⏳ queued — '+queued.length+' waiting to start…');
      setTimeout(tick,1500);
    } else {
      _cardMon.delete(match);
      if(sawActive){ setSt('#7ed87e','✓ done — reloading…'); setTimeout(()=>location.reload(),1400); }
    }
  };
  setTimeout(tick,300);
}
function startMatchMonitor(match, setSt){
  if(_matchMonOn) return;
  _matchMonOn=true;
  const wrap=document.getElementById('match-progress');
  let sawActive=false;
  const tick=async()=>{
    let data;
    try{ data=await (await fetch('/active-jobs')).json(); }
    catch(e){ setTimeout(tick,2000); return; }
    const mine=(data.jobs||[]).filter(j=>j.jtype==='match' && (j.match||'')===match);
    const running=mine.find(j=>j.status==='running');
    const queued=mine.filter(j=>j.status==='queued');
    if(running){
      sawActive=true;
      if(wrap){ wrap.classList.remove('hidden'); _renderMatchBars(wrap, running, running.job_id, queued); }
      let t='⏳ '+(running._msg||'working…')+' ('+(running.progress||0)+'%)';
      if(queued.length) t+=' · '+queued.length+' queued';
      if(setSt) setSt('#7ea8f7',t);
      setTimeout(tick,1200);
    } else if(queued.length){
      sawActive=true;
      // Nothing running yet: show the first queued job's bars + the rest behind it.
      if(wrap){ wrap.classList.remove('hidden'); _renderMatchBars(wrap, queued[0], queued[0].job_id, queued.slice(1)); }
      if(setSt) setSt('#7ea8f7','⏳ queued — '+queued.length+' waiting to start…');
      setTimeout(tick,1200);
    } else {
      _matchMonOn=false;
      if(sawActive){
        if(setSt) setSt('#7ed87e','✓ done — reloading…');
        setTimeout(()=>location.reload(),1300);
      }
    }
  };
  setTimeout(tick,300);
}
async function cancelJob(jobId){
  const b=document.getElementById('cancel-'+jobId);
  if(b){ b.disabled=true; b.textContent='⏹ Stopping…'; }
  try{ await fetch('/job/cancel',{method:'POST',
       headers:{'Content-Type':'application/x-www-form-urlencoded'},
       body:'job_id='+encodeURIComponent(jobId)}); }
  catch(e){ if(b){ b.disabled=false; b.textContent='⏹ Cancel'; } }
}
// Cancel the WHOLE generation queue: the running job stops after its current
// item and every queued job is skipped. Used by the progress cancel button.
async function cancelAll(){
  document.querySelectorAll('button[id^="cancel-"]').forEach(b=>{
    b.disabled=true; b.textContent='⏹ Stopping queue…'; });
  try{ await fetch('/job/cancel-all',{method:'POST',
       headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:''}); }
  catch(e){
    document.querySelectorAll('button[id^="cancel-"]').forEach(b=>{
      b.disabled=false; b.textContent='⏹ Cancel'; });
  }
}
async function reMatch(ev, scope, params){
  if(ev) ev.preventDefault();
  const labels={'full':'Regenerate this ENTIRE match end to end — re-plan all prompts, regenerate every keyframe, re-render all clips + outcomes, then reassemble the finals. Uses the text, image AND video models, so this is the slowest action. Other matches are untouched; the existing prompts, keyframes and clips for this match are replaced.',
                'match-clips':'Re-render ALL clips of this match (uses the video model, can take a while)?',
                'clip':'Re-render this single clip?',
                'reassemble':'Reassemble the final short/long videos from the existing clips? (fast, no model)',
                'outcomes':'Re-render all output clips for this fighter (uses the video model)?',
                'outcome':'Re-render this output clip?',
                'keyframes':'Regenerate ALL keyframe images for this match (uses the image model)? Existing keyframes are replaced; the clip videos are NOT re-rendered — click Re-render afterwards.',
                'keyframes-missing':'Generate only the MISSING keyframe images for this match (uses the image model)? Existing keyframes are kept; nothing is re-rendered.',
                'keyframe':'Regenerate this keyframe image (uses the image model)? The clip video is NOT re-rendered — click Re-render afterwards.',
                'replan':'Re-plan this match: rebuild its clip list and rewrite all clip prompts (frame budget from the Clip min/max frames settings). Only this match changes — other matches and outcomes are untouched. AFTERWARDS, in order: 1) Regenerate keyframes (the old keyframes match the PREVIOUS prompts and would anchor the video to the wrong image — causing static/low-motion clips), 2) Re-render all clips, 3) Reassemble finals.',
                'replan-outcomes':'Rewrite ONLY this match\\'s outcome prompts (the finish + victory shots for win / ko / retire / draw). The fight-clip prompts are left untouched. AFTERWARDS regenerate the outcome keyframes, then re-render the outcomes.',
                'clip-prompt':'Rewrite ONLY this single clip\\'s prompt (text model). The other clips are untouched and nothing is rendered. AFTERWARDS regenerate this clip\\'s keyframe (kf↻), then Re-render it.',
                'outcome-prompt':'Rewrite ONLY this single outcome\\'s prompts (its finish + victory shots, text model). The other outcomes and the fight clips are untouched and nothing is rendered. AFTERWARDS regenerate this outcome\\'s keyframes (kf↻), then Re-render it.'};
  const kf=(scope==='keyframes'||scope==='keyframe'||scope==='keyframes-missing');
  const kfMiss=(scope==='keyframes-missing');
  const isReplan=(scope==='replan'||scope==='replan-outcomes'||scope==='clip-prompt'||scope==='outcome-prompt');
  const isPrompt=(scope==='clip-prompt'||scope==='outcome-prompt');
  const isFull=(scope==='full');
  let fullPkg=true, fullUpl=false;
  if(isFull){
    fullPkg=!!(document.getElementById('full-package')||{checked:true}).checked;
    fullUpl=!!(document.getElementById('full-upload')||{}).checked;
    const extra=[]; if(fullPkg) extra.push('enhance ×2, generate odds + pack the upload ZIP');
    if(fullUpl) extra.push('upload to the configured Township endpoint');
    if(extra.length) labels.full += ' Afterwards it will '+extra.join(', then ')+'.';
  }
  if(!(await uiConfirm(labels[scope]||'Proceed?',
       {title:(isFull?'Regenerate whole match':(isPrompt?'Rewrite prompt':(isReplan?'Re-plan match prompts':(kfMiss?'Generate missing keyframes':(kf?'Regenerate keyframes':'Regenerate'))))),
        okText:(isFull?'Regenerate match':(scope==='reassemble'?'Reassemble':(isPrompt?'Rewrite prompt':(isReplan?'Re-plan':(kfMiss?'Generate missing':(kf?'Regenerate':'Re-render')))))),
        danger:(scope!=='reassemble'&&!kf&&!isReplan)})))return;
  const stEl=_findStatus(ev);
  const setSt=(c,t)=>{ if(stEl){ stEl.style.color=c; stEl.textContent=t; } };
  const fd=new FormData(); fd.append('scope',scope);
  for(const k in params) fd.append(k, params[k]);
  if(isFull){ fd.append('package', fullPkg?'1':'0'); fd.append('upload', fullUpl?'1':'0'); }
  setSt('#aaa','Starting…');
  let j;
  try{ j=await (await fetch('/matches/render',{method:'POST',body:fd})).json(); }
  catch(e){ setSt('#e07070','✗ '+e); return; }
  if(j.error){ setSt('#e07070','✗ '+j.error); return; }
  const det=document.getElementById('detail');
  const wrap=document.getElementById('match-progress');
  if(det && wrap){
    // Detail page: a SINGLE monitor drives the shared bars + status from the
    // server's job queue, so a second request submitted while one is running just
    // shows up as "queued" instead of fighting over the display.
    setSt('#7ea8f7','✓ '+(j.queued_msg||'submitted')+' — '+(scope==='reassemble'?'assembling…':'queued/running…'));
    startMatchMonitor(det.getAttribute('data-match'), setSt);
  } else {
    _pollJob(j.job_id, setSt);
  }
}
function _enhRecalcFps(){
  // Auto-fill Final FPS = Render FPS × Raise-FPS multiplier (preserves duration);
  // ×0/none → just the render fps. The user can still edit the field afterwards.
  const sel=document.getElementById('enh-fps'); const out=document.getElementById('enh-final-fps');
  if(!sel||!out) return;
  const base=parseInt(sel.getAttribute('data-render-fps'))||0;
  const mult=parseInt(sel.value)||0;
  out.value = (mult>1) ? base*mult : base;
}
async function enhanceMatch(ev, target, match){
  if(ev) ev.preventDefault();
  const up=(document.getElementById('enh-upscale')||{}).value||'0';
  const fps=(document.getElementById('enh-fps')||{}).value||'0';
  const finalFps=(document.getElementById('enh-final-fps')||{}).value||'0';
  const force=!!(document.getElementById('enh-force')||{}).checked;
  if(up==='0' && (fps==='0'||fps==='1')){ alert('Pick an Upscale factor and/or a FPS multiplier first.'); return; }
  const tlabel={finals:'final videos',outcomes:'outcome videos',all:'finals + outcomes'}[target]||target;
  const bits=[]; if(up!=='0') bits.push('upscale '+up+'×'); if(fps!=='0'&&fps!=='1') bits.push('FPS '+fps+'×');
  if(force) bits.push('force re-enhance');
  if(!(await uiConfirm('Enhance '+tlabel+' ('+bits.join(', ')+')? '+(force?'Existing enhanced files will be overwritten from the originals.':'New files are written alongside the originals.'),
       {title:'Enhance videos', okText:'Enhance'})))return;
  const stEl=document.getElementById('detail-status');
  const setSt=(c,t)=>{ if(stEl){ stEl.style.color=c; stEl.textContent=t; } };
  const fd=new FormData(); fd.append('scope','enhance'); fd.append('match',match);
  fd.append('target',target); fd.append('upscale',up); fd.append('fps',fps);
  fd.append('final_fps',finalFps); fd.append('force',force?'1':'0');
  setSt('#aaa','Starting…');
  let j;
  try{ j=await (await fetch('/matches/render',{method:'POST',body:fd})).json(); }
  catch(e){ setSt('#e07070','✗ '+e); return; }
  if(j.error){ setSt('#e07070','✗ '+j.error); return; }
  const det=document.getElementById('detail');
  const wrap=document.getElementById('match-progress');
  if(det && wrap){
    setSt('#7ea8f7','✓ submitted — queued/running…');
    startMatchMonitor(det.getAttribute('data-match'), setSt);
  } else {
    _pollJob(j.job_id, setSt);
  }
}
async function delVid(ev, scope, params){
  if(ev) ev.preventDefault();
  const labels={'clip':'Delete this clip video file?',
                'final':'Delete this assembled video file?',
                'match':'Delete ALL video files for this match (clips + finals)? The plan/prompts are kept so you can re-render.',
                'output':'Delete this output video file?',
                'outputs':'Delete ALL outcome video files for this match (and their upscaled variants)? Clips and finals are kept.',
                'keyframes':'Clear ALL keyframe images for this match? The next re-render will run keyframe-free until you regenerate them.',
                'keyframe':'Clear this keyframe image? The next re-render of it will run keyframe-free until you regenerate it.',
                'enhanced':'Remove only the UPSCALED / higher-FPS versions for this match (the *_2x / *_NxfpS files)? The originals are kept.',
                'match-purge':'Remove this match COMPLETELY — every clip, final, outcome, keyframe AND its entry in the plan (prompts.json)? This cannot be undone.'};
  if(!(await uiConfirm(labels[scope]||'Delete?',{title:'Remove videos', okText:'Delete', danger:true})))return;
  const stEl=_findStatus(ev);
  const setSt=(c,t)=>{ if(stEl){ stEl.style.color=c; stEl.textContent=t; } };
  const fd=new FormData(); fd.append('scope',scope);
  for(const k in params) fd.append(k, params[k]);
  let j;
  try{ j=await (await fetch('/matches/delete',{method:'POST',body:fd})).json(); }
  catch(e){ setSt('#e07070','✗ '+e); return; }
  if(j.error){ setSt('#e07070','✗ '+j.error); return; }
  if(scope==='match-purge'){
    setSt('#7ed87e','✓ match removed ('+(j.removed||0)+' file(s)) — returning to Matches…');
    setTimeout(()=>{ location.href = (window.ROOT_PATH||'') + '/matches'; },700);
    return;
  }
  setSt('#7ed87e','✓ removed '+(j.removed||0)+' file(s) — reloading…');
  setTimeout(()=>location.reload(),700);
}
async function saveMatch(ev, name){
  const root=document.getElementById('detail');
  const st=document.getElementById('detail-status');
  const setSt=(c,t)=>{ st.style.color=c; st.textContent=t; };
  const fd=new FormData(); fd.append('mode','match'); fd.append('name',name);
  ['f1','f2','env','referee','short_target','long_target','playback_fps'].forEach(k=>{
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
// ── Township upload: generate odds + ZIP, then chunked upload with a bar ──
function _upStatusEl(){ return document.querySelector('#detail .up-status'); }
async function prepOdds(ev, name){
  const st=_upStatusEl();
  const setSt=(c,t)=>{ if(st){ st.style.color=c; st.textContent=t; } };
  setSt('#aaa','🎲 Generating odds & packing ZIP…');
  const fd=new FormData(); fd.append('name',name);
  let j;
  try{ j=await (await fetch('/match/odds',{method:'POST',body:fd})).json(); }
  catch(e){ setSt('#e07070','✗ '+e); return; }
  if(j.error && !j.odds){ setSt('#e07070','✗ '+j.error); return; }
  if(!j.ok){
    const why=j.error||(j.missing&&j.missing.length?('missing '+j.missing.join(', ')):'incomplete');
    setSt('#e0a800','⚠ '+why);
  } else {
    setSt('#7ed87e','✓ Odds + ZIP ready');
  }
  // Reload so the odds table, badge and Upload button enabled-state refresh.
  setTimeout(()=>location.reload(), 700);
}
async function uploadMatch(ev, name){
  const btn=document.getElementById('upload-btn');
  const st=_upStatusEl();
  const setSt=(c,t)=>{ if(st){ st.style.color=c; st.textContent=t; } };
  const prog=document.querySelector('#detail .up-progress');
  const bar=document.querySelector('#detail .up-bar');
  const pct=document.querySelector('#detail .up-pct');
  if(btn) btn.disabled=true;
  if(prog) prog.style.display='block';
  const setBar=(p,msg)=>{ if(bar) bar.style.width=p+'%'; if(pct) pct.textContent=p+'% '+(msg||''); };
  setBar(0,'starting…'); setSt('#aaa','Uploading…');
  let j;
  try{ const fd=new FormData(); fd.append('name',name);
       j=await (await fetch('/match/upload',{method:'POST',body:fd})).json(); }
  catch(e){ setSt('#e07070','✗ '+e); if(btn) btn.disabled=false; return; }
  if(j.error){ setSt('#e07070','✗ '+j.error); if(btn) btn.disabled=false; return; }
  // Poll the job for progress until done/error.
  const jid=j.job_id;
  while(true){
    await new Promise(r=>setTimeout(r,500));
    let d;
    try{ d=await (await fetch('/job/'+jid)).json(); }catch(e){ continue; }
    setBar(d.progress||0, d._msg||'');
    if(d.status==='done'){ setSt('#7ed87e','✓ '+(d._msg||'uploaded')); setBar(100,'done');
      setTimeout(()=>location.reload(), 900); return; }
    if(d.status==='error'){ setSt('#e07070','✗ '+(d.error||d._msg||'upload failed'));
      if(btn) btn.disabled=false; return; }
  }
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
  const det=document.getElementById('detail');
  const detMatch=det?det.getAttribute('data-match'):null;
  let monStarted=false;
  // Collect the distinct matches that have jobs so the LIST page starts exactly
  // ONE monitor per card (not one poller per job → no blinking between the
  // running job and its queued siblings).
  const cardMatches=new Set();
  for(const j of (data.jobs||[])){
    if(j.jtype!=='match') continue;
    if(det && detMatch===j.match){
      // Detail page for this match: hand off to the single monitor (once).
      if(monStarted) continue;
      monStarted=true;
      const stEl=document.getElementById('detail-status');
      const setSt=(c,t)=>{ if(stEl){ stEl.style.color=c; stEl.textContent=t; } };
      startMatchMonitor(detMatch, setSt);
    } else if(j.match){
      cardMatches.add(j.match);
    }
  }
  for(const mm of cardMatches) startMatchCardMonitor(mm);
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
                f'  <button class="btn btn-danger" style="font-size:.8rem;padding:.35rem .8rem" '
                f'onclick="delVid(event,\'match-purge\',{{match:\'{_esc(mn)}\'}})" '
                f'title="Remove this match completely — files, keyframes and plan entry">🧨 Remove</button>'
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

        # ── New-match form: pick two fighters + an environment, then either build
        # the whole match end-to-end or just write its prompts. ──────────────────
        # Populate with the characters/environments that ACTUALLY exist (locally
        # saved profiles), not the full built-in pool.
        _char_opts = sorted(p["name"] for p in _list_profiles("character")
                            if not _is_referee_meta(p["meta"]))
        _env_opts = sorted(p["name"] for p in _list_profiles("environment"))
        _ref_opts = sorted(p["name"] for p in _list_profiles("character")
                           if _is_referee_meta(p["meta"]))

        def _nm_select(field, options, blank=None):
            opts = ([f'<option value="">{_esc(blank)}</option>'] if blank else [])
            opts += [f'<option value="{_esc(o)}">{_esc(o)}</option>' for o in options]
            return f'<select data-nm={field}>{"".join(opts)}</select>'

        new_match = (
            '<details class=card style="margin-bottom:1rem" id="new-match">'
            '  <summary style="cursor:pointer;font-weight:600">➕ New match</summary>'
            '  <div class=row3 style="margin-top:.6rem">'
            f'    <div><label>Fighter 1</label>{_nm_select("f1", _char_opts)}</div>'
            f'    <div><label>Fighter 2</label>{_nm_select("f2", _char_opts)}</div>'
            f'    <div><label>Environment</label>'
            f'{_nm_select("env", _env_opts, blank="(random / none)")}</div>'
            '  </div>'
            '  <div class=row style="margin-top:.4rem">'
            f'    <div><label>Referee <span class=hint>(officiates the outcome '
            f'victory clips)</span></label>'
            f'{_nm_select("referee", _ref_opts, blank="(auto / any)")}</div>'
            '  </div>'
            '  <div class=pf-actions style="margin-top:.6rem">'
            '    <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
            'title="Pick a random fighter pairing + environment + referee from your '
            'existing profiles" onclick="autogenMatch()">✨ Autogenerate</button>'
            '    <button class="btn btn-primary" style="font-size:.82rem;padding:.35rem .9rem" '
            'onclick="newMatch(\'full\')">🎬 Create whole match</button>'
            '    <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
            'onclick="newMatch(\'prompts\')">📝 Create prompts only</button>'
            '    <span class="nm-status" style="font-size:.76rem;color:#7ea8f7"></span>'
            '  </div>'
            '  <p class=hint style="margin:.5rem 0 0">“Whole match” runs text→image→video '
            'end-to-end (slow). “Prompts only” writes the clip + outcome prompts so you '
            'can review them, then generate keyframes and render from the match page.</p>'
            '</details>'
        )
        new_match_js = """
<script>
async function autogenMatch(){
  const root=document.getElementById('new-match');
  const st=root.querySelector('.nm-status');
  st.style.color='#aaa'; st.textContent='✨ Picking a matchup…';
  let j;
  try{ const r=await fetch('/matches/autogen',{method:'POST',body:new FormData()}); j=await r.json(); }
  catch(e){ st.style.color='#e07070'; st.textContent='✗ '+e; return; }
  if(j.error){ st.style.color='#e07070'; st.textContent='✗ '+j.error; return; }
  const f=j.fields||{};
  ['f1','f2','env','referee'].forEach(k=>{
    const el=root.querySelector('[data-nm='+k+']');
    if(el && f[k]!=null) el.value=f[k];
  });
  st.style.color='#7ed87e'; st.textContent='✓ Filled — review, then Create';
}
async function newMatch(mode){
  const root=document.getElementById('new-match');
  const st=root.querySelector('.nm-status');
  const f1=root.querySelector('[data-nm=f1]').value;
  const f2=root.querySelector('[data-nm=f2]').value;
  const env=root.querySelector('[data-nm=env]').value;
  const ref=root.querySelector('[data-nm=referee]').value;
  if(!f1||!f2){ st.style.color='#e07070'; st.textContent='Pick two fighters'; return; }
  if(f1===f2){ st.style.color='#e07070'; st.textContent='Pick two different fighters'; return; }
  const fd=new FormData();
  fd.append('f1',f1); fd.append('f2',f2); fd.append('env',env); fd.append('mode',mode);
  fd.append('referee',ref);
  st.style.color='#aaa'; st.textContent=(mode==='prompts'?'Writing prompts…':'Creating match…');
  let j;
  try{ const r=await fetch('/matches/create',{method:'POST',body:fd}); j=await r.json(); }
  catch(e){ st.style.color='#e07070'; st.textContent='✗ '+e; return; }
  if(j.error){ st.style.color='#e07070'; st.textContent='✗ '+j.error; return; }
  pollNewMatch(j.job_id, st);
}
function pollNewMatch(jobId, st){
  st.style.color='#7ea8f7';
  const poll=async()=>{
    let d;
    try{ d=await (await fetch('/job/'+jobId)).json(); }
    catch(e){ setTimeout(poll,1500); return; }
    if(d.status==='queued'){ st.textContent=d._msg||'⏳ queued…'; setTimeout(poll,1300); }
    else if(d.status==='running'){ st.textContent='⏳ '+(d._msg||('working… '+(d.progress||0)+'%')); setTimeout(poll,1500); }
    else if(d.status==='done'){ st.style.color='#7ed87e'; st.textContent='✓ '+(d._msg||'done')+' — reloading…'; setTimeout(()=>location.reload(),1400); }
    else { st.style.color='#e07070'; st.textContent='✗ '+(d.error||'failed'); }
  };
  setTimeout(poll,800);
}
</script>"""

        return (f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<h1>Matches</h1>'
                f'<a href="/matches" class="btn btn-secondary" style="font-size:.8rem">↻ Refresh</a></div>'
                f'<p class=hint style="margin-bottom:.8rem">Select a match to view, edit and '
                f'regenerate its clips, finals and outcomes. Videos load on the detail page.</p>'
                f'{new_match}{body}{_match_js}{new_match_js}')

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

        # Selectable rosters: locally-saved profiles + the built-in pools, so a
        # match's fighters/environment can be SWITCHED before re-rendering.
        # Populate with the characters/environments that ACTUALLY exist (locally
        # saved profiles), not the full built-in pool.
        _char_opts = sorted(p["name"] for p in _list_profiles("character")
                            if not _is_referee_meta(p["meta"]))
        _env_opts = sorted(p["name"] for p in _list_profiles("environment"))
        _ref_opts = sorted(p["name"] for p in _list_profiles("character")
                           if _is_referee_meta(p["meta"]))

        def _field_select(field, current, options):
            opts = list(options)
            if current and current not in opts:
                opts = [current] + opts
            inner = "".join(
                f'<option value="{_esc(o)}"{" selected" if o == current else ""}>'
                f'{_esc(o)}</option>' for o in opts)
            return f'<select data-field={field}>{inner}</select>'

        # Referee select with a blank "(auto / any)" option so the match can leave
        # the choice to whichever referee profile is available.
        _cur_ref = meta.get("referee") or ""
        _ref_field_select = (
            '<select data-field=referee>'
            f'<option value=""{" selected" if not _cur_ref else ""}>(auto / any)</option>'
            + ("".join(
                f'<option value="{_esc(o)}"{" selected" if o == _cur_ref else ""}>'
                f'{_esc(o)}</option>'
                for o in (_ref_opts if _cur_ref in _ref_opts or not _cur_ref
                          else [_cur_ref] + _ref_opts)))
            + '</select>')

        # Keyframes live alongside the videos; used as previews / posters below.
        kdir = vdir / "keyframes"
        def _kf_path(stem):
            return kdir / f"{stem}.png"
        # The first planned clip's keyframe stands in as the poster for finals.
        _first_idx = (plan_clips[0]["idx"] if plan_clips
                      else (min(clip_files) if clip_files else None))
        _finals_poster = (_kf_path(_clip_stem_fight(name, _first_idx))
                          if _first_idx is not None else None)

        finals_html = ""
        for k in ("short", "long"):
            if k in finals:
                finals_html += (
                    f'<div style="flex:1;min-width:220px">'
                    f'<div class=hint style="margin-bottom:.2rem">{k} ({_dur_str(finals[k])})</div>'
                    f'{_vid_tag(finals[k], poster=_finals_poster)}'
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
            kfp = _kf_path(_clip_stem_fight(name, idx))
            if vp:
                vid_html = _vid_tag(vp, 120, poster=kfp)
            else:
                vid_html = (_kf_img_tag(kfp, 120)
                            or '<div class=hint>not rendered</div>')
            rm_html = (f'<button class="btn btn-danger" style="font-size:.72rem;padding:.2rem .55rem" '
                       f'onclick="delVid(event,\'clip\',{{match:\'{_esc(name)}\',idx:\'{idx}\'}})">🗑</button>'
                       if vp else '')
            clip_tiles.append(
                f'<div class=card style="width:230px">'
                f'  <div class=hint style="display:flex;justify-content:space-between;align-items:center">'
                f'<span>clip {idx:02d}</span>'
                f'<span>'
                f'<a href="#" style="color:#9fdca0" title="Rewrite ONLY this clip\'s prompt (text model). Afterwards regenerate its keyframe and re-render." '
                f'onclick="reMatch(event,\'clip-prompt\',{{match:\'{_esc(name)}\',idx:\'{idx}\'}})">prompt↻</a> '
                f'<a href="#" style="color:#c79bf0" title="Regenerate this keyframe (image model)" '
                f'onclick="reMatch(event,\'keyframe\',{{match:\'{_esc(name)}\',idx:\'{idx}\'}})">kf↻</a> '
                f'<a href="#" style="color:#7eb8f7" '
                f'onclick="reMatch(event,\'clip\',{{match:\'{_esc(name)}\',idx:\'{idx}\'}})">re-render</a> {rm_html}</span></div>'
                f'  {vid_html}'
                f'  <textarea data-clip="{idx}" rows=2 style="margin-top:.3rem">{_esc(c.get("prompt",""))}</textarea>'
                f'</div>'
            )
        clip_tiles_html = "".join(clip_tiles) or '<span class=hint>No clips planned.</span>'

        # ── Outcomes for this match (per participating fighter) ────────────────
        rendered_out = {(fr, oc): p for (fr, oc, p) in info.get("outcomes", [])}
        _mfighters = {f1, f2} - {""}
        plan_out = {(o["fighter"], o["outcome"]): o for o in plan.get("outcome_plan", [])
                    if (o.get("match_name") == name
                        or (not o.get("match_name") and o.get("fighter") in _mfighters))}
        out_fighters = [x for x in (f1, f2) if x]
        # Include any fighters that appear in rendered/planned DECISIVE outcomes but
        # not in meta (the draw is handled once for the match, below).
        for (fr, _oc) in list(rendered_out) + list(plan_out):
            if _oc != "draw" and fr not in out_fighters:
                out_fighters.append(fr)

        def _otile(fr, oc):
            """One outcome card (the per-shot textareas live under the keyframe page;
            here we show the assembled video / keyframe + the finish prompt)."""
            p = rendered_out.get((fr, oc))
            o = plan_out.get((fr, oc), {})
            kfp = _kf_path(_clip_stem_outcome(fr, oc, o.get("match_name")))
            if p:
                vid = _vid_tag(p, 110, poster=kfp)
            else:
                vid = (_kf_img_tag(kfp, 110) or '<div class=hint>not rendered</div>')
            rm = (f'<button class="btn btn-danger" style="font-size:.72rem;padding:.2rem .55rem" '
                  f'onclick="delVid(event,\'output\',{{match:\'{_esc(name)}\',fighter:\'{_esc(fr)}\',outcome:\'{_esc(oc)}\'}})">🗑</button>'
                  if p else '')
            act = "re-render" if p else "render"
            # Show the two-shot sequence (finish → victory) as a hint so it's clear
            # the outcome video is assembled from those clips.
            _roles = " → ".join(s.get("role", "?") for s in (o.get("shots") or []))
            _seq = (f'<div class=hint style="font-size:.66rem;color:#8aa">{_esc(_roles)}</div>'
                    if _roles else '')
            return (
                f'<div class=card style="width:215px">'
                f'  <div class=hint style="display:flex;justify-content:space-between;align-items:center">'
                f'<span>{_esc(oc)}</span>'
                f'<span>'
                f'<a href="#" style="color:#9fdca0" title="Rewrite ONLY this outcome\'s prompts (finish + victory shots, text model). Afterwards regenerate its keyframes and re-render." '
                f'onclick="reMatch(event,\'outcome-prompt\',{{match:\'{_esc(name)}\',fighter:\'{_esc(fr)}\',outcome:\'{_esc(oc)}\'}})">prompt↻</a> '
                f'<a href="#" style="color:#c79bf0" title="Regenerate this keyframe (image model)" '
                f'onclick="reMatch(event,\'keyframe\',{{match:\'{_esc(name)}\',fighter:\'{_esc(fr)}\',outcome:\'{_esc(oc)}\'}})">kf↻</a> '
                f'<a href="#" style="color:#7eb8f7" '
                f'onclick="reMatch(event,\'outcome\',{{match:\'{_esc(name)}\',fighter:\'{_esc(fr)}\',outcome:\'{_esc(oc)}\'}})">{act}</a> {rm}</span></div>'
                f'  {vid}{_seq}'
                f'  <textarea data-outc="{_esc(fr)}|{_esc(oc)}" rows=2 style="margin-top:.3rem">{_esc(o.get("prompt",""))}</textarea>'
                f'</div>'
            )

        outcome_groups = []
        for fr in out_fighters:
            tiles = [_otile(fr, oc) for oc in ("win", "ko_win", "retire")]
            outcome_groups.append(
                f'<div style="margin-top:.5rem"><div class=hint style="font-weight:700;color:#bbb">'
                f'{_esc(fr)}</div>'
                f'<div style="display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.25rem">{"".join(tiles)}</div></div>'
            )
        # Exactly ONE draw per match (it concerns both fighters). Show it under the
        # fighter it's stored on (the owner; defaults to f1).
        draw_owner = next((fr for (fr, oc) in list(rendered_out) + list(plan_out)
                           if oc == "draw"), f1 or (out_fighters[0] if out_fighters else ""))
        if draw_owner:
            outcome_groups.append(
                f'<div style="margin-top:.5rem"><div class=hint style="font-weight:700;color:#bbb">'
                f'Match result — draw <span style="font-weight:400">({_esc(draw_owner)})</span></div>'
                f'<div style="display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.25rem">{_otile(draw_owner, "draw")}</div></div>'
            )
        outcomes_html = "".join(outcome_groups) or '<span class=hint>No outcomes planned for this match.</span>'

        # ── Township upload state + card ────────────────────────────────────────
        _up = _get_match_upload(out_dir, name)
        _cur_sig = match_video_signature(out_dir, name, f1, f2)
        _zip_ok = bool(_up.get("zip")) and Path(_up.get("zip", "")).exists()
        _sig_match = _up.get("sig") == _cur_sig
        _prepared = _zip_ok and _sig_match and not _up.get("missing")
        _uploaded = bool(_up.get("uploaded_at")) and _sig_match
        _cfg_ok, _cfg_missing = _upload_config_ready(default_args)

        # Badge shown in the header (clears automatically when the videos change,
        # because _sig_match goes false).
        if _uploaded:
            _upload_badge = (f'<span id=upload-badge style="font-size:.74rem;color:#2ecc71;'
                             f'border:1px solid #2ecc71;border-radius:10px;padding:.1rem .55rem">'
                             f'✓ Uploaded · match #{_esc(_up.get("match_number") or "?")}</span>')
        elif _up.get("uploaded_at"):
            _upload_badge = (f'<span id=upload-badge style="font-size:.74rem;color:#e0a800;'
                             f'border:1px solid #e0a800;border-radius:10px;padding:.1rem .55rem" '
                             f'title="Videos changed since the last upload — regenerate the ZIP and re-upload">'
                             f'⚠ Upload outdated</span>')
        else:
            _upload_badge = '<span id=upload-badge></span>'

        # Odds table (if any odds have been generated).
        _odds = _up.get("odds") or {}
        if _odds:
            _ord = ["under", "over", "win1", "win2", "ko1", "ko2", "ret1", "ret2", "draw"]
            _odds_cells = "".join(
                f'<div style="display:flex;justify-content:space-between;gap:.6rem;'
                f'padding:.1rem .4rem;background:#1a1a1a;border-radius:4px">'
                f'<span class=hint>{_esc(c)}</span><b>{_odds.get(c, "—")}</b></div>'
                for c in _ord)
            _arb = _up.get("arbitrage_ok")
            _arb_html = ('<span style="color:#2ecc71">✓ no arbitrage</span>' if _arb
                         else '<span style="color:#e07070">✗ arbitrage check failed</span>')
            _odds_html = (
                f'<div style="margin-top:.5rem"><div class=hint style="margin-bottom:.25rem">'
                f'Odds {_arb_html}</div>'
                f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.3rem">{_odds_cells}</div>'
                + (f'<div class=hint style="margin-top:.3rem;color:#e0a800">ZIP missing: '
                   f'{_esc(", ".join(_up.get("missing", [])))}</div>' if _up.get("missing") else "")
                + '</div>')
        else:
            _odds_html = '<div class=hint style="margin-top:.4rem">No odds generated yet.</div>'

        if not _cfg_ok:
            _upload_cfg_hint = (f'<div class=hint style="margin-top:.4rem;color:#e0a800">'
                                f'Upload not configured (missing: {_esc(", ".join(_cfg_missing))}). '
                                f'Set the server endpoint, token and fixture ID on the Run page.</div>')
        else:
            _upload_cfg_hint = ""

        _upload_html = (
            f'<div class=card>'
            f'  <div style="display:flex;justify-content:space-between;align-items:center">'
            f'    <h2 style="margin:0">Township upload</h2>{_upload_badge}</div>'
            f'  {_odds_html}'
            f'  <div class=pf-actions style="margin-top:.6rem">'
            f'    <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="prepOdds(event,\'{_esc(name)}\')">🎲 {"Regenerate" if _odds else "Generate"} odds &amp; ZIP</button>'
            f'    <button class="btn btn-primary" style="font-size:.82rem;padding:.35rem .9rem" '
            f'id=upload-btn onclick="uploadMatch(event,\'{_esc(name)}\')"'
            f'{"" if (_prepared and _cfg_ok) else " disabled"}>⬆ Upload to Township</button>'
            f'    <span class="up-status" style="font-size:.78rem;color:#7ea8f7"></span>'
            f'  </div>'
            f'  <div class=up-progress style="display:none;margin-top:.5rem">'
            f'    <div style="height:8px;background:#222;border-radius:5px;overflow:hidden">'
            f'      <div class=up-bar style="height:100%;width:0%;background:#7a3da8;transition:width .2s"></div></div>'
            f'    <div class=up-pct class=hint style="margin-top:.2rem">0%</div>'
            f'  </div>'
            f'  {_upload_cfg_hint}'
            f'  <p class=hint style="margin:.5rem 0 0">Packs OVER/UNDER + the six win '
            f'outcomes + DRAW (highest-quality variant of each) into a ZIP, generates '
            f'arbitrage-safe odds, and uploads to the configured fixture. The ZIP upload '
            f'is chunked so it works behind a reverse proxy.</p>'
            f'</div>')

        # Render fps (clips) + the auto final playtime fps after FPS-boost.
        _detail_eff_fps = int(getattr(default_args, "playback_fps", 0)
                              or getattr(default_args, "fps", 0) or 8)
        _chosen_fps = int(meta.get("playback_fps") or _detail_eff_fps)
        _detail_mult = int(getattr(default_args, "fps_multiplier", 0) or 0)
        _auto_final_fps = _chosen_fps * _detail_mult if _detail_mult > 1 else _chosen_fps
        if _detail_mult > 1:
            _detail_fps_hint = (f"clips render at {_chosen_fps} fps; FPS-boost ×{_detail_mult} "
                                f"→ auto final {_auto_final_fps} fps (duration preserved)")
        else:
            _detail_fps_hint = "render rate for the clips (no FPS-boost active)"

        return (
            f'<div id=detail data-match="{_esc(name or "")}">{back}'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin:.4rem 0">'
            f'<h1>🥊 {_esc(title)}</h1>'
            f'<span id=detail-status style="font-size:.8rem;color:#7ea8f7"></span></div>'
            # edit meta
            f'<div class=card>'
            f'  <div class=row3>'
            f'    <div><label>Fighter 1</label>{_field_select("f1", f1, _char_opts)}</div>'
            f'    <div><label>Fighter 2</label>{_field_select("f2", f2, _char_opts)}</div>'
            f'    <div><label>Environment</label>{_field_select("env", env, _env_opts)}</div>'
            f'  </div>'
            f'  <div class=row style="margin-top:.4rem">'
            f'    <div><label>Referee <span class=hint>(officiates the outcome '
            f'victory clips)</span></label>{_ref_field_select}</div>'
            f'  </div>'
            f'  <p class=hint style="margin-top:.25rem">Switch fighters/environment, then '
            f'<b>Save match</b> and re-render. Tip: also re-render the clip prompts so '
            f'their text matches the new fighters/location.</p>'
            f'  <div class=row3 style="margin-top:.4rem">'
            f'    <div><label>Short target (s)</label><input type=number data-field=short_target '
            f'value="{_esc(meta.get("short_target",45))}"></div>'
            f'    <div><label>Long target (s)</label><input type=number data-field=long_target '
            f'value="{_esc(meta.get("long_target",70))}"></div>'
            f'    <div><label>Render FPS <span class=hint>(clips)</span></label>'
            f'<input type=number min=0 max=120 data-field=playback_fps '
            f'value="{_esc(int(meta.get("playback_fps") or _detail_eff_fps))}">'
            f'<div class=hint>{_detail_fps_hint}</div></div>'
            f'  </div>'
            f'  <div class=pf-actions style="margin-top:.6rem">'
            f'    <button class="btn btn-primary" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="saveMatch(event,\'{_esc(name)}\')">💾 Save match</button>'
            f'    <button class="btn btn-primary" style="font-size:.82rem;padding:.35rem .9rem;background:#7a3da8;border-color:#7a3da8" '
            f'onclick="reMatch(event,\'full\',{{match:\'{_esc(name)}\'}})" '
            f'title="Regenerate this whole match end to end: prompts → keyframes → clips → outcomes → finals. Uses text, image and video models.">♻ Regenerate whole match</button>'
            f'    <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="reMatch(event,\'replan\',{{match:\'{_esc(name)}\'}})" '
            f'title="Rebuild this match\'s clip list + prompts at the current fps (more, shorter clips at higher fps). No video model.">📝 Re-plan prompts</button>'
            f'    <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="reMatch(event,\'replan-outcomes\',{{match:\'{_esc(name)}\'}})" '
            f'title="Rewrite ONLY this match\'s outcome prompts (finish + victory shots). Fight-clip prompts are left untouched. No video model.">📝 Re-plan outcome prompts</button>'
            f'    <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="reMatch(event,\'match-clips\',{{match:\'{_esc(name)}\'}})">♻ Re-render all clips</button>'
            f'    <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="reMatch(event,\'reassemble\',{{match:\'{_esc(name)}\'}})">🎞 Reassemble finals</button>'
            f'    <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="reMatch(event,\'outcomes\',{{match:\'{_esc(name)}\'}})">♻ Re-render all outcomes</button>'
            f'  </div>'
            # ── Whole-match finalize/package options (applied by Regenerate) ──
            f'  <div class=pf-actions style="margin-top:.3rem;align-items:center;gap:.9rem">'
            f'    <label style="display:flex;align-items:center;gap:.35rem;font-size:.8rem;white-space:nowrap" '
            f'title="After rendering, AI-upscale ×2 + interpolate ×2 the finals/outcomes, then generate arbitrage-safe odds and pack the upload ZIP (each video picked from its highest-quality variant and renamed to its slot).">'
            f'<input type=checkbox id=full-package checked> Finalize &amp; package <span class=hint>(enhance + odds + ZIP)</span></label>'
            f'    <label style="display:flex;align-items:center;gap:.35rem;font-size:.8rem;white-space:nowrap" '
            f'title="After packaging, upload to the configured Township endpoint. Requires the upload endpoint/token/fixture to be set on the Run page; otherwise it is skipped.">'
            f'<input type=checkbox id=full-upload {"checked" if getattr(default_args, "upload_after_render", False) else ""}> Upload after <span class=hint>(needs endpoint)</span></label>'
            f'    <span class=hint>↳ applied by “♻ Regenerate whole match”.</span>'
            f'  </div>'
            f'  <div class=pf-actions style="margin-top:.4rem">'
            f'    <a class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem;text-decoration:none" '
            f'href="/match/keyframes?name={_esc(name)}">🖼 Keyframes ▸</a>'
            f'    <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="reMatch(event,\'keyframes-missing\',{{match:\'{_esc(name)}\'}})">➕ Missing keyframes</button>'
            f'    <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="reMatch(event,\'keyframes\',{{match:\'{_esc(name)}\'}})">🖼 Regenerate keyframes</button>'
            f'    <button class="btn btn-danger" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="delVid(event,\'keyframes\',{{match:\'{_esc(name)}\'}})">🧹 Clear keyframes</button>'
            f'    <button class="btn btn-danger" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="delVid(event,\'outputs\',{{match:\'{_esc(name)}\'}})" '
            f'title="Remove only this match\'s outcome videos (and their upscaled variants). Clips and finals are kept.">🗑 Remove outcome videos</button>'
            f'    <button class="btn btn-danger" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="delVid(event,\'match\',{{match:\'{_esc(name)}\'}})">🗑 Remove all videos</button>'
            f'    <button class="btn btn-danger" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="delVid(event,\'match-purge\',{{match:\'{_esc(name)}\'}})" '
            f'title="Remove this match completely — files, keyframes, and its entry in the plan">'
            f'🧨 Remove match completely</button>'
            f'  </div>'
            f'</div>'
            # ── Enhance (upscale / raise FPS) ──
            f'<div class=card style="display:flex;align-items:flex-end;gap:.6rem;flex-wrap:wrap">'
            f'  <div style="min-width:130px"><label>Upscale</label>'
            f'    <select id=enh-upscale><option value=0>none</option>'
            f'      <option value=2>2× (super-res)</option><option value=4>4× (super-res)</option></select></div>'
            f'  <div style="min-width:130px"><label>Raise FPS</label>'
            f'    <select id=enh-fps data-render-fps="{_chosen_fps}" onchange="_enhRecalcFps()">'
            f'      <option value=0>none</option>'
            f'      <option value=2>2×</option><option value=3>3×</option><option value=4>4×</option></select></div>'
            f'  <div style="min-width:120px"><label>Final FPS <span class=hint>(playtime)</span></label>'
            f'    <input type=number id=enh-final-fps min=0 max=240 value="{_chosen_fps}" style="width:100%" '
            f'title="Play rate of the FPS-boosted video. Auto-updates from Render FPS × Raise FPS when you change the multiplier; edit to override (default keeps the same length).">'
            f'  </div>'
            f'  <label style="display:flex;align-items:center;gap:.35rem;font-size:.82rem;white-space:nowrap" '
            f'title="Re-run from the original video, overwriting any existing *_2x/_NxfpS file (otherwise an up-to-date enhanced file is skipped).">'
            f'<input type=checkbox id=enh-force> Force re-enhance</label>'
            f'  <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="enhanceMatch(event,\'finals\',\'{_esc(name)}\')">✨ Enhance finals</button>'
            f'  <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="enhanceMatch(event,\'outcomes\',\'{_esc(name)}\')">✨ Enhance outcomes</button>'
            f'  <button class="btn btn-primary" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="enhanceMatch(event,\'all\',\'{_esc(name)}\')">✨ Enhance all</button>'
            f'  <button class="btn btn-danger" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="delVid(event,\'enhanced\',{{match:\'{_esc(name)}\'}})" '
            f'title="Delete only the upscaled / higher-FPS versions (the *_2x / *_NxfpS files); originals are kept.">'
            f'🗑 Remove upscaled versions</button>'
            f'  <span class=hint style="flex-basis:100%;margin-top:.1rem">Writes new '
            f'<code>*_2x</code>/<code>*_NxfpS</code> files alongside the originals (non-destructive).</span>'
            f'</div>'
            f'<div id=match-progress class=hidden></div>'
            f'{_upload_html}'
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

    def _match_keyframes_html(name=None):
        """Dedicated keyframes view for a match: a thumbnail per clip + outcome
        keyframe, each with regenerate (image model) and clear buttons, plus
        match-level regenerate-all / clear-all. Reuses the same reMatch/delVid
        endpoints as the match detail page."""
        vdir, plan, fight_by_name, matches, _legacy = _scan_matches()
        back = (f'<a href="/match?name={_esc(name or "")}" '
                f'style="font-size:.85rem">‹ Back to match</a>')
        if not name or name not in matches:
            return f'<div class=card style="color:#666">Match not found. {back}</div>'
        meta = fight_by_name.get(name, {})
        info = matches[name]
        kdir = vdir / "keyframes"
        f1, f2 = meta.get("f1", ""), meta.get("f2", "")
        title = (f"{f1} vs {f2}" if f1
                 else name.replace("match_", "").replace("_", " "))
        # Wardrobe + environment used to compose the keyframe prompt shown below
        # each thumbnail, so the user sees (and can edit) exactly what is used.
        outfits = _load_wardrobe(out_dir)
        m_env, m_env_desc = meta.get("env", ""), meta.get("env_desc", "")
        m_fighters = [x for x in (f1, f2) if x]

        def _kf_tile(label, stem, regen_params, kf_text, data_attr):
            png = kdir / f"{stem}.png"
            if png.exists() and png.stat().st_size > 0:
                url = "/media/" + str(png.relative_to(out_dir)).replace("\\", "/")
                # cache-bust on mtime so a regenerated keyframe shows immediately
                img = (f'<a href="{url}?t={int(png.stat().st_mtime)}" target=_blank>'
                       f'<img src="{url}?t={int(png.stat().st_mtime)}" loading=lazy '
                       f'alt="{_esc(stem)}" style="width:100%;border-radius:6px;'
                       f'display:block"></a>')
                clr = (f'<button class="btn btn-danger" '
                       f'style="font-size:.72rem;padding:.2rem .55rem" '
                       f'onclick="delVid(event,\'keyframe\',{regen_params})">🗑</button>')
            else:
                img = ('<div class=hint style="height:120px;display:flex;'
                       'align-items:center;justify-content:center;'
                       'background:#1b1b1b;border-radius:6px">no keyframe</div>')
                clr = ''
            return (
                f'<div class=card style="width:215px">'
                f'  <div class=hint style="display:flex;justify-content:space-between;'
                f'align-items:center;margin-bottom:.25rem">'
                f'<span>{_esc(label)}</span>'
                f'<span><a href="#" style="color:#c79bf0" '
                f'title="Regenerate this keyframe (image model)" '
                f'onclick="reMatch(event,\'keyframe\',{regen_params})">↻ regenerate</a> '
                f'{clr}</span></div>'
                f'  {img}'
                f'  <textarea {data_attr} rows=3 placeholder="(auto — wardrobe + '
                f'environment)" style="margin-top:.3rem;font-size:.72rem" '
                f'title="Prompt used to generate this keyframe. Edit + Save, then '
                f'Regenerate.">{_esc(kf_text)}</textarea>'
                f'</div>'
            )

        # Clip keyframes (saved plan order, fall back to rendered files).
        plan_clips = meta.get("clips", [])
        clip_idxs = ([c["idx"] for c in plan_clips] if plan_clips
                     else sorted(int(p.stem.split("_clip")[-1])
                                 for p in info.get("clips", [])
                                 if p.stem.split("_clip")[-1].isdigit()))
        _clip_by_idx = {c["idx"]: c for c in plan_clips}

        def _clip_kf_text(idx):
            c = _clip_by_idx.get(idx, {})
            return (c.get("kf_prompt")
                    or _compose_kf_prompt(c.get("prompt", ""), m_fighters,
                                          m_env, m_env_desc, outfits))

        clip_tiles = "".join(
            _kf_tile(f"clip {idx:02d}", _clip_stem_fight(name, idx),
                     f'{{match:\'{_esc(name)}\',idx:\'{idx}\'}}',
                     _clip_kf_text(idx), f'data-kfclip="{idx}"')
            for idx in clip_idxs) or '<span class=hint>No clips planned.</span>'

        # Outcome keyframes (per participating fighter).
        _mf = {f1, f2} - {""}
        plan_out = [o for o in plan.get("outcome_plan", [])
                    if (o.get("match_name") == name
                        or (not o.get("match_name") and o.get("fighter") in _mf))]
        out_groups = []
        for fr in [x for x in (f1, f2) if x]:
            # Only outcomes that are actually planned can be (re)generated.
            ocs = [o for o in plan_out if o.get("fighter") == fr]
            if not ocs:
                continue
            tiles = []
            for o in ocs:
                oc = o.get("outcome")
                # Use the plan entry's actual match_name (may be None → legacy
                # "<fighter>_<outcome>" stem) so this matches the file the
                # generator writes; a "or name" fallback here would look for a
                # "<match>_<fighter>_<outcome>" that never gets created.
                base_stem = _clip_stem_outcome(fr, oc, o.get("match_name"))
                # The match is authoritative for location (outcome snapshots can be
                # stale) — match _generate_keyframes so the shown default is real.
                o_env = m_env or o.get("env")
                o_envd = m_env_desc or o.get("env_desc")
                # One tile per outcome shot (finish, then victory) — each anchors its
                # own clip. Legacy outcomes without shots show a single tile.
                _shots = o.get("shots") or [None]
                _seg_labels = {"finish": "finish", "final_exchange": "exchange",
                               "victory": "victory", "draw_decision": "decision"}
                for si, s in enumerate(_shots):
                    seg_stem = _outcome_seg_stem(base_stem, si)
                    if s is not None:
                        _role = _seg_labels.get(s.get("role"), f"clip {si+1}")
                        _sp = s.get("prompt") or o.get("prompt", "")
                        _kf_over = s.get("kf_prompt")
                    else:
                        _role = ""
                        _sp = o.get("prompt", "")
                        _kf_over = o.get("kf_prompt")
                    seg_label = f"{oc} · {_role}" if _role else oc
                    kf_text = (_kf_over
                               or _compose_kf_prompt(_sp, m_fighters,
                                                     o_env, o_envd, outfits))
                    tiles.append(_kf_tile(
                        seg_label, seg_stem,
                        f'{{match:\'{_esc(name)}\',fighter:\'{_esc(fr)}\','
                        f'outcome:\'{_esc(oc)}\',seg:\'{si}\'}}',
                        kf_text, f'data-kfoutc="{_esc(fr)}|{_esc(oc)}|{si}"'))
            out_groups.append(
                f'<div style="margin-top:.5rem"><div class=hint '
                f'style="font-weight:700;color:#bbb">{_esc(fr)}</div>'
                f'<div style="display:flex;gap:.5rem;flex-wrap:wrap;'
                f'margin-top:.25rem">{"".join(tiles)}</div></div>')
        outcomes_html = ("".join(out_groups)
                         or '<span class=hint>No outcomes for this match.</span>')

        return (
            f'<div id=detail data-match="{_esc(name)}">{back}'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;margin:.4rem 0">'
            f'<h1>🖼 Keyframes — {_esc(title)}</h1>'
            f'<span id=detail-status style="font-size:.8rem;color:#7ea8f7"></span></div>'
            f'<div class=card>'
            f'  <p class=hint style="margin:0 0 .5rem">Keyframes are the image→video '
            f'bridge stills. The text under each thumbnail is the <b>prompt used to '
            f'generate that keyframe</b> — edit it, hit <b>Save keyframe prompts</b>, '
            f'then <b>Regenerate</b>. Empty boxes are auto-composed from each '
            f'fighter\'s locked wardrobe (<code>wardrobe.json</code>) + the match '
            f'environment, so clothing colours and location stay consistent. After '
            f'regenerating, <b>Re-render</b> the matching clip on the match page.</p>'
            f'  <div class=pf-actions>'
            f'    <button class="btn btn-primary" style="font-size:.82rem;padding:.35rem .9rem;background:#2d7a4a;border-color:#2d7a4a" '
            f'onclick="saveKfPrompts(event,\'{_esc(name)}\')">💾 Save keyframe prompts</button>'
            f'    <button class="btn btn-primary" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="reMatch(event,\'keyframes-missing\',{{match:\'{_esc(name)}\'}})">➕ Generate missing</button>'
            f'    <button class="btn btn-secondary" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="reMatch(event,\'keyframes\',{{match:\'{_esc(name)}\'}})">🖼 Regenerate all</button>'
            f'    <button class="btn btn-danger" style="font-size:.82rem;padding:.35rem .9rem" '
            f'onclick="delVid(event,\'keyframes\',{{match:\'{_esc(name)}\'}})">🧹 Clear all</button>'
            f'  </div>'
            f'</div>'
            f'<div id=match-progress class=hidden></div>'
            f'<div class=section-title style="margin:.7rem 0 .3rem">Clip keyframes</div>'
            f'<div style="display:flex;gap:.5rem;flex-wrap:wrap">{clip_tiles}</div>'
            f'<div class=section-title style="margin:.9rem 0 .3rem">Outcome keyframes</div>'
            f'{outcomes_html}'
            f'</div>{_match_js}{_kf_prompt_js}'
        )

    def _wardrobe_html():
        """Edit each fighter's locked outfit (wardrobe.json). Every keyframe of a
        match dresses the fighter in this exact phrase, so editing it here fixes
        clothing colours/garments consistently across the whole match. Colour
        clashes (two fighters sharing a colour) are flagged."""
        wardrobe = _load_wardrobe(out_dir)

        # Which palette/colour word each outfit uses, to flag clashes.
        def _color_of(o):
            ol = str(o or "").lower()
            for c in _OUTFIT_COLORS:
                if c in ol:
                    return c
            for w in ol.split():
                if w in _COLOR_WORDS:
                    return w
            return None

        color_of = {nm: _color_of(o) for nm, o in wardrobe.items()}
        from collections import Counter as _Counter
        counts = _Counter(c for c in color_of.values() if c)
        clashes = {c for c, n in counts.items() if n > 1}

        rows = []
        for nm in sorted(wardrobe):
            o = wardrobe[nm]
            clash = color_of.get(nm) in clashes
            warn = (' <span class=hint style="color:#e0a060">⚠ shares colour</span>'
                    if clash else '')
            bstyle = 'border-color:#e0a060;' if clash else ''
            rows.append(
                f'<div class=card style="display:flex;gap:.7rem;align-items:center;'
                f'padding:.5rem .7rem">'
                f'  <label style="min-width:150px;font-weight:700;margin:0">'
                f'{_esc(nm)}</label>'
                f'  <input data-wf="{_esc(nm)}" value="{_esc(o)}" '
                f'style="flex:1;{bstyle}">{warn}'
                f'</div>')
        rows_html = "".join(rows) or '<span class=hint>No fighters found.</span>'

        clash_note = ''
        if clashes:
            clash_note = (
                f'<div class=card style="border-color:#e0a060">'
                f'<b style="color:#e0a060">⚠ {len(clashes)} colour clash'
                f'{"es" if len(clashes) > 1 else ""}</b> — two or more fighters '
                f'share a colour, which makes the image model swap their kits in a '
                f'match. Give them distinct colours, or hit '
                f'<b>Reshuffle distinct colours</b>.</div>')

        body = (
            f'<div id=wf-root>'
            f'{clash_note}'
            f'<div style="display:flex;flex-direction:column;gap:.4rem">{rows_html}</div>'
            f'<div class=pf-actions style="margin-top:.7rem">'
            f'  <button class="btn btn-primary" style="font-size:.85rem;padding:.4rem 1rem" '
            f'onclick="saveWardrobe(event)">💾 Save wardrobe</button>'
            f'  <button class="btn btn-secondary" style="font-size:.85rem;padding:.4rem 1rem" '
            f'onclick="reshuffleWardrobe(event)" '
            f'title="Give every fighter a distinct colour, keeping their garment">'
            f'🎨 Reshuffle distinct colours</button>'
            f'  <span id=wf-status style="font-size:.8rem;color:#7ea8f7"></span>'
            f'</div>'
            f'</div>'
        )

        script = """
<script>
function _gatherWardrobe(){
  const m={};
  document.querySelectorAll('[data-wf]').forEach(el=>{
    const v=el.value.trim(); if(v) m[el.getAttribute('data-wf')]=v; });
  return m;
}
async function saveWardrobe(ev){
  const st=document.getElementById('wf-status');
  const set=(c,t)=>{st.style.color=c;st.textContent=t;};
  set('#aaa','Saving…');
  const fd=new FormData(); fd.append('wardrobe', JSON.stringify(_gatherWardrobe()));
  try{
    const j=await (await fetch('/wardrobe/save',{method:'POST',body:fd})).json();
    if(j.error){ set('#e07070','✗ '+j.error); return; }
    set('#7ed87e','✓ Saved — regenerate the match keyframes to apply');
    setTimeout(()=>location.reload(),900);
  }catch(e){ set('#e07070','✗ '+e); }
}
async function reshuffleWardrobe(ev){
  if(!(await uiConfirm('Give every fighter a distinct colour (keeping each garment)? '
      +'This overwrites colours you set manually.',
      {title:'Reshuffle colours', okText:'Reshuffle'})))return;
  const st=document.getElementById('wf-status');
  const fd=new FormData(); fd.append('reshuffle','1');
  try{
    const j=await (await fetch('/wardrobe/save',{method:'POST',body:fd})).json();
    if(j.error){ st.style.color='#e07070'; st.textContent='✗ '+j.error; return; }
    location.reload();
  }catch(e){ st.style.color='#e07070'; st.textContent='✗ '+e; }
}
</script>"""

        return (f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<h1>👕 Wardrobe</h1></div>'
                f'<p class=hint style="margin-bottom:.8rem">Each fighter\'s <b>locked '
                f'outfit</b>. The same phrase is written into every keyframe of a match, '
                f'so clothing colours stay consistent shot to shot. Include a colour '
                f'(e.g. <code>navy blue boxing shorts</code>) and keep the two fighters '
                f'in a match clearly different. After editing, <b>Regenerate keyframes</b> '
                f'on the match to apply.</p>'
                f'{body}{script}')

    def _prompts_html():
        """Edit the global prompt templates used by the script (LLM system
        prompts + static fallback shot/outcome templates)."""
        cfg = prompts_config_snapshot()  # live values (defaults + any overrides)

        def ta(field, value, rows=3):
            return (f'<textarea data-pf="{_esc(field)}" rows={rows}>'
                    f'{_esc(value)}</textarea>')

        wst = cfg["win_shot_templates"]
        fnt = cfg.get("finish_shot_templates", {})
        outcome_blocks = "".join(
            f'<label>{_esc(k)} — FINISH templates <span class=hint>(decisive ending action; one per line)</span></label>'
            f'{ta("finish::"+k, chr(10).join(fnt.get(k, [])), 4)}'
            f'<label>{_esc(k)} — VICTORY templates <span class=hint>(referee raising the arm; one per line)</span></label>'
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
  const cfg={llm_system:'',llm_outcome_system:'',fight_shot_templates:[],win_shot_templates:{},finish_shot_templates:{}};
  root.querySelectorAll('[data-pf]').forEach(el=>{
    const f=el.getAttribute('data-pf'); const v=el.value;
    if(f==='llm_system'||f==='llm_outcome_system'){ cfg[f]=v; }
    else if(f==='fight_shot_templates'){ cfg.fight_shot_templates=v.split('\\n').map(s=>s.trim()).filter(Boolean); }
    else if(f.startsWith('finish::')){ cfg.finish_shot_templates[f.slice(8)]=v.split('\\n').map(s=>s.trim()).filter(Boolean); }
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

        def _public_prefix(self):
            """Reverse-proxy sub-path mount prefix (e.g. '/township'), or ''."""
            p = (self.headers.get("X-Forwarded-Prefix")
                 or self.headers.get("X-Script-Name") or "")
            p = p.strip().rstrip("/")
            return p if p.startswith("/") else (("/" + p) if p else "")

        def _route(self, path):
            """Strip the forwarded prefix so internal routing is mount-agnostic
            whether or not nginx already stripped it."""
            pref = self._public_prefix()
            if pref and (path == pref or path.startswith(pref + "/")):
                path = path[len(pref):] or "/"
            return path

        def _send(self, code, ctype, body):
            if isinstance(body, str): body = body.encode()
            if "text/html" in ctype:
                pref = self._public_prefix()
                if pref:
                    body = _mount_html(body.decode("utf-8", "replace"), pref).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = self._route(parsed.path).rstrip("/") or "/"

            if path == "/favicon.ico":
                # Bundled icon next to this script (tools/assets/favicon.ico).
                ico = Path(__file__).resolve().parent / "assets" / "favicon.ico"
                if ico.exists():
                    self._send(200, "image/x-icon", ico.read_bytes())
                else:
                    self._send(404, "text/plain; charset=utf-8", b"no favicon")

            elif path in ("/", ""):
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

            elif path == "/match/keyframes":
                qs = urllib.parse.parse_qs(parsed.query)
                nm = qs.get("name", [None])[0]
                html = _page("Keyframes", _match_keyframes_html(nm), "matches")
                self._send(200, "text/html; charset=utf-8", html)

            elif path == "/wardrobe":
                html = _page("Wardrobe", _wardrobe_html(), "wardrobe")
                self._send(200, "text/html; charset=utf-8", html)

            elif path == "/prompts":
                html = _page("Prompts", _prompts_html(), "prompts")
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
                # Running AND queued generation jobs, so a reloaded page (or the
                # match-detail progress monitor) can re-attach to in-flight work and
                # show what is queued behind it. Includes the per-item bars so the
                # monitor can render real progress without binding to one job id.
                import json as _j
                active = []
                with _jobs_lock:
                    for jid, j in _state["jobs"].items():
                        if (j.get("status") in ("running", "queued")
                                and j.get("jtype") in ("regen", "create", "train",
                                                       "match", "process")):
                            active.append({
                                "job_id": jid,
                                "kind": j.get("kind"),
                                "name": j.get("name"),
                                "jtype": j.get("jtype"),
                                "target": j.get("target"),
                                "scope": j.get("scope"),
                                "match": j.get("match"),
                                "status": j.get("status"),
                                "progress": j.get("progress", 0),
                                "items": j.get("items"),
                                "cancellable": j.get("cancellable"),
                                "cancelled": j.get("cancelled"),
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
            path = self._route(parsed.path)

            if path == "/stop":
                _state["abort"].set()
                import json as _j
                self._send(200, "application/json", _j.dumps({"ok": True}))
                return

            # Cancel a single in-flight job (graceful: stops after the current
            # clip/keyframe/output). The job's loop polls this flag.
            if path == "/job/cancel":
                import json as _j
                clen = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(clen)
                form = dict(urllib.parse.parse_qsl(raw.decode(errors="replace")))
                jid = (form.get("job_id") or "").strip()
                with _jobs_lock:
                    j = _state["jobs"].get(jid)
                    if j is None:
                        self._send(404, "application/json", _j.dumps({"error": "no such job"})); return
                    if j.get("status") == "running":
                        j["cancel"] = True
                        j["_msg"] = "⏹ cancelling — finishing the current item…"
                self._send(200, "application/json", _j.dumps({"ok": True}))
                return

            if path == "/job/cancel-all":
                # Cancel the WHOLE generation queue, not just the active job: flag the
                # running job to stop after its current item AND every job still
                # waiting in the queue so the worker skips them. Optionally scoped to
                # one match (`match` form field); with no match, cancels everything.
                import json as _j
                clen = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(clen) if clen else b""
                form = dict(urllib.parse.parse_qsl(raw.decode(errors="replace")))
                only_match = (form.get("match") or "").strip()
                n = 0
                with _jobs_lock:
                    for j in _state["jobs"].values():
                        if j.get("status") not in ("running", "queued"):
                            continue
                        if only_match and (j.get("match") or "") != only_match:
                            continue
                        j["cancel"] = True
                        j["_msg"] = ("⏹ cancelling — finishing the current item…"
                                     if j.get("status") == "running"
                                     else "⏹ cancelled — will be skipped")
                        n += 1
                self._send(200, "application/json", _j.dumps({"ok": True, "cancelled": n}))
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
                _enqueue_gen(job_id, _run_regen_job, job_id, kind, name, count, guide,
                             jtype="regen", kind=kind, name=name)
                self._send(200, "application/json", _j.dumps({"job_id": job_id}))
                return

            if path == "/profile/create":
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

                kind = _fv("kind"); name = _fv("name").strip()
                if (kind not in ("character", "environment") or not name
                        or "/" in name or "\\" in name or ".." in name):
                    self._send(400, "application/json",
                               _j.dumps({"error": "invalid kind/name"}))
                    return
                if (out_dir / (kind + "s") / name).exists():
                    self._send(400, "application/json",
                               _j.dumps({"error": "a profile with that name already exists"}))
                    return
                try:
                    count = max(1, min(8, int(_fv("count", "4") or 4)))
                except ValueError:
                    count = 4
                meta = {"prompt": _fv("prompt"), "description": _fv("description"),
                        "region": _fv("region")}
                if kind == "character":
                    meta["gender"] = _fv("gender", "male") or "male"
                    # A referee is a character profile tagged so it's kept out of
                    # the fighter pools and dressed as an official.
                    if _fv("role", "fighter").strip().lower() == "referee":
                        meta["role"] = "referee"
                job_id = _u.uuid4().hex[:12]
                _enqueue_gen(job_id, _run_create_profile_job, job_id, kind, name,
                             meta, count, jtype="create", kind=kind, name=name)
                self._send(200, "application/json", _j.dumps({"job_id": job_id}))
                return

            if path == "/profile/autogen":
                # Invent a fresh profile's form fields (name/region/description/
                # prompt[/gender]) so the create form can be one-click filled and
                # then reviewed/edited before submitting. Synchronous + fast.
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

                kind = _fv("kind")
                role = _fv("role", "fighter")
                if kind not in ("character", "environment"):
                    self._send(400, "application/json",
                               _j.dumps({"error": "invalid kind"})); return
                existing = [p["name"] for p in _list_profiles(kind)]
                client = CoderAIClient(default_args.base_url,
                                       getattr(default_args, "api_key", None))
                text_model = None
                if not getattr(default_args, "no_llm", False):
                    try:
                        text_model = pick_model(client, "text",
                                                getattr(default_args, "text_model", None))
                    except Exception:
                        text_model = getattr(default_args, "text_model", None)
                try:
                    fields = _autogen_profile_payload(client, text_model, kind,
                                                      role, existing)
                except Exception as e:
                    self._send(500, "application/json",
                               _j.dumps({"error": f"autogen failed: {e}"})); return
                self._send(200, "application/json",
                           _j.dumps({"ok": True, "fields": fields}))
                return

            if path == "/matches/autogen":
                # Pick a fresh fighter pairing (+ environment + referee) from the
                # EXISTING profiles so the new-match form can be one-click filled.
                import json as _j
                clen = int(self.headers.get("Content-Length", 0))
                self.rfile.read(clen)  # body unused
                _fighters = [p["name"] for p in _list_profiles("character")
                             if not _is_referee_meta(p["meta"])]
                _envs = [p["name"] for p in _list_profiles("environment")]
                _refs = [p["name"] for p in _list_profiles("character")
                         if _is_referee_meta(p["meta"])]
                if len(_fighters) < 2:
                    self._send(400, "application/json",
                               _j.dumps({"error": "need at least 2 fighter profiles"})); return
                _pair = random.sample(_fighters, 2)
                fields = {"f1": _pair[0], "f2": _pair[1],
                          "env": (random.choice(_envs) if _envs else ""),
                          "referee": (random.choice(_refs) if _refs else "")}
                self._send(200, "application/json",
                           _j.dumps({"ok": True, "fields": fields}))
                return

            if path == "/matches/create":
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

                f1 = _fv("f1").strip(); f2 = _fv("f2").strip(); env = _fv("env").strip()
                referee = _fv("referee").strip()
                for _x in (f1, f2, env, referee):
                    if "/" in _x or "\\" in _x or ".." in _x:
                        self._send(400, "application/json",
                                   _j.dumps({"error": "invalid name"}))
                        return
                if not f1 or not f2:
                    self._send(400, "application/json",
                               _j.dumps({"error": "two fighters required"}))
                    return
                if f1 == f2:
                    self._send(400, "application/json",
                               _j.dumps({"error": "pick two different fighters"}))
                    return
                prompts_only = _fv("mode") == "prompts"
                job_id = _u.uuid4().hex[:12]
                _enqueue_gen(job_id, _run_new_match_job, job_id, f1, f2, env,
                             prompts_only, referee, jtype="match",
                             match=None, scope=("prompts" if prompts_only else "full"),
                             cancellable=True)
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
                target = "video" if _fv("target") == "video" else "image"
                job_id = _u.uuid4().hex[:12]
                _enqueue_gen(job_id, _run_train_lora_job, job_id, kind, name, steps,
                             rank, target, jtype="train", kind=kind, name=name)
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
                if scope not in ("full", "match-clips", "clip", "reassemble", "outcomes",
                                 "outcome", "enhance", "keyframes",
                                 "keyframes-missing", "keyframe", "replan",
                                 "replan-outcomes", "clip-prompt", "outcome-prompt"):
                    self._send(400, "application/json",
                               _j.dumps({"error": "invalid scope"}))
                    return
                params = {}
                for k in ("match", "idx", "fighter", "outcome", "seg", "target", "upscale", "fps", "force", "final_fps", "package", "upload"):
                    val = _fv(k)
                    if val:
                        # Guard path-like fields against traversal.
                        if k in ("match", "fighter") and ("/" in val or "\\" in val or ".." in val):
                            self._send(400, "application/json",
                                       _j.dumps({"error": f"invalid {k}"}))
                            return
                        params[k] = val
                job_id = _u.uuid4().hex[:12]
                _enqueue_gen(job_id, _run_match_job, job_id, scope, params,
                             jtype="match", match=params.get("match"), scope=scope,
                             cancellable=True)
                self._send(200, "application/json", _j.dumps({"job_id": job_id}))
                return

            if path in ("/match/odds", "/match/upload"):
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

                name = _fv("name")
                if not name or "/" in name or "\\" in name or ".." in name:
                    self._send(400, "application/json", _j.dumps({"error": "invalid match name"}))
                    return

                if path == "/match/odds":
                    # Generate arbitrage-safe odds + pack the ZIP (fast, synchronous).
                    try:
                        res = prepare_match_odds_zip(out_dir, name, default_args, log=_web_log)
                    except Exception as e:
                        self._send(500, "application/json", _j.dumps({"error": str(e)}))
                        return
                    self._send(200, "application/json", _j.dumps({
                        "ok": res["ok"], "odds": res["odds"], "missing": res["missing"],
                        "arbitrage_ok": res["arbitrage_ok"], "error": res["error"],
                    }))
                    return

                # /match/upload — run in a background thread with a progress job so
                # the page can render a progress bar (uploads can be large/slow).
                job_id = _u.uuid4().hex[:12]
                with _jobs_lock:
                    _state["jobs"][job_id] = {
                        "status": "running", "progress": 0, "error": None,
                        "jtype": "upload", "match": name, "_msg": "starting…"}

                def _upload_worker(_jid=job_id, _name=name):
                    def _pcb(frac, label):
                        with _jobs_lock:
                            j = _state["jobs"].get(_jid)
                            if j:
                                j["progress"] = max(0, min(100, int(frac * 100)))
                                j["_msg"] = label
                    try:
                        res = upload_prepared_match(out_dir, _name, default_args,
                                                    log=_web_log, progress_cb=_pcb)
                        with _jobs_lock:
                            j = _state["jobs"].get(_jid)
                            if j:
                                if res["ok"]:
                                    j.update({"status": "done", "progress": 100,
                                              "_msg": f"uploaded — match #{res.get('match_number')}",
                                              "match_number": res.get("match_number")})
                                else:
                                    j.update({"status": "error", "error": res["error"],
                                              "_msg": f"✗ {res['error']}"})
                    except Exception as e:
                        with _jobs_lock:
                            j = _state["jobs"].get(_jid)
                            if j:
                                j.update({"status": "error", "error": str(e), "_msg": f"✗ {e}"})

                threading.Thread(target=_upload_worker, daemon=True,
                                 name=f"upload-{name}").start()
                self._send(200, "application/json", _j.dumps({"job_id": job_id}))
                return

            if path == "/wardrobe/save":
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

                if _fv("reshuffle"):
                    try:
                        _reshuffle_wardrobe(out_dir)
                    except Exception as e:
                        self._send(500, "application/json", _j.dumps({"error": f"cannot reshuffle: {e}"}))
                        return
                    self._send(200, "application/json", _j.dumps({"ok": True}))
                    return
                try:
                    wmap = _j.loads(_fv("wardrobe", "{}"))
                except Exception as e:
                    self._send(400, "application/json", _j.dumps({"error": f"bad wardrobe: {e}"}))
                    return
                if not isinstance(wmap, dict):
                    self._send(400, "application/json", _j.dumps({"error": "wardrobe must be an object"}))
                    return
                try:
                    _save_wardrobe(out_dir, wmap)
                except Exception as e:
                    self._send(500, "application/json", _j.dumps({"error": f"cannot save: {e}"}))
                    return
                self._send(200, "application/json", _j.dumps({"ok": True}))
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
                        if "referee" in form:
                            # Blank "(auto / any)" clears the explicit choice.
                            _ref = _fv("referee").strip()
                            if _ref and not _safe(_ref):
                                self._send(400, "application/json", _j.dumps({"error": "invalid referee"})); return
                            m["referee"] = _ref or None
                        if "env" in form:
                            _new_env = _fv("env") or None
                            if _new_env != m.get("env"):
                                m["env"] = _new_env
                                m["env_desc"] = (_env_description(_new_env)
                                                 if _new_env else "African township")
                                # Keep this match's outcome entries in sync so their
                                # keyframes/renders use the new location too.
                                for _o in data.get("outcome_plan", []):
                                    if _o.get("match_name") == nm:
                                        _o["env"] = m["env"]
                                        _o["env_desc"] = m["env_desc"]
                        for _tk in ("short_target", "long_target"):
                            if _tk in form:
                                try: m[_tk] = float(_fv(_tk))
                                except ValueError: pass
                        if "playback_fps" in form:
                            try:
                                _pf = int(float(_fv("playback_fps") or 0))
                                if _pf > 0:
                                    m["playback_fps"] = _pf
                                else:
                                    m.pop("playback_fps", None)
                            except ValueError:
                                pass
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
                            # Keep the FINISH shot in sync so the multi-clip render
                            # picks up the manual edit (the first segment's prompt).
                            if isinstance(o.get("shots"), list) and o["shots"]:
                                o["shots"][0]["prompt"] = o["prompt"]
                    elif mode == "outputs":
                        fr = _fv("fighter")
                        if not _safe(fr):
                            self._send(400, "application/json", _j.dumps({"error": "invalid fighter"})); return
                        for o in data.get("outcome_plan", []):
                            if o.get("fighter") == fr:
                                key = f"out_{o['outcome']}"
                                if key in form:
                                    o["prompt"] = _fv(key)
                                    if isinstance(o.get("shots"), list) and o["shots"]:
                                        o["shots"][0]["prompt"] = o["prompt"]
                    elif mode == "kfprompts":
                        # Per-keyframe prompt overrides. A submitted value equal to
                        # the freshly-composed default (or blank) CLEARS the override,
                        # so that keyframe keeps auto-tracking wardrobe/environment.
                        nm = _fv("name")
                        if not _safe(nm):
                            self._send(400, "application/json", _j.dumps({"error": "invalid name"})); return
                        m = next((x for x in data.get("fight_plan", []) if x.get("match_name") == nm), None)
                        if not m:
                            self._send(404, "application/json", _j.dumps({"error": "match not in prompts.json"})); return
                        _outfits = _load_wardrobe(out_dir)
                        _mf = [x for x in (m.get("f1"), m.get("f2")) if x]

                        def _apply_kf(entry, base, env, env_desc, submitted):
                            default = _compose_kf_prompt(base, _mf, env, env_desc, _outfits)
                            if not submitted.strip() or submitted.strip() == default.strip():
                                entry.pop("kf_prompt", None)
                            else:
                                entry["kf_prompt"] = submitted

                        for c in m.get("clips", []):
                            key = f"kfclip_{c['idx']}"
                            if key in form:
                                _apply_kf(c, c.get("prompt", ""), m.get("env"),
                                          m.get("env_desc"), _fv(key))
                        for fk in list(form.keys()):
                            if not fk.startswith("kfoutc_"):
                                continue
                            # Key is "<fighter>|<outcome>[|<seg>]" — seg selects which
                            # outcome shot's keyframe prompt (finish/victory) to set.
                            _payload = fk[7:]
                            _seg_i = None
                            _rest, _sep, _tail = _payload.rpartition("|")
                            if _sep and _tail.isdigit():
                                _seg_i = int(_tail)
                                _payload = _rest
                            fr_o, _, oc_o = _payload.rpartition("|")
                            if not fr_o or not oc_o:
                                continue
                            o = next((x for x in data.get("outcome_plan", [])
                                      if x.get("fighter") == fr_o
                                      and x.get("outcome") == oc_o
                                      and (x.get("match_name") in (None, nm))), None)
                            if o is None:
                                continue
                            _env = m.get("env") or o.get("env")
                            _envd = m.get("env_desc") or o.get("env_desc")
                            _shots = o.get("shots") or None
                            if _seg_i is not None and _shots and 0 <= _seg_i < len(_shots):
                                _apply_kf(_shots[_seg_i],
                                          _shots[_seg_i].get("prompt", ""),
                                          _env, _envd, _fv(fk))
                                if _seg_i == 0:
                                    # Mirror seg 0's override onto the outcome entry so
                                    # legacy single-keyframe paths stay consistent.
                                    o["kf_prompt"] = _shots[0].get("kf_prompt")
                                    if o["kf_prompt"] is None:
                                        o.pop("kf_prompt", None)
                            else:
                                _apply_kf(o, o.get("prompt", ""), _env, _envd, _fv(fk))
                    else:
                        self._send(400, "application/json", _j.dumps({"error": "invalid mode"})); return
                    try:
                        vdir.mkdir(parents=True, exist_ok=True)
                        prompts_file.write_text(json.dumps(data, indent=2))
                    except Exception as e:
                        self._send(500, "application/json", _j.dumps({"error": f"cannot save: {e}"})); return
                    # Editing a match invalidates any prepared/uploaded ZIP for it:
                    # drop the upload state so the "Uploaded" badge disappears and a
                    # fresh odds/ZIP must be generated before re-uploading.
                    if mode in ("match", "outputs", "kfprompts"):
                        _nm = _fv("name")
                        if _nm and _safe(_nm):
                            _clear_match_upload(out_dir, _nm)
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

                def _rm_video(p: Path):
                    # Remove a base video AND its enhanced (upscaled / fps) siblings,
                    # so deleting a video never leaves an orphaned *_2x / *_NxfpS file.
                    for _lbl, _vp in _video_variants(p):
                        _rm(_vp)

                if scope == "clip":
                    mn, idx = _fv("match"), _fv("idx")
                    if not _safe(mn):
                        self._send(400, "application/json", _j.dumps({"error": "invalid match"})); return
                    try:
                        _rm_video(vdir / f"{mn}_clip{int(idx):02d}.mp4")
                    except ValueError:
                        pass
                elif scope == "final":
                    mn, which = _fv("match"), _fv("which")
                    if not _safe(mn) or which not in ("short", "long"):
                        self._send(400, "application/json", _j.dumps({"error": "invalid args"})); return
                    _rm_video(vdir / f"{mn}_{which}.mp4")
                elif scope == "match":
                    mn = _fv("match")
                    if not _safe(mn):
                        self._send(400, "application/json", _j.dumps({"error": "invalid match"})); return
                    # Use the scanner so a sibling match (e.g. "<mn>_2") is not
                    # caught by a naive "<mn>_*" glob. _rm_video also sweeps each
                    # file's enhanced (upscaled / fps) siblings.
                    _, _, _, _matches_map, _ = _scan_matches()
                    _info = _matches_map.get(mn, {})
                    for _p in list(_info.get("finals", {}).values()):
                        _rm_video(_p)
                    for _p in _info.get("clips", []):
                        _rm_video(_p)
                    for (_f, _o, _p) in _info.get("outcomes", []):
                        _rm_video(_p)
                elif scope == "output":
                    mn, fr, oc = _fv("match"), _fv("fighter"), _fv("outcome")
                    if not _safe(fr) or not _safe(oc):
                        self._send(400, "application/json", _j.dumps({"error": "invalid args"})); return
                    if mn and _safe(mn):
                        _rm_video(vdir / f"{mn}_{fr}_{oc}.mp4")        # per-match outcome
                    else:
                        _rm_video(vdir / f"{fr}_{oc}.mp4")             # legacy per-fighter
                elif scope == "outputs":
                    mn, fr = _fv("match"), _fv("fighter")
                    if mn and _safe(mn):
                        _, _, _, _matches_map, _ = _scan_matches()
                        for (_f, _o, _p) in _matches_map.get(mn, {}).get("outcomes", []):
                            _rm_video(_p)
                    elif fr and _safe(fr):
                        for p in vdir.glob(f"{fr}_*.mp4"):
                            if "_clip" not in p.stem and not p.stem.startswith("match_"):
                                _rm_video(p)
                    else:
                        self._send(400, "application/json", _j.dumps({"error": "invalid args"})); return
                elif scope == "file":
                    fn = _fv("file")
                    if not _safe(fn) or not fn.endswith(".mp4"):
                        self._send(400, "application/json", _j.dumps({"error": "invalid file"})); return
                    _rm_video(vdir / fn)
                elif scope == "keyframes":
                    # Clear ALL keyframe PNGs for a match (its clips + its outcomes,
                    # whether or not the matching video was rendered).
                    mn = _fv("match")
                    if not _safe(mn):
                        self._send(400, "application/json", _j.dumps({"error": "invalid match"})); return
                    kdir = vdir / "keyframes"
                    for p in kdir.glob(f"{mn}_clip*.png"):
                        _rm(p)
                    _, _plan, _fbn, _matches_map, _ = _scan_matches()
                    # Rendered outcomes (have an mp4) — clear the base keyframe AND
                    # its per-segment siblings (finish + victory).
                    for (_f, _o, _p) in _matches_map.get(mn, {}).get("outcomes", []):
                        _base = Path(_p).stem
                        _rm(kdir / f"{_base}.png")
                        for _sp in kdir.glob(f"{_base}_s*.png"):
                            _rm(_sp)
                    # Planned outcomes (may have a keyframe but no rendered video).
                    _meta = _fbn.get(mn, {})
                    _mf = {_meta.get("f1"), _meta.get("f2")} - {None}
                    for o in _plan.get("outcome_plan", []):
                        if o.get("match_name") == mn or (
                                not o.get("match_name") and o.get("fighter") in _mf):
                            for _st in _outcome_kf_stems(o):
                                _rm(kdir / f"{_st}.png")
                elif scope == "keyframe":
                    # Clear one clip's (match+idx) or one outcome's (fighter+outcome) keyframe.
                    mn = _fv("match")
                    kdir = vdir / "keyframes"
                    fr, oc = _fv("fighter"), _fv("outcome")
                    if fr and oc:
                        if not _safe(fr) or not _safe(oc):
                            self._send(400, "application/json", _j.dumps({"error": "invalid args"})); return
                        stem = f"{mn}_{fr}_{oc}" if (mn and _safe(mn)) else f"{fr}_{oc}"
                        seg = _fv("seg")
                        if seg is not None and str(seg) != "":
                            # Clear just one segment's keyframe (finish or victory).
                            try:
                                si = int(seg)
                            except (TypeError, ValueError):
                                si = -1
                            _rm(kdir / f"{_outcome_seg_stem(stem, si)}.png")
                        else:
                            # Clear the base keyframe AND its per-segment siblings.
                            _rm(kdir / f"{stem}.png")
                            for _sp in kdir.glob(f"{stem}_s*.png"):
                                _rm(_sp)
                    else:
                        idx = _fv("idx")
                        if not _safe(mn):
                            self._send(400, "application/json", _j.dumps({"error": "invalid match"})); return
                        try:
                            _rm(kdir / f"{mn}_clip{int(idx):02d}.png")
                        except ValueError:
                            self._send(400, "application/json", _j.dumps({"error": "invalid idx"})); return
                elif scope == "enhanced":
                    # Remove ONLY the post-process enhanced variants (the
                    # *_2x / *_NxfpS files) for a match — finals, clips and
                    # outcomes — keeping every original intact.
                    mn = _fv("match")
                    if not _safe(mn):
                        self._send(400, "application/json", _j.dumps({"error": "invalid match"})); return
                    _, _, _, _matches_map, _ = _scan_matches()
                    _info = _matches_map.get(mn, {})
                    _bases = list(_info.get("finals", {}).values())
                    _bases += list(_info.get("clips", []))
                    _bases += [_p for (_f, _o, _p) in _info.get("outcomes", [])]
                    for _base in _bases:
                        # _video_variants returns the original first, then enhanced
                        # siblings — drop everything except the original.
                        for _lbl, _vp in _video_variants(_base)[1:]:
                            _rm(_vp)
                elif scope == "match-purge":
                    # Remove a match COMPLETELY: every video file, every keyframe,
                    # AND its entry in prompts.json (the plan + this match's
                    # per-match outcomes). Legacy GLOBAL per-fighter outcomes are
                    # left intact — they're shared with the fighter's other matches.
                    mn = _fv("match")
                    if not _safe(mn):
                        self._send(400, "application/json", _j.dumps({"error": "invalid match"})); return
                    kdir = vdir / "keyframes"
                    _, _plan, _fbn, _matches_map, _ = _scan_matches()
                    _info = _matches_map.get(mn, {})
                    # 1. Video files: finals (short/long), clips, outcomes — plus
                    #    each one's enhanced (upscaled / fps) siblings.
                    for _p in list(_info.get("finals", {}).values()):
                        _rm_video(_p)
                    for _p in _info.get("clips", []):
                        _rm_video(_p)
                    for (_f, _o, _p) in _info.get("outcomes", []):
                        _rm_video(_p)
                    # 2. Keyframes: clip keyframes + this match's outcome keyframes.
                    for p in kdir.glob(f"{mn}_clip*.png"):
                        _rm(p)
                    for (_f, _o, _p) in _info.get("outcomes", []):
                        _base = Path(_p).stem
                        _rm(kdir / f"{_base}.png")
                        for _sp in kdir.glob(f"{_base}_s*.png"):
                            _rm(_sp)
                    _meta = _fbn.get(mn, {})
                    _mf = {_meta.get("f1"), _meta.get("f2")} - {None}
                    for o in _plan.get("outcome_plan", []):
                        if o.get("match_name") == mn:
                            for _st in _outcome_kf_stems(o):
                                _rm(kdir / f"{_st}.png")
                    # 3. Strip the match from prompts.json (plan + its per-match outcomes).
                    try:
                        if prompts_file.exists():
                            _data = json.loads(prompts_file.read_text())
                            _before = len(_data.get("fight_plan", []))
                            _data["fight_plan"] = [x for x in _data.get("fight_plan", [])
                                                   if x.get("match_name") != mn]
                            _data["outcome_plan"] = [o for o in _data.get("outcome_plan", [])
                                                     if o.get("match_name") != mn]
                            if len(_data.get("fight_plan", [])) != _before:
                                prompts_file.write_text(json.dumps(_data, indent=2))
                    except Exception as e:
                        self._send(500, "application/json",
                                   _j.dumps({"error": f"removed files but could not update "
                                                      f"prompts.json: {e}"})); return
                else:
                    self._send(400, "application/json", _j.dumps({"error": "invalid scope"})); return
                # Removing any of a match's videos invalidates its prepared/uploaded
                # ZIP — drop the upload state so the badge clears and the ZIP must be
                # regenerated before re-uploading.
                _dmn = _fv("match")
                if _dmn and _safe(_dmn):
                    _clear_match_upload(out_dir, _dmn)
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
                    "upscale_model": _s(_fv("upscale_model")),
                    "upscale_model_2x": _s(_fv("upscale_model_2x")),
                    "upscale_model_4x": _s(_fv("upscale_model_4x")),
                    "interpolation_model": _s(_fv("interpolation_model")),
                    "no_llm": "no_llm" in form,
                    "out_dir": _fv("out_dir", "./township_output"),
                    "region": _s(_fv("region")),
                    "include_female": "include_female" in form,
                    "num_fighters": int(_fv("num_fighters", "0") or 0),
                    "num_environments": int(_fv("num_environments", "0") or 0),
                    "char_refs": int(_fv("char_refs", "4") or 4),
                    "env_refs": int(_fv("env_refs", "3") or 3),
                    "fps": int(_fv("fps", "8") or 8),
                    "playback_fps": int(_fv("playback_fps", "0") or 0),
                    "clip_delay": float(_fv("clip_delay", "5") or 5),
                    "upscale_factor": int(_fv("upscale_factor", "0") or 0),
                    "fps_multiplier": int(_fv("fps_multiplier", "0") or 0),
                    "matches": int(_fv("matches", "6") or 6),
                    "skip_videos": "skip_videos" in form,
                    "only_outcomes": "only_outcomes" in form,
                    "consistency": _fv("consistency", "keyframe"),
                    "keyframe_steps": int(_fv("keyframe_steps", "28") or 28),
                    "keyframe_size": _fv("keyframe_size", "832x480"),
                    "character_strength": float(_fv("character_strength", "0.7") or 0.7),
                    "lora_steps": int(_fv("lora_steps", "800") or 800),
                    "lora_rank": int(_fv("lora_rank", "16") or 16),
                    "lora_weight": float(_fv("lora_weight", "0.85") or 0.85),
                    "lora_train_base_model": _s(_fv("lora_train_base_model")) or "",
                    "no_env_loras": "env_loras" not in form,
                    "env_lora_steps": int(_fv("env_lora_steps", "800") or 800),
                    "env_lora_rank": int(_fv("env_lora_rank", "16") or 16),
                    "env_lora_weight": float(_fv("env_lora_weight", "0.8") or 0.8),
                    "video_lora_scale": float(_fv("video_lora_scale", "1.0") or 1.0),
                    "video_size": _fv("video_size", "832x480") or "832x480",
                    "clip_min_frames": int(_fv("clip_min_frames", "50") or 50),
                    "clip_max_frames": int(_fv("clip_max_frames", "70") or 70),
                    "single_clip_max_frames": int(_fv("single_clip_max_frames", "50") or 50),
                    "outcome_min_frames": int(_fv("outcome_min_frames", "96") or 96),
                    "outcome_max_frames": int(_fv("outcome_max_frames", "150") or 150),
                    "short_min": float(_fv("short_min", "40") or 40),
                    "short_max": float(_fv("short_max", "50") or 50),
                    "long_min": float(_fv("long_min", "65") or 65),
                    "long_max": float(_fv("long_max", "75") or 75),
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
                    "upload_endpoint": _fv("upload_endpoint", "") or "",
                    "upload_token": _fv("upload_token", "") or "",
                    "upload_fixture_id": _fv("upload_fixture_id", "") or "",
                    "upload_after_render": "upload_after_render" in form,
                    "odds_ranges": _collect_odds_ranges(_fv),
                }
                # Apply ALL saved settings to the live session immediately, so
                # subsequent per-profile jobs (regenerate, train LoRA), runs, AND a
                # page reload all reflect what was just saved. The Run page renders
                # from default_args, so updating only the connection keys (the old
                # behaviour) made every other field — e.g. video_lora_scale — snap
                # back to its launch value on reload even though the file was saved.
                for _k, _val in cfg.items():
                    setattr(default_args, _k, _val)
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
                _enqueue_gen(job_id, _run_process_job, job_id, fpath, op, param,
                             jtype="process")
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
            ns.upscale_model = _fv("upscale_model") or None
            ns.upscale_model_2x = _fv("upscale_model_2x") or None
            ns.upscale_model_4x = _fv("upscale_model_4x") or None
            ns.interpolation_model = _fv("interpolation_model") or None
            ns.no_llm       = "no_llm" in form
            ns.out_dir      = _fv("out_dir", "./township_output")
            ns.region       = _fv("region") or None
            ns.include_female = "include_female" in form
            ns.num_fighters    = int(_fv("num_fighters", "0") or 0)
            ns.num_environments = int(_fv("num_environments", "0") or 0)
            ns.char_refs        = int(_fv("char_refs", "4") or 4)
            ns.env_refs         = int(_fv("env_refs", "3") or 3)
            ns.fps          = int(_fv("fps", "8"))
            ns.playback_fps = int(_fv("playback_fps", "0") or 0)
            ns.clip_delay   = float(_fv("clip_delay", "5.0"))
            ns.upscale_factor = int(_fv("upscale_factor", "0") or 0)
            ns.fps_multiplier = int(_fv("fps_multiplier", "0") or 0)
            ns.matches      = int(_fv("matches", "6"))
            ns.skip_videos  = "skip_videos" in form
            ns.only_outcomes = "only_outcomes" in form
            # consistency config
            ns.consistency       = _fv("consistency", "keyframe")
            ns.keyframe_steps    = int(_fv("keyframe_steps", "28"))
            ns.keyframe_size     = _fv("keyframe_size", "832x480")
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
            ns.video_lora_scale  = float(_fv("video_lora_scale", "1.0"))
            ns.video_size        = _fv("video_size", "832x480") or "832x480"
            ns.clip_min_frames   = int(_fv("clip_min_frames", "50"))
            ns.clip_max_frames   = int(_fv("clip_max_frames", "70"))
            ns.single_clip_max_frames = int(_fv("single_clip_max_frames", "50"))
            ns.outcome_min_frames = int(_fv("outcome_min_frames", "96"))
            ns.outcome_max_frames = int(_fv("outcome_max_frames", "150"))
            ns.short_min         = float(_fv("short_min", "40"))
            ns.short_max         = float(_fv("short_max", "50"))
            ns.long_min          = float(_fv("long_min", "65"))
            ns.long_max          = float(_fv("long_max", "75"))
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
            # Township upload settings
            ns.upload_endpoint    = _fv("upload_endpoint", "") or ""
            ns.upload_token       = _fv("upload_token", "") or ""
            ns.upload_fixture_id  = _fv("upload_fixture_id", "") or ""
            ns.upload_after_render = "upload_after_render" in form
            ns.odds_ranges        = _collect_odds_ranges(_fv)

            # Apply the submitted connection/model settings to the live session
            # so later per-profile jobs (regenerate, train LoRA) use them too.
            for _k in ("base_url", "api_key", "image_model",
                       "video_model", "text_model", "upscale_model",
                       "upscale_model_2x", "upscale_model_4x", "interpolation_model",
                       "lora_train_base_model",
                       "upload_endpoint", "upload_token", "upload_fixture_id",
                       "upload_after_render", "odds_ranges"):
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
            # Full-run checkbox: also train Wan video LoRAs after image LoRAs.
            ns.video_loras = ("video_loras" in form)
            ns.only_video_loras = False
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
                elif step == "video-loras":
                    ns.skip_characters = True; ns.skip_environments = True
                    ns.skip_videos = True; ns.only_video_loras = True; ns.video_loras = True
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
                "prompts":      "Step 3 · Generate Matches Prompts",
                "loras":        "Step 4 · Train Character LoRAs",
                "video-loras":  "Step · Train Video (Wan) LoRAs",
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
                                          include_female=args.include_female,
                                          max_count=int(getattr(args, "num_fighters", 0) or 0),
                                          n_refs=int(getattr(args, "char_refs", 4) or 4))

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
            env_names = stage_environments(client, image_model, out_dir_r, region_filter=args.region,
                                           max_count=int(getattr(args, "num_environments", 0) or 0),
                                           n_refs=int(getattr(args, "env_refs", 3) or 3))

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

        # Video (Wan) LoRAs — trained against the configured video model and stored
        # separately (tagged with the model slug); image LoRAs above stay intact.
        _want_video_lora = (getattr(args, "video_loras", False)
                            or getattr(args, "only_video_loras", False))
        if _want_video_lora:
            _vm = video_model or pick_model(client, "video", args.video_model)
            if not _vm:
                _web_log("  ⚠ Video LoRA training needs a video model.")
            else:
                if char_names:
                    _web_log(f"  Training character VIDEO LoRAs ({_model_slug(_vm)}) "
                             f"for {len(char_names)} fighter(s)…")
                    stage_video_loras(client, _vm, out_dir_r, char_names or [],
                                      lora_steps=getattr(args, "lora_steps", 800),
                                      lora_rank=getattr(args, "lora_rank", 16))
                if not _no_env_loras and env_names:
                    _web_log(f"  Training environment VIDEO LoRAs ({_model_slug(_vm)}) "
                             f"for {len(env_names)} location(s)…")
                    stage_env_video_loras(client, _vm, out_dir_r, env_names or [],
                                          lora_steps=getattr(args, "env_lora_steps", 800),
                                          lora_rank=getattr(args, "env_lora_rank", 16))

        if getattr(args, "only_video_loras", False):
            _web_log("\n✓ Video LoRA step complete.")
        elif only_loras:
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
                keyframe_size=getattr(args, "keyframe_size", "832x480"),
                lora_weight=getattr(args, "lora_weight", 0.85),
                env_lora_map=env_lora_map, env_lora_weight=_env_lora_weight,
                video_lora_scale=getattr(args, "video_lora_scale", 1.0),
                video_size=getattr(args, "video_size", "832x480"),
                clip_min_frames=getattr(args, "clip_min_frames", CLIP_MIN_FRAMES),
                clip_max_frames=getattr(args, "clip_max_frames", CLIP_MAX_FRAMES),
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
                keyframe_size=getattr(args, "keyframe_size", "832x480"),
                lora_weight=getattr(args, "lora_weight", 0.85),
                clip_min_frames=getattr(args, "clip_min_frames", CLIP_MIN_FRAMES),
                clip_max_frames=getattr(args, "clip_max_frames", CLIP_MAX_FRAMES),
                env_lora_map=env_lora_map, env_lora_weight=_env_lora_weight,
                video_lora_scale=getattr(args, "video_lora_scale", 1.0),
                video_size=getattr(args, "video_size", "832x480"),
                short_min=getattr(args, "short_min", 40.0),
                short_max=getattr(args, "short_max", 50.0),
                long_min=getattr(args, "long_min", 65.0),
                long_max=getattr(args, "long_max", 75.0),
                single_clip_max_frames=getattr(args, "single_clip_max_frames", SINGLE_CLIP_MAX_FRAMES),
                outcome_min_frames=getattr(args, "outcome_min_frames", 96),
                outcome_max_frames=getattr(args, "outcome_max_frames", 150),
                playback_fps=getattr(args, "playback_fps", 0),
                upscale_factor=getattr(args, "upscale_factor", 0),
                fps_multiplier=getattr(args, "fps_multiplier", 0),
                upscale_model=_upscale_model_for(args, getattr(args, "upscale_factor", 0)) or None,
                interpolation_model=getattr(args, "interpolation_model", None),
            )

        # Optional: generate odds, pack ZIPs and upload every rendered match.
        if (getattr(args, "upload_after_render", False) and not args.skip_videos
                and not args.only_prompts and not getattr(args, "only_loras", False)
                and not getattr(args, "only_keyframes", False)):
            try:
                _auto_upload_all_matches(out_dir_r, args, log=_web_log)
            except Exception as e:
                _web_log(f"  ✗ auto-upload error: {e}")

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
    b) Outcome videos — win / ko_win / retire per fighter, plus ONE draw per
       match. Each is a two-shot sequence: the decisive FINISH (KO, retirement,
       last action) then the VICTORY (winner + referee raising the arm), assembled
       into one clip. Length via --outcome-min/max-frames.
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
    parser.add_argument("--upscale-model", default=None, metavar="MODEL_ID",
                        help="Generic AI super-resolution model (e.g. Real-ESRGAN) for "
                             "--upscale-factor. Fallback when no factor-specific model is set. "
                             "Optional: if omitted, CoderAI auto-selects a configured upscaler. "
                             "There is no ffmpeg fallback.")
    parser.add_argument("--upscale-model-2x", default=None, metavar="MODEL_ID",
                        help="Upscale model used specifically when the 2× factor is chosen "
                             "(overrides --upscale-model for 2×).")
    parser.add_argument("--upscale-model-4x", default=None, metavar="MODEL_ID",
                        help="Upscale model used specifically when the 4× factor is chosen "
                             "(overrides --upscale-model for 4×).")
    parser.add_argument("--interpolation-model", default=None, metavar="MODEL_ID",
                        help="AI frame-interpolation model (RIFE/FILM) for FPS raising "
                             "(--fps-multiplier). Optional: if omitted, CoderAI auto-selects a "
                             "configured interpolation model. There is no ffmpeg fallback.")
    parser.add_argument("--text-model",  default=None, metavar="MODEL_ID",
                        help="LLM for prompt generation (recommended for variety). Auto-selected if omitted.")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM prompt generation even if a text model is available.")
    parser.add_argument("--out-dir",   default="./township_output", metavar="DIR",
                        help="Output directory (default: ./township_output)")
    parser.add_argument("--fps",       type=int, default=8, metavar="N",
                        help="Generation/base FPS (default: 8).")
    parser.add_argument("--playback-fps", type=int, default=0, metavar="N",
                        help="Playback FPS for the encoded clips (0 = same as --fps). Wan makes a "
                             "fixed number of frames regardless of fps, so a HIGHER playback fps "
                             "plays them faster (less slow-motion). The clip count is planned on "
                             "this rate so the finals reach their target length at real speed.")
    parser.add_argument("--clip-delay", type=float, default=5.0, metavar="SECONDS",
                        help="Seconds between video clip requests (default: 5). Raise if rate-limited.")
    parser.add_argument("--upscale-factor", type=int, default=0, choices=[0, 2, 4], metavar="N",
                        help="Post-process the final + outcome videos with NxN super-resolution "
                             "(2 or 4; 0=off, default). Writes new *_2x/_4x files alongside.")
    parser.add_argument("--fps-multiplier", type=int, default=0, metavar="N",
                        help="Post-process the final + outcome videos by raising FPS Nx via frame "
                             "interpolation (e.g. 2; 0/1=off, default). Writes new *_NxfpS files.")
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
    parser.add_argument("--num-fighters", type=int, default=0, metavar="N",
                        help="How many fighters to generate from the pool (0 = whole pool).")
    parser.add_argument("--char-refs", type=int, default=4, metavar="N",
                        help="Reference images generated per character (default: 4). More = a "
                             "healthier LoRA at higher training steps; see the steps/weight guide.")

    # Environment control
    env_grp = parser.add_mutually_exclusive_group()
    env_grp.add_argument("--skip-environments", action="store_true",
                         help="Skip Stage 2 — do not generate any environment profiles.")
    env_grp.add_argument("--reuse-environments", action="store_true",
                         help="Skip Stage 2 and reuse ALL existing environment profiles already in CoderAI.")
    env_grp.add_argument("--environments", default=None, metavar="NAME,NAME,...",
                         help="Comma-separated environment profile names to use (skip generation).")
    parser.add_argument("--num-environments", type=int, default=0, metavar="N",
                        help="How many environments to generate from the pool (0 = whole pool).")
    parser.add_argument("--env-refs", type=int, default=3, metavar="N",
                        help="Reference images generated per environment (default: 3).")

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
    cons_grp.add_argument("--keyframe-size", default="832x480", metavar="WxH",
                          help="Keyframe image resolution (default: 832x480, 16:9 to match Wan).")
    cons_grp.add_argument("--video-size", default="832x480", metavar="WxH",
                          help="Video clip resolution (default: 832x480 — Wan2.2 native 16:9; "
                               "also 1280x720 for 720p).")
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
    cons_grp.add_argument("--clip-min-frames", type=int, default=CLIP_MIN_FRAMES, metavar="N",
                          help=f"Minimum frames per fight clip (default: {CLIP_MIN_FRAMES}). Clip "
                               "duration = frames / fps; kept within the model's safe length "
                               f"(≤{MODEL_MAX_FRAMES}).")
    cons_grp.add_argument("--clip-max-frames", type=int, default=CLIP_MAX_FRAMES, metavar="N",
                          help=f"Maximum frames per fight clip (default: {CLIP_MAX_FRAMES}). A clip "
                               f"longer than --single-clip-max-frames is split into chained, "
                               f"concatenated sub-renders (one continuous shot).")
    cons_grp.add_argument("--single-clip-max-frames", type=int, default=SINGLE_CLIP_MAX_FRAMES,
                          metavar="N",
                          help=f"Max frames in ONE model generation (default: {SINGLE_CLIP_MAX_FRAMES}, "
                               f"≤{MODEL_MAX_FRAMES}). Clips/outcomes longer than this are rendered as "
                               f"multiple parts chained via each part's last frame and concatenated "
                               f"into a single shot; the parts are discarded.")
    cons_grp.add_argument("--outcome-min-frames", type=int, default=96, metavar="N",
                          help="Minimum TOTAL frames per outcome video (default: 96). An outcome is "
                               "a two-clip sequence (finish → victory); this budget is split across "
                               "them.")
    cons_grp.add_argument("--outcome-max-frames", type=int, default=150, metavar="N",
                          help="Maximum TOTAL frames per outcome video (default: 150). Split across the "
                               "finish + victory clips, each chained when longer than "
                               "--single-clip-max-frames.")
    cons_grp.add_argument("--short-min", type=float, default=40.0, metavar="SEC",
                          help="Minimum duration (s) of the SHORT final assembly (default: 40).")
    cons_grp.add_argument("--short-max", type=float, default=50.0, metavar="SEC",
                          help="Maximum duration (s) of the SHORT final assembly (default: 50).")
    cons_grp.add_argument("--long-min", type=float, default=65.0, metavar="SEC",
                          help="Minimum duration (s) of the LONG final assembly (default: 65). "
                               "The clip count per match is derived from this target and the fps "
                               "so the long cut is always filled.")
    cons_grp.add_argument("--long-max", type=float, default=75.0, metavar="SEC",
                          help="Maximum duration (s) of the LONG final assembly (default: 75).")
    cons_grp.add_argument("--video-lora-scale", type=float, default=1.0, metavar="F",
                          help="Multiplier applied to the character + environment LoRA weights "
                               "at VIDEO render time only (keyframe LoRA weight is unaffected; "
                               "default: 1.0). Lower it (e.g. 0.5-0.7) when stacked LoRAs on a "
                               "distilled Wan2.2 expert desaturate or over-smooth the clip — it "
                               "trades identity strength for cleaner colour and motion.")
    cons_grp.add_argument("--video-loras", action="store_true",
                          help="Also train Wan VIDEO LoRAs (per fighter + environment) against the "
                               "configured --video-model. Stored separately (tagged with the model) "
                               "and applied to the video request. Heavy on large video models.")
    cons_grp.add_argument("--only-video-loras", action="store_true",
                          help="Train ONLY the Wan video LoRAs (skip everything else), against "
                               "--video-model. Implies --video-loras.")

    parser.add_argument("--cli-mode", action="store_true",
                        help="Run in CLI mode (default when --cli-mode is present). "
                             "Without this flag the script launches a web UI instead of processing.")
    parser.add_argument("--web-port", type=int, default=7788, metavar="PORT",
                        help="Port for the web UI (default: 7788, only used without --cli-mode).")
    parser.add_argument("--browser", action="store_true",
                        help="Auto-open a web browser at the UI URL on startup. Off by default "
                             "(avoids spawning a terminal text browser on headless servers).")

    # ── Township Combat League upload ─────────────────────────────────────────
    parser.add_argument("--upload-endpoint", default="", metavar="URL",
                        help="Base URL of the Township Combat League server to upload matches to "
                             "(e.g. https://townshipcombatleague.com).")
    parser.add_argument("--upload-token", default="", metavar="TOKEN",
                        help="Fixture-source API token for the upload endpoint.")
    parser.add_argument("--upload-fixture-id", default="", metavar="ID",
                        help="ID of an EXISTING fixture on the server to add uploaded matches to.")
    parser.add_argument("--upload-after-render", action="store_true",
                        help="After a full match render, also generate odds, pack the ZIP and "
                             "upload it to the Township server.")
    parser.add_argument("--odds-ranges", default=None,
                        help=argparse.SUPPRESS)  # JSON map col->[min,max]; usually set via the web UI.

    # Two-phase parse: pre-scan for -c/--config so the saved values become
    # parser defaults that explicit command-line arguments can still override.
    pre, _ = parser.parse_known_args()
    if pre.config:
        try:
            cfg = load_config(pre.config)
        except FileNotFoundError:
            # A missing config is not an error: the bundled launcher always passes
            # --config <out-dir>/township_config.json so a mapped config auto-loads,
            # but on a fresh install (no mapped dir) the file simply isn't there yet
            # — fall back to defaults instead of failing to start.
            cfg = {}
            _log(f"  Config {pre.config} not found — starting with defaults "
                 f"(it'll be created when you Save from the web UI)")
        except Exception as e:
            parser.error(f"cannot load config {pre.config}: {e}")
        if cfg:
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
                                      include_female=args.include_female,
                                      max_count=int(getattr(args, "num_fighters", 0) or 0),
                                      n_refs=int(getattr(args, "char_refs", 4) or 4))

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
        env_names = stage_environments(client, image_model, out_dir, region_filter=args.region,
                                       max_count=int(getattr(args, "num_environments", 0) or 0),
                                       n_refs=int(getattr(args, "env_refs", 3) or 3))

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

    # ── Stage 2.6: Video (Wan) LoRA training — against the video model ─────────
    if getattr(args, "video_loras", False) or getattr(args, "only_video_loras", False):
        _vm = video_model or pick_model(client, "video", args.video_model)
        if _vm and (char_names or []):
            stage_video_loras(client, _vm, out_dir, char_names or [],
                              lora_steps=getattr(args, "lora_steps", 800),
                              lora_rank=getattr(args, "lora_rank", 16))
        if (_vm and not getattr(args, "no_env_loras", False) and (env_names or [])):
            stage_env_video_loras(client, _vm, out_dir, env_names or [],
                                  lora_steps=getattr(args, "env_lora_steps", 800),
                                  lora_rank=getattr(args, "env_lora_rank", 16))
        if getattr(args, "only_video_loras", False):
            _log("\n✓ Video LoRA training complete.")
            return

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
            keyframe_size=getattr(args, "keyframe_size", "832x480"),
            clip_min_frames=getattr(args, "clip_min_frames", CLIP_MIN_FRAMES),
            clip_max_frames=getattr(args, "clip_max_frames", CLIP_MAX_FRAMES),
            lora_weight=getattr(args, "lora_weight", 0.85),
            env_lora_map=env_lora_map,
            env_lora_weight=getattr(args, "env_lora_weight", 0.8),
            video_lora_scale=getattr(args, "video_lora_scale", 1.0),
            video_size=getattr(args, "video_size", "832x480"),
            short_min=getattr(args, "short_min", 40.0),
            short_max=getattr(args, "short_max", 50.0),
            long_min=getattr(args, "long_min", 65.0),
            long_max=getattr(args, "long_max", 75.0),
            single_clip_max_frames=getattr(args, "single_clip_max_frames", SINGLE_CLIP_MAX_FRAMES),
            outcome_min_frames=getattr(args, "outcome_min_frames", 96),
            outcome_max_frames=getattr(args, "outcome_max_frames", 150),
            playback_fps=getattr(args, "playback_fps", 0),
            upscale_factor=getattr(args, "upscale_factor", 0),
            fps_multiplier=getattr(args, "fps_multiplier", 0),
            upscale_model=getattr(args, "upscale_model", None),
            interpolation_model=getattr(args, "interpolation_model", None),
        )

        if (getattr(args, "upload_after_render", False) and not args.only_prompts
                and not getattr(args, "only_loras", False)
                and not getattr(args, "only_keyframes", False)):
            try:
                _auto_upload_all_matches(out_dir, args, log=_log)
            except Exception as e:
                _log(f"  ✗ auto-upload error: {e}")

    _log("\n✓ Done.")


if __name__ == "__main__":
    main()
