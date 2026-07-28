#!/usr/bin/env python3
# CoderAI - colibri serve-mux PAUSE/RESUME patch
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net> — GPLv3 (see the main LICENSE).
"""Idempotently patch colibri's serve-mux loop (``c/colibri.c``) to add PAUSE/RESUME
control frames on stdin.

Rationale: colibri decodes autonomously once a request is in flight, so the only ways
coderai could throttle it for thermal protection were "let it finish" (CPU/GPU stays
hot) or SIGSTOP (freezes the process, and via killpg the parent engine too). A PAUSE
frame lets colibri idle the decode loop *between tokens* — keeping all KV/slot state —
and RESUME continues exactly where it left off, with the process staying alive and
responsive. This is what coderai's cooperative thermal pause drives instead of SIGSTOP.

Wire protocol additions (line-oriented, alongside SUBMIT/CANCEL):
    PAUSE\n   -> engine stops issuing forward passes, replies  PAUSED\n
    RESUME\n  -> engine resumes decoding,                      RESUMED\n

Safe to run repeatedly (each edit is guarded / self-idempotent).
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "c/colibri.c"
src = open(path, encoding="utf-8", errors="surrogateescape").read()
orig = src
applied = []

# 1) global pause flag, next to the existing soft-interrupt flag
if "g_paused" not in src:
    src = src.replace(
        "static volatile sig_atomic_t g_intr=0;",
        "static volatile sig_atomic_t g_intr=0;\n"
        "static volatile sig_atomic_t g_paused=0;   /* PAUSE/RESUME mux control (coderai thermal throttle) */",
        1)
    applied.append("g_paused flag")

# 2) handle PAUSE / RESUME command lines in mux_submit (line is already NUL-terminated,
#    trailing newline stripped) — right after the CANCEL handler's NOT_FOUND return.
_anchor = ('        printf("ERROR %llu NOT_FOUND\\n",id); fflush(stdout); free(line); return 0;\n'
           '    }\n')
if 'strcmp(line,"PAUSE")' not in src:
    if _anchor not in src:
        print("[patch-colibri] ERROR: CANCEL/NOT_FOUND anchor not found — upstream changed; "
              "patch NOT applied", file=sys.stderr)
        sys.exit(2)
    src = src.replace(
        _anchor,
        _anchor +
        '    if(!strcmp(line,"PAUSE")){  g_paused=1; printf("PAUSED\\n");  fflush(stdout); free(line); return 0; }\n'
        '    if(!strcmp(line,"RESUME")){ g_paused=0; printf("RESUMED\\n"); fflush(stdout); free(line); return 0; }\n',
        1)
    applied.append("PAUSE/RESUME handler")

# 3) while paused, block in select() awaiting the next command instead of busy-polling
_old_ptv = "struct timeval tv={0,0}, *ptv=active?&tv:NULL;"
if _old_ptv in src:
    src = src.replace(_old_ptv,
                      "struct timeval tv={0,0}, *ptv=(active && !g_paused)?&tv:NULL;", 1)
    applied.append("select-block-when-paused")

# 4) skip the forward pass while paused (state preserved; loop re-enters and blocks)
_old_dec = ("        active=0; for(int i=0;i<nctx;i++) active+=req[i].active;\n"
            "        if(!active){ if(eof) break; continue; }")
if _old_dec in src:
    src = src.replace(
        _old_dec,
        "        active=0; for(int i=0;i<nctx;i++) active+=req[i].active;\n"
        "        if(g_paused) continue;   /* paused: no forward pass; loop blocks in select() awaiting RESUME */\n"
        "        if(!active){ if(eof) break; continue; }", 1)
    applied.append("skip-decode-when-paused")

if src != orig:
    open(path, "w", encoding="utf-8", errors="surrogateescape").write(src)
    print("[patch-colibri] applied to %s: %s" % (path, ", ".join(applied)))
else:
    print("[patch-colibri] already patched (no change): %s" % path)
