#!/usr/bin/env python3
# CoderAI - kimi-k3-in-c serve-loop patch
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net> — GPLv3 (see the main LICENSE).
"""Idempotently patch FareedKhan-dev/kimi-k3-in-c (``src/cli/k3_run.c``) to add a
resident serve loop, so coderai can drive it like the other native engines instead of
paying k3's one-shot cold start (a ~108 GB dense-trunk read) on every request.

Upstream k3 is a batch CLI: it loads the model, runs ONE prompt, and exits. This patch
adds:

  * ``k3_serve_loop()`` — keeps the model / packed trunk / expert cache / tokenizer
    resident and answers many prompts over the SAME line-oriented mux stdin/stdout
    protocol colibri uses, so coderai's existing ``MuxEngine`` client drives it
    unchanged:
        engine → server : ``\\x01\\x01READY\\x01\\x01`` + ``STAT …``
        server → engine : ``SUBMIT <id> <slot> <bytes> <max> <temp> <top_p>\\n<payload>\\n``
        engine → server : ``DATA <id> <n>\\n<n bytes>\\n`` … ``DONE <id> STAT …``
    The payload is a fully rendered prompt (the coderai backend owns the Kimi-K3 chat
    template); the engine tokenizes it with the loaded tokenizer. Greedy, single-slot,
    incremental decode; KV / recurrent state is reset between turns (stateless HTTP).

  * a ``SERVE=1`` branch in ``main()`` that, after the model/trunk/cache are loaded,
    loads the tokenizer (``--tok`` dir, else the model dir) and enters the serve loop —
    reusing all of the one-shot setup rather than duplicating it.

Serve context capacity comes from ``K3_MAXT`` (else ``CTX``, else 4096).

Safe to run repeatedly (each edit is guarded / self-idempotent).
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "src/cli/k3_run.c"
src = open(path, encoding="utf-8", errors="surrogateescape").read()
orig = src
applied = []

# --------------------------------------------------------------------------- #
# 1) the serve loop + a SERVE detector, inserted just before main()
# --------------------------------------------------------------------------- #
_SERVE_FN = r'''
/* ================= coderai serve loop (added by packaging/patch-k3.py) =========
 * Keep the model, packed trunk, expert cache and tokenizer RESIDENT and answer many
 * prompts over the colibri-compatible mux stdin/stdout protocol, so coderai's MuxEngine
 * can drive k3 without paying the ~108 GB cold-start trunk read per request. One prompt
 * at a time (single KV slot), greedy incremental decode, state reset between turns. */
static int k3_serve_loop(Weights *w, const K3Cfg *c, K3Cache *cache, Tok *tok, int maxt)
{
    const int NL = c->n_layers;
    const int E  = c->hidden;
    const int maxb = c->n_layers / c->attn_res_block + 2;
    const int P  = c->kda_heads * c->kda_head_dim;
    const size_t kper = (size_t)P * c->kda_head_dim + (size_t)3 * P * (c->conv_k - 1);
    size_t sc_need = k3_layer_scratch(c, maxt);
    { size_t ic = k3_mla_scratch_cached(c, maxt, maxt, 1); if (ic > sc_need) sc_need = ic; }

    float *h   = (float *)malloc((size_t)maxt * E * sizeof(float));
    float *br  = (float *)malloc((size_t)maxt * maxb * E * sizeof(float));
    float *ks  = (float *)malloc(kper * (size_t)NL * sizeof(float));
    float *sc  = (float *)malloc(sc_need * sizeof(float));
    float *lg  = (float *)malloc((size_t)c->vocab * sizeof(float));
    int   *seq = (int   *)malloc((size_t)maxt * sizeof(int));
    size_t payload_cap = (size_t)1 << 20;
    char  *payload = (char *)malloc(payload_cap);
    if (!h || !br || !ks || !sc || !lg || !seq || !payload) {
        fprintf(stderr, "k3 serve: OOM allocating serve buffers\n"); return 1; }

    w->mla_slot = (int *)malloc((size_t)NL * sizeof(int));
    if (!w->mla_slot) { fprintf(stderr, "k3 serve: OOM mla_slot\n"); return 1; }
    w->n_mla = 0;
    for (int L = 0; L < NL; L++) w->mla_slot[L] = k3_is_mla(c, L) ? w->n_mla++ : -1;
    w->kv_cap = maxt;
    const size_t kvper = (size_t)maxt * c->n_heads * (c->qk_nope + c->v_head);
    const size_t rpper = (size_t)maxt * c->qk_rope;
    w->kvc   = (float *)calloc(kvper * (size_t)w->n_mla, sizeof(float));
    w->ropec = (float *)calloc(rpper * (size_t)w->n_mla, sizeof(float));
    if (!w->kvc || !w->ropec) { fprintf(stderr, "k3 serve: OOM KV cache\n"); return 1; }

    const int eos = tok_id_of(tok, "<|end_of_msg|>");

    setvbuf(stdin,  NULL, _IONBF, 0);
    setvbuf(stdout, NULL, _IONBF, 0);
    fputs("\x01\x01READY\x01\x01\n", stdout);
    printf("STAT 0 0.0 0.0 %.2f 0 0\n", peak_rss_bytes() / 1e9);
    fflush(stdout);

    char line[512], dbuf[1024];
    while (fgets(line, sizeof line, stdin)) {
        char cmd[16] = {0}, id[64] = {0};
        int slot = 0, plen = 0, maxtok = 0; float temp = 0.f, top_p = 1.f;
        int nf = sscanf(line, "%15s %63s %d %d %d %f %f",
                        cmd, id, &slot, &plen, &maxtok, &temp, &top_p);
        (void)slot; (void)temp; (void)top_p;
        if (nf < 2) continue;
        if (strcmp(cmd, "SUBMIT")) continue;   /* CANCEL/STOP between turns: nothing to do */
        if (nf < 5 || plen < 0) { printf("ERROR %s BAD_FRAME\n", id); fflush(stdout); continue; }
        if ((size_t)plen + 1 > payload_cap) {
            char *n2 = (char *)realloc(payload, (size_t)plen + 1);
            if (!n2) { printf("ERROR %s OOM\n", id); fflush(stdout); return 1; }
            payload = n2; payload_cap = (size_t)plen + 1;
        }
        if (plen > 0 && fread(payload, 1, (size_t)plen, stdin) != (size_t)plen) break;
        payload[plen] = 0;
        (void)fgetc(stdin);   /* trailing newline after the payload */

        /* fresh conversation each turn (stateless HTTP): drop carried recurrent/KV */
        memset(ks, 0, kper * (size_t)NL * sizeof(float));
        if (w->n_mla) {
            memset(w->kvc,   0, kvper * (size_t)w->n_mla * sizeof(float));
            memset(w->ropec, 0, rpper * (size_t)w->n_mla * sizeof(float));
        }
        w->cached = 0;

        int np = tok_encode(tok, payload, plen, seq, maxt - 1);
        if (np <= 0) { printf("ERROR %s EMPTY_PROMPT\n", id); fflush(stdout); continue; }
        int cap_tok = maxtok > 0 ? maxtok : 8;
        if (np + cap_tok > maxt) cap_tok = maxt - np;
        if (cap_tok < 0) cap_tok = 0;

        const double t0 = now_s();
        int emitted = 0, limited = 1;
        if (forward(w, c, cache, seq, np, lg, sc, h, br, ks, NULL) != 0) {
            printf("DONE %s STAT 0 0.0 0.0 %.2f %d 0\n", id, peak_rss_bytes() / 1e9, np);
            fflush(stdout); continue;
        }
        w->cached = np;
        int t = argmax_(lg, c->vocab);
        int T = np;
        for (int g = 0; g < cap_tok; g++) {
            if (t == eos) { limited = 0; break; }
            int n = tok_decode(tok, &t, 1, dbuf, (int)sizeof dbuf);
            if (n > 0) {
                printf("DATA %s %d\n", id, n);
                fwrite(dbuf, 1, (size_t)n, stdout);
                fputc('\n', stdout);
                fflush(stdout);
            }
            emitted++;
            if (T >= maxt) break;
            seq[T++] = t;
            if (forward(w, c, cache, seq + w->cached, 1, lg, sc, h, br, ks, NULL) != 0) break;
            w->cached += 1;
            t = argmax_(lg, c->vocab);
        }
        double dt = now_s() - t0;
        double tps = (emitted > 0 && dt > 0.0) ? emitted / dt : 0.0;
        printf("DONE %s STAT %d %.3f 0.0 %.2f %d %d\n",
               id, emitted, tps, peak_rss_bytes() / 1e9, np, limited);
        fflush(stdout);
    }

    free(h); free(br); free(ks); free(sc); free(lg); free(seq); free(payload);
    return 0;
}

'''

_MAIN_ANCHOR = "int main(int argc, char **argv)\n{"
if "k3_serve_loop" not in src:
    if _MAIN_ANCHOR not in src:
        print("[patch-k3] ERROR: main() anchor not found — upstream changed; patch NOT applied",
              file=sys.stderr)
        sys.exit(2)
    src = src.replace(_MAIN_ANCHOR, _SERVE_FN + _MAIN_ANCHOR, 1)
    applied.append("k3_serve_loop()")

# --------------------------------------------------------------------------- #
# 2) SERVE detector near the top of main()
# --------------------------------------------------------------------------- #
_ARGC_ANCHOR = "    if (argc < 2) { usage(stderr); return 2; }\n"
if "int serving =" not in src:
    if _ARGC_ANCHOR not in src:
        print("[patch-k3] ERROR: argc-check anchor not found — upstream changed; patch NOT applied",
              file=sys.stderr)
        sys.exit(2)
    src = src.replace(
        _ARGC_ANCHOR,
        _ARGC_ANCHOR +
        "\n    /* coderai: resident serve mode (SERVE=1) over the mux stdin/stdout "
        "protocol. */\n"
        "    int serving = getenv(\"SERVE\") && getenv(\"SERVE\")[0] == '1';\n",
        1)
    applied.append("SERVE detector")

# --------------------------------------------------------------------------- #
# 3) let serve mode start without a --ids/--prompt source
# --------------------------------------------------------------------------- #
_NSRC_ANCHOR = "        if (nsrc == 0) {"
if _NSRC_ANCHOR in src:
    src = src.replace(_NSRC_ANCHOR, "        if (nsrc == 0 && !serving) {", 1)
    applied.append("skip prompt-required when serving")

# --------------------------------------------------------------------------- #
# 4) enter the serve loop after the expert cache is initialised
# --------------------------------------------------------------------------- #
_CACHE_ANCHOR = ("    K3Cache cache;\n"
                 "    if (k3_cache_init(&cache, &st, &c, (int64_t)(cache_gb * 1e9)) != 0) return 1;\n")
if "return k3_serve_loop(" not in src:
    if _CACHE_ANCHOR not in src:
        print("[patch-k3] ERROR: k3_cache_init anchor not found — upstream changed; "
              "patch NOT applied", file=sys.stderr)
        sys.exit(2)
    src = src.replace(
        _CACHE_ANCHOR,
        _CACHE_ANCHOR +
        "\n    if (serving) {\n"
        "        const char *k3_td = tok_dir ? tok_dir : dir;\n"
        "        if (!have_tok) {\n"
        "            /* Refuse cleanly if there is no tokenizer to load — k3_tok_load has\n"
        "             * no error return and would otherwise crash on a bad/empty dir. */\n"
        "            char k3_tp[4096]; int k3_tok_ok = 0;\n"
        "            const char *k3_tc[] = {\"tokenizer.json\", \"tiktoken.model\", \"tokenizer.model\"};\n"
        "            for (int k3i = 0; k3i < 3; k3i++) {\n"
        "                snprintf(k3_tp, sizeof k3_tp, \"%s/%s\", k3_td, k3_tc[k3i]);\n"
        "                FILE *k3_tf = fopen(k3_tp, \"rb\");\n"
        "                if (k3_tf) { fclose(k3_tf); k3_tok_ok = 1; break; }\n"
        "            }\n"
        "            if (!k3_tok_ok) {\n"
        "                fprintf(stderr, \"k3 serve: no tokenizer under '%s' (need \"\n"
        "                        \"tokenizer.json / tiktoken.model); set --tok\\n\", k3_td);\n"
        "                return 2;\n"
        "            }\n"
        "            k3_tok_load(&tok, k3_td); have_tok = 1;\n"
        "        }\n"
        "        int serve_maxt = getenv(\"K3_MAXT\") ? atoi(getenv(\"K3_MAXT\"))\n"
        "                        : getenv(\"CTX\") ? atoi(getenv(\"CTX\")) : 4096;\n"
        "        if (serve_maxt < 32) serve_maxt = 4096;\n"
        "        return k3_serve_loop(&w, &c, &cache, &tok, serve_maxt);\n"
        "    }\n",
        1)
    applied.append("serve branch in main()")

if src != orig:
    open(path, "w", encoding="utf-8", errors="surrogateescape").write(src)
    print("[patch-k3] applied to %s: %s" % (path, ", ".join(applied)))
else:
    print("[patch-k3] already patched (no change): %s" % path)
