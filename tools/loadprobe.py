#!/usr/bin/env python3
"""Watch whether a coderai install is ANSWERING while it is under load.

Throughput alone cannot tell "the upstreams are slow" from "the server is not
running". Those need opposite responses -- rent more capacity, or find what is
blocking the loop -- and they look identical in a completion count. So this
samples the one thing that separates them: how long the front takes to answer a
request that does no work.

`/healthz` touches no model state and returns the instant the loop is free, so
its latency IS event-loop lag, observed from where a client stands. A run where
p99 healthz stays at a few milliseconds while documents crawl is a capacity
problem. A run where healthz goes to seconds is not: nothing was being served
at all, and no amount of extra capacity would have helped.

    tools/loadprobe.py --base http://127.0.0.1:18776 --out run.tsv
    tools/loadprobe.py --base ... --token $TOK --interval 1 --duration 3600

Each row carries the latency distribution for the window just ended rather than
one instantaneous reading, because a stall that is over before the next sample
leaves no trace in a point measurement. If `--token` is given it also records
the server's own loop-lag counters from /metrics, which cover the windows
between samples, and the in-flight page count the OCR subsystem reports.

Writes TSV to --out (and stdout), append-safe, one header. Ctrl-C to stop.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def _get(url, token="", timeout=20):
    """(seconds, status, body-or-None). Never raises."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "coderai-loadprobe/1.0",
                      **({"Authorization": f"Bearer {token}"} if token else {})})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return time.monotonic() - t0, r.status, body
    except urllib.error.HTTPError as e:
        return time.monotonic() - t0, e.code, None
    except Exception:
        # A timeout is the measurement, not an error: record the ceiling.
        return time.monotonic() - t0, 0, None


def _pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[i]


def _loop_metrics(base, token):
    """(worst_lag, stalls, blocked_seconds) from /metrics, or (None, ...)."""
    _, status, body = _get(base.rstrip("/") + "/metrics", token, timeout=20)
    if status != 200 or not body:
        return None, None, None
    worst = stalls = blocked = None
    for line in body.decode("utf-8", "replace").splitlines():
        if line.startswith("coderai_loop_lag_worst_seconds "):
            worst = float(line.split()[-1])
        elif line.startswith("coderai_loop_stalls_total "):
            stalls = int(float(line.split()[-1]))
        elif line.startswith("coderai_loop_blocked_seconds_total "):
            blocked = float(line.split()[-1])
    return worst, stalls, blocked


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True,
                    help="front base URL, e.g. http://127.0.0.1:18776")
    ap.add_argument("--out", default="", help="TSV file to append to")
    ap.add_argument("--token", default="", help="API token, for /metrics")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="seconds between probes (default 1)")
    ap.add_argument("--window", type=float, default=30.0,
                    help="seconds per reported row (default 30)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after this many seconds (0 = until Ctrl-C)")
    ap.add_argument("--timeout", type=float, default=20.0,
                    help="probe timeout; a timeout is recorded, not retried")
    a = ap.parse_args(argv)

    base = a.base.rstrip("/")
    cols = ("ts", "n", "ok", "fail", "p50_ms", "p90_ms", "p99_ms", "max_ms",
            "loop_worst_s", "stalls", "blocked_s")
    fh = open(a.out, "a") if a.out else None
    if fh is not None and fh.tell() == 0:
        fh.write("\t".join(cols) + "\n")
        fh.flush()
    print("\t".join(cols), flush=True)

    started = time.monotonic()
    lat, ok, fail = [], 0, 0
    window_end = time.monotonic() + a.window
    try:
        while True:
            secs, status, _ = _get(base + "/healthz", timeout=a.timeout)
            lat.append(secs * 1000.0)
            if 200 <= status < 500:
                ok += 1
            else:
                fail += 1                 # 0 = no answer within --timeout
            now = time.monotonic()
            if now >= window_end:
                worst, stalls, blocked = (_loop_metrics(base, a.token)
                                          if a.token else (None, None, None))
                row = (time.strftime("%H:%M:%S"), str(len(lat)), str(ok),
                       str(fail),
                       f"{_pct(lat, 50):.1f}", f"{_pct(lat, 90):.1f}",
                       f"{_pct(lat, 99):.1f}", f"{max(lat):.1f}",
                       "-" if worst is None else f"{worst:.2f}",
                       "-" if stalls is None else str(stalls),
                       "-" if blocked is None else f"{blocked:.1f}")
                line = "\t".join(row)
                print(line, flush=True)
                if fh is not None:
                    fh.write(line + "\n")
                    fh.flush()
                lat, ok, fail = [], 0, 0
                window_end = now + a.window
            if a.duration and (now - started) >= a.duration:
                break
            # The probe's own cost comes out of the interval, so a slow answer
            # does not also stretch the sampling period.
            time.sleep(max(0.0, a.interval - (time.monotonic() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        if fh is not None:
            fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
