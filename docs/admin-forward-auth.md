# Admin GUI: trusted forward-auth (reuse an upstream login)

**Status: IMPLEMENTED in 0.2.85.** (Originally a feature request; the
requirements below are what shipped, with the differences noted at the end.) This is what CoderAI needs to add so
a reverse proxy that has *already* authenticated a user can hand that identity to
CoderAI's admin GUI, instead of the user logging into CoderAI a second time.

## Why

In the Digesta deployment CoderAI runs as a GPU-less RunPod orchestrator behind
nginx on `digesta.tech`, exposed at `/coderai/`. The operators are already
Digesta administrators with a Digesta login (session cookie on the same host,
membership in the `admins` group). We want **those same people, and only those
people, to reach the CoderAI admin GUI with the login they already have** — no
separate CoderAI password to provision, rotate and share, and no second login
screen.

Today that is not possible. CoderAI's admin GUI (`codai/admin/auth.py`) has only
its own login form and HMAC-signed session cookies, with users in `auth.json`.
There is no hook to trust an identity asserted by the proxy: no `X-Forwarded-User`
/ `REMOTE_USER` handling, no OIDC, no "disable the form" switch. So the best we
can do at the proxy is gate `/coderai/` with nginx `auth_request` against a small
Digesta endpoint (only Digesta admins get through) — but the user still meets
CoderAI's own login behind that gate. This request removes that second login.

We considered, and rejected, doing it entirely on our side (Digesta minting a
valid CoderAI session cookie by reading CoderAI's `secret_key` and reproducing
the cookie format). It works but couples the two products through a private
on-disk secret and an undocumented cookie layout, and breaks on any change to
either. A first-class CoderAI feature is the right place for it.

## What we need

A **trusted forward-auth mode** for the admin GUI. When enabled and the request
arrives from a trusted upstream, CoderAI accepts the user identity from a request
header and establishes an admin session for that user **without** showing the
login form.

### 1. A switch, off by default

A config key (and matching env), e.g.:

```json
"admin": {
  "forward_auth": {
    "enabled": true,
    "user_header": "X-Forwarded-User",
    "groups_header": "X-Forwarded-Groups",
    "admin_group": "admins",
    "shared_secret_header": "X-Coderai-Proxy-Secret",
    "trusted_proxies": ["127.0.0.1", "::1"],
    "auto_create_users": true,
    "keep_local_login": false
  }
}
```

Env equivalents for the container (`CODERAI_ADMIN_FORWARD_AUTH=1`,
`CODERAI_ADMIN_FORWARD_AUTH_SECRET=…`, …) so it can be set from the quadlet
without editing the mounted config.

### 2. The behaviour

- When `forward_auth.enabled` and the request passes the trust check (below),
  read the username from `user_header`. If `auto_create_users`, create the admin
  user on first sight; otherwise require it to already exist in `auth.json`.
- Grant admin scope when `groups_header` contains `admin_group` (or, if no groups
  header is sent, treat every forward-authed user as admin — make this explicit
  in config, don't guess).
- Issue CoderAI's normal signed session for that user, exactly as a successful
  form login would, so the rest of the GUI is unchanged.
- `keep_local_login: false` disables the password form entirely (the proxy is the
  only way in); `true` leaves the form as a fallback for direct/loopback access.
- Logout: since the real session lives upstream, a logout in CoderAI should clear
  the CoderAI session and redirect to a configurable `post_logout_url` (the proxy
  / Digesta logout), not loop back into an auto-login.

### 3. The security contract — this is the part that must be right

A trusted header is a forged-identity hole if it is ever honoured from an
untrusted source. CoderAI must:

- **Only** honour `user_header` when the immediate peer is in `trusted_proxies`
  **and** the request carries the correct `shared_secret_header` value. Fail
  **closed** (fall back to the form, or 401) if either is missing — never trust
  the header on its strength alone.
- **Strip / ignore** any `user_header` and `groups_header` present on a request
  that does **not** come from a trusted proxy, so a client cannot send
  `X-Forwarded-User: admin` directly.
- Keep the readiness probes (`/healthz`, `/health`, `/v1/health`) exempt, as now.
- Leave the `/v1/*` API on its own bearer tokens — this feature is **GUI-only**
  and must not change how the API authenticates.

### 4. What the proxy side then looks like (our end, for reference)

nginx authenticates against Digesta, then forwards the identity plus the shared
secret on to CoderAI:

```nginx
location ^~ /coderai/ {
    auth_request /_digesta_admin;                 # 200 only for a Digesta admin session
    auth_request_set $cr_user $upstream_http_x_user;

    proxy_pass         http://127.0.0.1:8777/;    # full image, internal nginx 8776
    proxy_set_header   X-Forwarded-User   $cr_user;
    proxy_set_header   X-Forwarded-Groups admins;
    proxy_set_header   X-Coderai-Proxy-Secret  "<shared secret, from a file>";
    proxy_set_header   X-Forwarded-Prefix /coderai;
    proxy_set_header   X-Forwarded-Proto  $scheme;
    proxy_read_timeout 3600s;
    # client-supplied copies are dropped by not forwarding them + CoderAI's own strip
}

location = /_digesta_admin {                       # Digesta validates its own session cookie
    internal;
    proxy_pass http://127.0.0.1:8611/api/authz/coderai;  # 204 for admin, 401 otherwise
    proxy_pass_request_body off;
    proxy_set_header Content-Length "";
    proxy_set_header X-Original-URI $request_uri;
}
```

Digesta gets a tiny endpoint (`GET /api/authz/coderai` → `204` if the session is
an `admins` member, else `401`, echoing the username in `X-User`). That part is
ours to build once CoderAI accepts the header.

## Acceptance

- With `forward_auth.enabled` and a correct secret from a trusted proxy, a request
  to `/coderai/admin` lands in the dashboard **with no login form**, as the user
  named in `X-Forwarded-User`, with admin rights.
- The same request **without** the secret, or with the header set by a client
  directly (not via the trusted proxy), does **not** authenticate — it gets the
  form or 401, never an admin session.
- `/v1/*` behaviour is unchanged (bearer tokens).
- `keep_local_login: true` still allows the password form on loopback.

## Context (current prod state)

Full image `ghcr.io/nextime/coderai` **0.2.83** runs GPU-less on the Digesta host,
GUI at `https://digesta.tech/coderai/` behind nginx **basic-auth** for now (that
basic-auth is what this feature replaces). Two separate issues on that deployment
are written up in `/AI/sentenze/coderai-surya-fix.md` and are **not** part of this
request: engine#0 crash-looping GPU-less despite `backend.type=cpu` +
`server.engine_specs`, and `/v1/*` serving `200` without a bearer (tokens not
enforced). This forward-auth feature is independent of both.


---

## Prod activation status — 08/10/2026 (code 0.2.87): one bug left

Forward-auth has shipped (`admin.forward_auth` in config + `CODERAI_ADMIN_FORWARD_AUTH` /
`CODERAI_ADMIN_FORWARD_AUTH_SECRET` env), `/v1/*` now enforces the bearer (401), and
engine#0 starts clean GPU-less. The admin-GUI forward-auth is **almost working on the
Digesta prod host**, with **one reproducible bug left, only on the rootless published-port
path**.

### Prod setup (rootless podman + quadlet)
- Image `ghcr.io/nextime/coderai:latest`, in-image code upgraded to **0.2.87** with the
  baked-in upgrader (`CODERAI_UPGRADE_REF=master` → fetched 0.2.87, `podman commit` back).
  The upgrader also accepts **`CODERAI_UPGRADE_REF=production`** to track the `production`
  branch — we used `master` only because `production` was lagging (0.2.65 at last check).
  Say if prod should track `production` instead.
- `admin.forward_auth`: `enabled`, `shared_secret` set, `admin_group=admins`,
  `trusted_proxies=["127.0.0.1","::1","10.89.0.178"]`, `keep_local_login=true`.
  **Note:** config-only `enabled:true` did **not** turn the feature on — setting the env
  `CODERAI_ADMIN_FORWARD_AUTH=1` (+ `_SECRET`) in the quadlet was required. Worth checking
  whether `enabled` from `config.json` alone is meant to suffice.
- The container's own IP on `digesta-net` is pinned to `10.89.0.178`
  (`Network=digesta-net:ip=10.89.0.178`), because rootless netavark SNATs host→published-port
  traffic to the container's own eth0 IP.
- Topology: host nginx → `127.0.0.1:8777` (published) → container internal nginx `:8776` →
  front (uvicorn) `:18776`.

### What works (0.2.87, tested from inside the container)
- **Direct to the front** `127.0.0.1:18776/admin`, with `X-Coderai-Proxy-Secret` +
  `X-Forwarded-User` + `X-Forwarded-Groups: admins` → **`200` + `Set-Cookie: session=…`**.
  The front forward-auth logic is correct.
- **Through the internal nginx** `127.0.0.1:8776/admin` (same headers, `X-Forwarded-Prefix:
  /coderai`) → **also `200` + `Set-Cookie`**. So the internal nginx forwards the custom
  headers to the front correctly (this hop was broken on 0.2.85/0.2.86; 0.2.87 fixed it).

### The one bug
- **Host → published port** `127.0.0.1:8777/admin` (same headers) → **`302` → `/coderai/login`,
  no `Set-Cookie`**. SSO is not minted. This is the exact path the real host nginx uses, so
  SSO cannot be switched on until it works here.

### Why it's puzzling (for the maintainer)
- The front's uvicorn **access log shows the client for host→8777 as `10.89.0.178`** — which
  **is** in `trusted_proxies`.
- The shared secret matches (sha256-verified; and the in-container `:8776` hop mints with the
  same secret through the same internal nginx).
- The internal nginx forwards the headers (proven by the minting `:8776` hop).
- So the **only** difference between the minting `:8776` hop and the failing `:8777` hop is the
  **source IP into the internal nginx**: `127.0.0.1` (in-container) vs `10.89.0.178` (host via
  rootlessport). Both are in `trusted_proxies`, yet only the loopback one mints.

### Likely area to check
- How `_fa_trusted` resolves the **peer** for the published-port path vs what uvicorn logs as
  the client. If the proxy-headers middleware (`--proxy-headers` / `--forwarded-allow-ips`)
  computes `request.client.host` from `X-Forwarded-For` for the access log, but `_fa_trusted`
  reads a **different** value (the raw socket peer, or a different XFF element), the two can
  disagree — the log says `10.89.0.178` while the trust check sees something not in
  `trusted_proxies`.
- Less likely: the secret header dropped only on the rootlessport path (the `:8776` hop
  forwards it fine).
- Fastest confirmation: log, inside `_fa_trusted`, the actual `peer` value and whether the
  secret header is present, for one host→8777 request.

### Ops state
GUI still behind nginx **basic-auth** on prod (not swapped to `auth_request`): swapping before
host→8777 mints would lock everyone out (no coderai passwords; SSO would be the only way in).
The Digesta side is ready — `GET /api/authz/coderai` returns `204` + `X-User` for a Digesta
admin, `401` otherwise, for nginx `auth_request`.

### Update — 0.2.87, sharper repro (the bug is BEFORE the front, on the published-port path)
Two things ruled out, one thing pinned down:

- **Not a `trusted_proxies` string mismatch.** Broadened it to
  `["127.0.0.1","::1","::ffff:127.0.0.1","10.89.0.178","::ffff:10.89.0.178","10.89.0.1","::ffff:10.89.0.1"]`
  (covering IPv4-mapped-IPv6 and gateway forms) and restarted — host→8777 `/admin` **still** `302`s.

- **The `/admin` request never reaches the front on the published-port path.** For host→`127.0.0.1:8777/admin`
  the response is `302 → /coderai/login` with **`Server: nginx`**, and the **front's uvicorn access log shows
  no `/admin` line at all** (while it logs `/healthz` 200 from the same host→8777 path). So the **internal
  nginx returns the 302 itself**, without proxying `/admin` to the front.

- **Yet the same `/admin` DOES reach the front and mint from inside the container.** `curl …
  http://127.0.0.1:8776/admin` (internal nginx, source `127.0.0.1`) with the same secret + `X-Forwarded-*`
  headers → `200 + Set-Cookie`. And `http://127.0.0.1:18776/admin` (front direct) likewise mints.

So the internal nginx treats `/admin` differently **by source**: proxied to the front from loopback
(`127.0.0.1`), but short-circuited to a `/coderai/login` 302 from the rootless published-port source
(`10.89.0.178`). `/healthz` is proxied to the front from both sources. The nginx.conf we can read
(`/etc/nginx/nginx.conf`: `map` blocks for `X-Forwarded-*`, sub-app `location`s, `location / { proxy_pass
http://coderai; }`) has **no `/admin` block and no source/`allow`/`deny`/`auth_request`/`geo` rule** — so the
gate is somewhere we can't see (an included conf, a baked front/nginx behaviour, or the front proxy upstream
failing for `/admin` from non-loopback and mapping to a login redirect). **This is the thing to fix**: `/admin`
must reach the front over the published-port path exactly as `/healthz` does. A quick repro on any rootless
podman host: publish the full image's `8776` on a loopback port and `curl host:PORT/admin` with the secret +
`X-Forwarded-User`/`-Groups` headers — it 302s to `/coderai/login` and never hits the front, whereas the same
from inside the container mints.

We also tried to **bypass the internal nginx** by publishing the front (`18776`) on its own loopback host
port and pointing at it directly — but the **front binds `127.0.0.1:18776` *inside the container*** (loopback
only), so a published port to `18776` reaches the container's eth0 where nothing listens (connection refused).
The internal nginx on `8776` is the **only** thing reachable from the host, so the `/admin` short-circuit
cannot be worked around at the proxy/ops layer — **the fix has to be in the image** (internal nginx and/or the
front), making `/admin` reach the front over the published-port path the same way `/healthz` already does.

Until this is fixed, the host nginx on prod stays on **basic-auth** (not swapped to `auth_request`), to avoid
locking everyone out of the GUI via the one path nginx uses. Everything else (Digesta `/api/authz/coderai`
endpoint, the quadlet env + config, the pinned container IP) is ready for the swap the moment host→8777 `/admin`
mints.

---

## As shipped (0.2.85)

Config lives at `admin.forward_auth` in `config.json`, is editable from
**Settings → Single sign-on (trusted reverse proxy)**, and is **off by default**:

```json
"admin": {
  "forward_auth": {
    "enabled": false,
    "user_header": "X-Forwarded-User",
    "groups_header": "X-Forwarded-Groups",
    "admin_group": "admins",
    "shared_secret_header": "X-Coderai-Proxy-Secret",
    "shared_secret": "",
    "trusted_proxies": ["127.0.0.1", "::1"],
    "auto_create_users": true,
    "keep_local_login": false,
    "admin_without_groups": false,
    "post_logout_url": ""
  }
}
```

Env overrides for a unit file that must not edit a mounted config:
`CODERAI_ADMIN_FORWARD_AUTH=1|0` and `CODERAI_ADMIN_FORWARD_AUTH_SECRET`. The env
switch can turn it **off** as well as on.

**Differences from the request, all deliberate:**

- `shared_secret` is a config/env value, not only a header *name*. The request
  named the header but not where the expected value comes from.
- **Enabled with no secret stays OFF**, with one line in the log. Without a
  secret there is nothing distinguishing the proxy from a client that sends the
  header, so honouring it would be worse than not running the feature.
- `admin_without_groups` is the explicit decision the request asked for: when
  the proxy sends no groups header, `false` (default) makes the user a
  non-admin, `true` makes every forward-authed user an admin.
- The username from the header is validated (no slashes, ≤128 chars) before it
  is used to look up or create an account.
- An auto-created account gets a random 32-byte password nobody is told, and is
  never asked to change it — it exists only to be reached through the proxy.
- `GET /admin/api/settings` reports `shared_secret_set: true|false` and never
  the secret, so opening Settings does not hand it to the browser. Saving with
  the field blank keeps the stored value.
- A real signed session is minted and injected into the current request, not
  just a rendered page: the dashboard's own `/admin/api/*` calls authenticate by
  session cookie, and a page whose AJAX 401s is not a working GUI.

**Unchanged, as required:** `/v1/*` is untouched by this feature and keeps its
bearer tokens; `/healthz`, `/health` and `/v1/health` stay credential-free.

Separately, and found while testing this: the **front was not enforcing the API
bearer at all** — `GET /v1/models` answered 200 with no header on 0.2.83. Fixed
in 0.2.84; the front now refuses `/v1` without credentials instead of falling
open. That was the second item in the "not part of this request" note, and it
was real.

---

## Answer to the 08/10/2026 report: found, and it is one cause, not two

Both symptoms — `enabled: true` in `config.json` not switching the feature on, and
host→`:8777` refusing to mint while in-container→`:8776` minted — are the **same**
problem. Nothing is wrong with the trust logic.

### 1. The peer checked against `trusted_proxies` is not the socket peer

`_fa_trusted` reads `request.client.host`. uvicorn installs
`ProxyHeadersMiddleware` **by default**, with `trusted_hosts="127.0.0.1"`. The
internal nginx connects to the front over loopback, so that middleware trusts the
hop and **rewrites `scope["client"]` from `X-Forwarded-For`** before any CoderAI
middleware runs. Reproduced exactly (`tests/test_forward_auth_peer_trust.py`):

| hop | nginx sends `X-Forwarded-For` | peer the app sees |
|---|---|---|
| in-container → `:8776` | `127.0.0.1` | `127.0.0.1` |
| host → `:8777` | `10.89.0.178` | `10.89.0.178` |
| real nginx → `:8777` | `203.0.113.5, 10.89.0.178` | `10.89.0.178` |

So the access log and the trust check **do** agree — that was a correct
observation, and it is why the rejection looked impossible. The peer really is
`10.89.0.178`.

### 2. …but the list it was compared against was the default

`admin.forward_auth.trusted_proxies` defaults to `["127.0.0.1", "::1"]`. The
config loader is fine — a `config.json` carrying `admin.forward_auth` round-trips
correctly, and `enabled: true` from the file alone **does** suffice (verified; there
is no second bug there). Which means: **the front was not reading your
`config.json` at all.** That is exactly what "config-only `enabled: true` did not
work, the env var was required" tells us, and it is the known path trap:

> `coderai-entrypoint` creates `$CODERAI_CONFIG_DIR/coderai` and symlinks
> `~/.coderai` to it. The app resolves config from the HOME-style path, so a
> `config.json` at **`$CODERAI_CONFIG_DIR/config.json`** is silently ignored and a
> default is written alongside it.

With the feature forced on by env and the secret supplied by env, everything came
from env — except `trusted_proxies`, which stayed at the default. Loopback is in
that default; `10.89.0.178` is not. Hence: `:8776` mints, `:8777` does not.

### The fix on your side

Move the file to **`$CODERAI_CONFIG_DIR/coderai/config.json`** and your whole
`admin.forward_auth` block takes effect — then you can drop
`CODERAI_ADMIN_FORWARD_AUTH` / `_SECRET` from the quadlet entirely if you like.
Check which file is live before anything else:

```bash
podman exec coderai sh -lc 'ls -l ~/.coderai/config.json; \
  python3 -c "import json;print(json.load(open(\"$HOME/.coderai/config.json\")).get(\"admin\"))"'
```

### The fix on ours (0.2.88)

- **`CODERAI_ADMIN_FORWARD_AUTH_TRUSTED_PROXIES`** (comma- or space-separated) now
  overrides the list, so a unit file that can enable the feature and supply its
  secret can also supply its peers. Enabling by env and then rejecting every
  request from your own proxy was a gap in the contract, not a misuse. An unset or
  blank value falls through to `config.json`.
- **A refused attempt now says why**, once per peer and reason per 60s:

  ```
  [front] admin forward-auth refused: peer is not in trusted_proxies
          ['127.0.0.1', '::1'] (peer '10.89.0.178')
  [front] admin forward-auth refused: no X-Coderai-Proxy-Secret header (peer '10.89.0.178')
  [front] admin forward-auth refused: X-Coderai-Proxy-Secret does not match (peer '10.89.0.178')
  ```

  The log never prints the secret or the value sent. This is the instrumentation
  the report asked for, made permanent: a refusal is a 302 to the login form,
  which is indistinguishable from "no session yet".
- `_fa_trusted`'s docstring no longer says "immediate peer", because it is not —
  believing it was is what made this look impossible.

### What goes in `trusted_proxies`

On a container network, the address the proxy is **SNATed to** — for you
`10.89.0.178` — even though the TCP connection into the front is over loopback.
Keeping `127.0.0.1` and `::1` as well is right: that is the in-container hop you
test with.

### Two answers to your questions

- **Track `production`.** It was lagging when you looked; it now carries 0.2.88 and
  will stay current. Use `CODERAI_UPGRADE_REF=production`.
- **Switching nginx from basic-auth to `auth_request` is safe once host→8777
  mints**, and you no longer have to choose: with `keep_local_login: true` the
  password form stays as a loopback fallback, so a broken SSO cannot lock you out.
  Verify the mint through the real path first, then swap.

---

## Answer to the 0.2.87 follow-up: the 302 IS the front refusing

The sharper repro is good work, but both pillars of "the bug is before the front"
have innocent explanations, and the test that was meant to rule out
`trusted_proxies` could not have.

### `Server: nginx` does not mean nginx wrote the response

nginx sets its own `Server` header on **proxied** responses too, unless you add
`proxy_pass_header Server`. So that header is present whether the 302 came from
nginx or was passed through from the front. It carries no information here.

### The missing `/admin` access-log line is a log filter, not a missing request

`codai/frontproxy/app.py` has `_PollNoiseFilter`, installed on the front's
uvicorn **access** handler unless `--debug-web`:

```python
_READ = ("GET", "HEAD", "OPTIONS")
_WEB_PREFIXES = ("/admin", "/static", "/login", "/logout")
```

A `GET /admin` is **dropped from the access log, from every source**. `/healthz`
is not in that list, which is exactly the asymmetry that was observed:

| request | access log |
|---|---|
| `GET /admin` | **dropped** |
| `GET /healthz` | logged |
| `GET /coderai/admin` | logged (the prefix is not stripped yet) |

So "no `/admin` line" is the expected output for a request the front *did*
handle — the in-container test that minted was not logged either. Re-run any of
this with `--debug-web` and the lines appear. There is no nginx rule to find,
which is also why none was found: `location /` proxies `/admin` to the front
from both sources, as `/healthz` demonstrates.

A `302 → /coderai/login` with no `Set-Cookie` is precisely what the front
returns when forward-auth declines: the middleware calls `_fa_login`, gets
nothing, and falls through to the normal "no session" redirect.

### Why broadening `trusted_proxies` changed nothing

It was broadened in `config.json` — the file we established the app is **not
reading**, which is why `enabled: true` there did nothing and
`CODERAI_ADMIN_FORWARD_AUTH=1` was needed. Editing an unread file cannot change
behaviour, so that test does not exonerate the peer check. The list actually in
force was still the default `["127.0.0.1", "::1"]`: loopback in, container
address out. One cause, all three symptoms.

### And a second, independent bug the `::ffff:` instinct was right about

Adding `::ffff:10.89.0.178` was the correct hunch. The internal nginx listens on
**both** `8776` and `[::]:8776`, so a connection landing on the v6 socket makes
`$remote_addr` — and therefore `X-Forwarded-For`, and therefore the peer — the
IPv4-mapped form. The check compared **strings**, so `::ffff:10.89.0.178` was
refused even with `10.89.0.178` listed, and nothing in the configuration
explained the refusal.

Fixed in **0.2.90**: peers are compared as addresses. A mapped v6 peer matches
the plain v4 entry and vice versa, and **CIDR entries work**, so a container
network is trusted as `10.89.0.0/24` instead of pinning one address with
`Network=digesta-net:ip=…`. A non-address entry is still compared literally.

## Do this, in this order

**1. Upgrade.** `production` carries 0.2.90.

```bash
CODERAI_UPGRADE_REF=production coderai-docker --podman --upgrade \
    ghcr.io/nextime/coderai:latest
systemctl --user restart coderai.service
podman exec coderai grep -m1 __version__ /opt/coderai/app/codai/__init__.py
```

Or pull the signed image — `ghcr.io/nextime/coderai:0.2.88` is published and
`cosign verify`s; 0.2.89/0.2.90 are reachable by `--upgrade`.

**2. Put the peer list where it is read, by env, so the config path stops
mattering** — in the quadlet, beside the two variables already there:

```ini
Environment=CODERAI_ADMIN_FORWARD_AUTH=1
Environment=CODERAI_ADMIN_FORWARD_AUTH_SECRET=…
Environment=CODERAI_ADMIN_FORWARD_AUTH_TRUSTED_PROXIES=127.0.0.1,::1,10.89.0.0/24
```

New in 0.2.88. Previously the switch and the secret could come from a unit file
but the peers could only come from `config.json` — so the feature could be
enabled by env and then reject every request from its own proxy.

**3. Then one request settles it.** 0.2.88+ says why it refused, once per peer
and reason per 60s:

```
[front] admin forward-auth refused: peer is not in trusted_proxies
        ['127.0.0.1', '::1'] (peer '10.89.0.178')
```

If the list in that line is the default pair, `config.json` is still unread and
step 2 is what makes it work. If it names the peer and the line is about the
secret instead, the peer check passed and the secret is the remaining problem.
Either way the next move is in the message rather than in a repro.

**4. Fix the config path anyway**, so Settings and everything else apply:

```bash
podman exec coderai sh -lc 'ls -l ~/.coderai/config.json; \
  python3 -c "import json;print(json.load(open(\"$HOME/.coderai/config.json\")).get(\"admin\"))"'
```

The app resolves config from the HOME-style `$CODERAI_CONFIG_DIR/coderai`; a
file at `$CODERAI_CONFIG_DIR/config.json` is silently ignored.

**5. Swap nginx to `auth_request`** once host→8777 mints. With
`keep_local_login: true` the password form stays as a loopback fallback, so a
broken SSO cannot lock anyone out — the basic-auth caution was right while the
mint was unproven, and stops being necessary once it works.

### Not worth pursuing

- There is no `/admin` short-circuit in nginx to find.
- Publishing the front's `18776` cannot work: it binds `127.0.0.1` inside the
  container by design, so the internal nginx is the only reachable entry point.
  That part of the write-up is correct and the conclusion stands — the fix had to
  be in the image. It is, just not in nginx.
