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
