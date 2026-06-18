# Running CoderAI and the `tools/` web UIs behind nginx

Everything here works behind an nginx (or any) reverse proxy. There are two
ways to mount each service; pick per service:

* **Subdomain / root location** — the service owns `/` of a `server_name`
  (e.g. `coderai.example.com`). Works for *every* service with no app changes.
* **Sub-path** — the service lives under a path (e.g. `example.com/coderai/`).
  Supported by **CoderAI** and **`tools/video_editor.py`**. The other
  `tools/` UIs currently need the subdomain/root form (see the table).

| Service                         | Root / subdomain | Sub-path (`/foo/`) |
|---------------------------------|:----------------:|:------------------:|
| CoderAI server (`codai`)        | ✅               | ✅                 |
| `tools/video_editor.py`         | ✅               | ✅                 |
| `tools/videogen.py`             | ✅               | ⚠️ needs work      |
| `tools/review_outputs.py`       | ✅               | ⚠️ needs work      |
| `tools/gen_township_fighters.py`| ✅               | ⚠️ needs work      |

## Headers every proxy block needs

CoderAI builds public URLs (image/video/audio output links, redirects, admin
links) from these headers via `codai/api/urlutils.py`, and `video_editor.py`
honours `X-Forwarded-Prefix` for sub-path mounting:

```nginx
proxy_set_header Host              $host;
proxy_set_header X-Real-IP         $remote_addr;
proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;
proxy_set_header X-Forwarded-Host  $host;
# Sub-path mounts only — tells the app its public prefix:
# proxy_set_header X-Forwarded-Prefix /coderai;
```

Also important for AI workloads:

```nginx
client_max_body_size 1024m;   # large image/audio/video uploads
proxy_read_timeout   3600s;   # long generations / renders
proxy_send_timeout   3600s;
proxy_buffering      off;     # required for SSE streaming (chat, progress)
```

## CoderAI — subdomain (root)

```nginx
server {
    listen 443 ssl;
    server_name coderai.example.com;
    # ssl_certificate ... ; ssl_certificate_key ... ;

    client_max_body_size 1024m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host  $host;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        proxy_buffering off;            # SSE: streamed chat + task progress
    }
}
```

Optionally pin the public URL instead of trusting headers: start CoderAI with
`--url https://coderai.example.com`.

## CoderAI — sub-path (`https://example.com/coderai/`)

```nginx
location /coderai/ {
    proxy_pass http://127.0.0.1:8000/;   # trailing slash strips the prefix
    proxy_http_version 1.1;
    proxy_set_header Host              $host;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Host  $host;
    proxy_set_header X-Forwarded-Prefix /coderai;   # <-- the key line
    proxy_read_timeout 3600s;
    proxy_buffering off;
}
```

CoderAI reads `X-Forwarded-Prefix` into the ASGI `root_path`, so `request.url`,
redirects, `{{ root_path }}` template links, the `ROOT_PATH` JS global, and all
generated file URLs become `/coderai/...` automatically.

## `tools/video_editor.py`

Start it bound to localhost (default) and proxy to it. It works at root and at
a sub-path. For a sub-path, set `X-Forwarded-Prefix`; the page injects a
matching `<base href>` and all its API/media/render URLs are relative, so they
resolve correctly under any mount. It also strips the prefix server-side, so it
works whether or not nginx strips it.

```nginx
# Sub-path: https://example.com/editor/
location /editor/ {
    proxy_pass http://127.0.0.1:8420/;
    proxy_http_version 1.1;
    proxy_set_header Host              $host;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Prefix /editor;
    proxy_read_timeout 3600s;     # long ffmpeg renders
    proxy_send_timeout 3600s;
    proxy_request_buffering off;  # stream large uploads straight through
    client_max_body_size 4096m;   # video/music uploads from the browser machine
}
```

Run with `--no-browser` on a server. The video editor talks to CoderAI over
`--base-url` server-side (not from the browser), so the browser only ever needs
to reach the editor's own origin. Source files can be picked from the server's
media directory or uploaded from the browser machine (hence the larger
`client_max_body_size` / `proxy_request_buffering off` above).

## `tools/videogen.py`, `review_outputs.py`, `gen_township_fighters.py`

Mount each at the root of its own `server_name` (or a dedicated port). These
UIs use absolute (`/...`) asset and API paths plus SSE, so they expect to own
`/`:

```nginx
server {
    listen 443 ssl;
    server_name videogen.example.com;
    location / {
        proxy_pass http://127.0.0.1:7860;   # the tool's --port
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 3600s;
        proxy_buffering off;                # these stream progress over SSE
    }
}
```

Sub-path mounting for these three needs their client URLs made relative (the
same change already applied to `video_editor.py`).
