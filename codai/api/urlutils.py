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

"""Proxy-aware URL utilities.

Consults standard reverse-proxy headers (X-Forwarded-Proto, X-Forwarded-Host,
X-Forwarded-Port, X-Forwarded-Prefix / X-Script-Name) to build the correct
public-facing base URL for file references and redirects.

Priority order when url_setting == 'auto':
  1. X-Forwarded-Host  — overrides host (and implicitly port)
  2. X-Forwarded-Proto — overrides scheme
  3. X-Forwarded-Port  — overrides port when not already in the host
  4. X-Forwarded-Prefix / X-Script-Name / ASGI root_path — path prefix
  5. Host header + configured global_args.port + global_args.https as fallback
"""


def get_public_prefix(request) -> str:
    """Return the path prefix string (no trailing slash), e.g. '/coderai'."""
    if request is None:
        return ""
    prefix = (
        request.headers.get("x-forwarded-prefix")
        or request.headers.get("x-script-name")
        or request.scope.get("root_path", "")
        or ""
    )
    return prefix.rstrip("/")


def get_base_url(request) -> str:
    """Return the public-facing base URL (no trailing slash).

    When global_args.url is set to anything other than 'auto', that value is
    returned verbatim (allowing administrators to pin a specific URL).
    """
    from codai.api.state import get_global_args
    global_args = get_global_args()

    url_setting = getattr(global_args, "url", "auto") if global_args else "auto"
    if url_setting and url_setting != "auto":
        return url_setting.rstrip("/")

    if request is None:
        port = getattr(global_args, "port", 8000) if global_args else 8000
        return f"http://127.0.0.1:{port}"

    headers = request.headers
    fwd_proto = headers.get("x-forwarded-proto") or headers.get("x-scheme")
    fwd_host = headers.get("x-forwarded-host")
    fwd_port = headers.get("x-forwarded-port")
    prefix = get_public_prefix(request)

    if fwd_host:
        # Forwarded host may already carry a port (e.g. "example.com:8443")
        if ":" not in fwd_host and fwd_port:
            host_part = f"{fwd_host}:{fwd_port}"
        else:
            host_part = fwd_host
        proto = fwd_proto or "https"
        # Strip default ports so URLs stay clean
        if proto == "https" and host_part.endswith(":443"):
            host_part = host_part[:-4]
        elif proto == "http" and host_part.endswith(":80"):
            host_part = host_part[:-3]
        return f"{proto}://{host_part}{prefix}"

    # No X-Forwarded-Host — fall back to Host header
    host = headers.get("host", "")
    if not host:
        host = request.client.host if request.client else "127.0.0.1"

    # If Host header already carries a port, honour it; otherwise append the
    # configured server port so the URL is always absolute and unambiguous.
    if ":" not in host:
        port = getattr(global_args, "port", 8000) if global_args else 8000
        if fwd_port:
            host = f"{host}:{fwd_port}"
        else:
            host = f"{host}:{port}"

    use_https = (
        getattr(global_args, "https", False) or getattr(global_args, "pubkey", None)
    )
    proto = fwd_proto or ("https" if use_https else "http")

    # Strip default ports
    if proto == "https" and host.endswith(":443"):
        host = host[:-4]
    elif proto == "http" and host.endswith(":80"):
        host = host[:-3]

    return f"{proto}://{host}{prefix}"


def build_file_url(filename: str, request) -> str:
    """Return the public URL for a generated file served via /v1/files/{filename}."""
    return f"{get_base_url(request)}/v1/files/{filename}"
