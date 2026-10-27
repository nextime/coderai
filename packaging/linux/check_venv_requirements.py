#!/usr/bin/env python3
"""Did this build quietly lose a package the last image had?

The OCI image is a *capture* of a virtualenv: whatever is in the venv ships,
and whatever has fallen out of it is gone from every install that pulls the
image. Nothing in the build log says so, so the image simply loses a feature.

Not hypothetical. `gguf` was uninstalled from the dev venv at some point. It is
in requirements.txt, it was in the previous image, and the rebuilt image could
not load a single GGUF component — reported by diffusers as

    Unable to load weights from checkpoint file for '<file>.gguf'

because its loader takes the .gguf branch, fails to import `gguf`, and that
ImportError is swallowed into a generic OSError about the file. Nothing in the
message mentions a missing package.

requirements.txt is deliberately NOT the reference: it lists optional extras
this venv has never had (vosk, timm, edge-tts, …), so comparing against it
cries wolf. The previous image is the honest baseline — it is what this
install ships today.

Usage:
    check_venv_requirements.py <venv-python> <previous-image> [docker-bin...]

Exit 0 when nothing was lost (or there is no previous image to compare with),
1 when packages disappeared.
"""
import subprocess
import sys

_LIST_IN_VENV = (
    "import importlib.metadata as m;"
    "print('\\n'.join(sorted({(d.metadata['Name'] or '').strip().lower()"
    " for d in m.distributions()} - {''})))"
)

#: dist-info directories in the image, which is the same information without
#: needing a working interpreter inside it.
_LIST_IN_IMAGE = (
    "ls /opt/coderai/python/lib/python3*/site-packages 2>/dev/null"
    " | sed -n 's/^\\(.*\\)-[0-9][^-]*\\.dist-info$/\\1/p'"
    " | tr 'A-Z' 'a-z' | sort -u"
)


def _norm(name: str) -> str:
    return name.replace("_", "-").strip().lower()


def _real(name: str) -> bool:
    """pip leaves a `~`-prefixed dist-info behind when an install is
    interrupted ("Ignoring invalid distribution ~yping-extensions"). It is
    rubbish in both directions and must not read as a lost package."""
    return bool(name) and not name.startswith("~")


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    python, image = argv[1], argv[2]
    docker = argv[3:] or ["docker"]

    probe = subprocess.run(docker + ["image", "inspect", image],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        print(f"[req-check] no previous image '{image}' to compare against — "
              f"nothing to check")
        return 0

    before = subprocess.run(
        docker + ["run", "--rm", "--entrypoint", "/bin/sh", image, "-c", _LIST_IN_IMAGE],
        capture_output=True, text=True)
    if before.returncode != 0 or not before.stdout.strip():
        print(f"[req-check] could not list packages in '{image}' — skipping")
        return 0

    now = subprocess.run([python, "-c", _LIST_IN_VENV],
                         capture_output=True, text=True)
    if now.returncode != 0:
        print(now.stderr.strip() or "could not list the venv's packages",
              file=sys.stderr)
        return 2

    had = {_norm(n) for n in before.stdout.split() if _real(n)}
    has = {_norm(n) for n in now.stdout.split() if _real(n)}
    lost = sorted(had - has)
    if not lost:
        print(f"[req-check] the venv still has every package '{image}' shipped "
              f"({len(had)} distributions)")
        return 0

    print(f"[req-check] these packages are in '{image}' but NOT in the venv "
          f"being captured, so this image would ship without them:")
    for name in lost:
        print(f"  - {name}")
    print("[req-check] install them into the venv and build again, or pass "
          "--no-req-check if they are deliberately dropped.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
