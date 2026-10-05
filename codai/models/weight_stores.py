"""Where model weights live OUTSIDE the HF and GGUF caches.

An engine that fetches its own weights materialises them wherever it was pointed,
which is not necessarily a cache coderai scans. colibri is the case that exposed
this: its GLM-5.2 container sat in ``/AI/offloads/colibri/models--mastouri--…`` at
400 GB, while the HF cache held a 4 KB shell of the same repo. "Free disk" deleted
the shell, reported success, and freed nothing — truthfully, and uselessly.

Two rules follow from that, and both are implemented here rather than in the
endpoint, so every caller gets them:

  * a delete reports the BYTES it freed, so "success" can never again mean 4 KB;
  * the roots searched are extensible, so an engine that keeps weights of its own
    is covered by declaring them rather than by someone remembering to patch the
    delete path.

Nothing here deletes by pattern or prefix. A directory is only ever removed when
its name is exactly the HF cache spelling of the model being deleted, which is
what keeps a one-level sibling scan safe.
"""
import os
import shutil
from typing import Iterable, List, Optional, Tuple

# Roots declared by engines at import time: (path, label).
_REGISTERED: List[Tuple[str, str]] = []


def register_weight_root(path: str, label: str = "") -> None:
    """Declare a directory where an engine keeps model weights.

    Engines call this at import. Safe to call repeatedly and with a path that does
    not exist yet — a root is only used if it is a directory when a delete runs.
    """
    path = os.path.abspath(os.path.expanduser((path or "").strip()))
    if not path:
        return
    if not any(p == path for p, _ in _REGISTERED):
        _REGISTERED.append((path, label or os.path.basename(path)))


def registered_roots() -> List[Tuple[str, str]]:
    return list(_REGISTERED)


def hf_dir_name(model_id: str) -> str:
    """The directory name huggingface_hub gives a repo: ``models--org--name``."""
    return "models--" + (model_id or "").strip().replace("/", "--")


def candidate_roots(config=None) -> List[Tuple[str, str]]:
    """Directories that may hold weights for any model, as (path, label).

    Beyond what engines register:

      * the configured disk-offload directory, and
      * its SIBLINGS, one level only.

    The sibling scan exists because engines are pointed at sibling roots by
    convention — coderai's own offload is ``…/offloads/nvidia`` and colibri's
    weights landed in ``…/offloads/colibri``. It widens where we LOOK; it does not
    widen what we delete, which stays an exact ``models--…`` name match.
    """
    roots: List[Tuple[str, str]] = []
    seen = set()

    def _add(path, label):
        if not path:
            return
        path = os.path.abspath(os.path.expanduser(str(path).strip()))
        if path and path not in seen and os.path.isdir(path):
            seen.add(path)
            roots.append((path, label))

    for path, label in _REGISTERED:
        _add(path, label)

    offload = ""
    try:
        if config is not None:
            offload = (getattr(getattr(config, "offload", None), "directory", "") or "").strip()
    except Exception:
        offload = ""
    if offload:
        offload = os.path.abspath(os.path.expanduser(offload))
        _add(offload, "offload")
        parent = os.path.dirname(offload)
        if os.path.isdir(parent):
            try:
                for entry in sorted(os.listdir(parent)):
                    _add(os.path.join(parent, entry), f"offload sibling: {entry}")
            except OSError:
                pass
    return roots


def _iter_repo_dirs(root: str, wanted: str) -> Iterable[str]:
    """Directories under `root` whose name is exactly `wanted`, at depth 0 or 1.

    Depth 1 covers an HF cache root given as ``<x>`` when the repos actually sit in
    ``<x>/hub``, which is how huggingface_hub lays them out.
    """
    direct = os.path.join(root, wanted)
    if os.path.isdir(direct):
        yield direct
    try:
        children = sorted(os.listdir(root))
    except OSError:
        return
    for child in children:
        nested = os.path.join(root, child, wanted)
        if os.path.isdir(nested):
            yield nested


def dir_size(path: str) -> int:
    """Bytes a directory occupies, counting each piece of data once.

    Two kinds of double-counting to avoid, both of which would overstate how much
    a delete is about to free:

      * symlinks are neither followed nor counted — an HF snapshot is a tree of
        links into ``blobs/``;
      * hardlinks are counted once, by (device, inode), since a cache that links
        rather than copies would otherwise report the weights once per link.
    """
    total = 0
    seen_inodes = set()
    for dirpath, dirnames, filenames in os.walk(path, onerror=lambda e: None):
        dirnames[:] = [d for d in dirnames
                       if not os.path.islink(os.path.join(dirpath, d))]
        for name in filenames:
            full = os.path.join(dirpath, name)
            try:
                st = os.lstat(full)
            except OSError:
                continue
            if os.path.islink(full):
                continue
            if st.st_nlink > 1:
                key = (st.st_dev, st.st_ino)
                if key in seen_inodes:
                    continue
                seen_inodes.add(key)
            total += st.st_size
    return total


def prune_dangling_links(model_id: str, config=None) -> List[str]:
    """Remove cache entries that are symlinks to a store that no longer exists.

    The HF cache can hold the model as a SYMLINK into an engine's own directory
    rather than as a copy — which is how coderai's cache showed 4 KB for a 400 GB
    model. Once the store is purged that link dangles, and a dangling entry makes
    the model look present to anything that only stats the name.
    """
    wanted = hf_dir_name(model_id)
    if not wanted or wanted == "models--":
        return []
    removed = []
    roots = [p for p, _ in candidate_roots(config)]
    try:
        from codai.models.cache import get_all_cache_dirs
        roots.extend(d for d in get_all_cache_dirs().values() if d)
    except Exception:
        pass
    checked = set()
    for root in roots:
        for candidate in (os.path.join(root, wanted),
                          os.path.join(root, "hub", wanted)):
            if candidate in checked:
                continue
            checked.add(candidate)
            if os.path.islink(candidate) and not os.path.exists(candidate):
                try:
                    os.unlink(candidate)
                    removed.append(candidate)
                except OSError:
                    pass
    return removed


def find_external_weights(model_id: str, config=None) -> List[Tuple[str, int]]:
    """Every outside-the-cache directory holding `model_id`, as (path, bytes)."""
    wanted = hf_dir_name(model_id)
    if not wanted or wanted == "models--":
        return []
    found: List[Tuple[str, int]] = []
    seen = set()
    for root, _label in candidate_roots(config):
        for path in _iter_repo_dirs(root, wanted):
            real = os.path.realpath(path)
            if real in seen:
                continue
            seen.add(real)
            found.append((path, dir_size(path)))
    return found


def purge_external_weights(model_id: str, config=None,
                           dry_run: bool = False) -> Tuple[int, List[str]]:
    """Delete those directories. Returns (bytes freed, paths removed)."""
    freed = 0
    removed: List[str] = []
    for path, size in find_external_weights(model_id, config):
        if dry_run:
            freed += size
            removed.append(path)
            continue
        try:
            shutil.rmtree(path)
        except OSError:
            continue
        freed += size
        removed.append(path)
    if not dry_run:
        # A cache entry pointing INTO the store we just removed now dangles.
        removed.extend(prune_dangling_links(model_id, config))
    return freed, removed


def human_bytes(n: Optional[int]) -> str:
    """Sizes as the admin UI shows them."""
    try:
        n = float(n or 0)
    except (TypeError, ValueError):
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"
