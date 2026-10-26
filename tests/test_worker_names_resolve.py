"""A worker's own helpers have to exist.

`codai/ocr/workers/*` run OUTSIDE this package: the surya worker runs in its
own venv and may not import `codai` at all, so nothing here imports them and no
test ever executed a line of them. A misspelled helper is therefore invisible
until a page reaches a pod, where it surfaces as

    500 {"detail": "surya ocr failed: name '_opt' is not defined"}

which is what it did -- after the image was built, pushed, signed, and the pod
had booted. The layout pass called `_opt("layout", True)`; the accessor is
named `_page_override`. A source-string assertion had "checked" that line and
proved only that the text was present.

Narrow on purpose: `_`-prefixed names in Load context are the module's own
helpers, so an unbound one is a typo rather than a global from elsewhere. No
imports, no third-party linter, nothing added to requirements.
"""
import ast
import builtins
import glob

import pytest

WORKER_FILES = sorted(glob.glob("codai/ocr/workers/*.py"))

#: The whole package, because the same mistake is not confined to the workers.
#: Run over it, this check found `_QUANT_RE` -- used twice in
#: codai/admin/routes.py and defined nowhere, lost in a rewrite, so browsing
#: any HuggingFace repo containing a .gguf raised NameError. Nothing had ever
#: executed those two lines either.
PACKAGE_FILES = sorted(glob.glob("codai/**/*.py", recursive=True))


def _bound_names(tree):
    out = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
            args = getattr(node, "args", None)
            if args is not None:
                for a in (list(args.posonlyargs) + list(args.args)
                          + list(args.kwonlyargs) + [args.vararg, args.kwarg]):
                    if a is not None:
                        out.add(a.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                out.add((al.asname or al.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            out.add(node.id)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            out.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.add(node.name)
    return out


def _undefined(path):
    tree = ast.parse(open(path).read())
    known = _bound_names(tree)
    found = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                and node.id.startswith("_") and node.id not in known):
            found.setdefault(node.id, node.lineno)
    return found


def test_there_are_worker_files_to_check():
    """A glob that matches nothing would make every test below vacuous."""
    assert WORKER_FILES, "no worker modules found — has the path changed?"


@pytest.mark.parametrize("path", WORKER_FILES)
def test_every_private_helper_a_worker_uses_is_defined(path):
    bad = _undefined(path)
    assert not bad, "%s uses undefined private name(s): %s" % (
        path, ", ".join("%s (line %d)" % (n, l) for n, l in sorted(bad.items())))


def test_the_check_would_have_caught_the_bug_that_prompted_it():
    """Guarding the guard: a checker that passes everything is worse than none."""
    src = ("def _page_override(name, default):\n    return default\n"
           "def go(img):\n    return _opt('layout', True)\n")
    tree = ast.parse(src)
    known = _bound_names(tree)
    assert "_opt" not in known
    assert "_page_override" in known


@pytest.mark.parametrize("path", PACKAGE_FILES)
def test_every_private_helper_in_the_package_is_defined(path):
    """Not only the workers: a private name used and never bound is a
    NameError waiting for the first caller, wherever it lives."""
    bad = _undefined(path)
    assert not bad, "%s uses undefined private name(s): %s" % (
        path, ", ".join("%s (line %d)" % (n, l) for n, l in sorted(bad.items())))


def test_the_package_sweep_covers_a_realistic_number_of_files():
    """A glob that silently matched three files would make the sweep useless."""
    assert len(PACKAGE_FILES) > 100, len(PACKAGE_FILES)


def test_the_quant_label_regex_the_sweep_found_missing_now_exists():
    """It is used by the HF repo browser to label each GGUF file."""
    from codai.admin.routes import _QUANT_RE
    assert _QUANT_RE.search("TheDrummer_Artemis-31B-v1.2-Q4_K_M.gguf").group(1).upper() == "Q4_K_M"
    assert _QUANT_RE.search("Gemma-4-14B-A4B-Heretic.i1-IQ4_XS.gguf").group(1).upper() == "IQ4_XS"
    assert _QUANT_RE.search("m-Q4_K_M-00001-of-00002.gguf").group(1).upper() == "Q4_K_M"
    # No label in the name: reported as unknown rather than guessed at.
    assert _QUANT_RE.search("surya-2.gguf") is None
