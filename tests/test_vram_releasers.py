"""Every external VRAM releaser must match the manager's calling convention.

The manager calls each one as `fn(needed_gb)` and sums what they return. A
releaser that declares no parameter does not fail loudly: the TypeError is caught
and printed as a warning, and the VRAM it holds is simply never freed. That is
what happened to the F5-TTS engine — found in a production log as

    Warning in external VRAM releaser: release_f5_engine() takes 0 positional
    arguments but 1 was given

during a FULL VRAM CLEANUP. This checks the shape statically, so a new releaser
cannot be registered with the wrong signature.
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTER = "register_external_vram_releaser"


def _registrations():
    """(file, callable-name) for every register_external_vram_releaser(...) call."""
    for path in (ROOT / "codai").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:                                   # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name != REGISTER:
                continue
            arg = node.args[0]
            target = arg.attr if isinstance(arg, ast.Attribute) else getattr(arg, "id", None)
            if target:
                yield path, target, tree


def _find_def(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def test_there_are_releasers_to_check():
    """A rename that silently emptied this list would make every case below vacuous."""
    assert len(list(_registrations())) >= 4


@pytest.mark.parametrize("path,target,tree", list(_registrations()),
                         ids=lambda v: v if isinstance(v, str) else "")
def test_a_releaser_accepts_needed_gb(path, target, tree):
    fn = _find_def(tree, target)
    if fn is None:
        pytest.skip(f"{target} is not defined in {path.name}")
    params = [a.arg for a in fn.args.args if a.arg != "self"]
    assert params, (
        f"{path.relative_to(ROOT)}:{fn.lineno} {target}() takes no argument, but the "
        f"manager calls every releaser as fn(needed_gb) — it would raise TypeError "
        f"into the warning path and free nothing")
    # Extra parameters are fine only if the manager's single argument still
    # satisfies them — anything beyond the first must have a default.
    required = len(params) - len(fn.args.defaults)
    assert required <= 1, (
        f"{target}() requires {required} arguments; the manager passes exactly one")


@pytest.mark.parametrize("path,target,tree", list(_registrations()),
                         ids=lambda v: v if isinstance(v, str) else "")
def test_a_releaser_returns_a_number(path, target, tree):
    """The manager adds up the return values, so a releaser that returns nothing
    reports 0 GB freed and the eviction loop cannot tell it worked."""
    fn = _find_def(tree, target)
    if fn is None:
        pytest.skip(f"{target} is not defined in {path.name}")
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
    assert returns, f"{target}() never returns a value"
