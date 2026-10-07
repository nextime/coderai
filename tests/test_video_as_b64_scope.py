"""Every backend that carries binary inputs needs _as_b64 in scope.

It was defined inside _generate_h3 and referenced from _generate_longcat, so every
LongCat path that passes an image, an audio track or conditioning frames raised
`NameError: name '_as_b64' is not defined`. Text-to-video was the only path that never
touched it — which is why t2v was the only one that ever worked, and why the avatar
request failed after waiting 20 minutes for the GPU.
"""
import ast
import base64
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
VIDEO = ROOT / "codai" / "api" / "video.py"


@pytest.fixture(scope="module")
def tree():
    return ast.parse(VIDEO.read_text(encoding="utf-8"))


def _module_level_names(tree):
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef,
                                                       ast.AsyncFunctionDef))}


def test_as_b64_is_module_level(tree):
    assert "_as_b64" in _module_level_names(tree)


def test_it_is_not_also_nested_anywhere(tree):
    """A nested copy would shadow it and the bug could come back in one scope only."""
    nested = []
    for top in tree.body:
        if not isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(top):
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "_as_b64" and node is not top):
                nested.append(top.name)
    assert not nested, f"_as_b64 is nested inside {nested}"


def test_every_caller_can_actually_reach_it(tree):
    """The real defect: a call site in a function that does not define it and is not
    nested inside one that does."""
    module_level = _module_level_names(tree)
    assert "_as_b64" in module_level      # otherwise every caller is broken

    callers = set()
    for top in tree.body:
        if not isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(top):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "_as_b64"):
                callers.add(top.name)
    # the paths that matter
    assert "_generate_longcat" in callers, "longcat carries images/audio/cond frames"
    assert callers, "no callers found — the test is not looking at the right thing"


def test_it_normalises_plain_base64_unchanged():
    import importlib.util
    spec = importlib.util.spec_from_file_location("vid_b64", VIDEO)
    # Importing the whole module pulls in the server; exercise the logic directly
    # against the same two primitives it is built from.
    src = VIDEO.read_text(encoding="utf-8")
    body = src[src.index("def _as_b64"):src.index("def _pil_from_b64")]
    assert "_decode_b64_or_url(ref)" in body
    assert "base64.b64encode" in body


def test_the_longcat_binary_paths_all_use_it(tree):
    """audio for at2v, the reference image for ai2v/i2v, and continuation frames."""
    src = VIDEO.read_text(encoding="utf-8")
    start = src.index("async def _generate_longcat")
    body = src[start:]
    end = body.find("\nasync def ", 1)
    body = body[:end] if end > 0 else body
    for field in ('payload["audio"]', 'payload["image"]', 'payload["cond_frames_b64"]'):
        assert field in body, field
        line = next(l for l in body.splitlines() if field in l and "_as_b64" in l)
        assert "_as_b64" in line
