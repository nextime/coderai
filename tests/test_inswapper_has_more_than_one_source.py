"""Face swapping cannot depend on a repo the Hub has disabled.

Measured 2026-11-01: the publisher's repo now redirects
(deepinsight/inswapper -> deepinsight-unofficial/inswapper) to a repo the
Hub serves with ``"disabled": true``. The download therefore fails in under
a second with

    An error happened while trying to locate the file on the Hub and we
    cannot find the requested files in the local cache. Please check your
    connection and try again or make sure your Internet connection is on.

which is the same sentence an unplugged cable produces, and sent a working
install hunting for a firewall. The network was fine: another public repo
answered 200 from the same container in the same second.

So: try every known source, and when they are all gone say where to put the
file instead of blaming the connection.
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = io.open(os.path.join(ROOT, 'codai/api/faceswap.py'), encoding='utf-8').read()


def _repos():
    import ast
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, 'id', None) == '_INSWAPPER_HF_REPOS' for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise AssertionError('_INSWAPPER_HF_REPOS not found')


def test_there_is_more_than_one_source():
    assert len(_repos()) >= 2


def test_every_source_is_a_repo_id():
    for r in _repos():
        assert r.count('/') == 1 and ' ' not in r, r


def test_the_original_publisher_is_tried_first():
    """If it ever comes back it is the source to prefer, and trying it costs
    one request against a cache miss that was going to happen anyway."""
    assert _repos()[0] == 'deepinsight/inswapper'


def test_the_disabled_repo_is_recorded_as_the_reason():
    assert 'disabled' in SRC and 'deepinsight-unofficial' in SRC, (
        'without the reason, the mirror list looks like redundancy to tidy up')


def _ensure_module():
    """faceswap imports cv2/insightface at module scope; load the source in a
    namespace with just what _ensure_model touches."""
    return SRC


def test_a_later_source_is_tried_when_the_first_one_fails(tmp_path, monkeypatch):
    import types
    mod = types.ModuleType('fs_probe')
    target = tmp_path / 'models' / 'inswapper_128.onnx'
    tried = []

    def _dl(repo_id=None, filename=None, local_dir=None, **kw):
        tried.append(repo_id)
        if len(tried) < 3:
            raise RuntimeError('gone')
        p = os.path.join(local_dir, filename)
        io.open(p, 'w').write('weights')
        return p

    fake_hub = types.ModuleType('huggingface_hub')
    fake_hub.hf_hub_download = _dl
    monkeypatch.setitem(sys.modules, 'huggingface_hub', fake_hub)

    i = SRC.index('def _ensure_model():')
    src = SRC[i:SRC.index('\ndef ', i + 10)]
    ns = {'os': os, '_INSWAPPER_MODEL_PATH': str(target),
          '_INSWAPPER_HF_FILE': 'inswapper_128.onnx',
          '_INSWAPPER_HF_REPOS': ('a/one', 'b/two', 'c/three', 'd/four'),
          'print': lambda *a, **k: None}
    exec(compile(src, 'faceswap-slice', 'exec'), ns)
    ns['_ensure_model']()
    assert tried == ['a/one', 'b/two', 'c/three'], tried
    assert os.path.exists(str(target))
    assert mod is not None


def test_when_every_source_fails_the_error_names_the_path(tmp_path, monkeypatch):
    import types
    fake_hub = types.ModuleType('huggingface_hub')

    def _dl(**kw):
        raise RuntimeError('gone')
    fake_hub.hf_hub_download = _dl
    monkeypatch.setitem(sys.modules, 'huggingface_hub', fake_hub)

    target = tmp_path / 'models' / 'inswapper_128.onnx'
    i = SRC.index('def _ensure_model():')
    src = SRC[i:SRC.index('\ndef ', i + 10)]
    ns = {'os': os, '_INSWAPPER_MODEL_PATH': str(target),
          '_INSWAPPER_HF_FILE': 'inswapper_128.onnx',
          '_INSWAPPER_HF_REPOS': ('a/one', 'b/two'),
          'print': lambda *a, **k: None}
    exec(compile(src, 'faceswap-slice', 'exec'), ns)
    try:
        ns['_ensure_model']()
    except RuntimeError as e:
        msg = str(e)
    else:
        raise AssertionError('a total failure must raise')
    assert str(target) in msg, msg
    assert 'a/one' in msg and 'b/two' in msg, 'say what was tried'
    assert 'connection' not in msg.split('Tried:')[0].lower(), (
        'the connection was fine; blaming it is what cost the hours')


def test_an_already_present_file_is_not_downloaded_again(tmp_path, monkeypatch):
    import types
    fake_hub = types.ModuleType('huggingface_hub')

    def _dl(**kw):
        raise AssertionError('must not download')
    fake_hub.hf_hub_download = _dl
    monkeypatch.setitem(sys.modules, 'huggingface_hub', fake_hub)

    target = tmp_path / 'inswapper_128.onnx'
    io.open(str(target), 'w').write('x')
    i = SRC.index('def _ensure_model():')
    src = SRC[i:SRC.index('\ndef ', i + 10)]
    ns = {'os': os, '_INSWAPPER_MODEL_PATH': str(target),
          '_INSWAPPER_HF_FILE': 'inswapper_128.onnx',
          '_INSWAPPER_HF_REPOS': ('a/one',),
          'print': lambda *a, **k: None}
    exec(compile(src, 'faceswap-slice', 'exec'), ns)
    ns['_ensure_model']()
