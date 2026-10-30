"""Loading a GGUF whose multi-token-prediction block was quantized away.

A Qwen3.5-Next / Qwen4-expert checkpoint ends with one or more NextN blocks: a
speculative-decoding draft head, counted by ``<arch>.nextn_predict_layers``.
Several quantizers drop those blocks — they are dead weight without a draft
runner — but leave ``<arch>.block_count`` counting them. llama.cpp then asks for
a block the file does not contain and fails the load with a bare

    missing tensor 'blk.64.attn_norm.weight'

which reads exactly like a truncated download. The file is intact: it declares
``block_count`` blocks and carries ``block_count - nextn_predict_layers`` of
them. Overriding the declared count at load time (llama.cpp's own
``--override-kv``, ``kv_overrides=`` in llama-cpp-python) loads the model as what
it actually is — the dense stack, without the draft head.

The override is only offered when the file's shape *proves* that is the
situation: the blocks present are a contiguous run from 0, and exactly
``nextn_predict_layers`` of them are missing from the end. A download that
really is truncated loses blocks from wherever the transfer stopped and does not
match, so it still fails loudly instead of being silently reinterpreted.
"""

import struct

_LAYOUT_CACHE: dict = {}


def _read_str(f):
    ln = struct.unpack('<Q', f.read(8))[0]
    return f.read(ln).decode('utf-8', 'replace')


def _read_val(f, vt):
    if vt == 0:  return struct.unpack('<B', f.read(1))[0]
    if vt == 1:  return struct.unpack('<b', f.read(1))[0]
    if vt == 2:  return struct.unpack('<H', f.read(2))[0]
    if vt == 3:  return struct.unpack('<h', f.read(2))[0]
    if vt == 4:  return struct.unpack('<I', f.read(4))[0]
    if vt == 5:  return struct.unpack('<i', f.read(4))[0]
    if vt == 6:  return struct.unpack('<f', f.read(4))[0]
    if vt == 7:  return struct.unpack('<?', f.read(1))[0]
    if vt == 8:  return _read_str(f)
    if vt == 10: return struct.unpack('<Q', f.read(8))[0]
    if vt == 11: return struct.unpack('<q', f.read(8))[0]
    if vt == 12: return struct.unpack('<d', f.read(8))[0]
    if vt == 9:
        et = struct.unpack('<I', f.read(4))[0]
        cnt = struct.unpack('<Q', f.read(8))[0]
        return [_read_val(f, et) for _ in range(cnt)]
    raise ValueError(f"unknown gguf value type {vt}")


def read_layout(path) -> dict:
    """Declared vs. present block layout of a GGUF, and its tensor types.

    Returns ``{arch, block_count, nextn, blocks_present, contiguous,
    tensor_types}``, or ``{}`` when the header cannot be read. Unlike the other
    GGUF probes in this package this one walks the tensor *index* as well — the
    whole question is which blocks the file actually carries — but it still reads
    no tensor data.

    ``tensor_types`` is the ggml type id histogram, ``{type_id: count}``. It is
    collected here because this walk already passes every type field, and
    :mod:`codai.backends.gguf_preflight` needs it to answer whether the loaded
    llama.cpp can read the file at all. Four GGUF header parsers in one package
    is three too many, so this one is shared rather than copied.
    """
    if not path:
        return {}
    if path in _LAYOUT_CACHE:
        return _LAYOUT_CACHE[path]
    out: dict = {}
    try:
        with open(path, 'rb') as f:
            if f.read(4) != b'GGUF':
                _LAYOUT_CACHE[path] = {}
                return {}
            struct.unpack('<I', f.read(4))                      # version
            n_tensors = struct.unpack('<Q', f.read(8))[0]
            n_kv = struct.unpack('<Q', f.read(8))[0]
            arch, block_count, nextn = None, 0, 0
            array_lengths = {}
            for _ in range(n_kv):
                key = _read_str(f)
                val = _read_val(f, struct.unpack('<I', f.read(4))[0])
                if isinstance(val, list):
                    array_lengths[key] = len(val)
                if key == 'general.architecture':
                    arch = str(val)
                elif key.endswith('.block_count'):
                    block_count = int(val)
                elif key.endswith('.nextn_predict_layers'):
                    nextn = int(val)
            blocks = set()
            types: dict = {}
            for _ in range(n_tensors):
                name = _read_str(f)
                n_dims = struct.unpack('<I', f.read(4))[0]
                f.read(8 * n_dims)                              # dims
                ttype = struct.unpack('<I', f.read(4))[0]       # ggml type
                f.read(8)                                       # offset
                types[ttype] = types.get(ttype, 0) + 1
                if name.startswith('blk.'):
                    part = name.split('.', 2)[1]
                    if part.isdigit():
                        blocks.add(int(part))
            out = {
                'arch': arch,
                'block_count': block_count,
                'nextn': nextn,
                'blocks_present': len(blocks),
                'contiguous': bool(blocks) and blocks == set(range(len(blocks))),
                'tensor_types': types,
                # Metadata arrays with one entry PER LAYER. llama.cpp validates
                # their length against block_count, so they decide whether the
                # count can be overridden at all -- see block_count_override.
                'per_layer_arrays': {k: n for k, n in array_lengths.items()
                                     if block_count and n == block_count},
            }
    except Exception:
        out = {}
    _LAYOUT_CACHE[path] = out
    return out


def block_count_override(path):
    """Metadata overrides that make a de-MTP'd GGUF load as what it is.

    Returns ``{"<arch>.block_count": n, "<arch>.nextn_predict_layers": 0}``, or
    ``None`` whenever the file's shape does not prove the NextN blocks were
    dropped — including a file that is simply missing blocks, which must keep
    failing.

    Both keys are set, not just the count. Whether llama.cpp loads
    ``block_count`` blocks or ``block_count - nextn_predict_layers`` of them has
    changed between builds: the build this was written against does the former
    (it asked for ``blk.64`` of a 65/1 file), and a build doing the latter would
    read a corrected count of 64 as 63 and quietly leave a real layer out of the
    model. Saying "64 blocks, none of them NextN" is true under both.
    """
    layout = read_layout(path)
    if not layout or not layout.get('arch'):
        return None
    declared = int(layout.get('block_count') or 0)
    nextn = int(layout.get('nextn') or 0)
    present = int(layout.get('blocks_present') or 0)
    if nextn <= 0 or declared <= 0 or present <= 0:
        return None
    if not layout.get('contiguous'):
        return None
    if declared - present != nextn:
        return None
    # A per-layer metadata ARRAY pins the count. llama.cpp validates arrays like
    # `<arch>.attention.recurrent_layers` against block_count, and kv_overrides
    # can only carry a scalar -- bool, int, float or str -- so there is no way to
    # shorten the array to match. Overriding the count regardless does not fix
    # the load, it only changes which error it fails with: measured live on
    # 2026-10-30, a Qwen3.5 quant went from
    #     missing tensor 'blk.64.attn_norm.weight'
    # to
    #     key qwen35.attention.recurrent_layers has wrong array length;
    #     expected 64, got 65
    # which is strictly worse -- the first names the real problem. So decline,
    # and let the honest error stand. unrepairable_reason() says why.
    if layout.get('per_layer_arrays'):
        return None
    return {f"{layout['arch']}.block_count": present,
            f"{layout['arch']}.nextn_predict_layers": 0}


def unrepairable_reason(path) -> str:
    """Why a dropped-MTP GGUF could not be repaired, or '' when nothing applies.

    Only speaks for the one case worth explaining: the file IS a quant whose
    NextN block was dropped, but a per-layer array pins the block count, so the
    repair is impossible rather than merely unnecessary.
    """
    layout = read_layout(path)
    if not layout or not layout.get('arch'):
        return ''
    declared = int(layout.get('block_count') or 0)
    nextn = int(layout.get('nextn') or 0)
    present = int(layout.get('blocks_present') or 0)
    arrays = layout.get('per_layer_arrays') or {}
    if not (nextn > 0 and declared > 0 and present > 0
            and layout.get('contiguous') and declared - present == nextn
            and arrays):
        return ''
    names = ", ".join(f"{k} ({n} entries)" for k, n in sorted(arrays.items()))
    return (
        f"this quant dropped its {nextn} NextN (multi-token-prediction) block, so "
        f"it declares {declared} blocks and carries {present}. Correcting the "
        f"declared count would normally fix that, but {names} has one entry per "
        f"layer and llama.cpp checks its length against block_count -- and a "
        f"metadata override can only carry a single value, not a shorter array. "
        f"So the count cannot be corrected without rewriting the file. Use a "
        f"quant that keeps its NextN block (or one built without it), or a "
        f"llama.cpp that derives the layer count from nextn_predict_layers.")


def explain(path) -> str:
    """One line saying what the override does, for the load log."""
    layout = read_layout(path)
    return (f"GGUF declares {layout.get('block_count')} blocks but carries "
            f"{layout.get('blocks_present')}: the "
            f"{layout.get('nextn')} NextN (multi-token-prediction) block(s) were "
            f"dropped by the quantizer. Overriding "
            f"{layout.get('arch')}.block_count to {layout.get('blocks_present')} "
            f"— the model serves as its dense stack, without the draft head.")
