#!/usr/bin/env python3
"""Make a GGUF self-consistent after its NextN/MTP block was quantized away.

Some Qwen3.5-Next quants drop the trailing NextN (multi-token-prediction) block
— it is useless without a draft runner — but leave ``<arch>.block_count``
counting it. llama.cpp loads blocks ``0 … block_count-1``, so it asks for a block
the file does not contain and fails with

    missing tensor 'blk.64.attn_norm.weight'

which reads like a corrupt download. The file is intact; the header promises one
block too many.

coderai corrects that at LOAD time where it can (codai/backends/gguf_mtp.py,
through llama.cpp's own metadata override). That cannot work when a per-layer
metadata ARRAY also pins the count — ``qwen35.attention.recurrent_layers`` has
one entry per block, llama.cpp validates its length against block_count, and an
override can only carry a scalar, never a shorter array. Correct the count and
the array disagrees; leave it and the block is missing.

This tool resolves it in the file instead: it lowers the declared count and drops
the trailing entry of each per-layer array, which is exactly the entry describing
the block that is not there. Tensor data is copied byte-for-byte and the data
section keeps its alignment, so every tensor offset stays valid.

It refuses anything it cannot prove is this case: the blocks present must be a
contiguous run from zero, and exactly ``nextn_predict_layers`` of them must be
missing from the end. A download that really is truncated loses blocks from
wherever the transfer stopped and will not match.

    python3 tools/gguf_fix_dropped_mtp.py MODEL.gguf [-o OUT.gguf] [--inspect]
"""

import argparse
import os
import shutil
import struct
import sys

GGUF_MAGIC = b'GGUF'

# GGUF metadata value types.
(T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL,
 T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64) = range(13)

_FIXED = {T_UINT8: '<B', T_INT8: '<b', T_UINT16: '<H', T_INT16: '<h',
          T_UINT32: '<I', T_INT32: '<i', T_FLOAT32: '<f', T_BOOL: '<?',
          T_UINT64: '<Q', T_INT64: '<q', T_FLOAT64: '<d'}


class Reader:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError("unexpected end of file reading the header")
        return b

    def u32(self):
        return struct.unpack('<I', self.raw(4))[0]

    def u64(self):
        return struct.unpack('<Q', self.raw(8))[0]

    def string(self):
        return self.raw(self.u64()).decode('utf-8', 'replace')

    def value(self, vtype):
        if vtype in _FIXED:
            fmt = _FIXED[vtype]
            return struct.unpack(fmt, self.raw(struct.calcsize(fmt)))[0]
        if vtype == T_STRING:
            return self.string()
        if vtype == T_ARRAY:
            etype = self.u32()
            count = self.u64()
            return (etype, [self.value(etype) for _ in range(count)])
        raise ValueError(f"unknown GGUF value type {vtype}")


def _w_string(s):
    b = s.encode('utf-8')
    return struct.pack('<Q', len(b)) + b


def _w_value(vtype, val):
    if vtype in _FIXED:
        return struct.pack(_FIXED[vtype], val)
    if vtype == T_STRING:
        return _w_string(val)
    if vtype == T_ARRAY:
        etype, items = val
        out = [struct.pack('<I', etype), struct.pack('<Q', len(items))]
        out += [_w_value(etype, it) for it in items]
        return b''.join(out)
    raise ValueError(f"cannot write GGUF value type {vtype}")


def read_header(path):
    """(version, kvs, tensors, data_offset). kvs keeps order and types."""
    with open(path, 'rb') as f:
        r = Reader(f)
        if r.raw(4) != GGUF_MAGIC:
            raise ValueError(f"{path} is not a GGUF file")
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        kvs = []
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            kvs.append([key, vtype, r.value(vtype)])
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            ttype = r.u32()
            offset = r.u64()
            tensors.append([name, dims, ttype, offset])
        align = 32
        for key, _, val in kvs:
            if key == 'general.alignment':
                align = int(val) or 32
        pos = f.tell()
        pad = (align - (pos % align)) % align
        return version, kvs, tensors, pos + pad, align


def describe(path):
    version, kvs, tensors, data_off, align = read_header(path)
    arch = next((v for k, _, v in kvs if k == 'general.architecture'), None)
    declared = next((int(v) for k, _, v in kvs if k.endswith('.block_count')), 0)
    nextn = next((int(v) for k, _, v in kvs
                  if k.endswith('.nextn_predict_layers')), 0)
    blocks = sorted({int(n.split('.')[1]) for n, *_ in tensors
                     if n.startswith('blk.') and n.split('.')[1].isdigit()})
    per_layer = [(k, len(v[1])) for k, t, v in kvs
                 if t == T_ARRAY and declared and len(v[1]) == declared]
    return {
        'version': version, 'arch': arch, 'declared': declared, 'nextn': nextn,
        'present': len(blocks),
        'contiguous': bool(blocks) and blocks == list(range(len(blocks))),
        'per_layer_arrays': per_layer, 'n_tensors': len(tensors),
        'data_offset': data_off, 'alignment': align,
        'file_size': os.path.getsize(path),
    }


def repair(path, out_path):
    info = describe(path)
    arch, declared = info['arch'], info['declared']
    nextn, present = info['nextn'], info['present']

    if not arch or declared <= 0:
        raise SystemExit("refusing: no architecture or block_count in the header")
    if nextn <= 0:
        raise SystemExit(f"refusing: {arch}.nextn_predict_layers is {nextn}; this "
                         f"file does not claim to have NextN blocks, so a missing "
                         f"block is not the known case")
    if not info['contiguous']:
        raise SystemExit("refusing: the blocks present are not a contiguous run "
                         "from 0 — blocks missing from the middle mean a damaged "
                         "file, which must not be reinterpreted into a model that "
                         "loads and answers nonsense")
    if declared - present != nextn:
        raise SystemExit(f"refusing: {declared} declared, {present} present, "
                         f"nextn={nextn} — the arithmetic does not add up, so this "
                         f"is not simply a dropped draft head")

    version, kvs, tensors, data_off, align = read_header(path)
    changes = []
    for kv in kvs:
        key, vtype, val = kv
        if key.endswith('.block_count'):
            kv[2] = present
            changes.append(f"{key}: {declared} -> {present}")
        elif key.endswith('.nextn_predict_layers'):
            kv[2] = 0
            changes.append(f"{key}: {nextn} -> 0")
        elif vtype == T_ARRAY and len(val[1]) == declared:
            etype, items = val
            kv[2] = (etype, items[:present])
            changes.append(f"{key}: {declared} -> {present} entries "
                           f"(dropped the trailing entry, which described the "
                           f"block that is not in the file)")

    body = [GGUF_MAGIC, struct.pack('<I', version),
            struct.pack('<Q', len(tensors)), struct.pack('<Q', len(kvs))]
    for key, vtype, val in kvs:
        body += [_w_string(key), struct.pack('<I', vtype), _w_value(vtype, val)]
    for name, dims, ttype, offset in tensors:
        body += [_w_string(name), struct.pack('<I', len(dims))]
        body += [struct.pack('<Q', d) for d in dims]
        body += [struct.pack('<I', ttype), struct.pack('<Q', offset)]
    header = b''.join(body)
    pad = (align - (len(header) % align)) % align
    header += b'\x00' * pad

    # Tensor offsets are relative to the start of the data section, so the data
    # must keep its alignment but need not keep its absolute position.
    src_size = os.path.getsize(path)
    data_bytes = src_size - data_off
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(out_path))).free
    need = len(header) + data_bytes
    if free < need + (1 << 30):
        raise SystemExit(f"refusing: need {need/1e9:.1f} GB plus headroom, "
                         f"{free/1e9:.1f} GB free")

    print("changes:")
    for c in changes:
        print("   ", c)
    print(f"copying {data_bytes/1e9:.2f} GB of tensor data "
          f"(header {len(header)} bytes, alignment {align})")
    tmp = out_path + ".partial"
    with open(path, 'rb') as src, open(tmp, 'wb') as dst:
        dst.write(header)
        src.seek(data_off)
        while True:
            chunk = src.read(16 << 20)
            if not chunk:
                break
            dst.write(chunk)
    os.replace(tmp, out_path)
    return changes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path')
    ap.add_argument('-o', '--out')
    ap.add_argument('--inspect', action='store_true',
                    help='report the layout and change nothing')
    a = ap.parse_args()

    info = describe(a.path)
    print(f"{os.path.basename(a.path)}")
    for k in ('arch', 'declared', 'nextn', 'present', 'contiguous',
              'per_layer_arrays', 'n_tensors', 'data_offset', 'alignment'):
        print(f"   {k}: {info[k]}")
    if a.inspect:
        return 0

    out = a.out or (os.path.splitext(a.path)[0] + '.mtpfixed.gguf')
    if os.path.exists(out):
        raise SystemExit(f"refusing: {out} already exists")
    repair(a.path, out)
    after = describe(out)
    print("\nafter:")
    for k in ('declared', 'nextn', 'present', 'per_layer_arrays', 'n_tensors',
              'data_offset', 'file_size'):
        print(f"   {k}: {after[k]}")
    ok = (after['declared'] == after['present'] and after['nextn'] == 0
          and all(n == after['declared'] for _, n in after['per_layer_arrays'])
          and after['n_tensors'] == info['n_tensors'])
    print("\nself-consistent:", ok)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
