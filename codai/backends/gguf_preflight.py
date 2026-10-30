"""Can this build read this file at all — answered before anything is loaded.

A GGUF that the installed llama.cpp cannot read fails late and expensively. By
the time the load is attempted, models have already been evicted to make room,
VRAM has been reserved, and for a large file minutes have gone into reading it
— and the error that comes back is llama.cpp's, phrased for whoever wrote
llama.cpp. Two such files turned up on one afternoon:

  * a Qwen3.8-Flash-Next quant whose architecture (``qwen4exp``) this build does
    not implement, which cost a whole diagnosis to establish;
  * a 27B "Flash" quant that looked perfect — right architecture, every block
    present, an mmproj alongside — and carried 402 weight tensors in ggml type
    143, a rotation-based format from a fork. Nothing in the file's metadata
    says so; the type ids are in the tensor index, which no error message reads
    back to you.

The checks here are metadata-only: the GGUF header and tensor index, no tensor
data, no model load, no eviction. That is the whole point — a check that has to
load the model to decide is not a check that can run before loading the model.

Deliberately NOT here: a list of architectures this build supports. There is no
API for it, llama.cpp already reports ``unknown model architecture: '<name>'``
in its own log, and the loader surfaces that. Guessing the list from the shared
library's strings would invent a second source of truth that can be wrong in the
direction that blocks a model which would have worked.
"""

from codai.backends.gguf_mtp import read_layout


class GgufUnreadable(RuntimeError):
    """This build cannot read this file, decided from its header.

    Its own type, because the VRAM-reservation path turns a failed reservation
    into a printed warning and carries on — correctly, since a reservation that
    could not be made is often still a load that works. This is not that: it is
    a verdict, and a verdict swallowed into a warning is a bug that reappears as
    the original confusing failure a minute later. The host-RAM refusal is
    handled the same way for the same reason.
    """

    def __init__(self, message, path=None, types=None):
        super().__init__(message)
        self.path = path
        self.types = dict(types or {})


def unsupported_tensor_types(path, type_count) -> dict:
    """``{type_id: tensor_count}`` for types this build has no enum value for.

    ``type_count`` is llama.cpp's own ``GGML_TYPE_COUNT``, so the bound comes
    from the loaded library rather than from a list kept here.

    Only ids at or above that bound are reported, and that conservatism is the
    design. The Python binding's ``GGML_TYPE_*`` constants are an INCOMPLETE
    mirror of the compiled enum -- it defines 0-3, 6-29 and 39-42, while 30 is
    BF16, which plenty of files we serve use and which loads perfectly. Treating
    "absent from the binding's constants" as "unsupported" would reject those
    outright. An id below the bound that the build still cannot handle is left
    to llama.cpp to refuse in its own words; this only catches what is provably
    outside the enum.
    """
    try:
        bound = int(type_count)
    except (TypeError, ValueError):
        return {}
    if bound <= 0:
        return {}
    types = (read_layout(path) or {}).get('tensor_types') or {}
    return {tid: n for tid, n in sorted(types.items()) if int(tid) >= bound}


def explain_tensor_types(path, unsupported: dict, type_count) -> str:
    """Why this file cannot be read, in terms of the file rather than of ggml."""
    layout = read_layout(path) or {}
    total = sum((layout.get('tensor_types') or {}).values()) or 0
    affected = sum(unsupported.values())
    listed = ", ".join(f"type {tid} ({n} tensors)"
                       for tid, n in sorted(unsupported.items()))
    return (
        f"this GGUF stores {affected} of its {total} tensors in a quantization "
        f"format the installed llama.cpp does not implement: {listed}. The "
        f"highest type this build knows is {int(type_count) - 1} "
        f"(GGML_TYPE_COUNT={int(type_count)}), so these come from a fork or a "
        f"newer ggml -- a rotation-folded quant (PRISM and similar) is the usual "
        f"source, and nothing in the file's metadata announces it. Refusing here "
        f"rather than after eviction and a full read of the weights. Use a quant "
        f"in a standard format, or run it on a build that implements these types."
    )


def architecture(path) -> str:
    """``general.architecture``, or '' when it cannot be read.

    Logged before a load so that llama.cpp's own ``unknown model architecture``
    is attributable to a name without re-reading the file by hand.
    """
    return str((read_layout(path) or {}).get('arch') or '')
