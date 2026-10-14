#!/usr/bin/env python3
"""Rasterise PDF pages in a process of their own.

pdfium permits exactly one caller at a time. In-process that means a lock, and a
lock means all of an install's rasterisation runs on one core -- which became the
ceiling on OCR throughput the moment 0.3.66 made the serialisation explicit.
pdfium IS safe one-document-per-process, so the way to get parallelism back is a
page off the main process entirely.

Run as a SCRIPT, never imported: `import codai.ocr` costs 377 MB and 1.5 s (it
reaches codai.models.manager and litellm), and paying that per worker in a change
whose whole purpose is bounding memory would be self-defeating. This file imports
pypdfium2 and PIL and nothing else -- about 25 MB.

Protocol, newline-delimited JSON with raw binary payloads, on stdin/stdout:

    -> {"cmd": "open", "dpi": 250, "nbytes": 123456}  + <nbytes of PDF>
    <- {"ok": true, "pages": 40, "est": 691000000}

    -> {"cmd": "render", "start": 0, "count": 4}
    <- {"ok": true, "pages": [{"w": 2066, "h": 2924, "nbytes": 18122952}, ...]}
       + <concatenated raw RGB bytes, in that order>

    -> {"cmd": "close"}                     release the document, keep the process
    <- {"ok": true}

    -> {"cmd": "shutdown"}
    <- {"ok": true}                         then exit

Raw RGB rather than PNG: the parent rebuilds the page with Image.frombytes, and
a page that is about to be read by an OCR engine must not pass through a lossy
codec, while PNG would cost more CPU than the pipe it saves.

Errors answer {"ok": false, "error": str, "status": int} and leave the process
usable -- one undecodable document must not cost the pool a worker.
"""

import json
import sys


class _Doc:
    """The one document this worker holds open."""

    def __init__(self):
        self.pdf = None
        self.scale = 1.0
        self.n = 0

    def close(self):
        pdf, self.pdf = self.pdf, None
        self.n = 0
        if pdf is not None:
            try:
                pdf.close()
            except Exception:
                pass


def _reply(obj, payload=b""):
    out = sys.stdout.buffer
    out.write(json.dumps(obj).encode("utf-8") + b"\n")
    if payload:
        out.write(payload)
    out.flush()


def _read_exactly(n: int) -> bytes:
    """Read n bytes or die: a short read means the parent is gone or the frame is
    desynchronised, and guessing at a partial PDF is how a valid document gets
    reported as corrupt."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sys.stdin.buffer.read(n - len(buf))
        if not chunk:
            raise EOFError(f"stdin closed after {len(buf)} of {n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def main() -> int:
    try:
        import pypdfium2 as pdfium
        from PIL import Image          # noqa: F401  (frombytes is the parent's job)
    except Exception as e:
        _reply({"ready": False, "error": f"rasteriser deps missing: {e}"})
        return 1
    _reply({"ready": True})

    doc = _Doc()
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            break
        try:
            msg = json.loads(line.decode("utf-8"))
        except Exception as e:
            _reply({"ok": False, "error": f"bad request frame: {e}", "status": 500})
            continue
        cmd = msg.get("cmd")

        if cmd == "shutdown":
            doc.close()
            _reply({"ok": True})
            return 0

        if cmd == "open":
            doc.close()
            try:
                data = _read_exactly(int(msg.get("nbytes") or 0))
            except Exception as e:
                _reply({"ok": False, "error": str(e), "status": 500})
                return 1          # the frame is lost; the parent must replace us
            dpi = int(msg.get("dpi") or 200)
            try:
                doc.pdf = pdfium.PdfDocument(data)
                doc.scale = max(0.5, dpi / 72.0)
                doc.n = len(doc.pdf)
                # Page COUNT only. Sizing every page here cost 593 ms on a
                # 47-page document (it loads each page object), which is paid on
                # every chunk because the parent re-opens per chunk -- more than
                # the render it was meant to inform. The parent prices documents
                # itself, cheaply, with get_page_size().
            except Exception as e:
                doc.close()
                _reply({"ok": False,
                        "error": f"failed to rasterise PDF (pypdfium2): {e}",
                        "status": 400})
                continue
            _reply({"ok": True, "pages": doc.n})
            continue

        if cmd == "render":
            if doc.pdf is None:
                _reply({"ok": False, "error": "no document open", "status": 500})
                continue
            start = max(0, int(msg.get("start") or 0))
            count = max(0, int(msg.get("count") or 0))
            metas, blobs = [], []
            try:
                for i in range(start, min(doc.n, start + count)):
                    page = doc.pdf[i]
                    bitmap = page.render(scale=doc.scale)
                    img = bitmap.to_pil().convert("RGB")
                    raw = img.tobytes()
                    metas.append({"w": img.width, "h": img.height,
                                  "nbytes": len(raw)})
                    blobs.append(raw)
                    # Drop each page before rendering the next: this process is
                    # as bound by residency as its parent.
                    del img, bitmap, page
            except Exception as e:
                _reply({"ok": False,
                        "error": f"failed to rasterise PDF (pypdfium2): {e}",
                        "status": 400})
                continue
            _reply({"ok": True, "pages": metas}, b"".join(blobs))
            continue

        if cmd == "close":
            doc.close()
            _reply({"ok": True})
            continue

        _reply({"ok": False, "error": f"unknown cmd {cmd!r}", "status": 500})

    doc.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
