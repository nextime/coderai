# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""docTR (Mindee) OCR engine.

Apache-2.0, pure-PyTorch (uses the torch already in the stack). Detection +
recognition predictor; docTR groups words into blocks/lines with reading order, so
transcription comes out ordered. Weaker on complex table layout than PaddleOCR/Surya.
"""

from typing import List

from codai.ocr.base import OcrEngine, OcrPage, OcrLine, OcrError


class DoctrEngine(OcrEngine):
    name = "doctr"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._model = None

    def load(self) -> None:
        try:
            from doctr.models import ocr_predictor
        except Exception as e:
            raise OcrError(
                "docTR is not installed. Install it out of band: "
                f"`pip install python-doctr[torch]`. Import error: {e}",
                status=503,
            )
        try:
            self._model = ocr_predictor(
                det_arch=self.cfg.doctr_det_arch or "db_resnet50",
                reco_arch=self.cfg.doctr_reco_arch or "crnn_vgg16_bn",
                pretrained=True,
            )
        except Exception as e:
            raise OcrError(f"failed to initialise docTR predictor: {e}", status=500)

        # Device placement: default is GPU when torch sees CUDA. Force CPU if requested.
        if not self.cfg.doctr_use_gpu:
            try:
                self._model = self._model.cpu()
            except Exception:
                pass
        else:
            try:
                import torch
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
            except Exception:
                pass

    def cleanup(self) -> None:
        self._model = None
        self._loaded = False
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def vram_gb(self) -> float:
        return 0.7   # measured ~0.66 GB peak per instance

    def recognize_image(self, image) -> OcrPage:
        import numpy as np

        arr = np.array(image.convert("RGB"))
        h, w = arr.shape[0], arr.shape[1]
        page = OcrPage(index=0, width=int(w), height=int(h))

        try:
            result = self._model([arr])
            export = result.export()
        except Exception as e:
            raise OcrError(f"docTR recognition failed: {e}", status=500)

        lines: List[OcrLine] = []
        pages = export.get("pages") or []
        if pages:
            pg = pages[0]
            dims = pg.get("dimensions") or (h, w)   # (height, width)
            ph, pw = float(dims[0]), float(dims[1])
            for block in pg.get("blocks", []):
                for line in block.get("lines", []):
                    words = line.get("words", [])
                    text = " ".join(str(wd.get("value", "")) for wd in words).strip()
                    if not text:
                        continue
                    confs = [float(wd.get("confidence", 0.0)) for wd in words] or [0.0]
                    geom = line.get("geometry") or ((0, 0), (0, 0))
                    (x0, y0), (x1, y1) = geom
                    bbox = [x0 * pw, y0 * ph, x1 * pw, y1 * ph]
                    lines.append(OcrLine(text=text, bbox=bbox, conf=sum(confs) / len(confs)))

        page.lines = lines
        page.text = "\n".join(l.text for l in lines)
        return page
