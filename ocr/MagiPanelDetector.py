from PIL import Image
import cv2
import numpy as np

from ocr.PanelDetector import PanelDetector

import logging
import os
import warnings
import torch
from transformers import AutoModel

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

def load_magi() -> AutoModel:
    logger.info("loading Magi v2 model...")
    model = AutoModel.from_pretrained(
        "ragavsachdeva/magiv2", trust_remote_code=True
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Magi loaded on GPU")
    else:
        logger.info("Magi loaded on CPU")
    return model

class MagiPanelDetector(PanelDetector):
    """
    Panel detector backed by the Magi model.

    Usage:
        from transformers import AutoModel
        magi = AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True)
        detector = MagiPanelDetector(magi)
        panels = detector._find_panel_dividers(image)
    """

    def __init__(self, magi_model: AutoModel):
        """
        Args:
            magi_model:     Loaded Magi AutoModel instance.
        """
        import torch
        from PIL import Image as PILImage

        self._magi = magi_model
        self._torch = torch
        self._PILImage = PILImage

    def _find_panel_dividers(self, img: cv2.typing.MatLike) -> list[tuple]:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_np = np.array(Image.fromarray(img_rgb).convert("L").convert("RGB"))

        character_bank = {"images": [], "names": []}
        with torch.no_grad():
            results = self._magi.do_chapter_wide_prediction(
                [img_np], character_bank, use_tqdm=False, do_ocr=False
            )

        panels = results[0]["panels"]
        panel_rects = [
            (int(p[0]), int(p[1]), int(p[2] - p[0]), int(p[3] - p[1]))
            for p in panels
        ]
        logger.debug("Magi raw panels: %d → %s", len(panel_rects), panel_rects)

        def contains(outer, inner):
            ox, oy, ow, oh = outer
            ix, iy, iw, ih = inner
            return ox <= ix and oy <= iy and ox + ow >= ix + iw and oy + oh >= iy + ih

        panel_rects = [
            p for i, p in enumerate(panel_rects)
            if not any(self._contains(p, other) for j, other in enumerate(panel_rects) if i != j)
        ]
        logger.debug("Magi panels after containment filter: %d → %s",
                     len(panel_rects), panel_rects)

        return panel_rects