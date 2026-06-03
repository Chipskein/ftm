from abc import ABC, abstractmethod
import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class EngineOCR(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, img_path: str, output_dir: str) -> list:
        pass

    def loadImage(self, image_path: str) -> cv2.typing.MatLike:
        return cv2.imread(image_path)

    def detect_spread_split(self, img: cv2.typing.MatLike) -> int | None:
        h, w = img.shape[:2]
        if w <= h: return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        center, band_half = w // 2, int(w * 0.075) # 0.15 total band
        band = gray[:, center - band_half : center + band_half]
        
        white_pct = (band >= 240).astype(np.float32).mean(axis=0)
        gutter_cols = np.where(white_pct >= 0.85)[0]
        
        if len(gutter_cols) == 0: return None
        return (center - band_half) + int(gutter_cols.mean())
    
    def crop_bubble(
        self,
        img: cv2.typing.MatLike,
        x: int, y: int, w: int, h: int,
        crops_dir: str,
        base_name: str,
        bubble_id: int,
    ) -> str | None:
        img_h, img_w = img.shape[:2]
        y0, y1 = max(0, y), min(img_h, y + h)
        x0, x1 = max(0, x), min(img_w, x + w)
        if y1 <= y0 or x1 <= x0:
            return None
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        crop_path = os.path.join(crops_dir, f"{base_name}_bubble_{bubble_id:03d}.png")
        cv2.imwrite(crop_path, crop)
        return crop_path
