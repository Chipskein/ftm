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

    def preProcessImage(self, image: cv2.typing.MatLike) -> dict:
        """Common enhancement pipeline: upscale -> sharpen -> invert."""
        img_up = self._upscale(image, scale=2)
        img_enhanced = self._sharpen(self._enhance_contrast_clahe(img_up))
        img_inv = cv2.bitwise_not(img_enhanced)
        return {"up": img_up, "enhanced": img_enhanced, "inv": img_inv}

    # --- Shared Quality Assessment ---

    def assess_and_fix_quality(self, img: cv2.typing.MatLike) -> cv2.typing.MatLike:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(gray.std())
        resolution = min(gray.shape[:2])

        logger.info(f"Quality: Sharp={sharpness:.1f}, Contrast={contrast:.1f}, Res={resolution}px")

        if resolution < 800:
            logger.warning("Low resolution detected.")
        if contrast < 35:
            logger.info("Applying CLAHE contrast enhancement.")
            img = self._enhance_contrast_clahe(img)
        if sharpness < 120:
            logger.info("Applying unsharp mask.")
            img = self._sharpen(img)
        
        return img

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

    @staticmethod
    def _upscale(img: cv2.typing.MatLike, scale: int) -> cv2.typing.MatLike:
        h, w = img.shape[:2]
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _enhance_contrast_clahe(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _sharpen(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)
    
    @staticmethod
    def _iou(bbox_a: tuple, bbox_b: tuple) -> float:
        """IoU between two (x, y, w, h) boxes."""
        ax, ay, aw, ah = bbox_a
        bx, by, bw, bh = bbox_b

        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax + aw, bx + bw)
        iy2 = min(ay + ah, by + bh)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0

        union = aw * ah + bw * bh - inter
        return inter / union