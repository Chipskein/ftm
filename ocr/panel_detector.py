from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np


class PanelDetector(ABC):
    """
    Abstract interface for manga panel detectors.

    All implementations must return panels as list[tuple[x, y, w, h]],
    which is the format expected by EazyOCR._find_panel_dividers.
    """

    @abstractmethod
    def _find_panel_dividers(self, img: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detect panels in a BGR numpy image.

        Returns:
            List of (x, y, w, h) tuples, one per detected panel.
        """
        ...

    def detect_from_file(self, path: str | Path) -> list[tuple[int, int, int, int]]:
        """Convenience method to detect panels from a file path."""
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return self._find_panel_dividers(img)


class YOLOPanelDetector(PanelDetector):
    """
    Panel detector backed by a trained YOLOv8 model.

    Usage:
        detector = YOLOPanelDetector("runs/detect/train/weights/best.pt")
        panels = detector._find_panel_dividers(image)
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 1024,
    ):
        from ultralytics import YOLO

        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self._model = YOLO(str(model_path))

    def _find_panel_dividers(self, img: np.ndarray) -> list[tuple[int, int, int, int]]:
        results = self._model(
            img,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        panels = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            panels.append((x1, y1, x2 - x1, y2 - y1))

        panels.sort(key=lambda p: (p[1], p[0]))
        return panels


class MagiPanelDetector(PanelDetector):
    """
    Panel detector backed by the Magi model.

    Usage:
        from transformers import AutoModel
        magi = AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True)
        detector = MagiPanelDetector(magi)
        panels = detector._find_panel_dividers(image)
    """

    def __init__(self, magi_model, max_area_ratio: float = 0.90):
        """
        Args:
            magi_model:     Loaded Magi AutoModel instance.
            max_area_ratio: Panels covering more than this fraction of the image
                            are treated as full-page false positives and removed.
        """
        import torch
        from PIL import Image as PILImage

        self._magi = magi_model
        self._max_area_ratio = max_area_ratio
        self._torch = torch
        self._PILImage = PILImage

    def _find_panel_dividers(self, img: np.ndarray) -> list[tuple[int, int, int, int]]:
        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_np = np.array(self._PILImage.fromarray(img_rgb).convert("L").convert("RGB"))

        character_bank = {"images": [], "names": []}
        with self._torch.no_grad():
            results = self._magi.do_chapter_wide_prediction(
                [img_np], character_bank, use_tqdm=False, do_ocr=False
            )

        raw = results[0]["panels"]
        panels = [
            (int(p[0]), int(p[1]), int(p[2] - p[0]), int(p[3] - p[1]))
            for p in raw
        ]

        # Remove full-page false positives
        panels = [
            p for p in panels
            if (p[2] * p[3]) / img_area < self._max_area_ratio
        ]

        # Remove panels contained by another
        panels = [
            p for i, p in enumerate(panels)
            if not any(
                self._contains(other, p)
                for j, other in enumerate(panels) if i != j
            )
        ]

        panels.sort(key=lambda p: (p[1], p[0]))
        return panels

    @staticmethod
    def _contains(outer: tuple, inner: tuple, margin: int = 10) -> bool:
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner
        return (
            ox - margin <= ix
            and oy - margin <= iy
            and ox + ow + margin >= ix + iw
            and oy + oh + margin >= iy + ih
        )


class OpenCVPanelDetector(PanelDetector):
    """
    Classical contour-based panel detector (no model needed).
    Works well for manga with clear black borders.

    Usage:
        detector = OpenCVPanelDetector()
        panels = detector._find_panel_dividers(image)
    """

    def __init__(
        self,
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.95,
        binary_threshold: int = 200,
        containment_margin: int = 10,
        dilate_iterations: int = 1,
    ):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.binary_threshold = binary_threshold
        self.containment_margin = containment_margin
        self.dilate_iterations = dilate_iterations

    def _find_panel_dividers(self, img: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h_img, w_img = gray.shape[:2]
        total_area = h_img * w_img

        _, binary = cv2.threshold(
            gray, self.binary_threshold, 255, cv2.THRESH_BINARY_INV
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=self.dilate_iterations)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < total_area * self.min_area_ratio:
                continue
            if area > total_area * self.max_area_ratio:
                continue
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)
            candidates.append((x, y, w, h))

        # Remove nested panels
        panels = [
            p for i, p in enumerate(candidates)
            if not any(
                self._contains(candidates[j], p, self.containment_margin)
                for j in range(len(candidates)) if j != i
            )
        ]

        panels.sort(key=lambda p: (p[1], p[0]))
        return panels

    @staticmethod
    def _contains(outer: tuple, inner: tuple, margin: int = 10) -> bool:
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner
        return (
            ox - margin <= ix
            and oy - margin <= iy
            and ox + ow + margin >= ix + iw
            and oy + oh + margin >= iy + ih
        )
