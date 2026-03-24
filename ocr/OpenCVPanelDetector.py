import cv2
import numpy as np
from ocr.PanelDetector import PanelDetector

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

    def _find_panel_dividers(self, img: np.ndarray) -> list[tuple]:
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