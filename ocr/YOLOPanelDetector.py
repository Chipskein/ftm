from pathlib import Path
import numpy as np
from ocr.PanelDetector import PanelDetector

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

    def _find_panel_dividers(self, img: np.ndarray) -> list[tuple]:
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