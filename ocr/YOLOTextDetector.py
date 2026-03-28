import logging
import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .EngineOCR import EngineOCR
from .types.BubbleZone import BubbleZone
from ocr.PanelDetector import PanelDetector
from utils.resource import ResourceMonitor

logger = logging.getLogger(__name__)

CLS_TEXT = 0


class YOLOTextDetector(EngineOCR):
    """
    OCR engine that uses a YOLOv8s-seg custom model to detect and locate
    text / onomatopoeia regions in manga pages.

    YOLO is solely responsible for detection — no secondary OCR engine is used.

    Expected model classes
    ----------------------
    0: text
    1: onomatopoeia
    """

    def __init__(
        self,
        model_path: str,
        panel_detector: PanelDetector,
        conf_threshold: float = 0.25,
        imgsz: int = 1024,
        debug: bool = False,
        monitor: ResourceMonitor | None = None,
        use_cpu: bool = False,
    ):
        super().__init__("YOLOTextDetector")
        self.panel_detector = panel_detector
        self.conf_threshold = conf_threshold
        self.imgsz          = imgsz
        self.monitor        = monitor
        self.debug          = debug
        self.use_cpu        = use_cpu

        device = "cuda" if torch.cuda.is_available() and not use_cpu else "cpu"
        self.model = YOLO(model_path)
        self.model.to(device)

        if debug:
            logger.setLevel(logging.DEBUG)

        logger.debug("YOLOTextDetector initialised (device=%s, model=%s)", device, model_path)
        logger.debug("confidence threshold: %.2f  imgsz: %d", conf_threshold, imgsz)

    # ------------------------------------------------------------------ #
    # Abstract method implementations                                      #
    # ------------------------------------------------------------------ #

    def loadImage(self, image_path: str) -> cv2.typing.MatLike:
        return cv2.imread(image_path)

    def preProcessImage(self, image: cv2.typing.MatLike) -> dict:
        """Returns upscaled, contrast-enhanced and inverted variants."""
        img_up       = self._upscale(image, scale=2)
        img_enhanced = self._sharpen(self._enhance_contrast_clahe(img_up))
        img_inv      = cv2.bitwise_not(img_enhanced)
        return {"up": img_up, "enhanced": img_enhanced, "inv": img_inv}

    def run(self, img_path: str, output_dir: str) -> list[BubbleZone]:
        if not os.path.exists(img_path):
            logger.error("image not found: %s", img_path)
            return []

        img = self.loadImage(img_path)
        h, w = img.shape[:2]
        logger.info("processing image %s (%dx%d)", os.path.basename(img_path), w, h)

        quality = self._assess_quality(img)
        logger.info(
            "quality — sharpness=%.1f  contrast=%.1f  resolution=%dpx",
            quality["sharpness"], quality["contrast"], quality["resolution"],
        )
        if quality["low_res"]:
            logger.warning("low resolution (%dpx short side) — detection quality may suffer",
                           quality["resolution"])
        if quality["needs_contrast"]:
            logger.info("contrast low (%.1f) — applying CLAHE", quality["contrast"])
            img = self._enhance_contrast_clahe(img)
        if quality["needs_sharpen"]:
            logger.info("sharpness low (%.1f) — applying unsharp mask", quality["sharpness"])
            img = self._sharpen(img)

        split_x = self._detect_spread_split(img)
        if split_x is not None:
            logger.debug("spread detected — split at x=%d", split_x)
            left  = img[:, :split_x]
            right = img[:, split_x:]
            results_left  = self._run_single(left,  img_path, output_dir, suffix="_L")
            results_right = self._run_single(
                right, img_path, output_dir,
                suffix="_R", x_offset=split_x,
                id_offset=len(results_left),
            )
            results = results_left + results_right
            logger.info("spread — left=%d  right=%d  total=%d bubbles",
                        len(results_left), len(results_right), len(results))
            return results

        return self._run_single(img, img_path, output_dir)

    # ------------------------------------------------------------------ #
    # Core pipeline                                                        #
    # ------------------------------------------------------------------ #

    def _run_single(
        self,
        img: cv2.typing.MatLike,
        img_path: str,
        output_dir: str,
        suffix: str = "",
        x_offset: int = 0,
        id_offset: int = 0,
    ) -> list[BubbleZone]:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(img_path))[0] + suffix
        crops_dir = os.path.join(output_dir, "crops_" + base_name)
        os.makedirs(crops_dir, exist_ok=True)

        detections = [d for d in self._detect(img) if d["class_id"] == CLS_TEXT]
        logger.debug("YOLO detections (text only): %d", len(detections))

        if self.debug:
            debug_img = self._draw_detections(img.copy(), detections)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_debug.png"), debug_img)

        img_h, img_w = img.shape[:2]
        results: list[BubbleZone] = []

        for det_idx, det in enumerate(detections):
            bx, by, bw, bh = det["bbox"]
            confidence      = det["confidence"]

            # ── Save crop ────────────────────────────────────────────── #
            x1 = max(bx, 0);          y1 = max(by, 0)
            x2 = min(bx + bw, img_w); y2 = min(by + bh, img_h)
            crop = img[y1:y2, x1:x2].copy()
            crop_path = os.path.join(crops_dir, f"{det_idx:04d}.png")
            cv2.imwrite(crop_path, crop)
            logger.debug("saved crop: %s", crop_path)

            bubble = BubbleZone(
                id=id_offset + len(results),
                x=bx + x_offset,
                y=by,
                w=bw,
                h=bh,
                crop=crop_path,
            )
            results.append(bubble)

        logger.info("%s — %d bubble(s) detected, crops saved to %s", base_name, len(results), crops_dir)
        return results

    # ------------------------------------------------------------------ #
    # YOLO inference                                                       #
    # ------------------------------------------------------------------ #

    def _detect(self, img: cv2.typing.MatLike) -> list[dict]:
        """
        Run YOLOv8s-seg on *img* and return a list of detection dicts:
            {
                "bbox":       (x, y, w, h),          # int pixel coords
                "class_id":   int,
                "confidence": float,
                "mask":       np.ndarray | None,      # H×W uint8 binary
            }
        """
        yolo_results = list(self.model.predict(source=img, conf=0.25, imgsz=1024, verbose=True))
        if not yolo_results:
            return []

        result       = yolo_results[0]
        img_h, img_w = img.shape[:2]
        detections: list[dict] = []

        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bw, bh = x2 - x1, y2 - y1
            if bw < 5 or bh < 5:
                continue

            detections.append({
                "bbox":       (x1, y1, bw, bh),
                "class_id":   int(box.cls[0]),
                "confidence": float(box.conf[0]),
            })

        logger.debug("YOLO — detections=%d", len(detections))
        return detections

    # ------------------------------------------------------------------ #
    # Debug visualisation                                                  #
    # ------------------------------------------------------------------ #

    def _draw_detections(
        self, img: cv2.typing.MatLike, detections: list[dict]
    ) -> cv2.typing.MatLike:
        for det in detections:
            bx, by, bw, bh = det["bbox"]
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 200, 0), 2)
            label = f"text {det['confidence']:.2f}"
            cv2.putText(img, label, (bx, max(by - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
        return img

    # ------------------------------------------------------------------ #
    # Spread detection                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_spread_split(
        img: cv2.typing.MatLike,
        search_band: float = 0.15,
        white_thresh: int = 240,
        min_white_col_pct: float = 0.85,
    ) -> int | None:
        h, w = img.shape[:2]
        if w <= h:
            return None

        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        center = w // 2
        half   = int(w * search_band / 2)
        band   = gray[:, center - half : center + half]

        col_white_pct = (band >= white_thresh).astype(np.float32).mean(axis=0)
        gutter_cols   = np.where(col_white_pct >= min_white_col_pct)[0]
        if len(gutter_cols) == 0:
            return None

        return (center - half) + int(gutter_cols.mean())

    # ------------------------------------------------------------------ #
    # Static image-processing utilities                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _assess_quality(img: cv2.typing.MatLike) -> dict:
        gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast   = float(gray.std())
        resolution = min(gray.shape)
        return {
            "sharpness":      round(sharpness, 1),
            "contrast":       round(contrast,  1),
            "resolution":     resolution,
            "needs_sharpen":  sharpness < 120,
            "needs_contrast": contrast  <  35,
            "low_res":        resolution < 800,
        }

    @staticmethod
    def _upscale(img: cv2.typing.MatLike, scale: int = 2) -> cv2.typing.MatLike:
        h, w = img.shape[:2]
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _enhance_contrast_clahe(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
        lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l       = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _sharpen(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)