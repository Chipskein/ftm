import logging
import os
import time

import cv2
import torch
from ultralytics import YOLO

from .EngineOCR import EngineOCR
from ...dto.BubbleZone import BubbleZone
import numpy as np

logger = logging.getLogger(__name__)

CLS_TEXT = 0
CLS_ONOMATOPEIA = 1
CLS_COLORS = {
    CLS_TEXT:        (0, 200, 0),
    CLS_ONOMATOPEIA: (0, 0, 255)
}

class YOLOTextDetector(EngineOCR):
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        imgsz: int = 1024,
        debug: bool = False,
        use_cpu: bool = False,
    ):
        super().__init__("YOLOTextDetector")
        self.conf_threshold = conf_threshold
        self.imgsz          = imgsz
        self.debug          = debug
        self.use_cpu        = use_cpu

        device = "cuda" if torch.cuda.is_available() and not use_cpu else "cpu"
        self.model = YOLO(model_path)
        self.model.to(device)

        if debug:
            logger.setLevel(logging.DEBUG)

        logger.debug("YOLOTextDetector initialised (device=%s, model=%s)", device, model_path)
        logger.debug("confidence threshold: %.2f  imgsz: %d", conf_threshold, imgsz)

    def run(self, img_path: str, output_dir: str) -> list[BubbleZone]:
        if not os.path.exists(img_path):
            logger.error("image not found: %s", img_path)
            return []

        img = self.loadImage(img_path)
        h, w = img.shape[:2]
        logger.info("processing image %s (%dx%d)", os.path.basename(img_path), w, h)

        split_x = self.detect_spread_split(img)
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

        start_detection_time = time.perf_counter()
        raw_detections = self._detect(img)
        if self.debug:
            debug_img = self._draw_detections(img.copy(), raw_detections)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_raw_debug.png"), debug_img)
        detection_time = time.perf_counter() - start_detection_time
        
        text_detections = [d for d in raw_detections if d["class_id"] == CLS_TEXT]
        num_raw_detection = len(text_detections)
        
        start_group = time.perf_counter()
        detections = self._merge_overlapping(text_detections)
        group_time = time.perf_counter() - start_group
        num_kept_detection = len(detections)

        if self.debug:
            debug_img = self._draw_detections(img.copy(), detections)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_debug.png"), debug_img)
        
        results: list[BubbleZone] = []
        for _ , det in enumerate(detections):
            bx, by, bw, bh = det["bbox"]
            bubble_id = id_offset + len(results) + 1
            crop_path = self.crop_bubble(
                img, 
                bx, 
                by, 
                bw, 
                bh, 
                crops_dir, 
                base_name, 
                bubble_id
            )
            logger.debug("saved crop: %s", crop_path)
            bubble = BubbleZone(
                id=bubble_id,
                x=bx + x_offset,
                y=by,
                w=bw,
                h=bh,
                crop=crop_path,
                area=bw * bh,
                detection_method="yolo",
                detection_rects_total= num_raw_detection,
                detection_rects_kept= num_kept_detection,
                detection_time_s= detection_time,
                grouping_time_s= group_time
            )
            results.append(bubble)

        logger.info("%s — %d bubble(s) detected, crops saved to %s", base_name, len(results), crops_dir)
        return results

    def _detect(self, img: cv2.typing.MatLike) -> list[dict]:
        yolo_results = list(self.model.predict(source=img, conf=self.conf_threshold, imgsz=self.imgsz, verbose=True))
        if not yolo_results:
            return []

        result       = yolo_results[0]
        detections: list[dict] = []

        for _ , box in enumerate(result.boxes):
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

    def _merge_overlapping(self, detections: list[dict], iou_threshold: float = 0.1) -> list[dict]:
        if not detections:
            return []

        merged = True
        current = detections[:]

        while merged:
            merged = False
            result = []
            used = [False] * len(current)

            for i in range(len(current)):
                if used[i]:
                    continue

                group = [current[i]]
                used[i] = True

                for j in range(i + 1, len(current)):
                    if used[j]:
                        continue

                    ax1, ay1, aw, ah = current[i]["bbox"]
                    bx1, by1, bw, bh = current[j]["bbox"]
                    ix1 = max(ax1, bx1)
                    iy1 = max(ay1, by1)
                    ix2 = min(ax1 + aw, bx1 + bw)
                    iy2 = min(ay1 + ah, by1 + bh)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    union = aw * ah + bw * bh - inter
                    iou = inter / union if union > 0 else 0.0

                    if iou > iou_threshold:
                        group.append(current[j])
                        used[j] = True
                        merged = True

                x1 = min(d["bbox"][0] for d in group)
                y1 = min(d["bbox"][1] for d in group)
                x2 = max(d["bbox"][0] + d["bbox"][2] for d in group)
                y2 = max(d["bbox"][1] + d["bbox"][3] for d in group)
                best = max(group, key=lambda d: d["confidence"])
                result.append({
                    "bbox":       (x1, y1, x2 - x1, y2 - y1),
                    "class_id":   best["class_id"],
                    "confidence": best["confidence"],
                })

            current = result

        return current

    def _draw_detections(
        self, img: cv2.typing.MatLike, detections: list[dict]
    ) -> cv2.typing.MatLike:
        for det in detections:
            bx, by, bw, bh = det["bbox"]
            color = CLS_COLORS.get(det["class_id"], (200, 200, 200))
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, 2)
            class_id = det["class_id"]
            class_name = self.model.names[class_id]
            label = f"{class_name} {det['confidence']:.2f}"
            cv2.putText(img, label, (bx, max(by - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return img