import logging
import os
import tempfile

from .panel.PanelDetector import PanelDetector
from profiler.ResourceMonitor import ResourceMonitor

# Disable oneDNN/MKL-DNN before ANY paddle import.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_DISABLE_ONEDNN"] = "1"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_new_executor_use_interpretermcore"] = "0"

import cv2
import numpy as np
import torch

try:
    import paddle
    paddle.set_flags({"FLAGS_use_mkldnn": False})
except Exception:
    pass

from paddleocr import PaddleOCR, draw_ocr
from PIL import Image as PILImage
from .EngineOCR import EngineOCR
from dto.BubbleZone import BubbleZone

logger = logging.getLogger(__name__)

# --- Tunable constants ---

# Minimum bubble area as fraction of total image area (filters panel borders, tiny blobs)
_BUBBLE_MIN_AREA_FRAC     = 0.002
# Maximum bubble area as fraction of total image area (filters full-page regions)
_BUBBLE_MAX_AREA_FRAC     = 0.25
# Padding around each bubble crop before OCR (pixels, original scale)
_BUBBLE_PAD               = 6
# OCR confidence threshold
_CONF_THRESHOLD           = 0.45
# IoU threshold for deduplication across the two OCR variants
_IOU_MERGE_THRESHOLD      = 0.3
# Minimum solidity (contour_area / hull_area) — filters jagged non-bubble shapes
_BUBBLE_MIN_SOLIDITY      = 0.55
# Furigana filter: discard OCR boxes shorter than this fraction of bubble height
_FURIGANA_MIN_HEIGHT_FRAC = 0.12


class PaddleOCREngine(EngineOCR):
    def __init__(
        self,
        panel_detector: PanelDetector,
        debug: bool = False,
        monitor: ResourceMonitor | None = None,
    ):
        super().__init__("PaddleOCR")
        self.panel_detector = panel_detector
        self.debug = debug
        self.monitor = monitor

        if debug:
            logger.setLevel(logging.DEBUG)

        logger.info("loading PaddleOCR (lang=japan)...")
        self._ocr = self._init_paddle()
        logger.info("PaddleOCR ready  gpu=%s", torch.cuda.is_available())

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

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
            logger.info(
                "spread — left=%d  right=%d  total=%d bubbles",
                len(results_left), len(results_right), len(results),
            )
            return results

        return self._run_single(img, img_path, output_dir)

    # ------------------------------------------------------------------
    # Core single-image pipeline
    # ------------------------------------------------------------------

    def _run_single(
        self,
        img: cv2.typing.MatLike,
        img_path: str,
        output_dir: str,
        suffix: str = "",
        panels: list[tuple] | None = None,
        x_offset: int = 0,
        id_offset: int = 0,
    ) -> list[BubbleZone]:
        img_h, img_w = img.shape[:2]
        base_name = os.path.splitext(os.path.basename(img_path))[0] + suffix

        # Save preprocessed variants (used by debug + future passes)
        variants = self.preProcessImage(img)
        enhanced_path = f"{output_dir}/{base_name}_enhanced.png"
        inv_path      = f"{output_dir}/{base_name}_inv.png"
        cv2.imwrite(enhanced_path, variants["enhanced"])
        cv2.imwrite(inv_path,      variants["inv"])
        if self.debug:
            cv2.imwrite(f"{output_dir}/{base_name}_up.png", variants["up"])
            logger.debug("preprocessed variants saved to %s", output_dir)

        # ── Step 1: locate speech bubble regions ──────────────────────
        bubbles = self._detect_bubbles(img)
        logger.info("%s — %d speech bubbles found", base_name, len(bubbles))

        if self.debug:
            self._save_bubble_debug(img, bubbles, output_dir, base_name)

        # ── Step 2: OCR inside each bubble ────────────────────────────
        crops_dir = os.path.join(output_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

        results: list[BubbleZone]                                      = []
        all_deduped: list[tuple[tuple[int,int,int,int], str, float]]   = []

        for i, (bx, by, bw, bh) in enumerate(bubbles):
            bubble_id   = id_offset + i
            bubble_h_px = bh

            # Pad & clamp crop to image bounds
            x0 = max(0, bx - _BUBBLE_PAD)
            y0 = max(0, by - _BUBBLE_PAD)
            x1 = min(img_w, bx + bw + _BUBBLE_PAD)
            y1 = min(img_h, by + bh + _BUBBLE_PAD)

            bubble_crop = img[y0:y1, x0:x1]

            # OCR on normal + inverted crop to catch both polarities
            raw  = self._ocr_on_array(bubble_crop)
            raw += self._ocr_on_array(cv2.bitwise_not(bubble_crop))

            # Filter: confidence + furigana height
            filtered = []
            for bbox, text, conf in raw:
                if conf < _CONF_THRESHOLD:
                    continue
                _, _, _, bh_ocr = bbox
                if bh_ocr < _FURIGANA_MIN_HEIGHT_FRAC * bubble_h_px:
                    logger.debug(
                        "furigana filtered: %r  h=%d  bubble_h=%d",
                        text, bh_ocr, bubble_h_px,
                    )
                    continue
                filtered.append((bbox, text, conf))

            deduped = self._deduplicate(filtered)

            if not deduped:
                logger.debug("bubble %03d — no text after filtering", bubble_id)
                continue

            # Merge text lines → single string (vertical manga reading order)
            merged_text = self._merge_lines(deduped)

            # Translate OCR coords (relative to crop) → original image space
            abs_deduped = [
                ((cx + x0, cy + y0, cw, ch), text, conf)
                for (cx, cy, cw, ch), text, conf in deduped
            ]
            all_deduped.extend(abs_deduped)

            abs_x = bx + x_offset

            crop_path = self.crop_bubble(
                img, abs_x, by, bw, bh,
                crops_dir=crops_dir,
                base_name=base_name,
                bubble_id=bubble_id,
            )

            zone: BubbleZone = {
                "id":      bubble_id,
                "x":       abs_x,
                "y":       by,
                "w":       bw,
                "h":       bh,
                "jp_text": merged_text,
            }
            if crop_path:
                zone["crop"] = crop_path

            results.append(zone)
            logger.debug(
                "bubble %03d  pos=(%d,%d)  size=(%dx%d)  text=%r",
                bubble_id, abs_x, by, bw, bh, merged_text,
            )

        logger.info("%s — %d bubbles with text", base_name, len(results))

        if self.debug and all_deduped:
            self._save_debug_visualization(img, all_deduped, output_dir, base_name)

        return results

    # ------------------------------------------------------------------
    # Speech bubble detection
    # ------------------------------------------------------------------

    def _detect_bubbles(self, img: cv2.typing.MatLike) -> list[tuple[int,int,int,int]]:
        """
        Detect speech bubble regions via white-blob morphology.

        Pipeline:
          1. Grayscale → binary threshold (white regions)
          2. Morphological close to seal bubble outlines
          3. Flood-fill hole-filling so bubble interiors are solid white
          4. Filter contours by: area, solidity, aspect ratio
          5. Return (x, y, w, h) sorted in manga reading order (right→left, top→bottom)
        """
        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold: isolate white speech bubbles
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Close small gaps in bubble outlines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Fill holes so bubble interiors are solid
        filled = self._fill_holes(closed)

        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bubbles: list[tuple[int,int,int,int]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < _BUBBLE_MIN_AREA_FRAC * img_area:
                continue
            if area > _BUBBLE_MAX_AREA_FRAC * img_area:
                continue

            # Solidity: reject screentones, jagged panel edges
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            if hull_area == 0:
                continue
            if (area / hull_area) < _BUBBLE_MIN_SOLIDITY:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Reject extreme aspect ratios (horizontal rules, panel borders)
            aspect = w / h if h > 0 else 0
            if aspect > 6 or aspect < 0.15:
                continue

            bubbles.append((x, y, w, h))

        # Manga reading order: right-to-left columns, top-to-bottom within column
        bubbles.sort(key=lambda b: (-b[0], b[1]))
        return bubbles

    @staticmethod
    def _fill_holes(binary: np.ndarray) -> np.ndarray:
        """Flood-fill from the image border to find background, invert to fill blobs."""
        h, w  = binary.shape
        flood = binary.copy()
        mask  = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, mask, (0, 0), 255)
        return binary | cv2.bitwise_not(flood)

    # ------------------------------------------------------------------
    # OCR helpers
    # ------------------------------------------------------------------

    def _init_paddle(self) -> PaddleOCR:
        return PaddleOCR(
            use_angle_cls=True,
            lang="japan",
            use_gpu=torch.cuda.is_available(),
            show_log=False,
            # Handle small bubble crops without downscaling
            det_limit_side_len=960,
            det_limit_type="max",
        )

    def _ocr_on_array(
        self, img_arr: np.ndarray
    ) -> list[tuple[tuple[int,int,int,int], str, float]]:
        """Write array to a temp file, run OCR, delete temp file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cv2.imwrite(tmp_path, img_arr)
            return self._ocr_on_file(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _ocr_on_file(
        self, image_path: str
    ) -> list[tuple[tuple[int,int,int,int], str, float]]:
        result = self._ocr.ocr(image_path, cls=True)
        detections = []
        if not result or result[0] is None:
            return detections
        for line in result[0]:
            quad, (text, conf) = line
            detections.append((self._quad_to_xywh(quad), text, float(conf)))
        return detections

    @staticmethod
    def _quad_to_xywh(quad: list) -> tuple[int,int,int,int]:
        xs = [pt[0] for pt in quad]
        ys = [pt[1] for pt in quad]
        x, y = int(min(xs)), int(min(ys))
        return x, y, int(max(xs)) - x, int(max(ys)) - y

    @staticmethod
    def _merge_lines(
        detections: list[tuple[tuple[int,int,int,int], str, float]]
    ) -> str:
        """
        Merge OCR lines into one string using vertical manga reading order:
        right-to-left columns (x descending), top-to-bottom within each column.
        """
        sorted_lines = sorted(detections, key=lambda d: (-d[0][0], d[0][1]))
        return "".join(text for _, text, _ in sorted_lines)

    def _deduplicate(
        self,
        detections: list[tuple[tuple[int,int,int,int], str, float]],
    ) -> list[tuple[tuple[int,int,int,int], str, float]]:
        sorted_dets = sorted(detections, key=lambda d: d[2], reverse=True)
        kept: list[tuple[tuple[int,int,int,int], str, float]] = []
        for det in sorted_dets:
            bbox, text, conf = det
            if not any(self._iou(bbox, k[0]) > _IOU_MERGE_THRESHOLD for k in kept):
                kept.append(det)
        return kept

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    def _save_bubble_debug(
        self,
        img: cv2.typing.MatLike,
        bubbles: list[tuple[int,int,int,int]],
        output_dir: str,
        base_name: str,
    ) -> None:
        """Save image with detected bubble bounding boxes drawn in blue."""
        vis = img.copy()
        for i, (x, y, w, h) in enumerate(bubbles):
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 100, 0), 2)
            cv2.putText(vis, f"#{i}", (x + 2, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
        out = os.path.join(output_dir, f"{base_name}_bubbles.jpg")
        cv2.imwrite(out, vis)
        logger.debug("bubble debug saved → %s", out)

    def _save_debug_visualization(
        self,
        img: cv2.typing.MatLike,
        detections: list[tuple[tuple[int,int,int,int], str, float]],
        output_dir: str,
        base_name: str,
    ) -> None:
        """Save image with final OCR detections drawn in green."""
        quads, texts, scores = [], [], []
        for (x, y, w, h), text, conf in detections:
            quads.append([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
            texts.append(text)
            scores.append(conf)

        out_path = os.path.join(output_dir, f"{base_name}_detections.jpg")
        try:
            pil_img   = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            annotated = draw_ocr(pil_img, quads, texts, scores)
            PILImage.fromarray(annotated).save(out_path)
            logger.debug("detection debug saved → %s", out_path)
        except Exception as exc:
            logger.warning("draw_ocr failed (%s), using cv2 fallback", exc)
            vis = img.copy()
            for (x, y, w, h), text, conf in detections:
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(vis, f"{text}({conf:.2f})", (x, max(y - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.imwrite(out_path, vis)
            logger.debug("cv2 fallback detection debug saved → %s", out_path)