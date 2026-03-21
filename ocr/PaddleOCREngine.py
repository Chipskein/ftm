import json
import logging
import os

# Disable oneDNN/MKL-DNN before ANY paddle import.
# In PaddlePaddle v3 the static executor loads oneDNN ops from compiled
# .pdmodel files — env vars alone are not enough, we also need paddle.set_flags.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_DISABLE_ONEDNN"] = "1"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_new_executor_use_interpretermcore"] = "0"

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import paddle
    paddle.set_flags({"FLAGS_use_mkldnn": False})
except Exception:
    pass

from paddleocr import PaddleOCR
from transformers import AutoModel

from .EngineOCR import EngineOCR
from .types.BubbleZone import BubbleZone

logger = logging.getLogger(__name__)


class PaddleOCREngine(EngineOCR):

    MIN_SCORE    = 0.3   # detection confidence threshold
    MIN_BUBBLE_W = 20
    MIN_BUBBLE_H = 20

    def __init__(
        self,
        magi_model: AutoModel,
        debug: bool = False,
    ):
        super().__init__("PaddleOCR")
        self.magi  = magi_model
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)

        logger.info("loading PaddleOCR (lang=japan)...")
        self._ocr = self._init_paddle()
        logger.info("PaddleOCR ready  gpu=%s", torch.cuda.is_available())

    # ------------------------------------------------------------------ #
    # Abstract method implementations                                      #
    # ------------------------------------------------------------------ #

    def loadImage(self, image_path: str) -> cv2.typing.MatLike:
        return cv2.imread(image_path)

    def preProcessImage(self, image: cv2.typing.MatLike) -> dict:
        img_up       = self._upscale(image, scale=2)
        img_enhanced = self._sharpen(self._enhance_contrast_clahe(img_up))
        img_inv      = cv2.bitwise_not(img_enhanced)
        return {"up": img_up, "enhanced": img_enhanced, "inv": img_inv}

    def run(self, img_path: str, output_dir: str) -> list[BubbleZone]:
        if not os.path.exists(img_path):
            logger.error("image not found: %s", img_path)
            return []

        img = self.loadImage(img_path)
        img_h, img_w = img.shape[:2]
        logger.info("processing %s (%dx%d)", os.path.basename(img_path), img_w, img_h)

        split_x = self._detect_spread_split(img)
        if split_x is not None:
            logger.debug("spread detected — split at x=%d", split_x)
            left  = img[:, :split_x]
            right = img[:, split_x:]
            results_left  = self._run_single(left,  img_path, output_dir, suffix="_L")
            results_right = self._run_single(right, img_path, output_dir, suffix="_R",
                                             x_offset=split_x,
                                             id_offset=len(results_left))
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
        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w

        base_name = os.path.splitext(os.path.basename(img_path))[0] + suffix
        variants  = self.preProcessImage(img)

        enhanced_path = f"{output_dir}/{base_name}_enhanced.png"
        inv_path      = f"{output_dir}/{base_name}_inv.png"
        cv2.imwrite(enhanced_path, variants["enhanced"])
        cv2.imwrite(inv_path, variants["inv"])
        if self.debug:
            cv2.imwrite(f"{output_dir}/{base_name}_up.png", variants["up"])
            logger.debug("intermediate images saved to %s", output_dir)

        logger.debug("running PaddleOCR on 3 variants...")
        rects = self._detect_text_rects(img_path, enhanced_path, inv_path)
        logger.debug("raw text rects: %d", len(rects))

        logger.debug("detecting panels with Magi...")
        panels = self._find_panel_dividers(img)
        logger.info("panels found: %d", len(panels))

        panel_groups = self._group_rects_by_panel(rects, panels)
        logger.debug("rects per panel: %s", {k: len(v) for k, v in panel_groups.items()})

        crops_dir = os.path.join(output_dir, "crops_" + base_name)
        os.makedirs(crops_dir, exist_ok=True)

        out     = img.copy()
        results: list[BubbleZone] = []

        for idx, group_rects in panel_groups.items():
            if idx >= 0:
                px, py, pw, ph = panels[idx]
                ratio = (pw * ph) / img_area
                max_h = ph * (0.45 if ratio > 0.15 else 0.35)
                max_w = pw * 0.70
                clip  = (px, py, px + pw, py + ph)
            else:
                ratio = 1.0
                max_h, max_w = None, None
                clip  = (0, 0, img_w, img_h)

            if idx == -1 and group_rects:
                logger.warning("panel -1: %d rect(s) outside all panels", len(group_rects))

            if ratio > 0.30:   gap_x, gap_y = 10, 10
            elif ratio > 0.10: gap_x, gap_y = 18, 18
            else:              gap_x, gap_y = 10, 12

            bubbles = self._group_nearby_rects(
                group_rects, gap_x=gap_x, gap_y=gap_y, max_w=max_w, max_h=max_h,
            )
            bubbles = self._merge_overlapping_rects(bubbles)
            bubbles = [b for b in bubbles if b[2] >= self.MIN_BUBBLE_W and b[3] >= self.MIN_BUBBLE_H]
            logger.debug("panel %d — ratio=%.2f  gap=(%d,%d)  → %d bubble(s)",
                         idx, ratio, gap_x, gap_y, len(bubbles))

            color = (0, 255, 0) if idx >= 0 else (0, 165, 255)
            for x, y, w, h in bubbles:
                x1 = max(x, clip[0]);      y1 = max(y, clip[1])
                x2 = min(x + w, clip[2]);  y2 = min(y + h, clip[3])
                if x2 - x1 < self.MIN_BUBBLE_W or y2 - y1 < self.MIN_BUBBLE_H:
                    logger.debug("bubble clipped to nothing in panel %d, skipping", idx)
                    continue
                x, y, w, h = x1, y1, x2 - x1, y2 - y1

                bubble_id = id_offset + len(results) + 1
                crop_path = self._crop_bubble(img, x, y, w, h, crops_dir, base_name, bubble_id)
                if crop_path is None:
                    logger.warning("empty crop skipped (bubble_id=%d)", bubble_id)
                    continue
                logger.debug("[%d] (%d,%d,%d,%d) panel=%d → %s",
                             bubble_id, x, y, w, h, idx, os.path.basename(crop_path))
                results.append(BubbleZone(
                    id=bubble_id, x=x + x_offset, y=y, w=w, h=h,
                    crop=crop_path, jp_text="", en_text="", translated_text="",
                ))
                cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
                label_y = min(y + 14, y + h - 4)
                cv2.putText(out, f"P{idx}#{bubble_id}", (x + 3, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        for i, (px, py, pw, ph) in enumerate(panels):
            cv2.rectangle(out, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
            cv2.putText(out, f"Panel {i+1}", (px + 5, py + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        result_path = f"{output_dir}/{base_name}_result.png"
        cv2.imwrite(result_path, out)

        json_path = f"{output_dir}/{base_name}_bubbles.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info("done — %d bubble(s) | result → %s | crops → %s/",
                    len(results), result_path, crops_dir)
        logger.debug("JSON → %s", json_path)
        return results

    # ------------------------------------------------------------------ #
    # Spread detection (double-page)                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_spread_split(
        img: cv2.typing.MatLike,
        search_band: float = 0.15,
        white_thresh: int = 240,
        min_white_col_pct: float = 0.85,
    ) -> int | None:
        """
        Detect a double-page spread and return the x coordinate of the gutter.
        Returns None if the image is portrait or no clear white gutter is found.
        Only images wider than they are tall are considered spreads.
        """
        h, w = img.shape[:2]
        if w <= h:
            return None

        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        center  = w // 2
        band_half = int(w * search_band / 2)
        x_start = center - band_half
        x_end   = center + band_half

        band           = gray[:, x_start:x_end]
        white_mask     = (band >= white_thresh).astype(np.float32)
        col_white_pct  = white_mask.mean(axis=0)

        gutter_cols = np.where(col_white_pct >= min_white_col_pct)[0]
        if len(gutter_cols) == 0:
            return None

        return x_start + int(gutter_cols.mean())

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _init_paddle(self) -> "PaddleOCR":
        """
        Initialise PaddleOCR, trying constructors in this order:
          1. v3 server models  (PP-OCRv4_server_* — best accuracy for Japanese)
          2. v3 mobile models  (PP-OCRv4_mobile_* — oneDNN-free fallback)
          3. v3 default        (lang= only, let PaddleOCR pick)
          4. v2 API            (use_angle_cls= signature)

        oneDNN is disabled globally via env vars at module load and via
        paddle.set_flags() above, so server models should work on most hardware.
        Mobile models are kept only as a last resort because they sacrifice
        detection accuracy — especially for dense vertical Japanese text.
        """
        # 1. v3 server models — highest accuracy, oneDNN disabled via env
        try:
            ocr = PaddleOCR(
                lang="japan",
                ocr_version="PP-OCRv4",
                det_model_name="PP-OCRv4_server_det",
                rec_model_name="PP-OCRv4_server_rec",
            )
            logger.debug("PaddleOCR initialised with v3 server models (best accuracy)")
            return ocr
        except TypeError:
            pass
        except Exception as e:
            logger.warning("v3 server init failed (%s) — falling back to mobile models", e)

        # 2. v3 mobile models — lighter, oneDNN-free, less accurate
        try:
            ocr = PaddleOCR(
                lang="japan",
                ocr_version="PP-OCRv4",
                det_model_name="PP-OCRv4_mobile_det",
                rec_model_name="PP-OCRv4_mobile_rec",
            )
            logger.debug("PaddleOCR initialised with v3 mobile models (fallback)")
            return ocr
        except TypeError:
            pass
        except Exception as e:
            logger.warning("v3 mobile init failed (%s) — trying v3 default", e)

        # 3. v3 default — let PaddleOCR pick models
        try:
            ocr = PaddleOCR(lang="japan")
            logger.debug("PaddleOCR initialised with v3 default constructor")
            return ocr
        except TypeError:
            pass

        # 4. v2 API: PaddleOCR(use_angle_cls=True, lang=..., use_gpu=...)
        try:
            ocr = PaddleOCR(
                use_angle_cls=True,
                lang="japan",
                show_log=False,
                use_gpu=torch.cuda.is_available(),
            )
            logger.debug("PaddleOCR initialised with v2 constructor")
            return ocr
        except TypeError as e:
            raise RuntimeError(
                f"Could not initialise PaddleOCR — unknown API version: {e}"
            )

    def _detect_text_rects(
        self, img_path: str, enhanced_path: str, inv_path: str
    ) -> list[tuple]:
        rects = []
        for path, scale in ((img_path, 1), (enhanced_path, 0.5), (inv_path, 0.5)):
            boxes = self._paddle_boxes(path)
            logger.debug("PaddleOCR hits — %s: %d", os.path.basename(path), len(boxes))
            for x, y, w, h in boxes:
                rects.append((int(x * scale), int(y * scale),
                               int(w * scale), int(h * scale)))
        return rects

    def _paddle_boxes(self, img_path: str) -> list[tuple]:
        """
        Run PaddleOCR detection and return (x, y, w, h) for each text region.
        Supports both v2 (.ocr()) and v3 (.predict()) APIs.
        """
        # v3 API
        if hasattr(self._ocr, 'predict'):
            raw = self._ocr.predict(img_path)
            return self._parse_v3(raw)
        # v2 API
        raw = self._ocr.ocr(img_path, cls=True)
        return self._parse_v2(raw)

    def _parse_v3(self, raw) -> list[tuple]:
        """Parse PaddleOCR v3 predict() output.

        Key fix: use det_poly (DB Net detection quadrilateral) not rec_poly.
        rec_poly is the recognition crop region — it can be smaller or shifted
        relative to the actual detected text boundary. det_poly is the raw
        output of the detection stage and gives the tightest, most accurate bbox.
        Fall back to rec_poly only when det_poly is absent (older v3 builds).
        """
        boxes, low_score = [], 0
        if not raw:
            return boxes
        # v3 returns a list of Result objects per image
        for result in raw:
            for item in (result if isinstance(result, list) else [result]):
                try:
                    if isinstance(item, dict):
                        # Prefer det_poly; fall back to rec_poly for old builds
                        quad  = item.get('det_poly') or item.get('rec_poly')
                        score = item.get('rec_score', 1.0)
                    else:
                        quad  = getattr(item, 'det_poly', None) or getattr(item, 'rec_poly', None)
                        score = getattr(item, 'rec_score', 1.0)
                except (KeyError, AttributeError):
                    continue
                if quad is None:
                    continue
                if score < self.MIN_SCORE:
                    low_score += 1
                    continue
                pts = np.array(quad, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                if w > 0 and h > 0:
                    boxes.append((x, y, w, h))
        if low_score:
            logger.debug('discarded %d low-score detections (< %.2f)', low_score, self.MIN_SCORE)
        return boxes

    def _parse_v2(self, raw) -> list[tuple]:
        """Parse PaddleOCR v2 ocr() output."""
        boxes, low_score = [], 0
        if not raw or raw[0] is None:
            return boxes
        for line in raw[0]:
            quad, (text, score) = line
            if score < self.MIN_SCORE:
                low_score += 1
                continue
            pts = np.array(quad, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            if w > 0 and h > 0:
                boxes.append((x, y, w, h))
        if low_score:
            logger.debug('discarded %d low-score detections (< %.2f)', low_score, self.MIN_SCORE)
        return boxes

    def _find_panel_dividers(self, img: cv2.typing.MatLike) -> list[tuple]:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_np  = np.array(Image.fromarray(img_rgb).convert("L").convert("RGB"))
        with torch.no_grad():
            results = self.magi.do_chapter_wide_prediction(
                [img_np], {"images": [], "names": []}, use_tqdm=False, do_ocr=False
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
            if not any(contains(p, other) for j, other in enumerate(panel_rects) if i != j)
        ]
        logger.debug("Magi panels after filter: %d → %s", len(panel_rects), panel_rects)
        return panel_rects

    def _group_rects_by_panel(
        self, rects: list[tuple], panels: list[tuple]
    ) -> dict[int, list]:
        panel_groups: dict[int, list] = {i: [] for i in range(len(panels))}
        panel_groups[-1] = []
        for rx, ry, rw, rh in rects:
            rc_x, rc_y = rx + rw / 2, ry + rh / 2
            idx = next(
                (i for i, (px, py, pw, ph) in enumerate(panels)
                 if px <= rc_x <= px + pw and py <= rc_y <= py + ph),
                -1,
            )
            if idx == -1 and panels:
                best_area, best_idx = 0, -1
                for i, (px, py, pw, ph) in enumerate(panels):
                    ix1 = max(rx, px);           iy1 = max(ry, py)
                    ix2 = min(rx + rw, px + pw); iy2 = min(ry + rh, py + ph)
                    overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    if overlap > best_area:
                        best_area, best_idx = overlap, i
                if best_area >= rw * rh * 0.30:
                    logger.debug(
                        "rect (%d,%d,%d,%d) reassigned to panel %d via overlap (%.0f%%)",
                        rx, ry, rw, rh, best_idx, 100 * best_area / (rw * rh)
                    )
                    idx = best_idx
            panel_groups[idx].append((rx, ry, rw, rh))
        return panel_groups

    @staticmethod
    def _crop_bubble(
        img: cv2.typing.MatLike,
        x: int, y: int, w: int, h: int,
        crops_dir: str, base_name: str, bubble_id: int,
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

    def _group_nearby_rects(
        self, rects: list[tuple],
        gap_x: int = 20, gap_y: int = 20,
        max_h: float | None = None, max_w: float | None = None,
        h_dividers: list | None = None,
        v_dividers: list | None = None,
    ) -> list[tuple]:
        """
        Cluster nearby rects into bubble candidates.

        h_dividers / v_dividers: pixel positions of horizontal / vertical
        panel boundaries. A merged bbox that crosses a divider is rejected,
        preventing characters from two adjacent panels being folded into one
        bubble (bug that was absent in EasyOCR but present here).
        """
        if not rects:
            return []

        h_dividers = h_dividers or []
        v_dividers = v_dividers or []

        def cluster_bbox(cluster):
            x  = min(r[0] for r in cluster)
            y  = min(r[1] for r in cluster)
            x2 = max(r[0] + r[2] for r in cluster)
            y2 = max(r[1] + r[3] for r in cluster)
            return (x, y, x2 - x, y2 - y)

        def bbox_crosses_divider(x, y, w, h):
            return (
                any(x < d < x + w for d in v_dividers) or
                any(y < d < y + h for d in h_dividers)
            )

        def near(a, b):
            return (a[0] - gap_x < b[0] + b[2] and a[0] + a[2] + gap_x > b[0] and
                    a[1] - gap_y < b[1] + b[3] and a[1] + a[3] + gap_y > b[1])

        clusters = [[r] for r in rects]
        changed  = True
        while changed:
            changed, merged, used = False, [], set()
            for i, ci in enumerate(clusters):
                if i in used:
                    continue
                group = list(ci)
                for j, cj in enumerate(clusters):
                    if j <= i or j in used:
                        continue
                    if any(near(a, b) for a in ci for b in cj):
                        candidate      = group + list(cj)
                        bx, by, bw, bh = cluster_bbox(candidate)
                        if max_h and bh > max_h:              continue
                        if max_w and bw > max_w:              continue
                        if bbox_crosses_divider(bx, by, bw, bh): continue
                        group = candidate
                        used.add(j)
                        changed = True
                used.add(i)
                merged.append(group)
            clusters = merged
        return [cluster_bbox(c) for c in clusters]

    def _merge_overlapping_rects(
        self, rects: list[tuple], overlap_threshold: float = 0.05
    ) -> list[tuple]:
        if not rects:
            return []

        def overlaps(a, b):
            ax, ay, aw, ah = a;  bx, by, bw, bh = b
            ix  = max(ax, bx);   iy  = max(ay, by)
            ix2 = min(ax+aw, bx+bw); iy2 = min(ay+ah, by+bh)
            inter = max(0, ix2-ix) * max(0, iy2-iy)
            return inter > 0 and (inter / min(aw*ah, bw*bh)) >= overlap_threshold

        def union(a, b):
            ax, ay, aw, ah = a;  bx, by, bw, bh = b
            x  = min(ax, bx);    y  = min(ay, by)
            x2 = max(ax+aw, bx+bw); y2 = max(ay+ah, by+bh)
            return (x, y, x2-x, y2-y)

        changed = True
        while changed:
            changed, merged, used = False, [], set()
            for i, a in enumerate(rects):
                if i in used: continue
                current = a
                for j, b in enumerate(rects):
                    if j <= i or j in used: continue
                    if overlaps(current, b):
                        current = union(current, b)
                        used.add(j); changed = True
                used.add(i); merged.append(current)
            rects = merged
        return rects

    # ------------------------------------------------------------------ #
    # Static image processing utilities                                    #
    # ------------------------------------------------------------------ #

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