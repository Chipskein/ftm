import json
import logging
import os

import cv2
import numpy as np
import pytesseract

from panel.PanelDetector import PanelDetector
from profiler.ResourceMonitor import ResourceMonitor
from .EngineOCR import EngineOCR
from dto.BubbleZone import BubbleZone

logger = logging.getLogger(__name__)

class TesseractOCR(EngineOCR):

    MIN_CONF     = 65
    MIN_BUBBLE_W = 30
    MIN_BUBBLE_H = 30

    def __init__(
        self,
        panel_detector: PanelDetector,
        debug: bool = False,
        monitor: ResourceMonitor | None = None,
    ):
        super().__init__("TesseractOCR")
        self.panel_detector = panel_detector
        self.debug   = debug
        self.monitor = monitor
        # PSM 11 — sparse text, best for manga's scattered layout
        self.tesseract_cfg = "--oem 1 --psm 11"
        self.lang = "jpn+jpn_vert"
        if debug:
            logger.setLevel(logging.DEBUG)
        logger.debug("TesseractOCR initialised  lang=%s  cfg=%s", self.lang, self.tesseract_cfg)

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
        logger.info("processing image %s (%dx%d)", os.path.basename(img_path), img_w, img_h)

        # Adaptive preprocessing based on image quality
        quality = self._assess_quality(img)
        logger.info(
            "quality — sharpness=%.1f  contrast=%.1f  resolution=%dpx",
            quality["sharpness"], quality["contrast"], quality["resolution"],
        )
        if quality["low_res"]:
            logger.warning("low resolution (%dpx short side) — OCR quality may suffer",
                           quality["resolution"])
        if quality["needs_contrast"]:
            logger.info("contrast low (%.1f) — applying CLAHE", quality["contrast"])
            img = self._enhance_contrast_clahe(img)
        if quality["needs_sharpen"]:
            logger.info("sharpness low (%.1f) — applying sharpen", quality["sharpness"])
            img = self._sharpen(img)

        # Double-page spread handling
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

        logger.debug("running Tesseract on 2 variants...")
        rects = self._detect_text_rects(img_path, enhanced_path, inv_path)
        logger.debug("raw text rects: %d", len(rects))

        logger.debug("detecting panels...")
        panels = self.panel_detector._find_panel_dividers(img)
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
                logger.debug("[%d] bubble at (%d,%d,%d,%d) panel=%d → %s",
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
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _detect_text_rects(
        self, img_path: str, enhanced_path: str, inv_path: str
    ) -> list[tuple]:
        """
        Run Tesseract on original + best variant (2 instead of 3).
        _best_variant selects enhanced or inverted based on background brightness,
        cutting duplicate detections roughly in half vs the old 3-variant approach.
        """
        img     = cv2.imread(img_path)
        best    = self._best_variant(img)
        best_path = enhanced_path.replace("_enhanced", "_best")
        cv2.imwrite(best_path, best)

        rects = []
        for path, scale in ((img_path, 1), (best_path, 0.5)):
            src  = cv2.imread(path)
            boxes = self._tesseract_boxes(src)
            logger.debug("Tesseract hits — %s: %d", os.path.basename(path), len(boxes))
            for x, y, w, h in boxes:
                rects.append((int(x * scale), int(y * scale),
                               int(w * scale), int(h * scale)))
        return rects

    def _tesseract_boxes(self, img: cv2.typing.MatLike) -> list[tuple]:
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = pytesseract.image_to_data(
            rgb, lang=self.lang, config=self.tesseract_cfg,
            output_type=pytesseract.Output.DICT,
        )
        boxes, low_conf = [], 0
        for i, text in enumerate(data["text"]):
            if not str(text).strip():
                continue
            conf = int(data["conf"][i])
            if conf < self.MIN_CONF:
                low_conf += 1
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            if w > 0 and h > 0:
                boxes.append((x, y, w, h))
        if low_conf:
            logger.debug("discarded %d low-confidence detections (< %d)", low_conf, self.MIN_CONF)
        return boxes


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
        """
        h, w = img.shape[:2]
        if w <= h:
            return None

        gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        center    = w // 2
        band_half = int(w * search_band / 2)
        x_start   = center - band_half
        x_end     = center + band_half

        band          = gray[:, x_start:x_end]
        white_mask    = (band >= white_thresh).astype(np.float32)
        col_white_pct = white_mask.mean(axis=0)

        gutter_cols = np.where(col_white_pct >= min_white_col_pct)[0]
        if len(gutter_cols) == 0:
            return None

        return x_start + int(gutter_cols.mean())

    # ------------------------------------------------------------------ #
    # Static helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _best_variant(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """
        Return the most useful upscaled variant for this image.
        Dark backgrounds (white text on black) get inverted;
        light backgrounds get enhanced only.
        """
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean   = gray.mean()
        img_up = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2),
                            interpolation=cv2.INTER_CUBIC)
        enhanced = TesseractOCR._sharpen(TesseractOCR._enhance_contrast_clahe(img_up))
        if mean < 100:
            return cv2.bitwise_not(enhanced)
        return enhanced

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
        self,
        rects: list[tuple],
        gap_x: int = 20, gap_y: int = 20,
        max_h: float | None = None, max_w: float | None = None,
        h_dividers: list | None = None,
        v_dividers: list | None = None,
    ) -> list[tuple]:
        """
        Cluster nearby rects into bubble candidates.
        h_dividers/v_dividers prevent merging across panel boundaries.
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
                        if max_h and bh > max_h:                  continue
                        if max_w and bw > max_w:                  continue
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
    def _assess_quality(img: cv2.typing.MatLike) -> dict:
        """
        Assess sharpness, contrast and resolution of the full page image.
        Thresholds tuned for scanned manga pages:
          sharpness  < 120  → blurry scan, apply unsharp mask
          contrast   <  35  → faded/low-contrast, apply CLAHE
          resolution < 800  → very small image, log a warning
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast   = float(gray.std())
        resolution = min(h, w)

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