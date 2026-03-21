import json
import logging
import os
import warnings

import cv2
import easyocr
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

from .EngineOCR import EngineOCR
from .types.BubbleZone import BubbleZone

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def load_magi() -> AutoModel:
    logger.info("loading Magi v2 model...")
    model = AutoModel.from_pretrained(
        "ragavsachdeva/magiv2", trust_remote_code=True
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Magi loaded on GPU")
    else:
        logger.info("Magi loaded on CPU")
    return model


class EazyOCR(EngineOCR):

    def __init__(self, magi_model: AutoModel, debug: bool = False):
        super().__init__("EazyOCR")
        self.magi = magi_model
        self.debug = debug
        self.reader = easyocr.Reader(["ja"], gpu=torch.cuda.is_available())
        self.reader_cfg = dict(
            detail=1, text_threshold=0.2, link_threshold=0.1,
            low_text=0.2, contrast_ths=0.05, adjust_contrast=0.5,
        )
        if debug:
            logger.setLevel(logging.DEBUG)
        logger.debug("EazyOCR initialised (gpu=%s)", torch.cuda.is_available())

    # ------------------------------------------------------------------ #
    # Abstract method implementations                                      #
    # ------------------------------------------------------------------ #

    def loadImage(self, image_path: str) -> cv2.typing.MatLike:
        return cv2.imread(image_path)

    def preProcessImage(self, image: cv2.typing.MatLike) -> dict:
        """Returns a dict with upscaled, enhanced, and inverted variants."""
        img_up = self._upscale(image, scale=2)
        img_enhanced = self._sharpen(self._enhance_contrast_clahe(img_up))
        img_inv = cv2.bitwise_not(img_enhanced)
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
            logger.warning("low resolution (%dpx short side) — OCR quality may suffer",
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

        variants = self.preProcessImage(img)
        img_up, img_enhanced, img_inv = (
            variants["up"], variants["enhanced"], variants["inv"]
        )

        base_name = os.path.splitext(os.path.basename(img_path))[0] + suffix
        enhanced_path = f"{output_dir}/{base_name}_enhanced.png"
        inv_path = f"{output_dir}/{base_name}_inv.png"

        cv2.imwrite(enhanced_path, img_enhanced)
        cv2.imwrite(inv_path, img_inv)
        if self.debug:
            cv2.imwrite(f"{output_dir}/{base_name}_up.png", img_up)
            logger.debug("intermediate images saved to %s", output_dir)

        logger.debug("running EasyOCR on 3 variants...")
        rects = self._detect_text_rects(img_path, enhanced_path, inv_path)
        logger.debug("raw text rects: %d", len(rects))

        logger.debug("detecting panels with Magi...")
        panels = self._find_panel_dividers(img)
        logger.info("panels found: %d", len(panels))

        panel_groups = self._group_rects_by_panel(rects, panels)
        logger.debug("rects per panel: %s",
                     {k: len(v) for k, v in panel_groups.items()})

        crops_dir = os.path.join(output_dir, "crops_" + base_name)
        os.makedirs(crops_dir, exist_ok=True)

        out = img.copy()
        results: list[BubbleZone] = []

        for idx, group_rects in panel_groups.items():
            if idx >= 0:
                px, py, pw, ph = panels[idx]
                ratio = (pw * ph) / img_area
                max_h, max_w = ph * 0.4, pw * 0.6
            else:
                ratio = 1.0
                max_h, max_w = None, None

            # Scale gaps to panel size — small panels need tighter gaps
            # to avoid merging separate bubbles that are close together.
            if ratio > 0.30:
                gap_x, gap_y = 10, 10
            elif ratio > 0.10:
                gap_x, gap_y = 18, 18
            else:
                # Very small panel — keep gaps tight, bubbles are close by definition
                gap_x, gap_y = 10, 12

            # max_h/max_w: a single bubble can't be taller/wider than this fraction.
            # Use a tighter limit for small panels so two stacked bubbles can't merge.
            if idx >= 0:
                max_h = ph * (0.45 if ratio > 0.15 else 0.35)
                max_w = pw * 0.70

            bubbles = self._group_nearby_rects(
                group_rects, gap_x=gap_x, gap_y=gap_y,
                max_h=max_h, max_w=max_w,
            )
            bubbles = self._merge_overlapping_rects(bubbles)
            bubbles = [b for b in bubbles if b[2] >= 20 and b[3] >= 20]
            logger.debug("panel %d — ratio=%.2f  gap=(%d,%d)  max_h=%.0f  max_w=%.0f  → %d bubble(s)",
                         idx, ratio, gap_x, gap_y,
                         max_h if max_h else -1,
                         max_w if max_w else -1,
                         len(bubbles))

            if idx == -1 and group_rects:
                logger.warning("panel -1: %d rect(s) outside all panels — may be misdetected",
                               len(group_rects))

            color = (0, 255, 0) if idx >= 0 else (0, 165, 255)
            # Clip bubble coords to panel bounds to avoid grabbing panel border pixels
            if idx >= 0:
                px, py, pw, ph = panels[idx]
                clip = (px, py, px + pw, py + ph)
            else:
                clip = (0, 0, img.shape[1], img.shape[0])

            for x, y, w, h in bubbles:
                # Clip to panel bounds
                x1 = max(x, clip[0])
                y1 = max(y, clip[1])
                x2 = min(x + w, clip[2])
                y2 = min(y + h, clip[3])
                if x2 - x1 < 20 or y2 - y1 < 20:
                    logger.debug("bubble clipped to nothing in panel %d, skipping", idx)
                    continue
                x, y, w, h = x1, y1, x2 - x1, y2 - y1

                bubble_id = id_offset + len(results) + 1
                crop_path = self._crop_bubble(
                    img, x, y, w, h, crops_dir, base_name, bubble_id
                )
                if crop_path is None:
                    logger.warning("empty crop skipped (bubble_id=%d, x=%d y=%d w=%d h=%d)",
                                   bubble_id, x, y, w, h)
                    continue
                results.append(
                    BubbleZone(
                        id=bubble_id,
                        x=x + x_offset, y=y, w=w, h=h,
                        crop=crop_path,
                        jp_text="",
                        en_text="",
                        translated_text="",
                    )
                )
                cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
                # Label inside bubble top-left to avoid overlap with neighbouring bubbles
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

        logger.info(
            "done — %d bubble(s) | result → %s | crops → %s/",
            len(results), result_path, crops_dir,
        )
        logger.debug("JSON → %s", json_path)

        return results

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _crop_bubble(
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

    def _detect_text_rects(
        self, img_path: str, enhanced_path: str, inv_path: str
    ) -> list[tuple]:
        r1 = self.reader.readtext(img_path, **self.reader_cfg)
        r2 = self.reader.readtext(enhanced_path, **self.reader_cfg)
        r3 = self.reader.readtext(inv_path, **self.reader_cfg)
        logger.debug("EasyOCR hits — original=%d  enhanced=%d  inverted=%d",
                     len(r1), len(r2), len(r3))

        rects = []
        for (bbox, _, _) in r1:
            x, y, w, h = cv2.boundingRect(np.array(bbox, dtype=np.int32))
            rects.append((x, y, w, h))
        for (bbox, _, _) in r2 + r3:
            x, y, w, h = cv2.boundingRect(np.array(bbox, dtype=np.int32))
            rects.append((x // 2, y // 2, w // 2, h // 2))
        return rects

    def _find_panel_dividers(self, img: cv2.typing.MatLike) -> list[tuple]:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_np = np.array(Image.fromarray(img_rgb).convert("L").convert("RGB"))

        character_bank = {"images": [], "names": []}
        with torch.no_grad():
            results = self.magi.do_chapter_wide_prediction(
                [img_np], character_bank, use_tqdm=False, do_ocr=False
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
        logger.debug("Magi panels after containment filter: %d → %s",
                     len(panel_rects), panel_rects)

        return panel_rects

    def _group_rects_by_panel(
        self, rects: list[tuple], panels: list[tuple]
    ) -> dict[int, list]:
        panel_groups: dict[int, list] = {i: [] for i in range(len(panels))}
        panel_groups[-1] = []
        for rx, ry, rw, rh in rects:
            rc_x, rc_y = rx + rw / 2, ry + rh / 2

            # Primary: centroid inside panel
            idx = next(
                (i for i, (px, py, pw, ph) in enumerate(panels)
                 if px <= rc_x <= px + pw and py <= rc_y <= py + ph),
                -1,
            )

            # Fallback: centroid missed — assign to panel with largest overlap area
            if idx == -1 and panels:
                best_area, best_idx = 0, -1
                for i, (px, py, pw, ph) in enumerate(panels):
                    ix1 = max(rx, px);          iy1 = max(ry, py)
                    ix2 = min(rx + rw, px + pw); iy2 = min(ry + rh, py + ph)
                    overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    if overlap > best_area:
                        best_area = overlap
                        best_idx  = i
                # Only reassign if overlap covers at least 30% of the rect
                if best_area >= rw * rh * 0.30:
                    logger.debug(
                        "rect (%d,%d,%d,%d) centroid missed panels — reassigned to panel %d via overlap (%.0f%%)",
                        rx, ry, rw, rh, best_idx, 100 * best_area / (rw * rh)
                    )
                    idx = best_idx

            panel_groups[idx].append((rx, ry, rw, rh))
        return panel_groups

    def _group_nearby_rects(
        self,
        rects: list[tuple],
        gap_x: int = 25,
        gap_y: int = 40,
        h_dividers: list | None = None,
        v_dividers: list | None = None,
        max_h: float | None = None,
        max_w: float | None = None,
    ) -> list[tuple]:
        if not rects:
            return []

        h_dividers = h_dividers or []
        v_dividers = v_dividers or []

        def cluster_bbox(cluster):
            x = min(r[0] for r in cluster)
            y = min(r[1] for r in cluster)
            x2 = max(r[0] + r[2] for r in cluster)
            y2 = max(r[1] + r[3] for r in cluster)
            return (x, y, x2 - x, y2 - y)

        def bbox_crosses_divider(x, y, w, h):
            return any(x < d < x + w for d in v_dividers) or \
                   any(y < d < y + h for d in h_dividers)

        def near(a, b):
            return (
                a[0] - gap_x < b[0] + b[2]
                and a[0] + a[2] + gap_x > b[0]
                and a[1] - gap_y < b[1] + b[3]
                and a[1] + a[3] + gap_y > b[1]
            )

        clusters = [[r] for r in rects]
        changed = True
        while changed:
            changed = False
            merged, used = [], set()
            for i, ci in enumerate(clusters):
                if i in used:
                    continue
                group = list(ci)
                for j, cj in enumerate(clusters):
                    if j <= i or j in used:
                        continue
                    if any(near(a, b) for a in ci for b in cj):
                        candidate = group + list(cj)
                        bx, by, bw, bh = cluster_bbox(candidate)
                        if bbox_crosses_divider(bx, by, bw, bh):
                            continue
                        if max_h and bh > max_h:
                            continue
                        if max_w and bw > max_w:
                            continue
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
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            ix = max(ax, bx);        iy = max(ay, by)
            ix2 = min(ax + aw, bx + bw); iy2 = min(ay + ah, by + bh)
            inter = max(0, ix2 - ix) * max(0, iy2 - iy)
            return inter > 0 and (inter / min(aw * ah, bw * bh)) >= overlap_threshold

        def union(a, b):
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            x = min(ax, bx);         y = min(ay, by)
            x2 = max(ax + aw, bx + bw); y2 = max(ay + ah, by + bh)
            return (x, y, x2 - x, y2 - y)

        changed = True
        while changed:
            changed = False
            merged, used = [], set()
            for i, a in enumerate(rects):
                if i in used:
                    continue
                current = a
                for j, b in enumerate(rects):
                    if j <= i or j in used:
                        continue
                    if overlaps(current, b):
                        current = union(current, b)
                        used.add(j)
                        changed = True
                used.add(i)
                merged.append(current)
            rects = merged

        return rects

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

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        center = w // 2
        band_half = int(w * search_band / 2)
        x_start = center - band_half
        x_end   = center + band_half

        band = gray[:, x_start:x_end]
        white_mask = (band >= white_thresh).astype(np.float32)
        col_white_pct = white_mask.mean(axis=0)

        gutter_cols = np.where(col_white_pct >= min_white_col_pct)[0]
        if len(gutter_cols) == 0:
            return None

        split_x = x_start + int(gutter_cols.mean())
        return split_x

    # ------------------------------------------------------------------ #
    # Static image processing utilities                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _assess_quality(img: cv2.typing.MatLike) -> dict:
        """
        Assess sharpness, contrast and resolution of the full page image.
        Returns a dict with metrics and enhancement flags.

        Thresholds tuned for scanned manga pages (full page, not crops):
          sharpness  < 120  → blurry scan, apply unsharp mask
          contrast   <  35  → faded/low-contrast, apply CLAHE
          resolution < 800  → very small image, log a warning
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast  = float(gray.std())
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
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _sharpen(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)