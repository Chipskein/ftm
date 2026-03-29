import json
import logging
import os

import cv2
import easyocr
import numpy as np
import torch

from .EngineOCR import EngineOCR
from dto.BubbleZone import BubbleZone
from profiler.ResourceMonitor import ResourceMonitor
from .panel.PanelDetector import PanelDetector

logger = logging.getLogger(__name__)

class EazyOCR(EngineOCR):

    def __init__(
            self, 
            panel_detector: PanelDetector,
            debug: bool = False,
            monitor: ResourceMonitor | None = None,
            use_cpu: bool = False
    ):
        super().__init__("EazyOCR")
        self.panel_detector = panel_detector
        self.monitor = monitor
        self.debug = debug
        self.use_cpu = use_cpu
        use_gpu = (not use_cpu) and torch.cuda.is_available()
        self.reader = easyocr.Reader(["ja"], gpu=use_gpu)
        self.reader_cfg = dict(
            detail=1, text_threshold=0.2, link_threshold=0.1,
            low_text=0.2, contrast_ths=0.05, adjust_contrast=0.5,
        )
        if debug:
            logger.setLevel(logging.DEBUG)
        logger.debug("EazyOCR initialised (gpu=%s)", use_gpu)
        logger.debug("EasyOCR config: %s", self.reader_cfg)
        logger.debug("Panel detector: %s", self.panel_detector.__class__.__name__)

    def run(self, img_path: str, output_dir: str) -> list[BubbleZone]:
        if not os.path.exists(img_path):
            logger.error("image not found: %s", img_path)
            return []

        img = self.loadImage(img_path)
        h, w = img.shape[:2]
        logger.info("processing image %s (%dx%d)", os.path.basename(img_path), w, h)

        img = self.assess_and_fix_quality(img)

        split_x = self.detect_spread_split(img)
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

        # Save the (possibly cropped) image so that all 3 EasyOCR passes run on
        # the same spatial region. Without this, r1 uses the original file on disk
        # (which may be the full spread) while r2/r3 use the cropped variants,
        # producing rects with mismatched coordinate spaces.
        orig_crop_path = f"{output_dir}/{base_name}_orig.png"
        cv2.imwrite(orig_crop_path, img)

        logger.debug("running EasyOCR on 3 variants...")
        rects = self._detect_text_rects(orig_crop_path, enhanced_path, inv_path, output_dir)
        logger.debug("raw text rects: %d", len(rects))

        rects = self._normalize_rects(rects)
        logger.debug("normalized text rects: %d", len(rects))

        logger.debug("detecting panels ...")
        panels = self.panel_detector._find_panel_dividers(img)
        logger.info("panels found: %d", len(panels))

        panel_groups = self._group_rects_by_panel(rects, panels, img_w, img_h)
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
            elif idx == -2:
                # Global rects — skip grouping entirely, just pass through as-is
                ratio = 1.0
                max_h, max_w = None, None
                pw, ph = img_w, img_h
            else:
                ratio = 1.0
                max_h, max_w = None, None
                pw, ph = img_w, img_h

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

            raw_rects = [(rx, ry, rw, rh) for rx, ry, rw, rh, *_ in group_rects]

            if idx == -2:
                # Global rects bypass the clustering step — keep each rect as its own bubble
                bubbles = raw_rects
            else:
                bubbles = self._group_nearby_rects(
                    raw_rects, gap_x=gap_x, gap_y=gap_y,
                    max_h=max_h, max_w=max_w,
                    panel_w=pw if idx >= 0 else None,
                    panel_h=ph if idx >= 0 else None,
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
            if idx == -2 and group_rects:
                logger.info("global: %d rect(s) touch 3+ panels — not grouped with any panel",
                            len(group_rects))

            if idx >= 0:
                color = (0, 255, 0)
            elif idx == -2:
                color = (255, 0, 255)   # magenta for globals
            else:
                color = (0, 165, 255)   # orange for void

            # Clip bubble coords to panel bounds to avoid grabbing panel border pixels
            if idx >= 0:
                px, py, pw, ph = panels[idx]
                clip = (px, py, px + pw, py + ph)
            else:
                clip = (0, 0, img.shape[1], img.shape[0])

            for x, y, w, h in bubbles:
                spanning = any(
                    r[4]
                    for r in group_rects
                    if len(r) > 4
                    and r[0] >= x and r[1] >= y
                    and r[0] + r[2] <= x + w and r[1] + r[3] <= y + h
                )

                if spanning:
                    x1, y1, x2, y2 = x, y, x + w, y + h
                else:
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
                crop_path = self.crop_bubble(
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

    def _detect_text_rects(
        self, 
        img_path: str, 
        enhanced_path: str, 
        inv_path: str, 
        output_dir: str
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
        
        if logger.isEnabledFor(logging.DEBUG):
            dbg = cv2.imread(img_path)
            for (x, y, w, h) in rects:
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            raw_detect_path = f"{output_dir}/{base_name}_rects_debug.png"
            cv2.imwrite(raw_detect_path, dbg)
            logger.debug("text rects debug image saved to %s", raw_detect_path)

        return rects

    @staticmethod
    def _normalize_rects(rects: list[tuple]) -> list[tuple]:
        """
        Fixes bounding boxes inflated by EasyOCR.

        For vertical text (H > W), EasyOCR sometimes returns a box that is far
        too wide — e.g. W=249 for a single column of kanji that should be ~50px.
        This causes the rect to bleed into a neighbouring panel and trigger the
        wrong classification rule.

        Rule: if the rect is vertical (H > W) AND W > H * MAX_W_RATIO, clamp the
        width to H * MAX_W_RATIO centred on the original horizontal midpoint.

        MAX_W_RATIO = 0.8  (a single vertical column is rarely wider than ~1/3
                             of its height; kanji are roughly square so one column
                             of N chars ≈ char_size wide, N*char_size tall → ratio ~1/N)
        """
        MAX_W_RATIO = 0.8
        result = []
        for rx, ry, rw, rh in rects:
            if rh > rw and rw > rh * MAX_W_RATIO:
                new_w = int(rh * MAX_W_RATIO)
                cx   = rx + rw // 2
                new_x = cx - new_w // 2
                logger.debug(
                    "normalize: vertical rect (%d,%d,%d,%d) width clamped → (%d,%d,%d,%d)",
                    rx, ry, rw, rh, new_x, ry, new_w, rh,
                )
                result.append((new_x, ry, new_w, rh))
            else:
                result.append((rx, ry, rw, rh))
        return result

    def _group_rects_by_panel(
        self, rects: list[tuple], panels: list[tuple], img_w: int, img_h: int
    ) -> dict[int, list]:
        """
        Classifies each text rect into a panel according to the following rules:

        - Global Rule: rect touches 3+ panels → marked global (key -2), not grouped
          with any panel.
        - Precedence Rule: "global" overrides all local assignment attempts.
        - Unit Ownership Rule: rect touches exactly 1 panel → belongs exclusively to it.
        - Dominant Panel Rule: rect touches 2 panels → belongs to the one with the
          largest intersection area.
        - Vacuum Rule: rect is half inside a panel and half in empty space → adopted
          entirely by that panel.
        - Safety Margin: a -2 px internal padding is applied before collision checks
          to avoid phantom border touches.
        """
        COLLISION_PADDING = 2   # px shrink applied to each rect before overlap test
        GLOBAL_THRESHOLD  = 3   # panels touched to be considered global

        panel_groups: dict[int, list] = {i: [] for i in range(len(panels))}
        panel_groups[-1]  = []   # no panel (void)
        panel_groups[-2]  = []   # global rects

        pre_merged = self._group_nearby_rects(rects, gap_x=2, gap_y=2)

        # Discard over-large pre-merges, keep originals
        max_premix_w = img_w * 0.1
        max_premix_h = img_h * 0.1
        clean: list[tuple] = []
        for mx, my, mw, mh in pre_merged:
            if mw > max_premix_w or mh > max_premix_h:
                clean.extend(
                    (rx, ry, rw, rh) for rx, ry, rw, rh in rects
                    if rx >= mx and ry >= my
                    and rx + rw <= mx + mw and ry + rh <= my + mh
                )
            else:
                clean.append((mx, my, mw, mh))
        rects = clean

        for rx, ry, rw, rh in rects:
            rect_area = rw * rh
            if rect_area == 0:
                continue

            # --- Safety Margin: shrink rect by COLLISION_PADDING before overlap test ---
            srx  = rx  + COLLISION_PADDING
            sry  = ry  + COLLISION_PADDING
            srx2 = rx  + rw - COLLISION_PADDING
            sry2 = ry  + rh - COLLISION_PADDING
            if srx2 <= srx or sry2 <= sry:
                # Rect is too small to survive padding — use original bounds
                srx, sry, srx2, sry2 = rx, ry, rx + rw, ry + rh

            # Compute overlap of padded rect with every panel
            overlaps: list[tuple[int, float]] = []
            for i, (px, py, pw, ph) in enumerate(panels):
                ix1 = max(srx, px);            iy1 = max(sry, py)
                ix2 = min(srx2, px + pw);      iy2 = min(sry2, py + ph)
                area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if area > 0:
                    overlaps.append((i, float(area)))

            touched = len(overlaps)

            # --- Global Rule ---
            if touched >= GLOBAL_THRESHOLD:
                logger.debug(
                    "rect (%d,%d,%d,%d) touches %d panels → GLOBAL",
                    rx, ry, rw, rh, touched,
                )
                panel_groups[-2].append((rx, ry, rw, rh, False))
                continue

            # --- Unit Ownership Rule + Vacuum Rule (AJUSTADA)---
            if touched == 1:
                idx = overlaps[0][0]
                area_interseccao = overlaps[0][1]
                percentual_no_painel = area_interseccao / rect_area
                
                # Se o retângulo estiver "equilibrado" entre painel e vácuo 
                # ou mais pra fora do que pra dentro, vira GLOBAL.
                # Use 0.50 ou 0.60 para ser a "fronteira" do vácuo.
                if percentual_no_painel < 0.60:
                    logger.debug(
                        "rect (%d,%d,%d,%d) parcialmente no vácuo (%.1f%%) → GLOBAL",
                        rx, ry, rw, rh, percentual_no_painel * 100
                    )
                    panel_groups[-2].append((rx, ry, rw, rh, False))
                else:
                    logger.debug(
                        "rect (%d,%d,%d,%d) touches 1 panel → panel %d",
                        rx, ry, rw, rh, idx,
                    )
                    panel_groups[idx].append((rx, ry, rw, rh, False))
                continue

            # --- Dominant Panel Rule ---
            # Compare percentage of the rect covered by each panel, not absolute
            # pixels — otherwise a giant panel always wins over a small one.
            if touched == 2:
                idx = max(overlaps, key=lambda t: t[1] / rect_area)[0]
                logger.debug(
                    "rect (%d,%d,%d,%d) touches 2 panels → dominant panel %d (%.0f%% of rect)",
                    rx, ry, rw, rh, idx,
                    100 * max(overlaps, key=lambda t: t[1] / rect_area)[1] / rect_area,
                )
                panel_groups[idx].append((rx, ry, rw, rh, False))
                continue

            # --- Extreme Vacuum Rule (Totalmente fora) ---
            if touched == 0:
                logger.debug("rect (%d,%d,%d,%d) totalmente no vácuo -> GLOBAL", rx, ry, rw, rh)
                panel_groups[-2].append((rx, ry, rw, rh, False))
                continue

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
        panel_w: int | None = None,
        panel_h: int | None = None,
    ) -> list[tuple]:
        """
        Groups nearby rects into clusters following these rules:

        - Origin Filter: only rects belonging to the same parent panel are ever
          passed together, so the filter is implicitly satisfied upstream.
        - Proximity Criterion: edge-to-edge distance must be < 5% of the panel
          dimension (fallback to gap_x / gap_y when panel dimensions are not given).
        - Horizontal Alignment: normal text (W > H) → group only when Y ranges overlap.
        - Vertical Alignment: vertical text (H > W) → group only when X ranges overlap.
        - Column Protection: for vertical text, horizontal tolerance is tighter than
          vertical tolerance (gap_x < gap_y).
        """
        if not rects:
            return []

        h_dividers = h_dividers or []
        v_dividers = v_dividers or []

        # Derive proximity limits from panel size (5% rule) when available
        if panel_w is not None:
            gap_x = max(4, int(panel_w * 0.05))
        if panel_h is not None:
            gap_y = max(4, int(panel_h * 0.05))

        def cluster_bbox(cluster):
            x  = min(r[0] for r in cluster)
            y  = min(r[1] for r in cluster)
            x2 = max(r[0] + r[2] for r in cluster)
            y2 = max(r[1] + r[3] for r in cluster)
            return (x, y, x2 - x, y2 - y)

        def bbox_crosses_divider(x, y, w, h):
            return (
                any(x < d < x + w for d in v_dividers)
                or any(y < d < y + h for d in h_dividers)
            )

        def is_vertical(rect) -> bool:
            return rect[3] > rect[2]   # H > W

        def edge_to_edge_dist(a, b) -> tuple[float, float]:
            """Returns (horizontal gap, vertical gap) between two rects (negative = overlap)."""
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            dx = max(0.0, float(max(ax, bx) - min(ax + aw, bx + bw)))
            dy = max(0.0, float(max(ay, by) - min(ay + ah, by + bh)))
            return dx, dy

        def y_overlap(a, b) -> bool:
            return a[1] < b[1] + b[3] and b[1] < a[1] + a[3]

        def x_overlap(a, b) -> bool:
            return a[0] < b[0] + b[2] and b[0] < a[0] + a[2]

        def can_merge(a, b) -> bool:
            """
            Decides whether two individual rects (not clusters) may be merged.

            Horizontal text (W > H):
              - Edge-to-edge X distance < gap_x  AND  Y ranges overlap.
            Vertical text (H > W):
              - Edge-to-edge Y distance < gap_y  AND  X ranges overlap.
              - Column protection: horizontal tolerance (gap_x) is kept tighter.
            Mixed pairs fall back to the looser rule (both gaps checked).
            """
            dx, dy = edge_to_edge_dist(a, b)
            a_vert = is_vertical(a)
            b_vert = is_vertical(b)

            def can_merge(a, b) -> bool:
                dx, dy = edge_to_edge_dist(a, b)
                a_vert = is_vertical(a)
                b_vert = is_vertical(b)

                if a_vert and b_vert:
                    # PROTEÇÃO DE COLUNA: 
                    # Reduzimos a tolerância horizontal (gap_x * 0.5) para evitar 
                    # agrupar duas colunas de fala distintas no mesmo balão.
                    return dy < gap_y and dx < (gap_x * 0.5) and x_overlap(a, b)

                if not a_vert and not b_vert:
                    # Texto Horizontal: exige sobreposição em Y (linhas)
                    return dx < gap_x and dy < gap_y and y_overlap(a, b)

                # Casos mistos: mantém a lógica padrão
                return dx < gap_x and dy < gap_y

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
                    # Check every pair across the two clusters
                    if any(can_merge(a, b) for a in ci for b in cj):
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
