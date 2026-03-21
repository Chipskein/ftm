"""
manga_bbox.py  [v3]
-------------
Generates a contour (polygon) that wraps the glyphs detected in a speech
bubble, following the real shape of the characters.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)


# ── Border component detection ────────────────────────────────────────────────

def find_border_components(binary, labels, stats, centroids, n_labels,
                           area_threshold=0.15, max_aspect_ratio=3.5,
                           min_area=4, border_margin_pct=0.12):
    h, w = binary.shape
    total_area = h * w
    # Cap scales with image size so small crops (e.g. 122px wide bubble crops)
    # don't discard edge glyphs. On a 122px image, the old hard cap of 40px
    # was ~33% of the width — enough to eat an entire column of kanji.
    margin_cap_x = max(6, min(w * 0.06, 40))
    margin_cap_y = max(6, min(h * 0.06, 40))
    margin_x = min(w * border_margin_pct, margin_cap_x)
    margin_y = min(h * border_margin_pct, margin_cap_y)

    border_ids = set()
    for i in range(1, n_labels):
        x, y, bw, bh, area = stats[i]
        cx, cy = centroids[i]

        if area < min_area:
            border_ids.add(i); continue

        if area > total_area * area_threshold:
            bbox_area_c1 = stats[i][2] * stats[i][3]
            density_c1   = area / bbox_area_c1 if bbox_area_c1 > 0 else 0
            if density_c1 < 0.35:
                border_ids.add(i); continue

        if bw > w * 0.4 or bh > h * 0.4:
            bbox_area_c5 = bw * bh
            density_c5   = area / bbox_area_c5 if bbox_area_c5 > 0 else 0
            if density_c5 < 0.35:
                border_ids.add(i); continue

        mask = (labels == i)
        touches_border = (mask[:, 0].sum()   > bh * 0.3 or
                          mask[:, w-1].sum() > bh * 0.3 or
                          mask[0, :].sum()   > bw * 0.3 or
                          mask[-1, :].sum()  > bw * 0.3)
        bbox_on_edge = (x == 0 or y == 0 or x + bw >= w or y + bh >= h)
        if touches_border or bbox_on_edge:
            bbox_area_c2       = bw * bh
            density_c2         = area / bbox_area_c2 if bbox_area_c2 > 0 else 0
            centroid_in_margin = (cx < margin_x or cx > w - margin_x or
                                  cy < margin_y or cy > h - margin_y)
            density_limit = 0.40 if bbox_on_edge else 0.35
            if density_c2 < density_limit or centroid_in_margin:
                border_ids.add(i); continue

        if bw == 0 or bh == 0:
            border_ids.add(i); continue
        ratio = max(bw, bh) / min(bw, bh)
        if ratio > 15:
            border_ids.add(i); continue
        if ratio > max_aspect_ratio:
            bbox_area_c3 = bw * bh
            density_c3   = area / bbox_area_c3 if bbox_area_c3 > 0 else 0
            if density_c3 < 0.35:
                border_ids.add(i); continue

        bbox_area = bw * bh
        if bbox_area > 0 and area > total_area * 0.005:
            if (area / bbox_area) < 0.20:
                border_ids.add(i); continue

        if area < total_area * 0.005:
            if (cx < margin_x or cx > w - margin_x or
                    cy < margin_y or cy > h - margin_y):
                border_ids.add(i); continue

        if area < total_area * 0.10:
            if (cx < margin_x or cx > w - margin_x or
                    cy < margin_y or cy > h - margin_y):
                bbox_area_c8 = bw * bh
                density_c8   = area / bbox_area_c8 if bbox_area_c8 > 0 else 0
                if density_c8 < 0.35:
                    border_ids.add(i); continue

    logger.debug("border filter: %d/%d components discarded", len(border_ids), n_labels - 1)
    return border_ids


def filter_text_glyphs(glyph_ids, centroids, stats, img_shape,
                       proximity_pct=0.80):
    if len(glyph_ids) < 1:
        return glyph_ids

    h, w = img_shape[:2]
    areas = sorted([(stats[i][4], i) for i in glyph_ids], reverse=True)
    anchors = [areas[0][1]]
    for k in range(1, len(areas)):
        if areas[k][0] < areas[0][0] * 0.30:
            break
        anchors.append(areas[k][1])

    anchor_centers = np.array([[centroids[i][0], centroids[i][1]] for i in anchors])
    ax_min = anchor_centers[:, 0].min()
    ax_max = anchor_centers[:, 0].max()
    ay_min = anchor_centers[:, 1].min()
    ay_max = anchor_centers[:, 1].max()

    anchor_diag = np.sqrt((ax_max - ax_min)**2 + (ay_max - ay_min)**2)
    margin = max(anchor_diag * proximity_pct, min(w, h) * 0.10)

    kept = set(anchors)
    for i in glyph_ids:
        if i in kept:
            continue
        cx, cy = centroids[i][0], centroids[i][1]
        if (ax_min - margin <= cx <= ax_max + margin and
                ay_min - margin <= cy <= ay_max + margin):
            kept.add(i)

    if len(anchors) > 0:
        min_anchor_area    = min(stats[i][4] for i in anchors)
        anchor_xs_arr      = [centroids[i][0] for i in anchors]
        cluster_cx_val     = float(np.mean(anchor_xs_arr))
        ax_span_val        = max(anchor_xs_arr) - min(anchor_xs_arr)
        x_outlier_thresh   = max(ax_span_val * 2.0, 60.0)
        before = len(kept)
        kept = {i for i in kept
                if not (stats[i][4] < min_anchor_area * 0.10 and
                        abs(centroids[i][0] - cluster_cx_val) > x_outlier_thresh)}
        removed = before - len(kept)
        if removed:
            logger.debug("glyph filter: removed %d x-outlier(s)", removed)

    result = list(kept) if kept else glyph_ids
    logger.debug("glyph filter: anchors=%d  kept=%d / %d", len(anchors), len(result), len(glyph_ids))
    return result


# ── Contour generation ────────────────────────────────────────────────────────

def collect_glyph_pixels(glyph_ids, labels):
    mask = np.zeros(labels.shape, dtype=np.uint8)
    for i in glyph_ids:
        mask[labels == i] = 255
    ys, xs = np.where(mask > 0)
    return np.column_stack((xs, ys))


def convex_hull_polygon(points):
    hull = ConvexHull(points)
    return points[hull.vertices]


def alpha_shape_contour(points, alpha_pct=0.05):
    if len(points) == 0:
        return points

    xs, ys = points[:, 0], points[:, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    W = int(x_max - x_min) + 1
    H = int(y_max - y_min) + 1

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[ys - y_min, xs - x_min] = 255

    gap_h = max(3, int(W * alpha_pct * 3))
    gap_v = max(3, int(H * alpha_pct * 0.5))
    if gap_h % 2 == 0: gap_h += 1
    if gap_v % 2 == 0: gap_v += 1
    k_wide  = cv2.getStructuringElement(cv2.MORPH_RECT, (gap_h, gap_v))
    merged  = cv2.dilate(mask, k_wide, iterations=1)

    tight = max(3, int(min(W, H) * alpha_pct * 0.4))
    if tight % 2 == 0: tight += 1
    k_tight = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tight, tight))
    merged  = cv2.dilate(merged, k_tight, iterations=1)
    eroded  = cv2.erode(merged, k_tight, iterations=1)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return points[ConvexHull(points).vertices]

    if len(contours) > 1:
        logger.debug("alpha_shape: %d contours — merging via convex hull", len(contours))
        all_pts = np.vstack(contours).reshape(-1, 2).astype(float)
        hull    = ConvexHull(all_pts)
        polygon = all_pts[hull.vertices].astype(int)
        polygon[:, 0] += x_min
        polygon[:, 1] += y_min
        return polygon

    main_contour = contours[0]
    epsilon = 0.005 * cv2.arcLength(main_contour, closed=True)
    approx  = cv2.approxPolyDP(main_contour, epsilon, closed=True)
    polygon = approx.reshape(-1, 2)
    polygon[:, 0] += x_min
    polygon[:, 1] += y_min
    return polygon


def expand_polygon(polygon, margin=4):
    centroid = polygon.mean(axis=0)
    vectors  = polygon.astype(float) - centroid
    norms    = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms    = np.where(norms == 0, 1, norms)
    expanded = polygon.astype(float) + (vectors / norms * margin)
    return np.round(expanded).astype(np.int32)


# ── Grain removal ─────────────────────────────────────────────────────────────

def remove_grain_dark_bg(gray):
    """Remove noise from dark textured backgrounds before binarisation."""
    h_img, w_img = gray.shape
    kernel3 = np.ones((3, 3), np.uint8)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    total_white = max(np.count_nonzero(thresh), 1)

    n0, _, st0, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    max_area = max((st0[i, cv2.CC_STAT_AREA] for i in range(1, n0)), default=0)
    fused = (max_area / total_white) > 0.40

    if fused:
        k5     = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(thresh, k5, iterations=1)
        binary = cv2.dilate(eroded, k5, iterations=1)
        n_after, _, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        logger.debug("grain: stroke-fusion detected — erode+dilate 5×5: %d→%d components",
                     n0 - 1, n_after - 1)
        return binary

    clean_full = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel3)

    n, lbl, st, _ = cv2.connectedComponentsWithStats(clean_full, connectivity=8)
    anchor_boxes = []
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        bw   = st[i, cv2.CC_STAT_WIDTH]
        bh   = st[i, cv2.CC_STAT_HEIGHT]
        if area > 50 and (area / (bw * bh)) > 0.3:
            x = st[i, cv2.CC_STAT_LEFT]
            y = st[i, cv2.CC_STAT_TOP]
            anchor_boxes.append((x, y, x + bw, y + bh))

    if not anchor_boxes:
        logger.debug("grain: no anchors found, returning clean_full")
        return clean_full

    pad = 20
    tx1 = max(0,         min(b[0] for b in anchor_boxes) - pad)
    ty1 = max(0,         min(b[1] for b in anchor_boxes) - pad)
    tx2 = min(w_img - 1, max(b[2] for b in anchor_boxes) + pad)
    ty2 = min(h_img - 1, max(b[3] for b in anchor_boxes) + pad)
    logger.debug("grain: text region [%d,%d]→[%d,%d]  (%dx%dpx of %dx%d)",
                 tx1, ty1, tx2, ty2, tx2 - tx1, ty2 - ty1, w_img, h_img)

    region_gray   = gray[ty1:ty2, tx1:tx2]
    _, region_thresh = cv2.threshold(region_gray, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    region_clean  = cv2.morphologyEx(region_thresh, cv2.MORPH_OPEN, kernel3)
    rn, rl, rs, _ = cv2.connectedComponentsWithStats(region_clean, connectivity=8)

    region_output = np.zeros_like(region_clean)
    kept = removed = 0
    for i in range(1, rn):
        area    = rs[i, cv2.CC_STAT_AREA]
        bw      = rs[i, cv2.CC_STAT_WIDTH]
        bh      = rs[i, cv2.CC_STAT_HEIGHT]
        solidez = area / (bw * bh) if bw * bh > 0 else 0
        if (bw <= 3 and bh <= 3) or area < 15 or solidez < 0.15:
            removed += 1
        else:
            region_output[rl == i] = 255
            kept += 1
    logger.debug("grain: kept=%d  removed=%d", kept, removed)

    output = clean_full.copy()
    output[ty1:ty2, tx1:tx2] = region_output
    return output


# ── Main pipeline ─────────────────────────────────────────────────────────────

def generate_glyph_bbox(
    image_path,
    output_path=None,
    mode="concave",
    alpha_pct=0.05,
    margin=4,
    draw_color=(0, 200, 80),
    draw_thickness=2,
    debug=False,
):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    p       = Path(image_path)
    out_dir = Path(output_path).parent if output_path else p.parent
    stem    = p.stem

    h_img, w_img = img_bgr.shape[:2]
    logger.debug("processing %s (%dx%d)  mode=%s  margin=%d",
                 p.name, w_img, h_img, mode, margin)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    hist_full     = cv2.calcHist([gray], [0], None, [256], [0, 256])
    bg_brightness = float(np.argmax(hist_full))
    logger.debug("bg brightness estimate: %.0f (%s)",
                 bg_brightness, "dark" if bg_brightness < 128 else "light")

    if bg_brightness < 128:
        binary = remove_grain_dark_bg(gray)
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        total_px  = h_img * w_img
        n_check, _, stats_check, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        max_area  = max((stats_check[i][4] for i in range(1, n_check)), default=0)
        if max_area > total_px * 0.10:
            fixed_t = max(60, int(bg_brightness * 0.40))
            logger.debug("large component detected — switching to fixed threshold %d", fixed_t)
            _, binary = cv2.threshold(gray, fixed_t, 255, cv2.THRESH_BINARY_INV)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    logger.debug("connected components: %d", n_labels - 1)

    border_ids = find_border_components(binary, labels, stats, centroids, n_labels)
    glyph_ids  = [i for i in range(1, n_labels) if i not in border_ids]
    glyph_ids  = filter_text_glyphs(glyph_ids, centroids, stats, img_bgr.shape)
    logger.debug("glyphs after filtering: %d", len(glyph_ids))

    if not glyph_ids:
        raise ValueError("No glyphs detected in image.")

    if debug:
        dbg = img_bgr.copy()

        # Red — border/discarded components
        for i in border_ids:
            gx, gy, gw, gh, area = stats[i]
            cv2.rectangle(dbg, (gx, gy), (gx + gw, gy + gh), (0, 0, 200), 1)
            cv2.putText(dbg, f"#{i} a={area}", (gx, max(gy - 3, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 200), 1)

        # Orange — passed border filter but removed by glyph filter
        all_non_border = set(range(1, n_labels)) - border_ids
        filtered_out   = all_non_border - set(glyph_ids)
        for i in filtered_out:
            gx, gy, gw, gh, area = stats[i]
            cv2.rectangle(dbg, (gx, gy), (gx + gw, gy + gh), (0, 140, 255), 1)
            cv2.putText(dbg, f"#{i} a={area}", (gx, max(gy - 3, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 140, 255), 1)

        # Green — kept as glyphs
        for i in glyph_ids:
            gx, gy, gw, gh, area = stats[i]
            cv2.rectangle(dbg, (gx, gy), (gx + gw, gy + gh), (0, 200, 80), 1)
            cv2.putText(dbg, f"#{i} a={area}", (gx, max(gy - 3, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 160, 60), 1)

        # Legend (bottom-left)
        legend = [
            ((0, 200, 80),  f"glyph  ({len(glyph_ids)})"),
            ((0, 140, 255), f"filtered  ({len(filtered_out)})"),
            ((0, 0, 200),   f"border  ({len(border_ids)})"),
        ]
        lh, lx = 14, 4
        for row, (color, text) in enumerate(legend):
            ly = dbg.shape[0] - (len(legend) - row) * lh - 2
            cv2.rectangle(dbg, (lx, ly - 10), (lx + 10, ly), color, -1)
            cv2.putText(dbg, text, (lx + 14, ly - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)

        dbg_path = str(out_dir / (stem + "_debug_glyphs.png"))
        cv2.imwrite(dbg_path, dbg)
        logger.debug(
            "debug image saved → %s  (green=%d  orange=%d  red=%d)",
            Path(dbg_path).name, len(glyph_ids), len(filtered_out), len(border_ids),
        )

    points = collect_glyph_pixels(glyph_ids, labels)
    if len(points) < 3:
        raise ValueError("Too few pixels to generate contour.")

    polygon = convex_hull_polygon(points) if mode == "convex" \
              else alpha_shape_contour(points, alpha_pct=alpha_pct)

    if margin > 0:
        polygon = expand_polygon(polygon, margin=margin)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, w_img - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, h_img - 1)

    logger.debug("polygon: %d vertices", len(polygon))

    # ── Annotated output ──────────────────────────────────────────────────────
    annotated = img_bgr.copy()
    pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(annotated, [pts], isClosed=True, color=draw_color, thickness=draw_thickness)

    if output_path is None:
        output_path = str(out_dir / (stem + "_bbox.png"))
    cv2.imwrite(output_path, annotated)

    # ── Crop ──────────────────────────────────────────────────────────────────
    x, y, w, h = cv2.boundingRect(pts)
    crop_path = str(out_dir / (stem + "_crop.png"))

    mask_poly = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_poly, [pts], 255)
    crop      = img_bgr[y:y+h, x:x+w].copy()
    mask_crop = mask_poly[y:y+h, x:x+w]
    bg        = np.full_like(crop, 255)
    crop      = np.where(mask_crop[:, :, np.newaxis] > 0, crop, bg)
    cv2.imwrite(crop_path, crop)

    # ── Masked image ──────────────────────────────────────────────────────────
    masked_path = str(out_dir / (stem + "_masked.png"))

    glyph_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for gid in glyph_ids:
        glyph_mask[labels == gid] = 255

    if margin > 0:
        k        = max(3, margin * 2 + 1)
        kernel_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        glyph_mask = cv2.dilate(glyph_mask, kernel_m, iterations=1)

    cx2, cy2  = w_img // 2, h_img // 2
    mx2, my2  = max(10, w_img // 6), max(10, h_img // 6)
    interior  = img_bgr[cy2-my2:cy2+my2, cx2-mx2:cx2+mx2]
    int_mask  = glyph_mask[cy2-my2:cy2+my2, cx2-mx2:cx2+mx2]
    bg_pixels = interior[int_mask == 0].reshape(-1, 3)
    if len(bg_pixels) == 0:
        bg_pixels = img_bgr.reshape(-1, 3)[glyph_mask.flatten() == 0]
    bg_color  = np.median(bg_pixels, axis=0).astype(np.uint8)
    logger.debug("background colour estimate: %s", bg_color.tolist())

    masked_img = img_bgr.copy()
    masked_img[glyph_mask == 0] = bg_color
    cv2.imwrite(masked_path, masked_img)

    result = {
        "polygon"  : polygon.tolist(),
        "bbox_rect": [x, y, w, h],
        "output"   : output_path,
        "crop"     : crop_path,
        "masked"   : masked_path,
    }

    logger.info("done — vertices=%d  rect=%s  masked → %s",
                len(polygon), result["bbox_rect"], Path(masked_path).name)
    logger.debug("bbox annotated → %s | crop → %s",
                 Path(output_path).name, Path(crop_path).name)

    return result