import os
import textwrap
from PIL import Image, ImageDraw, ImageFont
from ocr.types.BubbleZone import BubbleZone
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import re

class BubbleTypesetter:
    """
    Overwrites each detected bubble in the original image with translated text.

    If bubble["polygon"] is present:
      - Fills only inside the polygon shape with white (clips to balloon boundary)
      - Draws text centered inside the polygon bounding box

    Falls back to plain white rectangle fill when no polygon is available.
    """

    FILL_COLOR = (255, 255, 255)
    TEXT_COLOR = (0, 0, 0)
    PADDING    = 4       # px inside the text area
    SR_SCALE   = 2       # upscale factor used during refinement
    MIN_COVERAGE_RATIO = 0.05  # polygon must cover at least 5% of the bbox
    FONT_PATH = os.path.join("fonts", "KOMIKAX_.ttf")
    ONOMATOPOEIA_PATTERNS = [
        r'^[ァ-ン゛゜ーッ！？\s]+$',   # pure katakana (most JP onomatopoeia)
        r'^[A-Za-z]{2,}[!\?～〜]*$',   # romaji sound words: HA, BOOM, CRASH
        r'^[！？\?!～〜ー・\s]+$',       # pure punctuation/symbols only
    ]

    def _get_font(self, size: int):
        try:
            return ImageFont.truetype(self.FONT_PATH, size)
        except Exception as e:
            print(f"[WARN] Falha ao carregar fonte: {e}")
            return ImageFont.load_default()
        
    def _has_valid_polygon(self, polygon, w: int, h: int) -> bool:
        bbox_area = w * h
        if bbox_area == 0:
            return False
        poly_area = self._polygon_area(polygon)
        ratio = poly_area / bbox_area
        return ratio >= self.MIN_COVERAGE_RATIO

    def _get_region_brightness(self, img: Image.Image, x, y, w, h) -> float:
        """Returns 0.0 (black) to 255.0 (white) average brightness of a region."""
        crop = img.crop((x, y, x + w, y + h)).convert("L")
        pixels = list(crop.getdata())
        return sum(pixels) / len(pixels)
    
    def _blur_region(self, img: Image.Image, mask: Image.Image,
        tint_color: tuple = (255, 255, 255),
        blur_radius: int = 8, tint_alpha: int = 180) -> None:
        blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        img.paste(blurred, mask=mask)
        if tint_alpha > 0:
            scaled_mask = mask.point(lambda p: p * tint_alpha // 255)
            tint = Image.new("RGB", img.size, tint_color)
            img.paste(tint, mask=scaled_mask)

    def _sample_border_color(self, img: Image.Image, x, y, w, h, sample_px=6) -> tuple:
        """
        Samples pixels just outside the bubble bbox to get the true background color.
        Returns (r, g, b).
        """
        arr = np.array(img)
        H, W = arr.shape[:2]

        # Define a border ring just outside the bbox
        x0 = max(0,   x - sample_px)
        y0 = max(0,   y - sample_px)
        x1 = min(W,   x + w + sample_px)
        y1 = min(H,   y + h + sample_px)

        # Collect pixels in the outer ring, excluding the inner bbox
        samples = []
        for row in range(y0, y1):
            for col in range(x0, x1):
                if row < y or row > y + h or col < x or col > x + w:
                    samples.append(arr[row, col, :3])

        if not samples:
            return (255, 255, 255)

        avg = np.mean(samples, axis=0).astype(int)
        return (int(avg[0]), int(avg[1]), int(avg[2]))

    def _contrast_color(self, bg: tuple) -> tuple:
        """
        Returns black or white depending on which contrasts more with bg.
        Uses WCAG relative luminance formula.
        """
        r, g, b = [c / 255.0 for c in bg]
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return (0, 0, 0) if luminance > 0.5 else (255, 255, 255)

    def typeset(
        self,
        img_path: str,
        bubbles: list[BubbleZone],
        output_path: str,
    ) -> None:
        img = Image.open(img_path).convert("RGB")

        for bubble in bubbles:
            text = bubble.get("translated_text", "").strip()
            if not text:
                continue

            if self._is_onomatopoeia(text):
                continue

            x, y, w, h = bubble["x"], bubble["y"], bubble["w"], bubble["h"]
            polygon = bubble.get("polygon")

            if polygon:
                page_poly = self._to_page_coords(polygon, x, y, self.SR_SCALE)
                if not self._has_valid_polygon(page_poly, w, h):
                    continue
                self._draw_polygon_bubble(img, page_poly, x, y, w, h, text)
            else:
                self._draw_rect_bubble(img, x, y, w, h, text)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        img.save(output_path)
        print(f"[BubbleTypesetter] saved → {output_path}")

    # ------------------------------------------------------------------ #
    # Drawing                                                              #
    # ------------------------------------------------------------------ #

    def _draw_polygon_bubble(self, img, page_poly, x, y, w, h, text):
        bg_color  = self._sample_border_color(img, x, y, w, h)
        text_color = self._contrast_color(bg_color)

        mask = Image.new("L", img.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.polygon(page_poly, fill=255)
        mask_draw.rectangle([x, y, x + w, y + h], fill=255)

        self._blur_region(img, mask, bg_color)

        draw = ImageDraw.Draw(img)
        self._draw_text(draw, text, x, y, w, h, text_color)

    
    def _draw_rect_bubble(self, img, x, y, w, h, text):
        bg_color   = self._sample_border_color(img, x, y, w, h)
        text_color = self._contrast_color(bg_color)

        mask = Image.new("L", img.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([x, y, x + w, y + h], fill=255)

        self._blur_region(img, mask, bg_color)

        draw = ImageDraw.Draw(img)
        self._draw_text(draw, text, x, y, w, h, text_color)

    def _draw_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        rx: int, ry: int, rw: int, rh: int,
        text_color=None
    ) -> None:
        color = text_color if text_color is not None else self.TEXT_COLOR
        inner_w = rw - self.PADDING * 2
        inner_h = rh - self.PADDING * 2
        if inner_w <= 0 or inner_h <= 0:
            return

        font, wrapped = self._fit_text(text, inner_w, inner_h)

        bbox   = draw.textbbox((0, 0), wrapped, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        tx = rx + self.PADDING + max(0, (inner_w - text_w) // 2)
        ty = ry + self.PADDING + max(0, (inner_h - text_h) // 2)

        draw.text((tx, ty), wrapped, font=font, fill=color)

    # ------------------------------------------------------------------ #
    # Coordinate transform                                                 #
    # ------------------------------------------------------------------ #

    def _is_onomatopoeia(self, text: str) -> bool:
        """Return True if text looks like an onomatopoeia (skip typesetting)."""
        t = text.strip()
        return any(re.fullmatch(p, t) for p in self.ONOMATOPOEIA_PATTERNS)

    @staticmethod
    def _to_page_coords(
        polygon: list[list[int]],
        bubble_x: int,
        bubble_y: int,
        sr_scale: int,
    ) -> list[tuple[int, int]]:
        """
        Convert crop-local upscaled polygon coords → page image coords.

          page_x = bubble_x + (poly_x / sr_scale)
          page_y = bubble_y + (poly_y / sr_scale)
        """
        return [
            (
                int(bubble_x + pt[0] / sr_scale),
                int(bubble_y + pt[1] / sr_scale),
            )
            for pt in polygon
        ]

    # ------------------------------------------------------------------ #
    # Font fitting                                                         #
    # ------------------------------------------------------------------ #

    def _fit_text(
        self,
        text: str,
        max_w: int,
        max_h: int,
        min_size: int = 6,
        max_size: int = 40,
    ):
        dummy = Image.new("RGB", (1, 1))
        draw  = ImageDraw.Draw(dummy)

        for size in range(max_size, min_size - 1, -1):
            font = self._get_font(size)

            # estima largura média
            avg_char_w = max(1, int(self._avg_char_width(draw, font) * 1.1))

            # testa diferentes quebras de linha
            for chars_per_line in range(
                max(1, max_w // avg_char_w),
                0,
                -1
            ):
                wrapped = textwrap.fill(
                    text,
                    width=max(chars_per_line, 1),
                    break_long_words=False,
                    break_on_hyphens=True,
                )

                bbox = draw.textbbox((0, 0), wrapped, font=font, spacing=2)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]

                if text_w <= max_w and text_h <= max_h:
                    return font, wrapped

        # fallback extremo
        font = self._get_font(min_size)
        wrapped = textwrap.fill(text, width=10, break_long_words=False, break_on_hyphens=True)
        return font, wrapped

    @staticmethod
    def _avg_char_width(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
        sample = "abcdefghijklmnopqrstuvwxyz"
        bbox   = draw.textbbox((0, 0), sample, font=font)
        return (bbox[2] - bbox[0]) // len(sample)

    @staticmethod
    def _polygon_area(poly: list[tuple[int, int]]) -> float:
        """Shoelace formula."""
        n = len(poly)
        area = 0.0
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0