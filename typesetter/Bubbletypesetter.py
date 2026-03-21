import os
import textwrap

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from ocr.types.BubbleZone import BubbleZone


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
    FONT_PATH = os.path.join("fonts", "KOMIKAX_.ttf")

    def _get_font(self, size: int):
        try:
            return ImageFont.truetype(self.FONT_PATH, size)
        except Exception as e:
            print(f"[WARN] Falha ao carregar fonte: {e}")
            return ImageFont.load_default()

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

            x, y, w, h = bubble["x"], bubble["y"], bubble["w"], bubble["h"]
            polygon = bubble.get("polygon")

            if polygon:
                page_poly = self._to_page_coords(polygon, x, y, self.SR_SCALE)
                self._draw_polygon_bubble(img, page_poly, x, y, w, h, text)
            else:
                self._draw_rect_bubble(img, x, y, w, h, text)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        img.save(output_path)
        print(f"[BubbleTypesetter] saved → {output_path}")

    # ------------------------------------------------------------------ #
    # Drawing                                                              #
    # ------------------------------------------------------------------ #

    def _draw_polygon_bubble(
        self,
        img: Image.Image,
        page_poly: list[tuple[int, int]],
        x: int, y: int, w: int, h: int,
        text: str,
    ) -> None:
        """Fill polygon + original bbox, then draw text using bbox dims."""
        draw = ImageDraw.Draw(img)
        draw.polygon(page_poly, fill=self.FILL_COLOR)
        # Use the original bbox (x, y, w, h) for both erasing and text layout.
        # The polygon page-bbox is narrower than the original bbox which causes
        # text overflow; the original bbox correctly constrains line width (w)
        # while providing full height (h) for multi-line wrapping.
        draw.rectangle([x, y, x + w, y + h], fill=self.FILL_COLOR)
        self._draw_text(draw, text, x, y, w, h)

    def _draw_rect_bubble(
        self,
        img: Image.Image,
        x: int, y: int, w: int, h: int,
        text: str,
    ) -> None:
        draw = ImageDraw.Draw(img)
        draw.rectangle([x, y, x + w, y + h], fill=self.FILL_COLOR)
        self._draw_text(draw, text, x, y, w, h)

    def _draw_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        rx: int, ry: int, rw: int, rh: int,
    ) -> None:
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

        draw.text((tx, ty), wrapped, font=font, fill=self.TEXT_COLOR)

    # ------------------------------------------------------------------ #
    # Coordinate transform                                                 #
    # ------------------------------------------------------------------ #

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