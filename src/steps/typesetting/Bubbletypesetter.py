import os
import logging
from functools import lru_cache
import time
from typing import List, Tuple, Dict, Optional, Any
from ...profiler.ResourceMonitor import ResourceMonitor
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from ...assets import font

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BubbleTypesetter")

class BubbleTypesetter:
    DEFAULT_FONT_PATH = font("KOMIKAX_.ttf")
    PADDING = 2
    MAX_FONT_SIZE = 120
    MIN_FONT_SIZE = 6

    def __init__(
        self, 
        font_path: Optional[str] = None,
        monitor: ResourceMonitor | None = None
    ):
        self.monitor = monitor
        self.font_path = font_path or self.DEFAULT_FONT_PATH
        self._dummy_draw = ImageDraw.Draw(Image.new("L", (1, 1)))
        
        if self.monitor:
            self.monitor.set_label("Init Typesetter")

    @lru_cache(maxsize=256)
    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        if self.monitor:
            self.monitor.set_label("BubbleTypesetter._get_font")

        try:
            return ImageFont.truetype(self.font_path, size)
        except (OSError, AttributeError):
            logger.warning(f"Font {self.font_path} not found, falling back.")
            return ImageFont.load_default()

    def typeset(self, img_path: str, bubbles: List[Dict[str, Any]], output_path: str) -> List[Dict[str, Any]]:
        if self.monitor:
            self.monitor.set_label("BubbleTypesetter._typeset")

        if not os.path.exists(img_path):
            logger.error(f"Input image not found: {img_path}")
            return

        try:
            with Image.open(img_path) as img:
                canvas = img.convert("RGB")
                for i, bubble in enumerate(bubbles):
                    start_time = time.perf_counter()
                    self._process_bubble(canvas, bubble)
                    elapsed = time.perf_counter() - start_time
                    bubble["typesetting_characters"] = len(bubble["translated_text"])
                    bubble["typesetting_time_s"] = elapsed

                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                canvas.save(output_path, quality=95)
                logger.info(f"Successfully saved typeset image to {output_path}")
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
        
        return bubbles

    def _process_bubble(self, canvas: Image.Image, bubble: Dict[str, Any]) -> None:
        if self.monitor:
            self.monitor.set_label("BubbleTypesetter._process_bubble")

        text = bubble.get("translated_text", "").strip()
        if not text:
            return

        x, y, w, h = int(bubble["x"]), int(bubble["y"]), int(bubble["w"]), int(bubble["h"])
        bbox = (x, y, x + w, y + h)

        bg_color = self._sample_color(canvas, x, y, w, h)
        
        region_mask = Image.new("L", canvas.size, 0)
        ImageDraw.Draw(region_mask).rectangle(bbox, fill=255)
        self._apply_cleaning(canvas, region_mask, bg_color)

        text_color = self._get_contrast_color(bg_color)
        draw = ImageDraw.Draw(canvas)
        self._draw_centered_text(draw, text, (x, y, w, h), text_color)

    def _apply_cleaning(self, canvas: Image.Image, mask: Image.Image, color: Tuple[int, int, int]):
        if self.monitor:
            self.monitor.set_label("BubbleTypesetter._apply_cleaning")

        blurred = canvas.filter(ImageFilter.GaussianBlur(radius=6))
        canvas.paste(blurred, mask=mask)
        
        tint_layer = Image.new("RGB", canvas.size, color)
        tint_mask = mask.point(lambda p: p * 200 // 255)
        canvas.paste(tint_layer, mask=tint_mask)

    def _draw_centered_text(self, draw: ImageDraw.ImageDraw, text: str, rect: Tuple[int, int, int, int], color: Tuple):
        if self.monitor:
            self.monitor.set_label("BubbleTypesetter._draw_centered_text")

        x, y, w, h = rect
        inner_w, inner_h = w - (self.PADDING * 2), h - (self.PADDING * 2)
        
        if inner_w <= 0 or inner_h <= 0:
            return

        font, wrapped_text = self._fit_text(text, inner_w, inner_h)
        
        line_spacing = 4 if h > w * 1.5 else 2

        t_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center", spacing=line_spacing)
        tw, th = t_bbox[2] - t_bbox[0], t_bbox[3] - t_bbox[1]
        
        tx = x + (w - tw) // 2
        ty = y + (h - th) // 2
        
        draw.multiline_text((tx, ty), wrapped_text, font=font, fill=color, align="center", spacing=line_spacing)

    def _fit_text(self, text: str, max_w: int, max_h: int) -> Tuple[ImageFont.FreeTypeFont, str]:
        if self.monitor:
            self.monitor.set_label("BubbleTypesetter._fit_text")
        
        """Encontra o equilíbrio entre tamanho de fonte e quebra de linha."""
        for size in range(self.MAX_FONT_SIZE, self.MIN_FONT_SIZE - 1, -1):
            font = self._get_font(size)
            wrapped = self._wrap_text_pixel_width(text, font, max_w)
            
            bbox = self._dummy_draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=2)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            if w <= max_w and h <= max_h:
                return font, wrapped
        
        min_font = self._get_font(self.MIN_FONT_SIZE)
        return min_font, self._wrap_text_pixel_width(text, min_font, max_w)

    def _wrap_text_pixel_width(self, text: str, font: ImageFont.FreeTypeFont, max_px_width: int) -> str:
        if self.monitor:
            self.monitor.set_label("BubbleTypesetter._wrap_text_pixel_width")
        
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            test_line = " ".join(current_line + [word]) if current_line else word
            
            if font.getlength(test_line) <= max_px_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    remainder = word
                    while remainder:
                        chunk = ""
                        for char in remainder:
                            if font.getlength(chunk + char) <= max_px_width:
                                chunk += char
                            else:
                                break
                        if not chunk: chunk = remainder[0]
                        lines.append(chunk)
                        remainder = remainder[len(chunk):]
                    current_line = []

        if current_line:
            lines.append(" ".join(current_line))
            
        return "\n".join(lines)

    @staticmethod
    def _sample_color(img: Image.Image, x: int, y: int, w: int, h: int) -> Tuple[int, int, int]:
        img_w, img_h = img.size
        box = (max(0, x), max(0, y), min(img_w, x + w), min(img_h, y + h))
        
        if box[2] <= box[0] or box[3] <= box[1]:
            return (255, 255, 255)

        region = img.crop(box).convert("RGB")
        colors = region.getcolors(maxcolors=1024)

        if not colors:
            # Fallback
            region = region.quantize(colors=256).convert("RGB")
            colors = region.getcolors(maxcolors=256)

        return max(colors, key=lambda item: item[0])[1]

    @staticmethod
    def _get_contrast_color(bg_rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        r, g, b = [v / 255.0 for v in bg_rgb]
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return (0, 0, 0) if lum > 0.5 else (255, 255, 255)