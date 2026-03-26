import os
import textwrap
import numpy as np
import re
from PIL import Image, ImageDraw, ImageFont, ImageFilter
# Assuming BubbleZone is imported correctly from your environment
# from ocr.types.BubbleZone import BubbleZone 

class BubbleTypesetter:
    """
    Overwrites each detected bubble in the original image with translated text.
    """

    FILL_COLOR = (255, 255, 255)
    TEXT_COLOR = (0, 0, 0)
    PADDING    = 4       
    SR_SCALE   = 1       
    MIN_COVERAGE_RATIO = 0.05  
    DEFAULT_FONT_PATH = os.path.join("fonts", "KOMIKAX_.ttf")
    ONOMATOPOEIA_PATTERNS = [
        r'^[ァ-ン゛゜ーッ！？\s]+$',
        r'^[！？\?!～〜ー・\s]+$', 
    ]

    def __init__(self, font_path: str = None):
        """
        Initialize the typesetter.
        :param font_path: Optional path to a .ttf or .otf font file.
        """
        self.font_path = font_path if font_path else self.DEFAULT_FONT_PATH
        print(f"[DEBUG] BubbleTypesetter initialized with font: {self.font_path}")

    def _get_font(self, size: int):
        try:
            return ImageFont.truetype(self.font_path, size)
        except Exception as e:
            if self.font_path != self.DEFAULT_FONT_PATH:
                 try:
                     return ImageFont.truetype(self.DEFAULT_FONT_PATH, size)
                 except:
                     pass
            print(f"[WARN] Falha ao carregar fonte {self.font_path}: {e}")
            return ImageFont.load_default()
        
    def _has_valid_polygon(self, polygon, w: int, h: int) -> bool:
        bbox_area = w * h
        if bbox_area == 0:
            return False
        poly_area = self._polygon_area(polygon)
        ratio = poly_area / bbox_area
        is_valid = ratio >= self.MIN_COVERAGE_RATIO
        if not is_valid:
            print(f"[DEBUG] Polygon rejected: ratio {ratio:.2f} < {self.MIN_COVERAGE_RATIO}")
        return is_valid

    def _get_region_brightness(self, img: Image.Image, x, y, w, h) -> float:
        crop = img.crop((x, y, x + w, y + h)).convert("L")
        pixels = list(crop.getdata())
        brightness = sum(pixels) / len(pixels)
        return brightness
    
    def _blur_region(self, img: Image.Image, mask: Image.Image,
        tint_color: tuple = (255, 255, 255),
        blur_radius: int = 8, tint_alpha: int = 180) -> None:
        print(f"[DEBUG] Applying blur (r={blur_radius}) and tint {tint_color} with alpha {tint_alpha}")
        blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        img.paste(blurred, mask=mask)
        if tint_alpha > 0:
            scaled_mask = mask.point(lambda p: p * tint_alpha // 255)
            tint = Image.new("RGB", img.size, tint_color)
            img.paste(tint, mask=scaled_mask)

    def _sample_interior_color(self, img: Image.Image, x: int, y: int,w: int,h: int, padding: int = 10) -> tuple:
        """
        Samples a pixel inside the bubble, offset from the top-left corner 
        to avoid the black border/stroke.
        """
        width, height = img.size

        # Target coordinate: move 'padding' pixels down and right
        sample_x = max(0, min(x + padding, width - 1))
        sample_y = max(0, min(y + padding, height - 1))
        
        # Extract RGB
        pixel = img.getpixel((sample_x, sample_y))
        
        # Handle grayscale or RGBA safely
        if isinstance(pixel, int):
            return (pixel, pixel, pixel)
        return pixel[:3]

    def _sample_border_color(self, img: Image.Image, x, y, w, h, sample_px=6) -> tuple:
        arr = np.array(img)
        H, W = arr.shape[:2]
        x0, y0 = max(0, x - sample_px), max(0, y - sample_px)
        x1, y1 = min(W, x + w + sample_px), min(H, y + h + sample_px)

        samples = []
        for row in range(y0, y1):
            for col in range(x0, x1):
                if row < y or row > y + h or col < x or col > x + w:
                    samples.append(arr[row, col, :3])

        if not samples:
            print("[DEBUG] No border samples found, defaulting to white")
            return (255, 255, 255)

        avg = np.mean(samples, axis=0).astype(int)
        color = (int(avg[0]), int(avg[1]), int(avg[2]))
        print(f"[DEBUG] Sampled border color: {color} at ({x},{y})")
        return color

    def _contrast_color(self, bg: tuple) -> tuple:
        r, g, b = [c / 255.0 for c in bg]
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        contrast = (0, 0, 0) if luminance > 0.5 else (255, 255, 255)
        return contrast

    def typeset(
        self,
        img_path: str,
        bubbles: list,
        output_path: str,
    ) -> None:
        print(f"[DEBUG] Starting typesetting for: {img_path}")
        img = Image.open(img_path).convert("RGB")

        for i, bubble in enumerate(bubbles):
            text = bubble.get("translated_text", "").strip()
            if not text:
                continue

            if self._is_onomatopoeia(text):
                print(f"[DEBUG] Skipping Bubble {i}: Detected as onomatopoeia ('{text[:10]}...')")
                continue

            x, y, w, h = bubble["x"], bubble["y"], bubble["w"], bubble["h"]
            polygon = bubble.get("polygon")
            
            print(f"[DEBUG] Processing Bubble {i}: bbox=[{x}, {y}, {w}, {h}] text='{text[:20]}...'")

            if polygon:
                page_poly = self._to_page_coords(polygon, x, y, self.SR_SCALE)
                if not self._has_valid_polygon(page_poly, w, h):
                    print(f"[DEBUG] Bubble {i}: Invalid polygon area ratio, skipping.")
                    continue
                self._draw_polygon_bubble(img, page_poly, x, y, w, h, text)
            else:
                self._draw_rect_bubble(img, x, y, w, h, text)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        img.save(output_path)
        print(f"[BubbleTypesetter] saved → {output_path}")

    def _draw_polygon_bubble(self, img, page_poly, x, y, w, h, text):
        bg_color  = self._sample_interior_color(img, x, y, w, h)
        text_color = self._contrast_color(bg_color)

        mask = Image.new("L", img.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.polygon(page_poly, fill=255)
        # We also fill the bbox to ensure internal coverage
        mask_draw.rectangle([x, y, x + w, y + h], fill=255)

        self._blur_region(img, mask, bg_color)
        draw = ImageDraw.Draw(img)
        self._draw_text(draw, text, x, y, w, h, text_color)

    def _draw_rect_bubble(self, img, x, y, w, h, text):
        bg_color   = self._sample_interior_color(img, x, y, w, h)
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
            print(f"[DEBUG] Text area too small: {inner_w}x{inner_h}")
            return

        font, wrapped = self._fit_text(text, inner_w, inner_h)

        bbox   = draw.textbbox((0, 0), wrapped, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        tx = rx + self.PADDING + max(0, (inner_w - text_w) // 2)
        ty = ry + self.PADDING + max(0, (inner_h - text_h) // 2)

        draw.text((tx, ty), wrapped, font=font, fill=color)

    def _is_onomatopoeia(self, text: str) -> bool:
        t = text.strip()
        return any(re.fullmatch(p, t) for p in self.ONOMATOPOEIA_PATTERNS)

    @staticmethod
    def _to_page_coords(polygon, bubble_x, bubble_y, sr_scale):
        return [
            (int(bubble_x + pt[0] / sr_scale), int(bubble_y + pt[1] / sr_scale))
            for pt in polygon
        ]

    def _fit_text(self, text, max_w, max_h, min_size=6, max_size=40):
        dummy = Image.new("RGB", (1, 1))
        draw  = ImageDraw.Draw(dummy)

        for size in range(max_size, min_size - 1, -1):
            font = self._get_font(size)
            avg_char_w = max(1, int(self._avg_char_width(draw, font) * 1.1))

            for chars_per_line in range(max(1, max_w // avg_char_w), 0, -1):
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
                    print(f"[DEBUG] Text fit found: size {size}, {chars_per_line} chars/line")
                    return font, wrapped

        print(f"[DEBUG] No ideal fit found for '{text[:10]}...', using min_size {min_size}")
        font = self._get_font(min_size)
        wrapped = textwrap.fill(text, width=10, break_long_words=False, break_on_hyphens=True)
        return font, wrapped

    @staticmethod
    def _avg_char_width(draw, font):
        sample = "abcdefghijklmnopqrstuvwxyz"
        bbox = draw.textbbox((0, 0), sample, font=font)
        return (bbox[2] - bbox[0]) // len(sample)

    @staticmethod
    def _polygon_area(poly) -> float:
        n = len(poly)
        area = 0.0
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0