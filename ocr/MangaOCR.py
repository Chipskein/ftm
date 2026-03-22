import time
import re
import cv2
from manga_ocr import MangaOcr
import numpy as np

class MangaOCREngine:
    name = "MangaOCR"

    MIN_COVERAGE_RATIO = 0.15

    JUNK_PATTERNS = [
        r'^[\s\u3000]+$',                        # whitespace / ideographic space
        r'^[。、！？!?～〜ー・…「」『』（）\(\)]+$',  # pure punctuation
        r'^(.)\1{2,}$',                          # single char repeated 3+ times: ーーー
        r'^[A-Za-z0-9\s]+$',                     # pure latin / numbers
        r'^\W+$',                                # no word characters at all
    ]

    def __init__(self):
        self._model: MangaOcr | None = None

    def _load(self):
        if self._model is None:
            self._model = MangaOcr()

    def _is_junk(self, text: str) -> bool:
        """Return True if OCR output is noise with no real content."""
        t = text.strip()
        if len(t) <= 1:
            return True
        return any(re.fullmatch(p, t) for p in self.JUNK_PATTERNS)
    
    def _has_valid_polygon(self, polygon, w: int, h: int) -> bool:
        bbox_area = w * h
        if bbox_area == 0:
            return False
        poly_area = self._polygon_area(polygon)
        ratio = poly_area / bbox_area
        return ratio >= self.MIN_COVERAGE_RATIO

    @staticmethod
    def _has_japanese(text: str) -> bool:
        """Return True if text contains at least one Japanese character."""
        return bool(re.search(
            r'[\u3040-\u309F'   # Hiragana
            r'\u30A0-\u30FF'    # Katakana
            r'\u4E00-\u9FFF'    # CJK Kanji (common)
            r'\u3400-\u4DBF]',  # CJK Kanji (extended)
            text
        ))

    def extract(self, crop_path: str) -> str:
        t0 = time.perf_counter()
        try:
            self._load()
            text = self._model(crop_path).strip()
            
            if not self._has_japanese(text):
                return ""
            
            if self._is_junk(text):
                return ""
            
            return text
        except Exception as e:
            print(f"[MangaOCREngine] failed for {crop_path}: {e}")
            return ""