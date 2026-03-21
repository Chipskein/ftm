import time
from manga_ocr import MangaOcr


class MangaOCREngine:
    name = "MangaOCR"

    def __init__(self):
        self._model: MangaOcr | None = None

    def _load(self):
        if self._model is None:
            self._model = MangaOcr()

    def extract(self, crop_path: str) -> str:
        """
        Extract Japanese text from a crop image file.

        Args:
            crop_path: path to the image file (PNG/JPG)

        Returns:
            Extracted text string, or empty string on failure.
        """
        t0 = time.perf_counter()
        try:
            self._load()
            text = self._model(crop_path)
            return text.strip()
        except Exception as e:
            print(f"[MangaOCREngine] failed for {crop_path}: {e}")
            return ""