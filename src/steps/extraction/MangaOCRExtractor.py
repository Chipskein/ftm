import logging
from manga_ocr import MangaOcr

from ...profiler.decorator import step
from .Extractor import TextExtractor

logger = logging.getLogger(__name__)

class MangaOCRExtractor(TextExtractor):
    def __init__(
        self,
        use_cpu: bool = False,
        debug: bool = False
    ):
        super().__init__("MangaOCR")
        logger.info("Initializing MangaOCR model...")
        self.use_cpu = use_cpu
        self._model_instance = MangaOcr(force_cpu=self.use_cpu)
        self.debug = debug

    @step(lambda self, crop_path, *_: f"MangaOCRExtractor.extract from {crop_path}")
    def extract(self, crop_path: str) -> str:
        try:
            text = self._model_instance(crop_path).strip()
            if not text or not self._has_japanese(text) or self._is_junk(text):
                logger.warning(f"OCR result is empty, non-Japanese, or junk for {crop_path}: {text!r}")
                return ""

            return text
        except Exception as e:
            logger.error(f"OCR failed for {crop_path}: {e}", exc_info=True)
            return ""