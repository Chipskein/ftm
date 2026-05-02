from abc import ABC, abstractmethod
import re
import logging

logger = logging.getLogger(__name__)

class TextExtractor(ABC):
    _JUNK_RE = re.compile('|'.join([
        r'[\s\u3000]+',
        r'[。、！？!?～〜ー・…「」『』（）\(\)]+',
        r'(.)\1{2,}',
        r'[A-Za-z0-9\s]+',
        r'\W+',
        r'そういえば、'
    ]))

    _JAPANESE_RE = re.compile(
        r'[\u3040-\u309F'
        r'\u30A0-\u30FF'
        r'\u4E00-\u9FFF'
        r'\u3400-\u4DBF]'
    )

    def _is_junk(self, text: str) -> bool:
        t = text.strip()
        result = len(t) <= 1 or bool(self._JUNK_RE.fullmatch(t))
        if self.debug:
            logger.debug(f"Checking if text is junk: {t!r}")
            logger.debug(f"Is junk: {result}")
        return result
    
    def _has_japanese(self, text: str) -> bool:
        result = bool(self._JAPANESE_RE.search(text))
        if self.debug:
            logger.debug(f"Checking if text has Japanese: {text!r}")
            logger.debug(f"Has Japanese: {result}")
        return result

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def extract(self, crop_path: str) -> str:
        pass