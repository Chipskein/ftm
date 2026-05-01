from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class Translator(ABC):
    
    @abstractmethod
    def translate(self, text: str, lang: str) -> str:
        pass