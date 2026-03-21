from abc import ABC, abstractmethod
import cv2


class EngineOCR(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def loadImage(self, image_path: str) -> cv2.typing.MatLike:
        pass

    @abstractmethod
    def preProcessImage(self, image: cv2.typing.MatLike) -> dict:
        pass

    @abstractmethod
    def run(self, img_path: str, output_dir: str) -> list:
        pass