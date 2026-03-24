from ocr.types.BubbleZone import BubbleZone
from ocr.EngineOCR import EngineOCR
from utils.resource import ResourceMonitor

SUPPORTED_ENGINES = ("easy", "tesseract","paddle")

def build_engine(
    engine_name: str,
    debug: bool = False,
    monitor: ResourceMonitor | None = None,
) -> EngineOCR:
    
    from ocr.MagiPanelDetector import load_magi,MagiPanelDetector
    model = load_magi()
    panel_detector = MagiPanelDetector(magi_model=model)

    match engine_name:
        case "easy":
            from ocr.EazyOcr import EazyOCR
            return EazyOCR(panel_detector, debug, monitor)
        
        case "tesseract":
            from ocr.TesseractOCR import TesseractOCR
            return TesseractOCR(panel_detector, debug, monitor)
        
        case "paddle":
            from ocr.PaddleOCREngine import PaddleOCREngine
            return PaddleOCREngine(panel_detector, debug, monitor)
        
        case _:
            raise ValueError(
                f"Unsupported OCR engine: {engine_name!r}. "
                f"Choose from: {SUPPORTED_ENGINES}"
            )


def detect(
    image_path: str,
    engine_name: str,
    debug: bool,
    tmp_dir: str,
    monitor: ResourceMonitor | None = None,
) -> list[BubbleZone]:
    engine = build_engine(engine_name, debug, monitor)
    return engine.run(image_path, tmp_dir)