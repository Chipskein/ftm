from ocr.types.BubbleZone import BubbleZone
from ocr.EngineOCR import EngineOCR
from utils.resource import ResourceMonitor

SUPPORTED_ENGINES = ("easy", "tesseract","paddle","yolo")

def build_engine(
    engine_name: str,
    debug: bool = False,
    monitor: ResourceMonitor | None = None,
    use_cpu: bool = False,
) -> EngineOCR:
    
    from ocr.YOLOPanelDetector import YOLOPanelDetector
    panel_detector = YOLOPanelDetector("models/panel_detector_model.pt",use_cpu=use_cpu)

    match engine_name:
        case "easy":
            from ocr.EazyOcr import EazyOCR
            return EazyOCR(panel_detector, debug, monitor,use_cpu)
        
        #Really bad results should drop support for tesseract and paddle
        #case "tesseract":
        #    from ocr.TesseractOCR import TesseractOCR
        #    return TesseractOCR(panel_detector, debug, monitor,use_cpu)
        
        #case "paddle":
        #    from ocr.PaddleOCREngine import PaddleOCREngine
        #    return PaddleOCREngine(panel_detector, debug, monitor,use_cpu)
        
        case "yolo":
            from ocr.YOLOTextDetector import YOLOTextDetector
            return YOLOTextDetector(
                "models/yolo_text_detector.pt", 
                panel_detector,
                debug=debug,
                monitor=monitor,
                use_cpu=use_cpu
            )
        
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
    use_cpu: bool = False,
) -> list[BubbleZone]:
    engine = build_engine(engine_name, debug, monitor,use_cpu)
    return engine.run(image_path, tmp_dir)