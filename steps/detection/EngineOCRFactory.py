from typing import Optional
from dto.BubbleZone import BubbleZone
from .EngineOCR import EngineOCR
from profiler.ResourceMonitor import ResourceMonitor

class EngineOCRFactory:
    def __init__(
        self, 
        engine_name: str, 
        debug: bool = False, 
        use_cpu: bool = False,
        monitor: Optional[ResourceMonitor] = None
    ):
        self.debug = debug
        self.use_cpu = use_cpu
        self.monitor = monitor
        self._panel_detector = self._setup_panel_detector()
        self.engine: EngineOCR = self._setup_engine(engine_name)

    def _setup_panel_detector(self):
        from .panel.YOLOPanelDetector import YOLOPanelDetector
        return YOLOPanelDetector("models/panel_detector_model.pt", use_cpu = self.use_cpu)

    def _setup_engine(self, name: str) -> EngineOCR:
        match name:
            case "easy":
                from .EazyOcr import EazyOCR
                return EazyOCR(self._panel_detector, self.debug, self.monitor, self.use_cpu)
            
            #TODO: Really bad results should drop support for tesseract and paddle
            #case "tesseract":
            #    from .TesseractOCR import TesseractOCR
            #    return TesseractOCR(panel_detector, debug, monitor,use_cpu)
            
            #case "paddle":
            #    from .PaddleOCREngine import PaddleOCREngine
            #    return PaddleOCREngine(panel_detector, debug, monitor,use_cpu)
            
            case "yolo":
                from .YOLOTextDetector import YOLOTextDetector
                return YOLOTextDetector(
                    "models/yolo_text_detector.pt", 
                    panel_detector=self._panel_detector, 
                    debug=self.debug, 
                    monitor=self.monitor, 
                    use_cpu=self.use_cpu
                )
            case _:
                raise ValueError(f"Engine '{name}' is not supported.")

    def run(self, image_path: str, tmp_dir: str) -> list[BubbleZone]:
        return self.engine.run(image_path, tmp_dir)