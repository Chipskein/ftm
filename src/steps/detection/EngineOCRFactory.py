from typing import Optional
from ...dto.BubbleZone import BubbleZone
from .EngineOCR import EngineOCR
from ...profiler.decorator import step
from ...assets import model

class EngineOCRFactory:
    def __init__(
        self, 
        engine_name: str, 
        debug: bool = False, 
        use_cpu: bool = False
    ):
        self.debug = debug
        self.use_cpu = use_cpu
        
        self.engine: EngineOCR = self._setup_engine(engine_name)

    @step("EngineOCRFactory._setup_engine")
    def _setup_engine(self, name: str) -> EngineOCR:
        match name:
            case "yolo":
                from .YOLOTextDetector import YOLOTextDetector
                return YOLOTextDetector(
                    model("yolo_text_detector_1024_95pbt_5pbv.pt"), 
                    debug=self.debug, 
                    use_cpu=self.use_cpu
                )
            case _:
                raise ValueError(f"Engine '{name}' is not supported.")

    @step("EngineOCRFactory._run")
    def run(self, image_path: str, tmp_dir: str) -> list[BubbleZone]:
        return self.engine.run(image_path, tmp_dir)