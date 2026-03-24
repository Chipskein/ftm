from abc import ABC, abstractmethod
import numpy as np

class PanelDetector(ABC):
    """
    Abstract interface for manga panel detectors.

    All implementations must return panels as list[tuple[x, y, w, h]],
    which is the format expected by EazyOCR._find_panel_dividers.
    """

    @abstractmethod
    def _find_panel_dividers(self, img: np.ndarray) -> list[tuple]:
        """
        Detect panels in a BGR numpy image.

        Returns:
            List of (x, y, w, h) tuples, one per detected panel.
        """

    def _contains(self, outer: tuple, inner: tuple, margin: int = 0) -> bool:
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner
        return (
            ox - margin <= ix
            and oy - margin <= iy
            and ox + ow + margin >= ix + iw
            and oy + oh + margin >= iy + ih
        )
