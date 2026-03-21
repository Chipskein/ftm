import logging
import os

from PIL import Image
from super_image import EdsrModel, ImageLoader

from utils.bubble import generate_glyph_bbox
from .types.BubbleZone import BubbleZone

logger = logging.getLogger(__name__)


class BubbleRefiner:
    """
    Refines raw bubble crops produced by EazyOCR in two steps:

      1. Upscale  — EDSR 2× super-resolution improves glyph definition
                    before polygon detection
      2. Denoise  — manga_bbox generates a tight glyph polygon and
                    replaces the background with a solid colour, giving
                    a clean masked image ready for any OCR engine

    Updates each BubbleZone:
      - bubble["crop"]    → path to the masked image
      - bubble["polygon"] → [[x, y], ...] polygon in crop-local coordinates
    """

    def __init__(
        self,
        mode: str = "concave",
        margin: int = 4,
        alpha_pct: float = 0.05,
        scale: int = 2,
        debug: bool = False,
    ):
        self.mode = mode
        self.margin = margin
        self.alpha_pct = alpha_pct
        self.debug = debug
        self.scale = scale

        logger.info("loading EDSR model (scale=%dx)...", scale)
        self._sr_model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=scale)
        logger.info("EDSR model ready")

        if debug:
            logger.setLevel(logging.DEBUG)

    def refine(self, bubbles: list[BubbleZone]) -> list[BubbleZone]:
        """
        Refine crops in-place: upscale → manga_bbox mask.
        Bubbles whose crop file is missing or whose processing fails are left unchanged.
        """
        total = len(bubbles)
        skipped = 0
        failed = 0

        logger.info("refining %d bubble(s)...", total)

        for i, bubble in enumerate(bubbles, 1):
            crop_path = bubble["crop"]
            logger.debug("[%d/%d] processing %s", i, total, os.path.basename(crop_path))

            if not os.path.exists(crop_path):
                logger.warning("[%d/%d] crop not found, skipping: %s", i, total, crop_path)
                skipped += 1
                continue

            upscaled_path = self._upscale_crop(crop_path)
            result = self._refine_crop(upscaled_path or crop_path)

            if result is not None:
                bubble["crop"]    = result["masked"]
                bubble["polygon"] = result["polygon"]
                logger.debug("[%d/%d] ✓ masked → %s", i, total, os.path.basename(result["masked"]))
            else:
                failed += 1
                logger.warning("[%d/%d] refinement failed, keeping original crop", i, total)

        logger.info(
            "refine done — ok=%d  skipped=%d  failed=%d",
            total - skipped - failed, skipped, failed,
        )
        return bubbles

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _upscale_crop(self, crop_path: str) -> str | None:
        """EDSR super-resolution on a single crop. Returns upscaled path or None."""
        try:
            name, ext = os.path.splitext(crop_path)
            out_path = f"{name}_{self.scale}x{ext}"
            img = Image.open(crop_path)
            inputs = ImageLoader.load_image(img)
            preds = self._sr_model(inputs)
            ImageLoader.save_image(preds, out_path)
            logger.debug("upscaled → %s", os.path.basename(out_path))
            return out_path
        except Exception as e:
            logger.error("upscale failed for %s: %s", os.path.basename(crop_path), e)
            return None

    def _refine_crop(self, crop_path: str) -> dict | None:
        """Run generate_glyph_bbox on a single crop. Returns result dict or None."""
        try:
            return generate_glyph_bbox(
                image_path=crop_path,
                mode=self.mode,
                margin=self.margin,
                alpha_pct=self.alpha_pct,
                debug=self.debug,
            )
        except (ValueError, FileNotFoundError) as e:
            logger.error("manga_bbox failed for %s: %s", os.path.basename(crop_path), e)
            return None