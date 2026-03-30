import logging
import os
import tempfile

from .panel.PanelDetector import PanelDetector
from profiler.ResourceMonitor import ResourceMonitor

# Disable oneDNN/MKL-DNN before ANY paddle import.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_DISABLE_ONEDNN"] = "1"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_new_executor_use_interpretermcore"] = "0"

import cv2
import numpy as np
import torch

try:
    import paddle
    paddle.set_flags({"FLAGS_use_mkldnn": False})
except Exception:
    pass

from paddleocr import PaddleOCR, draw_ocr
from PIL import Image as PILImage
from .EngineOCR import EngineOCR
from dto.BubbleZone import BubbleZone

logger = logging.getLogger(__name__)

_PANEL_MIN_AREA       = 0.02
_TILE_OVERLAP         = 0.20
_TILE_SPLIT_HEIGHT    = 800
_IOU_MERGE_THRESHOLD  = 0.40
_GAP_MIN_HEIGHT       = 60


class PaddleOCREngine(EngineOCR):
    def __init__(
        self,
        panel_detector: PanelDetector,
        debug: bool = False,
        monitor: ResourceMonitor | None = None,
    ):
        super().__init__("PaddleOCR")
        self.panel_detector = panel_detector
        self.debug = debug
        self.monitor = monitor

        if debug:
            logger.setLevel(logging.DEBUG)

        logger.info("loading PaddleOCR (lang=japan)...")
        self._ocr = self._init_paddle()
        logger.info("PaddleOCR ready  gpu=%s", torch.cuda.is_available())

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, img_path: str, output_dir: str) -> list[BubbleZone]:
        if not os.path.exists(img_path):
            logger.error("image not found: %s", img_path)
            return []

        img = self.loadImage(img_path)
        h, w = img.shape[:2]
        logger.info("processing image %s (%dx%d)", os.path.basename(img_path), w, h)

        split_x = self.detect_spread_split(img)
        if split_x is not None:
            logger.debug("spread detected — split at x=%d", split_x)
            left  = img[:, :split_x]
            right = img[:, split_x:]

            results_left  = self._run_single(left,  img_path, output_dir, suffix="_L")
            results_right = self._run_single(
                right, img_path, output_dir,
                suffix="_R", x_offset=split_x,
                id_offset=len(results_left),
            )
            results = results_left + results_right
            logger.info(
                "spread — left=%d  right=%d  total=%d zones",
                len(results_left), len(results_right), len(results),
            )
            return results

        return self._run_single(img, img_path, output_dir)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _run_single(
        self,
        img: cv2.typing.MatLike,
        img_path: str,
        output_dir: str,
        suffix: str = "",
        panels: list[tuple] | None = None,
        x_offset: int = 0,
        id_offset: int = 0,
    ) -> list[BubbleZone]:
        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w
        base_name = os.path.splitext(os.path.basename(img_path))[0] + suffix

        variants = self.preProcessImage(img)
        cv2.imwrite(f"{output_dir}/{base_name}_enhanced.png", variants["enhanced"])
        cv2.imwrite(f"{output_dir}/{base_name}_inv.png",      variants["inv"])
        if self.debug:
            cv2.imwrite(f"{output_dir}/{base_name}_up.png", variants["up"])
            logger.debug("preprocessed variants saved to %s", output_dir)
            self._save_paddle_full_debug(img, output_dir, base_name)

        # ── Step 1: detecta painéis ───────────────────────────────────
        detected = self.panel_detector._find_panel_dividers(img)

        panels_filtered = [
            p for p in detected
            if (p[2] * p[3]) >= _PANEL_MIN_AREA * img_area
        ]
        panels_filtered.sort(key=lambda p: (-p[0], p[1]))

        if not panels_filtered:
            logger.warning("%s — nenhum painel detectado, usando imagem inteira", base_name)
            panels_filtered = [(0, 0, img_w, img_h)]

        logger.info("%s — %d painéis detectados", base_name, len(panels_filtered))

        # ── Step 2: regiões gap (não cobertas por painéis) ───────────
        # Cobre texto de narração, artigos e qualquer conteúdo fora dos painéis
        gap_regions = self._find_gap_regions(panels_filtered, img_w, img_h)
        logger.info("%s — %d regiões gap", base_name, len(gap_regions))

        if self.debug:
            self._save_region_debug(img, panels_filtered, gap_regions, output_dir, base_name)

        # ── Step 3: OCR em painéis + gaps ────────────────────────────
        crops_dir = os.path.join(output_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

        all_detections: list[tuple[tuple[int,int,int,int], str, float]] = []

        for region_idx, (rx, ry, rw, rh) in enumerate(panels_filtered + gap_regions):
            is_gap = region_idx >= len(panels_filtered)
            region_crop = img[ry:ry + rh, rx:rx + rw]

            raw = self._ocr_region(region_crop, is_gap=is_gap)
            logger.debug(
                "%s #%d @ (%d,%d) %dx%d → %d detecções",
                "gap" if is_gap else "painel",
                region_idx, rx, ry, rw, rh, len(raw),
            )

            for (cx, cy, cw, ch), text, conf in raw:
                all_detections.append(((cx + rx, cy + ry, cw, ch), text, conf))

        logger.info("%s — %d detecções brutas totais", base_name, len(all_detections))

        # ── Step 4: deduplicação global ───────────────────────────────
        deduped = all_detections#self._deduplicate(all_detections)
        logger.info("%s — %d detecções após deduplicação", base_name, len(deduped))

        if self.debug and deduped:
            self._save_debug_visualization(img, deduped, output_dir, base_name)

        # ── Step 5: monta BubbleZones ─────────────────────────────────
        results: list[BubbleZone] = []
        for i, ((cx, cy, cw, ch), text, conf) in enumerate(deduped):
            bubble_id = id_offset + i
            abs_x = cx + x_offset

            crop_path = self.crop_bubble(
                img, abs_x, cy, cw, ch,
                crops_dir=crops_dir,
                base_name=base_name,
                bubble_id=bubble_id,
            )

            zone: BubbleZone = {
                "id":      bubble_id,
                "x":       abs_x,
                "y":       cy,
                "w":       cw,
                "h":       ch,
                "jp_text": text,
            }
            if crop_path:
                zone["crop"] = crop_path

            results.append(zone)

        logger.info("%s — %d zonas de texto finais", base_name, len(results))
        return results

    # ------------------------------------------------------------------
    # Gap coverage
    # ------------------------------------------------------------------

    def _find_gap_regions(
        self,
        panels: list[tuple[int,int,int,int]],
        img_w: int,
        img_h: int,
    ) -> list[tuple[int,int,int,int]]:
        """
        Encontra faixas horizontais não cobertas por nenhum painel.
        Retorna regiões (x=0, y, w=img_w, h) com h >= _GAP_MIN_HEIGHT.

        Cobre: narração fora de painéis, artigos de texto, cabeçalhos, rodapés.
        """
        if not panels:
            return []

        covered = np.zeros(img_h, dtype=bool)
        for (px, py, pw, ph) in panels:
            covered[max(0, py) : min(img_h, py + ph)] = True

        gaps: list[tuple[int,int,int,int]] = []
        in_gap = False
        gap_start = 0

        for y in range(img_h):
            if not covered[y] and not in_gap:
                in_gap = True
                gap_start = y
            elif covered[y] and in_gap:
                in_gap = False
                gap_h = y - gap_start
                if gap_h >= _GAP_MIN_HEIGHT:
                    gaps.append((0, gap_start, img_w, gap_h))

        if in_gap:
            gap_h = img_h - gap_start
            if gap_h >= _GAP_MIN_HEIGHT:
                gaps.append((0, gap_start, img_w, gap_h))

        return gaps

    # ------------------------------------------------------------------
    # OCR por região
    # ------------------------------------------------------------------

    def _ocr_region(self, region: cv2.typing.MatLike, is_gap: bool = False) -> list[tuple[tuple[int,int,int,int], str, float]]:
        rh, rw = region.shape[:2]
        raw: list[tuple[tuple[int,int,int,int], str, float]] = []

        # Tiling dinâmico com overlap maior
        tiles = []
        if rh > _TILE_SPLIT_HEIGHT:
            step = _TILE_SPLIT_HEIGHT - int(_TILE_SPLIT_HEIGHT * _TILE_OVERLAP)
            for y in range(0, rh, step):
                y_end = min(y + _TILE_SPLIT_HEIGHT, rh)
                tiles.append((region[y:y_end, :], 0, y))
                if y_end == rh: break
        else:
            tiles = [(region, 0, 0)]

        for tile, tx, ty in tiles:
            # Testamos variantes para garantir detecção em fundos complexos (Benchmark-friendly)
            # 1. Original
            for bbox, text, conf in self._ocr_on_array(tile):
                raw.append(((bbox[0] + tx, bbox[1] + ty, bbox[2], bbox[3]), text, conf))

            # 2. Enhanced (Denoise + Adaptive)
            enhanced = self._enhance_for_manga(tile)
            for bbox, text, conf in self._ocr_on_array(enhanced):
                raw.append(((bbox[0] + tx, bbox[1] + ty, bbox[2], bbox[3]), text, conf))

        return self._deduplicate(raw)

    # ------------------------------------------------------------------
    # Pré-processamento
    # ------------------------------------------------------------------

    def _enhance_for_ocr(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # sharpen strokes
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        sharp = cv2.filter2D(gray, -1, kernel)

        # binarize
        _, thresh = cv2.threshold(
            sharp, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return thresh
    
    def _enhance_for_manga(self, img: np.ndarray) -> np.ndarray:
        """Melhoria focada em remover reticulado sem destruir o texto."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Suaviza o ruído do screentone
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Binarização adaptativa é melhor que Otsu para páginas de mangá com sombras
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    # ------------------------------------------------------------------
    # PaddleOCR
    # ------------------------------------------------------------------

    def _init_paddle(self) -> PaddleOCR:
        return PaddleOCR(
            lang="japan",
            ocr_version='PP-OCRv4', # v4 ainda é o SOTA estável
            use_gpu=torch.cuda.is_available(),
            
            # --- CONFIGURAÇÃO PARA BENCHMARK AGRESSIVO ---
            det_db_thresh=0.1,         # Aceita qualquer sinal de "tinta"
            det_db_box_thresh=0.2,     # Aceita caixas mesmo com certeza baixíssima
            det_db_unclip_ratio=3.5,   # Aumenta a caixa para "morder" o caractere vizinho
            
            # Isso impede que o Paddle ignore textos pequenos/finos
            det_limit_side=4000, 
            max_batch_size=10,
            
            # Tente True/False aqui: às vezes o 'dilation' ajuda a unir as letras verticais
            use_dilation=True, 
            
            # Importante: se o texto for muito denso, aumente o limite de detecções
            # det_model_dir='...' (Apenas se estiver usando modelos customizados)
        )

    def _ocr_on_array(self, img_arr: np.ndarray) -> list[tuple[tuple[int,int,int,int], str, float]]:
        """Upscale moderado: 3x às vezes gera ruído demais, 2x é o sweet spot do Paddle."""
        h, w = img_arr.shape[:2]
        img_up = cv2.resize(img_arr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cv2.imwrite(tmp_path, img_up)
            result = self._ocr.ocr(tmp_path, cls=False)
            detections = []
            if result and result[0]:
                for line in result[0]:
                    quad, (text, conf) = line
                    # Reescalar de volta (÷2)
                    xs = [pt[0] / 2 for pt in quad]
                    ys = [pt[1] / 2 for pt in quad]
                    x, y = int(min(xs)), int(min(ys))
                    detections.append(((x, y, int(max(xs)) - x, int(max(ys)) - y), text, float(conf)))
            return detections
        finally:
            if os.path.exists(tmp_path): os.unlink(tmp_path)

    def _ocr_on_file(
        self, image_path: str
    ) -> list[tuple[tuple[int,int,int,int], str, float]]:
        result = self._ocr.ocr(image_path, cls=True)
        detections = []
        if not result or result[0] is None:
            return detections
        for line in result[0]:
            quad, (text, conf) = line
            detections.append((self._quad_to_xywh(quad), text, float(conf)))
        return detections

    # ------------------------------------------------------------------
    # Utilitários estáticos
    # ------------------------------------------------------------------

    @staticmethod
    def _quad_to_xywh(quad: list) -> tuple[int,int,int,int]:
        xs = [pt[0] for pt in quad]
        ys = [pt[1] for pt in quad]
        x, y = int(min(xs)), int(min(ys))
        return x, y, int(max(xs)) - x, int(max(ys)) - y

    @staticmethod
    def _merge_lines(
        detections: list[tuple[tuple[int,int,int,int], str, float]]
    ) -> str:
        """Ordem de leitura manga: colunas direita→esquerda, cima→baixo."""
        sorted_lines = sorted(detections, key=lambda d: (-d[0][0], d[0][1]))
        return "".join(text for _, text, _ in sorted_lines)

    def _deduplicate(
        self,
        detections: list[tuple[tuple[int,int,int,int], str, float]],
    ) -> list[tuple[tuple[int,int,int,int], str, float]]:
        """Remove duplicatas por IoU, mantendo a detecção de maior confiança."""
        sorted_dets = sorted(detections, key=lambda d: d[2], reverse=True)
        kept: list[tuple[tuple[int,int,int,int], str, float]] = []
        for det in sorted_dets:
            bbox, text, conf = det
            if not any(self._iou(bbox, k[0]) > _IOU_MERGE_THRESHOLD for k in kept):
                kept.append(det)
        return kept

    # ------------------------------------------------------------------
    # Debug / visualização
    # ------------------------------------------------------------------

    def _save_paddle_full_debug(self, img, output_dir, base_name):
        tmp_path = os.path.join(output_dir, f"{base_name}_full_tmp.png")
        cv2.imwrite(tmp_path, img)
        try:
            result = self._ocr.ocr(tmp_path, cls=True)
            if not result or result[0] is None:
                return
            vis = img.copy()
            for line in result[0]:
                quad, (text, conf) = line
                pts = [(int(x), int(y)) for x, y in quad]
                for i in range(4):
                    cv2.line(vis, pts[i], pts[(i + 1) % 4], (0, 255, 0), 2)
                cv2.putText(
                    vis, f"{conf:.2f}", pts[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
                )
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_paddle_full.jpg"), vis)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _save_region_debug(
        self,
        img: cv2.typing.MatLike,
        panels: list[tuple[int,int,int,int]],
        gaps: list[tuple[int,int,int,int]],
        output_dir: str,
        base_name: str,
    ) -> None:
        """Painéis em laranja, gaps em ciano."""
        vis = img.copy()
        for i, (x, y, w, h) in enumerate(panels):
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(vis, f"P{i}", (x + 4, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
        for i, (x, y, w, h) in enumerate(gaps):
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 200, 0), 2)
            cv2.putText(vis, f"G{i}", (x + 4, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
        out = os.path.join(output_dir, f"{base_name}_regions.jpg")
        cv2.imwrite(out, vis)
        logger.debug("region debug saved → %s", out)

    def _save_debug_visualization(
        self,
        img: cv2.typing.MatLike,
        detections: list[tuple[tuple[int,int,int,int], str, float]],
        output_dir: str,
        base_name: str,
    ) -> None:
        """Detecções OCR finais desenhadas em verde."""
        quads, texts, scores = [], [], []
        for (x, y, w, h), text, conf in detections:
            quads.append([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
            texts.append(text)
            scores.append(conf)

        out_path = os.path.join(output_dir, f"{base_name}_detections.jpg")
        try:
            pil_img   = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            annotated = draw_ocr(pil_img, quads, texts, scores)
            PILImage.fromarray(annotated).save(out_path)
            logger.debug("detection debug saved → %s", out_path)
        except Exception as exc:
            logger.warning("draw_ocr failed (%s), usando fallback cv2", exc)
            vis = img.copy()
            for (x, y, w, h), text, conf in detections:
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    vis, f"{text}({conf:.2f})", (x, max(y - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1,
                )
            cv2.imwrite(out_path, vis)
            logger.debug("cv2 fallback detection debug saved → %s", out_path)