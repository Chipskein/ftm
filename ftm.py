#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore")

import json
import os
import time
import argparse
from typesetter.Bubbletypesetter import BubbleTypesetter
from utils.resource import ResourceMonitor
from detector.detector import detect
from ocr.BubbleRefiner import BubbleRefiner
from ocr.MangaOCR import MangaOCREngine
from translator.OllamaTranslator import OllamaTranslator

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

def run_step(label: str, fn):
    """Runs a pipeline step with timing and visual output."""
    print(f"\n┌── {label}")
    t0 = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - t0
        print(f"└── ✓ done  ({elapsed:.2f}s)")
        return result
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"└── ✗ FAILED after {elapsed:.2f}s → {e}")
        raise

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FTM")

    parser.add_argument("--engine", default="easy", choices=["easy", "tesseract","paddle"],
                        help="OCR engine to use during detection step")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Path to output image")
    parser.add_argument("--tmp", default="./tmp", help="Path to temporary directory")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Show all logs and save all intermediate results")
    parser.add_argument("--monitor", action="store_true", default=False,
                        help="Monitor CPU/RAM resource usage")
    parser.add_argument("--detection-only", action="store_true", default=False,
                        help="Only run detection step")
    parser.add_argument("--ocr-only", action="store_true", default=False,
                        help="Only run OCR step")
    parser.add_argument("--translate-lang", default="en", 
                        help="Language to translate to (default: en)", choices=["en", "pt"])
    parser.add_argument("--translate-model", default="translategemma:4b",
                        help="Ollama model to use for translation (default: translategemma:4b)", 
                        choices=["translategemma:4b", "translategemma:12b"])

    return parser.parse_args()

def main():
    args = parseArgs()

    image_path: str = args.image
    tmp_dir: str = args.tmp
    output_path: str = args.output if args.output else os.path.join(tmp_dir, "output.jpg")
    engine: str = args.engine.lower()
    debug: bool = args.debug
    use_monitor: bool = args.monitor

    os.makedirs(tmp_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if output_path == os.path.join(tmp_dir, "output.jpg"):
        output_path = os.path.join(tmp_dir, f"{base_name}_translated.jpg")

    monitor: ResourceMonitor | None = None
    if use_monitor:
        csv_path = os.path.join(tmp_dir, f"resources_{engine}.csv")
        monitor = ResourceMonitor(output_path=csv_path, interval=0.5)
        monitor.start(label=f"{engine}_start")

    pipeline_start = time.perf_counter()
    print(f"\n{'═' * 52}")
    print(f"  FTM  |  {os.path.basename(image_path)}  |  engine: {engine}")
    print(f"{'═' * 52}")

    bboxs = run_step(
        label="Detect speech bubbles",
        fn=lambda: detect(image_path, engine, debug, tmp_dir, monitor),
    )

    if args.detection_only:
        print("\n⚠  --detection-only flag set. Skipping OCR, translation, and typesetting steps.")
        if monitor:
            monitor.stop()
        return

    if not bboxs:
        print("\n⚠  No bounding boxes detected. Exiting.")
        if monitor:
            monitor.stop()
        return

    bubbles = run_step(
        label="Refine bubble crops",
        fn=lambda: BubbleRefiner(debug=debug).refine(bboxs),
    )

    def run_ocr():
        ocr = MangaOCREngine()
        for i, bubble in enumerate(bubbles):
            crop_path = bubble.get("crop")
            jp_text = ocr.extract(crop_path)
            bubble["jp_text"] = jp_text
            if debug:
                print(f"  [{i+1}/{len(bubbles)}] {crop_path} → {jp_text!r}")
        return bubbles

    bubbles = run_step(label="OCR — extract Japanese text", fn=run_ocr)

    if args.ocr_only:
        print("\n⚠  --ocr-only flag set. Skipping translation and typesetting steps.")
        if monitor:
            monitor.stop()
        return
    
    def run_translate(target_lang, field_name="translated_text"):
        translator = OllamaTranslator(source_lang="ja", model=args.translate_model)

        for i, bubble in enumerate(bubbles):
            translated = translator.translate(bubble["jp_text"], lang=target_lang)
            bubble[field_name] = translated

            if debug:
                print(f"  [{i+1}/{len(bubbles)}] {bubble['jp_text']!r} → {translated!r}")

        return bubbles
    
    bubbles = run_step(
        label="Translate JP → EN",
        fn=lambda: run_translate("en", field_name="en_text")
    )

    if args.translate_lang == "en":
        print("\n⚠  --translate-lang set to 'en'. Skipping JP → PT translation step.")
        for bubble in bubbles:
            bubble["translated_text"] = bubble["en_text"]
            
    else:
        bubbles = run_step(
            label=f"Translate JP → {str(args.translate_lang).upper()}",
            fn=lambda: run_translate(str(args.translate_lang).lower(), field_name="translated_text")
        )

    run_step(
        label="Typeset translated text",
        fn=lambda: BubbleTypesetter().typeset(
            img_path=image_path,
            bubbles=bubbles,
            output_path=output_path,
        ),
    )

    json_path = os.path.join(tmp_dir, f"{base_name}_bubbles.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bubbles, f, ensure_ascii=False, indent=2)

    total = time.perf_counter() - pipeline_start
    print(f"\n{'═' * 52}")
    print(f"  ✓ done in {total:.2f}s")
    print(f"  image   → {output_path}")
    print(f"  bubbles → {json_path}")
    print(f"{'═' * 52}\n")

    if monitor:
        monitor.stop()


if __name__ == "__main__":
    main()