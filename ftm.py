#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore")

import json
import os
import time
import argparse
import logging
from typesetter.Bubbletypesetter import BubbleTypesetter
from utils.resource import ResourceMonitor
from detector.detector import detect

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


def run_ocr(bubbles: list[dict], debug: bool) -> list[dict]:
    from ocr.MangaOCR import MangaOCREngine
    ocr = MangaOCREngine()
    for i, bubble in enumerate(bubbles):
        crop_path = bubble.get("crop")
        jp_text = ocr.extract(crop_path)
        bubble["jp_text"] = jp_text
        if debug:
            print(f"  [{i+1}/{len(bubbles)}] {crop_path} → {jp_text!r}")
    return bubbles


def run_translate(
    bubbles: list[dict],
    target_lang: str,
    model: str,
    ollama_host: str = "http://localhost:11434",
    ollama_model_temperature: float = 0,
    field_name: str = "translated_text",
    debug: bool = False,
) -> list[dict]:
    
    from translator.OllamaTranslator import OllamaTranslator
    translator = OllamaTranslator(
        source_lang="ja", 
        model=model, 
        ollama_host=ollama_host, 
        ollama_model_temperature=ollama_model_temperature
    )

    for i, bubble in enumerate(bubbles):
        translated = translator.translate(bubble["jp_text"], lang=target_lang)
        bubble[field_name] = translated
        if debug:
            print(f"  [{i+1}/{len(bubbles)}] {bubble['jp_text']!r} → {translated!r}")
    return bubbles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FTM")

    parser.add_argument(
        "--engine", default="easy", choices=["easy", "tesseract", "paddle"],
        help="OCR engine to use during detection step",
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Path to output image")
    parser.add_argument("--tmp", default="./tmp", help="Path to temporary directory")
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Show all logs and save all intermediate results",
    )
    parser.add_argument(
        "--monitor", action="store_true", default=False,
        help="Monitor CPU/RAM resource usage",
    )
    parser.add_argument(
        "--steps", dest="steps", nargs="+",
        choices=["detection", "refinement", "ocr", "translation", "typesetting"],
        default=["detection", "refinement", "ocr", "translation", "typesetting"],
        help="Run specific steps of the pipeline",
    )
    parser.add_argument(
        "--bubbles-json", dest="bubbles_json",
        help="Path to JSON file with bubble data (skips detection and refinement steps)",
    )
    parser.add_argument(
        "--translate-lang", default="en", choices=["en", "pt"],
        help="Language to translate to (default: en)",
    )
    parser.add_argument(
        "--translate-model", default="translategemma:4b",
        choices=["translategemma:4b", "translategemma:12b"],
        help="Ollama model to use for translation (default: translategemma:4b)",
    )
    parser.add_argument(
        '--ollama-host', default='http://localhost:11434',
        help='Custom host URL for Ollama API (default: http://localhost:11434)'
    )
    parser.add_argument(
        '--ollama-model-temperature', default=0.0,
        help='Temperature for option for model (default: 0.0)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    image_path: str = args.image
    tmp_dir: str = args.tmp
    engine: str = args.engine.lower()
    debug: bool = args.debug
    use_monitor: bool = args.monitor
    steps: list[str] = args.steps

    os.makedirs(tmp_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path: str = args.output or os.path.join(tmp_dir, f"{base_name}_translated.jpg")

    monitor: ResourceMonitor | None = None
    if use_monitor:
        csv_path = os.path.join(tmp_dir, f"resources_{engine}.csv")
        monitor = ResourceMonitor(output_path=csv_path, interval=0.5)
        monitor.start(label=f"{engine}_start")

    pipeline_start = time.perf_counter()
    print(f"\n{'═' * 52}")
    print(f"  FTM  |  {os.path.basename(image_path)}")
    print(f"\n{'═' * 52}")
    print(f"\n┌── Flags: ")
    print(f"  steps: {', '.join(steps)}")
    print(f"  engine: {engine}")
    print(f"  debug: {debug}")
    print(f"  monitor: {use_monitor}")
    print(f"  translate_model: {args.translate_model}")
    print(f"  translate_lang: {args.translate_lang}")
    print(f"  bubbles_json: {args.bubbles_json}")
    print(f"  output: {output_path}")
    print(f"  tmp: {tmp_dir}")
    print(f"  ollama_host: {args.ollama_host}")
    print(f"  ollama_model_temperature: {args.ollama_model_temperature}")
    print(f"└── ")

    try:
        bboxs = []
        if "detection" in steps and args.bubbles_json is None:
            bboxs = run_step(
                label="Detect speech bubbles",
                fn=lambda: detect(image_path, engine, debug, tmp_dir, monitor),
            )

        bubbles = []
        if "refinement" in steps and args.bubbles_json is None:
            from ocr.BubbleRefiner import BubbleRefiner
            bubbles = run_step(
                label="Refine bubble crops",
                fn=lambda: BubbleRefiner(debug=debug).refine(bboxs),
            )

        if args.bubbles_json:
            print(f"\n⚠  --bubbles-json set. Loading from {args.bubbles_json}.")
            with open(args.bubbles_json, "r", encoding="utf-8") as f:
                bubbles = json.load(f)

        if not bubbles:
            print("\n⚠  No bubbles to process. Exiting.")
            return

        if "ocr" in steps:
            bubbles = run_step(
                label="OCR — extract Japanese text",
                fn=lambda: run_ocr(bubbles, debug),
            )

        if "translation" in steps:
            bubbles = run_step(
                label="Translate JP → EN",
                fn=lambda: run_translate(
                    bubbles, "en", 
                    args.translate_model, 
                    args.ollama_host,
                    args.ollama_model_temperature,
                    "en_text", 
                    debug
                ),
            )

            if args.translate_lang == "en":
                print("\n⚠  --translate-lang is 'en'. Using EN as final output.")
                for bubble in bubbles:
                    bubble["translated_text"] = bubble["en_text"]
            else:
                bubbles = run_step(
                    label=f"Translate JP → {args.translate_lang.upper()}",
                    fn=lambda: run_translate(
                        bubbles, 
                        args.translate_lang, 
                        args.translate_model, 
                        args.ollama_host,
                        args.ollama_model_temperature,
                        "translated_text", 
                        debug
                    ),
                )

        if "typesetting" in steps:
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

    finally:
        if monitor:
            monitor.stop()


if __name__ == "__main__":
    main()