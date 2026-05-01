#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore")

import regex
import json
import os
import time
import argparse
import logging

from .profiler.ResourceMonitor import ResourceMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


class FTM:
    def __init__(self, args: argparse.Namespace):
        self.image_path   = args.image
        self.tmp_dir      = args.tmp
        self.engine_name  = "yolo"
        self.debug        = args.debug
        self.steps        = args.steps
        self.bubbles_json = args.bubbles_json
        self.translate_lang        = args.translate_lang
        self.translate_model       = args.translate_model
        self.ollama_host           = args.ollama_host
        self.ollama_model_temperature = args.ollama_model_temperature
        self.use_cpu_all  = args.use_cpu_all
        self.use_cpu_step = args.use_cpu_step or []

        os.makedirs(self.tmp_dir, exist_ok=True)
        self.base_name   = os.path.splitext(os.path.basename(self.image_path))[0]
        self.output_path = args.output or os.path.join(self.tmp_dir, f"{self.base_name}_translated.jpg")
        self.monitor     = self._setup_monitor()

    def _setup_monitor(self) -> ResourceMonitor | None:
        if not hasattr(self, '_use_monitor') or not self._use_monitor:
            return None
        csv_path = os.path.join(self.tmp_dir, f"resources_{self.base_name}_{self.engine_name}.csv")
        monitor = ResourceMonitor(output_path=csv_path, interval=0.5)
        monitor.start(label=f"{self.engine_name}_start")
        return monitor

    def _use_cpu(self, step: str) -> bool:
        return self.use_cpu_all or step in self.use_cpu_step

    def _detect(self) -> list[dict]:
        from .steps.detection.EngineOCRFactory import EngineOCRFactory
        engine = EngineOCRFactory(self.engine_name, self.debug, self._use_cpu("detection"), self.monitor)
        return engine.run(self.image_path, self.tmp_dir)

    def _extract(self, bubbles: list[dict]) -> list[dict]:
        from .steps.extraction.MangaOCR import MangaOCRExtractor
        extractor = MangaOCRExtractor(use_cpu=self._use_cpu("extraction"), monitor=self.monitor)
        for i, bubble in enumerate(bubbles):
            crop_path = bubble.get("crop")
            t0 = time.perf_counter()
            jp_text = extractor.extract(crop_path)
            bubble["extraction_symbols"] = len(regex.findall(r'\X', jp_text))
            bubble["extraction_time_s"]  = time.perf_counter() - t0
            bubble["jp_text"]            = jp_text
            if self.debug:
                print(f"  [{i+1}/{len(bubbles)}] {crop_path} → {jp_text!r}")
        return bubbles

    def _translate(self, bubbles: list[dict]) -> list[dict]:
        from .steps.translation.OllamaTranslator import OllamaTranslator
        translator = OllamaTranslator(
            source_lang="ja",
            model=self.translate_model,
            ollama_host=self.ollama_host,
            ollama_model_temperature=self.ollama_model_temperature,
            monitor=self.monitor,
        )
        for i, bubble in enumerate(bubbles):
            t0 = time.perf_counter()
            translated = translator.translate(bubble["jp_text"], lang=self.translate_lang)
            bubble["translation_symbols"] = len(regex.findall(r'\X', bubble["jp_text"]))
            bubble["translating_time_s"]  = time.perf_counter() - t0
            bubble["translated_text"]     = translated
            if self.debug:
                print(f"  [{i+1}/{len(bubbles)}] {bubble['jp_text']!r} → {translated!r}")
        return bubbles

    def _typeset(self, bubbles: list[dict]) -> list[dict]:
        from .steps.typesetting.Bubbletypesetter import BubbleTypesetter
        return BubbleTypesetter(monitor=self.monitor).typeset(
            img_path=self.image_path,
            bubbles=bubbles,
            output_path=self.output_path,
        )

    def _run_step(self, label: str, fn):
        print(f"\n┌── {label}")
        t0 = time.perf_counter()
        try:
            result = fn()
            print(f"└── ✓ done  ({time.perf_counter() - t0:.2f}s)")
            return result
        except Exception as e:
            print(f"└── ✗ FAILED after {time.perf_counter() - t0:.2f}s → {e}")
            raise

    def _print_header(self):
        print(f"\n{'═' * 52}")
        print(f"  FTM  |  {os.path.basename(self.image_path)}")
        print(f"\n{'═' * 52}")
        print(f"\n┌── Flags:")
        print(f"  steps:                    {', '.join(self.steps)}")
        print(f"  engine:                   {self.engine_name}")
        print(f"  debug:                    {self.debug}")
        print(f"  monitor:                  {self.monitor is not None}")
        print(f"  translate_model:          {self.translate_model}")
        print(f"  translate_lang:           {self.translate_lang}")
        print(f"  bubbles_json:             {self.bubbles_json}")
        print(f"  output:                   {self.output_path}")
        print(f"  tmp:                      {self.tmp_dir}")
        print(f"  ollama_host:              {self.ollama_host}")
        print(f"  ollama_model_temperature: {self.ollama_model_temperature}")
        print(f"  use_cpu_all:              {self.use_cpu_all}")
        print(f"  use_cpu_step:             {self.use_cpu_step}")
        print(f"└──")

    def run(self):
        self._print_header()
        t0 = time.perf_counter()

        try:
            bubbles = []

            if "detection" in self.steps and not self.bubbles_json:
                bubbles = self._run_step("Detect speech bubbles", self._detect)

            if self.bubbles_json:
                print(f"\n⚠  --bubbles-json set. Loading from {self.bubbles_json}.")
                with open(self.bubbles_json, "r", encoding="utf-8") as f:
                    bubbles = json.load(f)

            if not bubbles:
                print("\n⚠  No bubbles to process. Exiting.")
                return

            if "extraction" in self.steps:
                bubbles = self._run_step(
                    "Extraction — extract Japanese text",
                    lambda: self._extract(bubbles)
                )

            if "translation" in self.steps:
                bubbles = self._run_step(
                    f"Translate JP → {self.translate_lang.upper()}",
                    lambda: self._translate(bubbles)
                )

            if "typesetting" in self.steps:
                bubbles = self._run_step(
                    "Typeset translated text",
                    lambda: self._typeset(bubbles)
                )

            json_path = os.path.join(self.tmp_dir, f"{self.base_name}_bubbles.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(bubbles, f, ensure_ascii=False, indent=2)

            total = time.perf_counter() - t0
            print(f"\n{'═' * 52}")
            print(f"  ✓ done in {total:.2f}s")
            print(f"  image translated → {self.output_path}")
            print(f"  bubbles          → {json_path}")
            print(f"{'═' * 52}\n")

        finally:
            if self.monitor:
                self.monitor.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FTM - Ferramenta de Tradução de Mangás - A modular manga translation pipeline"
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to output image (default: image_translated.jpg in tmp dir)"
    )
    parser.add_argument(
        "--tmp", default="./tmp",
        help="Path to temporary directory (default: ./tmp)"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Show all logs and save all intermediate results (crops, debug images, etc.) in the tmp directory"
    )
    parser.add_argument(
        "--monitor", action="store_true", default=False,
        help="Monitor CPU/RAM resource usage and save to CSV in the tmp directory (requires psutil)"
    )
    parser.add_argument(
        "--steps", nargs="+",
        choices=["detection", "extraction", "translation", "typesetting"],
        default=["detection", "extraction", "translation", "typesetting"],
        help="Run specific steps of the pipeline (default: all steps: detection, extraction, translation, typesetting)"
    )
    parser.add_argument(
        "--bubbles-json", dest="bubbles_json", default=None,
        help="Path to JSON file with bubble data (skips detection step)"
    )
    parser.add_argument(
        "--translate-lang", default="en", choices=["en", "pt"],
        help="Language to translate to (default: en)"
    )
    parser.add_argument(
        "--translate-model", default="translategemma:4b",
        choices=["translategemma:4b", "translategemma:12b"],
        help="Ollama model to use for translation (default: translategemma:4b)"
    )
    parser.add_argument(
        "--ollama-host", default="http://localhost:11434",
        help="Custom host URL for Ollama API (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--ollama-model-temperature", default=0.0, type=float,
        help="Temperature for the model (default: 0.0)"
    )
    parser.add_argument(
        "--use-cpu-all", action="store_true", default=False,
        help="Use CPU for all steps (default: False)"
    )
    parser.add_argument(
        "--use-cpu-step", nargs="+", choices=["detection", "extraction"], default=[],
        help="Use CPU for specific steps (overrides --use-cpu-all)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = FTM(args)
    pipeline.run()


if __name__ == "__main__":
    main()