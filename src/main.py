#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore")

import argparse
import logging

from .FTM import FTM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

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
        "--huggingface_online_check", action="store_true", default=False,
        help="Check Hugging Face model availability at startup and warn if offline(Default use cache/ offline mode)"
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
        "--use-monitor", action="store_true", default=False,
        help="Monitor CPU/RAM resource usage and save to CSV in the tmp directory (requires psutil)"
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