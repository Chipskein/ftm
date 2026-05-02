# src/assets.py
from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "assets"
MODELS_DIR = ASSETS_DIR / "models"
FONTS_DIR  = ASSETS_DIR / "fonts"

def model(filename: str) -> Path:
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return path

def font(filename: str) -> Path:
    path = FONTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Font not found: {path}")
    return path