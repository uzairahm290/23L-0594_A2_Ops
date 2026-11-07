"""Project-wide constants. Anything that changes between machines comes from env vars."""
import os
from pathlib import Path

PROJECT_NAME = "23L-0594_A2_Ops"
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("VG_DATA_DIR", ROOT / "data" / "raw"))
SPLIT_DIR = ROOT / "data" / "splits"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
MODEL_DIR = ROOT / "models"

# Positive class = AI-generated. Label comes from the folder an image sits in.
CLASSES = ["real", "ai"]          # index 0 = real, 1 = ai
SEED = 42
IMG_SIZE = 224                     # MobileNetV2's native input size
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def label_from_folder(folder_name: str) -> str | None:
    """Kaggle folders are 'AiArtData' and 'RealArt'. Match loosely so renames don't break us."""
    name = folder_name.lower()
    if name.startswith("ai") or "aiart" in name or "fake" in name or "generated" in name:
        return "ai"
    if "real" in name:
        return "real"
    return None
