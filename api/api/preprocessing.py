"""Single source of truth for turning raw image bytes into model input.

Training (tf.data via py_function), the Flask app and the Vercel API all call these
exact functions, so there is no train/serve skew:
  1. decode with Pillow, first frame only (GIF / animated WEBP), force RGB
     (drops alpha, expands greyscale and palette images)
  2. resize to IMG_SIZE x IMG_SIZE with bilinear filtering; aspect ratio is NOT kept,
     the image is stretched (cropping would throw away content that may hold the
     generator artefacts we're looking for)
  3. scale to [-1, 1] with x / 127.5 - 1, the range MobileNetV2 was pre-trained on

api/api/preprocessing.py is a byte-identical copy (Vercel deploys api/ on its own);
tests/test_preprocessing.py fails if the two drift apart.
"""
import io

import numpy as np
from PIL import Image

try:
    from .config import IMG_SIZE
except ImportError:          # vendored copy inside api/ (Vercel) has no package
    IMG_SIZE = 224


class UndecodableImage(ValueError):
    pass


def load_rgb(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.seek(0)          # first frame for GIF/animated WEBP
        img.load()           # forces a full decode; truncated files fail here
    except Exception as exc:  # PIL raises many types
        raise UndecodableImage(str(exc)) from exc
    return img.convert("RGB")


def to_model_input(img: Image.Image, size: int = IMG_SIZE) -> np.ndarray:
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    arr = arr / 127.5 - 1.0
    return arr[np.newaxis, ...]  # (1, H, W, 3)


def bytes_to_input(data: bytes, size: int = IMG_SIZE) -> np.ndarray:
    return to_model_input(load_rgb(data), size)
