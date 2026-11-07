import numpy as np
import pytest
from conftest import ROOT, png_bytes

from visionguard.preprocessing import UndecodableImage, bytes_to_input


def test_api_copy_is_identical():
    a = (ROOT / "src/visionguard/preprocessing.py").read_bytes()
    b = (ROOT / "api/api/preprocessing.py").read_bytes()
    assert a == b, "run: cp src/visionguard/preprocessing.py api/api/preprocessing.py"


def test_shape_and_range():
    x = bytes_to_input(png_bytes(size=(640, 120)))
    assert x.shape == (1, 224, 224, 3) and x.dtype == np.float32
    assert -1.0 <= x.min() and x.max() <= 1.0


def test_grey_and_alpha_become_rgb():
    import io
    from PIL import Image
    for mode in ("L", "RGBA", "P"):
        buf = io.BytesIO()
        Image.new(mode, (30, 30)).save(buf, "PNG")
        assert bytes_to_input(buf.getvalue()).shape[-1] == 3


def test_garbage_raises():
    with pytest.raises(UndecodableImage):
        bytes_to_input(b"not an image")
