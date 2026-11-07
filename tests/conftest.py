import io
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "web")]


def png_bytes(color=(200, 30, 30), size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


class FakePredictor:
    """Stands in for the Keras/remote model so web tests run without TensorFlow."""
    backend = "fake"

    def predict(self, data):
        from app.inference import Prediction
        p = 0.9 if data[-20:].count(b"\x00") % 2 else 0.2
        label = "AI-generated" if p >= 0.5 else "Real"
        return Prediction(label, p, p if p >= 0.5 else 1 - p, 0.5, "test-model", "0", "fake", 3)

    def ready(self):
        return True


@pytest.fixture
def client(tmp_path):
    from app import create_app
    from app.db import Store
    from app.settings import Settings

    class S(Settings):
        DATABASE_URL = f"sqlite:///{tmp_path/'t.db'}"
        MAX_UPLOAD_MB = 1

    app = create_app(S, predictor=FakePredictor(), store=Store(S.DATABASE_URL))
    app.testing = True
    return app.test_client()
