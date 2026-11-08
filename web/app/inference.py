"""Task 4.1 - inference module.

The model is loaded ONCE when the app starts (create_app -> get_predictor). Loading a
Keras model takes seconds and hundreds of MB of RAM; doing that per request would make
every upload slow and let a handful of parallel users run the container out of memory.

Two backends share one interface:
  LocalPredictor   Keras model file on disk (champion copy mounted into the container)
  RemotePredictor  the model API deployed on Vercel (Task 7/8); Flask only forwards bytes
"""
import time
from dataclasses import dataclass

import requests

from visionguard.preprocessing import bytes_to_input


@dataclass
class Prediction:
    label: str            # "AI-generated" | "Real"
    p_ai: float           # sigmoid output = probability of the AI class
    confidence: float     # probability of the chosen label
    threshold: float
    model_name: str
    model_version: str
    backend: str
    latency_ms: int


def _decide(p_ai: float, threshold: float) -> tuple[str, float]:
    if p_ai >= threshold:
        return "AI-generated", p_ai
    return "Real", 1.0 - p_ai


class LocalPredictor:
    backend = "local"

    def __init__(self, model_path: str, threshold: float, name: str, version: str):
        import tensorflow as tf          # imported here so the remote image needn't ship TF
        self.model = tf.keras.models.load_model(model_path)
        self.size = int(self.model.input_shape[1])   # follow whatever size the champion was trained at
        self.threshold, self.name, self.version = threshold, name, version
        self.model.predict(bytes_to_input(_blank_png(), self.size), verbose=0)   # warm-up: first call builds the graph

    def predict(self, data: bytes) -> Prediction:
        t0 = time.perf_counter()
        x = bytes_to_input(data, self.size)          # identical to training preprocessing
        p = float(self.model.predict(x, verbose=0)[0, 0])
        label, conf = _decide(p, self.threshold)
        return Prediction(label, p, conf, self.threshold, self.name, self.version, self.backend,
                          int((time.perf_counter() - t0) * 1000))

    def ready(self) -> bool:
        return self.model is not None


class RemotePredictor:
    backend = "remote"

    def __init__(self, url: str, key: str, timeout: float):
        if not url:
            raise RuntimeError("INFERENCE_BACKEND=remote but MODEL_API_URL is empty")
        self.url, self.key, self.timeout = url.rstrip("/"), key, timeout
        self.session = requests.Session()           # reuse TLS connection across requests

    def predict(self, data: bytes) -> Prediction:
        t0 = time.perf_counter()
        try:
            r = self.session.post(f"{self.url}/predict", files={"image": ("upload", data)},
                                  headers={"X-API-Key": self.key}, timeout=self.timeout)
        except requests.RequestException as e:      # timeout / DNS / TLS: same 502 path as a bad status
            raise RemoteModelError(502, type(e).__name__) from e
        if r.status_code != 200:
            raise RemoteModelError(r.status_code, r.text[:200])
        j = r.json()
        return Prediction(j["label"], j["p_ai"], j["confidence"], j["threshold"],
                          j["model_name"], str(j["model_version"]), "remote:vercel",
                          int((time.perf_counter() - t0) * 1000))

    def ready(self) -> bool:
        try:
            return self.session.get(f"{self.url}/health", timeout=5).ok
        except requests.RequestException:
            return False


class RemoteModelError(RuntimeError):
    def __init__(self, status, body):
        super().__init__(f"model API returned {status}: {body}")
        self.status = status


def get_predictor(cfg):
    if cfg.BACKEND == "remote":
        return RemotePredictor(cfg.MODEL_API_URL, cfg.MODEL_API_KEY, cfg.MODEL_API_TIMEOUT)
    return LocalPredictor(cfg.MODEL_PATH, cfg.THRESHOLD, cfg.MODEL_NAME, cfg.MODEL_VERSION)


def _blank_png() -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "PNG")
    return buf.getvalue()
