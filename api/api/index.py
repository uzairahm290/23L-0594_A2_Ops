"""VisionGuard model API - deployed on Vercel (Task 7).

TensorFlow (~600 MB) can't fit in a Vercel function (250 MB unzipped limit), so the
champion Keras model is exported to TFLite and run with the LiteRT interpreter (a few MB).
Preprocessing is a byte-identical copy of the training code.

  GET  /health              -> model + version info (no key needed)
  POST /predict  image=<file>   header X-API-Key: <VG_API_KEY>
"""
import hmac
import json
import os
import time
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np
from flask import Flask, jsonify, request

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:                       # local dev with full TF installed
    from tensorflow.lite import Interpreter

try:
    from .preprocessing import UndecodableImage, bytes_to_input
except ImportError:
    from preprocessing import UndecodableImage, bytes_to_input

HERE = Path(__file__).resolve().parent
MODEL_FILE = Path(os.getenv("VG_MODEL_FILE", HERE.parent / "model" / "visionguard.tflite"))
INFO_FILE = MODEL_FILE.parent / "model_info.json"
MAX_BYTES = 4 * 1024 * 1024               # Vercel's own body limit is 4.5 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES + 64 * 1024


class _RestorePath:
    """vercel.json rewrites every path to /api/index?__path=<original>, so Flask would only ever
    see /api/index. Put the original path back before routing."""

    def __init__(self, wsgi):
        self.wsgi = wsgi

    def __call__(self, environ, start_response):
        q = parse_qs(environ.get("QUERY_STRING", ""), keep_blank_values=True)
        if "__path" in q:
            environ["PATH_INFO"] = "/" + q["__path"][0].lstrip("/")
        return self.wsgi(environ, start_response)


app.wsgi_app = _RestorePath(app.wsgi_app)

# --- load once per cold start, reused by every warm invocation ---
_info = json.loads(INFO_FILE.read_text()) if INFO_FILE.exists() else {}
MODEL_NAME = _info.get("name", "23L-0594_A2_Ops_visionguard")
MODEL_VERSION = str(_info.get("version", "1"))
THRESHOLD = float(os.getenv("VG_THRESHOLD", _info.get("threshold", 0.5)))
_interp = Interpreter(model_path=str(MODEL_FILE))
_interp.allocate_tensors()
_in = _interp.get_input_details()[0]
_out = _interp.get_output_details()[0]
IMG_SIZE = int(_in["shape"][1])


def _authorised() -> bool:
    expected = os.getenv("VG_API_KEY", "")
    given = request.headers.get("X-API-Key", "")
    # constant-time compare; an unset key on the server rejects everything
    return bool(expected) and hmac.compare_digest(expected, given)


@app.get("/")
@app.get("/health")
def health():
    return jsonify(status="ok", model_name=MODEL_NAME, model_version=MODEL_VERSION,
                   threshold=THRESHOLD, input_size=IMG_SIZE, runtime="tflite")


@app.post("/predict")
def predict():
    if not _authorised():
        return jsonify(error="missing or wrong X-API-Key"), 401
    f = request.files.get("image")
    if f is None:
        return jsonify(error="send the image as multipart field 'image'"), 400
    data = f.read()
    if not data:
        return jsonify(error="empty file"), 400
    t0 = time.perf_counter()
    try:
        x = bytes_to_input(data, IMG_SIZE)
    except UndecodableImage:
        return jsonify(error="file could not be decoded as an image"), 422
    _interp.set_tensor(_in["index"], x.astype(np.float32))
    _interp.invoke()
    p = float(_interp.get_tensor(_out["index"])[0, 0])
    is_ai = p >= THRESHOLD
    return jsonify(label="AI-generated" if is_ai else "Real", p_ai=round(p, 6),
                   confidence=round(p if is_ai else 1 - p, 6), threshold=THRESHOLD,
                   model_name=MODEL_NAME, model_version=MODEL_VERSION,
                   inference_ms=int((time.perf_counter() - t0) * 1000))


@app.errorhandler(413)
def too_big(_):
    return jsonify(error="file too large", limit_mb=MAX_BYTES / 1024 / 1024), 413
