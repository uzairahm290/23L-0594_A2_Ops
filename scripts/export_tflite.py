"""Convert the champion Keras model to TFLite for the Vercel API (TensorFlow itself is
far over Vercel's function size limit; the LiteRT interpreter is a few MB).

  python scripts/export_tflite.py models/champion/model.keras
Writes api/model/visionguard.tflite and checks it agrees with the Keras model.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1])
dst = ROOT / "api" / "model" / "visionguard.tflite"
model = tf.keras.models.load_model(src)

# via an exported SavedModel: from_keras_model crashes on EfficientNet under Keras 3
# ("Failed to infer result type(s)" on a BatchNorm variable read)
with tempfile.TemporaryDirectory() as tmp:
    model.export(tmp)
    conv = tf.lite.TFLiteConverter.from_saved_model(tmp)
    # float16 weights: half the size, same test verdicts. Dynamic-range int8 was 4.6 MB but
    # flipped 3/147 test labels on EfficientNet (max |keras - tflite| 0.38)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    tflite_bytes = conv.convert()
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_bytes(tflite_bytes)

info = src.parent / "model_info.json"
if info.exists():
    shutil.copy(info, dst.parent / "model_info.json")

interp = tf.lite.Interpreter(model_path=str(dst))
interp.allocate_tensors()
i, o = interp.get_input_details()[0], interp.get_output_details()[0]
rng = np.random.default_rng(0)
diffs = []
for _ in range(5):
    x = rng.uniform(-1, 1, (1, *i["shape"][1:])).astype(np.float32)
    interp.set_tensor(i["index"], x)
    interp.invoke()
    diffs.append(abs(float(interp.get_tensor(o["index"])[0, 0]) - float(model.predict(x, verbose=0)[0, 0])))
print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB); max |keras - tflite| on 5 inputs = {max(diffs):.4f}")
print(json.dumps({"input": i["shape"].tolist(), "output": o["shape"].tolist()}))
