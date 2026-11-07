"""Task 3.6 - load the champion model from the registry by alias and classify one image.

  python scripts/predict_by_alias.py path/to/image.jpg
Also writes the downloaded model to models/champion/ so Docker / the API can use the same file.
"""
import json
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visionguard import tracking  # noqa: E402
from visionguard.config import MODEL_DIR  # noqa: E402
from visionguard.preprocessing import bytes_to_input  # noqa: E402

NAME, ALIAS = "23L-0594_A2_Ops_visionguard", "champion"

tracking.setup("23L-0594_A2_Ops")
mv = MlflowClient().get_model_version_by_alias(NAME, ALIAS)
run = MlflowClient().get_run(mv.run_id)
threshold = float(run.data.params.get("threshold", 0.5))
print(f"{NAME}@{ALIAS} -> version {mv.version} (run {mv.run_id[:8]}), threshold {threshold:.3f}")

model = mlflow.tensorflow.load_model(f"models:/{NAME}@{ALIAS}")
x = bytes_to_input(Path(sys.argv[1]).read_bytes(), model.input_shape[1])   # size from the model itself
p = float(model.predict(x, verbose=0)[0, 0])
label = "AI-generated" if p >= threshold else "Real"
conf = p if p >= threshold else 1 - p
print(f"{sys.argv[1]}: {label}  p(AI)={p:.4f}  confidence={conf:.1%}")

out = MODEL_DIR / "champion"
out.mkdir(parents=True, exist_ok=True)
model.save(out / "model.keras")
(out / "model_info.json").write_text(json.dumps(
    {"name": NAME, "version": mv.version, "run_id": mv.run_id, "threshold": threshold}, indent=1))
print("saved champion copy to", out)
