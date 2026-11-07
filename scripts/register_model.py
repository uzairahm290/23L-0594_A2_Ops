"""Task 3.6 - register runs as versions of one model and point the 'champion' alias at the best.

  python scripts/register_model.py <run_id_v1> <run_id_v2> ... --champion <run_id>
Every run must have logged a 'model' MLflow artifact (train.py does this in manual mode).
"""
import argparse
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visionguard import tracking  # noqa: E402

NAME = "23L-0594_A2_Ops_visionguard"

ap = argparse.ArgumentParser()
ap.add_argument("run_ids", nargs="+")
ap.add_argument("--champion", required=True)
a = ap.parse_args()

tracking.setup("23L-0594_A2_Ops")
client = MlflowClient()
champ_version = None
for rid in a.run_ids:
    mv = mlflow.register_model(f"runs:/{rid}/model", NAME)
    print(f"registered run {rid[:8]} as {NAME} v{mv.version}")
    if rid == a.champion:
        champ_version = mv.version
client.set_registered_model_alias(NAME, "champion", champ_version)
print(f"alias 'champion' -> v{champ_version}")
for v in client.search_model_versions(f"name='{NAME}'"):
    print(f"  v{v.version}  run={v.run_id[:8]}  aliases={v.aliases}")
