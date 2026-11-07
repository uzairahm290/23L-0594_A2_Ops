"""Task 2.7 (apply) - Keras Tuner over lr, dropout and number of unfrozen MobileNetV2 blocks.

  python scripts/tune.py --trials 8 --epochs 6
Each trial: frozen head warm-up (3 epochs) then fine-tune with the sampled number of blocks.
Results: reports/tuner_results.csv (+ logged to MLflow as one parent run).
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras_tuner as kt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visionguard import tracking  # noqa: E402
from visionguard.config import PROJECT_NAME, REPORT_DIR, ROOT, SEED, SPLIT_DIR  # noqa: E402
from visionguard.data import make_dataset, read_split  # noqa: E402
from visionguard.models import compile_model, transfer_model, unfreeze_last  # noqa: E402


class VGHyperModel(kt.HyperModel):
    def build(self, hp):
        model, base = transfer_model("mobilenetv2", dropout=hp.Float("dropout", 0.1, 0.5, step=0.1))
        n = hp.Int("unfreeze_blocks", 0, 3)
        if n:
            unfreeze_last(base, "mobilenetv2", n)
        compile_model(model, hp.Choice("lr", [1e-3, 3e-4, 1e-4, 3e-5]))
        return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=6)
    a = ap.parse_args()
    tf.keras.utils.set_random_seed(SEED)
    tr, va = read_split("train", SPLIT_DIR), read_split("val", SPLIT_DIR)
    ds_tr, ds_va = make_dataset(tr, training=True), make_dataset(va)

    tuner = kt.RandomSearch(VGHyperModel(), objective=kt.Objective("val_auc", "max"),
                            max_trials=a.trials, seed=SEED, overwrite=True,
                            directory=str(ROOT / "tuner"), project_name="visionguard")
    tuner.search(ds_tr, validation_data=ds_va, epochs=a.epochs, verbose=2,
                 callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=2)])

    rows = []
    for t in tuner.oracle.get_best_trials(num_trials=a.trials):
        rows.append({"trial": t.trial_id, **t.hyperparameters.values, "val_auc": t.score})
    df = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
    REPORT_DIR.mkdir(exist_ok=True)
    df.to_csv(REPORT_DIR / "tuner_results.csv", index=False)
    print(df.to_string(index=False))
    for hp in ("lr", "dropout", "unfreeze_blocks"):     # crude importance: spread of mean score per value
        g = df.groupby(hp)["val_auc"].mean()
        print(f"{hp:<16} mean val_auc by value: {g.round(4).to_dict()}  spread={g.max() - g.min():.4f}")

    tracking.setup(PROJECT_NAME)
    with mlflow.start_run(run_name="keras_tuner_random_search"):
        mlflow.set_tags(tracking.repro_tags(SEED, {"stage": "tuning"}))
        mlflow.log_params({f"best_{k}": v for k, v in tuner.get_best_hyperparameters()[0].values.items()})
        mlflow.log_metric("best_val_auc", float(df.val_auc.max()))
        mlflow.log_artifact(str(REPORT_DIR / "tuner_results.csv"))


if __name__ == "__main__":
    main()
