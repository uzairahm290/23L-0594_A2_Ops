"""Train one VisionGuard model and log everything to MLflow (local or DagsHub).

Examples
  python scripts/train.py --model cnn --run-name cnn_baseline
  python scripts/train.py --model mobilenetv2 --run-name mnv2_lr1e-3
  python scripts/train.py --model mobilenetv2 --tracking autolog --run-name mnv2_autolog
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visionguard import evaluation as ev  # noqa: E402
from visionguard import tracking  # noqa: E402
from visionguard.config import MODEL_DIR, PROJECT_NAME, SEED, SPLIT_DIR  # noqa: E402
from visionguard.data import make_dataset, read_split  # noqa: E402
from visionguard.models import baseline_cnn, compile_model, transfer_model, unfreeze_last  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenetv2", choices=["cnn", "mobilenetv2", "efficientnetb0"])
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft-lr", type=float, default=1e-5, help="stage-2 LR (transfer only)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--ft-epochs", type=int, default=15)
    ap.add_argument("--unfreeze-blocks", type=int, default=1, help="0 = never unfreeze")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--augment", default="light", choices=["none", "light", "strong"])
    ap.add_argument("--tracking", default="manual", choices=["manual", "autolog"])
    ap.add_argument("--experiment", default=PROJECT_NAME)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--register", default=None, help="register the model under this name")
    return ap.parse_args()


def callbacks(ckpt: Path, extra):
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1),
        *extra,
    ]


def merge(h1: dict, h2: dict | None) -> dict:
    if not h2:
        return h1
    return {k: list(h1.get(k, [])) + list(h2.get(k, [])) for k in h1}


def main():
    a = parse()
    tf.keras.utils.set_random_seed(SEED)          # python, numpy and TF seeds in one call
    run_name = a.run_name or f"{a.model}_lr{a.lr}_{a.augment}_{a.img_size}"
    out = MODEL_DIR / run_name
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)

    tr, va, te = (read_split(s, SPLIT_DIR) for s in ("train", "val", "test"))
    ds_tr = make_dataset(tr, a.img_size, a.batch_size, training=True, augment=a.augment)
    ds_va = make_dataset(va, a.img_size, a.batch_size)
    ds_te = make_dataset(te, a.img_size, a.batch_size)

    uri = tracking.setup(a.experiment)
    print(f"MLflow tracking -> {uri}")
    if a.tracking == "autolog":
        # autolog captures params of fit()/optimizer, per-epoch metrics, model summary and
        # the model itself, but knows nothing about our data version, threshold or test set.
        mlflow.tensorflow.autolog(log_models=True, log_datasets=False, checkpoint=False)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tracking.repro_tags(SEED, {"model_type": a.model, "tracking_mode": a.tracking}))
        params = vars(a).copy()
        params.update(train_n=len(tr), val_n=len(va), test_n=len(te))
        if a.tracking == "manual":
            mlflow.log_params(params)
        else:   # autolog logs fit/optimizer params; we only add what it can't see
            mlflow.log_params({k: params[k] for k in ("model", "img_size", "augment", "dropout",
                                                       "unfreeze_blocks", "ft_lr", "train_n", "val_n", "test_n")})

        manual_cb = lambda off: [tracking.MLflowEpochLogger(off)] if a.tracking == "manual" else []  # noqa: E731

        # ---------------- stage 1 ----------------
        if a.model == "cnn":
            model, base = baseline_cnn(a.img_size, a.dropout), None
        else:
            model, base = transfer_model(a.model, a.img_size, a.dropout)
        compile_model(model, a.lr)
        (out / "model_summary.txt").unlink(missing_ok=True)
        model.summary(print_fn=lambda s: open(out / "model_summary.txt", "a").write(s + "\n"))
        stage1_trainable = [w.name for w in model.trainable_weights]
        h1 = model.fit(ds_tr, validation_data=ds_va, epochs=a.epochs,
                       callbacks=callbacks(out / "best.keras", manual_cb(0)), verbose=2).history
        n1 = len(h1["loss"])

        # ---------------- stage 2 (fine-tune last block) ----------------
        h2, stage2_layers = None, []
        if base is not None and a.unfreeze_blocks > 0:
            stage2_layers = unfreeze_last(base, a.model, a.unfreeze_blocks)
            compile_model(model, a.ft_lr)      # recompile so the new trainable flags take effect
            h2 = model.fit(ds_tr, validation_data=ds_va, initial_epoch=n1, epochs=n1 + a.ft_epochs,
                           callbacks=callbacks(out / "best.keras", manual_cb(0)), verbose=2).history
        (out / "trainable_layers.json").write_text(json.dumps(
            {"stage1_trainable_weights": stage1_trainable, "stage2_unfrozen_backbone_layers": stage2_layers}, indent=1))

        hist = merge(h1, h2)
        best_epoch = int(np.argmin(hist["val_loss"])) + 1
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_val_loss", float(np.min(hist["val_loss"])))
        print(f"best epoch (lowest val_loss): {best_epoch}")
        ev.plot_curves(hist, best_epoch, n1 + 1 if h2 else None, fig / "curves.png",
                       f"{run_name}: loss / accuracy")

        # restore_best_weights only covers the last fit(); reload the global best checkpoint
        model = tf.keras.models.load_model(out / "best.keras")

        # ---------------- threshold on VAL, then TEST exactly once ----------------
        p_val = model.predict(ds_va, verbose=0).ravel()
        thr, rule = ev.choose_threshold(va.label.values, p_val)
        p_te = model.predict(ds_te, verbose=0).ravel()
        m = ev.metrics_at(te.label.values, p_te, thr)
        mlflow.log_param("threshold", round(thr, 4))
        mlflow.log_param("threshold_rule", rule)
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in m.items()})
        print(json.dumps({"threshold": thr, "rule": rule, **m}, indent=1))

        ev.plot_confusion(m, fig / "confusion_matrix.png", f"{run_name} test @ {thr:.2f}")
        ev.plot_roc(te.label.values, p_te, thr, fig / "roc_curve.png", f"{run_name} test ROC")
        te.assign(p_ai=p_te, pred=(p_te >= thr).astype(int)).to_csv(out / "test_predictions.csv", index=False)
        (out / "serving_config.json").write_text(json.dumps(
            {"model": run_name, "img_size": a.img_size, "threshold": thr, "preprocess": "x/127.5-1",
             "classes": ["real", "ai"], "mlflow_run_id": run.info.run_id}, indent=1))
        tracking.write_requirements(out / "requirements-freeze.txt")

        # ---------------- artifacts ----------------
        mlflow.log_artifacts(str(fig), "figures")
        mlflow.log_artifacts(str(SPLIT_DIR), "splits")
        for f in ("requirements-freeze.txt", "serving_config.json", "test_predictions.csv",
                  "model_summary.txt", "trainable_layers.json"):
            mlflow.log_artifact(str(out / f))
        mlflow.log_artifact(str(out / "best.keras"), "keras_model")
        if a.tracking == "manual" or a.register:
            tracking.log_keras_model(model, a.register)

        pd.DataFrame([{"run": run_name, **m, "threshold": thr, "best_epoch": best_epoch}]).to_csv(
            out / "test_metrics.csv", index=False)
        print(f"run_id={run.info.run_id}  artifacts in {out}")


if __name__ == "__main__":
    main()
