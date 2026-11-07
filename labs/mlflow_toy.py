"""Task 3.1 / 3.2 toy run. Tiny Keras model on MNIST, logs params, per-epoch metrics, a plot
and the model.

  python labs/mlflow_toy.py --lr 1e-3          # local  (open with: mlflow ui --port 5000)
  python labs/mlflow_toy.py --lr 1e-2
  python labs/mlflow_toy.py --lr 1e-3 --dagshub   # same, but against DagsHub (reads .env)
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import tensorflow as tf  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--dagshub", action="store_true")
a = ap.parse_args()

if a.dagshub:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])   # creds via MLFLOW_TRACKING_USERNAME/PASSWORD
else:
    mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("23L-0594_A2_Ops_toy")

tf.keras.utils.set_random_seed(42)
(x, y), (xt, yt) = tf.keras.datasets.mnist.load_data()
x, xt = x[:10000] / 255.0, xt[:2000] / 255.0
y, yt = y[:10000], yt[:2000]
model = tf.keras.Sequential([tf.keras.layers.Input((28, 28)), tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(64, activation="relu"),
                             tf.keras.layers.Dense(10, activation="softmax")])
model.compile(tf.keras.optimizers.Adam(a.lr), "sparse_categorical_crossentropy", ["accuracy"])


class LogEpoch(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch + 1)


with mlflow.start_run(run_name=f"toy_lr_{a.lr}"):
    mlflow.set_tags({"purpose": "tool-lab", "roll_number": "23L-0594"})
    mlflow.log_params({"lr": a.lr, "epochs": a.epochs, "units": 64, "train_n": len(x)})
    h = model.fit(x, y, validation_data=(xt, yt), epochs=a.epochs, callbacks=[LogEpoch()], verbose=2)
    plt.plot(h.history["loss"], label="train")
    plt.plot(h.history["val_loss"], label="val")
    plt.legend(); plt.title(f"loss, lr={a.lr}")
    plt.savefig("toy_loss.png"); plt.close()
    mlflow.log_artifact("toy_loss.png", "plots")
    try:
        mlflow.tensorflow.log_model(model, name="model")
    except TypeError:
        mlflow.tensorflow.log_model(model, artifact_path="model")
    print("final val_accuracy:", round(h.history["val_accuracy"][-1], 4))
