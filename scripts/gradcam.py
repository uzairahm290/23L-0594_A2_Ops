"""Bonus - Grad-CAM for 4 correct and 4 wrong test predictions.

  python scripts/gradcam.py models/<run_name>
Works for the MobileNetV2 transfer model (last conv layer 'out_relu') and the baseline CNN ('relu4').
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visionguard.preprocessing import bytes_to_input  # noqa: E402


def build_grad_model(model):
    """Return f(x) -> (conv_maps, prob). Handles a nested backbone by rebuilding the graph."""
    names = [l.name for l in model.layers]
    if "relu4" in names:                                     # baseline CNN
        conv = model.get_layer("relu4").output
        return tf.keras.Model(model.inputs, [conv, model.output])
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    last_conv = "out_relu" if "out_relu" in [l.name for l in base.layers] else "top_activation"   # MobileNetV2 | EfficientNet
    inner = tf.keras.Model(base.inputs, [base.get_layer(last_conv).output, base.output])
    x = inp = tf.keras.Input(model.input_shape[1:])
    for layer in model.layers[1:]:
        if layer is base:
            conv, x = inner(x)
        else:
            x = layer(x)
    return tf.keras.Model(inp, [conv, x])


def gradcam(gm, x):
    with tf.GradientTape() as tape:
        conv, p = gm(x)
        target = p[:, 0]
    grads = tape.gradient(target, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * weights[:, None, None, :], axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    return tf.image.resize(cam[..., None], x.shape[1:3]).numpy()[..., 0], float(p[0, 0])


run_dir = Path(sys.argv[1])
model = tf.keras.models.load_model(run_dir / "best.keras")
gm = build_grad_model(model)
df = pd.read_csv(run_dir / "test_predictions.csv")
picks = pd.concat([df[df.label == df.pred].sample(4, random_state=1),
                   df[df.label != df.pred].head(4)])
fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.6))
for j, (_, r) in enumerate(picks.iterrows()):
    x = bytes_to_input(Path(r.path).read_bytes(), model.input_shape[1])
    cam, p = gradcam(gm, x)
    img = (x[0] + 1) / 2
    axes[0, j].imshow(img)
    axes[0, j].set_title(f"{'OK' if r.label == r.pred else 'WRONG'} true {'AI' if r.label else 'real'}",
                         fontsize=9, color="#2f6f4f" if r.label == r.pred else "#b91c1c")
    axes[1, j].imshow(img)
    axes[1, j].imshow(cam, cmap="jet", alpha=0.45)
    axes[1, j].set_title(f"p(AI)={p:.2f}", fontsize=9)
    for a in axes[:, j]:
        a.axis("off")
fig.suptitle("Grad-CAM: left 4 correct, right 4 wrong")
fig.tight_layout()
fig.savefig(run_dir / "figures" / "gradcam.png", dpi=120)
print("saved", run_dir / "figures" / "gradcam.png")
