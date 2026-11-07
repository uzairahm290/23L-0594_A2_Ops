"""Task 2.2 / 2.3 model definitions."""
import os

import tensorflow as tf
from tensorflow.keras import layers

from .config import IMG_SIZE

BACKBONES = {
    "mobilenetv2": (tf.keras.applications.MobileNetV2, "block_16"),
    "efficientnetb0": (tf.keras.applications.EfficientNetB0, "block7a"),
}


def baseline_cnn(img_size: int = IMG_SIZE, dropout: float = 0.3) -> tf.keras.Model:
    """Four conv blocks, each doubling filters while halving resolution."""
    inp = layers.Input((img_size, img_size, 3), name="image")
    x = inp
    for i, f in enumerate([32, 64, 128, 256], start=1):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)       # stabilises training, allows higher LR
        x = layers.ReLU(name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)            # halve H,W; cheap translation invariance
    x = layers.GlobalAveragePooling2D(name="gap")(x)           # 256 numbers instead of a huge Flatten
    x = layers.Dropout(dropout, name="dropout")(x)             # regularise the small dense head
    x = layers.Dense(64, activation="relu", name="fc")(x)
    out = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return tf.keras.Model(inp, out, name="baseline_cnn")


def transfer_model(backbone: str = "mobilenetv2", img_size: int = IMG_SIZE,
                   dropout: float = 0.3) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Frozen ImageNet backbone + our own head. Inputs are already in [-1,1] which is
    exactly what MobileNetV2.preprocess_input produces, so no extra rescaling layer."""
    cls, _ = BACKBONES[backbone]
    # VG_BACKBONE_WEIGHTS=none only for offline smoke tests
    weights = None if os.getenv("VG_BACKBONE_WEIGHTS", "imagenet") == "none" else "imagenet"
    base = cls(include_top=False, weights=weights, input_shape=(img_size, img_size, 3))
    base.trainable = False
    inp = layers.Input((img_size, img_size, 3), name="image")
    x = inp
    if backbone == "efficientnetb0":
        # EfficientNet has its own Rescaling inside and expects [0,255]
        x = layers.Rescaling(127.5, offset=127.5, name="to_0_255")(x)
    x = base(x, training=False)        # keep BatchNorm in inference mode even when unfrozen
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return tf.keras.Model(inp, out, name=f"{backbone}_transfer"), base


def unfreeze_last(base: tf.keras.Model, backbone: str = "mobilenetv2", n_blocks: int = 1) -> list[str]:
    """Unfreeze the last n blocks of the backbone; BatchNorm layers stay frozen."""
    prefix = BACKBONES[backbone][1]
    if backbone == "mobilenetv2":
        idx = [16 - k for k in range(n_blocks)]
        prefixes = tuple(f"block_{i}_" for i in idx) + ("Conv_1",)
    else:
        prefixes = (prefix, "top_")
    base.trainable = True
    names = []
    for layer in base.layers:
        on = layer.name.startswith(prefixes) and not isinstance(layer, layers.BatchNormalization)
        layer.trainable = on
        if on:
            names.append(layer.name)
    return names


def compile_model(model, lr: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model
