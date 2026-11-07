"""Task 2.1 - tf.data input pipeline.

    split CSV -> (path, label)
      -> py_function: read bytes, PIL decode -> RGB -> resize -> [-1,1]   (shared with serving)
      -> cache            decoded tensors kept in RAM after epoch 1 (dataset is ~1k images)
      -> shuffle          train only, reshuffled each epoch
      -> batch
      -> augment          train only, on GPU/CPU as part of the graph
      -> prefetch         overlap input prep with the training step

Why PIL inside py_function instead of tf.io.decode_image: the Kaggle set contains WEBP
and palette GIFs that TF's decoder rejects, and using the exact same function as the
Flask app means there is zero chance of train/serve skew in decoding or resizing.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import IMG_SIZE, ROOT, SEED
from .preprocessing import load_rgb, to_model_input

AUTOTUNE = tf.data.AUTOTUNE


def read_split(name: str, split_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(split_dir / f"{name}.csv")
    df["path"] = df["path"].map(lambda p: str(p if Path(p).is_absolute() else ROOT / p))
    return df


def _load(path, img_size):
    data = Path(path.numpy().decode()).read_bytes()
    return to_model_input(load_rgb(data), int(img_size))[0]


def make_augmenter(strength: str = "light") -> tf.keras.Sequential:
    if strength == "none":
        return tf.keras.Sequential(name="no_aug")
    layers = [tf.keras.layers.RandomFlip("horizontal", seed=SEED),
              tf.keras.layers.RandomRotation(0.05, seed=SEED),
              tf.keras.layers.RandomZoom(0.1, seed=SEED)]
    if strength == "strong":
        layers += [tf.keras.layers.RandomContrast(0.2, value_range=(-1.0, 1.0), seed=SEED),
                   tf.keras.layers.RandomTranslation(0.1, 0.1, seed=SEED)]
    return tf.keras.Sequential(layers, name=f"aug_{strength}")


def make_dataset(df: pd.DataFrame, img_size: int = IMG_SIZE, batch_size: int = 32,
                 training: bool = False, augment: str = "light") -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values.astype(np.float32)))

    def load(path, label):
        img = tf.py_function(_load, [path, img_size], tf.float32)
        img.set_shape((img_size, img_size, 3))
        return img, label

    ds = ds.map(load, num_parallel_calls=AUTOTUNE).cache()
    if training:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if training and augment != "none":
        aug = make_augmenter(augment)
        # augmentation layers expect [0,255]-ish scale-agnostic ops; all ours are geometric
        # or contrast (scale-invariant), so running them on [-1,1] is fine. Clip keeps range.
        ds = ds.map(lambda x, y: (tf.clip_by_value(aug(x, training=True), -1.0, 1.0), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)
