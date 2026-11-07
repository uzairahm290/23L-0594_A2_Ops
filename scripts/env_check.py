"""Task 1.1 - print exact library versions and whether TF sees a GPU."""
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

PKGS = ["tensorflow", "tensorflow-cpu", "keras", "numpy", "pandas", "pillow",
        "matplotlib", "scikit-learn", "mlflow", "keras-tuner", "dvc", "flask"]

print(f"python      {sys.version.split()[0]}  ({platform.platform()})")
for p in PKGS:
    try:
        print(f"{p:<12}{version(p)}")
    except PackageNotFoundError:
        pass

import tensorflow as tf  # noqa: E402

gpus = tf.config.list_physical_devices("GPU")
print(f"\nTF built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"GPUs visible      : {gpus if gpus else 'none - training runs on CPU'}")
