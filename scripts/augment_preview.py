"""Task 2.1 figure - one augmented training batch."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visionguard.config import FIG_DIR, SPLIT_DIR  # noqa: E402
from visionguard.data import make_dataset, read_split  # noqa: E402

strength = sys.argv[1] if len(sys.argv) > 1 else "light"
ds = make_dataset(read_split("train", SPLIT_DIR), batch_size=16, training=True, augment=strength)
x, y = next(iter(ds))
fig, axes = plt.subplots(2, 8, figsize=(17, 4.8))
for img, lab, ax in zip(x.numpy(), y.numpy(), axes.ravel()):
    ax.imshow((img + 1) / 2)          # undo [-1,1] scaling for display
    ax.set_title("AI" if lab else "real", fontsize=9, color="#c2410c" if lab else "#2f6f4f")
    ax.axis("off")
fig.suptitle(f"One training batch after '{strength}' augmentation (224x224, stretched)")
fig.tight_layout()
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / f"augmented_batch_{strength}.png", dpi=120)
print("saved", FIG_DIR / f"augmented_batch_{strength}.png", "batch shape", tuple(x.shape))
