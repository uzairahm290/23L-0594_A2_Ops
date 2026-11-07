"""Task 1.3 - sample grid per class + width/height/file-size histograms per class."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visionguard.config import FIG_DIR, REPORT_DIR, ROOT, SEED  # noqa: E402

COLORS = {"real": "#2f6f4f", "ai": "#c2410c"}


def sample_grid(df, n=8):
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.8))
    for r, cls in enumerate(["real", "ai"]):
        picks = df[df["class"] == cls].sample(min(n, (df["class"] == cls).sum()), random_state=SEED)
        for c, ax in enumerate(axes[r]):
            ax.axis("off")
            if c < len(picks):
                row = picks.iloc[c]
                with Image.open(ROOT / row.path) as im:
                    im.seek(0)
                    ax.imshow(im.convert("RGB"))
                ax.set_title(f"{int(row.width)}x{int(row.height)}", fontsize=8)
        axes[r][0].text(-0.15, 0.5, cls.upper(), transform=axes[r][0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=12, color=COLORS[cls], weight="bold")
    fig.suptitle("Random samples per class (seed 42)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_samples.png", dpi=130)
    plt.close(fig)


def hists(df):
    cols = [("width", "Width (px)", False), ("height", "Height (px)", False),
            ("file_size", "File size (KB, log scale)", True)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (col, title, log) in zip(axes, cols):
        vals_all = df[col] / (1024 if col == "file_size" else 1)
        bins = (np.logspace(np.log10(max(vals_all.min(), 1)), np.log10(vals_all.max()), 40)
                if log else np.linspace(vals_all.min(), vals_all.max(), 40))
        for cls in ["real", "ai"]:
            v = df[df["class"] == cls][col] / (1024 if col == "file_size" else 1)
            ax.hist(v, bins=bins, alpha=0.55, label=f"{cls} (median {v.median():.0f})", color=COLORS[cls])
        if log:
            ax.set_xscale("log")
        ax.set_title(title)
        ax.set_ylabel("images")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_histograms.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REPORT_DIR / "audit.csv")
    df = df[df.readable & ~df.duplicate]
    sample_grid(df)
    hists(df)
    print("saved", FIG_DIR / "eda_samples.png", "and", FIG_DIR / "eda_histograms.png")
