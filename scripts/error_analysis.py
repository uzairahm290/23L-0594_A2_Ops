"""Task 2.6 - grid of 10 misclassified test images (most confident mistakes first).
Write your hypothesis per image into reports/error_hypotheses.csv afterwards.

  python scripts/error_analysis.py models/<run_name>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

run_dir = Path(sys.argv[1])
df = pd.read_csv(run_dir / "test_predictions.csv")
wrong = df[df.label != df.pred].copy()
if wrong.empty:
    sys.exit("no misclassified test images - nothing to analyse")
wrong["confidence_wrong"] = (wrong.p_ai - 0.5).abs()
wrong = wrong.sort_values("confidence_wrong", ascending=False).head(10)

fig, axes = plt.subplots(2, 5, figsize=(16, 7.5))
rows = []
for i, (ax, (_, r)) in enumerate(zip(axes.ravel(), wrong.iterrows()), start=1):
    with Image.open(r.path) as im:
        im.seek(0)
        w, h = im.size
        ax.imshow(im.convert("RGB"))
    truth = "AI" if r.label else "real"
    ax.set_title(f"#{i} true {truth}, p(AI)={r.p_ai:.2f}\n{w}x{h} {Path(r.path).suffix}", fontsize=9)
    ax.axis("off")
    rows.append({"id": i, "path": r.path, "true": truth, "p_ai": round(r.p_ai, 3), "size": f"{w}x{h}",
                 "hypothesis": ""})
for ax in axes.ravel()[len(wrong):]:
    ax.axis("off")
fig.suptitle(f"Misclassified test images - {run_dir.name}")
fig.tight_layout()
fig.savefig(run_dir / "figures" / "misclassified.png", dpi=120)
pd.DataFrame(rows).to_csv(run_dir / "error_hypotheses.csv", index=False)
print(f"{len(df[df.label != df.pred])} test errors; top {len(wrong)} saved to {run_dir/'figures/misclassified.png'}")
print(pd.DataFrame(rows)[["id", "true", "p_ai", "size"]].to_string(index=False))
