"""Task 3.5 - pull every run of the experiment from MLflow into one table + chart."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visionguard import tracking  # noqa: E402
from visionguard.config import FIG_DIR, PROJECT_NAME, REPORT_DIR  # noqa: E402

tracking.setup(PROJECT_NAME)
df = mlflow.search_runs(experiment_names=[PROJECT_NAME])
df = df[df["metrics.test_f1"].notna()]
if "params.opt_learning_rate" in df:          # autolog names it differently
    df["params.lr"] = df["params.lr"].fillna(df["params.opt_learning_rate"])
cols = {"tags.mlflow.runName": "run", "params.model": "model", "params.lr": "lr",
        "params.augment": "aug", "params.img_size": "size", "params.unfreeze_blocks": "unfrozen",
        "metrics.test_roc_auc": "auc", "metrics.test_f1": "f1", "metrics.test_precision": "precision",
        "metrics.test_recall": "recall", "metrics.test_real_flagged_rate": "real_flagged",
        "metrics.best_val_loss": "best_val_loss", "run_id": "run_id"}
t = df[[c for c in cols if c in df]].rename(columns=cols).sort_values("run")
REPORT_DIR.mkdir(exist_ok=True)
t.to_csv(REPORT_DIR / "run_comparison.csv", index=False)
print(t.drop(columns="run_id").round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4.2))
x = range(len(t))
w = 0.27
for k, (m, c) in enumerate([("auc", "#1f4e79"), ("f1", "#c2410c"), ("real_flagged", "#6b7280")]):
    ax.bar([i + (k - 1) * w for i in x], t[m], w, label=m, color=c)
ax.set_xticks(list(x), t["run"], rotation=20, ha="right", fontsize=8)
ax.set_ylim(0, 1)
ax.legend()
ax.set_title("Test metrics per run (real_flagged: lower is better)")
fig.tight_layout()
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / "run_comparison.png", dpi=130)
print("saved", FIG_DIR / "run_comparison.png")
