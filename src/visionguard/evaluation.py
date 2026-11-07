"""Task 2.4 / 2.5 - curves, threshold choice, test metrics and plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score, roc_curve)

MAX_REAL_FLAGGED = 0.10   # we tolerate at most 10% of genuine photos being called AI


def choose_threshold(y_val, p_val, max_fpr: float = MAX_REAL_FLAGGED) -> tuple[float, str]:
    """Calling a real photo fake is the worse error for VisionGuard (it accuses an honest
    photographer / journalist). So: among thresholds whose false-positive rate on REAL
    validation images is <= max_fpr, take the one with the highest recall on AI images.
    If no threshold meets the cap (tiny/odd val set) fall back to Youden's J."""
    fpr, tpr, thr = roc_curve(y_val, p_val)
    ok = np.where(fpr <= max_fpr)[0]
    if len(ok):
        i = ok[np.argmax(tpr[ok])]
        rule = f"max recall s.t. FPR<={max_fpr:.2f}"
    else:
        i = int(np.argmax(tpr - fpr))
        rule = "Youden J (FPR cap unreachable)"
    t = float(np.clip(thr[i], 0.0, 1.0))
    return t, rule


def metrics_at(y, p, t) -> dict:
    yhat = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0),
            "f1": f1_score(y, yhat, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
            "real_flagged_rate": fp / max(fp + tn, 1),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def plot_curves(hist: dict, best_epoch: int, stage2_start: int | None, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ep = np.arange(1, len(hist["loss"]) + 1)
    for ax, key in zip(axes, ["loss", "accuracy"]):
        ax.plot(ep, hist[key], label=f"train {key}", color="#1f4e79")
        ax.plot(ep, hist[f"val_{key}"], label=f"val {key}", color="#c2410c")
        ax.axvline(best_epoch, ls="--", color="#555", lw=1)
        ax.annotate(f"best epoch {best_epoch}", (best_epoch, ax.get_ylim()[1]),
                    xytext=(4, -14), textcoords="offset points", fontsize=8)
        if stage2_start:
            ax.axvspan(stage2_start - 0.5, ep[-1] + 0.5, color="#f3e8d2", alpha=0.6, label="fine-tune stage")
        ax.set_xlabel("epoch")
        ax.set_title(key)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(m: dict, path, title):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Oranges")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred real", "pred AI"])
    ax.set_yticks([0, 1], ["true real", "true AI"])
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_roc(y, p, t, path, title):
    fpr, tpr, thr = roc_curve(y, p)
    yhat = p >= t
    op_fpr = (yhat & (y == 0)).sum() / max((y == 0).sum(), 1)
    op_tpr = (yhat & (y == 1)).sum() / max((y == 1).sum(), 1)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fpr, tpr, color="#1f4e79", label=f"AUC = {roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], ls=":", color="#999")
    ax.scatter([op_fpr], [op_tpr], color="#c2410c", zorder=3, label=f"threshold {t:.2f}")
    ax.set_xlabel("false positive rate (real flagged as AI)")
    ax.set_ylabel("true positive rate (AI caught)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
