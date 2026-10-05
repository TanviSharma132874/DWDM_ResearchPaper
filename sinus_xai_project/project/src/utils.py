"""Seeding, metrics and plotting helpers."""
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
from . import config


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dirs():
    for d in (config.RAW_DIR, config.PROCESSED_DIR, config.EMB_DIR, config.MODELS_DIR,
              config.FIG_DIR, config.GRADCAM_DIR, config.TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Binary metrics; positive class = 1 (disease). Sensitivity == recall of class 1."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)), "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if both else float("nan"),
        "pr_auc": average_precision_score(y_true, y_prob) if both else float("nan"),
    }


def plot_confusion(y_true, y_prob, title, path, class_names=("Healthy", "Disease")):
    cm = confusion_matrix(y_true, (np.asarray(y_prob) >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_curves(entries, roc_path, pr_path, title_suffix=""):
    """entries: list of (label, y_true, y_prob). Overlays ROC and PR curves."""
    for kind, path in (("roc", roc_path), ("pr", pr_path)):
        fig, ax = plt.subplots(figsize=(5, 4.5))
        for label, yt, yp in entries:
            yt = np.asarray(yt)
            if len(np.unique(yt)) < 2:
                continue
            if kind == "roc":
                x, y, _ = roc_curve(yt, yp)
                ax.plot(x, y, label=f"{label} (AUC={roc_auc_score(yt, yp):.2f})")
            else:
                p, r, _ = precision_recall_curve(yt, yp)
                ax.plot(r, p, label=f"{label} (AP={average_precision_score(yt, yp):.2f})")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k:", lw=1)
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
            ax.set_title("ROC " + title_suffix)
        else:
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall " + title_suffix)
        ax.legend(fontsize=6, loc="best")
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_table(df: pd.DataFrame, name: str):
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TABLE_DIR / name
    df.to_csv(path, index=False)
    return path
