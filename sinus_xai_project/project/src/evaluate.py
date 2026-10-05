"""Evaluation: metrics, comparison tables, confusion matrices, ROC/PR curves, ablation."""
import numpy as np
import pandas as pd

from . import config
from .dataset import class_names_from, get_splits
from .utils import compute_metrics, ensure_dirs, plot_confusion, plot_curves, save_table

CNN_MODELS = {   # key -> (architecture, checkpoint, display name)
    "resnet50": ("resnet50", config.RESNET_CKPT, "ResNet50"),
    "efficientnet_b0": ("efficientnet_b0", config.EFFNET_CKPT, "EfficientNet-B0"),
    "resnet50_frozen": ("resnet50", config.RESNET_FROZEN_CKPT, "ResNet50 (frozen backbone)"),
    "resnet50_finetuned": ("resnet50", config.RESNET_FT_CKPT, "ResNet50 (fine-tuned, stage 2)"),
}
CLASSICAL_DISPLAY = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
                     "svm": "SVM", "xgboost": "XGBoost", "lightgbm": "LightGBM"}


def cnn_predictions():
    """Run CNN checkpoints on val/test with deterministic transforms and cache results."""
    import torch
    from .dataset import get_loader
    from .models import load_checkpoint
    from .train import predict_loader
    device = config.get_device()
    df = get_splits()
    rows = []
    for key, (arch, ckpt, _) in CNN_MODELS.items():
        if not ckpt.exists():
            print(f"[evaluate] missing checkpoint for {key}: skipped")
            continue
        model = load_checkpoint(arch, ckpt, device)
        for split in ("val", "test"):
            loader, sub = get_loader(df, split, shuffle=False)
            y, p, idx = predict_loader(model, loader, device)
            for i, yt, pr in zip(idx, y, p):
                rows.append(dict(model=key, variant="cnn", split=split,
                                 path=sub["path"].iat[int(i)], y_true=int(yt), y_prob=float(pr)))
    out = pd.DataFrame(rows)
    out.to_csv(config.PRED_CNN_CSV, index=False)
    return out


def baseline_predictions(df):
    y_tr = df.loc[df["split"] == "train", "label"]
    maj = int(y_tr.mean() >= 0.5)                      # majority class from TRAIN only
    rows = []
    for split in ("val", "test"):
        for pth, yt in zip(df.loc[df["split"] == split, "path"], df.loc[df["split"] == split, "label"]):
            rows.append(dict(model="majority_baseline", variant="baseline", split=split,
                             path=pth, y_true=int(yt), y_prob=float(maj)))
    return pd.DataFrame(rows), maj


def _metrics_row(sub, label, **extra):
    m = compute_metrics(sub["y_true"], sub["y_prob"])
    return {"model": label, **extra, **m}


def evaluate_all():
    ensure_dirs()
    df = get_splits()
    neg, pos = class_names_from(df)
    base, maj = baseline_predictions(df)
    parts = [base]
    try:
        parts.append(cnn_predictions())
    except ImportError as e:
        print(f"[evaluate] torch unavailable, CNN models skipped: {e}")
    if config.PRED_CLASSICAL_CSV.exists():
        parts.append(pd.read_csv(config.PRED_CLASSICAL_CSV))
    else:
        print("[evaluate] classical predictions not found (run --train-classical)")
    P = pd.concat([p for p in parts if len(p)], ignore_index=True)

    # ---- headline comparison: final models, FULL embeddings for classical (no selection on test)
    order = [("majority_baseline", "baseline", "Majority baseline"),
             ("resnet50", "cnn", "ResNet50"), ("efficientnet_b0", "cnn", "EfficientNet-B0")] + \
            [(m, "full", f"{CLASSICAL_DISPLAY[m]} (ResNet50 embeddings)") for m in CLASSICAL_DISPLAY]
    for split, fname in (("test", "model_comparison.csv"), ("val", "model_comparison_val.csv")):
        rows, curves = [], []
        for mk, var, disp in order:
            sub = P[(P.model == mk) & (P.variant == var) & (P.split == split)]
            if sub.empty:
                continue
            rows.append(_metrics_row(sub, disp))
            if split == "test":
                safe = mk if var in ("baseline", "cnn") else f"{mk}_{var}"
                plot_confusion(sub.y_true, sub.y_prob, disp, config.FIG_DIR / f"confusion_{safe}.png",
                               (neg, pos))
                if mk != "majority_baseline":
                    curves.append((disp, sub.y_true.values, sub.y_prob.values))
        table = pd.DataFrame(rows)
        save_table(table, fname)
        if split == "test" and curves:
            plot_curves(curves, config.FIG_DIR / "roc_curves_test.png",
                        config.FIG_DIR / "pr_curves_test.png", "(test set)")
        print(f"\n=== {fname} ===")
        if not table.empty:
            print(table[["model", "n", "accuracy", "precision", "recall", "specificity", "f1",
                         "roc_auc", "pr_auc"]].round(3).to_string(index=False))

    # ---- full vs reduced embeddings
    cvt = config.TABLE_DIR / "classical_cv_results.csv"
    cv = pd.read_csv(cvt) if cvt.exists() else pd.DataFrame()
    fs_rows = []
    for (mk, var), sub in P[P.variant.isin(["full"] + config.SELECTION_METHODS)].groupby(["model", "variant"]):
        t = sub[sub.split == "test"]
        if t.empty:
            continue
        r = _metrics_row(t, CLASSICAL_DISPLAY.get(mk, mk), variant=var)
        if not cv.empty:
            c = cv[(cv.model == mk) & (cv.variant == var)]
            if len(c):
                r["cv_roc_auc_train"] = c["cv_roc_auc"].iat[0]
                r["n_features"] = c["n_features"].iat[0]
        fs_rows.append(r)
    fs = pd.DataFrame(fs_rows)
    if not fs.empty:
        save_table(fs, "feature_selection_comparison.csv")
        _plot_fs(fs)

    # ---- ablation (A-E). Model/selector choices use TRAIN cross-validation only.
    abl = []
    def add(label, sub):
        if len(sub):
            abl.append(_metrics_row(sub, label))
    T = P[P.split == "test"]
    add("A. Frozen ResNet50 (linear head)", T[T.model == "resnet50_frozen"])
    add("B. Fine-tuned ResNet50 (stage 2)", T[T.model == "resnet50_finetuned"])
    if not cv.empty:
        full_cv = cv[cv.variant == "full"]
        best_m = full_cv.sort_values("cv_roc_auc", ascending=False)["model"].iat[0]
        add(f"C. ResNet50 embeddings + classical ML (best by train-CV: {CLASSICAL_DISPLAY[best_m]})",
            T[(T.model == best_m) & (T.variant == "full")])
        for tag, picker in (("D. Full embeddings (mean of classical models)", lambda m: "full"),
                            ("E. Selected embeddings (mean of classical models, selector by train-CV)",
                             lambda m: cv[(cv.model == m) & (cv.variant != "full")]
                             .sort_values("cv_roc_auc", ascending=False)["variant"].iat[0]
                             if len(cv[(cv.model == m) & (cv.variant != "full")]) else None)):
            ms = []
            for m in cv.model.unique():
                v = picker(m)
                sub = T[(T.model == m) & (T.variant == v)] if v else pd.DataFrame()
                if len(sub):
                    ms.append(compute_metrics(sub.y_true, sub.y_prob))
            if ms:
                avg = {k: float(np.nanmean([x[k] for x in ms])) for k in ms[0]}
                abl.append({"model": tag, **avg})
    if abl:
        save_table(pd.DataFrame(abl), "ablation_study.csv")
        print("\n=== ablation_study.csv ===")
        print(pd.DataFrame(abl)[["model", "accuracy", "recall", "specificity", "f1", "roc_auc"]]
              .round(3).to_string(index=False))
    print(f"\nTables -> {config.TABLE_DIR}\nFigures -> {config.FIG_DIR}")
    print("NOTE: test set is small; differences between models are within sampling noise.")


def _plot_fs(fs):
    import matplotlib.pyplot as plt
    piv = fs.pivot_table(index="model", columns="variant", values="roc_auc")
    cols = [c for c in ["full"] + config.SELECTION_METHODS if c in piv.columns]
    ax = piv[cols].plot(kind="bar", figsize=(8, 4))
    ax.set_ylabel("Test ROC-AUC"); ax.set_title("Full vs reduced embeddings (test set)")
    ax.legend(fontsize=7); plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(config.FIG_DIR / "feature_selection_comparison.png", dpi=150); plt.close()
