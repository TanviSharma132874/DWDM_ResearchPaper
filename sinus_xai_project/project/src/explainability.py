"""LIME (images), SHAP (classical models on embeddings) and error analysis."""
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .embeddings import load_embeddings
from .utils import ensure_dirs, save_table


# ------------------------------------------------------------------ error analysis
def error_analysis():
    """Categorise ResNet50 test predictions as TP/TN/FP/FN; save table + Grad-CAM for errors."""
    from .gradcam import GradCAM, resnet_test_table, save_panel
    ensure_dirs()
    table, model, device, (neg, pos) = resnet_test_table()
    names = {0: neg, 1: pos}
    out = table[["path", "filename", "class_name", "label", "pred", "y_prob", "conf", "category"]].copy()
    out["actual_class"] = out["label"].map(names)
    out["predicted_class"] = out["pred"].map(names)
    save_table(out.drop(columns=["label", "pred"]), "error_analysis.csv")
    counts = table["category"].value_counts().to_dict()
    print(f"[error analysis] test outcomes: { {k: counts.get(k, 0) for k in ('TP','TN','FP','FN')} }")
    cammer = GradCAM(model, model.layer4)
    d = config.FIG_DIR / "error_analysis"
    for _, r in table[table.category.isin(["FP", "FN"])].sort_values("conf", ascending=False).head(8).iterrows():
        stem = os.path.splitext(r["filename"])[0][:40]
        title = (f"[{r['category']}] actual={names[int(r['label'])]} | "
                 f"predicted={names[int(r['pred'])]} (conf {r['conf']:.2f})")
        save_panel(cammer, model, r["abs_path"], title, d / f"{r['category']}_{stem}.png", device)
    cammer.remove()
    print(f"[error analysis] table -> error_analysis.csv ; error panels -> {d}")
    print("[error analysis] No medical interpretation is attached to errors (requires expert review).")


# ------------------------------------------------------------------------- LIME
def run_lime():
    import torch
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    from .gradcam import resnet_test_table, select_representative
    from .preprocessing import get_transforms, load_image
    ensure_dirs()
    table, model, device, (neg, pos) = resnet_test_table()
    names = {0: neg, 1: pos}
    sel = select_representative(table, per_group=2).head(config.LIME_N_IMAGES)
    tf = get_transforms(False)
    mean = np.array(config.IMAGENET_MEAN, dtype=np.float32)
    std = np.array(config.IMAGENET_STD, dtype=np.float32)

    def batch_predict(imgs):
        x = np.asarray(imgs, dtype=np.float32)
        if x.max() > 1.5:
            x = x / 255.0
        x = torch.from_numpy(((x - mean) / std).transpose(0, 3, 1, 2)).float().to(device)
        with torch.no_grad():
            return torch.softmax(model(x), 1).cpu().numpy()

    explainer = lime_image.LimeImageExplainer(random_state=config.SEED)
    for _, r in sel.iterrows():
        img = np.asarray(load_image(r["abs_path"]).resize((config.IMG_SIZE, config.IMG_SIZE)), dtype=np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp = explainer.explain_instance(img, batch_predict, top_labels=2, hide_color=0,
                                             num_samples=config.LIME_NUM_SAMPLES, batch_size=16,
                                             random_seed=config.SEED)
        lab = int(r["pred"])
        temp, mask = exp.get_image_and_mask(lab, positive_only=True, num_features=5, hide_rest=False)
        fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))
        ax[0].imshow(img); ax[0].set_title("Original", fontsize=9); ax[0].axis("off")
        ax[1].imshow(mark_boundaries(temp.astype(np.float32) / 255.0, mask))
        ax[1].set_title(f"LIME: superpixels supporting '{names[lab]}'", fontsize=9); ax[1].axis("off")
        fig.suptitle(f"[{r['category']}] actual={names[int(r['label'])]} | conf {r['conf']:.2f}", fontsize=9)
        fig.text(0.5, 0.01, "LIME = local surrogate explanation; not proof of causality.",
                 ha="center", fontsize=6.5, style="italic")
        fig.tight_layout(rect=(0, 0.04, 1, 0.92))
        stem = os.path.splitext(r["filename"])[0][:40]
        fig.savefig(config.FIG_DIR / f"lime_{r['category']}_{stem}.png", dpi=150); plt.close(fig)
    print(f"[lime] saved {len(sel)} explanations -> {config.FIG_DIR}")


# ------------------------------------------------------------------------- SHAP
def _pos_values(sv):
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def run_shap():
    """SHAP on the classical models (full ResNet50 embeddings).
    WHAT IT EXPLAINS: how each of the 2048 abstract embedding dimensions pushes a classical model's
    predicted probability of the disease class up/down. Dimensions are NOT anatomical regions, so
    this is NOT an image-level explanation (use Grad-CAM/LIME for that)."""
    import shap
    ensure_dirs()
    Xtr, ytr, _, _ = load_embeddings("train")
    Xte, yte, pte, _ = load_embeddings("test")
    files = sorted(config.CLASSICAL_DIR.glob("*__full.joblib"))
    if not files:
        raise FileNotFoundError("No classical models found. Run --train-classical first.")
    for f in files:
        name = f.name.split("__")[0]
        pipe = joblib.load(f)
        pre, clf = pipe[:-1], pipe[-1]
        A, B = pre.transform(Xtr), pre.transform(Xte)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if name == "logistic_regression":
                sv = shap.LinearExplainer(clf, A).shap_values(B)
            elif name == "svm":
                bg = shap.kmeans(A, config.SHAP_N_BACKGROUND)
                ex = shap.KernelExplainer(lambda z: clf.predict_proba(z)[:, 1], bg)
                sv = ex.shap_values(B, nsamples=200, silent=True)
            else:
                sv = shap.TreeExplainer(clf).shap_values(B)
        sv = _pos_values(sv)
        prob = pipe.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        feat = [f"emb_{i}" for i in range(B.shape[1])]

        # global importance (mean |SHAP|)
        imp = np.abs(sv).mean(0)
        top = np.argsort(imp)[::-1][:15]
        pd.DataFrame({"feature": [feat[i] for i in top], "mean_abs_shap": imp[top]}).to_csv(
            config.TABLE_DIR / f"shap_importance_{name}.csv", index=False)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.barh([feat[i] for i in top][::-1], imp[top][::-1], color="#3b6ea5")
        ax.set_xlabel("mean |SHAP value| (test set)")
        ax.set_title(f"{name}: global importance of embedding dimensions", fontsize=9)
        fig.tight_layout(); fig.savefig(config.FIG_DIR / f"shap_global_{name}.png", dpi=150); plt.close(fig)

        # summary (beeswarm) plot
        plt.figure()
        shap.summary_plot(sv, B, feature_names=feat, max_display=15, show=False)
        plt.title(f"SHAP summary: {name} (embedding dimensions)", fontsize=9)
        plt.savefig(config.FIG_DIR / f"shap_summary_{name}.png", dpi=150, bbox_inches="tight"); plt.close("all")

        # individual explanations: first TP, TN and (if any) an error
        picks = {}
        for i in range(len(yte)):
            cat = ("TP" if yte[i] == 1 and pred[i] == 1 else "TN" if yte[i] == 0 and pred[i] == 0
                   else "FP" if yte[i] == 0 else "FN")
            picks.setdefault(cat, i)
        for cat, i in list(picks.items())[:config.SHAP_N_REPRESENTATIVE + 1]:
            order = np.argsort(np.abs(sv[i]))[::-1][:10][::-1]
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.barh([feat[j] for j in order], sv[i][order],
                    color=["#c0392b" if v > 0 else "#2874a6" for v in sv[i][order]])
            ax.set_title(f"{name} [{cat}] P(disease)={prob[i]:.2f}\n{os.path.basename(str(pte[i]))[:45]}",
                         fontsize=8)
            ax.set_xlabel("SHAP value (red: pushes toward disease class)")
            fig.tight_layout()
            fig.savefig(config.FIG_DIR / f"shap_sample_{name}_{cat}.png", dpi=150); plt.close(fig)
        print(f"[shap] {name}: done")
    print("[shap] NOTE: SHAP explains abstract embedding dimensions of classical models, "
          "not image regions, and does not imply causality.")
