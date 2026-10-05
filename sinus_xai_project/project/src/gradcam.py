"""Grad-CAM for ResNet50 (target layer: layer4, last convolutional stage)."""
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from . import config
from .dataset import class_names_from, get_loader, get_splits
from .models import load_checkpoint
from .preprocessing import get_transforms, load_image
from .train import predict_loader
from .utils import ensure_dirs

DISCLAIMER = ("Grad-CAM shows where the model's decision was most sensitive. "
              "It is NOT evidence of disease location or causality.")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model, self.acts, self.grads = model, None, None
        for p in model.parameters():
            p.requires_grad_(True)
        self.handle = target_layer.register_forward_hook(self._forward)

    def _forward(self, module, inp, out):
        self.acts = out.detach()
        out.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def remove(self):
        self.handle.remove()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        self.model.zero_grad()
        out = self.model(x)
        prob = torch.softmax(out, 1)[0].detach().cpu().numpy()
        idx = int(out.argmax(1).item()) if class_idx is None else int(class_idx)
        out[0, idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), idx, prob


def overlay(img_rgb, cam, alpha=0.45):
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * img_rgb + alpha * heat), heat


def save_panel(cammer, model, abs_path, title, out_path, device):
    """Save original | heatmap | overlay as one figure."""
    img = load_image(abs_path).resize((config.IMG_SIZE, config.IMG_SIZE))
    x = get_transforms(False)(load_image(abs_path)).unsqueeze(0).to(device)
    cam, idx, prob = cammer(x)
    ov, heat = overlay(np.asarray(img), cam)
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.4))
    for a, im, t in zip(ax, (np.asarray(img), heat, ov), ("Original", "Grad-CAM heatmap", "Overlay")):
        a.imshow(im); a.set_title(t, fontsize=9); a.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.text(0.5, 0.01, DISCLAIMER, ha="center", fontsize=6.5, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)


def resnet_test_table():
    """ResNet50 predictions on the TEST split (deterministic transforms)."""
    if not config.RESNET_CKPT.exists():
        raise FileNotFoundError(f"{config.RESNET_CKPT} not found. Run --train-resnet first.")
    device = config.get_device()
    df = get_splits()
    loader, sub = get_loader(df, "test", shuffle=False)
    model = load_checkpoint("resnet50", config.RESNET_CKPT, device)
    y, p, idx = predict_loader(model, loader, device)
    t = sub.iloc[idx].copy().reset_index(drop=True)
    t["y_prob"] = p
    t["pred"] = (p >= 0.5).astype(int)
    t["conf"] = np.where(t["pred"] == 1, p, 1 - p)
    t["category"] = np.select(
        [(t.label == 1) & (t.pred == 1), (t.label == 0) & (t.pred == 0),
         (t.label == 0) & (t.pred == 1)], ["TP", "TN", "FP"], "FN")
    return t, model, device, class_names_from(df)


def select_representative(table, per_group=config.GRADCAM_PER_GROUP):
    """Most confident examples of each outcome (TP/TN = correct; FP/FN = incorrect)."""
    picks = []
    for cat in ("TP", "TN", "FP", "FN"):
        sub = table[table.category == cat].sort_values("conf", ascending=False)
        picks.append(sub.head(per_group))
    import pandas as pd
    return pd.concat(picks)


def run_gradcam():
    ensure_dirs()
    table, model, device, (neg, pos) = resnet_test_table()
    names = {0: neg, 1: pos}
    cammer = GradCAM(model, model.layer4)
    sel = select_representative(table)
    for _, r in sel.iterrows():
        stem = os.path.splitext(r["filename"])[0][:40]
        title = (f"[{r['category']}] actual={names[int(r['label'])]} | "
                 f"predicted={names[int(r['pred'])]} (conf {r['conf']:.2f})")
        save_panel(cammer, model, r["abs_path"], title,
                   config.GRADCAM_DIR / f"{r['category']}_{stem}.png", device)
    cammer.remove()
    print(f"[gradcam] saved {len(sel)} panels -> {config.GRADCAM_DIR}")
    missing = [c for c in ("TP", "TN", "FP", "FN") if c not in set(sel.category)]
    if missing:
        print(f"[gradcam] no test examples for: {missing}")
