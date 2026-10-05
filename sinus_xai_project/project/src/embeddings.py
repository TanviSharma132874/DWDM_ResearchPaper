"""Extract 2048-d ResNet50 penultimate-layer embeddings for train/val/test."""
import numpy as np

from . import config
from .dataset import get_loader, get_splits
from .utils import ensure_dirs


def extract_embeddings(force=False, ckpt=None):
    import torch
    from .models import embedding_model, load_checkpoint
    ensure_dirs()
    outs = {s: config.EMB_DIR / f"{s}.npz" for s in ("train", "val", "test")}
    if all(p.exists() for p in outs.values()) and not force:
        print("[embeddings] already extracted; skipping (use --force to redo)")
        return outs
    device = config.get_device()
    ckpt = ckpt or config.RESNET_CKPT
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found. Run --train-resnet first.")
    model = embedding_model(load_checkpoint("resnet50", ckpt, device)).to(device)
    df = get_splits()
    for s, path in outs.items():
        loader, sub = get_loader(df, s, shuffle=False)     # deterministic eval transforms
        feats, ys = [], []
        with torch.no_grad():
            for x, y, _ in loader:
                feats.append(model(x.to(device)).cpu().numpy()); ys.append(y.numpy())
        X, y = np.concatenate(feats), np.concatenate(ys)
        np.savez(path, X=X, y=y, paths=sub["path"].values, groups=sub["group"].values)
        print(f"[embeddings] {s}: {X.shape}")
    return outs


def load_embeddings(split):
    d = np.load(config.EMB_DIR / f"{split}.npz", allow_pickle=True)
    return d["X"], d["y"], d["paths"], d["groups"]
