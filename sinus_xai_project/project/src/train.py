"""Two-stage transfer-learning training (frozen head -> fine-tune upper layers)."""
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from . import config
from .dataset import class_weights, get_loader, get_splits
from .models import (build_model, freeze_backbone, head_parameters, load_checkpoint,
                     set_train_mode, unfreeze_upper)
from .utils import ensure_dirs, save_table, set_seed


@torch.no_grad()
def predict_loader(model, loader, device):
    """Returns (y_true, prob_positive, dataset_indices)."""
    model.eval()
    ys, ps, idx = [], [], []
    for x, y, i in loader:
        p = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
        ps.append(p); ys.append(y.numpy()); idx.append(i.numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(idx)


def _run_epoch(model, loader, criterion, device, optimizer=None):
    if optimizer is not None:
        set_train_mode(model)
    else:
        model.eval()
    tot, n = 0.0, 0
    ys, ps = [], []
    with torch.set_grad_enabled(optimizer is not None):
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if optimizer is not None:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot += loss.item() * len(y); n += len(y)
            ys.append(y.cpu().numpy()); ps.append(torch.softmax(out, 1)[:, 1].detach().cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    acc = float(((p >= 0.5) == y).mean())
    return tot / n, acc, auc


def _stage(model, name, stage, epochs, lr, tr, va, criterion, device, log, ckpt_stage):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best_loss, best_state, bad = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tl, ta, _ = _run_epoch(model, tr, criterion, device, opt)
        vl, va_acc, vauc = _run_epoch(model, va, criterion, device)
        sched.step(vl)
        log.append(dict(model=name, stage=stage, epoch=ep, train_loss=tl, train_acc=ta,
                        val_loss=vl, val_acc=va_acc, val_auc=vauc, lr=opt.param_groups[0]["lr"]))
        print(f"[{name}|{stage}] ep{ep:02d} train_loss={tl:.3f} val_loss={vl:.3f} "
              f"val_acc={va_acc:.2f} val_auc={vauc:.2f} ({time.time()-t0:.0f}s)")
        if vl < best_loss - 1e-4:
            best_loss, bad = vl, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt_stage)
        else:
            bad += 1
            if bad >= config.EARLY_STOP_PATIENCE:
                print(f"[{name}|{stage}] early stopping")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss


def train_model(name, force=False):
    """Trains `name` ('resnet50' | 'efficientnet_b0'). Validation data is used for early
    stopping / checkpoint selection only; test data is never touched here."""
    ensure_dirs(); set_seed()
    final = config.RESNET_CKPT if name == "resnet50" else config.EFFNET_CKPT
    frozen = config.RESNET_FROZEN_CKPT if name == "resnet50" else config.EFFNET_FROZEN_CKPT
    ft = config.RESNET_FT_CKPT if name == "resnet50" else config.MODELS_DIR / "efficientnet_b0_finetuned.pth"
    if final.exists() and not force:
        print(f"[{name}] checkpoint exists ({final.name}); skipping training (use --force to retrain)")
        return final
    device = config.get_device()
    print(f"[{name}] device: {device}")
    df = get_splits()
    tr, _ = get_loader(df, "train")
    va, _ = get_loader(df, "val")
    w, counts = class_weights(df)
    print(f"[{name}] train class counts {counts.astype(int).tolist()} -> loss weights {w.round(2).tolist()}")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w).to(device))
    model = build_model(name).to(device)
    log = []

    freeze_backbone(model, name)
    l1 = _stage(model, name, "head", config.EPOCHS_HEAD, config.LR_HEAD, tr, va, criterion, device, log, frozen)

    unfreeze_upper(model, name)
    l2 = _stage(model, name, "finetune", config.EPOCHS_FINETUNE, config.LR_FINETUNE,
                tr, va, criterion, device, log, ft)

    # best_model = lower validation loss of the two stages (selection on validation only)
    best = ft if (ft.exists() and l2 < l1) else frozen
    model.load_state_dict(torch.load(best, map_location=device))
    torch.save(model.state_dict(), final)
    print(f"[{name}] best stage: {'finetune' if best == ft else 'head'} "
          f"(val_loss head={l1:.3f}, finetune={l2:.3f}) -> saved {final.name}")
    save_table(pd.DataFrame(log), f"training_log_{name}.csv")
    return final
