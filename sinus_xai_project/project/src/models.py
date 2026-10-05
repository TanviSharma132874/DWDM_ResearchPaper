"""ResNet50 / EfficientNet-B0 builders, freezing helpers, checkpoint IO."""
import torch
import torch.nn as nn
from torchvision import models
from . import config


def build_model(name, pretrained=None, num_classes=2):
    pretrained = config.USE_PRETRAINED if pretrained is None else pretrained
    if name == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Sequential(nn.Dropout(config.DROPOUT), nn.Linear(m.fc.in_features, num_classes))
    elif name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(config.DROPOUT),
                                     nn.Linear(m.classifier[1].in_features, num_classes))
    else:
        raise ValueError(f"unknown model {name}")
    return m


def head_parameters(model, name):
    return (model.fc if name == "resnet50" else model.classifier).parameters()


def freeze_backbone(model, name):
    """Stage 1: only the classification head is trainable."""
    for p in model.parameters():
        p.requires_grad = False
    for p in head_parameters(model, name):
        p.requires_grad = True


def unfreeze_upper(model, name):
    """Stage 2: head + last residual stage (ResNet50 layer4) / last 2 feature blocks (EffNet-B0)."""
    freeze_backbone(model, name)
    blocks = [model.layer4] if name == "resnet50" else [model.features[-2], model.features[-1]]
    for b in blocks:
        for p in b.parameters():
            p.requires_grad = True


def set_train_mode(model):
    """Train mode but keep BatchNorm layers in eval mode (tiny dataset/batches -> unstable stats)."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def load_checkpoint(name, path, device=None):
    device = device or config.get_device()
    m = build_model(name, pretrained=False)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()


def embedding_model(model):
    """ResNet50 with the final Linear removed -> 2048-d penultimate features."""
    import copy
    m = copy.deepcopy(model)
    m.fc = nn.Identity()
    return m.eval()
