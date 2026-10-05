"""Image loading, corruption checks and transforms."""
from PIL import Image
from . import config


def load_image(path):
    """Load any supported image as 3-channel RGB (handles L, P, RGBA, 16-bit)."""
    im = Image.open(path)
    im.load()
    if im.mode in ("I;16", "I", "F"):
        import numpy as np
        a = np.asarray(im, dtype="float64")
        a = (a - a.min()) / max(a.max() - a.min(), 1e-8) * 255
        im = Image.fromarray(a.astype("uint8"))
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (0, 0, 0, 255))   # composite on black (CT background)
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")                               # grayscale -> 3 identical channels


def is_corrupted(path):
    try:
        with Image.open(path) as im:
            im.verify()
        load_image(path)
        return False
    except Exception:
        return True


def get_transforms(train: bool):
    from torchvision import transforms as T
    size = (config.IMG_SIZE, config.IMG_SIZE)
    norm = T.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
    if not train:   # validation / test / inference: deterministic only
        return T.Compose([T.Resize(size), T.ToTensor(), norm])
    ops = [T.Resize(size),
           T.RandomAffine(degrees=config.AUG_ROTATION_DEG,
                          translate=(config.AUG_TRANSLATE, config.AUG_TRANSLATE)),
           T.ColorJitter(brightness=config.AUG_BRIGHTNESS, contrast=config.AUG_CONTRAST)]
    if config.AUG_HFLIP:
        ops.append(T.RandomHorizontalFlip())
    return T.Compose(ops + [T.ToTensor(), norm])


def denormalize(tensor):
    """CHW normalized tensor -> HWC float array in [0,1]."""
    import torch
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
