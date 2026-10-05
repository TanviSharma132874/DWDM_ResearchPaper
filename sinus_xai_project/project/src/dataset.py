"""Dataset discovery, inspection, leakage-safe splitting, datasets and loaders."""
import hashlib
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from . import config
from .preprocessing import get_transforms, is_corrupted, load_image

try:
    from torch.utils.data import Dataset
except ImportError:          # allows --inspect without torch installed
    Dataset = object

POSITIVE_HINTS = ("unhealthy", "abnormal", "disease", "pathol", "sinusitis", "polyp", "patient")


class DatasetError(RuntimeError):
    pass


# ----------------------------------------------------------------- discovery
def find_images(root=config.RAW_DIR):
    root = os.fspath(root)
    out = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if os.path.splitext(f)[1].lower() in config.IMG_EXTS:
                out.append(os.path.join(dp, f))
    return sorted(out)


def _clean_class_name(name):
    n = re.sub(r"[\s_\-]*(classification|class|images?|dataset|folder)$", "", name.strip(), flags=re.I)
    return n.strip() or name.strip()


def _detect_classes(paths, root):
    """Class = name of the folder that directly contains the image.
    Valid only if >=2 distinct such folders exist and no image sits directly in root."""
    root = os.fspath(root)
    leaf = {p: os.path.basename(os.path.dirname(p)) for p in paths}
    if any(os.path.dirname(p) == root for p in paths):
        raise DatasetError(
            "Images found directly inside data/raw/ without class folders. Place images in one "
            "sub-folder per class (e.g. data/raw/Healthy/, data/raw/Unhealthy/).")
    names = sorted(set(leaf.values()))
    if len(names) < 2:
        raise DatasetError(
            f"Only one class folder found ({names}). Expected >=2 folders each containing images. "
            "If your layout is class/subject/image, flatten it or adapt _detect_classes().")
    if len(names) != 2:
        raise DatasetError(
            f"Found {len(names)} class folders {names}; this pipeline is binary (e.g. Healthy vs "
            "Unhealthy). Remove extra folders or adapt the labelling in src/dataset.py.")
    return leaf


def _positive_name(clean_names):
    if config.POSITIVE_CLASS:
        if config.POSITIVE_CLASS not in clean_names:
            raise DatasetError(f"config.POSITIVE_CLASS={config.POSITIVE_CLASS!r} not in {clean_names}")
        return config.POSITIVE_CLASS
    for n in clean_names:
        if any(h in n.lower() for h in POSITIVE_HINTS):
            return n
    raise DatasetError(f"Cannot tell which class is the disease class from {clean_names}. "
                       "Set POSITIVE_CLASS in src/config.py.")


# ------------------------------------------------------------ grouping (leakage)
def _dhash(path):
    a = np.asarray(load_image(path).convert("L").resize((17, 16), Image.BILINEAR), dtype=float)
    return (a[:, 1:] > a[:, :-1]).flatten()


def _file_md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _stem(filename):
    s = os.path.splitext(filename)[0].lower().strip()
    return re.sub(r"[\s_\-\(]*\d+\)?$", "", s)


def assign_groups(df):
    """No subject IDs exist, so build groups of related images via union-find on
    (1) identical bytes, (2) near-identical content (16x16 dHash), (3) same filename stem
    within a class. Whole groups go to one split."""
    n = len(df)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    md5 = [_file_md5(p) for p in df["abs_path"]]
    seen = {}
    for i, h in enumerate(md5):
        if h in seen:
            union(i, seen[h])
        seen[h] = i
    hashes = np.stack([_dhash(p) for p in df["abs_path"]])
    for i in range(n):
        d = np.count_nonzero(hashes[i + 1:] != hashes[i], axis=1)
        for j in np.where(d <= config.DHASH_MAX_DIST)[0]:
            # only merge within a class: cross-class near-dups would be label noise, flagged separately
            if df["label"].iat[i] == df["label"].iat[i + 1 + j]:
                union(i, i + 1 + j)
    if config.GROUP_BY_FILENAME_STEM:
        by = defaultdict(list)
        for i, (lab, fn) in enumerate(zip(df["label"], df["filename"])):
            by[(lab, _stem(fn))].append(i)
        for v in by.values():
            for j in v[1:]:
                union(v[0], j)
    roots = {r: k for k, r in enumerate(sorted({find(i) for i in range(n)}))}
    return [f"g{roots[find(i)]:03d}" for i in range(n)], md5


# ----------------------------------------------------------------- dataframe
def build_dataframe(root=config.RAW_DIR, check_corrupt=True):
    paths = find_images(root)
    if not paths:
        raise DatasetError(f"No images found under {root}. Extract the dataset into data/raw/ "
                           f"(supported: {sorted(config.IMG_EXTS)}).")
    leaf = _detect_classes(paths, root)
    clean = {raw: _clean_class_name(raw) for raw in set(leaf.values())}
    pos = _positive_name(sorted(clean.values()))
    rows, bad = [], []
    for p in paths:
        if check_corrupt and is_corrupted(p):
            bad.append(p)
            continue
        cname = clean[leaf[p]]
        with Image.open(p) as im:
            w, h = im.size
            mode = im.mode
        rows.append({"path": os.path.relpath(p, root).replace("\\", "/"), "abs_path": p,
                     "filename": os.path.basename(p), "class_name": cname,
                     "label": int(cname == pos), "width": w, "height": h, "mode": mode,
                     "ext": os.path.splitext(p)[1].lower()})
    df = pd.DataFrame(rows)
    if df["label"].nunique() < 2:
        raise DatasetError("Only one class remained after loading images.")
    df.attrs["bad"] = bad
    df.attrs["positive"] = pos
    df["group"], df["md5"] = assign_groups(df)
    return df


# ------------------------------------------------------------------ splitting
def make_splits(df, seed=config.SEED):
    """Group-aware, stratified ~70/15/15. n_splits = round(1/test_ratio); fold 0 = test,
    fold 1 = validation, rest = train. All images of a group stay together."""
    k = int(round(1.0 / config.TEST_RATIO))
    best = None
    for s in range(seed, seed + 50):      # pick the first seed where every split has both classes
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=s)
        folds = [te for _, te in sgkf.split(df, df["label"], df["group"])]
        split = np.array(["train"] * len(df), dtype=object)
        split[folds[0]] = "test"
        split[folds[1]] = "val"
        ok = all(df.loc[split == sp, "label"].nunique() == 2 for sp in ("train", "val", "test"))
        if ok:
            best = split
            break
    if best is None:
        raise DatasetError("Could not build a group-aware split with both classes in every split.")
    out = df.copy()
    out["split"] = best
    return out


def verify_no_leakage(df):
    for g, sub in df.groupby("group"):
        assert sub["split"].nunique() == 1, f"group {g} spans splits"
    for h, sub in df.groupby("md5"):
        assert sub["split"].nunique() == 1, f"identical file {h} spans splits"
    return True


def get_splits(force=False):
    """Load cached split table (data/processed/splits.csv) or build it."""
    if config.SPLIT_CSV.exists() and not force:
        df = pd.read_csv(config.SPLIT_CSV)
        df["abs_path"] = [os.path.join(os.fspath(config.RAW_DIR), p) for p in df["path"]]
        if all(os.path.exists(p) for p in df["abs_path"]) and \
                set(df["path"]) == {os.path.relpath(p, config.RAW_DIR).replace("\\", "/")
                                    for p in find_images()}:
            return df
    df = make_splits(build_dataframe())
    verify_no_leakage(df)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["abs_path"]).to_csv(config.SPLIT_CSV, index=False)
    return df


def class_names_from(df):
    pos = df.loc[df["label"] == 1, "class_name"].iloc[0]
    neg = df.loc[df["label"] == 0, "class_name"].iloc[0]
    return neg, pos


# ------------------------------------------------------------------ inspection
def inspect_dataset(force_split=True):
    df = build_dataframe()
    neg, pos = class_names_from(df)
    L = []
    L.append("=" * 62)
    L.append("DATASET REPORT")
    L.append("=" * 62)
    L.append(f"Root: {config.RAW_DIR}")
    L.append(f"Readable images: {len(df)}   Corrupted/unreadable: {len(df.attrs['bad'])}")
    for b in df.attrs["bad"]:
        L.append(f"   corrupted: {b}")
    L.append(f"Classes (from folder names): negative='{neg}' (0), positive='{pos}' (1)")
    for name, sub in df.groupby("class_name"):
        L.append(f"   {name}: {len(sub)} images")
    L.append(f"Formats: {dict(Counter(df['ext']))}")
    L.append(f"Colour modes: {dict(Counter(df['mode']))}  (all converted to 3-channel RGB)")
    L.append(f"Size (WxH): min {df['width'].min()}x{df['height'].min()}, "
             f"max {df['width'].max()}x{df['height'].max()}, "
             f"median {int(df['width'].median())}x{int(df['height'].median())}")
    low = " ".join(df["filename"]).lower()
    L.append("Modality/view hints in filenames: "
             f"CT={low.count('ct')>0}, MRI={('mri' in low) or ('mr-' in low)}, "
             f"coronal={low.count('coronal')}, axial={low.count('axial')}")
    L.append("Subject IDs: NOT available (files are individually named images, no patient IDs).")
    dup_files = int(df.duplicated("md5", keep=False).sum())
    ng = df["group"].nunique()
    L.append(f"Byte-identical duplicate files: {dup_files}")
    L.append(f"Related-image groups (md5 + dHash + filename stem): {ng} groups for {len(df)} images")
    for name, sub in df.groupby("class_name"):
        sizes = sorted(sub.groupby("group").size().tolist(), reverse=True)
        L.append(f"   {name}: {len(sizes)} groups, largest sizes {sizes[:8]}")
    L.append("NOTE: 'images' != 'subjects'. Groups are a conservative proxy for subjects; whole groups")
    L.append("      are kept in a single split to avoid duplicate/near-duplicate leakage.")
    sp = make_splits(df)
    verify_no_leakage(sp)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if force_split or not config.SPLIT_CSV.exists():
        sp.drop(columns=["abs_path"]).to_csv(config.SPLIT_CSV, index=False)
    L.append("Split (group-aware, stratified, seed %d) -> %s" % (config.SEED, config.SPLIT_CSV.name))
    for s in ("train", "val", "test"):
        sub = sp[sp["split"] == s]
        L.append(f"   {s:5s}: {len(sub):3d} images, {sub['group'].nunique():3d} groups, "
                 f"{neg}={int((sub['label']==0).sum())}, {pos}={int((sub['label']==1).sum())}")
    L.append("Leakage check (no group or identical file spans splits): PASSED")
    if (sp["split"] == "test").sum() < 40:
        L.append("WARNING: test set is very small -> metrics will have wide confidence intervals.")
    L.append("=" * 62)
    print("\n".join(L))
    return sp


# --------------------------------------------------------- torch dataset/loaders
class ImageDataset(Dataset):
    def __init__(self, df, train=False):
        self.paths = list(df["abs_path"])
        self.labels = list(df["label"])
        self.tf = get_transforms(train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(load_image(self.paths[i])), int(self.labels[i]), i


def get_loader(df, split, shuffle=None, batch_size=None):
    from torch.utils.data import DataLoader
    sub = df[df["split"] == split].reset_index(drop=True)
    train = split == "train"
    ds = ImageDataset(sub, train=train)         # augmentation only for train
    return DataLoader(ds, batch_size=batch_size or config.BATCH_SIZE,
                      shuffle=train if shuffle is None else shuffle,
                      num_workers=config.NUM_WORKERS, drop_last=False), sub


def class_weights(df):
    """Inverse-frequency weights computed on the TRAIN split only."""
    y = df.loc[df["split"] == "train", "label"].values
    counts = np.bincount(y, minlength=2).astype(float)
    return (counts.sum() / (2.0 * counts)).astype(np.float32), counts
