"""Central configuration. Edit values here instead of hard-coding them elsewhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- paths ----
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
EMB_DIR = PROCESSED_DIR / "embeddings"
SPLIT_CSV = PROCESSED_DIR / "splits.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
GRADCAM_DIR = RESULTS_DIR / "gradcam"
TABLE_DIR = RESULTS_DIR / "tables"

RESNET_CKPT = MODELS_DIR / "resnet50_best.pth"            # best val loss over both stages
RESNET_FROZEN_CKPT = MODELS_DIR / "resnet50_frozen.pth"   # stage 1 only (ablation A)
RESNET_FT_CKPT = MODELS_DIR / "resnet50_finetuned.pth"    # stage 2 only (ablation B)
EFFNET_CKPT = MODELS_DIR / "efficientnet_b0_best.pth"
EFFNET_FROZEN_CKPT = MODELS_DIR / "efficientnet_b0_frozen.pth"
PRED_CLASSICAL_CSV = TABLE_DIR / "predictions_classical.csv"
PRED_CNN_CSV = TABLE_DIR / "predictions_cnn.csv"
CLASSICAL_DIR = MODELS_DIR / "classical"

# ---- data ----
IMG_SIZE = 224
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
# Positive (disease) class. None = auto-detect from folder names (e.g. "Unhealthy").
POSITIVE_CLASS = None
# Near-duplicate grouping (no subject IDs exist in this dataset -> see README).
DHASH_MAX_DIST = 10          # of 256 bits (16x16 dHash)
GROUP_BY_FILENAME_STEM = True

# ---- augmentation (training only) ----
AUG_ROTATION_DEG = 10
AUG_TRANSLATE = 0.05
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.2
# Off by default: lateral markers/annotations may differ and laterality is not verified.
AUG_HFLIP = False

# ---- training ----
BATCH_SIZE = 16
NUM_WORKERS = 0              # 0 is safest on Windows
EPOCHS_HEAD = 15             # stage 1 (frozen backbone)
EPOCHS_FINETUNE = 15         # stage 2 (upper layers)
LR_HEAD = 1e-3
LR_FINETUNE = 1e-5
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 5
DROPOUT = 0.3
USE_PRETRAINED = True        # downloads ImageNet weights on first use

# ---- classical ML / feature selection ----
CV_FOLDS = 5
N_SELECTED_FEATURES = 64
PCA_VARIANCE = 0.95
SELECTION_METHODS = ["kbest", "l1", "rf", "xgb", "pca"]

# ---- explainability ----
GRADCAM_PER_GROUP = 2        # per TP/TN/FP/FN
LIME_N_IMAGES = 6
LIME_NUM_SAMPLES = 500
SHAP_N_BACKGROUND = 10
SHAP_N_REPRESENTATIVE = 3


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
