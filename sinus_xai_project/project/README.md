# Explainable AI-Based Sinus Disease Detection Using Deep Learning and Transfer Learning

**Research / educational prototype. Not a medical device. Not clinically validated. Does not diagnose patients.**

## Objective
Classify paranasal-sinus CT images as `Healthy` vs `Unhealthy` (labels are taken from the dataset's folder
names; the project makes no claim about a specific disease such as chronic rhinosinusitis). It compares a
majority baseline, two fine-tuned CNNs, and classical ML on deep embeddings, and explains predictions with
Grad-CAM, LIME and SHAP.

## Dataset
Place the extracted dataset inside `data/raw/` (any sub-folder depth; the folder that directly contains the
images is the class). Expected layout of the provided archive:

```
data/raw/DATASET_SININUSE/Healthy classification/*.jpeg|png
data/raw/DATASET_SININUSE/Unhealthy classification/*.jpeg|png
```

Observed when this project was built (re-verify with `--inspect`; nothing is hard-coded):
137 images (37 Healthy, 100 Unhealthy), JPEG/PNG, mixed colour modes (L/P/RGB/RGBA), 250x178 to 1749x1856 px,
CT filenames (coronal/axial hints). **There are no subject IDs - the files are individual, web-style
images, not 137 patients**, and 16 files are byte-identical duplicates.

**Leakage handling.** Because subject IDs are missing, images are grouped (union-find over byte-identical
files, near-identical 16x16 dHash, and same filename stem within a class). Whole groups are assigned to a
single split (`StratifiedGroupKFold`, ~70/15/15, fold 0 = test, fold 1 = validation). The split is checked
for leakage and saved to `data/processed/splits.csv`. Groups are a heuristic proxy for subjects; images
of one patient with unrelated filenames and appearance could still be split.

## Methodology
1. **Preprocessing** - corrupted-file check, any mode -> 3-channel RGB (alpha composited on black), resize
   224x224, ImageNet normalisation.
2. **Augmentation (train only)** - rotation +-10 deg, translation 5%, brightness/contrast 0.2. Horizontal
   flip is available (`AUG_HFLIP`) but off by default. Validation/test use deterministic transforms.
3. **Class imbalance** - inverse-frequency class-weighted loss (weights from the train split); class weights
   / `scale_pos_weight` in classical models. No SMOTE, no synthetic images.
4. **Baseline** - majority class of the training split.
5. **CNNs** - ImageNet-pretrained ResNet50 and EfficientNet-B0. Stage 1: frozen backbone, train head.
   Stage 2: unfreeze `layer4` (ResNet50) / last two blocks (EfficientNet-B0) at LR 1e-5. AdamW,
   ReduceLROnPlateau, early stopping on validation loss, BatchNorm kept in eval mode. Checkpoints:
   `models/resnet50_best.pth`, `models/efficientnet_b0_best.pth` (plus `*_frozen.pth`, `*_finetuned.pth`
   for the ablation).
6. **Embeddings** - 2048-d ResNet50 penultimate-layer features for train/val/test
   (`data/processed/embeddings/*.npz`).
7. **Classical ML** - Logistic Regression, Random Forest, SVM, XGBoost, LightGBM on the embeddings.
   Small grid search with group-aware 5-fold CV on the **training split only**.
8. **Feature selection** - SelectKBest (ANOVA F), L1/LASSO, Random-Forest importance, XGBoost importance
   (top 64 features each) and PCA (95% variance), compared with the full embeddings. Selectors live inside
   the CV pipeline, so they are fit on training folds only.
9. **Ablation** - A frozen ResNet50, B fine-tuned ResNet50, C embeddings + best classical model (by
   train-CV), D full embeddings, E selected embeddings (selector chosen by train-CV).

## Explainability
- **Grad-CAM** (ResNet50 `layer4`): original / heatmap / overlay for the most confident TP, TN, FP, FN test
  images -> `results/gradcam/`. Shows where the model was sensitive; it does **not** show disease location or causality.
- **LIME**: superpixel explanations for a small subset (`LIME_N_IMAGES`, `LIME_NUM_SAMPLES` in config) -> `results/figures/lime_*.png`.
- **SHAP**: applied to the classical models on the embeddings (Tree/Linear/Kernel explainers). It explains how
  abstract embedding dimensions move a classical model's output - **not image regions** - and implies no causality.
  Outputs: `shap_global_*.png`, `shap_summary_*.png`, `shap_sample_*.png`, `results/tables/shap_importance_*.csv`.
- **Error analysis**: `results/tables/error_analysis.csv` (TP/TN/FP/FN, actual, predicted, confidence) and
  Grad-CAM panels of the most confident errors in `results/figures/error_analysis/`. No medical interpretation is attached.

## Metrics
Accuracy, precision, recall, sensitivity, specificity, F1, ROC-AUC, PR-AUC (positive class = disease
class, threshold 0.5), confusion matrices, ROC and PR curves. Reported on the held-out **test** split only;
the validation split is used for early stopping and checkpoint choice. Outputs: `results/tables/model_comparison.csv`
(headline: baseline, ResNet50, EfficientNet-B0, classical models on full embeddings),
`model_comparison_val.csv`, `feature_selection_comparison.csv`, `ablation_study.csv`, `classical_cv_results.csv`.

## Install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```
For GPU, install the CUDA build of PyTorch from pytorch.org first. ImageNet weights are downloaded on first
use (internet required once); use `--no-pretrained` to skip (much worse performance).

## Run
```bash
python scripts/run_pipeline.py --inspect              # dataset report + leakage-safe split
python scripts/run_pipeline.py --train-resnet
python scripts/run_pipeline.py --train-efficientnet
python scripts/run_pipeline.py --extract-embeddings
python scripts/run_pipeline.py --train-classical
python scripts/run_pipeline.py --evaluate
python scripts/run_pipeline.py --explain              # Grad-CAM, error analysis, LIME, SHAP
python scripts/run_pipeline.py --all                  # everything, in order
```
Existing checkpoints/embeddings/results are reused; add `--force` to recompute. Optional:
`--batch-size 8 --epochs-head 10 --epochs-finetune 10`. All other settings are in `src/config.py`.
`python main.py --all` is equivalent.

## Project structure
```
data/raw/ data/processed/      dataset (not committed), splits.csv, embeddings
src/config.py                  all settings          src/dataset.py   discovery, grouping, split, loaders
src/preprocessing.py           loading/transforms    src/models.py    ResNet50, EfficientNet-B0
src/train.py                   2-stage training      src/embeddings.py  feature extraction
src/classical_ml.py            LR/RF/SVM/XGB/LGBM    src/feature_selection.py  selectors
src/evaluate.py                metrics, tables, ablation
src/gradcam.py                 Grad-CAM              src/explainability.py  LIME, SHAP, error analysis
scripts/run_pipeline.py, main.py   CLI
models/ results/{figures,gradcam,tables}/
```

## Limitations
- Tiny, web-sourced dataset (~137 images). The test split holds ~23 images, so every metric has a very wide
  confidence interval and model rankings are not reliable. The validation split has only a handful of
  Healthy images.
- No subject IDs: leakage prevention is heuristic (see above). Image provenance, scanner, windowing and
  annotations may differ between classes, so models can learn shortcuts rather than pathology.
- Embeddings come from a CNN fine-tuned on the training images, so training-set embeddings are
  optimistic for the downstream classical models (they are still evaluated on held-out data).
- Single split, no external validation, no calibration analysis. CT only (no MRI present), mixed views.
- Grad-CAM, LIME and SHAP are post-hoc and are not proof of causality or clinical reasoning.

## Future work
Subject-level data with patient IDs and external validation; repeated/nested CV with confidence intervals;
calibration; additional modalities; **active clinical feature acquisition** (this dataset has no clinical
variables such as eosinophils, CRP, IgE, SNOT-22 or CT scores, so it is intentionally not implemented and no
test costs are assumed); expert review of explanations.

## Disclaimer
For research and education only. Outputs must not be used for diagnosis, treatment or any clinical decision.
