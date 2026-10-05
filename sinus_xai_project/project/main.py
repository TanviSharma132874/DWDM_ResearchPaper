"""Command-line interface for the sinus-disease XAI pipeline."""
import argparse
import sys
import traceback

from src import config
from src.utils import ensure_dirs, set_seed


def _explain():
    from src.explainability import error_analysis, run_lime, run_shap
    from src.gradcam import run_gradcam
    ok = True
    for name, fn in (("Grad-CAM", run_gradcam), ("Error analysis", error_analysis),
                     ("LIME", run_lime), ("SHAP", run_shap)):
        print(f"\n--- {name} ---")
        try:
            fn()
        except Exception:
            ok = False
            print(f"[{name}] FAILED:")
            traceback.print_exc()
    return ok


def build_parser():
    p = argparse.ArgumentParser(description="Explainable sinus disease classification (research prototype)")
    for flag, h in (("--inspect", "inspect data/raw and build the leakage-safe split"),
                    ("--train-resnet", "train ResNet50 (2-stage transfer learning)"),
                    ("--train-efficientnet", "train EfficientNet-B0 (2-stage transfer learning)"),
                    ("--extract-embeddings", "extract ResNet50 embeddings (train/val/test)"),
                    ("--train-classical", "train LR/RF/SVM/XGBoost/LightGBM + feature selection"),
                    ("--evaluate", "metrics, comparison tables, curves, ablation"),
                    ("--explain", "Grad-CAM, error analysis, LIME, SHAP"),
                    ("--all", "run everything in order"),
                    ("--force", "retrain / recompute even if outputs exist")):
        p.add_argument(flag, action="store_true", help=h)
    p.add_argument("--batch-size", type=int, help="override config.BATCH_SIZE")
    p.add_argument("--epochs-head", type=int, help="override config.EPOCHS_HEAD")
    p.add_argument("--epochs-finetune", type=int, help="override config.EPOCHS_FINETUNE")
    p.add_argument("--no-pretrained", action="store_true", help="random init (no weight download)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.epochs_head:
        config.EPOCHS_HEAD = args.epochs_head
    if args.epochs_finetune:
        config.EPOCHS_FINETUNE = args.epochs_finetune
    if args.no_pretrained:
        config.USE_PRETRAINED = False
    if not any(v for k, v in vars(args).items() if k.replace("_", "-") in
               ("inspect", "train-resnet", "train-efficientnet", "extract-embeddings",
                "train-classical", "evaluate", "explain", "all")):
        build_parser().print_help()
        return 0
    ensure_dirs(); set_seed()
    from src.dataset import DatasetError
    try:
        if args.inspect or args.all:
            from src.dataset import inspect_dataset
            inspect_dataset()
        if args.train_resnet or args.all:
            from src.train import train_model
            train_model("resnet50", force=args.force)
        if args.train_efficientnet or args.all:
            from src.train import train_model
            train_model("efficientnet_b0", force=args.force)
        if args.extract_embeddings or args.all:
            from src.embeddings import extract_embeddings
            extract_embeddings(force=args.force)
        if args.train_classical or args.all:
            from src.classical_ml import train_classical
            train_classical(force=args.force)
        if args.evaluate or args.all:
            from src.evaluate import evaluate_all
            evaluate_all()
        if args.explain or args.all:
            if not _explain():
                return 1
    except (DatasetError, FileNotFoundError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
