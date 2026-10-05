"""Classical ML on ResNet50 embeddings, with train-only, group-aware cross-validation."""
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .embeddings import load_embeddings
from .feature_selection import make_selector
from .utils import ensure_dirs, save_table

MODEL_NAMES = ["logistic_regression", "random_forest", "svm", "xgboost", "lightgbm"]


def _model_and_grid(name, y_train):
    seed = config.SEED
    if name == "logistic_regression":
        return (LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed),
                {"clf__C": [0.01, 0.1, 1.0]})
    if name == "random_forest":
        return (RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
                {"clf__n_estimators": [200], "clf__max_depth": [None, 5]})
    if name == "svm":
        return (SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
                {"clf__C": [0.1, 1.0, 10.0]})
    spw = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return (XGBClassifier(scale_pos_weight=spw, eval_metric="logloss", random_state=seed, n_jobs=-1),
                {"clf__max_depth": [2, 3], "clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1]})
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return (LGBMClassifier(class_weight="balanced", min_child_samples=5, verbose=-1,
                               random_state=seed, n_jobs=1),
                {"clf__num_leaves": [7, 15], "clf__n_estimators": [100]})
    raise ValueError(name)


def build_pipeline(model_name, method, y_train):
    clf, grid = _model_and_grid(model_name, y_train)
    steps = [("scale", StandardScaler())]
    sel = make_selector(method, y_train)
    if sel is not None:
        steps.append(("sel", sel))
    steps.append(("clf", clf))
    return Pipeline(steps), grid


def train_classical(force=False, methods=None):
    ensure_dirs()
    if config.PRED_CLASSICAL_CSV.exists() and not force:
        print("[classical] results exist; skipping (use --force to retrain)")
        return
    for s in ("train", "val", "test"):
        if not (config.EMB_DIR / f"{s}.npz").exists():
            raise FileNotFoundError("Embeddings missing. Run --extract-embeddings first.")
    Xtr, ytr, ptr, gtr = load_embeddings("train")
    splits = {s: load_embeddings(s) for s in ("val", "test")}
    methods = ["full"] + list(methods or config.SELECTION_METHODS)
    cv = list(StratifiedGroupKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED)
              .split(Xtr, ytr, gtr))                         # CV on TRAIN only, grouped
    pred_rows, cv_rows = [], []
    config.CLASSICAL_DIR.mkdir(parents=True, exist_ok=True)
    for mname in MODEL_NAMES:
        for method in methods:
            try:
                pipe, grid = build_pipeline(mname, method, ytr)
            except ImportError as e:
                print(f"[classical] skipping {mname}/{method}: {e}")
                continue
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=1, refit=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gs.fit(Xtr, ytr)
            n_feat = (gs.best_estimator_[:-1].transform(Xtr[:2]).shape[1])
            print(f"[classical] {mname:20s} {method:5s} cv_auc={gs.best_score_:.3f} "
                  f"features={n_feat} params={gs.best_params_}")
            cv_rows.append(dict(model=mname, variant=method, cv_roc_auc=gs.best_score_,
                                n_features=n_feat, best_params=str(gs.best_params_)))
            for split, (X, y, paths, _) in splits.items():
                prob = gs.predict_proba(X)[:, 1]
                for pth, yt, pr in zip(paths, y, prob):
                    pred_rows.append(dict(model=mname, variant=method, split=split,
                                          path=pth, y_true=int(yt), y_prob=float(pr)))
            if method == "full":
                joblib.dump(gs.best_estimator_, config.CLASSICAL_DIR / f"{mname}__full.joblib")
    save_table(pd.DataFrame(cv_rows), "classical_cv_results.csv")
    pd.DataFrame(pred_rows).to_csv(config.PRED_CLASSICAL_CSV, index=False)
