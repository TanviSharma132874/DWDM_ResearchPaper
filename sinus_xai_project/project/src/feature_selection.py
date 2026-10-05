"""Feature selectors for deep embeddings. All are sklearn transformers that are placed
INSIDE the CV pipeline, so they are re-fit on training folds only (no val/test leakage)."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from . import config

METHOD_LABELS = {"full": "Full embeddings", "kbest": "SelectKBest (ANOVA F)",
                 "l1": "L1/LASSO", "rf": "RF importance", "xgb": "XGBoost importance",
                 "pca": "PCA"}


def make_selector(method, y_train=None):
    k = config.N_SELECTED_FEATURES
    if method == "full":
        return None
    if method == "kbest":
        return SelectKBest(f_classif, k=k)
    if method == "l1":
        est = LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                                 class_weight="balanced", max_iter=1000)
        return SelectFromModel(est, max_features=k, threshold=-np.inf)
    if method == "rf":
        est = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     random_state=config.SEED, n_jobs=-1)
        return SelectFromModel(est, max_features=k, threshold=-np.inf)
    if method == "xgb":
        from xgboost import XGBClassifier
        est = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                            random_state=config.SEED, eval_metric="logloss", n_jobs=-1)
        return SelectFromModel(est, max_features=k, threshold=-np.inf)
    if method == "pca":
        return PCA(n_components=config.PCA_VARIANCE, svd_solver="full", random_state=config.SEED)
    raise ValueError(method)
