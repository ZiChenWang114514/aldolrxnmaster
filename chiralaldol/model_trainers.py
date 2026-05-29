"""Shared model training functions for AldolRxnMaster."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_sample_weight

from .config import N_CLASSES, N_JOBS


def train_xgb(X_tr, y_tr, X_val=None, y_val=None, n_jobs=N_JOBS):
    """XGBoost with 3-config grid search (balanced sample weights)."""
    sw = compute_sample_weight("balanced", y_tr)
    configs = [
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1,
         "subsample": 0.8, "colsample_bytree": 0.7},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.6},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.15,
         "subsample": 0.9, "colsample_bytree": 0.8},
    ]

    if X_val is None:
        # No validation set — use first config only (for stacking etc.)
        cfg = configs[0].copy()
        cfg.update({"objective": "multi:softprob", "num_class": N_CLASSES,
                    "tree_method": "hist", "random_state": 42, "n_jobs": n_jobs,
                    "verbosity": 0, "gamma": 0.1, "reg_lambda": 1.0})
        m = xgb.XGBClassifier(**cfg)
        m.fit(X_tr, y_tr, sample_weight=sw)
        return m

    best_m, best_acc = None, 0
    for cfg in configs:
        cfg.update({"objective": "multi:softprob", "num_class": N_CLASSES,
                    "tree_method": "hist", "random_state": 42, "n_jobs": n_jobs,
                    "verbosity": 0, "gamma": 0.1, "reg_lambda": 1.0})
        m = xgb.XGBClassifier(**cfg)
        m.fit(X_tr, y_tr, sample_weight=sw)
        acc = balanced_accuracy_score(y_val, m.predict(X_val))
        if acc > best_acc:
            best_acc, best_m = acc, m
    return best_m


def train_et(X_tr, y_tr, X_val=None, y_val=None, n_jobs=N_JOBS):
    """ExtraTrees with balanced class weight."""
    m = ExtraTreesClassifier(n_estimators=300, max_depth=None, random_state=42,
                              n_jobs=n_jobs, class_weight="balanced")
    m.fit(X_tr, y_tr)
    return m


def train_rf(X_tr, y_tr, X_val=None, y_val=None, n_jobs=N_JOBS):
    """RandomForest with balanced class weight."""
    m = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42,
                                n_jobs=n_jobs, class_weight="balanced")
    m.fit(X_tr, y_tr)
    return m


def train_lgbm(X_tr, y_tr, X_val=None, y_val=None, n_jobs=N_JOBS):
    """LightGBM with balanced sample weights."""
    from lightgbm import LGBMClassifier
    sw = compute_sample_weight("balanced", y_tr)
    m = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
                        colsample_bytree=0.7, random_state=42, n_jobs=n_jobs, verbose=-1)
    m.fit(X_tr, y_tr, sample_weight=sw)
    return m


def train_knn(X_tr, y_tr, X_val=None, y_val=None, k=5):
    """KNN baseline."""
    m = KNeighborsClassifier(n_neighbors=k)
    m.fit(X_tr, y_tr)
    return m


class MajorityClassifier:
    """Baseline: always predicts the majority class."""

    def fit(self, X, y, **kw):
        self.majority = int(pd.Series(y).mode()[0])
        self.n_classes = len(np.unique(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.majority)

    def predict_proba(self, X):
        p = np.zeros((len(X), self.n_classes))
        p[:, self.majority] = 1.0
        return p
