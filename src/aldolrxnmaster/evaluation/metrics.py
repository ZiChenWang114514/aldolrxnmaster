"""Evaluation metrics for 4-class stereochemistry prediction."""

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """Compute comprehensive metrics for 4-class joint prediction."""
    metrics = {}

    # Primary metrics
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["joint_accuracy"] = accuracy_score(y_true, y_pred)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class F1
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, f in enumerate(f1_per):
        metrics[f"f1_class{i}"] = f

    # Marginal accuracies (Ca and Cb separately)
    ca_true = (y_true >= 2).astype(int)
    ca_pred = (y_pred >= 2).astype(int)
    cb_true = (y_true % 2).astype(int)
    cb_pred = (y_pred % 2).astype(int)
    metrics["ca_accuracy"] = accuracy_score(ca_true, ca_pred)
    metrics["cb_accuracy"] = accuracy_score(cb_true, cb_pred)

    # syn/anti accuracy
    sa_true = (ca_true == cb_true).astype(int)
    sa_pred = (ca_pred == cb_pred).astype(int)
    metrics["sa_accuracy"] = accuracy_score(sa_true, sa_pred)

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist()

    return metrics


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            s = metric_fn(y_true[idx], y_pred[idx])
            scores.append(s)
        except Exception:
            continue
    if not scores:
        return 0.0, 0.0, 0.0
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    lo = np.percentile(scores, alpha * 100)
    hi = np.percentile(scores, (1 - alpha) * 100)
    return np.mean(scores), lo, hi


def compute_metrics_with_ci(y_true, y_pred, n_boot=1000):
    """Compute key metrics with bootstrap 95% CI."""
    from functools import partial

    results = {}

    metric_fns = {
        "balanced_accuracy": balanced_accuracy_score,
        "mcc": matthews_corrcoef,
        "joint_accuracy": accuracy_score,
        "f1_macro": partial(f1_score, average="macro", zero_division=0),
    }

    for name, fn in metric_fns.items():
        mean, lo, hi = bootstrap_ci(y_true, y_pred, fn, n_boot=n_boot)
        results[name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}

    return results
