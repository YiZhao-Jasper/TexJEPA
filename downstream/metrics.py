"""Evaluation metrics for multi-label chest X-ray classification."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_multilabel_metrics(y_true, y_score, class_names=None):
    """Compute per-class and mean AUC-ROC and mAP for multi-label classification.

    Args:
        y_true: np.ndarray of shape [N, C], binary ground truth.
        y_score: np.ndarray of shape [N, C], predicted probabilities.
        class_names: optional list of class names for reporting.

    Returns:
        dict with 'mean_auc', 'mean_ap', and per-class metrics.
    """
    n_classes = y_true.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    aucs, aps = [], []
    per_class = {}

    for i in range(n_classes):
        gt = y_true[:, i]
        sc = y_score[:, i]
        n_pos = gt.sum()

        if n_pos == 0 or n_pos == len(gt):
            per_class[class_names[i]] = {"auc": float("nan"), "ap": float("nan"),
                                         "n_pos": int(n_pos), "skipped": True}
            continue

        auc = roc_auc_score(gt, sc)
        ap = average_precision_score(gt, sc)
        aucs.append(auc)
        aps.append(ap)
        per_class[class_names[i]] = {"auc": auc, "ap": ap, "n_pos": int(n_pos),
                                     "skipped": False}

    mean_auc = np.mean(aucs) if aucs else float("nan")
    mean_ap = np.mean(aps) if aps else float("nan")

    return {
        "mean_auc": mean_auc,
        "mean_ap": mean_ap,
        "n_evaluated_classes": len(aucs),
        "per_class": per_class,
    }
