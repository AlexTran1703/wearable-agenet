"""Beat-level metrics for N / S / V: per class precision (+P), sensitivity (Se), specificity (Sp), F1, AUROC,
PR-AUC (one-vs-rest on the class probabilities); macro-F1, MCC, balanced accuracy."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

CLASSES = ("N", "S", "V")


def per_class(y, p):
    out = {}
    for k, c in enumerate(CLASSES):
        tp = int(((y == k) & (p == k)).sum()); fp = int(((y != k) & (p == k)).sum())
        fn = int(((y == k) & (p != k)).sum()); tn = int(len(y) - tp - fp - fn)
        se = tp / (tp + fn) if tp + fn else np.nan
        pp = tp / (tp + fp) if tp + fp else np.nan
        sp = tn / (tn + fp) if tn + fp else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
        out[c] = {"P": pp, "Se": se, "Sp": sp, "F1": f1, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "n": int((y == k).sum())}
    return out


def macro_f1(y, p):
    return float(np.nanmean([per_class(y, p)[c]["F1"] for c in CLASSES]))


def full(y, prob):
    p = prob.argmax(1)
    out = {"per_class": per_class(y, p), "macro_F1": macro_f1(y, p), "MCC": float(matthews_corrcoef(y, p)),
           "balanced_acc": float(np.nanmean([per_class(y, p)[c]["Se"] for c in CLASSES])), "n": int(len(y))}
    for k, c in enumerate(CLASSES):
        t = (y == k).astype(int)
        if 0 < t.sum() < len(t):
            out["per_class"][c]["AUROC"] = float(roc_auc_score(t, prob[:, k]))
            out["per_class"][c]["PR_AUC"] = float(average_precision_score(t, prob[:, k]))
    return out
