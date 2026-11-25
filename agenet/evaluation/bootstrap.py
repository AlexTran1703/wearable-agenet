"""Record-level bootstrap: resample DS2 records with replacement (2,000 times), pool their beat predictions, recompute
the metric; 95 % percentile interval. The record (patient) is the unit of generalisation, not the correlated beat."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import matthews_corrcoef

from .metrics import per_class

METRICS = {
    "macro_F1": lambda y, p: np.nanmean([per_class(y, p)[c]["F1"] for c in "NSV"]),
    "S_F1": lambda y, p: per_class(y, p)["S"]["F1"], "V_F1": lambda y, p: per_class(y, p)["V"]["F1"],
    "S_Se": lambda y, p: per_class(y, p)["S"]["Se"], "V_Se": lambda y, p: per_class(y, p)["V"]["Se"],
    "MCC": lambda y, p: matthews_corrcoef(y, p),
}


def record_bootstrap(y, p, rec, n=2000, seed=0):
    recs = np.unique(rec)
    idx = {r: np.where(rec == r)[0] for r in recs}
    rng = np.random.default_rng(seed)
    point = {k: float(f(y, p)) for k, f in METRICS.items()}
    boots = {k: [] for k in METRICS}
    for _ in range(n):
        take = np.concatenate([idx[r] for r in rng.choice(recs, len(recs), replace=True)])
        for k, f in METRICS.items():
            boots[k].append(f(y[take], p[take]))
    return {k: {"point": point[k], "lo": float(np.nanpercentile(boots[k], 2.5)),
                "hi": float(np.nanpercentile(boots[k], 97.5))} for k in METRICS}


def paired_bootstrap_delta(y, pa, pb, rec, metric="macro_F1", n=2000, seed=0):
    """Delta = metric(A) - metric(B) on the same resampled records."""
    recs = np.unique(rec)
    idx = {r: np.where(rec == r)[0] for r in recs}
    rng = np.random.default_rng(seed)
    f = METRICS[metric]
    d = []
    for _ in range(n):
        take = np.concatenate([idx[r] for r in rng.choice(recs, len(recs), replace=True)])
        d.append(f(y[take], pa[take]) - f(y[take], pb[take]))
    d = np.array(d)
    return {"delta": float(f(y, pa) - f(y, pb)), "lo": float(np.nanpercentile(d, 2.5)), "hi": float(np.nanpercentile(d, 97.5))}
