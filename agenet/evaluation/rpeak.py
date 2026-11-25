"""R-peak detectors (Pan-Tompkins and Hamilton via NeuroKit2, XQRS via WFDB) on the 120-Hz preprocessed lead,
refined to the largest |amplitude| within +-50 ms, and one-to-one matching to reference beats within +-100 ms."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FS  # noqa: E402

TOL = int(0.100 * FS)
REFINE = int(0.050 * FS)
DETECTORS = ("hamilton", "xqrs", "pantompkins")


def detect(x, name):
    if name == "xqrs":
        from wfdb import processing
        q = processing.XQRS(sig=x, fs=FS); q.detect(verbose=False)
        r = np.asarray(q.qrs_inds)
    else:
        import neurokit2 as nk
        method = {"pantompkins": "pantompkins1985", "hamilton": "hamilton2002"}[name]
        _, info = nk.ecg_peaks(x, sampling_rate=FS, method=method)
        r = np.asarray(info["ECG_R_Peaks"])
    r = r[(r >= 0) & (r < len(x))].astype(np.int64)
    ref = np.array([max(0, p - REFINE) + int(np.argmax(np.abs(x[max(0, p - REFINE):p + REFINE + 1]))) for p in r], np.int64)
    return np.unique(ref)


def match(det, ref):
    """One-to-one matching: returns det_of_ref (index into det or -1) for each reference beat."""
    out = np.full(len(ref), -1)
    used = np.zeros(len(det), bool)
    j = 0
    for i, r in enumerate(ref):
        while j < len(det) and det[j] < r - TOL:
            j += 1
        best, bd = -1, TOL + 1
        for k in range(j, len(det)):
            if det[k] > r + TOL:
                break
            if not used[k] and abs(det[k] - r) < bd:
                best, bd = k, abs(det[k] - r)
        if best >= 0:
            out[i] = best; used[best] = True
    return out
