"""Record preprocessing at 120 Hz.

Lead II (MIT-BIH MLII, INCART II; SVDB: the more Lead-II-like of ECG1 / ECG2, see lead_ii_like) in mV -> 2nd-order Butterworth 0.5-40 Hz, zero-phase, at the native rate ->
polyphase resampling to 120 Hz. Expert beat annotations give the R positions (refined to the largest |amplitude|
within +-3 samples). All beat annotations are kept for the R-peak series; only N / S / V are classified.
Cache: cache/<db>/<rec>.npz with x (signal, float32), rpk (R samples at 120 Hz), y (0 / 1 / 2, -1 = not classified).
"""
from __future__ import annotations

import sys
from math import gcd
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, resample_poly, sosfiltfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import CACHE, DB_ROOT, FS  # noqa: E402
from data.aami_mapping import BEAT_SYMBOLS, label  # noqa: E402
from data.patient_split import DS1, DS2, INCART, svdb_records  # noqa: E402


def lead_ii_like(sig, sos, samples):
    """SVDB channels are unnamed (ECG1 / ECG2). Choose, per record and without labels, the channel that looks most like
    Lead II: the larger median signed amplitude at the annotated beat positions (Lead II has a tall positive R)."""
    best, score = 0, -np.inf
    for c in range(sig.shape[1]):
        xc = sosfiltfilt(sos, np.nan_to_num(sig[:, c].astype(np.float64)))
        idx = samples[(samples >= 0) & (samples < len(xc))]
        v = np.median(xc[idx])
        if v > score:
            best, score = c, v
    return best


def load_record(db: str, rec: str):
    r = wfdb.rdrecord(str(DB_ROOT[db] / rec))
    names = [s.strip() for s in r.sig_name]
    fs = float(r.fs)
    sos = butter(2, [0.5, 40.0], btype="bandpass", fs=fs, output="sos")
    a = wfdb.rdann(str(DB_ROOT[db] / rec), "atr")
    if db == "svdb":
        ch = lead_ii_like(r.p_signal, sos, a.sample)
    else:
        ch = next(names.index(n) for n in ("MLII", "II") if n in names)
    x = sosfiltfilt(sos, np.nan_to_num(r.p_signal[:, ch].astype(np.float64)))
    g = gcd(FS, int(round(fs)))
    x = resample_poly(x, FS // g, int(round(fs)) // g)
    sym = np.asarray(a.symbol)
    keep = np.isin(sym, list(BEAT_SYMBOLS))
    rpk = np.round(np.asarray(a.sample)[keep] * FS / fs).astype(np.int64)
    sym = sym[keep]
    inside = (rpk >= 0) & (rpk < len(x) - 1)       # eight INCART records have an annotation at a negative sample
    rpk, sym = rpk[inside], sym[inside]
    rpk = np.array([max(0, p - 3) + int(np.argmax(np.abs(x[max(0, p - 3):p + 4]))) for p in rpk], np.int64)
    y = np.array([label(s) for s in sym], np.int64)
    return x.astype(np.float32), rpk, y


def main():
    for db, recs in (("mitdb", DS1 + DS2), ("incart", INCART), ("svdb", svdb_records())):
        (CACHE / db).mkdir(parents=True, exist_ok=True)
        tot = np.zeros(3, int)
        for rec in recs:
            x, rpk, y = load_record(db, rec)
            np.savez_compressed(CACHE / db / f"{rec}.npz", x=x, rpk=rpk, y=y)
            tot += np.bincount(y[y >= 0], minlength=3)
        print(f"{db}: {len(recs)} records, N {tot[0]} S {tot[1]} V {tot[2]}", flush=True)


if __name__ == "__main__":
    main()
