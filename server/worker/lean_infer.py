"""AGENet-B3 analysis engine for the EC2 server, without PyTorch (NumPy + SciPy + ONNX Runtime; R peaks: Hamilton
(vendored, deployed) or WFDB XQRS, chosen by the "detector" field of models/<run>_stats.json).

Re-implements, line for line, the inference-time input construction of agenet/data/context_windows.Split for the
B3 model (beat window, record median template, 10-s beat tokens with patient-relative timing, RR ratios, standardised
with the training statistics), so a 1-GiB instance can run it. Equivalence with the PyTorch pipeline is checked by
verify_lean.py (labels and probabilities). Assets (shipped next to this file):
  models/<run>.onnx        the locked AGENet-B3 network
  models/<run>_stats.json  training-split standardisation statistics (timing_r, rr4) and the locked QRS detector
"""
from __future__ import annotations

import json
import time
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

HERE = Path(__file__).resolve().parent
RUN = "nv_B3_wce_s1337"
FS = 120                                    # processing rate (Hz)
BEAT_PRE, BEAT_POST = 42, 54                # [-0.35, +0.45] s
CTX_HALF = 5 * FS                           # context fit test (+-5 s inside the record)
HALF = int(round(5.0 * FS))                 # token window +-5 s
MAXB = int(round(6.4 * 5.0))                # 32 tokens
LOCAL_HALF = 10 * FS
REFINE = int(0.050 * FS)                    # detector fiducial refinement +-50 ms
CLASSES = ("N", "S", "V")
INPUTS = ("beat", "tdiff", "rr4", "seq", "seq_tdiff", "timing_r", "len", "ctr")


def _z(a, axis=-1):
    return (a - a.mean(axis, keepdims=True)) / (a.std(axis, keepdims=True) + 1e-6)


def _window_mean(rpk, v, half):
    c = np.r_[0.0, np.cumsum(v)]
    lo, hi = np.searchsorted(rpk, rpk - half, "left"), np.searchsorted(rpk, rpk + half, "right")
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def preprocess(lead_mv: np.ndarray, fs: int) -> np.ndarray:
    x = sosfiltfilt(butter(2, [0.5, 40.0], btype="bandpass", fs=fs, output="sos"), lead_mv)
    g = gcd(FS, fs)
    return resample_poly(x, FS // g, fs // g)


def detect_xqrs(x: np.ndarray) -> np.ndarray:
    from wfdb import processing
    q = processing.XQRS(sig=x, fs=FS)
    q.detect(verbose=False)
    r = np.asarray(q.qrs_inds)
    r = r[(r >= 0) & (r < len(x))].astype(np.int64)
    ref = np.array([max(0, p - REFINE) + int(np.argmax(np.abs(x[max(0, p - REFINE):p + REFINE + 1]))) for p in r], np.int64)
    return np.unique(ref)


def _refine(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    r = r[(r >= 0) & (r < len(x))].astype(np.int64)
    ref = np.array([max(0, p - REFINE) + int(np.argmax(np.abs(x[max(0, p - REFINE):p + REFINE + 1]))) for p in r], np.int64)
    return np.unique(ref)


def _hamilton_peaks(signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Hamilton (2002) QRS detector, vendored verbatim from NeuroKit2 0.3.0 (_ecg_findpeaks_hamilton, itself from
    py-ecg-detectors) so the EC2 worker needs no NeuroKit2/pandas. Equivalence: verify_lean.py --detector hamilton."""
    from bisect import insort
    from collections import deque
    from scipy.signal import lfilter
    signal = np.asarray(signal, dtype=float)
    diff = np.abs(np.diff(signal))
    b = np.ones(int(0.08 * sampling_rate)) / int(0.08 * sampling_rate)
    ma = lfilter(b, [1], diff)
    ma[0: len(b) * 2] = 0
    n_pks, s_pks, RR = deque([], maxlen=8), deque([], maxlen=8), deque([], maxlen=8)
    n_pks_ave = s_pks_ave = RR_ave = th = 0.0
    QRS, idx, peaks = [0], [], []
    for i in range(1, len(ma) - 1):
        if ma[i - 1] < ma[i] and ma[i + 1] < ma[i]:
            peak = i
            peaks.append(peak)
            if ma[peak] > th and (peak - QRS[-1]) > 0.3 * sampling_rate:
                QRS.append(peak)
                idx.append(peak)
                s_pks.append(ma[peak])
                s_pks_ave = np.mean(s_pks)
                if RR_ave != 0.0 and QRS[-1] - QRS[-2] > 1.5 * RR_ave:
                    missed_peaks = peaks[idx[-2] + 1: idx[-1]]
                    missed_peak_added = False
                    for missed_peak in missed_peaks:
                        if missed_peak - peaks[idx[-2]] > int(0.36 * sampling_rate) and ma[missed_peak] > 0.5 * th:
                            insort(QRS, missed_peak)
                            missed_peak_added = True
                            break
                    if missed_peak_added:
                        continue
                if len(QRS) > 2:
                    RR.append(QRS[-1] - QRS[-2])
                    RR_ave = int(np.mean(RR))
            else:
                n_pks.append(ma[peak])
                n_pks_ave = np.mean(n_pks)
            th = n_pks_ave + 0.45 * (s_pks_ave - n_pks_ave)
    QRS.pop(0)
    return np.array(QRS, dtype="int")


def detect_hamilton(x: np.ndarray) -> np.ndarray:
    return _refine(x, _hamilton_peaks(x, FS))


DETECTORS = {"xqrs": lambda x: detect_xqrs(x), "hamilton": lambda x: detect_hamilton(x)}


def features(x: np.ndarray, rpk: np.ndarray, stats: dict, y=None):
    """Inputs of AGENet-B3 for every beat whose 10-s context lies inside the record (as in training)."""
    x = x.astype(np.float32)
    y = np.zeros(len(rpk), np.int64) if y is None else y
    pad = CTX_HALF + max(BEAT_PRE, BEAT_POST)
    xp = np.pad(x, (pad, pad))
    n = len(rpk)
    rr = np.diff(rpk) / FS
    pre, post = np.r_[rr[0], rr], np.r_[rr, rr[-1]]
    med = np.median(rr)
    local = _window_mean(rpk, pre, LOCAL_HALF)
    W = _z(xp[(rpk + pad)[:, None] + np.arange(-BEAT_PRE, BEAT_POST)[None]])
    T = np.median(W, 0, keepdims=True)
    D = W - T
    ok = (y >= 0) & (rpk - CTX_HALF >= 0) & (rpk + CTX_HALF <= len(x))
    ok[[0, -1]] = False
    i = np.where(ok)[0]
    ar = np.arange(MAXB)
    lo = np.searchsorted(rpk, rpk[i] - HALF, "left")
    hi = np.searchsorted(rpk, rpk[i] + HALF, "right")
    J = lo[:, None] + ar[None]
    valid = J < hi[:, None]
    for r in np.where(hi - lo > MAXB)[0]:
        j = np.arange(lo[r], hi[r])
        J[r] = np.sort(j[np.argsort(np.abs(rpk[j] - rpk[i[r]]))[:MAXB]])
    Jc = np.minimum(J, n - 1)
    ln = valid.sum(1)
    ctr = np.argmax(J == i[:, None], 1)
    offs = (rpk[Jc] - rpk[i][:, None]) / HALF
    lg = np.log
    tr = np.stack([lg(pre[Jc] / med), lg(post[Jc] / med), lg(pre[Jc] / local[Jc]), offs], -1).astype(np.float32)
    tr[~valid] = 0
    rr4 = np.stack([lg(pre[i] / med), lg(post[i] / med), lg(pre[i] / post[i]), lg(pre[i] / local[i])], 1)
    mask = ln[:, None] > ar[None]
    m, s = np.asarray(stats["timing_r"][0]), np.asarray(stats["timing_r"][1])
    tr = (tr - m) / s
    tr[~mask] = 0
    rr4 = (rr4 - np.asarray(stats["rr4"][0])) / np.asarray(stats["rr4"][1])
    seq_idx = np.where(valid, J, -1)
    sv = (seq_idx >= 0)[..., None]
    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)   # noqa: E731
    feeds = {"beat": f32(W[i]), "tdiff": f32(D[i]), "rr4": f32(rr4),
             "seq": f32(W[np.maximum(seq_idx, 0)] * sv), "seq_tdiff": f32(D[np.maximum(seq_idx, 0)] * sv),
             "timing_r": f32(tr), "len": ln.astype(np.int64), "ctr": ctr.astype(np.int64)}
    return feeds, i


class LeanEngine:
    def __init__(self, run: str = RUN, threads: int = 1):
        import onnxruntime as ort
        t0 = time.perf_counter()
        meta = json.loads((HERE / "models" / f"{run}_stats.json").read_text())
        self.stats, self.detector, self.run = meta["stats"], meta.get("detector", "xqrs"), run
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(HERE / "models" / f"{run}.onnx"), so, providers=["CPUExecutionProvider"])
        self.load_ms = 1e3 * (time.perf_counter() - t0)

    def analyse(self, payload: bytes, fs: int, n_leads: int, lead: int = 0, uv_per_count: float = 1.0):
        t = {}
        t0 = time.perf_counter()
        a = np.frombuffer(payload, "<i2").reshape(-1, n_leads)[:, lead].astype(np.float64) * uv_per_count / 1000.0
        x = preprocess(a, fs)
        t["preprocess_ms"] = 1e3 * (time.perf_counter() - t0); t1 = time.perf_counter()
        rpk = DETECTORS[self.detector](x)
        t["qrs_ms"] = 1e3 * (time.perf_counter() - t1); t1 = time.perf_counter()
        if len(rpk) < 3:
            return {"error": "fewer than 3 beats detected", "timing": t}
        feeds, idx = features(x, rpk, self.stats)
        t["features_ms"] = 1e3 * (time.perf_counter() - t1); t1 = time.perf_counter()
        prob = self.sess.run(None, feeds)[0] if len(idx) else np.zeros((0, 3), np.float32)
        t["inference_ms"] = 1e3 * (time.perf_counter() - t1)
        pred = prob.argmax(1)
        r_native = np.round(rpk[idx] * fs / FS).astype(int)
        return {"model": self.run, "runtime": "onnxruntime (no PyTorch)", "detector": self.detector, "fs_in": fs,
                "fs_model": FS, "n_samples": int(len(a)), "n_detected": int(len(rpk)), "n_classified": int(len(pred)),
                "counts": {c: int((pred == k).sum()) for k, c in enumerate(CLASSES)},
                "beats": {"r_sample": r_native.tolist(), "label": [CLASSES[k] for k in pred],
                          "prob": np.round(prob, 4).tolist()},
                "timing": {k: round(v, 2) for k, v in t.items()}}
