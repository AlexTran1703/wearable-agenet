"""Central-beat + 10-s context dataset, built once per split and kept on the GPU (vectorised per record).

For every classified (N / S / V) central beat b_i whose full 10-s context lies inside its record:
  beat      96 samples, [-0.35, +0.45] s around R_i, z-scored per window
  tdiff     beat minus the record template (median of its beats, optionally robust; no labels)  (tmpl models)
  seq       the beats (any type) with R within +-ctx_s of R_i (default 5 s), in time order, max 6.4*ctx_s, padded; each a beat window (default 96 samples = [-0.35, +0.45] s; win_s = (pre, post) in s)
  timing    per token, absolute: [pre-RR (s), post-RR (s), pre-RR / record median RR, (R_j - R_i) / 5 s]
  timing_r  per token, patient-relative: [log pre/med, log post/med, log pre/local, (R_j - R_i) / 5 s]
  rr3       pre-RR, post-RR, mean of the 10 previous RR (s)                                       (A4)
  rr4       patient-relative RR of b_i: [log pre/med, log post/med, log pre/post, log pre/local]   (rr-injection)
  irr / timing_i  rhythm irregularity of b_i / of every token (see irregularity(); U2)
  med = record median RR; local = mean RR of the beats within +-10 s of the beat.
The label is always the central beat; the 10-s window is context only.
Timing values and RR vectors are standardised with training-split statistics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BEAT_LEN, BEAT_POST, BEAT_PRE, CACHE, CTX_HALF, CTX_LEN, FS, MAX_CTX_BEATS  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_HALF = 10 * FS


def _z(a, axis=-1):
    return (a - a.mean(axis, keepdims=True)) / (a.std(axis, keepdims=True) + 1e-6)


def _window_mean(rpk, v, half):
    """Mean of v over the beats whose R lies within +-half samples of each beat."""
    c = np.r_[0.0, np.cumsum(v)]
    lo, hi = np.searchsorted(rpk, rpk - half, "left"), np.searchsorted(rpk, rpk + half, "right")
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def template(W, robust=False, drop=0.15):
    """Record template from all beat windows, no labels. robust: drop the `drop` fraction of beats farthest (L2) from
    the plain median, then take the median of the rest (limits contamination by frequent ectopic beats)."""
    T0 = np.median(W, 0, keepdims=True)
    if not robust or len(W) < 10:
        return T0
    d = np.linalg.norm(W - T0, axis=1)
    return np.median(W[d <= np.quantile(d, 1 - drop)], 0, keepdims=True)


def irregularity(rpk, pre, half=LOCAL_HALF, thr=0.15):
    """U2: label-free local rhythm irregularity per beat over +-10 s: [log coefficient of variation of the RR,
    fraction of successive RR changing by > 15 %, log pre-RR / local median RR]. Separates an early beat in a regular
    rhythm (SVEB) from a short interval inside an irregular rhythm (e.g. AF) without any rhythm labels."""
    lo, hi = np.searchsorted(rpk, rpk - half, "left"), np.searchsorted(rpk, rpk + half, "right")
    ch = np.r_[0.0, np.abs(np.diff(pre)) / pre[:-1] > thr]
    out = np.empty((len(rpk), 3), np.float32)
    for k in range(len(rpk)):
        w = pre[lo[k]:hi[k]]
        out[k] = (np.log(w.std() / w.mean() + 1e-3), ch[lo[k] + 1:hi[k]].mean() if hi[k] - lo[k] > 1 else 0.0,
                  np.log(pre[k] / np.median(w)))
    return out


class Split:
    def __init__(self, db: str, recs: list[str], stats=None, ctx_s: float = 5.0, robust: bool = False,
                 win_s: tuple | None = None):
        wpre, wpost = (BEAT_PRE, BEAT_POST) if win_s is None else (int(round(win_s[0] * FS)), int(round(win_s[1] * FS)))
        half = int(round(ctx_s * FS))                          # context half-width (samples)
        maxb = int(round(6.4 * ctx_s))                          # token cap: 32 at +-5 s (192 bpm)
        self.ctx_s, self.max_beats = ctx_s, maxb
        P = {k: [] for k in ("sig", "beats", "tdiff", "timing", "timing_r", "seq_idx", "seq_off", "seq_len", "seq_ctr",
                             "tgt_pos", "tgt_beat", "rr3", "rr4", "timing_i", "irr", "y", "rec")}
        off_sig, off_beat = 0, 0
        ar = np.arange(maxb)
        for rec in recs:                                         # "rec" or "<db>/<rec>"
            dbr, rr_ = rec.split("/") if "/" in rec else (db, rec)
            d = np.load(CACHE / dbr / f"{rr_}.npz")
            x, rpk, y = d["x"], d["rpk"], d["y"]
            if len(rpk) < 3:
                continue
            pad = CTX_HALF + max(wpre, wpost)
            xp = np.pad(x, (pad, pad))
            n = len(rpk)
            rr = np.diff(rpk) / FS
            pre, post = np.r_[rr[0], rr], np.r_[rr, rr[-1]]
            med = np.median(rr)
            local = _window_mean(rpk, pre, LOCAL_HALF)
            irr = irregularity(rpk, pre)
            W = _z(xp[(rpk + pad)[:, None] + np.arange(-wpre, wpost)[None]])
            P["beats"].append(W); P["sig"].append(xp)
            P["tdiff"].append(W - template(W, robust))
            prev10 = np.array([pre[max(1, i - 10):i].mean() if i > 1 else pre[1] for i in range(n)])
            if "elig" in d.files:                                # explicit eligibility (paired automatic-peak runs)
                ok = d["elig"].astype(bool) & (y >= 0)
            else:
                ok = (y >= 0) & (rpk - CTX_HALF >= 0) & (rpk + CTX_HALF <= len(x))   # same beats for every ctx_s
                ok[[0, -1]] = False
            i = np.where(ok)[0]
            lo = np.searchsorted(rpk, rpk[i] - half, "left")
            hi = np.searchsorted(rpk, rpk[i] + half, "right")
            J = lo[:, None] + ar[None]
            valid = J < hi[:, None]
            for r in np.where(hi - lo > maxb)[0]:              # too many beats: keep the maxb nearest
                j = np.arange(lo[r], hi[r])
                J[r] = np.sort(j[np.argsort(np.abs(rpk[j] - rpk[i[r]]))[:maxb]])
            Jc = np.minimum(J, n - 1)
            ln = valid.sum(1)
            ctr = np.argmax(J == i[:, None], 1)
            offs = (rpk[Jc] - rpk[i][:, None]) / half
            lg = np.log
            tm = np.stack([pre[Jc], post[Jc], pre[Jc] / med, offs], -1).astype(np.float32)
            tr_ = np.stack([lg(pre[Jc] / med), lg(post[Jc] / med), lg(pre[Jc] / local[Jc]), offs], -1).astype(np.float32)
            tm[~valid] = 0; tr_[~valid] = 0
            ti = irr[Jc].astype(np.float32); ti[~valid] = 0
            P["timing"].append(tm); P["timing_r"].append(tr_); P["timing_i"].append(ti)
            P["seq_idx"].append(np.where(valid, off_beat + J, -1)); P["seq_len"].append(ln); P["seq_ctr"].append(ctr)
            P["seq_off"].append(np.where(valid, rpk[Jc] - rpk[i][:, None], 0))
            P["tgt_pos"].append(off_sig + rpk[i] + pad); P["tgt_beat"].append(off_beat + i)
            P["rr3"].append(np.stack([pre[i], post[i], prev10[i]], 1))
            P["rr4"].append(np.stack([lg(pre[i] / med), lg(post[i] / med), lg(pre[i] / post[i]), lg(pre[i] / local[i])], 1))
            P["irr"].append(irr[i]); P["y"].append(y[i]); P["rec"].append(np.full(len(i), rec))
            off_sig += len(xp); off_beat += n
        cat = {k: np.concatenate(v) for k, v in P.items()}
        mask = cat["seq_len"][:, None] > ar[None]
        if stats is None:
            ms = lambda a: (a.mean(0), a.std(0) + 1e-6)     # noqa: E731
            stats = {"timing": ms(cat["timing"][mask]), "timing_r": ms(cat["timing_r"][mask]),
                     "rr3": ms(cat["rr3"]), "rr4": ms(cat["rr4"]), "irr": ms(cat["irr"])}
        stats = {**stats, "irr": stats.get("irr", (np.zeros(3, np.float32), np.ones(3, np.float32)))}   # old checkpoints
        self.stats = stats
        cat["timing_i"] = (cat["timing_i"] - stats["irr"][0]) / stats["irr"][1]
        cat["timing_i"][~mask] = 0
        cat["irr"] = (cat["irr"] - stats["irr"][0]) / stats["irr"][1]
        for k in ("timing", "timing_r"):
            cat[k] = (cat[k] - stats[k][0]) / stats[k][1]
            cat[k][~mask] = 0
        for k in ("rr3", "rr4"):
            cat[k] = (cat[k] - stats[k][0]) / stats[k][1]
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=DEV)   # noqa: E731
        lng = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long, device=DEV)   # noqa: E731
        self.sig, self.beats, self.tdiff = f(cat["sig"]), f(cat["beats"]), f(cat["tdiff"])
        self.timing, self.timing_r, self.rr3, self.rr4 = f(cat["timing"]), f(cat["timing_r"]), f(cat["rr3"]), f(cat["rr4"])
        self.timing_i, self.irr = f(cat["timing_i"]), f(cat["irr"])
        for k in ("tgt_pos", "tgt_beat", "seq_idx", "seq_off", "seq_len", "seq_ctr", "y"):
            setattr(self, k, lng(cat[k]))
        self.rec = cat["rec"]
        self.win = torch.arange(-wpre, wpost, device=DEV)
        self.ctx_off = torch.arange(-CTX_HALF, CTX_HALF, device=DEV)

    def __len__(self):
        return len(self.y)

    def batch(self, idx: torch.Tensor, need_raw: bool, need_seq: bool, aug=None):
        out = {"beat": self.beats[self.tgt_beat[idx]], "tdiff": self.tdiff[self.tgt_beat[idx]],
               "rr3": self.rr3[idx], "rr4": self.rr4[idx], "irr": self.irr[idx], "y": self.y[idx]}
        if need_raw:
            c = self.sig[self.tgt_pos[idx][:, None] + self.ctx_off]
            out["raw"] = (c - c.mean(1, keepdim=True)) / (c.std(1, keepdim=True) + 1e-6)
        if need_seq:
            si = self.seq_idx[idx]
            out["seq"] = self.beats[si.clamp(min=0)] * (si >= 0)[..., None]
            out["seq_tdiff"] = self.tdiff[si.clamp(min=0)] * (si >= 0)[..., None]
            out["timing"], out["timing_r"], out["timing_i"] = self.timing[idx], self.timing_r[idx], self.timing_i[idx]
            out["len"], out["ctr"] = self.seq_len[idx], self.seq_ctr[idx]
        if aug is not None:
            aug(self, idx, out)
        return out


assert CTX_LEN == 2 * CTX_HALF and BEAT_LEN == BEAT_PRE + BEAT_POST
