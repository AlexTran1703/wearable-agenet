"""Training-time augmentation on the GPU (training split only; validation and tests are never augmented).

Following Zahid et al. 2022 (NSTDB noise added to training beats), each training sample is augmented with
probability p:
  noise   a contiguous NSTDB segment (bw / em / ma, either channel; 0.5-40 Hz, 120 Hz, unit std) added at a random
          SNR in [snr_lo, snr_hi] dB to the central beat and to every context token at its true time offset, so the
          whole 10-s context carries the same noise realisation; windows are re-z-scored afterwards
  jitter  the central-beat window is re-cut from the signal around R_i + U{-j..j} samples (automatic-detector R error)
"""
from __future__ import annotations

import sys
from math import gcd
from pathlib import Path

import numpy as np
import torch
import wfdb
from scipy.signal import butter, resample_poly, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BEAT_POST, BEAT_PRE, FS  # noqa: E402

NSTDB = Path(r"E:\ecg_quality\mit-bih-noise-stress-test-database-1.0.0")


def noise_bank(device):
    sos = butter(2, [0.5, 40.0], btype="bandpass", fs=360, output="sos")
    g = gcd(FS, 360)
    out = []
    for nm in ("bw", "em", "ma"):
        s = wfdb.rdrecord(str(NSTDB / nm)).p_signal
        for c in range(s.shape[1]):
            v = resample_poly(sosfiltfilt(sos, s[:, c]), FS // g, 360 // g)
            out.append((v - v.mean()) / (v.std() + 1e-9))
    L = min(map(len, out))
    return torch.as_tensor(np.stack([o[:L] for o in out]), dtype=torch.float32, device=device)


class Augment:
    def __init__(self, device, p=0.5, snr=(6.0, 24.0), jitter=2):
        self.bank, self.p, self.snr, self.jit = noise_bank(device), p, snr, jitter
        self.win = torch.arange(-BEAT_PRE, BEAT_POST, device=device)
        self.margin = 6 * FS + BEAT_POST                                   # covers +-5 s of tokens + windows

    @staticmethod
    def _z(a):
        return (a - a.mean(-1, keepdim=True)) / (a.std(-1, keepdim=True) + 1e-6)

    def __call__(self, S, idx, out):
        B, dev = len(idx), idx.device
        on = torch.rand(B, device=dev) < self.p
        if not on.any():
            return
        beat0 = out["beat"]
        if self.jit:
            sh = torch.randint(-self.jit, self.jit + 1, (B,), device=dev) * on
            beat = self._z(S.sig[(S.tgt_pos[idx] + sh)[:, None] + self.win])
        else:
            beat = beat0
        sigma = 10 ** (-(torch.rand(B, device=dev) * (self.snr[1] - self.snr[0]) + self.snr[0]) / 20) * on
        src = torch.randint(0, self.bank.shape[0], (B,), device=dev)
        start = torch.randint(self.margin, self.bank.shape[1] - self.margin, (B,), device=dev)
        n = self.bank[src[:, None], start[:, None] + self.win]
        out["beat"] = torch.where(on[:, None], self._z(beat + sigma[:, None] * n), beat0)
        out["tdiff"] = out["tdiff"] + (out["beat"] - beat0)
        if "seq" in out:
            pos = start[:, None, None] + S.seq_off[idx][..., None] + self.win
            ns = self.bank[src[:, None, None], pos]
            valid = (S.seq_idx[idx] >= 0)[..., None]
            s0 = out["seq"]
            s1 = torch.where(on[:, None, None] & valid, self._z(s0 + sigma[:, None, None] * ns), s0)
            out["seq"] = s1
            out["seq_tdiff"] = out["seq_tdiff"] + (s1 - s0)


class AcqAugment:
    """U4: acquisition-robustness augmentation (training only), each part applied per sample with probability p:
      low-pass    random cut-off 20-40 Hz (smooth spectral mask, rFFT) on the central beat and all tokens
                  (other recorders / SVDB 128 Hz bandwidth)
      wander      random linear trend across each window (residual baseline), slope up to +-0.3 std per window
      jitter      central beat re-cut around R_i + U{-2..2} samples (automatic-detector R error)
      template    record template perturbed: T' = a T + eps, a ~ U(0.9, 1.1), eps ~ N(0, 0.05^2) (imperfect template)
    The difference channel is recomputed as beat - T' so it stays consistent with the perturbed inputs."""

    def __init__(self, device, p=0.5, jitter=2, fs=120):
        self.p, self.jit, self.fs, self.dev = p, jitter, fs, device

    def _lowpass(self, x, fc):                        # x (..., T); fc broadcastable (...,)
        T = x.shape[-1]
        f = torch.fft.rfftfreq(T, 1 / self.fs, device=x.device)
        m = torch.sigmoid((fc[..., None] - f) / 2.0)
        return torch.fft.irfft(torch.fft.rfft(x, dim=-1) * m, n=T, dim=-1)

    @staticmethod
    def _z(a):
        return (a - a.mean(-1, keepdim=True)) / (a.std(-1, keepdim=True) + 1e-6)

    def __call__(self, S, idx, out):
        B, dev = len(idx), idx.device
        on = lambda: torch.rand(B, device=dev) < self.p    # noqa: E731
        T_c = out["beat"] - out["tdiff"]                   # record template seen by the central beat
        beat = out["beat"]
        j = on()
        if self.jit and j.any():
            sh = torch.randint(-self.jit, self.jit + 1, (B,), device=dev) * j
            beat = torch.where(j[:, None], self._z(S.sig[(S.tgt_pos[idx] + sh)[:, None] + S.win]), beat)
        lp, fc = on(), 20 + 20 * torch.rand(B, device=dev)
        wd, slope = on(), (torch.rand(B, device=dev) * 2 - 1) * 0.3
        ramp = torch.linspace(-0.5, 0.5, beat.shape[-1], device=dev)
        def acq(x, extra_dims):                          # noqa: E306
            sel = lambda v: v.view(B, *([1] * extra_dims))   # noqa: E731
            y = torch.where(sel(lp), self._lowpass(x, fc.view(B, *([1] * (extra_dims - 1)))), x)
            y = y + sel(wd * slope) * ramp
            return torch.where(sel(lp | wd), self._z(y), x)
        beat = acq(beat, 1)
        tp = on()
        a = torch.where(tp, 0.9 + 0.2 * torch.rand(B, device=dev), torch.ones(B, device=dev))
        eps = torch.randn_like(T_c) * 0.05 * tp[:, None]
        T_new = a[:, None] * T_c + eps
        out["beat"], out["tdiff"] = beat, beat - T_new
        if "seq" in out:
            valid = (S.seq_idx[idx] >= 0)[..., None]
            T_s = out["seq"] - out["seq_tdiff"]
            s1 = torch.where(valid, acq(out["seq"], 2), out["seq"])
            T_s2 = a[:, None, None] * T_s + torch.randn_like(T_s) * 0.05 * tp[:, None, None]
            out["seq"], out["seq_tdiff"] = s1, torch.where(valid, s1 - T_s2, out["seq_tdiff"])
