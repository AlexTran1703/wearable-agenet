"""AGENet: central-beat morphology (residual CNN + CBAM-1D) + 10-s beat-sequence rhythm context (shared beat encoder
+ GRU), gated fusion -> N / SVEB / VEB.

Beat branch (central beat, 1 x 96 = [-0.35, +0.45] s at 120 Hz):
  Conv1D 1->16 k7, BN, ReLU | ResBlock 16->24 k7 | CBAM-1D(24) | MaxPool 2 | ResBlock 24->32 k5 | CBAM-1D(32) |
  global average pool -> m (32)
Context branch (beats with R within +-5 s of the central R, in time order, up to 32):
  shared token encoder per beat: Conv1D 1->8 k7 s2, BN, ReLU | Conv1D 8->16 k5 s2, BN, ReLU | GAP (16)
    ++ 4 timing values [pre-RR, post-RR, pre-RR / record median RR, offset from the central R] -> Linear 20->24, ReLU
    -> e_j (24)
  GRU 24->32 (last hidden state)                              -> h (32)
  BiGRU 24->32 per direction, output at the central beat (64) -> Linear 64->32 -> h (32)
Gated fusion: g = sigmoid(W_g [m; h] + b_g), z = g * m + (1 - g) * h  (32)
Head: Linear 32->16, ReLU, dropout 0.2, Linear 16->3.

Compared models
  A0 Beat-CNN        beat branch without CBAM, no context
  A1 Beat-CBAM       beat branch with CBAM, no context
  A2 AGENet-GRU      A1 + GRU context, gated fusion          (main architecture)
  A3 AGENet-BiGRU    A1 + BiGRU context, gated fusion
  A4 AGENet-GRU+RR   A2 + explicit [RR_pre, RR_post, RR_local] concatenated to z before the head
Enhancements of A3 (literature mechanisms for inter-patient SVEB, selected on validation only)
  rel   patient-relative token timing [log pre/med, log post/med, log pre/local, offset] instead of seconds
        (RR relative to the patient's own rate: Zhang 2021, Wang 2020, Zahid 2022, Farag 2023)
  inj   feature injection: central-beat [log pre/med, log post/med, log pre/post, log pre/local] -> head (Zahid 2022)
  tmpl  second input channel = beat minus the record's median beat (patient-adaptive morphology), beat + tokens
  aug   NSTDB noise + R jitter on training beats (train/augment.py; Zahid 2022)
B4 options on top of B3 (opts, selected greedily by patient-wise 5-fold CV inside DS1; notes/04_agenet_b4_plan.md)
  ch3    3 input channels [b, T, b - T] (beat, record template, difference) in both branches
  catt   central-query attention over the BiGRU sequence: q = W_q h_c, a = softmax(K q / sqrt d),
         h = Linear([h_c ; sum_j a_j W_v h_j])
  irr    U2: rhythm-irregularity features per token (+3 timing values) and for the central beat (+3 at the head)
  rrmlp  RR ratios transformed phi(r) = ReLU(Linear(4, 8)); head Linear(40, 24), ReLU, dropout 0.25, Linear(24, 3)
  (robust template and subject-balanced sampling are data / training options: data/context_windows.py, train/train.py)
"""
import torch
import torch.nn as nn

from .cbam1d import CBAM1d

VARIANTS = {
    "A0": dict(cbam=False, ctx=None, rr=False),
    "A1": dict(cbam=True, ctx=None, rr=False),
    "A2": dict(cbam=True, ctx="gru", rr=False),
    "A3": dict(cbam=True, ctx="bigru", rr=False),
    "A4": dict(cbam=True, ctx="gru", rr=True),
    "B1": dict(cbam=True, ctx="bigru", rr=False, rel=True),
    "B2": dict(cbam=True, ctx="bigru", rr=False, rel=True, inj=True),
    "B3": dict(cbam=True, ctx="bigru", rr=False, rel=True, inj=True, tmpl=True),
    "B4": dict(cbam=True, ctx="bigru", rr=False, rel=False, inj=True),
}
NAMES = {"A0": "Beat-CNN", "A1": "Beat-CBAM", "A2": "AGENet-GRU", "A3": "AGENet-BiGRU", "A4": "AGENet-GRU+RR",
         "B1": "A3 + relative RR tokens", "B2": "A3 + relative RR tokens + RR injection",
         "B3": "A3 + relative RR tokens + RR injection + template channel", "B4": "A3 + RR injection"}


def cbr(ci, co, k, s=1):
    return nn.Sequential(nn.Conv1d(ci, co, k, s, k // 2, bias=False), nn.BatchNorm1d(co), nn.ReLU())


class ResBlock(nn.Module):
    def __init__(self, ci, co, k):
        super().__init__()
        self.body = nn.Sequential(cbr(ci, co, k), nn.Conv1d(co, co, k, 1, k // 2, bias=False), nn.BatchNorm1d(co))
        self.skip = nn.Sequential(nn.Conv1d(ci, co, 1, bias=False), nn.BatchNorm1d(co)) if ci != co else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class BeatBranch(nn.Module):
    def __init__(self, cbam: bool, in_ch: int = 1):
        super().__init__()
        self.stem = cbr(in_ch, 16, 7)
        self.res1, self.res2 = ResBlock(16, 24, 7), ResBlock(24, 32, 5)
        self.cbam1 = CBAM1d(24) if cbam else nn.Identity()
        self.cbam2 = CBAM1d(32) if cbam else nn.Identity()
        self.pool = nn.MaxPool1d(2)

    def forward(self, beat):                                   # (B, C, 96)
        x = self.cbam1(self.res1(self.stem(beat)))            # (B, 24, 96)
        x = self.cbam2(self.res2(self.pool(x)))               # (B, 32, 48)
        return x.mean(-1)                                      # (B, 32)


class TokenEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, n_time: int = 4):
        super().__init__()
        self.cnn = nn.Sequential(cbr(in_ch, 8, 7, 2), cbr(8, 16, 5, 2))
        self.proj = nn.Sequential(nn.Linear(16 + n_time, 24), nn.ReLU())

    def forward(self, seq, timing):                            # (B, L, C, 96), (B, L, 4)
        B, L, C, T = seq.shape
        e = self.cnn(seq.reshape(B * L, C, T)).mean(-1).reshape(B, L, 16)
        return self.proj(torch.cat([e, timing], -1))          # (B, L, 24)


class ContextBranch(nn.Module):
    """Masked fixed-length recurrence (no packing, no GPU->CPU sync). Sequences are left-aligned with zero padding:
    the forward GRU state at step len-1 is unaffected by the padding that follows; the backward direction runs a second
    GRU over each sequence reversed within its own length, so it also never reads padding. Mathematically identical to
    a packed (Bi)GRU."""

    def __init__(self, bidirectional: bool, in_ch: int = 1, catt: bool = False, n_time: int = 4):
        super().__init__()
        self.bi, self.catt = bidirectional, catt
        self.tokens = TokenEncoder(in_ch, n_time)
        self.gru = nn.GRU(24, 32, batch_first=True)
        if bidirectional:
            self.gru_bwd = nn.GRU(24, 32, batch_first=True)
            self.proj = nn.Linear(64, 32)
        if catt:
            assert bidirectional
            self.q, self.k, self.v = nn.Linear(64, 32), nn.Linear(64, 32), nn.Linear(64, 32)
            self.proj = nn.Linear(64 + 32, 32)
            self.last_attn = None

    def forward(self, seq, timing, length, center):
        e = self.tokens(seq, timing)                           # (B, L, 24)
        B, L, _ = e.shape
        ar = torch.arange(B, device=e.device)
        fwd, _ = self.gru(e)
        if not self.bi:
            return fwd[ar, length - 1]                         # last valid step (B, 32)
        t = torch.arange(L, device=e.device)[None]
        rev = (length[:, None] - 1 - t).clamp(min=0)           # per-sample reversal of the valid part
        bwd, _ = self.gru_bwd(e[ar[:, None], rev])
        if not self.catt:
            return self.proj(torch.cat([fwd[ar, center], bwd[ar, length - 1 - center]], -1))
        H = torch.cat([fwd, bwd[ar[:, None], rev]], -1)        # (B, L, 64), backward re-aligned to time order
        hc = H[ar, center]                                      # central beat state (same as the non-attention path)
        sc = (self.k(H) @ self.q(hc)[..., None])[..., 0] / 32 ** 0.5
        sc = sc.masked_fill(t >= length[:, None], float("-inf"))
        a = torch.softmax(sc, 1)
        self.last_attn = a.detach()
        return self.proj(torch.cat([hc, (a[..., None] * self.v(H)).sum(1)], -1))


class GatedFusion(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.gate = nn.Linear(2 * dim, dim)
        self.last_gate = None

    def forward(self, m, h):
        g = torch.sigmoid(self.gate(torch.cat([m, h], -1)))
        self.last_gate = g.detach()
        return g * m + (1 - g) * h


class AGENet(nn.Module):
    def __init__(self, variant: str = "A2", opts=()):
        super().__init__()
        base = {"rel": False, "inj": False, "tmpl": False, "ch3": False, "catt": False, "rrmlp": False, "irr": False}
        self.variant, self.opts = variant, tuple(opts)
        cfg = {**base, **VARIANTS[variant], **{o: True for o in opts if o in base}}
        self.cfg = cfg
        ch = 3 if cfg["ch3"] else 2 if cfg["tmpl"] else 1
        self.beat = BeatBranch(cfg["cbam"], ch)
        if cfg["ctx"]:
            self.ctx = ContextBranch(cfg["ctx"] == "bigru", ch, cfg["catt"], 7 if cfg["irr"] else 4)
            self.fuse = GatedFusion(32)
        if cfg["rrmlp"]:
            assert cfg["inj"]
            self.rrphi = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(32 + 8 + (3 if cfg["rr"] else 0), 24), nn.ReLU(), nn.Dropout(0.25),
                                      nn.Linear(24, 3))
        else:
            extra = (3 if cfg["rr"] else 0) + (4 if cfg["inj"] else 0) + (3 if cfg["irr"] else 0)
            self.head = nn.Sequential(nn.Linear(32 + extra, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 3))

    @property
    def needs(self):
        return False, self.cfg["ctx"] is not None             # (raw context, beat sequence)

    def _chans(self, x, d, dim):
        if self.cfg["ch3"]:
            return torch.stack([x, x - d, d], dim)              # beat, template (= beat - diff), difference
        return torch.stack([x, d], dim) if self.cfg["tmpl"] else x.unsqueeze(dim)

    def forward(self, b):
        z = self.beat(self._chans(b["beat"], b["tdiff"], 1))
        if self.cfg["ctx"]:
            seq = self._chans(b["seq"], b["seq_tdiff"], 2)
            tim = b["timing_r" if self.cfg["rel"] else "timing"]
            if self.cfg["irr"]:
                tim = torch.cat([tim, b["timing_i"]], -1)
            z = self.fuse(z, self.ctx(seq, tim, b["len"], b["ctr"]))
        if self.cfg["rr"]:
            z = torch.cat([z, b["rr3"]], -1)
        if self.cfg["inj"]:
            z = torch.cat([z, self.rrphi(b["rr4"]) if self.cfg["rrmlp"] else b["rr4"]], -1)
        if self.cfg["irr"]:
            z = torch.cat([z, b["irr"]], -1)
        return self.head(z)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
