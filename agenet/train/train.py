"""Train one AGENet run and store its predictions.

  python train/train.py --variant A2 --loss wce --seeds 1337

Protocol: training split = DS1 minus validation minus 201; model selection on the patient-disjoint validation
records only (max validation macro-F1). AdamW lr 1e-3, weight decay 1e-4, batch 256, deterministic cuDNN, up to 100 epochs, early
stopping (patience 12), ReduceLROnPlateau on validation macro-F1 (factor 0.5, patience 4), gradient clipping 1.0.
After training, the best-validation checkpoint predicts the validation set and the zero-shot test sets DS2, INCART and
SVDB (MIT-BIH Supraventricular); none of them influences training or selection.
Output: runs/<variant>_<loss>_s<seed>/ {model.pt, history.json, preds_{val,ds2,incart,svdb}.npz, metrics.json}
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CACHE, RUNS  # noqa: E402
from data.context_windows import Split  # noqa: E402
from data.patient_split import DS1, EXCLUDED, INCART, TEST, TRAIN, VAL, svdb_records  # noqa: E402
from evaluation.metrics import full, macro_f1  # noqa: E402
from models.agenet import NAMES, AGENet, count_parameters  # noqa: E402
from train.augment import AcqAugment, Augment  # noqa: E402
from train.losses import class_weights, make_loss  # noqa: E402

MAX_EPOCHS, PATIENCE = 100, 12
FIXED_EPOCHS, STEP = 35, 10          # no-validation protocol (Zahid 2022 schedule shape): lr x0.1 every 10 epochs, last epoch kept


@torch.no_grad()
def predict(model, S, bs=1024):
    model.eval()
    raw, seq = model.needs
    out = []
    for s in range(0, len(S), bs):
        idx = torch.arange(s, min(len(S), s + bs), device=S.y.device)
        out.append(torch.softmax(model(S.batch(idx, raw, seq)), 1))
    return torch.cat(out).cpu().numpy()


def subject_sampler(tr, seed):
    """Patient -> class -> beat sampling (B4c): each draw picks a training record uniformly, then one of the classes
    present in it uniformly, then a beat of that class uniformly; len(tr) draws per epoch."""
    y = tr.y.cpu().numpy(); rec = tr.rec
    keys = sorted({(r, c) for r, c in zip(rec, y)})
    groups = [np.where((rec == r) & (y == c))[0] for r, c in keys]
    recs = sorted({r for r, _ in keys})
    g_of = [[k for k, (r2, _) in enumerate(keys) if r2 == r] for r in recs]
    g0, ng = np.array([g[0] for g in g_of]), np.array([len(g) for g in g_of])
    flat = np.concatenate(groups); start = np.r_[0, np.cumsum([len(g) for g in groups])[:-1]]
    glen = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    def draw(n):
        r = rng.integers(len(recs), size=n)
        g = g0[r] + (rng.random(n) * ng[r]).astype(int)
        return torch.as_tensor(flat[start[g] + (rng.random(n) * glen[g]).astype(int)], device=tr.y.device)
    return draw


def run(variant, loss_name, seed, splits=None, aug=False, batch=256, lr=1e-3, tag="", opts=()):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False   # one seed -> one result
    tr, va, tests = splits or load_splits()
    model = AGENet(variant, opts).to(tr.y.device)
    draw = subject_sampler(tr, seed) if "subj" in opts else None
    raw, seq = model.needs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    augf = AcqAugment(tr.y.device) if "acq" in opts else (Augment(tr.y.device) if aug else None)
    noval = va is None
    sched = (torch.optim.lr_scheduler.StepLR(opt, STEP, 0.1) if noval else
             torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4))
    lf = make_loss(loss_name, torch.ones(3, device=tr.y.device) if draw else class_weights(tr.y))   # sampling already balances
    best, best_state, best_ep, wait, hist = -1.0, None, 0, 0, []
    t0 = time.time()
    for ep in range(1, (FIXED_EPOCHS if noval else MAX_EPOCHS) + 1):
        model.train()
        perm = draw(len(tr)) if draw else torch.randperm(len(tr), device=tr.y.device)
        tot = torch.zeros((), device=tr.y.device)
        for s in range(0, len(tr), batch):
            b = tr.batch(perm[s:s + batch], raw, seq, augf)
            loss = lf(model(b), b["y"])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.detach()
        if noval:                                         # no model selection: fixed schedule, last epoch
            sched.step()
            hist.append({"epoch": ep, "loss": float(tot) / (len(tr) // batch + 1), "lr": opt.param_groups[0]["lr"]})
            best_state, best_ep = copy.deepcopy(model.state_dict()), ep
            continue
        f1 = macro_f1(va.y.cpu().numpy(), predict(model, va).argmax(1))
        sched.step(f1)
        hist.append({"epoch": ep, "loss": float(tot) / (len(tr) // batch + 1), "val_macro_F1": f1,
                     "lr": opt.param_groups[0]["lr"]})
        if f1 > best:
            best, best_state, best_ep, wait = f1, copy.deepcopy(model.state_dict()), ep, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    name = f"{'nv_' if noval else ''}{variant}_{loss_name}{'_aug' if aug else ''}{tag}_s{seed}"
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "variant": variant, "opts": list(opts), "stats": tr.stats}, d / "model.pt")
    metrics = {"variant": variant, "name": NAMES[variant], "loss": loss_name, "opts": list(opts), "aug": aug, "batch": batch, "lr": lr, "seed": seed, "protocol": "noval" if noval else "val", "train_records": len(set(tr.rec)), "params": count_parameters(model),
               "best_epoch": best_ep, "epochs_run": len(hist), "train_s": time.time() - t0}
    for tag, S in (*(() if noval else (("val", va),)), *tests.items()):
        prob = predict(model, S)
        np.savez_compressed(d / f"preds_{tag}.npz", prob=prob, y=S.y.cpu().numpy(), rec=S.rec)
        metrics[tag] = full(S.y.cpu().numpy(), prob)
    (d / "history.json").write_text(json.dumps(hist, indent=1))
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    pc = lambda m, c: m["per_class"][c]["F1"] * 100   # noqa: E731
    print(f"{name} ({NAMES[variant]}): {metrics['params']:,} params, best ep {best_ep}/{len(hist)} | "
          + " | ".join(f"{t} macro {metrics[t]['macro_F1'] * 100:.1f} S {pc(metrics[t], 'S'):.1f} V {pc(metrics[t], 'V'):.1f}"
                       for t in ("val", "intra_test", "ds2", "incart", "svdb") if t in metrics) + f" ({metrics['train_s']:.0f}s)", flush=True)
    return metrics


CV_FOLDS = 5


def cv_folds():
    """Patient-wise 5-fold split of the 21 DS1 records (201 excluded): records sorted by SVEB count and dealt
    round-robin, so every fold holds SVEB patients. Fixed before any CV run."""
    recs = [r for r in DS1 if r not in EXCLUDED]
    nS = {r: int((np.load(CACHE / "mitdb" / f"{r}.npz")["y"] == 1).sum()) for r in recs}
    order = sorted(recs, key=lambda r: (-nS[r], r))
    return [order[k::CV_FOLDS] for k in range(CV_FOLDS)]


PER_TARGET = ("timing", "timing_r", "rr3", "rr4", "timing_i", "irr", "tgt_pos", "tgt_beat", "seq_idx", "seq_off",
              "seq_len", "seq_ctr", "y")
INTRA_SEED = 2024                                          # fixed split seed, independent of the training seed


def subset(S, idx):
    """View of Split S restricted to target beats idx (signals, beat bank and templates shared)."""
    T = copy.copy(S)
    it = torch.as_tensor(idx, device=S.y.device)
    for k in PER_TARGET:
        setattr(T, k, getattr(S, k)[it])
    T.rec = S.rec[idx]
    return T


def intra_split(S):
    """Conventional intra-subject benchmark: random beat-level 64/16/20 split of all 44 MIT-BIH records, stratified
    by class (beats of one patient fall in train and test). Label-free normalisation statistics and record templates
    come from the whole record. Validation 16 % is reported only, never used for selection (fixed schedule)."""
    rng = np.random.default_rng(INTRA_SEED)
    y = S.y.cpu().numpy()
    tr, va, te = [], [], []
    for c in range(3):
        i = rng.permutation(np.where(y == c)[0])
        a, b = int(round(0.64 * len(i))), int(round(0.80 * len(i)))
        tr.append(i[:a]); va.append(i[a:b]); te.append(i[b:])
    return [np.sort(np.concatenate(z)) for z in (tr, va, te)]


def load_splits(protocol="val", ctx_s=5.0, fold=None, robust=False, win_s=None):
    """val: TRAIN (16 records) + 5 validation patients. noval: all DS1 except 201 (21 records), no validation.
    noval201: all 22 DS1 records including 201 (the literature setting; 201 and DS2 record 202 are the same patient)."""
    if protocol == "intra":
        S = Split("mitdb", DS1 + TEST, ctx_s=ctx_s, robust=robust, win_s=win_s)
        tr, va, te = intra_split(S)
        return subset(S, tr), None, {"intra_test": subset(S, te), "intra_val": subset(S, va)}
    if protocol == "cv":                                   # train on 4 folds, predict the held-out fold only
        held = cv_folds()[fold]
        tr = Split("mitdb", [r for r in DS1 if r not in EXCLUDED + held], ctx_s=ctx_s, robust=robust, win_s=win_s)
        return tr, None, {"fold": Split("mitdb", held, tr.stats, ctx_s, robust, win_s)}
    recs = {"val": TRAIN, "noval": [r for r in DS1 if r not in EXCLUDED], "noval201": DS1}[protocol]
    tr = Split("mitdb", recs, ctx_s=ctx_s, robust=robust, win_s=win_s)
    tests = {"ds2": Split("mitdb", TEST, tr.stats, ctx_s, robust, win_s), "incart": Split("incart", INCART, tr.stats, ctx_s, robust, win_s),
             "svdb": Split("svdb", svdb_records(), tr.stats, ctx_s, robust, win_s)}
    return tr, (Split("mitdb", VAL, tr.stats, ctx_s, robust, win_s) if protocol == "val" else None), tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--loss", default="wce", choices=["wce", "focal1", "focal15", "focal2"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1337])
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--protocol", default="val", choices=["val", "noval", "noval201", "cv", "intra"])
    ap.add_argument("--ctx", type=float, default=5.0)
    ap.add_argument("--win", default="", help="beat window pre,post in s, e.g. 0.5,0.5 (default 0.35,0.45)")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--opts", default="", help="comma list: ch3,robust,subj,catt,rrmlp,irr,acq")
    a = ap.parse_args()
    opts = tuple(o for o in a.opts.split(",") if o)
    win = tuple(float(v) for v in a.win.split(",")) if a.win else None
    splits = load_splits(a.protocol, a.ctx, a.fold, "robust" in opts, win)
    tag = a.tag + ("_201" if a.protocol == "noval201" else "") + (f"_ctx{a.ctx:g}" if a.ctx != 5.0 else "") + (f"_w{win[0]:g}-{win[1]:g}" if win else "")
    if opts:
        tag = "_o-" + "+".join(opts) + tag
    if a.protocol == "cv":
        tag = f"_cv{a.fold}" + tag
    if a.protocol == "intra":
        tag = "_intra" + tag
    for s in a.seeds:
        run(a.variant, a.loss, s, splits, a.aug, a.batch, a.lr, tag, opts)


if __name__ == "__main__":
    main()
