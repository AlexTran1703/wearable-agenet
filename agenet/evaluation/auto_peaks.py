"""Paired reference-vs-automatic R-peak evaluation of the locked AGENet on DS2, INCART and SVDB (Section 3.5.1).

  python evaluation/auto_peaks.py nv_B3_wce_s1337

Both arms use the same DS1-trained weights, the same preprocessed channel per record and the same training statistics.
  channel      MIT-BIH MLII, INCART II, SVDB ECG1 for every record (prespecified; the Section 3.2 lead-II-like rule uses
               reference beat times and is not used here). The SVDB reference arm is therefore re-scored on ECG1.
  automatic    Hamilton (deployed), XQRS or Pan-Tompkins on the preprocessed 120-Hz channel. Detections are
               matched one-to-one within +-100 ms to ALL reference beat annotations (N/S/V/F/Q). A detection matched to
               an N/S/V beat inherits its class; matched to F/Q it is excluded; unmatched it is "spurious" (label 3).
  eligibility  a reference N/S/V beat is eligible when its 10-s context lies inside the record (>= 5 s from both ends;
               the first and last annotated beats excluded), as in Section 3.2. In the automatic arm a matched detection
               takes the eligibility of its reference beat; an unmatched detection is eligible when it lies >= 5 s from
               both ends. Eligible reference beats without a detection are FN of their class; eligible spurious
               detections are FP of the class predicted for them (N included).
  intervals    2,000 paired bootstrap draws: the same resampled units score both arms. Units: MIT-BIH DS2 record,
               INCART patient (from the '# patient k' header field; 32 patients), SVDB record.
  detection    R-peak Se / PPV / F1 over all reference beat annotations of the whole record (all beat types).
Output: results/auto_peaks.json for Hamilton (+ per-beat arrays in results/auto_peaks_beats.npz);
  python evaluation/auto_peaks.py nv_B3_wce_s1337 pantompkins   -> results/auto_peaks_pantompkins.json (comparison only).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data.context_windows as cw  # noqa: E402
from common import CACHE, CTX_HALF, DB_ROOT, RESULTS, RUNS  # noqa: E402
from data.patient_split import INCART, TEST, svdb_records  # noqa: E402
from data.preprocess import load_record  # noqa: E402
from evaluation.rpeak import detect, match  # noqa: E402

WORK = CACHE.parent / "cache_paired"   # per-detector subfolder set in main()
NB, SEED = 2000, 0
MISSED = 4                                   # prediction token for an eligible reference beat without a detection


def svdb_ecg1(rec):
    """load_record for SVDB with the channel fixed to ECG1 (index 0) instead of the lead-II-like rule."""
    import data.preprocess as pp
    orig = pp.lead_ii_like
    pp.lead_ii_like = lambda sig, sos, samples: 0
    try:
        return load_record("svdb", rec)
    finally:
        pp.lead_ii_like = orig


def ref_eligible(rpk, y, n):
    ok = (y >= 0) & (rpk - CTX_HALF >= 0) & (rpk + CTX_HALF <= n)
    ok[[0, -1]] = False
    return ok


def incart_patient(rec):
    txt = (DB_ROOT["incart"] / f"{rec}.hea").read_text()
    k = int(re.search(r"#\s*patient\s+(\d+)", txt).group(1))
    return f"P{k:02d}"


def build(db, recs, detector):
    """Reference and automatic caches for one database; returns per-record detection and missed-beat counts."""
    ref_dir, auto_dir = WORK / f"{db}_ref", WORK / f"{db}_auto"
    ref_dir.mkdir(parents=True, exist_ok=True); auto_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for rec in recs:
        if db == "svdb":
            x, rpk, y = svdb_ecg1(rec)
        else:
            d = np.load(CACHE / db / f"{rec}.npz"); x, rpk, y = d["x"], d["rpk"], d["y"]
        np.savez_compressed(ref_dir / f"{rec}.npz", x=x, rpk=rpk, y=y)
        det = detect(x.astype(np.float64), detector)
        m = match(det, rpk)                                   # det index per reference beat, -1 = missed
        elig_ref = ref_eligible(rpk, y, len(x))
        yd = np.full(len(det), 3, np.int64)
        ed = (det - CTX_HALF >= 0) & (det + CTX_HALF <= len(x))
        hit = m >= 0
        yd[m[hit]] = y[hit]                                   # F / Q stay -1 (excluded)
        ed[m[hit]] = elig_ref[hit]
        np.savez_compressed(auto_dir / f"{rec}.npz", x=x, rpk=det, y=yd, elig=ed)
        miss = elig_ref & ~hit
        info[rec] = {"missed": np.bincount(y[miss], minlength=3).tolist(),
                     "spurious_eligible": int(((yd == 3) & ed).sum()),
                     "det_tp": int(hit.sum()), "det_fn": int((~hit).sum()), "det_fp": int(len(det) - hit.sum())}
    return ref_dir, auto_dir, info


def predict_split(model, stats, cache_dir, recs):
    from train.train import predict
    saved = cw.CACHE
    cw.CACHE = cache_dir.parent
    try:
        S = cw.Split(cache_dir.name, recs, stats)
    finally:
        cw.CACHE = saved
    return S.y.cpu().numpy(), predict(model, S).argmax(1), np.asarray(S.rec)


def scores(y, p):
    out = {}
    for k, c in enumerate("NSV"):
        tp = int(((y == k) & (p == k)).sum()); fp = int(((y != k) & (p == k)).sum()); fn = int(((y == k) & (p != k)).sum())
        se = tp / (tp + fn) if tp + fn else np.nan
        pp = tp / (tp + fp) if tp + fp else np.nan
        out[c] = {"Se": se, "P": pp, "F1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan,
                  "TP": tp, "FP": fp, "FN": fn}
    out["macro_F1"] = float(np.nanmean([out[c]["F1"] for c in "NSV"]))
    return out


KEYS = ["macro_F1"] + [f"{c}_{m}" for c in "NSV" for m in ("Se", "P", "F1")]


def flat(s):
    return {"macro_F1": s["macro_F1"], **{f"{c}_{m}": s[c][m] for c in "NSV" for m in ("Se", "P", "F1")}}


def paired_bootstrap(arms, units):
    """arms: {name: (y, p, unit)}; the same unit draw scores every arm."""
    uniq = np.unique(np.concatenate([u for _, _, u in arms.values()]))
    idx = {a: {u: np.where(uu == u)[0] for u in uniq} for a, (_, _, uu) in arms.items()}
    rng = np.random.default_rng(SEED)
    boots = {a: {k: [] for k in KEYS} for a in arms}
    for _ in range(NB):
        draw = rng.choice(uniq, len(uniq), replace=True)
        for a, (y, p, _) in arms.items():
            take = np.concatenate([idx[a][u] for u in draw])
            f = flat(scores(y[take], p[take]))
            for k in KEYS:
                boots[a][k].append(f[k])
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]   # noqa: E731
    out = {a: {k: ci(boots[a][k]) for k in KEYS} for a in arms}
    out["delta"] = {k: ci(np.array(boots["automatic"][k]) - np.array(boots["reference"][k])) for k in KEYS}
    return out


def main(run, detector=None):
    global WORK
    locked = "hamilton"                         # detector deployed on EC2
    detector = detector or locked
    WORK = CACHE.parent / f"cache_paired_{detector}"
    tag = "" if detector == locked else f"_{detector}"
    ck = torch.load(RUNS / run / "model.pt", weights_only=False)
    from models.agenet import AGENet
    model = AGENet(ck["variant"], ck.get("opts", ())).to(cw.DEV)
    model.load_state_dict(ck["state_dict"])
    stats = ck["stats"]
    res, beats = {"run": run, "detector": detector, "locked_detector": locked, "n_boot": NB, "seed": SEED}, {}
    for db, recs, unit_of, unit_name in (("mitdb", TEST, lambda r: r, "record"),
                                         ("incart", INCART, incart_patient, "patient"),
                                         ("svdb", svdb_records(), lambda r: r, "record")):
        ref_dir, auto_dir, info = build(db, recs, detector)
        yr, pr, rr = predict_split(model, stats, ref_dir, recs)
        ya, pa, ra = predict_split(model, stats, auto_dir, recs)
        # eligible missed reference beats enter the automatic arm as FN of their class
        my, mr = [], []
        for rec in recs:
            for k, n in enumerate(info[rec]["missed"]):
                my += [k] * n; mr += [rec] * n
        ya = np.r_[ya, np.array(my, np.int64)]; pa = np.r_[pa, np.full(len(my), MISSED)]; ra = np.r_[ra, np.array(mr, dtype=ra.dtype)]
        unit = np.vectorize(unit_of)
        arms = {"reference": (yr, pr, unit(rr)), "automatic": (ya, pa, unit(ra))}
        sr, sa = scores(yr, pr), scores(ya, pa)
        tp = sum(v["det_tp"] for v in info.values()); fn = sum(v["det_fn"] for v in info.values())
        fp = sum(v["det_fp"] for v in info.values())
        se, ppv = tp / (tp + fn), tp / (tp + fp)
        name = {"mitdb": "DS2", "incart": "INCART", "svdb": "SVDB"}[db]
        res[name] = {
            "channel": {"mitdb": "MLII", "incart": "II", "svdb": "ECG1 (all records)"}[db],
            "bootstrap_unit": unit_name, "n_units": int(len(np.unique(unit(rr)))),
            "reference": sr, "automatic": sa,
            "delta": {k: flat(sa)[k] - flat(sr)[k] for k in KEYS},
            "ci": paired_bootstrap(arms, unit_name),
            "eligible_reference_NSV": np.bincount(yr, minlength=3).tolist(),
            "missed_eligible_NSV": np.sum([v["missed"] for v in info.values()], 0).tolist(),
            "spurious_eligible": int(sum(v["spurious_eligible"] for v in info.values())),
            "spurious_eligible_predicted_NSV": np.bincount(pa[ya == 3], minlength=3).tolist(),
            "detection_all_beats": {"Se": se, "PPV": ppv, "F1": 2 * se * ppv / (se + ppv), "TP": tp, "FN": fn, "FP": fp},
        }
        beats[f"{name}_ref_y"], beats[f"{name}_ref_p"], beats[f"{name}_ref_rec"] = yr, pr, rr
        beats[f"{name}_auto_y"], beats[f"{name}_auto_p"], beats[f"{name}_auto_rec"] = ya, pa, ra
        d = res[name]
        print(f"{name:7s} ref macro {100 * sr['macro_F1']:.1f} S {100 * sr['S']['F1']:.1f} V {100 * sr['V']['F1']:.1f} | "
              f"auto macro {100 * sa['macro_F1']:.1f} S {100 * sa['S']['F1']:.1f} V {100 * sa['V']['F1']:.1f} | "
              f"det Se {100 * se:.2f} PPV {100 * ppv:.2f} | missed {d['missed_eligible_NSV']} spurious {d['spurious_eligible']}",
              flush=True)
    (RESULTS / f"auto_peaks{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(RESULTS / f"auto_peaks{tag}_beats.npz", **beats)
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nv_B3_wce_s1337", sys.argv[2] if len(sys.argv) > 2 else None)
