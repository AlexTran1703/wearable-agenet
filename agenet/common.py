"""Paths and constants for AGENet (MDPI BioMedInformatics)."""
from __future__ import annotations

import os
from pathlib import Path

AGENET = Path(__file__).resolve().parent
MDPI = AGENET.parent
CACHE = AGENET / "cache"
RUNS = AGENET / "runs"
RESULTS = AGENET / "results"              # component results; paper copies go to mdpi/results
PAPER_RESULTS = MDPI / "results"

DATASET = Path(os.environ.get("AGENET_DATASET", AGENET.parent / "physionet"))   # PhysioNet databases
DB_ROOT = {"mitdb": DATASET / "mit-bih-arrhythmia", "incart": DATASET / "st-petersburg-incart",
           "svdb": DATASET / "mit-bih-supraventricular-arrhythmia"}

FS = 120                                  # processing rate
BEAT_PRE, BEAT_POST = 42, 54              # central beat [-0.35, +0.45] s -> 96 samples
BEAT_LEN = BEAT_PRE + BEAT_POST
CTX_HALF = 5 * FS                         # 10-s context [-5, +5] s -> 1,200 samples
CTX_LEN = 2 * CTX_HALF
MAX_CTX_BEATS = 32                        # beats inside the 10-s window (covers >= 190 bpm)
