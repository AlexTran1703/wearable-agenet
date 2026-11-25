"""Strict patient-independent split (fixed before any model was trained).

de Chazal DS1 / DS2 with the 201 / 202 overlap removed: records 201 and 202 come from the same patient and sit on
opposite sides of the classic split, so 201 is excluded from training and 202 stays in the DS2 test set.
Validation = a patient-disjoint DS1 subset chosen by a fixed rule: hold roughly 20 % of the remaining DS1 SVEB and
30 % of the DS1 VEB beats while keeping the largest SVEB source (record 209, 383 beats) in training, because SVEB
examples are the scarcest training resource. Resulting validation: 220, 223, 119, 215, 122 (170 S, 1,081 V).
Training: the other 16 DS1 records (645 S, 2,509 V). DS2 is used for nothing but the final evaluation.
"""
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
       "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
       "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]
EXCLUDED = ["201"]                                  # same patient as 202 (DS2)
VAL = ["220", "223", "119", "215", "122"]
TRAIN = [r for r in DS1 if r not in VAL + EXCLUDED]
TEST = DS2
INCART = [f"I{i:02d}" for i in range(1, 76)]

assert not set(TRAIN) & set(VAL) and not set(TRAIN) & set(TEST) and not set(VAL) & set(TEST)


def svdb_records():
    """The 78 SVDB records listed in its RECORDS file (zero-shot external test, like INCART)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import DB_ROOT
    return (DB_ROOT["svdb"] / "RECORDS").read_text().split()
