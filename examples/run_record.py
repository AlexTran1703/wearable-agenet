"""Analyse one stored five-minute record with the deployed pipeline (Hamilton R peaks + locked AGENet, ONNX).

  python examples/run_record.py data/simulator/N075_01.BIN
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server" / "worker"))
from lean_infer import LeanEngine  # noqa: E402

path = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "simulator" / "N075_01.BIN")
manifest = {Path(r["file"]).name: r for r in csv.DictReader(open(ROOT / "data" / "simulator" / "MANIFEST.CSV"))}
m = manifest[path.name]
out = LeanEngine().analyse(path.read_bytes(), int(m["fs"]), 1, 0, float(m["uV_per_count"]))
print(json.dumps({"record": path.name, "condition": m["condition"], "detector": out["detector"],
                  "counts": out["counts"], "n_classified": out["n_classified"]}, indent=1))
