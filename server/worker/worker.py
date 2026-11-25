"""Inference worker for the Go gateway on EC2 (bound to 127.0.0.1 only; never exposed to the network).

POST /infer?fs=200&leads=1&lead=0&uv=0.7324   body: int16 LE counts; uv = microvolts per count  -> analysis JSON
GET  /health                                                                   -> model, detector, onnx file
Runs the lean engine (no PyTorch), so it fits a 1-GiB EC2 instance.
The Go service owns authentication, integrity checks, storage and the public API; this process only analyses ECG.
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lean_infer import RUN, LeanEngine  # noqa: E402  (NumPy + ONNX Runtime; verified identical to the PyTorch pipeline)

ENGINE = LeanEngine()
LOCK = threading.Lock()


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"status": "ok", "model": RUN, "onnx": f"{RUN}.onnx", "model_load_ms": round(ENGINE.load_ms, 1), "detector": ENGINE.detector})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/infer":
            return self._send(404, {"error": "not found"})
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            with LOCK:
                out = ENGINE.analyse(body, int(q.get("fs", 200)), int(q.get("leads", 1)), int(q.get("lead", 0)),
                                     float(q.get("uv", 1.0)))
            self._send(422 if "error" in out else 200, out)
        except Exception as e:                                 # malformed input -> 400, never crash the worker
            self._send(400, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    print(f"worker ready on 127.0.0.1:{port} ({RUN}, detector {ENGINE.detector})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
