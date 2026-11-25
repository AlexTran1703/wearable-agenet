"""Send line commands to the ESP-12F node over serial and print its replies (never prints secrets: the firmware does
not echo credentials). Usage: python serial_cmd.py "INFO" "WIFI" "UPLOAD /ECG/PVC1_01.BIN"
Each command waits until a line starting with OK / ERR / PONG / INFO (or the timeout) arrives."""
from __future__ import annotations

import sys
import time

import serial

PORT, BAUD = "COM9", 460800


def run(cmds, timeout=420):
    with serial.Serial(PORT, BAUD, timeout=0.5) as s:
        s.dtr = False; s.rts = False
        time.sleep(1.5); s.reset_input_buffer()
        out = []
        for c in cmds:
            s.write((c + "\n").encode())
            t0 = time.time()
            while time.time() - t0 < timeout:
                line = s.readline().decode(errors="replace").strip()
                if not line:
                    continue
                print(f"[{time.time() - t0:7.2f}s] {line}", flush=True)
                out.append(line)
                if line.startswith(("OK,", "ERR", "PONG", "INFO,")):
                    break
        return out


if __name__ == "__main__":
    run(sys.argv[1:])


def put_files(files, base):
    """Copy local files to the SD card with PUT; verifies the SHA-256 reported by the node."""
    import hashlib
    from pathlib import Path
    ok = 0
    with serial.Serial(PORT, BAUD, timeout=2) as s:
        s.dtr = False; s.rts = False
        time.sleep(1.5); s.reset_input_buffer()
        for f in files:
            data = Path(f).read_bytes()
            dst = "/" + Path(f).relative_to(base).as_posix()
            s.write(f"PUT {dst} {len(data)}\n".encode())
            line = s.readline().decode(errors="replace").strip()
            if not line.startswith("READY"):
                print("no READY for", dst, line); continue
            t0 = time.time()
            for i in range(0, len(data), 512):                   # one '.' acknowledgement per 512-byte chunk
                s.write(data[i:i + 512]); s.flush()
                if s.read(1) != b".":
                    break
            reply = ""
            while not reply.startswith(("OK,PUT", "ERR")) and time.time() - t0 < 60:
                reply = s.readline().decode(errors="replace").strip()
            good = reply.startswith("OK,PUT") and reply.rsplit(",", 1)[-1] == hashlib.sha256(data).hexdigest()
            ok += good
            print(f"{dst} {len(data)} B {time.time() - t0:.1f}s {'sha ok' if good else 'FAILED ' + reply}", flush=True)
    return ok
