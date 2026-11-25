# AGENet

This repository contains the central parts of the study: the AGENet model, the patient-wise data split, training and the
paired R-peak benchmark, the cloud server (Go gateway and Python analysis worker), the ESP8266 firmware and the
42 five-minute simulator recordings.

## Contents

```
agenet/                 AGENet model, preprocessing, DS1/DS2 split, training, paired R-peak benchmark
  models/               AGENet (35,209 parameters) and the 1-D CBAM block
  data/                 preprocessing (0.5-40 Hz, 120 Hz), AAMI N/S/V mapping, patient split, beat/context windows
  train/                training (fixed 35-epoch schedule, no validation set, seed 1337)
  evaluation/           metrics, record/patient bootstrap, R-peak detectors, paired reference-vs-automatic benchmark
  runs/nv_B3_wce_s1337/ locked AGENet weights used in the paper
server/
  gateway/              Go HTTPS gateway: HMAC/nonce/timestamp/SHA-256 checks, MongoDB storage
  worker/               analysis worker (NumPy + ONNX Runtime): Hamilton R peaks + locked AGENet (ONNX)
  config/               placeholder configuration files (*.example)
firmware/               ESP8266 (ESP-12F) firmware: microSD storage and authenticated HTTPS upload (PlatformIO)
data/simulator/         42 five-minute ECG records (7 simulator runs x 6) and MANIFEST.CSV
examples/run_record.py  analyse one stored record with the deployed pipeline
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start: analyse a stored record

```bash
python examples/run_record.py data/simulator/N075_01.BIN
```

Expected output (identical to the cloud results reported in the paper):

| Record | Programmed rhythm | N | S | V |
|---|---|---|---|---|
| `N075_01.BIN` | sinus, 75 bpm | 362 | 0 | 0 |
| `PAC1_01.BIN` | premature atrial contractions | 313 | 78 | 0 |
| `PVC1_01.BIN` | premature ventricular contractions | 310 | 0 | 50 |

## Reproduce the classification results

1. Download the PhysioNet databases into `physionet/` (or set `AGENET_DATASET`):
   `mit-bih-arrhythmia/` ([MIT-BIH Arrhythmia](https://doi.org/10.13026/C2F305)),
   `st-petersburg-incart/` ([INCART](https://doi.org/10.13026/C2V88N)) and
   `mit-bih-supraventricular-arrhythmia/` ([SVDB](https://doi.org/10.13026/C2V30W)).
2. Run from `agenet/`:

```bash
cd agenet
python data/preprocess.py                                                   # filtered 120-Hz caches
python train/train.py --variant B3 --loss wce --protocol noval --seeds 1337 # trains on the 21 DS1 records
python evaluation/auto_peaks.py nv_B3_wce_s1337                             # reference vs Hamilton R peaks
python evaluation/auto_peaks.py nv_B3_wce_s1337 xqrs                        # comparison detectors
python evaluation/auto_peaks.py nv_B3_wce_s1337 pantompkins
```

The patient-wise split is defined in `agenet/data/patient_split.py`: DS1 (21 training records; record 201 is excluded
because it comes from the same patient as DS2 record 202) and DS2 (22 test records). The paced records 102, 104, 107 and
217 are not used. The locked weights in `agenet/runs/nv_B3_wce_s1337/` can be evaluated without retraining.

## Simulator records

`data/simulator/*.BIN`: 42 five-minute records, lead II, 200 Hz, 60,000 samples each, stored as 12-bit ADC codes in
little-endian 16-bit words (120,000 bytes). One code corresponds to 0.732 µV (gain 1100, 3.3-V reference, offset 2048).
`MANIFEST.CSV` gives, per record, the rhythm condition, sample count, SHA-256 digest and reference N/S/V beat counts.

## Server

```bash
# analysis worker (listens on 127.0.0.1:8090 only)
python server/worker/worker.py 8090

# gateway
cd server/gateway && go build -o gateway .
./gateway -addr :443 \
  -devices ../config/device_key.env -mongo-uri-file ../config/mongodb_uri \
  -tls-cert <tls.crt> -tls-key <tls.key> -worker http://127.0.0.1:8090
```

Create `server/config/device_key.env` and `server/config/mongodb_uri` from the `*.example` files.

## Firmware

1. Copy `firmware/include/secrets.example.h` to `firmware/include/secrets.h` and fill in your Wi-Fi settings, device
   key, gateway URL and certificate fingerprint.
2. Copy the records in `data/simulator/` to the `/ECG/` folder of the microSD card.
3. Build and flash with PlatformIO: `cd firmware && pio run -t upload` (set `upload_port` in `platformio.ini`).
4. Send commands over serial, e.g. `python tools/serial_cmd.py WIFI "UPLOAD /ECG/N075_01.BIN"`.

## Security note

No credentials are included. `secrets.h`, `device_key.env`, `mongodb_uri`, TLS keys and certificates are git-ignored;
only placeholder `*.example` files are provided. Generate your own device key (for example `openssl rand -hex 32`).

## Citation and license

Please cite the paper above. License: <to be added>.
