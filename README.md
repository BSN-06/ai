# AI-Assisted Multi-Sensor Surveillance and Event Prioritization System

**OFFLINE ACADEMIC RESEARCH PROTOTYPE.** It is not connected to operational military infrastructure, live sensors or
classified systems. Event priority is a **recommendation for human review**: it does not judge intent, does not predict
attacks, and is not an operational decision.

---

## 1. What it does

| Module | Input | Method | Status |
|---|---|---|---|
| **A** Satellite/aerial detection | Image upload (optional tiling for large scenes) | Ultralytics **YOLOv8n**, COCO-pretrained, fine-tuning pipeline for **xView** | Real inference ✔ · fine-tuning needs the xView dataset |
| **B** Thermal detection | Thermal image upload | **Separate** YOLOv8n with its own checkpoint/classes, fine-tuning pipeline for **FLIR ADAS** (thermal only) | Real inference ✔ · fine-tuning needs FLIR ADAS |
| **C** Radar tracking | Recorded detections CSV/JSON (x/y metres or lat/lon) | Linear **Kalman filter** (constant velocity) + **global-nearest-neighbour** (Hungarian) association, track creation/termination | Working ✔ (classical, no neural model) |
| **D** Fusion & prioritization | Common observation schema | Rule-based, uncertainty-aware correlation → grouping → duplicate collapse → explainable LOW/MEDIUM/HIGH | Working ✔ (**not** a trained fusion model) |

Three execution modes are kept separate and labelled everywhere:

* **Mode A: Real inference.** You upload an image or data file, and the actual model or tracker runs on it. A missing checkpoint returns an error, never a made-up prediction.
* **Mode B: Recorded-data replay.** Recorded observations are replayed in timestamp order with start, pause, resume, reset and speed controls. Each record is validated, stored, correlated, and creates or updates events. The UI labels this **"RECORDED DATA REPLAY — NOT A LIVE SENSOR FEED"**.
* **Mode C: Synthetic integration tests.** 14 deterministic scenarios run through the real software path, and the report shows expected vs. actual results as PASS/FAIL. The UI labels this **"SYNTHETIC DEMONSTRATION — NOT LIVE SENSOR DATA"**.

```
Sensor input ─┬─ Image  → Module A / B (YOLO) ─┐
              ├─ Radar  → Module C (KF + GNN) ─┤
              └─ Replay / synthetic records ───┤
                                               ▼
                 Common observation schema (Pydantic) + validation  PASS / WARNING / FAIL
                                               ▼
                 Temporal window + CRS check + haversine distance (geo positions only)
                                               ▼
                 Pairwise decisions: CORRELATED / NOT_CORRELATED / INSUFFICIENT_METADATA
                                               ▼
                 Connected-component grouping, duplicates collapsed (not double-counted)
                                               ▼
                 Explainable priority factors → Event (LOW / MEDIUM / HIGH) → human review
                                               ▼
                 Dashboard (React) · Reports (DB + JSON files)
```

**Stack:** FastAPI · Pydantic v2 · SQLAlchemy 2 + SQLite · PyTorch · Ultralytics YOLO · OpenCV · NumPy/Pandas/SciPy ·
pytest. The frontend uses React 18 + TypeScript + Vite + Recharts.

---

## 2. Requirements

| | Minimum | Verified with |
|---|---|---|
| Python | 3.10+ | 3.13 |
| Node.js | 18+ | 22 |
| RAM | 8 GB (16 GB+ for training) | — |
| Disk | ~6 GB for dependencies; **+20–60 GB** for xView/FLIR raw data plus converted tiles | — |
| GPU | Optional. CUDA is detected at runtime and never assumed | CPU-only |

Training on the full xView or FLIR datasets takes **days on a CPU** and hours on a modern GPU.

---

## 3. Windows quick start

```bat
REM 1) One-time backend setup: creates backend\venv, installs packages, initialises the DB, caches YOLOv8n weights
setup_backend.bat

REM 2) Start the API  → http://localhost:8000   (interactive docs: http://localhost:8000/docs)
start_backend.bat

REM 3) In a second terminal: start the dashboard → http://localhost:5173
start_frontend.bat

REM 4) Run the automated tests
run_tests.bat
REM    ...skipping the real-model tests (no torch / offline):
run_tests.bat -k "not real_models"
```

For a **CUDA GPU**, install the matching PyTorch build first, following
<https://pytorch.org/get-started/locally/>. Then run `pip install -r requirements.txt` inside `backend\venv`.

### Manual commands (any OS)

```bat
cd backend
python -m venv venv
venv\Scripts\activate            &REM Linux/macOS: source venv/bin/activate
pip install -r requirements.txt  &REM or requirements_core.txt (no torch/ultralytics)
python scripts\init_backend.py
python -m uvicorn app.main:app --port 8000

cd ..\frontend
npm install
npm run dev
```

### Trying it without any dataset

```bat
cd backend & venv\Scripts\activate
python ..\generate_sample_radar.py     &REM simulated radar CSV with reference ids → upload on the Radar page
python ..\generate_sample_replay.py    &REM simulated recorded observations → Evaluation → Recorded-data replay
```

You can also go to **Evaluation → Synthetic integration tests → Run**, or upload any photo on the Satellite or Thermal
page. A photo gives real YOLO inference with COCO-pretrained weights; it is not satellite performance.

---

## 4. Datasets (manual; nothing is downloaded automatically)

Place datasets under `backend\data\datasets\`, or add another folder to the `ALLOWED_DATASET_ROOTS` environment variable.
Paths outside these roots are rejected.

| Dataset | Access | Steps |
|---|---|---|
| **xView** (satellite, 60 classes) | Register at <https://challenge.xviewdataset.org/> | Extract `train_images` and `xView_train.geojson` to `data\datasets\xview_raw\`, then run `python scripts\convert_dataset.py --dataset xView --source xview_raw` |
| **FLIR ADAS** (thermal) | Form at <https://oem.flir.com/en-in/solutions/automotive/adas-dataset-form/> | Extract to `data\datasets\flir_adas_raw\`, then run `python scripts\convert_dataset.py --dataset FLIR_ADAS --source flir_adas_raw` |
| **DLR marine radar / AIS** | [DLR page](https://www.dlr.de/en/kn/about-us/departments/nautical-systems/real-world-marine-radar-datasets-for-target-tracking); access may need contact | Export detections as CSV (`timestamp`, `x,y` or `latitude,longitude`, optional `mmsi`/`truth_id`), then upload on the Radar page or run `scripts\evaluate_radar.py` |
| DOTA (optional) | <https://captain-whu.github.io/DOTA/dataset.html> | Registered but **not converted**: DOTA uses oriented boxes, which need an OBB model |

**What the converters do**

* **xView:** maps `type_id` to 60 contiguous classes (the official mapping) and tiles images to 640 px chips with 64 px overlap. Boxes are clipped to each tile and kept when ≥50 % of the box is visible. Official test labels are not public, so the official *train* images are split 80/10/10 by a hash of the image id. All tiles from one image stay in one split, which prevents leakage.
* **FLIR ADAS:** reads every COCO `*.json`, skips any folder containing `rgb` (visible-light images), and preserves FLIR's own train/val/test folders. Category names are kept exactly as given; classes with no instances are reported, not dropped.
* **Validation** (Datasets page or `POST /api/datasets/validate/{id}`) checks `dataset.yaml`, image/label pairing, label syntax, class-id range, coordinate range, required splits and cross-split leakage. It also saves a report.

---

## 5. Training & evaluation commands

The commands below use the same code path as the dashboard and write to the same database.

```bat
cd backend & venv\Scripts\activate

REM Satellite (Module A)
python scripts\train.py    --model satellite_yolov8n --dataset xView --epochs 50 --batch 16 --imgsz 640 --device cpu
python scripts\evaluate.py --model satellite_yolov8n --dataset xView --split test

REM Thermal (Module B)
python scripts\train.py    --model thermal_yolov8n --dataset FLIR_ADAS --epochs 50 --batch 16 --imgsz 640 --device cpu
python scripts\evaluate.py --model thermal_yolov8n --dataset FLIR_ADAS --split val

REM Radar (Module C): metrics only when reference columns exist
python scripts\evaluate_radar.py --file data\datasets\dlr_radar\recording.csv --gate 50 --max-misses 5
```

Use `--device cuda` (or `0`) when a GPU is available. Training refuses to start unless the dataset is **VALIDATED**,
and only one training job runs at a time. Per-epoch losses and validation mAP are saved, and **Cancel** stops the
job after the current batch.

Checkpoints are written to `backend\data\training_runs\<model>_<job>\weights\best.pt`. The model registry is
updated (status TRAINED) only when that file exists. After training, the checkpoint is automatically evaluated on
the held-out `test` split (or `val` if there is no test split).

**How metrics are computed.** `app/services/evaluation.py` compares each predicted box with the ground-truth labels:

* precision and recall at confidence 0.25 and IoU 0.5
* COCO-style 101-point AP per class, giving mAP@0.5 and mAP@0.5:0.95
* a confusion matrix (including background)
* mean latency per image

If the model's class mapping differs from the dataset's (a PRETRAINED COCO model, for example), the evaluation
report says so.

---

## 6. Model choices

* **YOLOv8n (both detectors).** It runs on a CPU, has maintained pretrained weights, and handles the axis-aligned
  boxes that both xView and FLIR provide. Ultralytics YOLOv8 has three parts:
  * a CSPDarknet-style backbone with C2f blocks and SPPF
  * a PAN-FPN neck working at strides 8, 16 and 32
  * a decoupled, anchor-free head with DFL box regression

  For small xView objects on a GPU, prefer `yolov8s/m` (change `variant` in the registry) together with tiling.
  YOLO is not optimal everywhere: oriented-box data such as DOTA would need an OBB model or a two-stage detector.
* **Radar.** Structured detections or tracks suit a Kalman filter: a linear constant-velocity model with Gaussian
  noise and two-point velocity initialisation. Global-nearest-neighbour association avoids the order dependence of
  greedy matching. Probabilistic association (JPDA/MHT) was not added, because nothing in the available data shows
  it is needed. Raw radar signals or images are **not** supported, and no neural radar model is claimed.
* **Fusion.** Rule-based, because no synchronised, labelled multimodal dataset is available. The decision rules:
  * Pixel boxes are never treated as positions.
  * Matching class labels alone never correlate two observations.
  * A mismatched or unknown CRS blocks distance comparison.
  * Missing timestamps or positions give `INSUFFICIENT_METADATA`.

  Every decision stores its evidence, reason, missing metadata and configuration.
* **Priority factors** (`app/services/prioritization.py`):
  * number of independent sensor types
  * mean confidence
  * correlation outcomes
  * class agreement
  * completeness
  * uncertainty
  * number of supporting observations
  * freshness relative to a reference time

  Duplicates are reported and excluded from support. Thresholds: HIGH ≥ 60, MEDIUM ≥ 30 (configurable).

---

## 7. Verified status (on the development machine, Linux, CPU-only)

| Item | Status |
|---|---|
| Backend starts, DB initialises, frontend builds and connects through the Vite proxy | ✔ verified |
| Automated tests | ✔ **149 passed** (`pytest tests`) |
| Real YOLO inference (satellite, tiled and untiled; thermal with modality check) | ✔ verified with COCO-pretrained weights |
| Missing checkpoint → explicit error, no fallback prediction | ✔ tested |
| Training pipeline end to end (1 epoch on a tiny generated fixture, then held-out evaluation and registry update) | ✔ tested. Software check only; the numbers are meaningless |
| Radar parsing (cartesian + geographic), tracking, reference-based metrics, missing-record warnings | ✔ tested |
| Correlation, duplicates, grouping, prioritization, event review + history, export | ✔ tested |
| Recorded replay: chronological order, skipped records, events created/updated, report | ✔ tested |
| Synthetic suite: 14/14 scenarios with real expected-vs-actual checks | ✔ tested |
| All dashboard pages driven in headless Chromium with no console errors | ✔ verified |
| **Fine-tuned satellite model / xView metrics** | ✖ **not done**: dataset not installed |
| **Fine-tuned thermal model / FLIR metrics** | ✖ **not done**: dataset not installed |
| **Radar accuracy on DLR data** | ✖ **not done**: dataset not available |
| Windows `.bat` scripts | Written to match the project files; **not executed** (the development machine runs Linux) |

Both registered models are honestly **PRETRAINED / NOT_EVALUATED**. No accuracy numbers are shown anywhere until an
evaluation job has computed them.

---

## 8. Known limitations

1. No datasets are bundled, so the detectors are COCO-pretrained only. COCO classes (car, person, …) are not xView or FLIR classes.
2. Uploaded images carry no capture time or georeference unless the operator supplies them. Those observations are therefore stored without a timestamp or position, and correlation reports `INSUFFICIENT_METADATA`. This is by design.
3. Replay processes previously generated observations; it does not re-run models on raw imagery.
4. Radar: single-hypothesis GNN tracking with a constant-velocity model, which may struggle with maneuvering or dense targets. The lat/lon to local-plane projection is equirectangular, which is fine over tens of kilometres.
5. Priority weights and thresholds are hand-set rules, not learned or validated against operator decisions.
6. SQLite with background threads suits a single-user research prototype only. It is not production-ready and is not certified.
7. The thermal check is a saturation heuristic. It flags colour images but is not a modality classifier.

---

## 9. Project layout

```
backend/
  app/
    main.py                     FastAPI app, lifespan, CORS, registry seeding
    core/config.py              settings (env-overridable)
    db/database.py, models.py   SQLAlchemy engine, tables (+ forward-only column migration)
    schemas/schemas.py          Pydantic schemas incl. the common observation schema
    api/routes.py               all REST endpoints
    services/
      observation_normalizer.py normalisation + PASS/WARNING/FAIL validation
      detector_base.py          shared YOLO loading/inference/tiling/drawing
      satellite_detector.py     Module A
      thermal_detector.py       Module B (+ modality check)
      radar_tracking.py         Module C (Kalman + GNN, parsing, evaluation)
      correlation.py            pairwise rules, duplicates, grouping
      prioritization.py         explainable review priority
      fusion_pipeline.py        correlation → events orchestration
      datasets.py               xView/COCO→YOLO conversion, validation, safe paths
      evaluation.py             precision/recall/mAP/confusion from predictions vs labels
      training.py               background training/evaluation jobs
      replay.py                 recorded-data replay engine
      synthetic_scenarios.py    deterministic integration suite
      reporting.py              report persistence (DB + JSON files)
  scripts/                      init_backend, convert_dataset, train, evaluate, evaluate_radar
  tests/                        pytest suite (149 tests)
frontend/src/
  pages/                        Overview, Satellite, Thermal, Radar, Fusion, Models, Training, Evaluation, Datasets
  components/                   Layout, Sidebar, TopBar, ui kit, ImageAnalysis
  utils/api.ts, types/index.ts  typed API client
setup_backend.bat · start_backend.bat · start_frontend.bat · run_tests.bat
generate_sample_radar.py · generate_sample_replay.py   (SIMULATED sample files)
```

Runtime data lives in `backend/data/` (git-ignored): `surveillance.db`, `uploads/`, `datasets/`, `checkpoints/`,
`training_runs/` and `reports/`.

## 10. Main API endpoints

`GET /api/health` · `GET /api/system/stats` · `GET /api/datasets` · `POST /api/datasets/validate/{id}` ·
`POST /api/datasets/{id}/convert` · `GET /api/models` · `POST /api/inference/{satellite|thermal|radar}` ·
`GET|POST /api/observations` · `POST /api/correlation/run` · `POST /api/events/auto-generate` ·
`GET /api/events/{id}` · `PATCH /api/events/{id}/review` · `GET /api/events/{id}/export` ·
`GET /api/training/preflight` · `POST /api/training/start` · `POST /api/training/cancel/{id}` ·
`POST /api/evaluation/start` · `POST /api/replay/load` · `POST /api/replay/control` · `GET /api/replay/status` ·
`POST /api/synthetic/run` · `GET /api/reports` · `GET /api/reports/export/{id}` · `GET /api/reports/download/{id}`.
Full interactive docs are at `/docs`.

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate `backend\venv`, then `pip install -r requirements.txt` |
| Inference returns 503 "pretrained weights…" | You need internet once to fetch `yolov8n.pt`, or copy it to `backend\data\checkpoints\pretrained\` |
| Training button disabled | Read the pre-flight panel. Usually the dataset is not converted or VALIDATED |
| "Path is outside the allowed dataset roots" | Move the data under `backend\data\datasets` or set `ALLOWED_DATASET_ROOTS` |
| Frontend shows "Backend unreachable" | Start the backend on port 8000 first |
| `npm` not recognised | Install Node.js 18+ from nodejs.org |
| Port in use | `python -m uvicorn app.main:app --port 8001` and update `frontend\vite.config.ts` |
