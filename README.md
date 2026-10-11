# AI-Assisted Multi-Sensor Surveillance and Event Prioritization System

**OFFLINE RESEARCH PROTOTYPE — not connected to operational military infrastructure,
live military sensors, or classified systems.**

---

## Overview

A complete research prototype that demonstrates multi-sensor observation ingestion,
correlation, and event prioritization using:

| Module | Sensor | Technology |
|--------|--------|------------|
| A | Satellite / aerial imagery | YOLOv8 object detection |
| B | Thermal / infrared imagery | YOLOv8 (separate instance) |
| C | Radar detections / tracks | Kalman filter + nearest-neighbour association |
| D | Sensor fusion & events | Rule-based uncertainty-aware pipeline |

The fusion component is **rule-based** — it is not a trained deep-learning model.
Priority is a **human review recommendation**, not a determination of hostile intent.

---

## Requirements

| Dependency | Minimum version | Notes |
|-----------|----------------|-------|
| Python | 3.10+ | 3.14 tested |
| Node.js | 18+ | v24 tested |
| npm | 9+ | v11 tested |
| Disk space | ~5 GB | For Python deps + model weights |
| RAM | 8 GB | 16 GB recommended for training |
| GPU | Optional | CUDA 11.8+ for faster inference/training |

---

## Quick Start (Windows)

### 1 — Backend setup (first time only)

```bat
setup_backend.bat
```

This creates a Python virtualenv in `backend\venv\` and installs all dependencies.

### 2 — Start backend

```bat
start_backend.bat
```

The API starts at **http://localhost:8000**  
Interactive docs: **http://localhost:8000/docs**

### 3 — Start frontend (separate terminal)

```bat
start_frontend.bat
```

The dashboard opens at **http://localhost:5173**

### 4 — Run tests

```bat
run_tests.bat
```

---

## Manual Commands

```bat
REM Backend
cd backend
venv\Scripts\activate
python -m uvicorn app.main:app --reload --port 8000

REM Frontend
cd frontend
npm run dev

REM Tests
cd backend
venv\Scripts\activate
python -m pytest tests/ -v
```

---

## Application Pages

| URL | Page | Description |
|-----|------|-------------|
| `/` | Overview Dashboard | KPIs, recent observations, events, training status |
| `/satellite` | Satellite Analysis | Upload image → YOLO inference → detections |
| `/thermal` | Thermal Analysis | Thermal image inference (separate model) |
| `/radar` | Radar Analysis | Upload CSV/JSON → Kalman filter tracking |
| `/fusion` | Sensor Fusion | Correlation decisions + event prioritization |
| `/models` | Model Management | Architecture, status, metrics, limitations |
| `/training` | Training | Configure and launch fine-tuning jobs |
| `/evaluation` | Test & Evaluation | Real eval / replay / synthetic integration test |
| `/datasets` | Datasets | Dataset registry, paths, setup instructions |

---

## Dataset Setup (Manual — Required for Training)

No datasets are downloaded automatically. Each requires registration.

### xView (Satellite)
1. Register at https://challenge.xviewdataset.org/
2. Download training images + labels (~20 GB)
3. Extract to `backend\data\datasets\xview\`
4. Convert GeoJSON annotations to YOLO format
5. Set local path in the Datasets page → Validate

### FLIR ADAS (Thermal)
1. Register at https://www.flir.com/oem/adas/adas-dataset-form/
2. Download FLIR ADAS dataset (~14 GB)
3. Extract to `backend\data\datasets\flir_adas\`
4. Convert COCO annotations to YOLO format
5. Set local path in the Datasets page → Validate

### DLR Marine Radar (Radar)
1. Contact DLR for access (maritime radar data — NOT aircraft radar)
2. Extract to `backend\data\datasets\dlr_radar\`
3. Import CSV/JSON via the Radar Analysis page

---

## Model Status

| Model | Status | Notes |
|-------|--------|-------|
| satellite_yolov8n | PRETRAINED (COCO) | Fine-tune on xView/DOTA after dataset download |
| thermal_yolov8n | PRETRAINED (COCO) | Fine-tune on FLIR ADAS after dataset download |
| Radar tracker | Rule-based Kalman | No neural model — classical tracking |

Models are labelled **PRETRAINED** until fine-tuning completes. No fake accuracy values are shown.

---

## Sample Data Generation

Generate test files to try the system without real datasets:

```bat
cd backend
venv\Scripts\activate

REM Radar detection test file
python ..\generate_sample_radar.py
REM → creates sample_radar_data.csv, upload on Radar Analysis page

REM Replay observation test file
python ..\generate_sample_replay.py
REM → creates sample_replay_observations.json, upload on Evaluation → Replay tab
```

---

## Synthetic Integration Test

Without any datasets, run the built-in synthetic test from the **Evaluation** page
(Synthetic Integration Test tab) to verify the full software pipeline:

- Observation creation and validation
- Missing metadata handling
- Cross-sensor correlation
- Duplicate detection
- Event prioritization
- Event review workflow

All synthetic records are clearly labelled **SYNTHETIC DEMONSTRATION — NOT LIVE SENSOR DATA**.

---

## Training Commands (Manual)

Once a dataset is installed and validated:

```bat
cd backend
venv\Scripts\activate

REM Satellite model — adjust paths and epochs as needed
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='data/datasets/xview/dataset.yaml', epochs=50, imgsz=640, device='cpu')
"

REM Thermal model
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='data/datasets/flir_adas/dataset.yaml', epochs=50, imgsz=640, device='cpu')
"
```

Or use the Training page in the dashboard to configure and launch jobs.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | System health check |
| GET | `/api/system/stats` | Dashboard statistics |
| GET | `/api/datasets` | List datasets |
| POST | `/api/datasets/validate/{id}` | Validate dataset files |
| GET | `/api/models` | List models |
| POST | `/api/inference/satellite` | Run satellite inference |
| POST | `/api/inference/thermal` | Run thermal inference |
| POST | `/api/inference/radar` | Import radar detections |
| GET | `/api/radar/tracks` | Get all radar tracks |
| GET | `/api/observations` | List observations |
| POST | `/api/correlation/run` | Run correlation |
| GET | `/api/events` | List events |
| PATCH | `/api/events/{id}/review` | Update review status |
| POST | `/api/training/start` | Start training job |
| POST | `/api/evaluation/start` | Start evaluation job |
| POST | `/api/replay/load` | Load replay file |
| POST | `/api/replay/control` | Control replay playback |
| POST | `/api/synthetic/run` | Run synthetic scenarios |
| GET | `/api/reports` | List evaluation reports |

Full interactive documentation: http://localhost:8000/docs

---

## Architecture

```
Sensor Input
    ├── Satellite Image → YOLOv8 (Module A) ─────────┐
    ├── Thermal Image   → YOLOv8 (Module B) ─────────┤
    └── Radar CSV/JSON  → Kalman Tracker (Module C) ──┤
                                                       ▼
                                          Common Observation Schema
                                                (Pydantic + SQLite)
                                                       ▼
                                          Metadata Validation
                                          (timestamp, CRS, coords)
                                                       ▼
                                          Rule-Based Correlation
                                          (temporal + spatial)
                                                       ▼
                                          Event Prioritization
                                          (LOW / MEDIUM / HIGH)
                                                       ▼
                                          Human Review Dashboard
                                          (React + Recharts)
```

---

## Known Limitations

1. **No datasets are installed** — training and real evaluation cannot run until datasets are downloaded
2. **Models are PRETRAINED on COCO** — not fine-tuned on satellite or thermal-specific data
3. **Radar tracker** processes maritime radar data only — not aircraft radar
4. **Fusion pipeline is rule-based** — not a trained multimodal model
5. **No GPU assumed** — CPU inference is slow for large images; use tiled inference for high-res aerial images
6. **SQLite** — suitable for research prototype; not for production multi-user deployment
7. **Pixel coordinates ≠ geographic coordinates** — spatial correlation requires valid lat/lon, not pixel bboxes

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Backend won't start | Check `venv\Scripts\activate` ran; check port 8000 not in use |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside venv |
| Frontend can't reach backend | Ensure backend is on port 8000; check CORS in config.py |
| `npm` not recognized | Install Node.js from nodejs.org |
| Inference returns error | Model loads from ultralytics on first use (needs internet once) |
| Training fails immediately | Dataset local path not set; dataset not VALIDATED |
| Tests fail on imports | Run pytest from inside `backend/` with venv activated |

---

## Project Structure

```
hackathon(pjm)/
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app, lifespan, CORS
│   │   ├── core/config.py        # Settings, directories
│   │   ├── db/
│   │   │   ├── database.py       # SQLAlchemy engine + session
│   │   │   └── models.py         # ORM models (all tables)
│   │   ├── schemas/schemas.py    # Pydantic schemas
│   │   ├── api/routes.py         # All API endpoints
│   │   └── services/
│   │       ├── satellite_detector.py   # Module A — YOLO satellite
│   │       ├── thermal_detector.py     # Module B — YOLO thermal
│   │       ├── radar_tracking.py       # Module C — Kalman tracker
│   │       ├── correlation.py          # Rule-based fusion
│   │       ├── prioritization.py       # Event priority scoring
│   │       ├── replay.py               # Recorded-data replay engine
│   │       └── synthetic_scenarios.py  # Integration test scenarios
│   ├── tests/                    # pytest test suite
│   ├── requirements.txt
│   └── pytest.ini
├── frontend/
│   ├── src/
│   │   ├── pages/                # 9 dashboard pages
│   │   ├── components/           # Sidebar, TopBar, Layout
│   │   ├── hooks/                # usePolling, useHealth
│   │   ├── utils/                # api.ts, format.ts
│   │   └── types/index.ts        # TypeScript interfaces
│   ├── package.json
│   └── vite.config.ts
├── setup_backend.bat
├── start_backend.bat
├── start_frontend.bat
├── run_tests.bat
├── generate_sample_radar.py
├── generate_sample_replay.py
└── README.md
```
