"""
Real-model integration tests (Ultralytics + PyTorch required; skipped otherwise).

These verify that the software runs actual inference / training / evaluation end to end.
The generated images and tiny dataset are SOFTWARE TEST FIXTURES; the resulting numbers say
nothing about real satellite or thermal detection performance.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ultralytics = pytest.importorskip("ultralytics")

from app.services.detector_base import pretrained_weights_path  # noqa: E402


def _png_bytes(color=True, size=(320, 240)):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    if color:
        img[:, :, 2] = 200
        img[60:180, 100:220] = (30, 180, 40)
    else:
        img[:] = 60
        img[60:180, 100:220] = 220
    return cv2.imencode(".png", img)[1].tobytes()


def test_real_satellite_inference_tiled(client):
    r = client.post("/api/inference/satellite", params={"tile_size": 256, "confidence": 0.05},
                    files={"file": ("scene.png", _png_bytes(size=(700, 500)), "image/png")})
    if r.status_code == 503:
        pytest.skip(f"Pretrained weights unavailable: {r.json()}")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["provenance"] == "real_inference"
    assert d["model_status"] == "PRETRAINED"
    assert d["tiled"] is True
    assert d["inference_time_ms"] > 0
    assert any("timestamp" in w for w in d["warnings"])  # no capture time was given -> not invented
    assert client.get(d["annotated_image_url"]).status_code == 200
    for det in d["detections"]:
        assert 0 <= det["bbox_x1"] <= det["bbox_x2"] <= 700


def test_real_thermal_inference_flags_colour_image(client):
    r = client.post("/api/inference/thermal", files={"file": ("c.png", _png_bytes(color=True), "image/png")})
    if r.status_code == 503:
        pytest.skip("weights unavailable")
    d = r.json()
    assert d["modality_check"]["thermal_like"] is False
    assert any("MODALITY MISMATCH" in w for w in d["warnings"])
    r = client.post("/api/inference/thermal", files={"file": ("g.png", _png_bytes(color=False), "image/png")})
    assert r.json()["modality_check"]["thermal_like"] is True


def test_requested_missing_checkpoint_returns_error(client, db_session):
    from app.db import models as dbm
    from app.db.database import SessionLocal
    s = SessionLocal()
    m = dbm.ModelRegistry(name="sat_missing_ckpt", sensor_type="satellite", architecture="YOLOv8",
                          variant="yolov8n.pt", checkpoint_path="/nonexistent/best.pt", trained=True, status="TRAINED")
    s.add(m); s.commit()
    mid = m.id
    s.close()
    r = client.post("/api/inference/satellite", params={"model_id": mid},
                    files={"file": ("a.png", _png_bytes(), "image/png")})
    assert r.status_code == 503
    assert "Checkpoint not found" in r.json()["detail"]
    s = SessionLocal(); s.delete(s.get(dbm.ModelRegistry, mid)); s.commit(); s.close()


def _tiny_yolo_dataset(root: Path):
    """Bright squares on dark background; class 0 = square. Software fixture only."""
    rng = np.random.default_rng(0)
    for split, n in (("train", 8), ("val", 4), ("test", 4)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.full((96, 96, 3), 20, np.uint8)
            x, y, s = int(rng.integers(10, 50)), int(rng.integers(10, 50)), 30
            img[y:y + s, x:x + s] = 230
            cv2.imwrite(str(root / "images" / split / f"{split}_{i}.jpg"), img)
            (root / "labels" / split / f"{split}_{i}.txt").write_text(
                f"0 {(x + s / 2) / 96:.6f} {(y + s / 2) / 96:.6f} {s / 96:.6f} {s / 96:.6f}")
    from app.services.datasets import write_dataset_yaml
    write_dataset_yaml(root, ["square"])


def test_real_training_job_end_to_end(client, tmp_path, monkeypatch):
    """Runs a genuine 1-epoch CPU training job + held-out evaluation through the API."""
    from app.core.config import settings, BASE_DIR
    from app.db.database import SessionLocal
    from app.db import models as dbm
    if not pretrained_weights_path("yolov8n.pt").exists():
        pytest.skip("pretrained weights not available offline")
    root = (BASE_DIR / settings.DATASET_DIR / "pytest_squares_yolo")
    import shutil
    shutil.rmtree(root, ignore_errors=True)
    _tiny_yolo_dataset(root)

    s = SessionLocal()
    ds = dbm.Dataset(name="PYTEST_SQUARES", sensor_type="satellite", validation_status="NOT_VALIDATED")
    model = dbm.ModelRegistry(name="pytest_sat_model", sensor_type="satellite", architecture="YOLOv8",
                              variant="yolov8n.pt", status="PRETRAINED")
    s.add_all([ds, model]); s.commit()
    ds_id, model_id = ds.id, model.id
    s.close()
    try:
        assert client.patch(f"/api/datasets/{ds_id}/path", params={"local_path": str(root)}).status_code == 200
        v = client.post(f"/api/datasets/validate/{ds_id}").json()
        assert v["status"] == "VALIDATED", v
        r = client.post("/api/training/start", json={"model_id": model_id, "dataset_name": "PYTEST_SQUARES",
                                                     "epochs": 1, "batch_size": 4, "image_size": 96, "device": "cpu"})
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]
        for _ in range(600):
            job = client.get(f"/api/training/jobs/{job_id}").json()
            if job["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
                break
            time.sleep(0.5)
        assert job["status"] == "COMPLETED", job["error_message"]
        assert job["epoch_history"] and job["epoch_history"][0]["epoch"] == 1
        assert Path(job["checkpoint_path"]).exists()
        m = client.get(f"/api/models/{model_id}").json()
        assert m["trained"] is True and m["status"] == "TRAINED"
        for _ in range(240):
            evs = [e for e in client.get("/api/evaluation/jobs").json() if e["model_id"] == model_id]
            if evs and evs[0]["status"] in ("COMPLETED", "FAILED"):
                break
            time.sleep(0.5)
        assert evs[0]["status"] == "COMPLETED", evs[0]["error_message"]
        assert evs[0]["dataset_split"] == "test" and evs[0]["num_images"] == 4
        assert evs[0]["map50"] is not None  # computed from predictions vs labels (value not asserted)
    finally:
        s = SessionLocal()
        for j in s.query(dbm.EvaluationJob).filter(dbm.EvaluationJob.model_id == model_id).all():
            s.delete(j)
        for j in s.query(dbm.TrainingJob).filter(dbm.TrainingJob.model_id == model_id).all():
            s.delete(j)
        s.delete(s.get(dbm.ModelRegistry, model_id)); s.delete(s.get(dbm.Dataset, ds_id)); s.commit(); s.close()
        shutil.rmtree(root, ignore_errors=True)
