"""
API routes for the surveillance system.
Domains: system, datasets, models, inference, observations, correlation, events, training,
evaluation, replay, synthetic, reports, files, radar.

OFFLINE RESEARCH PROTOTYPE. Execution modes are kept separate and labelled:
  real_inference  – uploaded image/data processed by an actual model or tracker
  recorded_replay – recorded observations replayed in timestamp order (never a live feed)
  synthetic       – deterministic integration scenarios (never real sensor data)
"""
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import settings, BASE_DIR
from app.db import models as dbm
from app.db.database import get_db
from app.schemas.schemas import (
    SystemHealth, ObservationCreate, EventCreate, EventReviewUpdate, DatasetResponse, ModelResponse,
    TrainingJobCreate, EvaluationJobCreate, ReplayControl,
)
from app.services.observation_normalizer import validate_observation, observation_to_dict

logger = logging.getLogger(__name__)
router = APIRouter()

DATA_DIR = BASE_DIR / "data"


def _iso(dt):
    return dt.isoformat() if dt else None


def _log(db: Session, level: str, component: str, message: str, details: Optional[dict] = None):
    db.add(dbm.ProcessingLog(level=level, component=component, message=message, details=details))
    db.commit()


# ═══════════════════════════════ SYSTEM ═══════════════════════════════

@router.get("/health", response_model=SystemHealth)
def system_health(db: Session = Depends(get_db)):
    """System health check with component status."""
    from app.services.detector_base import detect_device
    device = detect_device()
    return SystemHealth(
        status="operational",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        database="connected",
        models_available=db.query(dbm.ModelRegistry).count(),
        datasets_registered=db.query(dbm.Dataset).count(),
        total_observations=db.query(dbm.Observation).count(),
        total_events=db.query(dbm.Event).count(),
        pending_events=db.query(dbm.Event).filter(dbm.Event.review_status == "PENDING").count(),
        gpu_available=device == "cuda",
        device=device,
        timestamp=datetime.utcnow(),
    )


def _dependency_status() -> dict:
    out = {}
    for mod in ("torch", "ultralytics", "cv2", "scipy"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "installed")
        except ImportError:
            out[mod] = None
    return out


@router.get("/system/stats")
def system_stats(db: Session = Depends(get_db)):
    """Dashboard statistics — every number comes from database records or in-memory service state."""
    from app.services.replay import replay_engine
    from app.services.radar_tracking import radar_tracker

    obs_by_sensor = {s: db.query(dbm.Observation).filter(dbm.Observation.sensor_type == s).count()
                     for s in ("satellite", "thermal", "radar")}
    obs_by_prov = {}
    for p in ("real_inference", "recorded", "synthetic"):
        obs_by_prov[p] = db.query(dbm.Observation).filter(dbm.Observation.data_provenance == p).count()
    events_by_priority = {p: db.query(dbm.Event).filter(dbm.Event.review_priority == p).count()
                          for p in ("LOW", "MEDIUM", "HIGH")}
    events_by_status = {s: db.query(dbm.Event).filter(dbm.Event.review_status == s).count()
                        for s in ("PENDING", "REVIEWING", "REVIEWED", "DISMISSED")}
    recent_obs = db.query(dbm.Observation).order_by(dbm.Observation.created_at.desc()).limit(15).all()
    recent_events = db.query(dbm.Event).order_by(dbm.Event.created_at.desc()).limit(10).all()
    training_jobs = db.query(dbm.TrainingJob).order_by(dbm.TrainingJob.created_at.desc()).limit(5).all()
    eval_jobs = db.query(dbm.EvaluationJob).order_by(dbm.EvaluationJob.created_at.desc()).limit(5).all()
    errors = db.query(dbm.ProcessingLog).filter(dbm.ProcessingLog.level.in_(["ERROR", "WARNING"])) \
        .order_by(dbm.ProcessingLog.timestamp.desc()).limit(10).all()
    activity = db.query(dbm.ProcessingLog).order_by(dbm.ProcessingLog.timestamp.desc()).limit(15).all()
    latency_logs = db.query(dbm.ProcessingLog).filter(dbm.ProcessingLog.component.in_(
        ["satellite_inference", "thermal_inference", "radar_tracking"])) \
        .order_by(dbm.ProcessingLog.timestamp.desc()).limit(30).all()
    models = db.query(dbm.ModelRegistry).all()
    datasets = db.query(dbm.Dataset).all()
    rstatus = replay_engine.get_status()
    mode = "recorded_replay" if rstatus["status"] in ("PLAYING", "PAUSED") else "idle"

    return {
        "observations": {"total": db.query(dbm.Observation).count(), "by_sensor": obs_by_sensor,
                         "by_provenance": obs_by_prov},
        "events": {"total": db.query(dbm.Event).count(), "pending": events_by_status["PENDING"],
                   "by_priority": events_by_priority, "by_status": events_by_status},
        "models": {
            "total": len(models),
            "trained": sum(1 for m in models if m.trained),
            "items": [{"id": m.id, "name": m.name, "sensor_type": m.sensor_type, "status": m.status,
                       "evaluation_status": m.evaluation_status,
                       "checkpoint_exists": bool(m.checkpoint_path and Path(m.checkpoint_path).exists())}
                      for m in models],
            "radar_tracker": {"type": "Kalman filter + GNN association (classical, no neural model)",
                              "available": True},
        },
        "datasets": {"total": len(datasets),
                     "items": [{"name": d.name, "sensor_type": d.sensor_type,
                                "validation_status": d.validation_status} for d in datasets]},
        "dependencies": _dependency_status(),
        "processing_mode": mode,
        "replay": {k: rstatus[k] for k in ("status", "processed_records", "total_records", "current_timestamp",
                                           "replay_speed", "queue_remaining", "label")},
        "radar": radar_tracker.get_status(),
        "training_jobs": [{"id": j.id, "status": j.status, "model_id": j.model_id, "dataset_name": j.dataset_name,
                           "current_epoch": j.current_epoch, "total_epochs": j.total_epochs,
                           "created_at": _iso(j.created_at)} for j in training_jobs],
        "evaluation_jobs": [{"id": j.id, "status": j.status, "model_id": j.model_id, "dataset_name": j.dataset_name,
                             "map50": j.map50, "created_at": _iso(j.created_at)} for j in eval_jobs],
        "recent_observations": [{
            "id": o.id, "sensor_type": o.sensor_type, "object_class": o.object_class, "confidence": o.confidence,
            "data_provenance": o.data_provenance, "observation_timestamp": _iso(o.observation_timestamp),
            "created_at": _iso(o.created_at)} for o in recent_obs],
        "recent_events": [{
            "id": e.id, "review_priority": e.review_priority, "review_status": e.review_status,
            "source_sensors": e.source_sensors, "evidence_summary": e.evidence_summary,
            "data_provenance": e.data_provenance, "created_at": _iso(e.created_at)} for e in recent_events],
        "latency": [{"component": l.component, "latency_ms": (l.details or {}).get("latency_ms"),
                     "timestamp": _iso(l.timestamp)} for l in reversed(latency_logs)],
        "errors": [{"level": e.level, "component": e.component, "message": e.message,
                    "timestamp": _iso(e.timestamp)} for e in errors],
        "activity": [{"level": a.level, "component": a.component, "message": a.message,
                      "timestamp": _iso(a.timestamp)} for a in activity],
    }


@router.get("/system/logs")
def get_system_logs(level: Optional[str] = Query(None), limit: int = Query(50, ge=1, le=500),
                    offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    query = db.query(dbm.ProcessingLog)
    if level:
        query = query.filter(dbm.ProcessingLog.level == level.upper())
    total = query.count()
    logs = query.order_by(dbm.ProcessingLog.timestamp.desc()).offset(offset).limit(limit).all()
    return {"total": total, "logs": [{"id": l.id, "level": l.level, "component": l.component, "message": l.message,
                                      "details": l.details, "timestamp": _iso(l.timestamp)} for l in logs]}


# ═══════════════════════════════ DATASETS ═══════════════════════════════

def _dataset_or_404(db, dataset_id):
    ds = db.get(dbm.Dataset, dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    return ds


@router.get("/datasets", response_model=list[DatasetResponse])
def list_datasets(db: Session = Depends(get_db)):
    return db.query(dbm.Dataset).all()


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    return _dataset_or_404(db, dataset_id)


@router.post("/datasets/register")
def register_dataset(name: str, sensor_type: str, local_path: Optional[str] = None, db: Session = Depends(get_db)):
    from app.services.datasets import resolve_dataset_path
    if sensor_type not in ("satellite", "thermal", "radar"):
        raise HTTPException(400, "sensor_type must be satellite, thermal or radar")
    path = None
    if local_path:
        try:
            path = str(resolve_dataset_path(local_path))
        except ValueError as e:
            raise HTTPException(400, str(e))
    existing = db.query(dbm.Dataset).filter(dbm.Dataset.name == name).first()
    if existing:
        if path:
            existing.local_path, existing.validation_status = path, "NOT_VALIDATED"
            db.commit()
        return {"status": "updated", "dataset_id": existing.id}
    ds = dbm.Dataset(name=name, sensor_type=sensor_type, local_path=path, validation_status="NOT_VALIDATED")
    db.add(ds)
    db.commit()
    return {"status": "registered", "dataset_id": ds.id}


@router.patch("/datasets/{dataset_id}/path")
def update_dataset_path(dataset_id: str, local_path: str, db: Session = Depends(get_db)):
    from app.services.datasets import resolve_dataset_path
    ds = _dataset_or_404(db, dataset_id)
    try:
        ds.local_path = str(resolve_dataset_path(local_path))
    except ValueError as e:
        raise HTTPException(400, str(e))
    ds.validation_status = "NOT_VALIDATED"
    db.commit()
    return {"status": "updated", "local_path": ds.local_path}


@router.get("/datasets/{dataset_id}/inspect")
def inspect_dataset(dataset_id: str, db: Session = Depends(get_db)):
    from app.services.datasets import inspect_directory
    ds = _dataset_or_404(db, dataset_id)
    if not ds.local_path:
        return {"exists": False, "message": "No local path set."}
    return inspect_directory(Path(ds.local_path))


def _apply_validation(db, ds, report: dict):
    ds.validation_status = report["status"]
    splits = report.get("splits", {})
    ds.train_count = splits.get("train", {}).get("images")
    ds.val_count = splits.get("val", {}).get("images")
    ds.test_count = splits.get("test", {}).get("images")
    if splits:
        ds.file_count = sum(v.get("images", 0) for v in splits.values())
    if report.get("classes"):
        ds.classes = report["classes"]
        ds.num_classes = len(report["classes"])
    msgs = report.get("errors", []) + report.get("warnings", [])
    ds.validation_message = " | ".join(msgs) if msgs else "All integrity checks passed."
    db.commit()


@router.post("/datasets/validate/{dataset_id}")
def validate_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Validate a dataset (YOLO layout, label syntax, class ids, splits, leakage) and save a report."""
    from app.services.datasets import validate_yolo_dataset, inspect_directory
    from app.services.reporting import save_report
    ds = _dataset_or_404(db, dataset_id)
    if not ds.local_path or not Path(ds.local_path).exists():
        ds.validation_status = "NOT_FOUND"
        ds.validation_message = f"Local path does not exist: {ds.local_path}"
        db.commit()
        return {"status": "NOT_FOUND", "message": ds.validation_message}
    if ds.sensor_type == "radar":
        info = inspect_directory(Path(ds.local_path))
        csvs = list(Path(ds.local_path).rglob("*.csv")) + list(Path(ds.local_path).rglob("*.json"))
        report = {"status": "VALIDATED" if csvs else "EMPTY", "errors": [] if csvs else ["No CSV/JSON files found."],
                  "warnings": [], "splits": {}, "files": [str(c) for c in csvs[:50]], "info": info}
        ds.file_count = len(csvs)
    else:
        report = validate_yolo_dataset(Path(ds.local_path))
    _apply_validation(db, ds, report)
    rep = save_report(db, report_type="dataset_validation", test_mode="dataset", source=ds.name,
                      num_samples=ds.file_count or 0, metrics={"status": report["status"]},
                      warnings=report.get("warnings"), completion_status=report["status"], body=report)
    return {"status": ds.validation_status, "message": ds.validation_message, "file_count": ds.file_count,
            "report": report, "report_id": rep.id}


@router.post("/datasets/{dataset_id}/convert")
def convert_dataset(dataset_id: str, source_path: str = Query(..., description="Raw dataset folder"),
                    tile_size: int = Query(640, ge=256, le=2048), max_images: Optional[int] = Query(None, ge=1),
                    db: Session = Depends(get_db)):
    """Convert a raw xView (GeoJSON) or FLIR/COCO dataset into YOLO format in the background."""
    from app.services.datasets import resolve_dataset_path, dataset_output_dir, inspect_directory
    from app.services.training import start_in_background
    ds = _dataset_or_404(db, dataset_id)
    try:
        src = resolve_dataset_path(source_path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    info = inspect_directory(src)
    if not info["exists"]:
        raise HTTPException(404, f"Source path not found: {src}")
    if info["format"] not in ("xview_geojson", "coco_json"):
        raise HTTPException(400, f"Unsupported or undetected raw format '{info['format']}'. "
                                 "Expected xView .geojson or COCO .json annotations.")
    out = dataset_output_dir(ds.name)
    ds.validation_status, ds.validation_message = "CONVERTING", f"Converting {info['format']} from {src}"
    db.commit()

    def _job(ds_id, fmt, src, out):
        from app.db.database import SessionLocal
        from app.services import datasets as svc
        s = SessionLocal()
        try:
            if fmt == "xview_geojson":
                res = svc.convert_xview_to_yolo(src, out, tile_size=tile_size, max_images=max_images)
            else:
                res = svc.convert_coco_to_yolo(src, out, modality_filter="thermal" if ds.sensor_type == "thermal" else None)
            d = s.get(dbm.Dataset, ds_id)
            d.local_path = str(out)
            report = svc.validate_yolo_dataset(out)
            _apply_validation(s, d, report)
            s.add(dbm.ProcessingLog(level="INFO", component="dataset_conversion",
                                    message=f"Converted {d.name}", details=res))
            s.commit()
        except Exception as e:
            d = s.get(dbm.Dataset, ds_id)
            d.validation_status, d.validation_message = "CONVERSION_FAILED", str(e)
            s.add(dbm.ProcessingLog(level="ERROR", component="dataset_conversion", message=str(e)))
            s.commit()
        finally:
            s.close()

    start_in_background(_job, ds.id, info["format"], src, out)
    return {"status": "CONVERTING", "source": str(src), "output": str(out), "format": info["format"]}


# ═══════════════════════════════ MODELS ═══════════════════════════════

@router.get("/models", response_model=list[ModelResponse])
def list_models(db: Session = Depends(get_db)):
    return db.query(dbm.ModelRegistry).all()


@router.get("/models/{model_id}", response_model=ModelResponse)
def get_model(model_id: str, db: Session = Depends(get_db)):
    model = db.get(dbm.ModelRegistry, model_id)
    if not model:
        raise HTTPException(404, "Model not found")
    return model


@router.get("/models/{model_id}/load-status")
def get_model_load_status(model_id: str, db: Session = Depends(get_db)):
    from app.services.satellite_detector import satellite_detector
    from app.services.thermal_detector import thermal_detector
    model = db.get(dbm.ModelRegistry, model_id)
    if not model:
        raise HTTPException(404, "Model not found")
    det = {"satellite": satellite_detector, "thermal": thermal_detector}.get(model.sensor_type)
    return {"model_id": model_id, "db_status": model.status,
            "checkpoint_exists": bool(model.checkpoint_path and Path(model.checkpoint_path).exists()),
            "in_memory_loaded": bool(det and det.is_loaded), "runtime_status": det.get_status() if det else {}}


# ═══════════════════════════════ INFERENCE ═══════════════════════════════

def _save_upload(file: UploadFile, subdir: str = "uploads", kind: str = "any") -> str:
    """Stream an upload to a server-generated filename, enforcing extension and size limits."""
    upload_dir = DATA_DIR / subdir
    upload_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    allowed = {"image": settings.ALLOWED_IMAGE_EXTENSIONS, "data": settings.ALLOWED_DATA_EXTENSIONS}.get(
        kind, settings.ALLOWED_IMAGE_EXTENSIONS + settings.ALLOWED_DATA_EXTENSIONS)
    if ext not in allowed:
        raise HTTPException(400, f"File type '{ext}' not allowed. Allowed: {allowed}")
    file_path = upload_dir / f"{uuid.uuid4()}{ext}"
    limit = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    size = 0
    with open(file_path, "wb") as f:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                f.close()
                file_path.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit")
            f.write(chunk)
    if size == 0:
        file_path.unlink(missing_ok=True)
        raise HTTPException(400, "Uploaded file is empty")
    return str(file_path)


def _resolve_model(db, sensor: str, model_id: Optional[str]) -> Optional[dbm.ModelRegistry]:
    if model_id:
        m = db.get(dbm.ModelRegistry, model_id)
        if not m or m.sensor_type != sensor:
            raise HTTPException(404, f"No {sensor} model with id {model_id}")
        return m
    return db.query(dbm.ModelRegistry).filter(dbm.ModelRegistry.sensor_type == sensor) \
        .order_by(dbm.ModelRegistry.trained.desc()).first()


def _image_inference(sensor: str, file: UploadFile, confidence: float, iou: float, model_id: Optional[str],
                     tile_size: Optional[int], captured_at: Optional[str], latitude: Optional[float],
                     longitude: Optional[float], position_uncertainty_m: Optional[float], db: Session):
    import cv2
    from app.services.satellite_detector import satellite_detector
    from app.services.thermal_detector import thermal_detector
    detector = satellite_detector if sensor == "satellite" else thermal_detector

    file_path = _save_upload(file, kind="image")
    if cv2.imread(file_path) is None:
        Path(file_path).unlink(missing_ok=True)
        raise HTTPException(400, "Uploaded file is not a readable image.")
    if (latitude is None) != (longitude is None):
        raise HTTPException(400, "Provide both latitude and longitude, or neither.")

    model = _resolve_model(db, sensor, model_id)
    load = detector.load_model(checkpoint_path=model.checkpoint_path if model else None,
                               variant=(model.variant if model else None) or "yolov8n.pt")
    if load["status"] == "error":
        _log(db, "ERROR", f"{sensor}_inference", load["message"])
        raise HTTPException(503, load["message"])

    result = detector.run_inference(file_path, confidence, iou, tile_size=tile_size)
    if result["status"] == "error":
        _log(db, "ERROR", f"{sensor}_inference", result["message"])
        raise HTTPException(500, result["message"])

    ts = None
    if captured_at:
        from app.services.observation_normalizer import parse_timestamp
        try:
            ts, _ = parse_timestamp(captured_at)
        except ValueError as e:
            raise HTTPException(400, str(e))
    warnings = []
    if ts is None:
        warnings.append("No capture timestamp supplied — observations stored without observation_timestamp "
                        "(processing time is kept in metadata). Temporal correlation will be INSUFFICIENT_METADATA.")
    if latitude is None:
        warnings.append("No georeference supplied — pixel boxes are NOT geographic positions; "
                        "spatial correlation will be INSUFFICIENT_METADATA.")

    obs_ids = []
    for det_idx, det in enumerate(result["detections"]):
        raw = {
            "sensor_type": sensor, "observation_timestamp": ts, "is_utc": ts is not None,
            "source_dataset": "user_upload", "source_record_id": f"{Path(file_path).name}#det{det_idx}",
            "object_class": det["class_name"], "confidence": det["confidence"],
            "bbox_x1": det["bbox_x1"], "bbox_y1": det["bbox_y1"], "bbox_x2": det["bbox_x2"], "bbox_y2": det["bbox_y2"],
            "latitude": latitude, "longitude": longitude,
            "coordinate_crs": "WGS84" if latitude is not None else None,
            "uncertainty": position_uncertainty_m if latitude is not None else None,
            "model_name": result["model_name"], "model_version": result.get("model_status"),
            "checkpoint_id": result["checkpoint"], "data_provenance": "real_inference",
            "image_path": file_path, "annotated_image_path": result["annotated_image_path"],
            "metadata_json": {"processed_at": result["timestamp"], "inference_time_ms": result["inference_time_ms"],
                              "georeference": "operator-supplied scene location" if latitude is not None else None,
                              "class_id": det["class_id"]},
        }
        v = validate_observation(raw)
        if v["observation"] is None:
            warnings.append(f"Detection rejected by validation: {v['errors']}")
            continue
        o = dbm.Observation(**v["observation"].model_dump())
        db.add(o)
        db.flush()
        obs_ids.append(o.id)
    db.commit()
    _log(db, "WARNING" if result.get("warning") else "INFO", f"{sensor}_inference",
         f"{sensor} inference: {result['num_detections']} detections in {result['inference_time_ms']} ms",
         {"latency_ms": result["inference_time_ms"], "model": result["model_name"],
          "model_status": result.get("model_status"), "file": Path(file_path).name})
    result["observation_ids"] = obs_ids
    result["warnings"] = warnings + ([result["warning"]] if result.get("warning") else [])
    result["image_url"] = f"/api/files/uploads/{Path(file_path).name}"
    result["annotated_image_url"] = f"/api/files/uploads/{Path(result['annotated_image_path']).name}"
    result["model_registry_id"] = model.id if model else None
    return result


@router.post("/inference/satellite")
def satellite_inference(file: UploadFile = File(...), confidence: float = Query(0.25, ge=0.0, le=1.0),
                        iou: float = Query(0.45, ge=0.0, le=1.0), model_id: Optional[str] = Query(None),
                        tile_size: Optional[int] = Query(None, ge=256, le=4096),
                        captured_at: Optional[str] = Query(None), latitude: Optional[float] = Query(None, ge=-90, le=90),
                        longitude: Optional[float] = Query(None, ge=-180, le=180),
                        position_uncertainty_m: Optional[float] = Query(None, ge=0), db: Session = Depends(get_db)):
    """Module A — real satellite/aerial detection (optional tiled inference for large scenes)."""
    return _image_inference("satellite", file, confidence, iou, model_id, tile_size, captured_at,
                            latitude, longitude, position_uncertainty_m, db)


@router.post("/inference/thermal")
def thermal_inference(file: UploadFile = File(...), confidence: float = Query(0.25, ge=0.0, le=1.0),
                      iou: float = Query(0.45, ge=0.0, le=1.0), model_id: Optional[str] = Query(None),
                      captured_at: Optional[str] = Query(None), latitude: Optional[float] = Query(None, ge=-90, le=90),
                      longitude: Optional[float] = Query(None, ge=-180, le=180),
                      position_uncertainty_m: Optional[float] = Query(None, ge=0), db: Session = Depends(get_db)):
    """Module B — real thermal detection with modality check."""
    return _image_inference("thermal", file, confidence, iou, model_id, None, captured_at,
                            latitude, longitude, position_uncertainty_m, db)


@router.post("/inference/radar")
def radar_import(file: UploadFile = File(...), reset: bool = Query(True),
                 association_threshold: Optional[float] = Query(None, gt=0),
                 max_misses: Optional[int] = Query(None, ge=1, le=100),
                 source_dataset: str = Query("user_upload"), db: Session = Depends(get_db)):
    """Module C — parse recorded radar detections and run Kalman/GNN tracking."""
    import numpy as np
    from app.services.radar_tracking import radar_tracker
    from app.services.reporting import save_report

    file_path = _save_upload(file, "radar_data", kind="data")
    ext = Path(file_path).suffix.lower()
    if ext not in (".csv", ".json"):
        raise HTTPException(400, f"Unsupported radar data format: {ext}. Use CSV or JSON.")
    if reset:
        radar_tracker.reset()
    if association_threshold:
        radar_tracker.association_threshold = association_threshold
    if max_misses:
        radar_tracker.max_misses = max_misses
    try:
        detections = radar_tracker.parse_csv_data(file_path) if ext == ".csv" else radar_tracker.parse_json_data(file_path)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))
    if not detections:
        raise HTTPException(400, "No valid radar detections found in file.")

    result = radar_tracker.process_detections(detections)
    tracks = radar_tracker.get_tracks()
    obs_ids = []
    for track in tracks:
        last = track["history"][-1]
        lat = lon = None
        if radar_tracker.coordinate_mode == "geographic_local_enu" and radar_tracker.geo_origin:
            lat0, lon0 = radar_tracker.geo_origin
            lat = lat0 + np.degrees(last["estimated_y"] / 6371000.0)
            lon = lon0 + np.degrees(last["estimated_x"] / (6371000.0 * np.cos(np.radians(lat0))))
        raw = {
            "sensor_type": "radar", "observation_timestamp": last["timestamp"], "is_utc": True,
            "source_dataset": source_dataset, "source_record_id": f"{Path(file_path).name}#{track['track_id']}",
            "radar_track_id": track["track_id"], "position_x": last["estimated_x"], "position_y": last["estimated_y"],
            "velocity_x": last["velocity_x"], "velocity_y": last["velocity_y"], "uncertainty": last["uncertainty"],
            "latitude": lat, "longitude": lon, "coordinate_crs": "WGS84" if lat is not None else None,
            "model_name": "kalman_cv_gnn", "data_provenance": "real_inference",
            "metadata_json": {"track_status": track["status"], "num_observations": track["num_observations"],
                              "coordinate_mode": radar_tracker.coordinate_mode, "units": radar_tracker.units,
                              "position_is": "Kalman estimate (not raw measurement)"},
        }
        v = validate_observation(raw)
        if v["observation"] is not None:
            o = dbm.Observation(**v["observation"].model_dump())
            db.add(o); db.flush()
            obs_ids.append(o.id)
    db.commit()
    evaluation = radar_tracker.evaluate()
    _log(db, "INFO", "radar_tracking", f"Radar: {len(detections)} detections -> {len(tracks)} tracks",
         {"latency_ms": result["processing_time_ms"]})
    rep = save_report(db, report_type="radar_tracking", test_mode="real_evaluation"
                      if evaluation["metrics_available"] else "recorded_data",
                      source=f"{source_dataset}:{file.filename}", model_checkpoint="kalman_cv_gnn (classical)",
                      num_samples=len(detections), metrics=evaluation, warnings=radar_tracker.warnings,
                      limitations=["Classical tracker; maritime radar/AIS data does not demonstrate aircraft tracking.",
                                   *evaluation.get("metrics_unavailable", [])])
    result.update({"tracks": tracks, "observation_ids": obs_ids, "evaluation": evaluation,
                   "warnings": radar_tracker.warnings, "report_id": rep.id,
                   "tracker_status": radar_tracker.get_status()})
    return result


# ═══════════════════════════════ OBSERVATIONS ═══════════════════════════════

def _obs_out(o: dbm.Observation) -> dict:
    return {
        "id": o.id, "sensor_type": o.sensor_type, "observation_timestamp": _iso(o.observation_timestamp),
        "is_utc": o.is_utc, "source_dataset": o.source_dataset, "source_record_id": o.source_record_id,
        "object_class": o.object_class, "confidence": o.confidence,
        "bbox_x1": o.bbox_x1, "bbox_y1": o.bbox_y1, "bbox_x2": o.bbox_x2, "bbox_y2": o.bbox_y2,
        "latitude": o.latitude, "longitude": o.longitude, "coordinate_crs": o.coordinate_crs,
        "radar_track_id": o.radar_track_id, "position_x": o.position_x, "position_y": o.position_y,
        "velocity_x": o.velocity_x, "velocity_y": o.velocity_y, "uncertainty": o.uncertainty,
        "model_name": o.model_name, "model_version": o.model_version, "checkpoint_id": o.checkpoint_id,
        "data_provenance": o.data_provenance, "processing_status": o.processing_status,
        "image_path": o.image_path, "annotated_image_path": o.annotated_image_path,
        "image_url": f"/api/files/uploads/{Path(o.image_path).name}" if o.image_path else None,
        "metadata_json": o.metadata_json, "created_at": _iso(o.created_at),
    }


@router.get("/observations")
def list_observations(sensor_type: Optional[str] = Query(None), provenance: Optional[str] = Query(None),
                      limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                      db: Session = Depends(get_db)):
    query = db.query(dbm.Observation)
    if sensor_type:
        query = query.filter(dbm.Observation.sensor_type == sensor_type)
    if provenance:
        query = query.filter(dbm.Observation.data_provenance == provenance)
    total = query.count()
    rows = query.order_by(dbm.Observation.created_at.desc()).offset(offset).limit(limit).all()
    return {"total": total, "limit": limit, "offset": offset, "observations": [_obs_out(o) for o in rows]}


@router.get("/observations/{observation_id}")
def get_observation(observation_id: str, db: Session = Depends(get_db)):
    o = db.get(dbm.Observation, observation_id)
    if not o:
        raise HTTPException(404, "Observation not found")
    return _obs_out(o)


@router.post("/observations")
def create_observation(obs_data: ObservationCreate, db: Session = Depends(get_db)):
    """Validate and store an observation in the common schema."""
    v = validate_observation(obs_data.model_dump())
    if v["status"] == "FAIL":
        raise HTTPException(422, {"status": "FAIL", "errors": v["errors"]})
    obs = dbm.Observation(**v["observation"].model_dump())
    db.add(obs)
    db.commit()
    return {"id": obs.id, "status": "created", "validation": v["status"], "warnings": v["warnings"]}


@router.post("/observations/validate")
def validate_observation_endpoint(raw: dict = Body(...)):
    """Dry-run validation (nothing is stored)."""
    v = validate_observation(raw)
    return {"status": v["status"], "errors": v["errors"], "warnings": v["warnings"]}


# ═══════════════════════════════ CORRELATION ═══════════════════════════════

def _select_observations(db, observation_ids: list[str], limit: int = 50):
    if observation_ids:
        return db.query(dbm.Observation).filter(dbm.Observation.id.in_(observation_ids)).all()
    return db.query(dbm.Observation).order_by(dbm.Observation.created_at.desc()).limit(limit).all()


@router.post("/correlation/run")
def run_correlation(observation_ids: list[str] = Body(default=[]),
                    temporal_window: Optional[float] = Query(None, gt=0),
                    spatial_threshold: Optional[float] = Query(None, gt=0),
                    db: Session = Depends(get_db)):
    """Pairwise rule-based correlation (no events created). Decisions are persisted."""
    from app.services.correlation import correlate_observation_group, check_duplicates, group_correlated
    from app.services.fusion_pipeline import save_decisions
    observations = _select_observations(db, observation_ids, limit=20)
    if len(observations) < 2:
        return {"status": "insufficient_data", "message": "Need at least 2 observations for correlation."}
    obs_dicts = [observation_to_dict(o) for o in observations]
    config = {}
    if temporal_window:
        config["temporal_window_seconds"] = temporal_window
    if spatial_threshold:
        config["spatial_distance_threshold_m"] = spatial_threshold
    results = correlate_observation_group(obs_dicts, config)
    duplicates = check_duplicates(obs_dicts)
    groups = group_correlated(obs_dicts, results, duplicates)
    save_decisions(db, results)
    db.commit()
    summary = {s: sum(1 for r in results if r["decision_status"] == s)
               for s in ("CORRELATED", "NOT_CORRELATED", "INSUFFICIENT_METADATA")}
    return {"status": "completed", "correlations": results, "duplicates": duplicates,
            "groups": [{"observation_ids": g["observation_ids"], "duplicate_ids": g["duplicate_ids"]} for g in groups],
            "summary": summary, "total_pairs_checked": len(results)}


@router.get("/correlation/history")
def correlation_history(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                        db: Session = Depends(get_db)):
    q = db.query(dbm.CorrelationDecision)
    decisions = q.order_by(dbm.CorrelationDecision.created_at.desc()).offset(offset).limit(limit).all()
    return {"total": q.count(), "decisions": [{
        "id": d.id, "observation_ids": d.observation_ids, "decision_status": d.decision_status,
        "evidence": d.evidence, "reason": d.reason, "missing_metadata": d.missing_metadata,
        "temporal_delta_s": d.temporal_delta_s, "spatial_distance_m": d.spatial_distance_m,
        "config_params": d.config_params, "event_id": d.event_id, "created_at": _iso(d.created_at),
    } for d in decisions]}


# ═══════════════════════════════ EVENTS ═══════════════════════════════

def _event_out(e: dbm.Event) -> dict:
    return {
        "id": e.id, "review_priority": e.review_priority, "source_sensors": e.source_sensors,
        "observation_ids": e.observation_ids, "evidence_summary": e.evidence_summary,
        "uncertainty": e.uncertainty, "missing_information": e.missing_information,
        "priority_explanation": e.priority_explanation, "review_status": e.review_status,
        "reviewer_notes": e.reviewer_notes, "data_provenance": e.data_provenance,
        "created_at": _iso(e.created_at), "updated_at": _iso(e.updated_at),
    }


@router.get("/events")
def list_events(priority: Optional[str] = Query(None), status: Optional[str] = Query(None),
                provenance: Optional[str] = Query(None),
                limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    query = db.query(dbm.Event)
    if priority:
        query = query.filter(dbm.Event.review_priority == priority.upper())
    if status:
        query = query.filter(dbm.Event.review_status == status.upper())
    if provenance:
        query = query.filter(dbm.Event.data_provenance == provenance)
    total = query.count()
    events = query.order_by(dbm.Event.created_at.desc()).offset(offset).limit(limit).all()
    return {"total": total, "limit": limit, "offset": offset, "events": [_event_out(e) for e in events]}


@router.post("/events")
def create_event(event_data: EventCreate, db: Session = Depends(get_db)):
    event = dbm.Event(**event_data.model_dump())
    db.add(event)
    db.commit()
    return {"id": event.id, "status": "created", "review_priority": event.review_priority}


@router.post("/events/auto-generate")
def auto_generate_events(observation_ids: list[str] = Body(default=[]),
                         include_singletons: bool = Query(False),
                         temporal_window: Optional[float] = Query(None, gt=0),
                         spatial_threshold: Optional[float] = Query(None, gt=0),
                         db: Session = Depends(get_db)):
    """
    Correlate observations, group CORRELATED pairs into connected groups and create one event per group.
    Observations already linked to an event are skipped when no explicit ids are given.
    """
    from app.services.fusion_pipeline import run_fusion
    if observation_ids:
        observations = _select_observations(db, observation_ids)
    else:
        linked = set()
        for (ids,) in db.query(dbm.Event.observation_ids).all():
            linked.update(ids or [])
        observations = [o for o in db.query(dbm.Observation).order_by(dbm.Observation.created_at.desc())
                        .limit(200).all() if o.id not in linked][:50]
    if not observations:
        return {"status": "no_observations", "events_created": 0, "events": []}
    config = {"reference_time": datetime.utcnow()}
    if temporal_window:
        config["temporal_window_seconds"] = temporal_window
    if spatial_threshold:
        config["spatial_distance_threshold_m"] = spatial_threshold
    res = run_fusion(db, observations, config, include_singletons=include_singletons)
    events = res["events"]
    out = {"status": "created" if events else "no_correlated_groups", "events_created": len(events),
           "observations_considered": len(observations),
           "events": [{"event_id": e["id"], "review_priority": e["review_priority"],
                       "observation_ids": e["observation_ids"], "evidence_summary": e["evidence_summary"],
                       "priority_explanation": e["priority_explanation"]} for e in events],
           "decisions_summary": {s: sum(1 for d in res["decisions"] if d["decision_status"] == s)
                                 for s in ("CORRELATED", "NOT_CORRELATED", "INSUFFICIENT_METADATA")}}
    if events:  # backwards-compatible single-event fields
        out.update({"event_id": events[0]["id"], "review_priority": events[0]["review_priority"],
                    "evidence_summary": events[0]["evidence_summary"],
                    "priority_explanation": events[0]["priority_explanation"]})
    return out


@router.get("/events/{event_id}")
def get_event(event_id: str, db: Session = Depends(get_db)):
    """Event with full traceability: linked observations, correlation decisions and review history."""
    event = db.get(dbm.Event, event_id)
    if not event:
        raise HTTPException(404, "Event not found")
    out = _event_out(event)
    ids = event.observation_ids or []
    out["observations"] = [_obs_out(o) for o in db.query(dbm.Observation).filter(dbm.Observation.id.in_(ids)).all()]
    out["correlations"] = [{
        "id": d.id, "observation_ids": d.observation_ids, "decision_status": d.decision_status,
        "reason": d.reason, "evidence": d.evidence, "missing_metadata": d.missing_metadata,
        "config_params": d.config_params} for d in
        db.query(dbm.CorrelationDecision).filter(dbm.CorrelationDecision.event_id == event_id).all()]
    out["review_history"] = [{"previous_status": h.previous_status, "new_status": h.new_status, "notes": h.notes,
                              "timestamp": _iso(h.timestamp)} for h in
                             db.query(dbm.EventReviewHistory).filter(dbm.EventReviewHistory.event_id == event_id)
                             .order_by(dbm.EventReviewHistory.timestamp).all()]
    return out


@router.patch("/events/{event_id}/review")
def update_event_review(event_id: str, review: EventReviewUpdate, db: Session = Depends(get_db)):
    event = db.get(dbm.Event, event_id)
    if not event:
        raise HTTPException(404, "Event not found")
    db.add(dbm.EventReviewHistory(event_id=event_id, previous_status=event.review_status,
                                  new_status=review.review_status, notes=review.notes))
    event.review_status = review.review_status
    if review.notes:
        event.reviewer_notes = review.notes
    event.updated_at = datetime.utcnow()
    db.commit()
    return {"status": "updated", "review_status": event.review_status}


@router.get("/events/{event_id}/export")
def export_event(event_id: str, db: Session = Depends(get_db)):
    """Event evidence package (JSON) for offline review."""
    data = get_event(event_id, db)
    data["export_note"] = "Review-priority recommendation only. Not an assessment of intent; not an operational decision."
    data["exported_at"] = datetime.utcnow().isoformat()
    return data


# ═══════════════════════════════ TRAINING ═══════════════════════════════

def _job_out(j: dbm.TrainingJob, detail: bool = False) -> dict:
    d = {"id": j.id, "model_id": j.model_id, "dataset_name": j.dataset_name, "status": j.status,
         "epochs": j.epochs, "batch_size": j.batch_size, "image_size": j.image_size,
         "learning_rate": j.learning_rate, "device": j.device, "current_epoch": j.current_epoch,
         "total_epochs": j.total_epochs, "train_loss": j.train_loss, "val_loss": j.val_loss,
         "best_map50": j.best_map50, "checkpoint_path": j.checkpoint_path, "error_message": j.error_message,
         "cancel_requested": bool(j.cancel_requested), "started_at": _iso(j.started_at),
         "completed_at": _iso(j.completed_at), "created_at": _iso(j.created_at),
         "epoch_history": j.epoch_history or []}
    if detail:
        d["config"] = j.config
        log = ""
        if j.log_path and Path(j.log_path).exists():
            log = "".join(Path(j.log_path).read_text(encoding="utf-8").splitlines(True)[-200:])
        d["log_tail"] = log
    return d


@router.get("/training/preflight")
def training_preflight(model_id: str, dataset_name: str, device: str = "cpu", db: Session = Depends(get_db)):
    from app.services.training import preflight
    ds = db.query(dbm.Dataset).filter(dbm.Dataset.name == dataset_name).first()
    problems = preflight(ds, device)
    import os
    return {"ok": not problems, "problems": problems, "cpu_count": os.cpu_count(),
            "device_available": _dependency_status(),
            "running_jobs": db.query(dbm.TrainingJob).filter(dbm.TrainingJob.status.in_(["PENDING", "RUNNING"])).count()}


@router.post("/training/start")
def start_training(job_data: TrainingJobCreate, db: Session = Depends(get_db)):
    """Start a fine-tuning job in a background worker (one at a time)."""
    from app.services.training import preflight, run_training_job, start_in_background
    model = db.get(dbm.ModelRegistry, job_data.model_id)
    if not model:
        raise HTTPException(404, "Model not found")
    if model.sensor_type not in ("satellite", "thermal"):
        raise HTTPException(400, "Only image detectors are trainable.")
    if db.query(dbm.TrainingJob).filter(dbm.TrainingJob.status.in_(["PENDING", "RUNNING"])).count():
        raise HTTPException(409, "A training job is already running. Wait for it to complete or cancel it.")
    ds = db.query(dbm.Dataset).filter(dbm.Dataset.name == job_data.dataset_name).first()
    if ds and ds.sensor_type != model.sensor_type:
        raise HTTPException(400, f"Dataset '{ds.name}' is {ds.sensor_type} data; model is {model.sensor_type}.")
    problems = preflight(ds, job_data.device)
    if problems:
        raise HTTPException(400, {"message": "Training prerequisites not met", "problems": problems})
    job = dbm.TrainingJob(model_id=model.id, dataset_name=job_data.dataset_name, epochs=job_data.epochs,
                          batch_size=job_data.batch_size, image_size=job_data.image_size,
                          learning_rate=job_data.learning_rate, device=job_data.device,
                          total_epochs=job_data.epochs, status="PENDING")
    db.add(job)
    db.commit()
    start_in_background(run_training_job, job.id)
    return {"job_id": job.id, "status": "PENDING", "message": "Training job started in background."}


@router.get("/training/jobs")
def list_training_jobs(db: Session = Depends(get_db)):
    jobs = db.query(dbm.TrainingJob).order_by(dbm.TrainingJob.created_at.desc()).limit(20).all()
    return [_job_out(j) for j in jobs]


@router.get("/training/jobs/{job_id}")
def get_training_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(dbm.TrainingJob, job_id)
    if not job:
        raise HTTPException(404, "Training job not found")
    return _job_out(job, detail=True)


@router.post("/training/cancel/{job_id}")
def cancel_training(job_id: str, db: Session = Depends(get_db)):
    """Cooperative cancellation: the worker stops after the current batch."""
    job = db.get(dbm.TrainingJob, job_id)
    if not job:
        raise HTTPException(404, "Training job not found")
    if job.status not in ("PENDING", "RUNNING"):
        return {"status": "cannot_cancel", "current_status": job.status}
    job.cancel_requested = True
    if job.status == "PENDING":
        job.status, job.error_message = "CANCELLED", "Cancelled by user before start."
    db.commit()
    return {"status": "cancellation_requested", "job_id": job_id}


# ═══════════════════════════════ EVALUATION ═══════════════════════════════

@router.post("/evaluation/start")
def start_evaluation(job_data: EvaluationJobCreate, max_images: Optional[int] = Query(None, ge=1),
                     db: Session = Depends(get_db)):
    """Evaluate a model checkpoint on a labelled split (metrics computed from predictions vs labels)."""
    from app.services.training import run_evaluation_job, start_in_background
    model = db.get(dbm.ModelRegistry, job_data.model_id)
    if not model:
        raise HTTPException(404, "Model not found")
    if job_data.dataset_split not in ("train", "val", "test"):
        raise HTTPException(400, "dataset_split must be train, val or test")
    ds = db.query(dbm.Dataset).filter(dbm.Dataset.name == job_data.dataset_name).first()
    if not ds or ds.validation_status != "VALIDATED":
        raise HTTPException(400, f"Dataset '{job_data.dataset_name}' is not installed and VALIDATED — "
                                 "no evaluation can be performed without reference labels.")
    job = dbm.EvaluationJob(model_id=model.id, dataset_name=job_data.dataset_name,
                            dataset_split=job_data.dataset_split, status="PENDING")
    db.add(job)
    db.commit()
    start_in_background(run_evaluation_job, job.id, max_images)
    return {"job_id": job.id, "status": "PENDING"}


@router.get("/evaluation/jobs")
def list_evaluation_jobs(db: Session = Depends(get_db)):
    jobs = db.query(dbm.EvaluationJob).order_by(dbm.EvaluationJob.created_at.desc()).limit(20).all()
    return [{
        "id": j.id, "model_id": j.model_id, "dataset_name": j.dataset_name, "dataset_split": j.dataset_split,
        "status": j.status, "precision": j.precision, "recall": j.recall, "map50": j.map50, "map50_95": j.map50_95,
        "per_class_metrics": j.per_class_metrics, "confusion_matrix": j.confusion_matrix,
        "avg_inference_ms": j.avg_inference_ms, "num_images": j.num_images, "checkpoint_path": j.checkpoint_path,
        "model_status_at_eval": j.model_status_at_eval, "warnings": j.warnings, "error_message": j.error_message,
        "report_path": j.report_path, "started_at": _iso(j.started_at), "completed_at": _iso(j.completed_at),
        "created_at": _iso(j.created_at),
    } for j in jobs]


# ═══════════════════════════════ REPLAY ═══════════════════════════════

def _replay_processor(record: dict, engine) -> dict:
    """Validate -> store -> correlate with recent replay observations -> create/update events."""
    from app.db.database import SessionLocal
    from app.services.correlation import correlate_observations, check_duplicates
    from app.services.fusion_pipeline import recompute_event, build_event, save_decisions
    from datetime import timedelta

    session = SessionLocal()
    try:
        v = validate_observation(record, defaults={"data_provenance": "recorded"})
        if v["observation"] is None:
            engine.stats["rejected"] += 1
            return {"status": "REJECTED", "errors": v["errors"]}
        data = v["observation"].model_dump()
        if data.get("data_provenance") not in ("recorded", "recorded_prediction"):
            data["data_provenance"] = "recorded"
        meta = dict(data.get("metadata_json") or {})
        meta.update({"replay_session": engine.session_id,
                     "note": "Replayed recorded observation — no new model inference executed."})
        data["metadata_json"] = meta
        obs = dbm.Observation(**data)
        session.add(obs)
        session.flush()
        engine.stats["saved"] += 1

        new = observation_to_dict(obs)
        window = timedelta(seconds=settings.TEMPORAL_WINDOW_SECONDS)
        recent = session.query(dbm.Observation).filter(
            dbm.Observation.id != obs.id,
            dbm.Observation.observation_timestamp >= obs.observation_timestamp - window,
            dbm.Observation.observation_timestamp <= obs.observation_timestamp,
        ).all()
        recent = [o for o in recent if (o.metadata_json or {}).get("replay_session") == engine.session_id]
        cfg = {"reference_time": obs.observation_timestamp}
        decisions = [correlate_observations(observation_to_dict(o), new) for o in recent]
        matched = [d for d in decisions if d["decision_status"] == "CORRELATED"]
        outcome = {"status": "STORED", "validation": v["status"], "observation_id": obs.id,
                   "correlated_with": [d["observation_ids"][0] for d in matched]}
        if matched:
            event_map = engine.__dict__.setdefault("_event_map", {})
            existing = next((event_map[d["observation_ids"][0]] for d in matched
                             if d["observation_ids"][0] in event_map), None)
            if existing:
                ev = session.get(dbm.Event, existing)
                ev.observation_ids = list(ev.observation_ids or []) + [obs.id]
                recompute_event(session, ev, cfg)
                for d in matched:
                    d["event_id"] = ev.id
                engine.stats["events_updated"] += 1
                action = "UPDATED"
            else:
                members = [observation_to_dict(session.get(dbm.Observation, d["observation_ids"][0])) for d in matched]
                dups = check_duplicates(members + [new])
                built = build_event(members + [new], matched, [], cfg)
                ev = dbm.Event(id=built["id"], review_priority=built["review_priority"],
                               source_sensors=built["source_sensors"],
                               observation_ids=[m["id"] for m in members] + [obs.id],
                               evidence_summary=built["evidence_summary"], uncertainty=built.get("uncertainty"),
                               missing_information=built.get("missing_information"),
                               priority_explanation=built["priority_explanation"], data_provenance="recorded")
                session.add(ev)
                for d in matched:
                    d["event_id"] = ev.id
                engine.stats["events_created"] += 1
                action = "CREATED"
                if dups:
                    outcome["duplicates"] = len(dups)
            for oid in ev.observation_ids:
                event_map[oid] = ev.id
            engine.event_timeline.append({
                "event_id": ev.id, "action": action, "review_priority": ev.review_priority,
                "recorded_timestamp": obs.observation_timestamp.isoformat(),
                "source_sensors": ev.source_sensors, "observation_count": len(ev.observation_ids)})
            outcome["event_id"], outcome["event_action"] = ev.id, action
        save_decisions(session, decisions)
        session.commit()
        return outcome
    finally:
        session.close()


def _replay_complete(engine):
    from app.db.database import SessionLocal
    from app.services.reporting import save_report
    s = SessionLocal()
    try:
        save_report(s, report_type="recorded_replay", test_mode="recorded_replay", source=engine.source_file,
                    num_samples=len(engine.processed_records),
                    metrics={**engine.stats, "skipped_records": len(engine.skipped_records),
                             "start_timestamp": str(engine.start_timestamp), "end_timestamp": str(engine.end_timestamp)},
                    warnings=engine.errors[-20:],
                    body={"session_id": engine.session_id, "event_timeline": engine.event_timeline,
                          "skipped": engine.skipped_records[:100]})
    finally:
        s.close()


@router.post("/replay/load")
def replay_load(file: UploadFile = File(...)):
    """Load a recorded observation file (JSON list / {"observations": [...]} or CSV)."""
    from app.services.replay import replay_engine
    import pandas as pd
    file_path = _save_upload(file, "replay_data", kind="data")
    ext = Path(file_path).suffix.lower()
    try:
        if ext == ".json":
            with open(file_path, encoding="utf-8") as f:
                records = json.load(f)
            if isinstance(records, dict) and "observations" in records:
                records = records["observations"]
            if not isinstance(records, list):
                raise HTTPException(400, "JSON must be a list of observations or {'observations': [...]}")
        elif ext == ".csv":
            records = pd.read_csv(file_path).to_dict(orient="records")
        else:
            raise HTTPException(400, f"Unsupported format: {ext}. Use JSON or CSV.")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid JSON: {e}")
    replay_engine.processor = _replay_processor
    replay_engine.on_complete = _replay_complete
    replay_engine.__dict__["_event_map"] = {}
    result = replay_engine.load_records(records, source_file=file.filename)
    if result.get("status") == "error":
        raise HTTPException(400, result)
    return result


@router.post("/replay/control")
def replay_control(control: ReplayControl):
    """start | pause | resume | reset | speed (speed may accompany any action)."""
    from app.services.replay import replay_engine
    if control.speed is not None:
        replay_engine.set_speed(control.speed)
    actions = {"start": replay_engine.start, "pause": replay_engine.pause,
               "resume": replay_engine.resume, "reset": replay_engine.reset}
    if control.action in actions:
        if control.action == "reset":
            replay_engine.__dict__["_event_map"] = {}
        result = actions[control.action]()
        result["replay_speed"] = replay_engine.replay_speed
        return result
    return replay_engine.get_status()


@router.get("/replay/status")
def replay_status():
    from app.services.replay import replay_engine
    return replay_engine.get_status()


@router.get("/replay/processed")
def get_replay_processed(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    from app.services.replay import replay_engine
    return {"status": replay_engine.status, "total_processed": len(replay_engine.processed_records),
            "records": replay_engine.get_processed(limit=limit, offset=offset),
            "label": "RECORDED DATA REPLAY — NOT A LIVE SENSOR FEED"}


# ═══════════════════════════════ SYNTHETIC ═══════════════════════════════

@router.post("/synthetic/run")
def run_synthetic_scenarios(db: Session = Depends(get_db)):
    """Run deterministic synthetic integration scenarios; each check reports expected vs actual."""
    from app.services.synthetic_scenarios import run_synthetic_suite
    return run_synthetic_suite(db)


@router.get("/synthetic/scenarios")
def list_synthetic_scenarios():
    from app.services.synthetic_scenarios import generate_synthetic_scenarios
    sc = generate_synthetic_scenarios("preview")
    return {"label": "SYNTHETIC DEMONSTRATION — NOT LIVE SENSOR DATA",
            "scenarios": {k: {"description": v["description"], "expected": v["expected"],
                              "observations": v["observations"]} for k, v in sc.items()}}


# ═══════════════════════════════ REPORTS ═══════════════════════════════

@router.get("/reports")
def list_reports(report_type: Optional[str] = Query(None), test_mode: Optional[str] = Query(None),
                 limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    from app.services.reporting import to_dict
    q = db.query(dbm.Report)
    if report_type:
        q = q.filter(dbm.Report.report_type == report_type)
    if test_mode:
        q = q.filter(dbm.Report.test_mode == test_mode)
    return [to_dict(r) for r in q.order_by(dbm.Report.created_at.desc()).offset(offset).limit(limit).all()]


@router.get("/reports/export/{report_id}")
def export_report(report_id: str, db: Session = Depends(get_db)):
    from app.services.reporting import to_dict
    rep = db.get(dbm.Report, report_id)
    if not rep:
        raise HTTPException(404, "Report not found")
    return to_dict(rep, include_body=True)


@router.get("/reports/download/{report_id}")
def download_report(report_id: str, db: Session = Depends(get_db)):
    rep = db.get(dbm.Report, report_id)
    if not rep or not rep.file_path or not Path(rep.file_path).exists():
        raise HTTPException(404, "Report file not found")
    return FileResponse(rep.file_path, media_type="application/json", filename=Path(rep.file_path).name)


@router.post("/reports/fusion-tests")
def fusion_test_report(db: Session = Depends(get_db)):
    """Summarise stored correlation decisions and events as a fusion software report."""
    from app.services.reporting import save_report, to_dict
    dec = db.query(dbm.CorrelationDecision).all()
    evs = db.query(dbm.Event).all()
    metrics = {
        "decisions": len(dec),
        **{s: sum(1 for d in dec if d.decision_status == s) for s in ("CORRELATED", "NOT_CORRELATED", "INSUFFICIENT_METADATA")},
        "events": len(evs),
        "events_by_provenance": {p: sum(1 for e in evs if e.data_provenance == p) for p in ("real", "recorded", "synthetic")},
    }
    rep = save_report(db, report_type="fusion_tests", test_mode="software", source="stored decisions",
                      num_samples=len(dec), metrics=metrics,
                      limitations=["Counts describe software behaviour; they are not fusion accuracy."])
    return to_dict(rep)


# ═══════════════════════════════ FILES ═══════════════════════════════

@router.get("/files/{file_type}/{filename}")
def serve_file(file_type: str, filename: str):
    """Serve uploaded/annotated images only from whitelisted folders (no path traversal)."""
    allowed_dirs = {"uploads": "uploads", "radar_data": "radar_data", "replay_data": "replay_data"}
    if file_type not in allowed_dirs:
        raise HTTPException(400, "Invalid file type")
    if filename != Path(filename).name or filename.startswith("."):
        raise HTTPException(400, "Invalid filename")
    base = (DATA_DIR / allowed_dirs[file_type]).resolve()
    file_path = (base / filename).resolve()
    if base not in file_path.parents or not file_path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(str(file_path))


# ═══════════════════════════════ RADAR TRACKS ═══════════════════════════════

@router.get("/radar/tracks")
def get_radar_tracks():
    from app.services.radar_tracking import radar_tracker
    return {"tracks": radar_tracker.get_tracks(), "status": radar_tracker.get_status(),
            "evaluation": radar_tracker.evaluate()}


@router.get("/radar/tracks/active")
def get_active_radar_tracks():
    from app.services.radar_tracking import radar_tracker
    tracks = radar_tracker.get_active_tracks()
    return {"tracks": tracks, "active_count": len(tracks)}


@router.delete("/radar/tracks/reset")
def reset_radar_tracks():
    from app.services.radar_tracking import radar_tracker
    radar_tracker.reset()
    return {"status": "reset", "message": "All radar tracks cleared."}
