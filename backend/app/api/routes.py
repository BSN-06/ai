"""
API routes for the surveillance system.
All routes organized by domain: system, datasets, models, inference, training,
evaluation, observations, correlation, events, replay, synthetic, and reports.
"""
import os
import uuid
import json
import shutil
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db import models as dbm
from app.schemas.schemas import (
    SystemHealth, ObservationCreate, ObservationResponse,
    EventCreate, EventResponse, EventReviewUpdate,
    CorrelationResult, DatasetResponse, ModelResponse,
    TrainingJobCreate, TrainingJobResponse,
    EvaluationJobCreate, EvaluationJobResponse,
    ReplaySessionResponse, ReplayControl, ReportResponse,
    InferenceRequest,
)
from app.core.config import settings, BASE_DIR

logger = logging.getLogger(__name__)

router = APIRouter()

# ═══════════════════════════════ SYSTEM ═══════════════════════════════

@router.get("/health", response_model=SystemHealth)
def system_health(db: Session = Depends(get_db)):
    """System health check with component status."""
    gpu_available = False
    device = "cpu"
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        device = "cuda" if gpu_available else "cpu"
    except ImportError:
        pass

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
        gpu_available=gpu_available,
        device=device,
        timestamp=datetime.utcnow(),
    )


@router.get("/system/stats")
def system_stats(db: Session = Depends(get_db)):
    """Detailed system statistics for the dashboard."""
    obs_by_sensor = {}
    for sensor in ["satellite", "thermal", "radar"]:
        obs_by_sensor[sensor] = db.query(dbm.Observation).filter(
            dbm.Observation.sensor_type == sensor
        ).count()

    events_by_priority = {}
    for priority in ["LOW", "MEDIUM", "HIGH"]:
        events_by_priority[priority] = db.query(dbm.Event).filter(
            dbm.Event.review_priority == priority
        ).count()

    recent_observations = db.query(dbm.Observation).order_by(
        dbm.Observation.created_at.desc()
    ).limit(10).all()

    recent_events = db.query(dbm.Event).order_by(
        dbm.Event.created_at.desc()
    ).limit(10).all()

    training_jobs = db.query(dbm.TrainingJob).order_by(
        dbm.TrainingJob.created_at.desc()
    ).limit(5).all()

    errors = db.query(dbm.ProcessingLog).filter(
        dbm.ProcessingLog.level == "ERROR"
    ).order_by(dbm.ProcessingLog.timestamp.desc()).limit(10).all()

    return {
        "observations": {
            "total": db.query(dbm.Observation).count(),
            "by_sensor": obs_by_sensor,
        },
        "events": {
            "total": db.query(dbm.Event).count(),
            "pending": db.query(dbm.Event).filter(dbm.Event.review_status == "PENDING").count(),
            "by_priority": events_by_priority,
        },
        "models": {
            "total": db.query(dbm.ModelRegistry).count(),
            "trained": db.query(dbm.ModelRegistry).filter(dbm.ModelRegistry.trained == True).count(),
        },
        "datasets": {
            "total": db.query(dbm.Dataset).count(),
        },
        "training_jobs": [
            {
                "id": j.id, "status": j.status, "model_id": j.model_id,
                "current_epoch": j.current_epoch, "total_epochs": j.total_epochs,
                "created_at": j.created_at.isoformat() if j.created_at else None,
            }
            for j in training_jobs
        ],
        "recent_observations": [
            {
                "id": o.id, "sensor_type": o.sensor_type, "object_class": o.object_class,
                "confidence": o.confidence, "data_provenance": o.data_provenance,
                "created_at": o.created_at.isoformat() if o.created_at else None,
            }
            for o in recent_observations
        ],
        "recent_events": [
            {
                "id": e.id, "review_priority": e.review_priority,
                "review_status": e.review_status, "source_sensors": e.source_sensors,
                "evidence_summary": e.evidence_summary,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in recent_events
        ],
        "errors": [
            {"level": e.level, "component": e.component, "message": e.message,
             "timestamp": e.timestamp.isoformat() if e.timestamp else None}
            for e in errors
        ],
    }


# ═══════════════════════════════ DATASETS ═══════════════════════════════

@router.get("/datasets", response_model=list[DatasetResponse])
def list_datasets(db: Session = Depends(get_db)):
    return db.query(dbm.Dataset).all()


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.query(dbm.Dataset).filter(dbm.Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    return ds


@router.post("/datasets/validate/{dataset_id}")
def validate_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Validate a dataset's local files and annotations."""
    ds = db.query(dbm.Dataset).filter(dbm.Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")

    if not ds.local_path or not Path(ds.local_path).exists():
        ds.validation_status = "NOT_FOUND"
        ds.validation_message = f"Local path does not exist: {ds.local_path}"
        db.commit()
        return {"status": "NOT_FOUND", "message": ds.validation_message}

    # Count files
    local_path = Path(ds.local_path)
    image_count = len(list(local_path.rglob("*.jpg")) + list(local_path.rglob("*.png")) + list(local_path.rglob("*.tif")))

    ds.file_count = image_count
    ds.validation_status = "VALIDATED" if image_count > 0 else "EMPTY"
    ds.validation_message = f"Found {image_count} image files."
    db.commit()

    return {"status": ds.validation_status, "message": ds.validation_message, "file_count": image_count}


# ═══════════════════════════════ MODELS ═══════════════════════════════

@router.get("/models", response_model=list[ModelResponse])
def list_models(db: Session = Depends(get_db)):
    return db.query(dbm.ModelRegistry).all()


@router.get("/models/{model_id}", response_model=ModelResponse)
def get_model(model_id: str, db: Session = Depends(get_db)):
    model = db.query(dbm.ModelRegistry).filter(dbm.ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(404, "Model not found")
    return model


# ═══════════════════════════════ INFERENCE ═══════════════════════════════

def _save_upload(file: UploadFile, subdir: str = "uploads") -> str:
    """Save uploaded file safely and return the path."""
    upload_dir = BASE_DIR / "data" / subdir
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Validate extension
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    allowed = settings.ALLOWED_IMAGE_EXTENSIONS + settings.ALLOWED_DATA_EXTENSIONS
    if ext not in allowed:
        raise HTTPException(400, f"File type {ext} not allowed. Allowed: {allowed}")

    # Safe filename
    safe_name = f"{uuid.uuid4()}{ext}"
    file_path = upload_dir / safe_name

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Check size
    if file_path.stat().st_size > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        file_path.unlink()
        raise HTTPException(413, f"File exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit")

    return str(file_path)


@router.post("/inference/satellite")
def satellite_inference(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.45, ge=0.0, le=1.0),
    model_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Run satellite image object detection."""
    from app.services.satellite_detector import satellite_detector

    file_path = _save_upload(file)

    # Load model if needed
    if not satellite_detector.is_loaded:
        checkpoint = None
        if model_id:
            model_rec = db.query(dbm.ModelRegistry).filter(dbm.ModelRegistry.id == model_id).first()
            if model_rec and model_rec.checkpoint_path:
                checkpoint = model_rec.checkpoint_path
        load_result = satellite_detector.load_model(checkpoint_path=checkpoint)
        if load_result["status"] == "error":
            raise HTTPException(500, load_result["message"])

    result = satellite_detector.run_inference(file_path, confidence, iou)

    if result["status"] == "error":
        raise HTTPException(500, result["message"])

    # Save observations to DB
    for det in result.get("detections", []):
        obs = dbm.Observation(
            sensor_type="satellite",
            observation_timestamp=datetime.utcnow(),
            object_class=det["class_name"],
            confidence=det["confidence"],
            bbox_x1=det["bbox_x1"], bbox_y1=det["bbox_y1"],
            bbox_x2=det["bbox_x2"], bbox_y2=det["bbox_y2"],
            model_name=result.get("model_name"),
            checkpoint_id=result.get("checkpoint"),
            data_provenance="real_inference",
            image_path=file_path,
            annotated_image_path=result.get("annotated_image_path"),
        )
        db.add(obs)
    db.commit()

    return result


@router.post("/inference/thermal")
def thermal_inference(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.45, ge=0.0, le=1.0),
    model_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Run thermal image object detection."""
    from app.services.thermal_detector import thermal_detector

    file_path = _save_upload(file)

    if not thermal_detector.is_loaded:
        checkpoint = None
        if model_id:
            model_rec = db.query(dbm.ModelRegistry).filter(dbm.ModelRegistry.id == model_id).first()
            if model_rec and model_rec.checkpoint_path:
                checkpoint = model_rec.checkpoint_path
        load_result = thermal_detector.load_model(checkpoint_path=checkpoint)
        if load_result["status"] == "error":
            raise HTTPException(500, load_result["message"])

    result = thermal_detector.run_inference(file_path, confidence, iou)

    if result["status"] == "error":
        raise HTTPException(500, result["message"])

    for det in result.get("detections", []):
        obs = dbm.Observation(
            sensor_type="thermal",
            observation_timestamp=datetime.utcnow(),
            object_class=det["class_name"],
            confidence=det["confidence"],
            bbox_x1=det["bbox_x1"], bbox_y1=det["bbox_y1"],
            bbox_x2=det["bbox_x2"], bbox_y2=det["bbox_y2"],
            model_name=result.get("model_name"),
            checkpoint_id=result.get("checkpoint"),
            data_provenance="real_inference",
            image_path=file_path,
            annotated_image_path=result.get("annotated_image_path"),
        )
        db.add(obs)
    db.commit()

    return result


@router.post("/inference/radar")
def radar_import(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Import and process radar detection data."""
    from app.services.radar_tracking import radar_tracker

    file_path = _save_upload(file, "radar_data")
    ext = Path(file_path).suffix.lower()

    try:
        if ext == ".csv":
            detections = radar_tracker.parse_csv_data(file_path)
        elif ext == ".json":
            detections = radar_tracker.parse_json_data(file_path)
        else:
            raise HTTPException(400, f"Unsupported radar data format: {ext}. Use CSV or JSON.")
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))

    result = radar_tracker.process_detections(detections)

    # Save radar observations to DB
    for track in radar_tracker.get_tracks():
        if track["history"]:
            last = track["history"][-1]
            obs = dbm.Observation(
                sensor_type="radar",
                observation_timestamp=datetime.fromisoformat(last["timestamp"]) if last.get("timestamp") else None,
                radar_track_id=track["track_id"],
                position_x=last.get("estimated_x"),
                position_y=last.get("estimated_y"),
                velocity_x=last.get("velocity_x"),
                velocity_y=last.get("velocity_y"),
                uncertainty=last.get("uncertainty"),
                data_provenance="real_inference",
            )
            db.add(obs)
    db.commit()

    result["tracks"] = radar_tracker.get_tracks()
    return result


# ═══════════════════════════════ OBSERVATIONS ═══════════════════════════════

@router.get("/observations")
def list_observations(
    sensor_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    query = db.query(dbm.Observation)
    if sensor_type:
        query = query.filter(dbm.Observation.sensor_type == sensor_type)
    total = query.count()
    observations = query.order_by(dbm.Observation.created_at.desc()).offset(offset).limit(limit).all()
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "observations": [
            {
                "id": o.id, "sensor_type": o.sensor_type,
                "observation_timestamp": o.observation_timestamp.isoformat() if o.observation_timestamp else None,
                "object_class": o.object_class, "confidence": o.confidence,
                "bbox_x1": o.bbox_x1, "bbox_y1": o.bbox_y1,
                "bbox_x2": o.bbox_x2, "bbox_y2": o.bbox_y2,
                "latitude": o.latitude, "longitude": o.longitude,
                "radar_track_id": o.radar_track_id,
                "position_x": o.position_x, "position_y": o.position_y,
                "velocity_x": o.velocity_x, "velocity_y": o.velocity_y,
                "uncertainty": o.uncertainty,
                "model_name": o.model_name, "data_provenance": o.data_provenance,
                "processing_status": o.processing_status,
                "created_at": o.created_at.isoformat() if o.created_at else None,
            }
            for o in observations
        ],
    }


@router.post("/observations")
def create_observation(obs_data: ObservationCreate, db: Session = Depends(get_db)):
    """Create a new observation record."""
    obs = dbm.Observation(**obs_data.model_dump())
    db.add(obs)
    db.commit()
    db.refresh(obs)
    return {"id": obs.id, "status": "created"}


# ═══════════════════════════════ CORRELATION ═══════════════════════════════

@router.post("/correlation/run")
def run_correlation(
    observation_ids: list[str] = [],
    temporal_window: Optional[float] = Query(None),
    spatial_threshold: Optional[float] = Query(None),
    db: Session = Depends(get_db),
):
    """Run correlation analysis on selected observations."""
    from app.services.correlation import correlate_observation_group, check_duplicates

    if not observation_ids:
        # Get recent observations if none specified
        observations = db.query(dbm.Observation).order_by(
            dbm.Observation.created_at.desc()
        ).limit(20).all()
    else:
        observations = db.query(dbm.Observation).filter(
            dbm.Observation.id.in_(observation_ids)
        ).all()

    if len(observations) < 2:
        return {"status": "insufficient_data", "message": "Need at least 2 observations for correlation."}

    obs_dicts = []
    for o in observations:
        obs_dicts.append({
            "id": o.id,
            "sensor_type": o.sensor_type,
            "observation_timestamp": o.observation_timestamp,
            "latitude": o.latitude,
            "longitude": o.longitude,
            "coordinate_crs": o.coordinate_crs,
            "object_class": o.object_class,
            "confidence": o.confidence,
            "bbox_x1": o.bbox_x1,
            "uncertainty": o.uncertainty,
            "data_provenance": o.data_provenance,
        })

    config = {}
    if temporal_window:
        config["temporal_window_seconds"] = temporal_window
    if spatial_threshold:
        config["spatial_distance_threshold_m"] = spatial_threshold

    correlation_results = correlate_observation_group(obs_dicts, config)
    duplicates = check_duplicates(obs_dicts)

    # Save correlation decisions to DB
    for cr in correlation_results:
        decision = dbm.CorrelationDecision(
            observation_ids=cr["observation_ids"],
            decision_status=cr["decision_status"],
            evidence=cr.get("evidence"),
            reason=cr.get("reason"),
            missing_metadata=cr.get("missing_metadata"),
            temporal_delta_s=cr.get("temporal_delta_s"),
            spatial_distance_m=cr.get("spatial_distance_m"),
            config_params=cr.get("config_params"),
        )
        db.add(decision)

    db.commit()

    return {
        "status": "completed",
        "correlations": correlation_results,
        "duplicates": duplicates,
        "total_pairs_checked": len(correlation_results),
    }


# ═══════════════════════════════ EVENTS ═══════════════════════════════

@router.get("/events")
def list_events(
    priority: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    query = db.query(dbm.Event)
    if priority:
        query = query.filter(dbm.Event.review_priority == priority.upper())
    if status:
        query = query.filter(dbm.Event.review_status == status.upper())
    total = query.count()
    events = query.order_by(dbm.Event.created_at.desc()).offset(offset).limit(limit).all()
    return {
        "total": total,
        "events": [
            {
                "id": e.id, "review_priority": e.review_priority,
                "source_sensors": e.source_sensors,
                "observation_ids": e.observation_ids,
                "evidence_summary": e.evidence_summary,
                "uncertainty": e.uncertainty,
                "missing_information": e.missing_information,
                "priority_explanation": e.priority_explanation,
                "review_status": e.review_status,
                "reviewer_notes": e.reviewer_notes,
                "data_provenance": e.data_provenance,
                "created_at": e.created_at.isoformat() if e.created_at else None,
                "updated_at": e.updated_at.isoformat() if e.updated_at else None,
            }
            for e in events
        ],
    }


@router.post("/events")
def create_event(event_data: EventCreate, db: Session = Depends(get_db)):
    """Create a new event from correlated observations."""
    event = dbm.Event(**event_data.model_dump())
    db.add(event)
    db.commit()
    db.refresh(event)
    return {"id": event.id, "status": "created", "review_priority": event.review_priority}


@router.post("/events/auto-generate")
def auto_generate_events(
    observation_ids: list[str] = [],
    db: Session = Depends(get_db),
):
    """Automatically correlate observations and generate prioritized events."""
    from app.services.correlation import correlate_observation_group
    from app.services.prioritization import calculate_priority

    if not observation_ids:
        observations = db.query(dbm.Observation).order_by(
            dbm.Observation.created_at.desc()
        ).limit(20).all()
    else:
        observations = db.query(dbm.Observation).filter(
            dbm.Observation.id.in_(observation_ids)
        ).all()

    if not observations:
        return {"status": "no_observations", "events_created": 0}

    obs_dicts = [{
        "id": o.id, "sensor_type": o.sensor_type,
        "observation_timestamp": o.observation_timestamp,
        "latitude": o.latitude, "longitude": o.longitude,
        "coordinate_crs": o.coordinate_crs,
        "object_class": o.object_class, "confidence": o.confidence,
        "bbox_x1": o.bbox_x1, "uncertainty": o.uncertainty,
        "data_provenance": o.data_provenance,
    } for o in observations]

    correlations = correlate_observation_group(obs_dicts)
    event_result = calculate_priority(obs_dicts, correlations)

    event = dbm.Event(
        review_priority=event_result["review_priority"],
        source_sensors=event_result["source_sensors"],
        observation_ids=event_result["observation_ids"],
        evidence_summary=event_result["evidence_summary"],
        uncertainty=event_result.get("uncertainty"),
        missing_information=event_result.get("missing_information"),
        priority_explanation=event_result["priority_explanation"],
        data_provenance=event_result.get("data_provenance", "real"),
    )
    db.add(event)
    db.commit()

    return {
        "status": "created",
        "event_id": event.id,
        "review_priority": event.review_priority,
        "evidence_summary": event.evidence_summary,
        "priority_explanation": event.priority_explanation,
    }


@router.get("/events/{event_id}")
def get_event(event_id: str, db: Session = Depends(get_db)):
    event = db.query(dbm.Event).filter(dbm.Event.id == event_id).first()
    if not event:
        raise HTTPException(404, "Event not found")
    return {
        "id": event.id,
        "review_priority": event.review_priority,
        "source_sensors": event.source_sensors,
        "observation_ids": event.observation_ids,
        "evidence_summary": event.evidence_summary,
        "uncertainty": event.uncertainty,
        "missing_information": event.missing_information,
        "priority_explanation": event.priority_explanation,
        "review_status": event.review_status,
        "reviewer_notes": event.reviewer_notes,
        "data_provenance": event.data_provenance,
        "created_at": event.created_at.isoformat() if event.created_at else None,
        "updated_at": event.updated_at.isoformat() if event.updated_at else None,
    }


@router.patch("/events/{event_id}/review")
def update_event_review(event_id: str, review: EventReviewUpdate, db: Session = Depends(get_db)):
    """Update event review status and notes."""
    event = db.query(dbm.Event).filter(dbm.Event.id == event_id).first()
    if not event:
        raise HTTPException(404, "Event not found")

    # Save history
    history = dbm.EventReviewHistory(
        event_id=event_id,
        previous_status=event.review_status,
        new_status=review.review_status.upper(),
        notes=review.notes,
    )
    db.add(history)

    event.review_status = review.review_status.upper()
    if review.notes:
        event.reviewer_notes = review.notes
    event.updated_at = datetime.utcnow()
    db.commit()

    return {"status": "updated", "review_status": event.review_status}


# ═══════════════════════════════ TRAINING ═══════════════════════════════

@router.post("/training/start")
def start_training(job_data: TrainingJobCreate, db: Session = Depends(get_db)):
    """Start a model training job in the background."""
    model = db.query(dbm.ModelRegistry).filter(dbm.ModelRegistry.id == job_data.model_id).first()
    if not model:
        raise HTTPException(404, "Model not found")

    # Check for existing running jobs
    running = db.query(dbm.TrainingJob).filter(dbm.TrainingJob.status == "RUNNING").count()
    if running > 0:
        raise HTTPException(409, "A training job is already running. Wait for it to complete.")

    job = dbm.TrainingJob(
        model_id=job_data.model_id,
        dataset_name=job_data.dataset_name,
        epochs=job_data.epochs,
        batch_size=job_data.batch_size,
        image_size=job_data.image_size,
        learning_rate=job_data.learning_rate,
        device=job_data.device,
        total_epochs=job_data.epochs,
        status="PENDING",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Start training in background thread
    def _run_training(job_id: str, model_id: str):
        from app.db.database import SessionLocal
        session = SessionLocal()
        try:
            job = session.query(dbm.TrainingJob).filter(dbm.TrainingJob.id == job_id).first()
            model = session.query(dbm.ModelRegistry).filter(dbm.ModelRegistry.id == model_id).first()
            if not job or not model:
                return

            job.status = "RUNNING"
            job.started_at = datetime.utcnow()
            model.status = "TRAINING"
            session.commit()

            try:
                from ultralytics import YOLO
                yolo_model = YOLO(model.variant or "yolov8n.pt")

                # Check dataset
                dataset_path = None
                ds = session.query(dbm.Dataset).filter(dbm.Dataset.name == job.dataset_name).first()
                if ds and ds.local_path:
                    dataset_path = ds.local_path

                if not dataset_path or not Path(dataset_path).exists():
                    job.status = "FAILED"
                    job.error_message = f"Dataset '{job.dataset_name}' not found at '{dataset_path}'"
                    model.status = "PRETRAINED"
                    session.commit()
                    return

                results = yolo_model.train(
                    data=dataset_path,
                    epochs=job.epochs,
                    batch=job.batch_size,
                    imgsz=job.image_size,
                    device=job.device,
                    project=str(BASE_DIR / "data" / "training_runs"),
                    name=f"{model.name}_{job.id[:8]}",
                    exist_ok=True,
                )

                # Update job and model
                job.status = "COMPLETED"
                job.completed_at = datetime.utcnow()

                # Find best checkpoint
                run_dir = BASE_DIR / "data" / "training_runs" / f"{model.name}_{job.id[:8]}"
                best_pt = run_dir / "weights" / "best.pt"
                if best_pt.exists():
                    job.checkpoint_path = str(best_pt)
                    model.checkpoint_path = str(best_pt)
                    model.trained = True
                    model.status = "TRAINED"
                    model.last_training_time = datetime.utcnow()

                session.commit()

            except Exception as e:
                job.status = "FAILED"
                job.error_message = str(e)
                model.status = "PRETRAINED" if not model.trained else model.status
                session.commit()
                logger.error(f"Training failed: {e}")

        finally:
            session.close()

    thread = threading.Thread(target=_run_training, args=(job.id, model.id), daemon=True)
    thread.start()

    return {
        "job_id": job.id,
        "status": "PENDING",
        "message": "Training job started in background.",
    }


@router.get("/training/jobs")
def list_training_jobs(db: Session = Depends(get_db)):
    jobs = db.query(dbm.TrainingJob).order_by(dbm.TrainingJob.created_at.desc()).limit(20).all()
    return [
        {
            "id": j.id, "model_id": j.model_id, "dataset_name": j.dataset_name,
            "status": j.status, "current_epoch": j.current_epoch,
            "total_epochs": j.total_epochs, "device": j.device,
            "train_loss": j.train_loss, "val_loss": j.val_loss,
            "best_map50": j.best_map50, "error_message": j.error_message,
            "started_at": j.started_at.isoformat() if j.started_at else None,
            "completed_at": j.completed_at.isoformat() if j.completed_at else None,
            "created_at": j.created_at.isoformat() if j.created_at else None,
        }
        for j in jobs
    ]


@router.get("/training/jobs/{job_id}")
def get_training_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(dbm.TrainingJob).filter(dbm.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "Training job not found")
    return {
        "id": job.id, "model_id": job.model_id, "dataset_name": job.dataset_name,
        "status": job.status, "epochs": job.epochs, "batch_size": job.batch_size,
        "image_size": job.image_size, "learning_rate": job.learning_rate,
        "device": job.device, "current_epoch": job.current_epoch,
        "total_epochs": job.total_epochs, "train_loss": job.train_loss,
        "val_loss": job.val_loss, "best_map50": job.best_map50,
        "checkpoint_path": job.checkpoint_path, "error_message": job.error_message,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }


# ═══════════════════════════════ EVALUATION ═══════════════════════════════

@router.post("/evaluation/start")
def start_evaluation(job_data: EvaluationJobCreate, db: Session = Depends(get_db)):
    """Start model evaluation in background."""
    model = db.query(dbm.ModelRegistry).filter(dbm.ModelRegistry.id == job_data.model_id).first()
    if not model:
        raise HTTPException(404, "Model not found")

    job = dbm.EvaluationJob(
        model_id=job_data.model_id,
        dataset_name=job_data.dataset_name,
        dataset_split=job_data.dataset_split,
        status="PENDING",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    def _run_eval(job_id: str, model_id: str):
        from app.db.database import SessionLocal
        session = SessionLocal()
        try:
            job = session.query(dbm.EvaluationJob).filter(dbm.EvaluationJob.id == job_id).first()
            model = session.query(dbm.ModelRegistry).filter(dbm.ModelRegistry.id == model_id).first()
            if not job or not model:
                return

            job.status = "RUNNING"
            job.started_at = datetime.utcnow()
            session.commit()

            try:
                from ultralytics import YOLO
                checkpoint = model.checkpoint_path or model.variant or "yolov8n.pt"
                yolo_model = YOLO(checkpoint)

                ds = session.query(dbm.Dataset).filter(dbm.Dataset.name == job.dataset_name).first()
                if not ds or not ds.local_path:
                    job.status = "FAILED"
                    job.error_message = f"Dataset '{job.dataset_name}' not found"
                    session.commit()
                    return

                metrics = yolo_model.val(data=ds.local_path, split=job.dataset_split)

                job.status = "COMPLETED"
                job.completed_at = datetime.utcnow()
                job.map50 = float(metrics.box.map50) if hasattr(metrics.box, 'map50') else None
                job.map50_95 = float(metrics.box.map) if hasattr(metrics.box, 'map') else None
                job.precision = float(metrics.box.mp) if hasattr(metrics.box, 'mp') else None
                job.recall = float(metrics.box.mr) if hasattr(metrics.box, 'mr') else None

                model.evaluation_status = "EVALUATED"
                model.metrics = {
                    "map50": job.map50, "map50_95": job.map50_95,
                    "precision": job.precision, "recall": job.recall,
                }
                session.commit()

            except Exception as e:
                job.status = "FAILED"
                job.error_message = str(e)
                session.commit()
                logger.error(f"Evaluation failed: {e}")
        finally:
            session.close()

    thread = threading.Thread(target=_run_eval, args=(job.id, model.id), daemon=True)
    thread.start()

    return {"job_id": job.id, "status": "PENDING"}


@router.get("/evaluation/jobs")
def list_evaluation_jobs(db: Session = Depends(get_db)):
    jobs = db.query(dbm.EvaluationJob).order_by(dbm.EvaluationJob.created_at.desc()).limit(20).all()
    return [
        {
            "id": j.id, "model_id": j.model_id, "dataset_name": j.dataset_name,
            "dataset_split": j.dataset_split, "status": j.status,
            "precision": j.precision, "recall": j.recall,
            "map50": j.map50, "map50_95": j.map50_95,
            "avg_inference_ms": j.avg_inference_ms,
            "num_images": j.num_images, "error_message": j.error_message,
            "started_at": j.started_at.isoformat() if j.started_at else None,
            "completed_at": j.completed_at.isoformat() if j.completed_at else None,
        }
        for j in jobs
    ]


# ═══════════════════════════════ REPLAY ═══════════════════════════════

@router.post("/replay/load")
def replay_load(file: UploadFile = File(...)):
    """Load recorded data for replay."""
    from app.services.replay import replay_engine

    file_path = _save_upload(file, "replay_data")
    ext = Path(file_path).suffix.lower()

    try:
        if ext == ".json":
            with open(file_path) as f:
                records = json.load(f)
            if isinstance(records, dict) and "observations" in records:
                records = records["observations"]
        elif ext == ".csv":
            import pandas as pd
            df = pd.read_csv(file_path)
            records = df.to_dict(orient="records")
        else:
            raise HTTPException(400, f"Unsupported format: {ext}")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid JSON: {e}")

    result = replay_engine.load_records(records)
    return result


@router.post("/replay/control")
def replay_control(control: ReplayControl):
    """Control replay playback."""
    from app.services.replay import replay_engine

    if control.action == "start":
        return replay_engine.start()
    elif control.action == "pause":
        return replay_engine.pause()
    elif control.action == "resume":
        return replay_engine.resume()
    elif control.action == "reset":
        return replay_engine.reset()

    if control.speed is not None:
        replay_engine.set_speed(control.speed)

    return replay_engine.get_status()


@router.get("/replay/status")
def replay_status():
    from app.services.replay import replay_engine
    return replay_engine.get_status()


# ═══════════════════════════════ SYNTHETIC ═══════════════════════════════

@router.post("/synthetic/run")
def run_synthetic_scenarios(db: Session = Depends(get_db)):
    """Run all synthetic integration test scenarios."""
    from app.services.synthetic_scenarios import generate_synthetic_scenarios
    from app.services.correlation import correlate_observation_group, check_duplicates
    from app.services.prioritization import calculate_priority

    scenarios = generate_synthetic_scenarios()
    results = {}

    for name, scenario in scenarios.items():
        obs_list = scenario["observations"]

        # Save observations to DB
        saved_ids = []
        for obs_data in obs_list:
            obs = dbm.Observation(
                id=obs_data.get("id", str(uuid.uuid4())),
                sensor_type=obs_data.get("sensor_type", "unknown"),
                observation_timestamp=datetime.fromisoformat(obs_data["observation_timestamp"]) if obs_data.get("observation_timestamp") else None,
                object_class=obs_data.get("object_class"),
                confidence=obs_data.get("confidence"),
                bbox_x1=obs_data.get("bbox_x1"), bbox_y1=obs_data.get("bbox_y1"),
                bbox_x2=obs_data.get("bbox_x2"), bbox_y2=obs_data.get("bbox_y2"),
                latitude=obs_data.get("latitude"), longitude=obs_data.get("longitude"),
                coordinate_crs=obs_data.get("coordinate_crs"),
                radar_track_id=obs_data.get("radar_track_id"),
                position_x=obs_data.get("position_x"), position_y=obs_data.get("position_y"),
                velocity_x=obs_data.get("velocity_x"), velocity_y=obs_data.get("velocity_y"),
                uncertainty=obs_data.get("uncertainty"),
                model_name=obs_data.get("model_name"),
                data_provenance="synthetic",
                processing_status="PROCESSED",
            )
            db.add(obs)
            saved_ids.append(obs.id)

        # Run correlation if multiple observations
        actual_correlation = None
        if len(obs_list) >= 2:
            correlations = correlate_observation_group(obs_list)
            actual_correlation = correlations[0]["decision_status"] if correlations else None

        # Run duplicate check
        actual_duplicates = None
        if len(obs_list) >= 2:
            dups = check_duplicates(obs_list)
            actual_duplicates = len(dups) > 0

        # Run prioritization
        correlations_for_priority = correlate_observation_group(obs_list) if len(obs_list) >= 2 else []
        event_result = calculate_priority(obs_list, correlations_for_priority)

        # Save event
        event = dbm.Event(
            review_priority=event_result["review_priority"],
            source_sensors=event_result["source_sensors"],
            observation_ids=saved_ids,
            evidence_summary=event_result["evidence_summary"],
            uncertainty=event_result.get("uncertainty"),
            missing_information=event_result.get("missing_information"),
            priority_explanation=event_result["priority_explanation"],
            data_provenance="synthetic",
        )
        db.add(event)

        results[name] = {
            "description": scenario["description"],
            "label": scenario["label"],
            "expected": scenario["expected"],
            "actual": {
                "observations_saved": len(saved_ids),
                "correlation_result": actual_correlation,
                "has_duplicates": actual_duplicates,
                "event_priority": event_result["review_priority"],
                "evidence_summary": event_result["evidence_summary"],
            },
            "pass": True,  # Will be evaluated below
        }

        # Check expected vs actual
        expected = scenario["expected"]
        if "correlation" in expected:
            results[name]["pass"] = actual_correlation == expected["correlation"]
        elif "duplicate_check" in expected:
            results[name]["pass"] = actual_duplicates == (expected["duplicate_check"] == "POTENTIAL_DUPLICATE")
        elif "event_priority" in expected:
            results[name]["pass"] = event_result["review_priority"] == expected["event_priority"]

    db.commit()

    passed = sum(1 for r in results.values() if r["pass"])
    total = len(results)

    return {
        "label": "SYNTHETIC INTEGRATION TEST — NOT LIVE SENSOR DATA",
        "total_scenarios": total,
        "passed": passed,
        "failed": total - passed,
        "scenarios": results,
    }


# ═══════════════════════════════ REPORTS ═══════════════════════════════

@router.get("/reports")
def list_reports(db: Session = Depends(get_db)):
    """List available evaluation and test reports."""
    reports = []

    # Evaluation job reports
    eval_jobs = db.query(dbm.EvaluationJob).filter(
        dbm.EvaluationJob.status == "COMPLETED"
    ).all()
    for j in eval_jobs:
        reports.append({
            "id": j.id,
            "report_type": "model_evaluation",
            "timestamp": j.completed_at.isoformat() if j.completed_at else None,
            "test_mode": "real_evaluation",
            "source": j.dataset_name,
            "model_checkpoint": None,
            "num_samples": j.num_images or 0,
            "metrics": {
                "precision": j.precision, "recall": j.recall,
                "map50": j.map50, "map50_95": j.map50_95,
            },
            "completion_status": j.status,
        })

    return reports


@router.get("/reports/export/{report_id}")
def export_report(report_id: str, db: Session = Depends(get_db)):
    """Export a report as JSON."""
    job = db.query(dbm.EvaluationJob).filter(dbm.EvaluationJob.id == report_id).first()
    if not job:
        raise HTTPException(404, "Report not found")

    return {
        "report_id": job.id,
        "type": "model_evaluation",
        "timestamp": job.completed_at.isoformat() if job.completed_at else None,
        "model_id": job.model_id,
        "dataset": job.dataset_name,
        "split": job.dataset_split,
        "metrics": {
            "precision": job.precision, "recall": job.recall,
            "map50": job.map50, "map50_95": job.map50_95,
            "avg_inference_ms": job.avg_inference_ms,
        },
        "num_images": job.num_images,
        "status": job.status,
    }


# ═══════════════════════════════ STATIC FILES ═══════════════════════════════

@router.get("/files/{file_type}/{filename}")
def serve_file(file_type: str, filename: str):
    """Serve uploaded/annotated images."""
    allowed_dirs = {"uploads": "uploads", "radar_data": "radar_data", "replay_data": "replay_data"}
    if file_type not in allowed_dirs:
        raise HTTPException(400, "Invalid file type")

    file_path = BASE_DIR / "data" / allowed_dirs[file_type] / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")

    from fastapi.responses import FileResponse
    return FileResponse(str(file_path))


# ═══════════════════════════════ RADAR TRACKS ═══════════════════════════════

@router.get("/radar/tracks")
def get_radar_tracks():
    """Get all current radar tracks from the in-memory tracker."""
    from app.services.radar_tracking import radar_tracker
    return {
        "tracks": radar_tracker.get_tracks(),
        "status": radar_tracker.get_status(),
    }


@router.get("/radar/tracks/active")
def get_active_radar_tracks():
    """Get currently active radar tracks."""
    from app.services.radar_tracking import radar_tracker
    return {
        "tracks": radar_tracker.get_active_tracks(),
        "active_count": len(radar_tracker.get_active_tracks()),
    }


@router.delete("/radar/tracks/reset")
def reset_radar_tracks():
    """Reset all radar tracks (clears in-memory state)."""
    from app.services.radar_tracking import radar_tracker
    radar_tracker.reset()
    return {"status": "reset", "message": "All radar tracks cleared."}


# ═══════════════════════════════ SYSTEM LOGS ═══════════════════════════════

@router.get("/system/logs")
def get_system_logs(
    level: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Get system processing logs with optional level filter."""
    query = db.query(dbm.ProcessingLog)
    if level:
        query = query.filter(dbm.ProcessingLog.level == level.upper())
    total = query.count()
    logs = query.order_by(dbm.ProcessingLog.timestamp.desc()).offset(offset).limit(limit).all()
    return {
        "total": total,
        "logs": [
            {
                "id": log.id,
                "level": log.level,
                "component": log.component,
                "message": log.message,
                "details": log.details,
                "timestamp": log.timestamp.isoformat() if log.timestamp else None,
            }
            for log in logs
        ],
    }


@router.post("/system/logs")
def create_log(
    level: str = "INFO",
    component: str = "",
    message: str = "",
    details: Optional[dict] = None,
    db: Session = Depends(get_db),
):
    """Create a processing log entry."""
    log = dbm.ProcessingLog(
        level=level.upper(),
        component=component,
        message=message,
        details=details,
    )
    db.add(log)
    db.commit()
    return {"status": "logged"}


# ═══════════════════════════════ REPLAY PROCESSED RECORDS ═══════════════════════════════

@router.get("/replay/processed")
def get_replay_processed(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Get processed records from the current replay session."""
    from app.services.replay import replay_engine
    records = replay_engine.get_processed(limit=limit, offset=offset)
    return {
        "status": replay_engine.status,
        "total_processed": len(replay_engine.processed_records),
        "records": records,
        "label": "RECORDED DATA REPLAY — NOT A LIVE SENSOR FEED",
    }


# ═══════════════════════════════ DATASET MANAGEMENT ═══════════════════════════════

@router.post("/datasets/register")
def register_dataset(
    name: str,
    sensor_type: str,
    local_path: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Register a new dataset path."""
    existing = db.query(dbm.Dataset).filter(dbm.Dataset.name == name).first()
    if existing:
        if local_path:
            existing.local_path = local_path
            db.commit()
        return {"status": "updated", "dataset_id": existing.id}

    ds = dbm.Dataset(
        name=name,
        sensor_type=sensor_type,
        local_path=local_path,
        validation_status="NOT_VALIDATED",
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return {"status": "registered", "dataset_id": ds.id}


@router.patch("/datasets/{dataset_id}/path")
def update_dataset_path(dataset_id: str, local_path: str, db: Session = Depends(get_db)):
    """Update the local path of a registered dataset."""
    ds = db.query(dbm.Dataset).filter(dbm.Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    ds.local_path = local_path
    ds.validation_status = "NOT_VALIDATED"
    db.commit()
    return {"status": "updated", "local_path": local_path}


# ═══════════════════════════════ TRAINING CANCEL ═══════════════════════════════

@router.post("/training/cancel/{job_id}")
def cancel_training(job_id: str, db: Session = Depends(get_db)):
    """Request cancellation of a training job."""
    job = db.query(dbm.TrainingJob).filter(dbm.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "Training job not found")
    if job.status not in ("PENDING", "RUNNING"):
        return {"status": "cannot_cancel", "current_status": job.status}
    job.status = "CANCELLED"
    job.error_message = "Cancelled by user."
    db.commit()
    return {"status": "cancellation_requested", "job_id": job_id}


# ═══════════════════════════════ MODEL STATUS UPDATE ═══════════════════════════════

@router.get("/models/{model_id}/load-status")
def get_model_load_status(model_id: str, db: Session = Depends(get_db)):
    """Check if a model is loaded in memory and its current status."""
    from app.services.satellite_detector import satellite_detector
    from app.services.thermal_detector import thermal_detector

    model = db.query(dbm.ModelRegistry).filter(dbm.ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(404, "Model not found")

    if model.sensor_type == "satellite":
        loaded = satellite_detector.is_loaded
        status = satellite_detector.get_status()
    elif model.sensor_type == "thermal":
        loaded = thermal_detector.is_loaded
        status = thermal_detector.get_status()
    else:
        loaded = False
        status = {}

    return {
        "model_id": model_id,
        "db_status": model.status,
        "in_memory_loaded": loaded,
        "runtime_status": status,
    }


# ═══════════════════════════════ CORRELATION HISTORY ═══════════════════════════════

@router.get("/correlation/history")
def correlation_history(
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Get recent correlation decisions."""
    decisions = db.query(dbm.CorrelationDecision).order_by(
        dbm.CorrelationDecision.created_at.desc()
    ).limit(limit).all()
    return {
        "total": db.query(dbm.CorrelationDecision).count(),
        "decisions": [
            {
                "id": d.id,
                "observation_ids": d.observation_ids,
                "decision_status": d.decision_status,
                "evidence": d.evidence,
                "reason": d.reason,
                "missing_metadata": d.missing_metadata,
                "temporal_delta_s": d.temporal_delta_s,
                "spatial_distance_m": d.spatial_distance_m,
                "event_id": d.event_id,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in decisions
        ],
    }
