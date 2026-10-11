"""
Pydantic schemas for API request/response validation and the common observation format.
"""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ───────────────────────────── Common Observation Schema ─────────────────────────────

class ObservationCreate(BaseModel):
    """Schema for creating a new observation from any sensor module."""
    sensor_type: str = Field(..., description="satellite | thermal | radar")
    observation_timestamp: Optional[datetime] = None
    is_utc: bool = True
    source_dataset: Optional[str] = None
    source_record_id: Optional[str] = None
    object_class: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    bbox_x1: Optional[float] = None
    bbox_y1: Optional[float] = None
    bbox_x2: Optional[float] = None
    bbox_y2: Optional[float] = None
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    coordinate_crs: Optional[str] = None
    radar_track_id: Optional[str] = None
    position_x: Optional[float] = None
    position_y: Optional[float] = None
    velocity_x: Optional[float] = None
    velocity_y: Optional[float] = None
    uncertainty: Optional[float] = Field(None, ge=0.0)
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    checkpoint_id: Optional[str] = None
    data_provenance: str = "real_inference"
    processing_status: str = "PROCESSED"
    image_path: Optional[str] = None
    annotated_image_path: Optional[str] = None
    metadata_json: Optional[dict] = None

    @field_validator("sensor_type")
    @classmethod
    def validate_sensor_type(cls, v: str) -> str:
        allowed = {"satellite", "thermal", "radar"}
        if v.lower() not in allowed:
            raise ValueError(f"sensor_type must be one of {allowed}")
        return v.lower()


class ObservationResponse(ObservationCreate):
    """Full observation record as returned by the API."""
    id: str
    created_at: datetime

    class Config:
        from_attributes = True


# ───────────────────────────── Events ─────────────────────────────

class EventCreate(BaseModel):
    review_priority: str = "LOW"
    source_sensors: Optional[list[str]] = None
    observation_ids: Optional[list[str]] = None
    evidence_summary: Optional[str] = None
    uncertainty: Optional[float] = None
    missing_information: Optional[list[str]] = None
    priority_explanation: Optional[str] = None
    data_provenance: str = "real"

    @field_validator("review_priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        allowed = {"LOW", "MEDIUM", "HIGH"}
        if v.upper() not in allowed:
            raise ValueError(f"review_priority must be one of {allowed}")
        return v.upper()


class EventResponse(EventCreate):
    id: str
    review_status: str
    reviewer_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class EventReviewUpdate(BaseModel):
    review_status: str
    notes: Optional[str] = None

    @field_validator("review_status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"PENDING", "REVIEWING", "REVIEWED", "DISMISSED"}
        if v.upper() not in allowed:
            raise ValueError(f"review_status must be one of {allowed}")
        return v.upper()


# ───────────────────────────── Correlation ─────────────────────────────

class CorrelationResult(BaseModel):
    id: Optional[str] = None
    observation_ids: list[str]
    decision_status: str  # CORRELATED, NOT_CORRELATED, INSUFFICIENT_METADATA
    evidence: Optional[dict] = None
    reason: Optional[str] = None
    missing_metadata: Optional[list[str]] = None
    temporal_delta_s: Optional[float] = None
    spatial_distance_m: Optional[float] = None
    config_params: Optional[dict] = None
    event_id: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ───────────────────────────── Datasets ─────────────────────────────

class DatasetResponse(BaseModel):
    id: str
    name: str
    sensor_type: str
    source_url: Optional[str] = None
    local_path: Optional[str] = None
    annotation_format: Optional[str] = None
    classes: Optional[list] = None
    num_classes: Optional[int] = None
    file_count: Optional[int] = None
    train_count: Optional[int] = None
    val_count: Optional[int] = None
    test_count: Optional[int] = None
    validation_status: str
    validation_message: Optional[str] = None
    access_requirements: Optional[str] = None
    setup_instructions: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ───────────────────────────── Models ─────────────────────────────

class ModelResponse(BaseModel):
    id: str
    name: str
    sensor_type: str
    architecture: str
    variant: Optional[str] = None
    backbone: Optional[str] = None
    neck: Optional[str] = None
    head: Optional[str] = None
    dataset_name: Optional[str] = None
    pretrained: bool
    trained: bool
    checkpoint_path: Optional[str] = None
    num_classes: Optional[int] = None
    class_names: Optional[list] = None
    input_size: int
    status: str
    last_training_time: Optional[datetime] = None
    evaluation_status: str
    metrics: Optional[dict] = None
    known_limitations: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ───────────────────────────── Training ─────────────────────────────

class TrainingJobCreate(BaseModel):
    model_id: str
    dataset_name: str
    epochs: int = Field(50, ge=1, le=1000)
    batch_size: int = Field(16, ge=1, le=128)
    image_size: int = Field(640, ge=32, le=2048)
    learning_rate: Optional[float] = Field(None, gt=0.0)
    device: str = "cpu"


class TrainingJobResponse(BaseModel):
    id: str
    model_id: str
    dataset_name: str
    status: str
    epochs: int
    batch_size: int
    image_size: int
    learning_rate: Optional[float] = None
    device: str
    current_epoch: int
    total_epochs: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    best_map50: Optional[float] = None
    checkpoint_path: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ───────────────────────────── Evaluation ─────────────────────────────

class EvaluationJobCreate(BaseModel):
    model_id: str
    dataset_name: str
    dataset_split: str = "val"


class EvaluationJobResponse(BaseModel):
    id: str
    model_id: str
    dataset_name: str
    dataset_split: str
    status: str
    num_images: Optional[int] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    map50: Optional[float] = None
    map50_95: Optional[float] = None
    per_class_metrics: Optional[dict] = None
    avg_inference_ms: Optional[float] = None
    error_message: Optional[str] = None
    report_path: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ───────────────────────────── Replay ─────────────────────────────

class ReplaySessionResponse(BaseModel):
    id: str
    status: str
    replay_speed: float
    total_records: int
    processed_records: int
    current_timestamp: Optional[datetime] = None
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    source_file: Optional[str] = None
    data_provenance: str
    created_at: datetime

    class Config:
        from_attributes = True


class ReplayControl(BaseModel):
    action: str  # start, pause, resume, reset
    speed: Optional[float] = Field(None, ge=0.1, le=100.0)

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        allowed = {"start", "pause", "resume", "reset"}
        if v.lower() not in allowed:
            raise ValueError(f"action must be one of {allowed}")
        return v.lower()


# ───────────────────────────── System ─────────────────────────────

class SystemHealth(BaseModel):
    status: str
    app_name: str
    version: str
    database: str
    models_available: int
    datasets_registered: int
    total_observations: int
    total_events: int
    pending_events: int
    gpu_available: bool
    device: str
    timestamp: datetime


# ───────────────────────────── Reports ─────────────────────────────

class ReportResponse(BaseModel):
    id: str
    report_type: str
    timestamp: datetime
    test_mode: str
    source: str
    model_checkpoint: Optional[str] = None
    num_samples: int
    metrics: Optional[dict] = None
    warnings: Optional[list[str]] = None
    limitations: Optional[list[str]] = None
    completion_status: str


# ───────────────────────────── Inference Request ─────────────────────────────

class InferenceRequest(BaseModel):
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    model_id: Optional[str] = None


# ───────────────────────────── Radar ─────────────────────────────

class RadarTrack(BaseModel):
    track_id: str
    timestamps: list[datetime]
    positions_x: list[float]
    positions_y: list[float]
    velocities_x: Optional[list[float]] = None
    velocities_y: Optional[list[float]] = None
    estimated_positions_x: Optional[list[float]] = None
    estimated_positions_y: Optional[list[float]] = None
    latitudes: Optional[list[float]] = None
    longitudes: Optional[list[float]] = None
    coordinate_crs: Optional[str] = None
    num_observations: int
    duration_s: Optional[float] = None
    status: str = "ACTIVE"
