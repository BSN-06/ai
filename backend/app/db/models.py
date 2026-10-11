"""SQLAlchemy ORM models for the surveillance system."""
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, Text, JSON,
    ForeignKey, Enum as SAEnum
)
from sqlalchemy.orm import relationship
from app.db.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class Dataset(Base):
    """Registered dataset metadata."""
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False, unique=True)
    sensor_type = Column(String, nullable=False)  # satellite, thermal, radar
    source_url = Column(String, nullable=True)
    local_path = Column(String, nullable=True)
    annotation_format = Column(String, nullable=True)
    classes = Column(JSON, nullable=True)
    num_classes = Column(Integer, nullable=True)
    file_count = Column(Integer, nullable=True)
    train_count = Column(Integer, nullable=True)
    val_count = Column(Integer, nullable=True)
    test_count = Column(Integer, nullable=True)
    validation_status = Column(String, default="NOT_VALIDATED")
    validation_message = Column(Text, nullable=True)
    access_requirements = Column(Text, nullable=True)
    setup_instructions = Column(Text, nullable=True)
    size_bytes = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ModelRegistry(Base):
    """Registered model metadata."""
    __tablename__ = "model_registry"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    sensor_type = Column(String, nullable=False)  # satellite, thermal
    architecture = Column(String, nullable=False)
    variant = Column(String, nullable=True)
    backbone = Column(String, nullable=True)
    neck = Column(String, nullable=True)
    head = Column(String, nullable=True)
    dataset_name = Column(String, nullable=True)
    pretrained = Column(Boolean, default=True)
    trained = Column(Boolean, default=False)
    checkpoint_path = Column(String, nullable=True)
    num_classes = Column(Integer, nullable=True)
    class_names = Column(JSON, nullable=True)
    input_size = Column(Integer, default=640)
    status = Column(String, default="PRETRAINED")  # PRETRAINED, TRAINING, TRAINED, FAILED
    last_training_time = Column(DateTime, nullable=True)
    evaluation_status = Column(String, default="NOT_EVALUATED")
    metrics = Column(JSON, nullable=True)
    known_limitations = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TrainingJob(Base):
    """Training job tracking."""
    __tablename__ = "training_jobs"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("model_registry.id"), nullable=False)
    dataset_name = Column(String, nullable=False)
    status = Column(String, default="PENDING")  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    epochs = Column(Integer, default=50)
    batch_size = Column(Integer, default=16)
    image_size = Column(Integer, default=640)
    learning_rate = Column(Float, nullable=True)
    device = Column(String, default="cpu")
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=50)
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    best_map50 = Column(Float, nullable=True)
    checkpoint_path = Column(String, nullable=True)
    log_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    config = Column(JSON, nullable=True)
    epoch_history = Column(JSON, nullable=True)  # [{epoch, train_box_loss, ..., map50}]
    cancel_requested = Column(Boolean, default=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    model = relationship("ModelRegistry", backref="training_jobs")


class EvaluationJob(Base):
    """Evaluation job tracking."""
    __tablename__ = "evaluation_jobs"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("model_registry.id"), nullable=False)
    dataset_name = Column(String, nullable=False)
    dataset_split = Column(String, default="val")
    status = Column(String, default="PENDING")
    num_images = Column(Integer, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    map50 = Column(Float, nullable=True)
    map50_95 = Column(Float, nullable=True)
    per_class_metrics = Column(JSON, nullable=True)
    confusion_matrix = Column(JSON, nullable=True)
    avg_inference_ms = Column(Float, nullable=True)
    checkpoint_path = Column(String, nullable=True)
    model_status_at_eval = Column(String, nullable=True)  # PRETRAINED / TRAINED
    warnings = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    report_path = Column(String, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    model = relationship("ModelRegistry", backref="evaluation_jobs")


class Observation(Base):
    """Common observation schema stored in DB."""
    __tablename__ = "observations"

    id = Column(String, primary_key=True, default=generate_uuid)
    sensor_type = Column(String, nullable=False)  # satellite, thermal, radar
    observation_timestamp = Column(DateTime, nullable=True)
    is_utc = Column(Boolean, default=True)
    source_dataset = Column(String, nullable=True)
    source_record_id = Column(String, nullable=True)
    object_class = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    bbox_x1 = Column(Float, nullable=True)
    bbox_y1 = Column(Float, nullable=True)
    bbox_x2 = Column(Float, nullable=True)
    bbox_y2 = Column(Float, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    coordinate_crs = Column(String, nullable=True)
    radar_track_id = Column(String, nullable=True)
    position_x = Column(Float, nullable=True)
    position_y = Column(Float, nullable=True)
    velocity_x = Column(Float, nullable=True)
    velocity_y = Column(Float, nullable=True)
    uncertainty = Column(Float, nullable=True)
    model_name = Column(String, nullable=True)
    model_version = Column(String, nullable=True)
    checkpoint_id = Column(String, nullable=True)
    data_provenance = Column(String, nullable=True)  # real_inference, recorded, synthetic
    processing_status = Column(String, default="PROCESSED")
    image_path = Column(String, nullable=True)
    annotated_image_path = Column(String, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class CorrelationDecision(Base):
    """Record of correlation decisions between observations."""
    __tablename__ = "correlation_decisions"

    id = Column(String, primary_key=True, default=generate_uuid)
    observation_ids = Column(JSON, nullable=False)
    decision_status = Column(String, nullable=False)  # CORRELATED, NOT_CORRELATED, INSUFFICIENT_METADATA
    evidence = Column(JSON, nullable=True)
    reason = Column(Text, nullable=True)
    missing_metadata = Column(JSON, nullable=True)
    temporal_delta_s = Column(Float, nullable=True)
    spatial_distance_m = Column(Float, nullable=True)
    config_params = Column(JSON, nullable=True)
    event_id = Column(String, ForeignKey("events.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Event(Base):
    """Prioritized event for human review."""
    __tablename__ = "events"

    id = Column(String, primary_key=True, default=generate_uuid)
    review_priority = Column(String, default="LOW")  # LOW, MEDIUM, HIGH
    source_sensors = Column(JSON, nullable=True)
    observation_ids = Column(JSON, nullable=True)
    evidence_summary = Column(Text, nullable=True)
    uncertainty = Column(Float, nullable=True)
    missing_information = Column(JSON, nullable=True)
    priority_explanation = Column(Text, nullable=True)
    review_status = Column(String, default="PENDING")  # PENDING, REVIEWING, REVIEWED, DISMISSED
    reviewer_notes = Column(Text, nullable=True)
    data_provenance = Column(String, nullable=True)  # real, synthetic
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    correlations = relationship("CorrelationDecision", backref="event")


class EventReviewHistory(Base):
    """Audit trail for event review changes."""
    __tablename__ = "event_review_history"

    id = Column(String, primary_key=True, default=generate_uuid)
    event_id = Column(String, ForeignKey("events.id"), nullable=False)
    previous_status = Column(String, nullable=True)
    new_status = Column(String, nullable=False)
    notes = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


class ProcessingLog(Base):
    """Processing and error logs."""
    __tablename__ = "processing_logs"

    id = Column(String, primary_key=True, default=generate_uuid)
    level = Column(String, default="INFO")
    component = Column(String, nullable=True)
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


class ReplaySession(Base):
    """Recorded-data replay session tracking."""
    __tablename__ = "replay_sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    status = Column(String, default="STOPPED")  # STOPPED, PLAYING, PAUSED, COMPLETED
    replay_speed = Column(Float, default=1.0)
    total_records = Column(Integer, default=0)
    processed_records = Column(Integer, default=0)
    current_timestamp = Column(DateTime, nullable=True)
    start_timestamp = Column(DateTime, nullable=True)
    end_timestamp = Column(DateTime, nullable=True)
    source_file = Column(String, nullable=True)
    data_provenance = Column(String, default="recorded")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Report(Base):
    """Persisted evaluation / test report (one row per report, JSON body on disk and in DB)."""
    __tablename__ = "reports"

    id = Column(String, primary_key=True, default=generate_uuid)
    report_type = Column(String, nullable=False)  # satellite_detection, thermal_detection, radar_tracking,
    #                                               training, dataset_validation, fusion_tests,
    #                                               recorded_replay, synthetic_integration
    test_mode = Column(String, nullable=False)  # real_evaluation, recorded_replay, synthetic, dataset, training
    source = Column(String, nullable=True)
    model_checkpoint = Column(String, nullable=True)
    num_samples = Column(Integer, default=0)
    metrics = Column(JSON, nullable=True)
    warnings = Column(JSON, nullable=True)
    limitations = Column(JSON, nullable=True)
    completion_status = Column(String, default="COMPLETED")
    body = Column(JSON, nullable=True)
    file_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
