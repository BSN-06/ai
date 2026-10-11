"""
AI-Assisted Multi-Sensor Surveillance and Event Prioritization System
Core configuration module.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    APP_NAME: str = "AI-Assisted Multi-Sensor Surveillance System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "sqlite:///./data/surveillance.db"
    
    # File storage
    UPLOAD_DIR: str = "./data/uploads"
    CHECKPOINT_DIR: str = "./data/checkpoints"
    DATASET_DIR: str = "./data/datasets"
    REPORT_DIR: str = "./data/reports"
    
    # Upload limits
    MAX_UPLOAD_SIZE_MB: int = 100
    ALLOWED_IMAGE_EXTENSIONS: list[str] = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
    ALLOWED_DATA_EXTENSIONS: list[str] = [".csv", ".json", ".txt", ".geojson"]
    
    # Model defaults
    SATELLITE_MODEL_VARIANT: str = "yolov8n.pt"
    THERMAL_MODEL_VARIANT: str = "yolov8n.pt"
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.25
    DEFAULT_IOU_THRESHOLD: float = 0.45
    
    # Correlation parameters
    TEMPORAL_WINDOW_SECONDS: float = 300.0  # 5 minutes
    SPATIAL_DISTANCE_THRESHOLD_M: float = 500.0  # 500 meters
    
    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    # Device
    FORCE_CPU: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure directories exist
BASE_DIR = Path(__file__).resolve().parent.parent.parent
for dir_attr in ["UPLOAD_DIR", "CHECKPOINT_DIR", "DATASET_DIR", "REPORT_DIR"]:
    dir_path = BASE_DIR / getattr(settings, dir_attr)
    dir_path.mkdir(parents=True, exist_ok=True)

# Also ensure data directory exists for the database
(BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
