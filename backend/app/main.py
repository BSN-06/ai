"""
AI-Assisted Multi-Sensor Surveillance and Event Prioritization System
Main FastAPI application entry point.

This is an OFFLINE RESEARCH PROTOTYPE.
It does not connect to operational military infrastructure, live military sensors,
or classified systems.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings, BASE_DIR
from app.db.database import init_db, SessionLocal
from app.db import models as dbm
from app.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def seed_initial_data():
    """Seed the database with dataset and model registry entries."""
    db = SessionLocal()
    try:
        # Only seed if empty
        if db.query(dbm.Dataset).count() > 0:
            return

        # ── Register known datasets ──
        datasets = [
            dbm.Dataset(
                name="xView",
                sensor_type="satellite",
                source_url="https://challenge.xviewdataset.org/",
                annotation_format="geojson_bboxes",
                num_classes=60,
                classes=["Fixed-wing Aircraft", "Small Aircraft", "Cargo Plane", "Helicopter",
                         "Passenger Vehicle", "Small Car", "Bus", "Pickup Truck", "Utility Truck",
                         "Truck", "Cargo Truck", "Truck w/Box", "Truck Tractor", "Trailer",
                         "Truck w/Flatbed", "Truck w/Liquid", "Crane Truck", "Railway Vehicle",
                         "Passenger Car", "Cargo Car", "Flat Car", "Tank car", "Locomotive",
                         "Maritime Vessel", "Motorboat", "Sailboat", "Tugboat", "Barge",
                         "Fishing Vessel", "Ferry", "Yacht", "Container Ship", "Oil Tanker",
                         "Engineering Vehicle", "Tower crane", "Container Crane", "Reach Stacker",
                         "Straddle Carrier", "Mobile Crane", "Dump Truck", "Haul Truck",
                         "Scraper/Tractor", "Front loader/Bulldozer", "Excavator", "Cement Mixer",
                         "Ground Grader", "Hut/Tent", "Shed", "Building", "Aircraft Hangar",
                         "Damaged Building", "Facility", "Construction Site", "Vehicle Lot",
                         "Helipad", "Storage Tank", "Shipping container lot",
                         "Shipping Container", "Pylon", "Tower"],
                validation_status="NOT_INSTALLED",
                access_requirements="Registration required at challenge.xviewdataset.org. "
                                    "Dataset is ~20GB. Contains satellite imagery with 60 object classes.",
                setup_instructions=(
                    "1. Visit https://challenge.xviewdataset.org/ and register.\n"
                    "2. Download the training images and labels.\n"
                    "3. Extract to backend/data/datasets/xview/\n"
                    "4. Convert annotations to YOLO format using the provided script.\n"
                    "5. Validate the dataset through the dashboard."
                ),
            ),
            dbm.Dataset(
                name="DOTA",
                sensor_type="satellite",
                source_url="https://captain-whu.github.io/DOTA/dataset.html",
                annotation_format="oriented_bboxes",
                num_classes=15,
                classes=["plane", "ship", "storage-tank", "baseball-diamond",
                         "tennis-court", "basketball-court", "ground-track-field",
                         "harbor", "bridge", "large-vehicle", "small-vehicle",
                         "helicopter", "roundabout", "soccer-ball-field",
                         "swimming-pool"],
                validation_status="NOT_INSTALLED",
                access_requirements="Download from DOTA website. Multiple versions available (v1.0, v1.5, v2.0).",
                setup_instructions=(
                    "1. Visit https://captain-whu.github.io/DOTA/dataset.html\n"
                    "2. Download the desired version.\n"
                    "3. Extract to backend/data/datasets/dota/\n"
                    "4. Convert annotations to YOLO format.\n"
                    "5. Validate through the dashboard."
                ),
            ),
            dbm.Dataset(
                name="FLIR_ADAS",
                sensor_type="thermal",
                source_url="https://www.flir.com/oem/adas/adas-dataset-form/",
                annotation_format="coco_json",
                num_classes=5,
                classes=["person", "car", "bicycle", "dog", "other_vehicle"],
                validation_status="NOT_INSTALLED",
                access_requirements="Registration required at FLIR/Teledyne website. "
                                    "Contains thermal + visible paired images. ~14GB.",
                setup_instructions=(
                    "1. Visit https://www.flir.com/oem/adas/adas-dataset-form/\n"
                    "2. Complete registration form.\n"
                    "3. Download FLIR ADAS dataset.\n"
                    "4. Extract to backend/data/datasets/flir_adas/\n"
                    "5. Convert COCO annotations to YOLO format.\n"
                    "6. Validate through the dashboard."
                ),
            ),
            dbm.Dataset(
                name="DLR_Marine_Radar",
                sensor_type="radar",
                source_url="https://www.dlr.de/en/kn/about-us/departments/nautical-systems/real-world-marine-radar-datasets-for-target-tracking",
                annotation_format="csv/custom",
                validation_status="NOT_INSTALLED",
                access_requirements="Access may require contacting DLR. Maritime radar data — NOT aircraft radar.",
                setup_instructions=(
                    "1. Visit the DLR dataset page.\n"
                    "2. Request access to the marine radar dataset.\n"
                    "3. Download and extract to backend/data/datasets/dlr_radar/\n"
                    "4. Import through the Radar Analysis page.\n"
                    "NOTE: This is MARITIME radar data, not aircraft radar."
                ),
            ),
        ]
        for ds in datasets:
            db.add(ds)

        # ── Register models ──
        models = [
            dbm.ModelRegistry(
                name="satellite_yolov8n",
                sensor_type="satellite",
                architecture="YOLOv8",
                variant="yolov8n.pt",
                backbone="CSPDarknet (Cross Stage Partial Darknet)",
                neck="PAN-FPN (Path Aggregation Network with Feature Pyramid Network)",
                head="Decoupled anchor-free detection head",
                dataset_name="xView",
                pretrained=True,
                trained=False,
                num_classes=80,
                input_size=640,
                status="PRETRAINED",
                evaluation_status="NOT_EVALUATED",
                known_limitations=(
                    "COCO-pretrained weights. NOT fine-tuned on satellite imagery. "
                    "COCO classes differ from xView classes. Performance on satellite imagery "
                    "may be limited without fine-tuning. Small objects in aerial views may be missed "
                    "without tiled inference."
                ),
            ),
            dbm.ModelRegistry(
                name="thermal_yolov8n",
                sensor_type="thermal",
                architecture="YOLOv8",
                variant="yolov8n.pt",
                backbone="CSPDarknet",
                neck="PAN-FPN",
                head="Decoupled anchor-free detection head",
                dataset_name="FLIR_ADAS",
                pretrained=True,
                trained=False,
                num_classes=80,
                input_size=640,
                status="PRETRAINED",
                evaluation_status="NOT_EVALUATED",
                known_limitations=(
                    "COCO-pretrained weights. NOT fine-tuned on thermal imagery. "
                    "The FLIR ADAS dataset contains automotive thermal data. "
                    "This does NOT prove military-target or aircraft-detection performance."
                ),
            ),
        ]
        for m in models:
            db.add(m)

        db.commit()
        logger.info("Seeded initial datasets and models.")
    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        db.rollback()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init DB and seed data."""
    logger.info("Starting AI-Assisted Multi-Sensor Surveillance System...")
    init_db()
    seed_initial_data()
    logger.info("Database initialized and seeded.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AI-Assisted Multi-Sensor Surveillance and Event Prioritization System. "
        "OFFLINE RESEARCH PROTOTYPE — not connected to operational infrastructure."
    ),
    lifespan=lifespan,
)

# CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for uploads
data_dir = BASE_DIR / "data"
data_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(data_dir)), name="static")

# Include API routes
app.include_router(router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
