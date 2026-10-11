"""
Initialise the database (tables + dataset/model registry) and cache the COCO-pretrained YOLOv8n
base weights used to initialise both detectors. Safe to run repeatedly.

  python scripts/init_backend.py [--skip-weights]
"""
import argparse

import _bootstrap  # noqa: F401
from app.db.database import init_db, db_path
from app.main import seed_initial_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-weights", action="store_true")
    a = ap.parse_args()
    init_db()
    seed_initial_data()
    print(f"Database initialised ({db_path}).")
    if a.skip_weights:
        return
    try:
        from ultralytics import YOLO
        from app.services.detector_base import pretrained_weights_path
        p = pretrained_weights_path("yolov8n.pt")
        YOLO(str(p))
        print(f"Pretrained base weights available: {p}")
    except ImportError:
        print("Ultralytics not installed — image inference/training unavailable until it is installed.")
    except Exception as e:
        print(f"Could not obtain pretrained weights (internet needed once): {e}")


if __name__ == "__main__":
    main()
