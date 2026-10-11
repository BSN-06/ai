"""
Fine-tune a registered detector on a VALIDATED dataset (runs in the foreground, same code path as the API).

  python scripts/train.py --model satellite_yolov8n --dataset xView     --epochs 50 --batch 16 --imgsz 640 --device cpu
  python scripts/train.py --model thermal_yolov8n   --dataset FLIR_ADAS --epochs 50 --batch 16 --imgsz 640 --device cpu
"""
import argparse

import _bootstrap  # noqa: F401
from app.db.database import SessionLocal, init_db
from app.db import models as dbm
from app.main import seed_initial_data
from app.services.training import preflight, run_training_job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    init_db(); seed_initial_data()
    s = SessionLocal()
    model = s.query(dbm.ModelRegistry).filter(dbm.ModelRegistry.name == a.model).first()
    ds = s.query(dbm.Dataset).filter(dbm.Dataset.name == a.dataset).first()
    if not model:
        raise SystemExit(f"Unknown model {a.model}")
    problems = preflight(ds, a.device)
    if problems:
        raise SystemExit("Cannot train:\n  - " + "\n  - ".join(problems))
    job = dbm.TrainingJob(model_id=model.id, dataset_name=a.dataset, epochs=a.epochs, batch_size=a.batch,
                          image_size=a.imgsz, learning_rate=a.lr, device=a.device, total_epochs=a.epochs)
    s.add(job); s.commit()
    print(f"Training job {job.id} (also visible in the dashboard Training page)")
    run_training_job(job.id)
    s.expire_all()
    job = s.get(dbm.TrainingJob, job.id)
    print(f"Status: {job.status}  checkpoint: {job.checkpoint_path}  error: {job.error_message}")


if __name__ == "__main__":
    main()
