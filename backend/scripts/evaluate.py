"""
Evaluate a detector on a labelled split; metrics are computed from predictions vs ground truth.

  python scripts/evaluate.py --model satellite_yolov8n --dataset xView --split test
  python scripts/evaluate.py --model thermal_yolov8n   --dataset FLIR_ADAS --split val [--max-images 500]
"""
import argparse
import json

import _bootstrap  # noqa: F401
from app.db.database import SessionLocal, init_db
from app.db import models as dbm
from app.main import seed_initial_data
from app.services.training import run_evaluation_job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--max-images", type=int, default=None)
    a = ap.parse_args()
    init_db(); seed_initial_data()
    s = SessionLocal()
    model = s.query(dbm.ModelRegistry).filter(dbm.ModelRegistry.name == a.model).first()
    if not model:
        raise SystemExit(f"Unknown model {a.model}")
    job = dbm.EvaluationJob(model_id=model.id, dataset_name=a.dataset, dataset_split=a.split)
    s.add(job); s.commit()
    run_evaluation_job(job.id, a.max_images)
    s.expire_all()
    job = s.get(dbm.EvaluationJob, job.id)
    print(json.dumps({"status": job.status, "error": job.error_message, "num_images": job.num_images,
                      "precision": job.precision, "recall": job.recall, "map50": job.map50,
                      "map50_95": job.map50_95, "avg_inference_ms": job.avg_inference_ms,
                      "model_status_at_eval": job.model_status_at_eval, "warnings": job.warnings,
                      "report": job.report_path}, indent=2))


if __name__ == "__main__":
    main()
