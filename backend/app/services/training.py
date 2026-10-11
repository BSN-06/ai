"""
Background training and evaluation jobs for the two image detectors.

- Jobs run in a worker thread, outside the HTTP request; status, per-epoch losses/metrics,
  logs and the configuration are persisted in the database and in the run directory.
- Only one training job runs at a time (a second request is rejected with 409).
- Cancellation is cooperative: the flag is checked after every training batch.
- A model is marked TRAINED only if training completed AND ``weights/best.pt`` exists.
- After training, the best checkpoint is evaluated on the held-out split (test, else val)
  with ``app.services.evaluation`` and a report is saved.
"""
from __future__ import annotations

import json
import logging
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.core.config import settings, BASE_DIR
from app.db import models as dbm

logger = logging.getLogger(__name__)

RUNS_DIR = BASE_DIR / "data" / "training_runs"
_lock = threading.Lock()


class JobCancelled(Exception):
    pass


def _session():
    from app.db.database import SessionLocal
    return SessionLocal()


def _log(run_dir: Path, msg: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "job.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()}Z {msg}\n")


def preflight(dataset: Optional[dbm.Dataset], device: str) -> list[str]:
    """Return blocking problems that prevent training (empty list = OK)."""
    problems = []
    if dataset is None:
        problems.append("Dataset is not registered.")
    elif not dataset.local_path:
        problems.append(f"Dataset '{dataset.name}' has no local path. Convert/register it first.")
    elif dataset.validation_status != "VALIDATED":
        problems.append(f"Dataset '{dataset.name}' is {dataset.validation_status}; it must be VALIDATED.")
    elif not (Path(dataset.local_path) / "dataset.yaml").exists():
        problems.append("dataset.yaml missing in dataset path.")
    try:
        import ultralytics  # noqa: F401
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            problems.append("CUDA requested but not available on this machine. Use device=cpu.")
    except ImportError as e:
        problems.append(f"Missing dependency: {e}. Install ultralytics and torch.")
    return problems


def run_training_job(job_id: str):
    """Worker entry point (runs in a background thread)."""
    if not _lock.acquire(blocking=False):
        s = _session()
        job = s.get(dbm.TrainingJob, job_id)
        job.status, job.error_message = "FAILED", "Another training job is running."
        s.commit(); s.close()
        return
    session = _session()
    run_dir = None
    try:
        job = session.get(dbm.TrainingJob, job_id)
        model = session.get(dbm.ModelRegistry, job.model_id)
        dataset = session.query(dbm.Dataset).filter(dbm.Dataset.name == job.dataset_name).first()
        run_name = f"{model.name}_{job.id[:8]}"
        run_dir = RUNS_DIR / run_name
        job.log_path = str(run_dir / "job.log")
        job.status, job.started_at = "RUNNING", datetime.utcnow()
        job.epoch_history = []
        model.status = "TRAINING"
        session.commit()
        _log(run_dir, f"Job {job.id} started for model {model.name} on dataset {job.dataset_name}")

        problems = preflight(dataset, job.device)
        if problems:
            raise RuntimeError("; ".join(problems))

        from ultralytics import YOLO
        from app.services.detector_base import pretrained_weights_path
        base = pretrained_weights_path(model.variant or "yolov8n.pt")
        yolo = YOLO(str(base))
        data_yaml = str(Path(dataset.local_path) / "dataset.yaml")
        config = {
            "data": data_yaml, "epochs": job.epochs, "batch": job.batch_size, "imgsz": job.image_size,
            "device": job.device, "project": str(RUNS_DIR), "name": run_name, "exist_ok": True,
            "pretrained": True, "verbose": False, "plots": False, "workers": 0,
        }
        if job.learning_rate:
            config["lr0"] = job.learning_rate
        job.config = {**config, "base_weights": str(base), "model_variant": model.variant}
        session.commit()
        with open(run_dir / "job_config.json", "w") as f:
            json.dump(job.config, f, indent=2)

        def on_batch_end(trainer):
            s = _session()
            try:
                if s.get(dbm.TrainingJob, job_id).cancel_requested:
                    raise JobCancelled()
            finally:
                s.close()

        def on_fit_epoch_end(trainer):
            s = _session()
            try:
                j = s.get(dbm.TrainingJob, job_id)
                losses = trainer.label_loss_items(trainer.tloss, prefix="train") if trainer.tloss is not None else {}
                metrics = {k: float(v) for k, v in (trainer.metrics or {}).items()}
                entry = {"epoch": trainer.epoch + 1, **{k: round(float(v), 5) for k, v in losses.items()},
                         **{k: round(v, 5) for k, v in metrics.items()}}
                hist = list(j.epoch_history or [])
                hist.append(entry)
                j.epoch_history = hist
                j.current_epoch = trainer.epoch + 1
                j.train_loss = round(sum(float(v) for v in losses.values()), 5) if losses else None
                vl = [v for k, v in metrics.items() if k.startswith("val/") and k.endswith("loss")]
                j.val_loss = round(sum(vl), 5) if vl else None
                m50 = metrics.get("metrics/mAP50(B)")
                if m50 is not None:
                    j.best_map50 = max(j.best_map50 or 0.0, m50)
                s.commit()
                _log(run_dir, f"epoch {entry['epoch']}/{job.epochs}: {entry}")
            finally:
                s.close()

        yolo.add_callback("on_train_batch_end", on_batch_end)
        yolo.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        yolo.train(**config)

        session.expire_all()
        job = session.get(dbm.TrainingJob, job_id)
        model = session.get(dbm.ModelRegistry, job.model_id)
        best = run_dir / "weights" / "best.pt"
        if not best.exists():
            raise RuntimeError("Training finished but weights/best.pt was not produced.")
        job.status, job.completed_at, job.checkpoint_path = "COMPLETED", datetime.utcnow(), str(best)
        model.checkpoint_path, model.trained, model.status = str(best), True, "TRAINED"
        model.last_training_time = datetime.utcnow()
        model.dataset_name = job.dataset_name
        model.class_names = list(dataset.classes or [])
        model.num_classes = len(model.class_names)
        session.commit()
        _log(run_dir, f"Training completed. Checkpoint: {best}")

        from app.services.reporting import save_report
        save_report(session, report_type="training", test_mode="training", source=job.dataset_name,
                    model_checkpoint=str(best), num_samples=dataset.train_count or 0,
                    metrics={"epochs": job.current_epoch, "best_map50_val": job.best_map50,
                             "final_train_loss": job.train_loss, "final_val_loss": job.val_loss},
                    body={"job_id": job.id, "config": job.config, "epoch_history": job.epoch_history},
                    limitations=["Validation mAP during training is computed by Ultralytics on the val split; "
                                 "see the separate held-out evaluation report."])

        # Held-out evaluation of the new checkpoint
        split = "test" if (dataset.test_count or 0) > 0 else "val"
        ev = dbm.EvaluationJob(model_id=model.id, dataset_name=dataset.name, dataset_split=split, status="PENDING")
        session.add(ev); session.commit()
        _log(run_dir, f"Starting held-out evaluation on '{split}' (job {ev.id})")
        run_evaluation_job(ev.id)
    except JobCancelled:
        _finish_failed(session, job_id, "CANCELLED", "Cancelled by user.", run_dir)
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")
        _finish_failed(session, job_id, "FAILED", f"{e}", run_dir, traceback.format_exc())
    finally:
        session.close()
        _lock.release()


def _finish_failed(session, job_id, status, msg, run_dir, tb: str = ""):
    session.rollback()
    session.expire_all()
    job = session.get(dbm.TrainingJob, job_id)
    model = session.get(dbm.ModelRegistry, job.model_id)
    job.status, job.error_message, job.completed_at = status, msg, datetime.utcnow()
    model.status = "TRAINED" if model.trained else "PRETRAINED"
    session.commit()
    if run_dir:
        _log(run_dir, f"{status}: {msg}\n{tb}")


def run_evaluation_job(job_id: str, max_images: Optional[int] = None, conf: float = 0.001):
    """
    Evaluate a model checkpoint on a labelled split with our own metric implementation.

    conf=0.001 is used for mAP (standard practice: score the full PR curve); precision/recall are
    reported at the configured default confidence threshold.
    """
    from app.services.evaluation import evaluate_detector_on_split
    from app.services.detector_base import YoloDetector
    from app.services.reporting import save_report
    import yaml

    session = _session()
    try:
        job = session.get(dbm.EvaluationJob, job_id)
        model = session.get(dbm.ModelRegistry, job.model_id)
        dataset = session.query(dbm.Dataset).filter(dbm.Dataset.name == job.dataset_name).first()
        job.status, job.started_at = "RUNNING", datetime.utcnow()
        session.commit()

        if not dataset or not dataset.local_path or dataset.validation_status != "VALIDATED":
            raise RuntimeError(f"Dataset '{job.dataset_name}' is not installed/validated — evaluation impossible.")
        root = Path(dataset.local_path)
        with open(root / "dataset.yaml", encoding="utf-8") as f:
            names = yaml.safe_load(f)["names"]
        ds_names = [names[k] for k in sorted(names, key=int)] if isinstance(names, dict) else list(names)

        det = YoloDetector()
        det.sensor_type = model.sensor_type
        info = det.load_model(checkpoint_path=model.checkpoint_path, variant=model.variant or "yolov8n.pt",
                              allow_pretrained=not model.checkpoint_path)
        if info["status"] != "loaded":
            raise RuntimeError(info.get("message"))
        warnings = []
        if det.class_names != ds_names:
            warnings.append(
                f"Model class mapping ({len(det.class_names)} classes, e.g. {det.class_names[:3]}) differs from "
                f"dataset mapping ({len(ds_names)} classes, e.g. {ds_names[:3]}). Metrics for a PRETRAINED "
                "COCO model on this dataset are not meaningful.")

        def progress(i, n):
            if i % 25 == 0 or i == n:
                s = _session()
                j = s.get(dbm.EvaluationJob, job_id)
                j.num_images = i
                s.commit(); s.close()

        metrics = evaluate_detector_on_split(
            lambda p: det.predict(p, conf=conf, iou=settings.DEFAULT_IOU_THRESHOLD),
            root, job.dataset_split, ds_names, conf_threshold=settings.DEFAULT_CONFIDENCE_THRESHOLD,
            max_images=max_images, progress=progress)

        session.expire_all()
        job = session.get(dbm.EvaluationJob, job_id)
        model = session.get(dbm.ModelRegistry, job.model_id)
        job.precision, job.recall = metrics["precision"], metrics["recall"]
        job.map50, job.map50_95 = metrics["map50"], metrics["map50_95"]
        job.per_class_metrics, job.confusion_matrix = metrics["per_class"], metrics["confusion_matrix"]
        job.avg_inference_ms, job.num_images = metrics["avg_inference_ms"], metrics["num_images"]
        job.checkpoint_path, job.model_status_at_eval = det.checkpoint_path, det.model_status
        job.warnings = warnings
        job.status, job.completed_at = "COMPLETED", datetime.utcnow()
        model.evaluation_status = "EVALUATED"
        model.metrics = {"map50": job.map50, "map50_95": job.map50_95, "precision": job.precision,
                         "recall": job.recall, "avg_inference_ms": job.avg_inference_ms,
                         "dataset": dataset.name, "split": job.dataset_split, "num_images": job.num_images,
                         "model_status_at_eval": det.model_status, "evaluated_at": job.completed_at.isoformat()}
        session.commit()
        rep = save_report(
            session, report_type=f"{model.sensor_type}_detection", test_mode="real_evaluation",
            source=f"{dataset.name}/{job.dataset_split}", model_checkpoint=det.checkpoint_path,
            num_samples=metrics["num_images"],
            metrics={k: metrics[k] for k in ("precision", "recall", "map50", "map50_95", "avg_inference_ms")},
            warnings=warnings,
            limitations=[f"Model status at evaluation: {det.model_status}.",
                         "Metrics apply only to this dataset split and domain."],
            body={"evaluation_job_id": job.id, "model": model.name, **metrics})
        job.report_path = rep.file_path
        session.commit()
    except Exception as e:
        logger.error(f"Evaluation job {job_id} failed: {e}")
        session.rollback()
        job = session.get(dbm.EvaluationJob, job_id)
        job.status, job.error_message, job.completed_at = "FAILED", str(e), datetime.utcnow()
        model = session.get(dbm.ModelRegistry, job.model_id)
        if model and model.evaluation_status != "EVALUATED":
            model.evaluation_status = "FAILED"
        session.commit()
    finally:
        session.close()


def start_in_background(target, *args) -> threading.Thread:
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()
    return t
