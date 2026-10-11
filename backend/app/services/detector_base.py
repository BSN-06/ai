"""
Shared Ultralytics YOLO detector logic for the satellite (Module A) and thermal (Module B) detectors.

Each sensor has its own singleton subclass with its own checkpoint, class mapping and status;
only the mechanics (loading, inference, tiling, drawing) are shared.

Checkpoint policy
- If a specific checkpoint is requested and the file is missing -> error (no silent fallback).
- If no fine-tuned checkpoint exists, the COCO-pretrained base weights may be loaded and are
  always labelled PRETRAINED (COCO classes, not the target dataset's classes).
"""
from __future__ import annotations

import logging
import tempfile
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.core.config import settings, BASE_DIR

logger = logging.getLogger(__name__)

PRETRAINED_DIR = BASE_DIR / settings.CHECKPOINT_DIR / "pretrained"


def detect_device(preferred: Optional[str] = None) -> str:
    """Return 'cuda' or 'cpu' based on availability (never assumes CUDA)."""
    if settings.FORCE_CPU or preferred == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda" if preferred in (None, "cuda", "auto") else preferred
    except ImportError:
        pass
    return "cpu"


def pretrained_weights_path(variant: str) -> Path:
    """Location where COCO-pretrained base weights are cached (downloaded once by Ultralytics)."""
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    return PRETRAINED_DIR / Path(variant).name


class YoloDetector:
    sensor_type = "generic"
    annotated_suffix = "_annotated"

    def __init__(self):
        self.model = None
        self.model_name: Optional[str] = None
        self.checkpoint_path: Optional[str] = None
        self.model_status: Optional[str] = None  # PRETRAINED | FINE_TUNED
        self.class_names: list[str] = []
        self.is_loaded = False
        self.device = detect_device()

    # ── Loading ──
    def load_model(self, checkpoint_path: Optional[str] = None, variant: str = "yolov8n.pt",
                   allow_pretrained: bool = True) -> dict:
        try:
            from ultralytics import YOLO
        except ImportError:
            return {"status": "error", "message": "Ultralytics YOLO is not installed. Run: pip install ultralytics"}

        if checkpoint_path:
            if not Path(checkpoint_path).exists():
                return {"status": "error",
                        "message": f"Checkpoint not found: {checkpoint_path}. No prediction was produced."}
            model_path, status = Path(checkpoint_path), "FINE_TUNED"
        elif allow_pretrained:
            model_path, status = pretrained_weights_path(variant), "PRETRAINED"
        else:
            return {"status": "error", "message": "No fine-tuned checkpoint registered for this model."}

        if self.is_loaded and self.checkpoint_path == str(model_path):
            return self._load_info()
        try:
            self.model = YOLO(str(model_path))
        except Exception as e:
            logger.error(f"Failed to load {self.sensor_type} model: {e}")
            msg = str(e)
            if status == "PRETRAINED":
                msg += (" (pretrained weights are downloaded by Ultralytics on first use — internet access "
                        f"is required once, or place {Path(variant).name} in {PRETRAINED_DIR})")
            return {"status": "error", "message": msg}
        self.checkpoint_path = str(model_path)
        self.model_status = status
        self.model_name = f"{self.sensor_type}_{Path(variant).stem}" if status == "PRETRAINED" \
            else f"{self.sensor_type}_{model_path.parent.parent.name or model_path.stem}"
        names = getattr(self.model, "names", None) or {}
        self.class_names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
        self.is_loaded = True
        return self._load_info()

    def _load_info(self) -> dict:
        return {
            "status": "loaded",
            "model_status": self.model_status,
            "model_name": self.model_name,
            "checkpoint": self.checkpoint_path,
            "device": self.device,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
        }

    # ── Inference ──
    def predict(self, image_path: str, conf: float = 0.25, iou: float = 0.45,
                tile_size: Optional[int] = None, tile_overlap: int = 64) -> list[dict]:
        """Raw detections for one image (used by inference API and evaluation)."""
        if tile_size:
            img = cv2.imread(image_path)
            if img is not None and (img.shape[1] > tile_size or img.shape[0] > tile_size):
                return self._tiled_inference(img, tile_size, tile_overlap, conf, iou)
        return self._single_inference(image_path, conf, iou)

    def run_inference(self, image_path: str, confidence_threshold: float = 0.25, iou_threshold: float = 0.45,
                      tile_size: Optional[int] = None, tile_overlap: int = 64) -> dict:
        if not self.is_loaded or self.model is None:
            return {"status": "error", "message": "Model not loaded. Call load_model() first."}
        if not Path(image_path).exists():
            return {"status": "error", "message": f"Image not found: {image_path}"}
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"status": "error", "message": f"Failed to read image: {image_path}"}
            h, w = img.shape[:2]
            start = time.perf_counter()
            tiled = bool(tile_size and (w > tile_size or h > tile_size))
            detections = self.predict(image_path, confidence_threshold, iou_threshold,
                                      tile_size if tiled else None, tile_overlap)
            inference_ms = (time.perf_counter() - start) * 1000
            annotated = self._draw_detections(image_path, detections)
            result = {
                "status": "success",
                "sensor_type": self.sensor_type,
                "image_path": image_path,
                "annotated_image_path": annotated,
                "image_width": w,
                "image_height": h,
                "num_detections": len(detections),
                "detections": detections,
                "inference_time_ms": round(inference_ms, 2),
                "model_name": self.model_name,
                "model_status": self.model_status,
                "checkpoint": self.checkpoint_path,
                "class_names": self.class_names,
                "confidence_threshold": confidence_threshold,
                "iou_threshold": iou_threshold,
                "device": self.device,
                "tiled": tiled,
                "tile_size": tile_size if tiled else None,
                "timestamp": datetime.utcnow().isoformat(),
                "provenance": "real_inference",
            }
            if self.model_status == "PRETRAINED":
                result["model_note"] = ("PRETRAINED (COCO) weights — classes are COCO classes, not the target "
                                        "dataset's classes. Not fine-tuned.")
            return result
        except Exception as e:
            logger.error(f"{self.sensor_type} inference failed: {e}")
            return {"status": "error", "message": str(e)}

    def _single_inference(self, image_path, conf, iou) -> list[dict]:
        results = self.model(image_path, conf=conf, iou=iou, device=self.device, verbose=False)
        return self._parse_results(results)

    def _tiled_inference(self, img: np.ndarray, tile_size: int, overlap: int, conf: float, iou: float) -> list[dict]:
        """Overlapping tiles -> per-tile detection -> shift to image coordinates -> class-wise NMS."""
        h, w = img.shape[:2]
        stride = max(tile_size - overlap, 1)
        all_dets = []
        ys = list(range(0, max(h - overlap, 1), stride))
        xs = list(range(0, max(w - overlap, 1), stride))
        for y in ys:
            for x in xs:
                tile = img[y:y + tile_size, x:x + tile_size]
                results = self.model(tile, conf=conf, iou=iou, device=self.device, verbose=False)
                for det in self._parse_results(results):
                    det["bbox_x1"] = round(det["bbox_x1"] + x, 2)
                    det["bbox_y1"] = round(det["bbox_y1"] + y, 2)
                    det["bbox_x2"] = round(det["bbox_x2"] + x, 2)
                    det["bbox_y2"] = round(det["bbox_y2"] + y, 2)
                    all_dets.append(det)
        return self._nms_across_tiles(all_dets, iou) if all_dets else all_dets

    def _nms_across_tiles(self, detections: list[dict], iou_threshold: float) -> list[dict]:
        """Class-wise NMS to remove duplicates at tile overlaps."""
        if not detections:
            return detections
        keep_all = []
        by_cls: dict[int, list[int]] = {}
        for i, d in enumerate(detections):
            by_cls.setdefault(d.get("class_id", 0), []).append(i)
        for idxs in by_cls.values():
            boxes = np.array([[detections[i][k] for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")] for i in idxs])
            scores = np.array([detections[i]["confidence"] for i in idxs])
            x1, y1, x2, y2 = boxes.T
            areas = (x2 - x1) * (y2 - y1)
            order = scores.argsort()[::-1]
            while order.size > 0:
                k = order[0]
                keep_all.append(idxs[k])
                xx1 = np.maximum(x1[k], x1[order[1:]]); yy1 = np.maximum(y1[k], y1[order[1:]])
                xx2 = np.minimum(x2[k], x2[order[1:]]); yy2 = np.minimum(y2[k], y2[order[1:]])
                inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
                ov = inter / (areas[k] + areas[order[1:]] - inter + 1e-6)
                order = order[np.where(ov <= iou_threshold)[0] + 1]
        keep_all.sort(key=lambda i: -detections[i]["confidence"])
        return [detections[i] for i in keep_all]

    def _parse_results(self, results) -> list[dict]:
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                dets.append({
                    "class_id": int(k),
                    "class_name": self.class_names[k] if k < len(self.class_names) else f"class_{k}",
                    "confidence": round(float(c), 4),
                    "bbox_x1": round(float(x1), 2), "bbox_y1": round(float(y1), 2),
                    "bbox_x2": round(float(x2), 2), "bbox_y2": round(float(y2), 2),
                })
        return dets

    def _color(self, name: str) -> tuple:
        h = zlib.crc32(name.encode()) & 0xFFFFFF  # deterministic across runs
        return (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)

    def _draw_detections(self, image_path: str, detections: list[dict]) -> str:
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        for det in detections:
            color = self._color(det["class_name"])
            x1, y1, x2, y2 = (int(det[k]) for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f'{det["class_name"]} {det["confidence"]:.2f}'
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, max(y1 - lh - 4, 0)), (x1 + lw, max(y1, lh + 4)), color, -1)
            cv2.putText(img, label, (x1, max(y1 - 2, lh)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        p = Path(image_path)
        out = str(p.parent / f"{p.stem}{self.annotated_suffix}{p.suffix}")
        cv2.imwrite(out, img)
        return out

    def unload(self):
        self.model = None
        self.is_loaded = False
        self.checkpoint_path = None

    def get_status(self) -> dict:
        return {
            "loaded": self.is_loaded,
            "model_name": self.model_name,
            "model_status": self.model_status,
            "checkpoint": self.checkpoint_path,
            "device": self.device,
            "num_classes": len(self.class_names),
        }
