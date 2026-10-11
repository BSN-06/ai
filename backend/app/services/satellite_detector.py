"""
Satellite image object detection service using Ultralytics YOLO.

Architecture decision:
- YOLOv8n (nano) is selected as the default for CPU-friendly inference and training.
- For satellite/aerial imagery, YOLOv8s or YOLOv8m may be preferable when GPU is available
  due to small object sizes in aerial views. The user can configure this.
- YOLOv8's architecture: CSPDarknet backbone, PAN-FPN neck, decoupled detection head.
- YOLO is NOT universally optimal for all aerial imagery tasks. For very small objects
  in high-resolution images, tiled inference is implemented to improve detection.
- The xView dataset contains 60 classes of objects in satellite imagery.
  Registration at challenge.xviewdataset.org is required for download.
"""
import logging
import time
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SatelliteDetector:
    """YOLO-based object detector for satellite/aerial imagery."""

    def __init__(self):
        self.model = None
        self.model_name: Optional[str] = None
        self.checkpoint_path: Optional[str] = None
        self.device: str = "cpu"
        self.class_names: list[str] = []
        self.is_loaded: bool = False
        self._detect_device()

    def _detect_device(self):
        """Detect available compute device."""
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                logger.info("No CUDA detected, using CPU")
        except ImportError:
            self.device = "cpu"
            logger.warning("PyTorch not available, using CPU")

    def load_model(self, checkpoint_path: Optional[str] = None, variant: str = "yolov8n.pt") -> dict:
        """
        Load a YOLO model for satellite detection.
        
        Args:
            checkpoint_path: Path to a fine-tuned checkpoint. If None, uses pretrained weights.
            variant: YOLO model variant (e.g., yolov8n.pt, yolov8s.pt).
            
        Returns:
            dict with model loading status and metadata.
        """
        try:
            from ultralytics import YOLO

            model_path = checkpoint_path if checkpoint_path and Path(checkpoint_path).exists() else variant
            is_pretrained = checkpoint_path is None or not Path(str(checkpoint_path)).exists()

            self.model = YOLO(model_path)
            self.model_name = variant if is_pretrained else Path(checkpoint_path).stem
            self.checkpoint_path = str(model_path)
            self.class_names = list(self.model.names.values()) if hasattr(self.model, 'names') and self.model.names else []
            self.is_loaded = True

            status = "PRETRAINED" if is_pretrained else "FINE_TUNED"
            if is_pretrained and checkpoint_path:
                logger.warning(f"Checkpoint {checkpoint_path} not found, falling back to pretrained {variant}")

            return {
                "status": "loaded",
                "model_status": status,
                "model_name": self.model_name,
                "variant": variant,
                "checkpoint": str(model_path),
                "device": self.device,
                "num_classes": len(self.class_names),
                "class_names": self.class_names[:20],  # Limit for display
                "architecture": {
                    "backbone": "CSPDarknet",
                    "neck": "PAN-FPN (Path Aggregation Network with Feature Pyramid)",
                    "head": "Decoupled detection head with anchor-free design",
                },
                "note": "Pretrained on COCO. Fine-tune on xView or DOTA for satellite-specific detection." if is_pretrained else "Fine-tuned checkpoint loaded."
            }
        except ImportError:
            logger.error("Ultralytics package not installed")
            return {"status": "error", "message": "Ultralytics YOLO not installed. Run: pip install ultralytics"}
        except Exception as e:
            logger.error(f"Failed to load satellite model: {e}")
            return {"status": "error", "message": str(e)}

    def run_inference(
        self,
        image_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        tile_size: Optional[int] = None,
        tile_overlap: int = 64,
    ) -> dict:
        """
        Run object detection on a satellite/aerial image.
        
        Supports optional tiled inference for large images where objects may be small
        relative to the full image resolution.
        
        Args:
            image_path: Path to the input image.
            confidence_threshold: Minimum detection confidence.
            iou_threshold: IoU threshold for NMS.
            tile_size: If set, use tiled inference with this tile dimension.
            tile_overlap: Pixel overlap between adjacent tiles.
            
        Returns:
            Detection results with bounding boxes, classes, and metadata.
        """
        if not self.is_loaded or self.model is None:
            return {"status": "error", "message": "Model not loaded. Call load_model() first."}

        if not Path(image_path).exists():
            return {"status": "error", "message": f"Image not found: {image_path}"}

        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"status": "error", "message": f"Failed to read image: {image_path}"}

            h, w = img.shape[:2]
            start_time = time.time()

            if tile_size and (w > tile_size * 1.5 or h > tile_size * 1.5):
                detections = self._tiled_inference(img, tile_size, tile_overlap, confidence_threshold, iou_threshold)
            else:
                detections = self._single_inference(image_path, confidence_threshold, iou_threshold)

            inference_time_ms = (time.time() - start_time) * 1000

            # Draw annotations
            annotated_path = self._draw_detections(image_path, detections)

            return {
                "status": "success",
                "image_path": image_path,
                "annotated_image_path": annotated_path,
                "image_width": w,
                "image_height": h,
                "num_detections": len(detections),
                "detections": detections,
                "inference_time_ms": round(inference_time_ms, 2),
                "model_name": self.model_name,
                "checkpoint": self.checkpoint_path,
                "confidence_threshold": confidence_threshold,
                "iou_threshold": iou_threshold,
                "device": self.device,
                "tiled": tile_size is not None,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {"status": "error", "message": str(e)}

    def _single_inference(self, image_path: str, conf: float, iou: float) -> list[dict]:
        """Run inference on a single image."""
        results = self.model(image_path, conf=conf, iou=iou, device=self.device, verbose=False)
        return self._parse_results(results)

    def _tiled_inference(self, img: np.ndarray, tile_size: int, overlap: int, conf: float, iou: float) -> list[dict]:
        """
        Run tiled inference on a large image.
        Splits the image into overlapping tiles, runs detection on each,
        and merges results with NMS to remove duplicates at tile boundaries.
        """
        h, w = img.shape[:2]
        stride = tile_size - overlap
        all_detections = []

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x2 = min(x + tile_size, w)
                y2 = min(y + tile_size, h)
                tile = img[y:y2, x:x2]

                # Save tile temporarily
                tile_path = f"/tmp/tile_{x}_{y}.jpg"
                cv2.imwrite(tile_path, tile)

                results = self.model(tile_path, conf=conf, iou=iou, device=self.device, verbose=False)
                tile_dets = self._parse_results(results)

                # Transform coordinates from tile space to original image space
                for det in tile_dets:
                    det["bbox_x1"] += x
                    det["bbox_y1"] += y
                    det["bbox_x2"] += x
                    det["bbox_y2"] += y
                    all_detections.append(det)

                # Cleanup
                Path(tile_path).unlink(missing_ok=True)

        # Apply NMS across all tiles to remove duplicate detections at boundaries
        if all_detections:
            all_detections = self._nms_across_tiles(all_detections, iou)

        return all_detections

    def _nms_across_tiles(self, detections: list[dict], iou_threshold: float) -> list[dict]:
        """Apply non-maximum suppression across tile boundaries."""
        if not detections:
            return detections

        boxes = np.array([[d["bbox_x1"], d["bbox_y1"], d["bbox_x2"], d["bbox_y2"]] for d in detections])
        scores = np.array([d["confidence"] for d in detections])

        # Simple NMS
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return [detections[i] for i in keep]

    def _parse_results(self, results) -> list[dict]:
        """Parse Ultralytics YOLO results into a list of detection dicts."""
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"

                    detections.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(conf, 4),
                        "bbox_x1": round(x1, 2),
                        "bbox_y1": round(y1, 2),
                        "bbox_x2": round(x2, 2),
                        "bbox_y2": round(y2, 2),
                    })
        return detections

    def _draw_detections(self, image_path: str, detections: list[dict]) -> str:
        """Draw bounding boxes and labels on the image and save the annotated version."""
        img = cv2.imread(image_path)
        if img is None:
            return image_path

        colors = {}
        for det in detections:
            cls_name = det["class_name"]
            if cls_name not in colors:
                # Generate deterministic color from class name
                hash_val = hash(cls_name) % 0xFFFFFF
                colors[cls_name] = (hash_val & 0xFF, (hash_val >> 8) & 0xFF, (hash_val >> 16) & 0xFF)

            color = colors[cls_name]
            x1, y1, x2, y2 = int(det["bbox_x1"]), int(det["bbox_y1"]), int(det["bbox_x2"]), int(det["bbox_y2"])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f'{cls_name} {det["confidence"]:.2f}'
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - label_h - 4), (x1 + label_w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save annotated image
        p = Path(image_path)
        annotated_path = str(p.parent / f"{p.stem}_annotated{p.suffix}")
        cv2.imwrite(annotated_path, img)
        return annotated_path

    def get_status(self) -> dict:
        """Get current model status."""
        return {
            "loaded": self.is_loaded,
            "model_name": self.model_name,
            "checkpoint": self.checkpoint_path,
            "device": self.device,
            "num_classes": len(self.class_names),
        }


# Singleton instance
satellite_detector = SatelliteDetector()
