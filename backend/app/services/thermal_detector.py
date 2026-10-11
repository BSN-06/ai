"""
Thermal / infrared image object detection service using Ultralytics YOLO.

Architecture decision:
- Separate YOLO model instance for thermal imagery.
- The FLIR ADAS dataset contains thermal images with classes like person, car, bicycle, dog, etc.
- This is automotive thermal data — it does NOT prove military-target detection capability.
- The thermal model has its own checkpoint, class mapping, and evaluation.
- YOLOv8n is the default variant for CPU-friendly inference.
"""
import logging
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ThermalDetector:
    """YOLO-based object detector for thermal/infrared imagery."""

    # FLIR ADAS typical classes (actual classes depend on dataset version)
    FLIR_CLASSES = ["person", "car", "bicycle", "dog", "other_vehicle"]

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
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self.device = "cpu"

    def load_model(self, checkpoint_path: Optional[str] = None, variant: str = "yolov8n.pt") -> dict:
        """
        Load a YOLO model for thermal image detection.
        
        Args:
            checkpoint_path: Path to a fine-tuned thermal checkpoint.
            variant: YOLO model variant.
            
        Returns:
            Model loading status and metadata.
        """
        try:
            from ultralytics import YOLO

            model_path = checkpoint_path if checkpoint_path and Path(checkpoint_path).exists() else variant
            is_pretrained = checkpoint_path is None or not Path(str(checkpoint_path)).exists()

            self.model = YOLO(model_path)
            self.model_name = f"thermal_{variant}" if is_pretrained else Path(checkpoint_path).stem
            self.checkpoint_path = str(model_path)
            self.class_names = list(self.model.names.values()) if hasattr(self.model, 'names') and self.model.names else []
            self.is_loaded = True

            status = "PRETRAINED" if is_pretrained else "FINE_TUNED"
            note = (
                "Using COCO-pretrained weights. Fine-tune on FLIR ADAS thermal dataset "
                "for thermal-specific detection. Current class mapping is from COCO, "
                "not thermal-specific classes."
            ) if is_pretrained else "Fine-tuned thermal checkpoint loaded."

            return {
                "status": "loaded",
                "model_status": status,
                "model_name": self.model_name,
                "variant": variant,
                "checkpoint": str(model_path),
                "device": self.device,
                "num_classes": len(self.class_names),
                "class_names": self.class_names[:20],
                "architecture": {
                    "backbone": "CSPDarknet",
                    "neck": "PAN-FPN",
                    "head": "Decoupled anchor-free detection head",
                },
                "note": note,
                "domain_warning": (
                    "This model is designed for thermal/infrared imagery. "
                    "Automotive thermal data does NOT prove military-target detection capability."
                ),
            }
        except ImportError:
            return {"status": "error", "message": "Ultralytics YOLO not installed."}
        except Exception as e:
            logger.error(f"Failed to load thermal model: {e}")
            return {"status": "error", "message": str(e)}

    def run_inference(
        self,
        image_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> dict:
        """
        Run thermal image detection.
        
        Note: If a visible-light image is passed instead of a thermal image,
        detection may still produce results (since the model can process any image),
        but the results should not be treated as thermal detection performance.
        """
        if not self.is_loaded or self.model is None:
            return {"status": "error", "message": "Thermal model not loaded."}

        if not Path(image_path).exists():
            return {"status": "error", "message": f"Image not found: {image_path}"}

        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"status": "error", "message": f"Failed to read image: {image_path}"}

            h, w = img.shape[:2]

            # Simple heuristic to detect if image might be thermal (grayscale-dominant)
            is_likely_thermal = self._check_thermal_characteristics(img)

            start_time = time.time()
            results = self.model(image_path, conf=confidence_threshold, iou=iou_threshold,
                                device=self.device, verbose=False)
            inference_time_ms = (time.time() - start_time) * 1000

            detections = self._parse_results(results)
            annotated_path = self._draw_detections(image_path, detections)

            result = {
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
                "device": self.device,
                "timestamp": datetime.utcnow().isoformat(),
            }

            if not is_likely_thermal:
                result["warning"] = (
                    "The uploaded image does not appear to be a thermal/infrared image "
                    "(it has significant color variation). Detection results from a visible-light "
                    "image should not be reported as thermal detection performance."
                )

            return result
        except Exception as e:
            logger.error(f"Thermal inference failed: {e}")
            return {"status": "error", "message": str(e)}

    def _check_thermal_characteristics(self, img: np.ndarray) -> bool:
        """
        Simple heuristic to check if an image has thermal-like characteristics.
        Thermal images typically have low color saturation (appear grayscale or single-channel tinted).
        This is NOT a reliable classifier — it's a warning heuristic only.
        """
        if len(img.shape) == 2:
            return True  # Grayscale
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_saturation = np.mean(hsv[:, :, 1])
        return mean_saturation < 40  # Low saturation suggests thermal-like

    def _parse_results(self, results) -> list[dict]:
        """Parse YOLO results into detection dicts."""
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
        """Draw bounding boxes on thermal image."""
        img = cv2.imread(image_path)
        if img is None:
            return image_path

        for det in detections:
            x1, y1, x2, y2 = int(det["bbox_x1"]), int(det["bbox_y1"]), int(det["bbox_x2"]), int(det["bbox_y2"])
            # Use bright green for visibility on thermal images
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f'{det["class_name"]} {det["confidence"]:.2f}'
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - label_h - 4), (x1 + label_w, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        p = Path(image_path)
        annotated_path = str(p.parent / f"{p.stem}_thermal_annotated{p.suffix}")
        cv2.imwrite(annotated_path, img)
        return annotated_path

    def get_status(self) -> dict:
        return {
            "loaded": self.is_loaded,
            "model_name": self.model_name,
            "checkpoint": self.checkpoint_path,
            "device": self.device,
            "num_classes": len(self.class_names),
        }


# Singleton instance
thermal_detector = ThermalDetector()
