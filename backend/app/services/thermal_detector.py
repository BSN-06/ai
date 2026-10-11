"""
Module B — Thermal / infrared image object detection (separate Ultralytics YOLO model).

- Own checkpoint, class mapping (taken from the FLIR ADAS COCO categories during conversion),
  training job, evaluation report and status — independent of the satellite model.
- Default: YOLOv8n initialised from COCO weights, fine-tuned on FLIR ADAS *thermal* images only
  (RGB images in the package are excluded by the converter).
- FLIR ADAS is automotive data (person, car, bicycle, ... ). Results do NOT demonstrate
  aircraft or military-target detection performance.
- Uploaded images are checked for thermal-like characteristics; a colour (visible-light) image
  is processed but flagged with an explicit modality-mismatch warning.
"""
import cv2
import numpy as np

from app.services.detector_base import YoloDetector


class ThermalDetector(YoloDetector):
    sensor_type = "thermal"
    annotated_suffix = "_thermal_annotated"

    def run_inference(self, image_path: str, confidence_threshold: float = 0.25,
                      iou_threshold: float = 0.45, tile_size=None, tile_overlap: int = 64) -> dict:
        result = super().run_inference(image_path, confidence_threshold, iou_threshold, tile_size, tile_overlap)
        if result.get("status") != "success":
            return result
        img = cv2.imread(image_path)
        thermal_like, saturation = self._check_thermal_characteristics(img)
        result["modality_check"] = {"thermal_like": thermal_like, "mean_saturation": round(saturation, 2),
                                    "method": "HSV mean-saturation heuristic (< 40 = thermal-like)"}
        if not thermal_like:
            result["warning"] = (
                "MODALITY MISMATCH: the uploaded image has significant colour saturation and does not "
                "appear to be a thermal/infrared image. Results must not be reported as thermal detection."
            )
        return result

    def _check_thermal_characteristics(self, img: np.ndarray) -> tuple[bool, float]:
        """Heuristic only: thermal images are single-channel or low-saturation pseudo-colour."""
        if img is None:
            return False, 0.0
        if len(img.shape) == 2 or img.shape[2] == 1:
            return True, 0.0
        sat = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]))
        return sat < 40, sat


# Singleton instance
thermal_detector = ThermalDetector()
