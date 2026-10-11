"""
Module A — Satellite / aerial image object detection (Ultralytics YOLO).

Model selection
- Default: YOLOv8n (``yolov8n.pt``), initialised from COCO-pretrained weights and fine-tuned on
  xView (tiled to 640 px chips). Chosen because it trains and runs on CPU; YOLOv8s/m is
  recommended when a GPU is available because xView objects are small.
- YOLOv8 architecture (as implemented in Ultralytics 8.x): CSPDarknet-style backbone with C2f
  blocks and SPPF; PAN-FPN multiscale neck (strides 8/16/32); decoupled, anchor-free detection
  head (separate classification and box-regression branches, DFL box loss).
- YOLO is not universally optimal: DOTA uses *oriented* boxes, for which an OBB model
  (e.g. ``yolov8n-obb.pt``) or a two-stage detector would be more appropriate. xView provides
  axis-aligned boxes, which is why an axis-aligned YOLO detector was selected.
- Large scenes are processed with tiled inference; boxes are mapped back from tile to
  original-image coordinates and merged with class-wise NMS.
"""
from app.services.detector_base import YoloDetector


class SatelliteDetector(YoloDetector):
    sensor_type = "satellite"
    annotated_suffix = "_annotated"


# Singleton instance
satellite_detector = SatelliteDetector()
