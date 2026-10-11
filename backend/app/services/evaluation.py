"""
Detection evaluation against ground-truth labels.

Metrics are computed here from actual predictions and reference boxes (no values are
taken from anywhere else):
- precision / recall at the chosen confidence threshold and IoU 0.5
- AP per class at IoU 0.50:0.05:0.95 (COCO-style 101-point interpolation)
- mAP@0.5 and mAP@0.5:0.95 (mean over classes that have ground truth)
- confusion matrix at IoU 0.5 (last row/column = background)
- mean inference latency per image

``evaluate_detector_on_split`` runs a loaded detector on every image of a YOLO split.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from app.services.datasets import IMAGE_EXTS

logger = logging.getLogger(__name__)

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (N,4) and b (M,4) in xyxy format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO 101-point interpolated AP: mean over recall thresholds of the max precision at recall >= t."""
    if len(recall) == 0:
        return 0.0
    envelope = np.flip(np.maximum.accumulate(np.flip(precision)))
    thresholds = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, thresholds, side="left")
    vals = np.array([envelope[i] if i < len(envelope) else 0.0 for i in idx])
    return float(vals.mean())


def compute_detection_metrics(
    images: list[dict],
    class_names: list[str],
    conf_threshold: float = 0.25,
) -> dict:
    """
    Args:
        images: [{"gt": [(cls, x1, y1, x2, y2), ...],
                  "pred": [(cls, conf, x1, y1, x2, y2), ...]}, ...]  (pixel or normalised, consistent)
        class_names: index -> name mapping of the model / dataset.
    """
    nc = len(class_names)
    tp_all, conf_all, cls_all = [], [], []
    n_gt = np.zeros(nc, dtype=int)
    confusion = np.zeros((nc + 1, nc + 1), dtype=int)  # rows = predicted, cols = true

    for im in images:
        gt = np.array(im.get("gt", []), dtype=float).reshape(-1, 5)
        pr = np.array(im.get("pred", []), dtype=float).reshape(-1, 6)
        for c in gt[:, 0].astype(int):
            if 0 <= c < nc:
                n_gt[c] += 1
        if len(pr):
            pr = pr[np.argsort(-pr[:, 1])]
        iou = box_iou(pr[:, 2:6], gt[:, 1:5]) if len(pr) and len(gt) else np.zeros((len(pr), len(gt)))
        tp = np.zeros((len(pr), len(IOU_THRESHOLDS)), dtype=bool)
        for ti, thr in enumerate(IOU_THRESHOLDS):
            matched = set()
            for pi in range(len(pr)):
                best, best_iou = -1, thr
                for gi in range(len(gt)):
                    if gi in matched or int(gt[gi, 0]) != int(pr[pi, 0]):
                        continue
                    if iou[pi, gi] >= best_iou:
                        best, best_iou = gi, iou[pi, gi]
                if best >= 0:
                    matched.add(best)
                    tp[pi, ti] = True
        tp_all.append(tp)
        conf_all.append(pr[:, 1] if len(pr) else np.zeros(0))
        cls_all.append(pr[:, 0].astype(int) if len(pr) else np.zeros(0, dtype=int))

        # Confusion matrix at IoU 0.5 for predictions above conf threshold (class-agnostic matching)
        keep = pr[pr[:, 1] >= conf_threshold] if len(pr) else pr
        kiou = box_iou(keep[:, 2:6], gt[:, 1:5]) if len(keep) and len(gt) else np.zeros((len(keep), len(gt)))
        used_gt = set()
        for pi in range(len(keep)):
            gi = int(np.argmax(kiou[pi])) if kiou.shape[1] else -1
            if gi >= 0 and kiou[pi, gi] >= 0.5 and gi not in used_gt:
                used_gt.add(gi)
                confusion[int(keep[pi, 0]), int(gt[gi, 0])] += 1
            else:
                confusion[int(keep[pi, 0]), nc] += 1
        for gi in range(len(gt)):
            if gi not in used_gt:
                confusion[nc, int(gt[gi, 0])] += 1

    tp = np.concatenate(tp_all) if tp_all else np.zeros((0, len(IOU_THRESHOLDS)), dtype=bool)
    conf = np.concatenate(conf_all) if conf_all else np.zeros(0)
    cls = np.concatenate(cls_all) if cls_all else np.zeros(0, dtype=int)

    per_class = {}
    aps50, aps = [], []
    tot_tp = tot_fp = 0
    for c in range(nc):
        mask = cls == c
        npos = n_gt[c]
        c_tp = tp[mask]
        c_conf = conf[mask]
        order = np.argsort(-c_conf)
        c_tp, c_conf = c_tp[order], c_conf[order]
        above = c_conf >= conf_threshold
        tp_at = int(c_tp[above, 0].sum())
        fp_at = int(above.sum() - tp_at)
        tot_tp += tp_at
        tot_fp += fp_at
        if npos == 0:
            if mask.any():
                per_class[class_names[c]] = {"gt_instances": 0, "predictions": int(mask.sum()),
                                             "note": "no ground truth; excluded from mAP"}
            continue
        ap_t = []
        for ti in range(len(IOU_THRESHOLDS)):
            tpc = np.cumsum(c_tp[:, ti])
            fpc = np.cumsum(~c_tp[:, ti])
            rec = tpc / npos
            prec = tpc / np.maximum(tpc + fpc, 1e-9)
            ap_t.append(_ap(rec, prec) if len(rec) else 0.0)
        aps50.append(ap_t[0])
        aps.append(float(np.mean(ap_t)))
        per_class[class_names[c]] = {
            "gt_instances": int(npos),
            "predictions": int(mask.sum()),
            "precision": round(tp_at / max(tp_at + fp_at, 1), 4) if (tp_at + fp_at) else 0.0,
            "recall": round(tp_at / npos, 4),
            "ap50": round(ap_t[0], 4),
            "ap50_95": round(float(np.mean(ap_t)), 4),
        }
    total_gt = int(n_gt.sum())
    return {
        "precision": round(tot_tp / max(tot_tp + tot_fp, 1), 4) if (tot_tp + tot_fp) else 0.0,
        "recall": round(tot_tp / total_gt, 4) if total_gt else None,
        "map50": round(float(np.mean(aps50)), 4) if aps50 else None,
        "map50_95": round(float(np.mean(aps)), 4) if aps else None,
        "classes_with_ground_truth": len(aps50),
        "ground_truth_instances": total_gt,
        "num_predictions": int(len(conf)),
        "per_class": per_class,
        "confusion_matrix": {"labels": class_names + ["background"], "matrix": confusion.tolist(),
                             "axes": "rows=predicted, cols=true"},
        "conf_threshold": conf_threshold,
        "iou_thresholds": [round(float(t), 2) for t in IOU_THRESHOLDS],
    }


def read_yolo_labels(label_path: Path, width: int, height: int) -> list[tuple]:
    """Read a YOLO label file and return pixel-space boxes [(cls, x1, y1, x2, y2)]."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().strip().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
        out.append((c, (cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height))
    return out


def evaluate_detector_on_split(
    predict_fn: Callable[[str], list[dict]],
    dataset_root: Path,
    split: str,
    class_names: list[str],
    conf_threshold: float = 0.25,
    max_images: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict:
    """
    Run ``predict_fn(image_path) -> [{class_id, confidence, bbox_x1..bbox_y2}]`` on each image
    of ``dataset_root/images/<split>`` and compare with ``labels/<split>``.

    The class mapping of the model MUST match the dataset's; mismatches are reported, not hidden.
    """
    import cv2

    img_dir = dataset_root / "images" / split
    if not img_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {img_dir}")
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if max_images:
        files = files[:max_images]
    if not files:
        raise ValueError(f"No images in {img_dir}")

    images, latencies = [], []
    for i, f in enumerate(files):
        if should_stop and should_stop():
            break
        im = cv2.imread(str(f))
        if im is None:
            continue
        h, w = im.shape[:2]
        gt = read_yolo_labels(dataset_root / "labels" / split / f"{f.stem}.txt", w, h)
        t0 = time.perf_counter()
        dets = predict_fn(str(f))
        latencies.append((time.perf_counter() - t0) * 1000)
        pred = [(d["class_id"], d["confidence"], d["bbox_x1"], d["bbox_y1"], d["bbox_x2"], d["bbox_y2"])
                for d in dets]
        images.append({"gt": gt, "pred": pred})
        if progress:
            progress(i + 1, len(files))

    metrics = compute_detection_metrics(images, class_names, conf_threshold)
    metrics["num_images"] = len(images)
    metrics["avg_inference_ms"] = round(float(np.mean(latencies)), 2) if latencies else None
    metrics["dataset_split"] = split
    return metrics
