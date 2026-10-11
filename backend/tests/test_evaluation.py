"""Detection-metric and radar-evaluation tests with hand-computable expected values."""
from datetime import datetime, timedelta

import pytest

from app.services.evaluation import compute_detection_metrics, box_iou
from app.services.radar_tracking import RadarTracker
import numpy as np


def test_box_iou():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[5, 0, 15, 10]], float)
    assert box_iou(a, b)[0, 0] == pytest.approx(50 / 150)


def test_perfect_predictions_give_map_one():
    images = [{"gt": [(0, 10, 10, 50, 50), (1, 60, 60, 90, 90)],
               "pred": [(0, 0.9, 10, 10, 50, 50), (1, 0.8, 60, 60, 90, 90)]}]
    m = compute_detection_metrics(images, ["a", "b"])
    assert m["map50"] == pytest.approx(1.0)
    assert m["map50_95"] == pytest.approx(1.0)
    assert m["precision"] == 1.0 and m["recall"] == 1.0


def test_false_positive_and_miss():
    images = [{"gt": [(0, 10, 10, 50, 50), (0, 100, 100, 140, 140)],
               "pred": [(0, 0.9, 10, 10, 50, 50), (0, 0.8, 300, 300, 340, 340)]}]
    m = compute_detection_metrics(images, ["a"])
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["map50"] == pytest.approx(0.5, abs=0.01)
    cm = m["confusion_matrix"]["matrix"]
    assert cm[0][0] == 1          # TP
    assert cm[0][1] == 1          # FP vs background
    assert cm[1][0] == 1          # missed GT


def test_wrong_class_is_not_true_positive():
    images = [{"gt": [(0, 10, 10, 50, 50)], "pred": [(1, 0.9, 10, 10, 50, 50)]}]
    m = compute_detection_metrics(images, ["a", "b"])
    assert m["map50"] == 0.0
    assert m["confusion_matrix"]["matrix"][1][0] == 1  # predicted b, true a


def test_no_ground_truth_gives_no_map():
    m = compute_detection_metrics([{"gt": [], "pred": []}], ["a"])
    assert m["map50"] is None and m["recall"] is None


def _two_targets(with_truth=True):
    base = datetime(2024, 1, 1)
    dets = []
    for i in range(20):
        t = base + timedelta(seconds=i)
        for tid, (x0, vx) in {"A": (0, 5), "B": (1000, -5)}.items():
            d = {"timestamp": t, "x": x0 + vx * i, "y": 0.0}
            if with_truth:
                d.update({"truth_id": tid, "truth_x": x0 + vx * i, "truth_y": 0.0})
            dets.append(d)
    return dets


def test_radar_evaluation_with_reference():
    tr = RadarTracker(association_threshold=50)
    tr.process_detections(_two_targets())
    ev = tr.evaluate()
    assert ev["reference_targets"] == 2
    assert ev["association_accuracy"] == 1.0
    assert ev["fragmentation"] == 0.0
    assert ev["id_switches"] == 0
    assert ev["position_rmse"] < 1.0
    assert "association_accuracy" in ev["metrics_available"]


def test_radar_evaluation_without_reference_claims_nothing():
    tr = RadarTracker(association_threshold=50)
    tr.process_detections(_two_targets(with_truth=False))
    ev = tr.evaluate()
    assert "association_accuracy" not in ev
    assert "position_rmse" not in ev
    assert ev["metrics_available"] == []
    assert "no tracking accuracy is claimed" in ev["note"]


def test_radar_geographic_input_projected_to_metres(tmp_path):
    p = tmp_path / "geo.csv"
    rows = ["timestamp,latitude,longitude,mmsi"]
    for i in range(10):
        rows.append(f"2024-01-01T00:00:{i:02d}Z,54.0,{10.0 + i * 0.0001:.6f},111")
    p.write_text("\n".join(rows))
    tr = RadarTracker(association_threshold=50)
    dets = tr.parse_csv_data(str(p))
    assert tr.coordinate_mode == "geographic_local_enu"
    assert dets[1]["x"] == pytest.approx(6.54, abs=0.1)  # 0.0001 deg lon at 54N ≈ 6.5 m
    tr.process_detections(dets)
    assert len(tr.get_tracks()) == 1
    assert tr.evaluate()["association_accuracy"] == 1.0


def test_radar_missing_records_reported(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("timestamp,x,y\n2024-01-01T00:00:00,1,2\n,3,4\n2024-01-01T00:00:02,abc,5\n")
    tr = RadarTracker()
    dets = tr.parse_csv_data(str(p))
    assert len(dets) == 1
    assert any("2 invalid" in w for w in tr.warnings)
