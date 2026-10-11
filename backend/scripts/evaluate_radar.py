"""
Run the Kalman/GNN tracker on a recorded radar CSV/JSON file and print the supported metrics.
Accuracy metrics are reported only when reference columns exist (truth_id / mmsi, truth_x / truth_y).

  python scripts/evaluate_radar.py --file data/datasets/dlr_radar/scenario1.csv [--gate 50] [--max-misses 5]
"""
import argparse
import json

import _bootstrap  # noqa: F401
from app.services.radar_tracking import RadarTracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--gate", type=float, default=50.0, help="Association gate (metres / input units)")
    ap.add_argument("--max-misses", type=int, default=5)
    a = ap.parse_args()
    tr = RadarTracker(association_threshold=a.gate, max_misses=a.max_misses)
    dets = tr.parse_csv_data(a.file) if a.file.lower().endswith(".csv") else tr.parse_json_data(a.file)
    print(json.dumps(tr.process_detections(dets), indent=2, default=str))
    print(json.dumps(tr.evaluate(), indent=2))
    if tr.warnings:
        print("Warnings:", tr.warnings)


if __name__ == "__main__":
    main()
