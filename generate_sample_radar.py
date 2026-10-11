"""
Generates a SIMULATED radar detection CSV for exercising the radar import / tracking feature.
It is NOT real radar data and must not be used to claim tracking accuracy on real sensors.

Run:    python generate_sample_radar.py
Output: sample_radar_data.csv  (timestamp, x, y [metres], target_id, truth_x, truth_y)

target_id / truth_x / truth_y are the simulation's reference values, so the tracker's
association and position-error metrics can be demonstrated on the simulated scenario.
"""
import csv
import math
import random
from datetime import datetime, timedelta

random.seed(42)
base_time = datetime(2024, 6, 15, 10, 0, 0)
rows = []
NOISE_M = 2.0

for i in range(60):
    t = base_time + timedelta(seconds=i * 2)
    # Target 1: straight line north-east
    tx1, ty1 = 100.0 + i * 6.0, 200.0 + i * 4.0
    # Target 2: circular track (first 40 scans)
    angle = math.radians(i * 9)
    tx2, ty2 = 500.0 + 80.0 * math.cos(angle), 500.0 + 80.0 * math.sin(angle)
    targets = [("SIM-1", tx1, ty1)] + ([("SIM-2", tx2, ty2)] if i < 40 else [])
    for tid, tx, ty in targets:
        if random.random() < 0.05:  # 5 % missed detections
            continue
        rows.append({"timestamp": t.isoformat() + "Z",
                     "x": round(tx + random.gauss(0, NOISE_M), 2), "y": round(ty + random.gauss(0, NOISE_M), 2),
                     "target_id": tid, "truth_x": round(tx, 2), "truth_y": round(ty, 2)})

with open("sample_radar_data.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"Generated {len(rows)} SIMULATED detections in sample_radar_data.csv")
print("Upload it on the Radar Analysis page. Metrics refer only to this simulated scenario.")
