"""
Generates a sample radar detection CSV for testing the radar import feature.
Run: python generate_sample_radar.py
Output: sample_radar_data.csv
"""
import csv
from datetime import datetime, timedelta
import math

base_time = datetime(2024, 6, 15, 10, 0, 0)
rows = []

# Two simulated targets moving in different directions
for i in range(60):
    t = base_time + timedelta(seconds=i * 2)

    # Target 1: moving northeast
    x1 = 100.0 + i * 3.0 + (i % 3) * 0.5
    y1 = 200.0 + i * 2.0 + (i % 4) * 0.3
    rows.append({'timestamp': t.isoformat(), 'x': round(x1, 2), 'y': round(y1, 2)})

    # Target 2: moving in a circle (only first 40 steps)
    if i < 40:
        angle = math.radians(i * 9)
        x2 = 500.0 + 80.0 * math.cos(angle)
        y2 = 500.0 + 80.0 * math.sin(angle)
        rows.append({'timestamp': t.isoformat(), 'x': round(x2, 2), 'y': round(y2, 2)})

# Sort by timestamp
rows.sort(key=lambda r: r['timestamp'])

output_file = 'sample_radar_data.csv'
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['timestamp', 'x', 'y'])
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} detections in {output_file}")
print("Upload this file on the Radar Analysis page to test track processing.")
