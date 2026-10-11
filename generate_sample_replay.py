"""
Generates a sample observation JSON file for testing the recorded-data replay feature.
Run: python generate_sample_replay.py
Output: sample_replay_observations.json
"""
import json
from datetime import datetime, timedelta
import uuid

base_time = datetime(2024, 6, 15, 10, 0, 0)
observations = []

sensors = [
    ('satellite', 'vehicle', 0.82, 34.0522, -118.2437),
    ('thermal',   'person',  0.71, 34.0525, -118.2440),
    ('radar',     None,      None, 34.0520, -118.2435),
    ('satellite', 'truck',   0.65, 34.0530, -118.2450),
    ('thermal',   'car',     0.88, 34.0515, -118.2430),
]

for i in range(30):
    sensor_type, obj_class, conf, lat, lon = sensors[i % len(sensors)]
    t = base_time + timedelta(minutes=i * 2)
    obs = {
        'id': str(uuid.uuid4()),
        'sensor_type': sensor_type,
        'observation_timestamp': t.isoformat(),
        'is_utc': True,
        'source_dataset': 'sample_replay',
        'object_class': obj_class,
        'confidence': conf,
        'latitude': round(lat + i * 0.0001, 6),
        'longitude': round(lon + i * 0.0001, 6),
        'coordinate_crs': 'WGS84',
        'data_provenance': 'recorded',
        'processing_status': 'PROCESSED',
    }
    if sensor_type == 'radar':
        obs['radar_track_id'] = f'T-{i:03d}'
        obs['position_x'] = 1000.0 + i * 5
        obs['position_y'] = 2000.0 + i * 3
        obs['velocity_x'] = 5.0
        obs['velocity_y'] = 2.5
        obs['uncertainty'] = 8.0 + i * 0.2
    observations.append(obs)

output = {'observations': observations}
with open('sample_replay_observations.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"Generated {len(observations)} observations in sample_replay_observations.json")
print("Upload this file on the Evaluation → Recorded Data Replay tab to test replay.")
