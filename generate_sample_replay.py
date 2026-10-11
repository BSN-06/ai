"""
Generates a SIMULATED recorded-observation file for the Recorded-Data Replay mode.
It is NOT real sensor data; it only exercises the replay → validation → correlation → event pipeline.

Run:    python generate_sample_replay.py
Output: sample_replay_observations.json

Scenario: three sites several km apart. Every 10 minutes one site gets a short burst of
satellite / thermal / radar observations (within ~60 s and ~50 m), so each burst should form
its own event. A few records deliberately lack metadata (missing coordinates / timestamp)
to show how the pipeline reports them.
"""
import json
import random
import uuid
from datetime import datetime, timedelta

random.seed(7)
base = datetime(2024, 6, 15, 10, 0, 0)
sites = [(34.0522, -118.2437), (34.0900, -118.3000), (34.0100, -118.1900)]
obs = []

for burst in range(9):
    lat0, lon0 = sites[burst % 3]
    t0 = base + timedelta(minutes=10 * burst)
    sensors = ["satellite", "thermal", "radar"] if burst % 4 != 3 else ["satellite"]  # some bursts single-sensor
    for k, sensor in enumerate(sensors):
        t = t0 + timedelta(seconds=20 * k + random.randint(0, 5))
        rec = {
            "id": f"REC-{burst:02d}-{sensor[:3].upper()}-{uuid.uuid4().hex[:6]}",
            "sensor_type": sensor,
            "observation_timestamp": t.isoformat() + "Z",
            "source_dataset": "simulated_recording",
            "latitude": round(lat0 + random.uniform(-0.0003, 0.0003), 6),
            "longitude": round(lon0 + random.uniform(-0.0003, 0.0003), 6),
            "coordinate_crs": "WGS84",
            "data_provenance": "recorded",
        }
        if sensor in ("satellite", "thermal"):
            rec.update({"object_class": random.choice(["vehicle", "truck", "person"]),
                        "confidence": round(random.uniform(0.45, 0.95), 2), "model_name": "recorded_prediction"})
        else:
            rec.update({"radar_track_id": f"T-{burst:03d}", "uncertainty": round(random.uniform(4, 30), 1),
                        "position_x": round(random.uniform(-500, 500), 1), "position_y": round(random.uniform(-500, 500), 1)})
        obs.append(rec)

# Metadata problems the pipeline must report (not hide)
obs.append({"id": "REC-NOGEO", "sensor_type": "thermal", "observation_timestamp": (base + timedelta(minutes=33)).isoformat() + "Z",
            "object_class": "person", "confidence": 0.6, "data_provenance": "recorded"})
obs.append({"id": "REC-NOTS", "sensor_type": "satellite", "latitude": 34.05, "longitude": -118.24,
            "coordinate_crs": "WGS84", "data_provenance": "recorded"})

random.shuffle(obs)  # replay must re-order by recorded timestamp
with open("sample_replay_observations.json", "w") as f:
    json.dump({"observations": obs}, f, indent=2)
print(f"Generated {len(obs)} SIMULATED recorded observations in sample_replay_observations.json")
print("Load it in Evaluation → Recorded-data replay. One record has no timestamp and will be skipped.")
