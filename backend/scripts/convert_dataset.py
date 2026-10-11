"""
Convert a raw dataset to YOLO format, register it and validate it.

  python scripts/convert_dataset.py --dataset xView     --source data/datasets/xview_raw [--tile-size 640]
  python scripts/convert_dataset.py --dataset FLIR_ADAS --source data/datasets/flir_adas_raw
"""
import argparse
import json

import _bootstrap  # noqa: F401
from app.db.database import SessionLocal, init_db
from app.db import models as dbm
from app.main import seed_initial_data
from app.services import datasets as svc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Registered dataset name (xView, FLIR_ADAS, DOTA)")
    ap.add_argument("--source", required=True, help="Raw dataset folder (inside backend/data/datasets)")
    ap.add_argument("--tile-size", type=int, default=640)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--max-images", type=int, default=None, help="Limit source images (quick trial)")
    args = ap.parse_args()

    init_db(); seed_initial_data()
    s = SessionLocal()
    ds = s.query(dbm.Dataset).filter(dbm.Dataset.name == args.dataset).first()
    if not ds:
        raise SystemExit(f"Unknown dataset '{args.dataset}'")
    src = svc.resolve_dataset_path(args.source)
    info = svc.inspect_directory(src)
    print("Detected:", json.dumps(info, indent=2))
    out = svc.dataset_output_dir(ds.name)
    if info["format"] == "xview_geojson":
        res = svc.convert_xview_to_yolo(src, out, tile_size=args.tile_size, overlap=args.overlap,
                                        max_images=args.max_images, progress=print)
    elif info["format"] == "coco_json":
        res = svc.convert_coco_to_yolo(src, out, modality_filter="thermal" if ds.sensor_type == "thermal" else None,
                                       progress=print)
    else:
        raise SystemExit(f"Unsupported raw format: {info['format']}")
    print("Conversion:", json.dumps({k: v for k, v in res.items() if k != "classes"}, indent=2))
    rep = svc.validate_yolo_dataset(out)
    ds.local_path = str(out)
    ds.validation_status = rep["status"]
    splits = rep.get("splits", {})
    ds.train_count = splits.get("train", {}).get("images")
    ds.val_count = splits.get("val", {}).get("images")
    ds.test_count = splits.get("test", {}).get("images")
    ds.file_count = sum(v.get("images", 0) for v in splits.values())
    ds.classes, ds.num_classes = rep["classes"], len(rep["classes"] or [])
    ds.validation_message = " | ".join(rep["errors"] + rep["warnings"]) or "All integrity checks passed."
    s.commit()
    print(f"Validation: {rep['status']}  errors={rep['errors']}  warnings={rep['warnings']}")


if __name__ == "__main__":
    main()
