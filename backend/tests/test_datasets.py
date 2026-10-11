"""
Dataset conversion and validation tests.

Fixtures are tiny generated files that mimic the xView / COCO *formats* so the converters and the
validator can be tested without the real datasets. They are not real data and prove nothing about
model performance.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services import datasets as svc


def _img(path: Path, w=200, h=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w, 3), 90, np.uint8))


def _make_coco(root: Path):
    for split in ("images_thermal_train", "images_thermal_val", "images_rgb_train"):
        d = root / split
        _img(d / "data" / "a.jpg")
        _img(d / "data" / "b.jpg")
        coco = {
            "images": [{"id": 1, "file_name": "data/a.jpg", "width": 200, "height": 160},
                       {"id": 2, "file_name": "data/b.jpg", "width": 200, "height": 160}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 20, 50, 40], "iscrowd": 0},
                            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [100, 50, 30, 60], "iscrowd": 0}],
            "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "bike"}, {"id": 3, "name": "car"}],
        }
        (d / "coco.json").write_text(json.dumps(coco))


def test_coco_conversion_preserves_classes_and_excludes_rgb(tmp_path):
    src, out = tmp_path / "flir", tmp_path / "flir_yolo"
    _make_coco(src)
    res = svc.convert_coco_to_yolo(src, out, modality_filter="thermal")
    assert res["classes"] == ["person", "bike", "car"]           # ascending category id, names unchanged
    assert res["classes_without_instances"] == ["bike"]          # reported, not silently dropped
    assert res["images_per_split"] == {"train": 2, "val": 2}     # RGB folder excluded
    assert not any("rgb" in f["file"] for f in res["annotation_files"])
    lbl = (out / "labels" / "train" / "images_thermal_train_a.txt").read_text().split()
    assert lbl[0] == "2"  # car -> index 2
    assert float(lbl[1]) == pytest.approx((10 + 25) / 200, abs=1e-5)
    assert float(lbl[2]) == pytest.approx((20 + 20) / 160, abs=1e-5)
    rep = svc.validate_yolo_dataset(out)
    assert rep["status"] == "VALIDATED", rep["errors"]
    assert rep["splits"]["train"]["images"] == 2


def test_xview_conversion_tiles_and_maps_boxes(tmp_path):
    src, out = tmp_path / "xview", tmp_path / "xview_yolo"
    _img(src / "train_images" / "10.tif", w=1000, h=700)
    # type_id 18 -> class index 5 ("Small Car"); box spans x 100..140, y 600..640 (second tile row)
    gj = {"features": [
        {"properties": {"image_id": "10.tif", "type_id": 18, "bounds_imcoords": "100,600,140,640"}},
        {"properties": {"image_id": "10.tif", "type_id": 75, "bounds_imcoords": "5,5,20,20"}},  # unmapped id
    ]}
    (src / "xView_train.geojson").write_text(json.dumps(gj))
    res = svc.convert_xview_to_yolo(src, out, tile_size=512, overlap=64, val_frac=0.0, test_frac=0.0)
    assert res["skipped_type_ids"] == {"75": 1}
    assert res["class_instances"] == {"Small Car": 1}
    # tile origin (0, 448) contains the box fully -> local coords (100..140, 152..192) in a 512x252 tile
    lbl = (out / "labels" / "train" / "10_0_448.txt").read_text().split()
    assert lbl[0] == "5"
    assert float(lbl[1]) == pytest.approx(120 / 512, abs=1e-5)
    assert float(lbl[2]) == pytest.approx((172) / 252, abs=1e-5)
    assert len(res["classes"]) == 60


def test_validation_detects_bad_labels_and_leakage(tmp_path):
    root = tmp_path / "ds"
    for split in ("train", "val"):
        _img(root / "images" / split / "same.jpg")
        (root / "labels" / split).mkdir(parents=True)
    (root / "labels" / "train" / "same.txt").write_text("7 0.5 0.5 0.1 0.1\n")   # class 7 out of range
    (root / "labels" / "val" / "same.txt").write_text("0 1.5 0.5 0.1 0.1\n")     # coord > 1
    svc.write_dataset_yaml(root, ["a", "b"])
    rep = svc.validate_yolo_dataset(root)
    assert rep["status"] == "INVALID"
    assert any("leakage" in e.lower() for e in rep["errors"])
    assert rep["splits"]["train"]["invalid_label_lines"] == 1


def test_validation_raw_dataset_needs_conversion(tmp_path):
    src = tmp_path / "raw"
    _make_coco(src)
    rep = svc.validate_yolo_dataset(src)
    assert rep["status"] == "RAW_NEEDS_CONVERSION"


def test_missing_path_not_found(tmp_path):
    assert svc.validate_yolo_dataset(tmp_path / "nope")["status"] == "NOT_FOUND"


def test_dataset_path_outside_allowed_roots_rejected():
    with pytest.raises(ValueError):
        svc.resolve_dataset_path("/etc")


def test_dataset_api_rejects_arbitrary_path(client):
    ds = client.get("/api/datasets").json()[0]
    r = client.patch(f"/api/datasets/{ds['id']}/path", params={"local_path": "/etc"})
    assert r.status_code == 400


def test_dataset_validate_api_not_installed(client):
    ds = client.get("/api/datasets").json()[0]
    r = client.post(f"/api/datasets/validate/{ds['id']}")
    assert r.status_code == 200
    assert r.json()["status"] == "NOT_FOUND"
