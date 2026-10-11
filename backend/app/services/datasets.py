"""
Dataset discovery, annotation conversion and integrity validation.

Supported conversions (run explicitly — nothing is downloaded automatically):
- xView  (GeoJSON, pixel bounds in ``bounds_imcoords``)  -> YOLO, with image tiling.
- COCO JSON (FLIR ADAS v1/v2 thermal, or any COCO detection set) -> YOLO.

Validation works on the converted YOLO layout:
    <root>/dataset.yaml
    <root>/images/{train,val,test}/*.jpg|png|tif
    <root>/labels/{train,val,test}/*.txt   (class cx cy w h, normalised)

Split policy and limitations
- xView test labels are not public. The converter therefore splits the official *train*
  images into train/val/test by a deterministic hash of the image id, so tiles from one
  source image never appear in two splits (prevents tile-level leakage).
- FLIR ADAS ships its own train/val (and v2 video test) splits; those are preserved.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional

import yaml

from app.core.config import settings, BASE_DIR

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SPLITS = ("train", "val", "test")

# Official xView type_id -> contiguous class index (60 classes), as used by the Ultralytics
# xView conversion. -1 means the type id is not one of the 60 evaluated classes.
XVIEW_CLASS2INDEX = [
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, -1, 9, 10, 11,
    12, 13, 14, 15, -1, -1, 16, 17, 18, 19, 20, 21, 22, -1, 23, 24, 25, -1, 26, 27, -1, 28, -1,
    29, 30, 31, 32, 33, 34, 35, 36, 37, -1, 38, 39, 40, 41, 42, 43, 44, 45, -1, -1, -1, -1, 46,
    47, 48, 49, -1, 50, 51, -1, 52, -1, -1, -1, 53, 54, -1, 55, -1, -1, 56, -1, 57, -1, 58, 59,
]
XVIEW_CLASSES = [
    "Fixed-wing Aircraft", "Small Aircraft", "Cargo Plane", "Helicopter", "Passenger Vehicle",
    "Small Car", "Bus", "Pickup Truck", "Utility Truck", "Truck", "Cargo Truck", "Truck w/Box",
    "Truck Tractor", "Trailer", "Truck w/Flatbed", "Truck w/Liquid", "Crane Truck", "Railway Vehicle",
    "Passenger Car", "Cargo Car", "Flat Car", "Tank car", "Locomotive", "Maritime Vessel", "Motorboat",
    "Sailboat", "Tugboat", "Barge", "Fishing Vessel", "Ferry", "Yacht", "Container Ship", "Oil Tanker",
    "Engineering Vehicle", "Tower crane", "Container Crane", "Reach Stacker", "Straddle Carrier",
    "Mobile Crane", "Dump Truck", "Haul Truck", "Scraper/Tractor", "Front loader/Bulldozer", "Excavator",
    "Cement Mixer", "Ground Grader", "Hut/Tent", "Shed", "Building", "Aircraft Hangar",
    "Damaged Building", "Facility", "Construction Site", "Vehicle Lot", "Helipad", "Storage Tank",
    "Shipping container lot", "Shipping Container", "Pylon", "Tower",
]


# ─────────────────────────── Safe paths ───────────────────────────

def allowed_roots() -> list[Path]:
    """Dataset paths are restricted to the dataset dir plus roots listed in ALLOWED_DATASET_ROOTS."""
    roots = [(BASE_DIR / settings.DATASET_DIR).resolve()]
    extra = os.environ.get("ALLOWED_DATASET_ROOTS", "")
    for p in extra.split(os.pathsep):
        if p.strip():
            roots.append(Path(p.strip()).resolve())
    return roots


def resolve_dataset_path(path: str) -> Path:
    """Resolve a user-supplied dataset path and refuse anything outside the allowed roots."""
    p = Path(path)
    if not p.is_absolute():
        p = (BASE_DIR / settings.DATASET_DIR / p)
    p = p.resolve()
    for root in allowed_roots():
        if p == root or root in p.parents:
            return p
    raise ValueError(
        f"Path '{path}' is outside the allowed dataset roots {[str(r) for r in allowed_roots()]}. "
        "Place datasets under backend/data/datasets or add the folder to ALLOWED_DATASET_ROOTS."
    )


def _place(src: Path, dst: Path):
    """Hard-link when possible (no extra disk), else copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _hash_split(key: str, val_frac: float, test_frac: float) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def write_dataset_yaml(root: Path, names: list[str], extra: Optional[dict] = None) -> Path:
    cfg = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(names)},
    }
    if (root / "images" / "test").exists() and any((root / "images" / "test").iterdir()):
        cfg["test"] = "images/test"
    if extra:
        cfg.update(extra)
    out = root / "dataset.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out


# ─────────────────────────── Discovery ───────────────────────────

def inspect_directory(path: Path) -> dict:
    """Describe what a dataset directory contains (format detection, no modification)."""
    if not path.exists():
        return {"exists": False, "format": None}
    images = [p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    geojson = list(path.rglob("*.geojson"))
    coco = [p for p in path.rglob("*.json") if _looks_like_coco(p)]
    fmt = "unknown"
    if (path / "dataset.yaml").exists() and (path / "labels").exists():
        fmt = "yolo"
    elif geojson:
        fmt = "xview_geojson"
    elif coco:
        fmt = "coco_json"
    elif list(path.rglob("*.csv")):
        fmt = "csv"
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return {
        "exists": True,
        "format": fmt,
        "image_count": len(images),
        "geojson_files": [str(p) for p in geojson[:5]],
        "coco_files": [str(p) for p in coco[:10]],
        "size_bytes": size,
    }


def _looks_like_coco(p: Path) -> bool:
    if p.stat().st_size < 20:
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            head = f.read(4096)
        return '"images"' in head or '"annotations"' in head or '"categories"' in head or \
            p.name in ("coco.json", "thermal_annotations.json")
    except Exception:
        return False


# ─────────────────────────── COCO -> YOLO ───────────────────────────

def _split_from_name(path: Path) -> Optional[str]:
    s = path.name.lower()  # only the annotation file's own folder name decides the split
    if "test" in s:
        return "test"
    if "val" in s:
        return "val"
    if "train" in s:
        return "train"
    return None


def convert_coco_to_yolo(
    src: Path,
    out: Path,
    modality_filter: Optional[str] = "thermal",
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Convert every COCO annotation file found under ``src`` into one YOLO dataset at ``out``.

    - Split is taken from the annotation file's directory name (train/val/test).
    - Category ids are mapped to contiguous indices in ascending id order, using the union of
      categories across all files. Names are preserved exactly; no class is dropped or renamed.
    - ``modality_filter='thermal'`` skips annotation files whose path contains 'rgb' so that
      visible-light FLIR images are never mixed into the thermal dataset.
    """
    coco_files = [p for p in src.rglob("*.json") if _looks_like_coco(p)]
    if modality_filter == "thermal":
        coco_files = [p for p in coco_files if "rgb" not in str(p).lower()]
    if not coco_files:
        raise ValueError(f"No COCO annotation JSON found under {src}")

    loaded = []
    categories: dict[int, str] = {}
    for cf in coco_files:
        with open(cf, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "images" not in data:
            continue
        for c in data.get("categories", []):
            categories[int(c["id"])] = c["name"]
        loaded.append((cf, data))
    if not categories:
        raise ValueError("COCO files contain no categories.")

    cat_ids = sorted(categories)
    cat_map = {cid: i for i, cid in enumerate(cat_ids)}
    names = [categories[c] for c in cat_ids]

    stats = {"files": [], "images": Counter(), "boxes": Counter(), "missing_images": 0,
             "skipped_boxes": 0, "class_instances": Counter()}
    for cf, data in loaded:
        split = _split_from_name(cf.parent) or "train"
        anns = defaultdict(list)
        for a in data.get("annotations", []):
            anns[a["image_id"]].append(a)
        n_img = 0
        for img in data["images"]:
            fname = img["file_name"]
            cand = [cf.parent / fname, cf.parent / "data" / Path(fname).name,
                    cf.parent / "thermal_8_bit" / Path(fname).name]
            img_path = next((c for c in cand if c.exists()), None)
            if img_path is None:
                stats["missing_images"] += 1
                continue
            w, h = img.get("width"), img.get("height")
            if not w or not h:
                import cv2
                im = cv2.imread(str(img_path))
                if im is None:
                    stats["missing_images"] += 1
                    continue
                h, w = im.shape[:2]
            stem = f"{cf.parent.name}_{Path(fname).stem}".replace(" ", "_")
            _place(img_path, out / "images" / split / f"{stem}{img_path.suffix.lower()}")
            lines = []
            for a in anns.get(img["id"], []):
                if a.get("iscrowd"):
                    continue
                x, y, bw, bh = a["bbox"]
                if bw <= 1 or bh <= 1 or a["category_id"] not in cat_map:
                    stats["skipped_boxes"] += 1
                    continue
                cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
                lines.append(f"{cat_map[a['category_id']]} {min(max(cx,0),1):.6f} {min(max(cy,0),1):.6f} "
                             f"{min(bw / w, 1):.6f} {min(bh / h, 1):.6f}")
                stats["class_instances"][names[cat_map[a['category_id']]]] += 1
            lbl = out / "labels" / split / f"{stem}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text("\n".join(lines))
            stats["boxes"][split] += len(lines)
            n_img += 1
        stats["images"][split] += n_img
        stats["files"].append({"file": str(cf), "split": split, "images": n_img})
        if progress:
            progress(f"Converted {cf} -> {split}: {n_img} images")

    zero = [n for n in names if stats["class_instances"][n] == 0]
    write_dataset_yaml(out, names, {"source_format": "coco_json", "source_path": str(src)})
    return {
        "output": str(out),
        "classes": names,
        "classes_without_instances": zero,
        "images_per_split": dict(stats["images"]),
        "boxes_per_split": dict(stats["boxes"]),
        "class_instances": dict(stats["class_instances"]),
        "missing_images": stats["missing_images"],
        "skipped_boxes": stats["skipped_boxes"],
        "annotation_files": stats["files"],
    }


# ─────────────────────────── xView -> YOLO (tiled) ───────────────────────────

def convert_xview_to_yolo(
    src: Path,
    out: Path,
    tile_size: int = 640,
    overlap: int = 64,
    min_visible: float = 0.5,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    max_images: Optional[int] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Convert xView GeoJSON labels to tiled YOLO data.

    Boxes are transformed from full-image pixel coordinates into each tile's coordinates; a box is
    kept in a tile when at least ``min_visible`` of its area lies inside the tile (clipped).
    """
    import cv2

    geojsons = sorted(src.rglob("*.geojson"))
    if not geojsons:
        raise ValueError(f"No xView .geojson label file found under {src}")
    with open(geojsons[0], "r", encoding="utf-8") as f:
        gj = json.load(f)

    boxes = defaultdict(list)
    skipped_types = Counter()
    for feat in gj.get("features", []):
        p = feat.get("properties", {})
        try:
            tid = int(p["type_id"])
            x1, y1, x2, y2 = [float(v) for v in str(p["bounds_imcoords"]).split(",")]
        except (KeyError, ValueError):
            skipped_types["malformed"] += 1
            continue
        cls = XVIEW_CLASS2INDEX[tid] if 0 <= tid < len(XVIEW_CLASS2INDEX) else -1
        if cls < 0:
            skipped_types[tid] += 1
            continue
        if x2 > x1 and y2 > y1:
            boxes[p["image_id"]].append((cls, x1, y1, x2, y2))

    image_index = {p.name: p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    stats = {"tiles": Counter(), "boxes": Counter(), "source_images": Counter(),
             "missing_images": 0, "class_instances": Counter()}
    stride = tile_size - overlap
    ids = sorted(boxes)
    if max_images:
        ids = ids[:max_images]
    for n, image_id in enumerate(ids):
        img_path = image_index.get(image_id)
        if img_path is None:
            stats["missing_images"] += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            stats["missing_images"] += 1
            continue
        split = _hash_split(image_id, val_frac, test_frac)
        stats["source_images"][split] += 1
        H, W = img.shape[:2]
        ys = list(range(0, max(H - overlap, 1), stride))
        xs = list(range(0, max(W - overlap, 1), stride))
        for ty in ys:
            for tx in xs:
                tile = img[ty:ty + tile_size, tx:tx + tile_size]
                th, tw = tile.shape[:2]
                lines = []
                for cls, x1, y1, x2, y2 in boxes[image_id]:
                    cx1, cy1 = max(x1, tx), max(y1, ty)
                    cx2, cy2 = min(x2, tx + tw), min(y2, ty + th)
                    if cx2 <= cx1 or cy2 <= cy1:
                        continue
                    if (cx2 - cx1) * (cy2 - cy1) < min_visible * (x2 - x1) * (y2 - y1):
                        continue
                    lx1, ly1, lx2, ly2 = cx1 - tx, cy1 - ty, cx2 - tx, cy2 - ty
                    lines.append(f"{cls} {(lx1 + lx2) / 2 / tw:.6f} {(ly1 + ly2) / 2 / th:.6f} "
                                 f"{(lx2 - lx1) / tw:.6f} {(ly2 - ly1) / th:.6f}")
                    stats["class_instances"][XVIEW_CLASSES[cls]] += 1
                stem = f"{Path(image_id).stem}_{tx}_{ty}"
                ip = out / "images" / split / f"{stem}.jpg"
                ip.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(ip), tile)
                lp = out / "labels" / split / f"{stem}.txt"
                lp.parent.mkdir(parents=True, exist_ok=True)
                lp.write_text("\n".join(lines))
                stats["tiles"][split] += 1
                stats["boxes"][split] += len(lines)
        if progress and n % 20 == 0:
            progress(f"Tiled {n + 1}/{len(ids)} xView images")

    write_dataset_yaml(out, XVIEW_CLASSES, {
        "source_format": "xview_geojson", "source_path": str(src),
        "tile_size": tile_size, "tile_overlap": overlap,
        "split_policy": "hash(image_id) over official train images; official test labels are not public",
    })
    return {
        "output": str(out),
        "classes": XVIEW_CLASSES,
        "source_images_per_split": dict(stats["source_images"]),
        "tiles_per_split": dict(stats["tiles"]),
        "boxes_per_split": dict(stats["boxes"]),
        "class_instances": dict(stats["class_instances"]),
        "missing_images": stats["missing_images"],
        "skipped_type_ids": {str(k): v for k, v in skipped_types.items()},
    }


# ─────────────────────────── YOLO validation ───────────────────────────

def validate_yolo_dataset(root: Path, max_label_files: Optional[int] = None) -> dict:
    """
    Check a YOLO dataset's integrity. Returns a report with status:
      VALIDATED  – usable for training and held-out evaluation
      INVALID    – structural errors (see ``errors``)
      NOT_FOUND / RAW_NEEDS_CONVERSION
    """
    report = {"path": str(root), "errors": [], "warnings": [], "splits": {}, "classes": None,
              "class_instances": {}, "status": "INVALID"}
    if not root.exists():
        report["status"] = "NOT_FOUND"
        report["errors"].append(f"Path does not exist: {root}")
        return report
    yaml_path = root / "dataset.yaml"
    if not yaml_path.exists():
        info = inspect_directory(root)
        report["detected_format"] = info["format"]
        if info["format"] in ("xview_geojson", "coco_json"):
            report["status"] = "RAW_NEEDS_CONVERSION"
            report["errors"].append(f"Raw {info['format']} dataset found — run the conversion step first.")
        else:
            report["errors"].append("dataset.yaml not found (expected YOLO layout).")
        return report

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    names = cfg.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    if not names:
        report["errors"].append("dataset.yaml has no 'names' (class mapping).")
        return report
    report["classes"] = names
    nc = len(names)

    instances = Counter()
    stems_by_split = {}
    for split in SPLITS:
        img_dir, lbl_dir = root / "images" / split, root / "labels" / split
        if not img_dir.exists():
            if split in ("train", "val"):
                report["errors"].append(f"Missing required split: images/{split}")
            continue
        imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        lbls = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.exists() else {}
        bad_lines, empty, checked = 0, 0, 0
        for stem, lp in list(lbls.items())[: max_label_files or None]:
            checked += 1
            text = lp.read_text().strip()
            if not text:
                empty += 1
                continue
            for line in text.splitlines():
                parts = line.split()
                try:
                    c = int(parts[0])
                    vals = [float(v) for v in parts[1:5]]
                    if len(parts) < 5 or not (0 <= c < nc) or any(v < 0 or v > 1 for v in vals):
                        raise ValueError
                    instances[names[c]] += 1
                except (ValueError, IndexError):
                    bad_lines += 1
        missing_lbl = sorted(set(imgs) - set(lbls))
        orphan = sorted(set(lbls) - set(imgs))
        report["splits"][split] = {
            "images": len(imgs), "labels": len(lbls), "images_without_labels": len(missing_lbl),
            "orphan_labels": len(orphan), "invalid_label_lines": bad_lines,
            "empty_label_files": empty, "label_files_checked": checked,
        }
        if bad_lines:
            report["errors"].append(f"{split}: {bad_lines} invalid label line(s) (bad class id or coords).")
        if missing_lbl:
            report["warnings"].append(f"{split}: {len(missing_lbl)} image(s) without a label file "
                                      "(treated as background by YOLO).")
        if orphan:
            report["warnings"].append(f"{split}: {len(orphan)} label file(s) without an image.")
        if not imgs and split in ("train", "val"):
            report["errors"].append(f"Split '{split}' contains no images.")
        stems_by_split[split] = set(imgs)

    # Leakage: identical file stems in more than one split
    for a in SPLITS:
        for b in SPLITS:
            if a < b and a in stems_by_split and b in stems_by_split:
                common = stems_by_split[a] & stems_by_split[b]
                if common:
                    report["errors"].append(f"Data leakage: {len(common)} file(s) appear in both {a} and {b}.")
    if "test" not in stems_by_split:
        report["warnings"].append("No test split — held-out evaluation will use 'val'.")
    zero = [n for n in names if instances[n] == 0]
    if zero:
        report["warnings"].append(f"{len(zero)} class(es) have no labelled instances: {zero[:10]}")
    report["class_instances"] = dict(instances)
    report["status"] = "INVALID" if report["errors"] else "VALIDATED"
    return report


def dataset_output_dir(name: str) -> Path:
    return (BASE_DIR / settings.DATASET_DIR / f"{name.lower()}_yolo").resolve()
