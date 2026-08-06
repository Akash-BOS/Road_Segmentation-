"""Create measured 3D point clouds from ONNX segmentation overlays.

The geometry mirrors the LCMS distress metric logic used in the C3D-Qt codebase:
419x1024 masks are treated as the metric mask space, linear cracks use skeleton
length plus distance-transform width, and pothole/patch depth is measured
relative to local road pixels when range data is available.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from lcms_unetpp.data import COLOR_MAP


VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}
CLASS_NAMES = {
    0: "BACKGROUND",
    1: "ALLIGATOR",
    2: "TRANSVERSE",
    3: "LONGITUDINAL",
    4: "POTHOLE",
    5: "PATCH",
}
METRIC_MASK_WIDTH = 419
METRIC_MASK_HEIGHT = 1024
DB_IMAGE_WIDTH = 4770
DB_IMAGE_HEIGHT = 2000
CRACK_PIXEL_SIZE_MM = 0.8
POTHOLE_PATCH_PIXEL_SIZE_MM = 10.0
ISOTROPIC_HEIGHT_SCALE = 5
DEPTH_SCALE_MM = 250.0 / 140.0
DEPTH_OUTLIER_MIN = -100.0
DEPTH_OUTLIER_MAX = 100.0


@dataclass
class DistressMetric:
    image: str
    instance_id: int
    class_id: int
    class_name: str
    pixel_area: int
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    bbox_x1_ui: float
    bbox_y1_ui: float
    bbox_x2_ui: float
    bbox_y2_ui: float
    length_mm: float
    width_mm: float
    area_mm2: float
    area_m2: float
    diameter_mm: float
    depth_min_mm: float
    depth_max_mm: float
    depth_avg_mm: float
    road_level: float
    severity: str
    width_severity: str
    depth_severity: str
    ply: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate measured PLY point clouds and metrics from model.onnx "
            "prediction masks/overlays."
        )
    )
    parser.add_argument("--prediction-dir", required=True, help="Folder containing masks/ and overlays/")
    parser.add_argument("--output-dir", default="", help="Defaults to prediction-dir/point_clouds")
    parser.add_argument("--mask-dir", default="", help="Defaults to prediction-dir/masks")
    parser.add_argument("--overlay-dir", default="", help="Defaults to prediction-dir/overlays")
    parser.add_argument("--image-dir", default="", help="Optional original images folder for point colors")
    parser.add_argument("--range-dir", default="", help="Optional filtered/range/depth images folder for Z and depth metrics")
    parser.add_argument("--raw-range-dir", default="", help="Optional raw range folder used for pothole/patch validity checks")
    parser.add_argument("--z-source", default="range", choices=["range", "depth-mm", "mask", "intensity", "flat"])
    parser.add_argument("--xy-scale", default=1.0, type=float, help="Scale applied to pixel X/Y coordinates")
    parser.add_argument("--z-scale", default=1.0, type=float, help="Scale applied to Z values in the PLY")
    parser.add_argument("--stride", default=1, type=int, help="Keep every Nth pixel in each direction")
    parser.add_argument("--include-background", action="store_true", help="Include class 0/background points in full-image cloud")
    parser.add_argument("--instances-only", action="store_true", help="Write one PLY per distress instance instead of one per image")
    parser.add_argument("--combined-name", default="", help="Also write one combined PLY with this filename")
    parser.add_argument("--max-points-per-file", default=0, type=int, help="Randomly downsample each cloud to this many points")
    parser.add_argument("--limit", default=0, type=int, help="Process only the first N prediction results")
    parser.add_argument("--lane-left", default=0, type=int, help="Zero labels left of this mask-space x coordinate")
    parser.add_argument("--lane-right", default=0, type=int, help="Zero labels right of this mask-space x coordinate")
    parser.add_argument("--component-margin", default=5, type=int)
    parser.add_argument("--merge-margin-crack", default=15, type=int)
    parser.add_argument("--merge-margin-area", default=0, type=int)
    parser.add_argument("--min-component-pixels", default=15, type=int)
    parser.add_argument("--area-crack-width-mm", default=3500.0, type=float)
    parser.add_argument("--class-heights", default="", help="Fallback class:height map for --z-source mask")
    parser.add_argument("--write-full-image-cloud", action="store_true", help="Write a full-image cloud in addition to instance clouds")
    return parser.parse_args()


def parse_class_heights(value: str) -> dict[int, float]:
    heights = {0: 0.0, 1: 1.0, 2: 2.0, 3: 2.0, 4: 4.0, 5: 1.5}
    if not value.strip():
        return heights
    for item in value.split(","):
        class_id, height = item.split(":", 1)
        heights[int(class_id.strip())] = float(height.strip())
    return heights


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTS)


def strip_prediction_suffix(stem: str) -> str:
    for suffix in ("_overlay", "_pred", "_range", "_depth", "_filtered"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def index_by_base_name(paths: list[Path]) -> dict[str, Path]:
    return {strip_prediction_suffix(path.stem): path for path in paths}


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return image


def read_scalar(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float32)


def resize_like(image: np.ndarray, shape: tuple[int, int], interpolation: int) -> np.ndarray:
    height, width = shape
    if image.shape[:2] == (height, width):
        return image
    return cv2.resize(image, (width, height), interpolation=interpolation)


def mask_to_labels(mask_rgb: np.ndarray) -> np.ndarray:
    if mask_rgb.ndim == 2:
        return mask_rgb.astype(np.uint8)

    labels = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    known = np.zeros(mask_rgb.shape[:2], dtype=bool)
    mask_i16 = mask_rgb.astype(np.int16)
    for color, class_id in COLOR_MAP.items():
        exact = np.all(mask_i16 == np.asarray(color, dtype=np.int16), axis=2)
        labels[exact] = class_id
        known |= exact

    if np.all(known):
        return labels

    palette = np.asarray(list(COLOR_MAP.keys()), dtype=np.int16)
    class_ids = np.asarray(list(COLOR_MAP.values()), dtype=np.uint8)
    pixels = mask_i16.reshape(-1, 3)
    distances = ((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    nearest = class_ids[np.argmin(distances, axis=1)].reshape(mask_rgb.shape[:2])
    labels[~known] = nearest[~known]
    return labels


def apply_lane_bounds(labels: np.ndarray, lane_left: int, lane_right: int) -> np.ndarray:
    out = labels.copy()
    if lane_left > 0:
        out[:, : min(lane_left, out.shape[1])] = 0
    if lane_right > 0:
        out[:, max(0, lane_right) :] = 0
    return out


def percentile(values: np.ndarray | list[float], pct: float) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, np.clip(pct, 0.0, 100.0), method="nearest"))


def valid_depth_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = values[mask > 0].astype(np.float32)
    valid = np.isfinite(selected)
    valid &= selected > DEPTH_OUTLIER_MIN
    valid &= selected < DEPTH_OUTLIER_MAX
    valid &= np.abs(selected) > 1e-5
    return np.sort(selected[valid])


def zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    img = (binary > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            padded = np.pad(img, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )
            if step == 0:
                condition = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                condition = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (img == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & condition
            if np.any(remove):
                img[remove] = 0
                changed = True
    return (img * 255).astype(np.uint8)


def skeletonize(binary: np.ndarray) -> np.ndarray:
    binary_u8 = ((binary > 0).astype(np.uint8) * 255)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(binary_u8, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return zhang_suen_thinning(binary_u8)


def skeleton_length_px(binary: np.ndarray) -> float:
    skel = skeletonize(binary)
    coords = np.argwhere(skel > 0)
    pixels = set(map(tuple, coords.tolist()))
    length = 0.0
    for y, x in pixels:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if (y + dy, x + dx) in pixels:
                    length += 1.0 if dx == 0 or dy == 0 else math.sqrt(2.0)
    return length / 2.0


def thickness_p90_mm(binary: np.ndarray) -> float:
    binary_u8 = ((binary > 0).astype(np.uint8) * 255)
    skel = skeletonize(binary_u8)
    dist = cv2.distanceTransform(binary_u8, cv2.DIST_L2, 5)
    samples = (2.0 * dist[skel > 0]).astype(np.float32)
    if samples.size == 0:
        return 0.0
    return percentile(samples.astype(np.int32), 90.0) * CRACK_PIXEL_SIZE_MM


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0, 0, 0, 0
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return x0, y0, x1 - x0, y1 - y0


def expand_box(box: tuple[int, int, int, int], margin: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = box
    height, width = shape
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(width, x + w + margin)
    y1 = min(height, y + h + margin)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (
        ax + aw + margin < bx
        or bx + bw + margin < ax
        or ay + ah + margin < by
        or by + bh + margin < ay
    )


def union_box(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = min(ax, bx)
    y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw)
    y1 = max(ay + ah, by + bh)
    return x0, y0, x1 - x0, y1 - y0


def merge_boxes(boxes: list[tuple[int, int, int, int]], margin: int) -> list[tuple[int, int, int, int]]:
    merged = boxes[:]
    changed = True
    while changed:
        changed = False
        out: list[tuple[int, int, int, int]] = []
        used = [False] * len(merged)
        for i, box in enumerate(merged):
            if used[i]:
                continue
            current = box
            used[i] = True
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if boxes_overlap(current, merged[j], margin):
                    current = union_box(current, merged[j])
                    used[j] = True
                    changed = True
            out.append(current)
        merged = out
    return merged


def component_boxes(class_mask: np.ndarray, margin: int, merge_margin: int, min_pixels: int) -> list[tuple[int, int, int, int]]:
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask.astype(np.uint8), 8, cv2.CV_32S)
    boxes = []
    for index in range(1, labels_count):
        if int(stats[index, cv2.CC_STAT_AREA]) < min_pixels:
            continue
        box = (
            int(stats[index, cv2.CC_STAT_LEFT]),
            int(stats[index, cv2.CC_STAT_TOP]),
            int(stats[index, cv2.CC_STAT_WIDTH]),
            int(stats[index, cv2.CC_STAT_HEIGHT]),
        )
        boxes.append(expand_box(box, margin, class_mask.shape))
    return merge_boxes(boxes, merge_margin)


def is_area_crack(class_id: int) -> bool:
    return CLASS_NAMES.get(class_id, "") == "ALLIGATOR"


def is_pothole_or_patch(class_id: int) -> bool:
    return CLASS_NAMES.get(class_id, "") in {"POTHOLE", "PATCH"}


def severity_from_depth(depth_mm: float) -> str:
    if depth_mm >= 50.0:
        return "HIGH"
    if depth_mm >= 25.0:
        return "MEDIUM"
    if depth_mm >= 10.0:
        return "LOW"
    return "DELAMINATION"


def severity_from_width(width_mm: float) -> str:
    if width_mm >= 20.0:
        return "HIGH"
    if width_mm >= 6.0:
        return "MEDIUM"
    if width_mm >= 3.0:
        return "LOW"
    return "HAIRLINE"


def measure_depth(
    instance_mask: np.ndarray,
    box: tuple[int, int, int, int],
    range_image: np.ndarray | None,
    raw_range_image: np.ndarray | None,
    class_id: int,
) -> tuple[float, float, float, float, np.ndarray]:
    depth_mm_map = np.zeros(instance_mask.shape, dtype=np.float32)
    if range_image is None:
        return 0.0, 0.0, 0.0, 0.0, depth_mm_map

    x, y, w, h = box
    roi_mask = instance_mask[y : y + h, x : x + w].astype(np.uint8)
    roi_range = range_image[y : y + h, x : x + w].astype(np.float32)
    road_mask = (roi_mask == 0).astype(np.uint8)

    distress_values = valid_depth_values(roi_range, roi_mask)
    road_values = valid_depth_values(roi_range, road_mask)
    if distress_values.size == 0 or road_values.size == 0:
        return 0.0, 0.0, 0.0, 0.0, depth_mm_map

    road_level = percentile(road_values, 90.0)
    if is_pothole_or_patch(class_id):
        depth_values = (road_level - distress_values) * DEPTH_SCALE_MM
        depth_values = depth_values[depth_values > 0.0]
        if raw_range_image is not None:
            raw_values = valid_depth_values(raw_range_image[y : y + h, x : x + w], roi_mask)
            if raw_values.size > 0 and percentile(raw_values, 50.0) < 40.0:
                depth_values = np.asarray([], dtype=np.float32)
    else:
        depth_values = np.abs(distress_values - road_level) * DEPTH_SCALE_MM

    if depth_values.size == 0:
        return road_level, 0.0, 0.0, 0.0, depth_mm_map

    roi_depth_mm = np.maximum((road_level - roi_range) * DEPTH_SCALE_MM, 0.0)
    if not is_pothole_or_patch(class_id):
        roi_depth_mm = np.abs((roi_range - road_level) * DEPTH_SCALE_MM)
    depth_mm_map[y : y + h, x : x + w][roi_mask > 0] = roi_depth_mm[roi_mask > 0]

    return (
        road_level,
        percentile(depth_values, 1.0),
        percentile(depth_values, 99.0),
        float(np.mean(depth_values)),
        depth_mm_map,
    )


def make_metric(
    base_name: str,
    instance_id: int,
    class_id: int,
    instance_mask: np.ndarray,
    box: tuple[int, int, int, int],
    range_image: np.ndarray | None,
    raw_range_image: np.ndarray | None,
    area_crack_width_mm: float,
    ply_path: Path,
) -> tuple[DistressMetric, np.ndarray]:
    x, y, w, h = box
    roi_mask = instance_mask[y : y + h, x : x + w]
    roi_isotropic = cv2.resize(
        roi_mask.astype(np.uint8),
        (max(1, w), max(1, h * ISOTROPIC_HEIGHT_SCALE)),
        interpolation=cv2.INTER_NEAREST,
    )
    pixel_area = int(np.count_nonzero(instance_mask))
    width_mm = thickness_p90_mm(instance_mask)
    length_mm = skeleton_length_px(instance_mask) * CRACK_PIXEL_SIZE_MM
    area_mm2 = (length_mm * width_mm) if width_mm > 0 else 0.0
    diameter_mm = 0.0

    if is_pothole_or_patch(class_id):
        length_mm = max(0, h - 10) * POTHOLE_PATCH_PIXEL_SIZE_MM
        width_mm = max(0, w - 10) * POTHOLE_PATCH_PIXEL_SIZE_MM
        area_mm2 = float(np.count_nonzero(roi_isotropic))
        diameter_mm = 2.0 * math.sqrt(area_mm2 / math.pi) if area_mm2 > 0 else 0.0
    elif is_area_crack(class_id):
        length_mm = h * 10000.0 / float(instance_mask.shape[0])
        width_mm = area_crack_width_mm
        area_mm2 = length_mm * width_mm

    road_level, depth_min, depth_max, depth_avg, depth_mm_map = measure_depth(
        instance_mask,
        box,
        range_image,
        raw_range_image,
        class_id,
    )

    depth_severity = severity_from_depth(depth_max if is_pothole_or_patch(class_id) else depth_avg)
    width_severity = severity_from_width(width_mm)
    severity = depth_severity if depth_max > 0 else width_severity

    sx_ui = DB_IMAGE_WIDTH / float(METRIC_MASK_WIDTH)
    sy_ui = DB_IMAGE_HEIGHT / float(METRIC_MASK_HEIGHT)
    metric = DistressMetric(
        image=base_name,
        instance_id=instance_id,
        class_id=class_id,
        class_name=CLASS_NAMES.get(class_id, f"CLASS_{class_id}"),
        pixel_area=pixel_area,
        bbox_x=x,
        bbox_y=y,
        bbox_width=w,
        bbox_height=h,
        bbox_x1_ui=x * sx_ui,
        bbox_y1_ui=y * sy_ui,
        bbox_x2_ui=(x + w) * sx_ui,
        bbox_y2_ui=(y + h) * sy_ui,
        length_mm=length_mm,
        width_mm=width_mm,
        area_mm2=area_mm2,
        area_m2=area_mm2 / 1_000_000.0,
        diameter_mm=diameter_mm,
        depth_min_mm=depth_min,
        depth_max_mm=depth_max,
        depth_avg_mm=depth_avg,
        road_level=road_level,
        severity=severity,
        width_severity=width_severity,
        depth_severity=depth_severity,
        ply=str(ply_path),
    )
    return metric, depth_mm_map


def make_z_values(
    labels: np.ndarray,
    colors: np.ndarray,
    range_image: np.ndarray | None,
    depth_mm_map: np.ndarray | None,
    z_source: str,
    class_heights: dict[int, float],
    z_scale: float,
) -> np.ndarray:
    if z_source == "range":
        if range_image is not None:
            values = range_image.astype(np.float32)
            finite = values[np.isfinite(values)]
            if finite.size:
                values = values - float(np.nanmin(finite))
            return values * z_scale
        z_source = "depth-mm"

    if z_source == "depth-mm":
        if depth_mm_map is not None and np.count_nonzero(depth_mm_map) > 0:
            return depth_mm_map.astype(np.float32) * z_scale
        z_source = "mask"

    if z_source == "intensity":
        gray = cv2.cvtColor(colors, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return (gray / 255.0) * z_scale

    if z_source == "flat":
        return np.zeros(labels.shape, dtype=np.float32)

    z = np.zeros(labels.shape, dtype=np.float32)
    for class_id, height in class_heights.items():
        z[labels == class_id] = height
    return z * z_scale


def make_point_cloud(
    labels: np.ndarray,
    colors: np.ndarray,
    z_values: np.ndarray,
    instance_ids: np.ndarray,
    depth_mm_map: np.ndarray,
    xy_scale: float,
    stride: int,
    include_background: bool,
) -> np.ndarray:
    labels_s = labels[::stride, ::stride]
    colors_s = colors[::stride, ::stride]
    z_s = z_values[::stride, ::stride]
    instance_s = instance_ids[::stride, ::stride]
    depth_s = depth_mm_map[::stride, ::stride]

    yy, xx = np.indices(labels_s.shape)
    valid = np.isfinite(z_s)
    if not include_background:
        valid &= labels_s != 0

    return np.column_stack(
        [
            xx[valid].astype(np.float32) * xy_scale,
            yy[valid].astype(np.float32) * xy_scale,
            z_s[valid].astype(np.float32),
            colors_s[valid, 0].astype(np.uint8),
            colors_s[valid, 1].astype(np.uint8),
            colors_s[valid, 2].astype(np.uint8),
            labels_s[valid].astype(np.uint8),
            instance_s[valid].astype(np.uint16),
            depth_s[valid].astype(np.float32),
        ]
    )


def downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    indices = np.random.default_rng(0).choice(len(points), size=max_points, replace=False)
    return points[np.sort(indices)]


def write_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property uchar label",
            "property ushort instance_id",
            "property float depth_mm",
            "end_header",
        ]
    )
    with path.open("w", encoding="utf-8") as file:
        file.write(header + "\n")
        for point in points:
            file.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(point[3])} {int(point[4])} {int(point[5])} "
                f"{int(point[6])} {int(point[7])} {point[8]:.6f}\n"
            )


def resolve_match(index: dict[str, Path], base_name: str, kind: str, warn: bool = True) -> Path | None:
    path = index.get(base_name)
    if path is None and warn:
        print(f"warning: missing {kind} for {base_name}")
    return path


def write_metrics(output_dir: Path, metrics: list[DistressMetric]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in metrics]
    (output_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not rows:
        return
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    prediction_dir = Path(args.prediction_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else prediction_dir / "masks"
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else prediction_dir / "overlays"
    output_dir = Path(args.output_dir) if args.output_dir else prediction_dir / "point_clouds"
    image_dir = Path(args.image_dir) if args.image_dir else None
    range_dir = Path(args.range_dir) if args.range_dir else None
    raw_range_dir = Path(args.raw_range_dir) if args.raw_range_dir else None
    stride = max(1, args.stride)
    class_heights = parse_class_heights(args.class_heights)

    mask_index = index_by_base_name(list_images(mask_dir))
    overlay_index = index_by_base_name(list_images(overlay_dir))
    image_index = index_by_base_name(list_images(image_dir)) if image_dir else {}
    range_index = index_by_base_name(list_images(range_dir)) if range_dir else {}
    raw_range_index = index_by_base_name(list_images(raw_range_dir)) if raw_range_dir else {}

    base_names = sorted(mask_index.keys() or overlay_index.keys())
    if args.limit > 0:
        base_names = base_names[: args.limit]
    if not base_names:
        raise FileNotFoundError(f"No prediction masks or overlays found under: {prediction_dir}")

    manifest = []
    all_metrics: list[DistressMetric] = []
    combined = []
    for base_name in base_names:
        mask_path = resolve_match(mask_index, base_name, "mask")
        overlay_path = resolve_match(overlay_index, base_name, "overlay", warn=False)
        if mask_path is None and overlay_path is None:
            continue

        label_source = read_rgb(mask_path) if mask_path is not None else read_rgb(overlay_path)
        labels = mask_to_labels(label_source)
        labels = apply_lane_bounds(labels, args.lane_left, args.lane_right)
        height, width = labels.shape

        color_path = image_index.get(base_name) or overlay_path or mask_path
        colors = resize_like(read_rgb(color_path), (height, width), cv2.INTER_LINEAR)

        range_image = None
        if range_dir is not None:
            range_path = resolve_match(range_index, base_name, "range image")
            if range_path is not None:
                range_image = resize_like(read_scalar(range_path), (height, width), cv2.INTER_LINEAR)

        raw_range_image = None
        if raw_range_dir is not None:
            raw_path = resolve_match(raw_range_index, base_name, "raw range image")
            if raw_path is not None:
                raw_range_image = resize_like(read_scalar(raw_path), (height, width), cv2.INTER_LINEAR)

        instance_ids = np.zeros(labels.shape, dtype=np.uint16)
        depth_mm_map = np.zeros(labels.shape, dtype=np.float32)
        image_metrics: list[DistressMetric] = []
        instance_id = 1

        for class_id in sorted(int(v) for v in np.unique(labels) if int(v) != 0):
            class_mask = (labels == class_id).astype(np.uint8)
            merge_margin = args.merge_margin_area if is_pothole_or_patch(class_id) else args.merge_margin_crack
            boxes = component_boxes(class_mask, args.component_margin, merge_margin, args.min_component_pixels)
            for box in boxes:
                x, y, w, h = box
                instance_mask = np.zeros_like(class_mask, dtype=np.uint8)
                roi_class = class_mask[y : y + h, x : x + w]
                instance_mask[y : y + h, x : x + w] = roi_class
                if np.count_nonzero(instance_mask) < args.min_component_pixels:
                    continue

                ply_name = f"{base_name}_inst{instance_id:03d}_{CLASS_NAMES.get(class_id, class_id)}.ply"
                ply_path = output_dir / "instances" / ply_name
                metric, instance_depth = make_metric(
                    base_name,
                    instance_id,
                    class_id,
                    instance_mask,
                    bbox_from_mask(instance_mask),
                    range_image,
                    raw_range_image,
                    args.area_crack_width_mm,
                    ply_path,
                )
                image_metrics.append(metric)
                all_metrics.append(metric)
                instance_ids[instance_mask > 0] = instance_id
                depth_mm_map[instance_mask > 0] = instance_depth[instance_mask > 0]

                inst_labels = np.where(instance_mask > 0, labels, 0).astype(np.uint8)
                inst_colors = colors.copy()
                inst_z = make_z_values(inst_labels, inst_colors, range_image, instance_depth, args.z_source, class_heights, args.z_scale)
                inst_points = make_point_cloud(
                    inst_labels,
                    inst_colors,
                    inst_z,
                    np.where(instance_mask > 0, instance_id, 0).astype(np.uint16),
                    instance_depth,
                    args.xy_scale,
                    stride,
                    include_background=False,
                )
                inst_points = downsample(inst_points, args.max_points_per_file)
                write_ply(ply_path, inst_points)
                if args.combined_name and args.instances_only:
                    combined.append(inst_points)

                instance_id += 1

        wrote_points = 0
        image_ply_path = output_dir / f"{base_name}.ply"
        if not args.instances_only or args.write_full_image_cloud:
            z_values = make_z_values(labels, colors, range_image, depth_mm_map, args.z_source, class_heights, args.z_scale)
            points = make_point_cloud(
                labels,
                colors,
                z_values,
                instance_ids,
                depth_mm_map,
                args.xy_scale,
                stride,
                include_background=args.include_background,
            )
            points = downsample(points, args.max_points_per_file)
            write_ply(image_ply_path, points)
            wrote_points = len(points)
            if args.combined_name:
                combined.append(points)

        manifest.append(
            {
                "name": base_name,
                "instances": len(image_metrics),
                "points": int(wrote_points),
                "ply": str(image_ply_path) if wrote_points else "",
                "metrics": [asdict(item) for item in image_metrics],
            }
        )
        print(f"processed {base_name} instances={len(image_metrics)} points={wrote_points}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_metrics(output_dir, all_metrics)

    if args.combined_name and combined:
        combined_points = np.concatenate(combined, axis=0)
        write_ply(output_dir / args.combined_name, combined_points)
        print(f"wrote {output_dir / args.combined_name} points={len(combined_points)}")

    print(f"wrote manifest {manifest_path}")
    print(f"wrote metrics {output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
