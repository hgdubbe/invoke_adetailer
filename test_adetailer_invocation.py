from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence


Box = tuple[int, int, int, int]
SortMode = Literal["confidence", "area", "left_to_right", "top_to_bottom"]
GuideSizeFor = Literal["bbox", "crop"]
TARGET_MODEL_PRESETS = {
    "face": "face_yolov8n.pt",
    "faces": "face_yolov8n.pt",
    "hand": "hand_yolov8n.pt",
    "hands": "hand_yolov8n.pt",
    "person": "person_yolov8n-seg.pt",
    "people": "person_yolov8n-seg.pt",
    "body": "person_yolov8n-seg.pt",
}
DEFAULT_DIFFUSERS_INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"
ADETAILER_HF_REPO = "Bingsu/adetailer"
ADETAILER_MODEL_FILENAMES = {
    "face_yolov8n.pt",
    "face_yolov8n_v2.pt",
    "face_yolov8s.pt",
    "face_yolov8m.pt",
    "hand_yolov8n.pt",
    "hand_yolov8s.pt",
    "person_yolov8n-seg.pt",
    "person_yolov8s-seg.pt",
    "person_yolov8m-seg.pt",
}


@dataclass(frozen=True)
class Detection:
    box: Box
    confidence: float
    class_name: str
    mask: Sequence[Sequence[int]] | None = None


@dataclass(frozen=True)
class DetailRegion:
    detection: Detection
    crop_box: Box
    mask_box: Box


def parse_class_filter(class_filter: str) -> set[str]:
    return {part.strip().lower() for part in class_filter.split(",") if part.strip()}


def resolve_detector_model(target: str, model_path: str) -> str:
    explicit_model = model_path.strip()
    if explicit_model and explicit_model.lower() != "auto":
        return explicit_model
    target_key = target.strip().lower()
    return TARGET_MODEL_PRESETS.get(target_key, "yolov8n.pt")


def resolve_detector_weight_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if path.exists() or path.name not in ADETAILER_MODEL_FILENAMES:
        return str(path)
    return _download_adetailer_weight(path.name)


def resolve_diffusers_model(model: str) -> str:
    explicit_model = model.strip()
    if explicit_model and explicit_model.lower() != "auto":
        return explicit_model
    env_model = os.environ.get("ADETAILER_DIFFUSERS_MODEL", "").strip()
    return env_model or DEFAULT_DIFFUSERS_INPAINT_MODEL


def _download_adetailer_weight(filename: str) -> str:
    try:
        return _hf_hub_download(ADETAILER_HF_REPO, filename, _adetailer_cache_dir())
    except Exception:
        return _download_adetailer_weight_direct(filename)


def _hf_hub_download(repo_id: str, filename: str, cache_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)


def _download_adetailer_weight_direct(filename: str) -> str:
    from urllib.request import urlretrieve

    cache_dir = Path(_adetailer_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename
    if target.exists():
        return str(target)
    url = f"https://huggingface.co/{ADETAILER_HF_REPO}/resolve/main/{filename}"
    urlretrieve(url, target)
    return str(target)


def _adetailer_cache_dir() -> str:
    configured = os.environ.get("ADETAILER_MODEL_CACHE", "").strip()
    if configured:
        return str(Path(configured).expanduser())
    return str(Path.home() / ".cache" / "invokeai-adetailer")


def filter_detections(
    detections: Iterable[Detection],
    confidence_threshold: float,
    class_filter: str = "",
    max_detections: int = 0,
    sort_by: SortMode = "confidence",
    image_size: tuple[int, int] | None = None,
    min_area_ratio: float = 0.0,
    max_area_ratio: float = 1.0,
) -> list[Detection]:
    allowed_classes = parse_class_filter(class_filter)
    image_area = image_size[0] * image_size[1] if image_size is not None else None
    kept = [
        detection
        for detection in detections
        if detection.confidence >= confidence_threshold
        and (not allowed_classes or detection.class_name.lower() in allowed_classes)
        and _is_area_ratio_allowed(detection.box, image_area, min_area_ratio, max_area_ratio)
    ]
    kept.sort(key=_sort_key(sort_by), reverse=sort_by in {"confidence", "area"})
    if max_detections > 0:
        return kept[:max_detections]
    return kept


def expand_box(box: Box, image_size: tuple[int, int], padding: int = 0, crop_factor: float = 1.0) -> Box:
    width, height = image_size
    x1, y1, x2, y2 = box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    scaled_width = box_width * max(1.0, crop_factor)
    scaled_height = box_height * max(1.0, crop_factor)

    left = round(center_x - scaled_width / 2) - padding
    top = round(center_y - scaled_height / 2) - padding
    right = round(center_x + scaled_width / 2) + padding
    bottom = round(center_y + scaled_height / 2) + padding

    return (
        _clamp(left, 0, width),
        _clamp(top, 0, height),
        _clamp(right, 0, width),
        _clamp(bottom, 0, height),
    )


def build_combined_mask(
    detections: Sequence[Detection],
    image_size: tuple[int, int],
    mask_padding: int = 0,
) -> list[list[int]]:
    width, height = image_size
    mask = [[0 for _ in range(width)] for _ in range(height)]

    for detection in detections:
        if detection.mask is not None:
            for y, row in enumerate(detection.mask[:height]):
                output_row = mask[y]
                for x, value in enumerate(row[:width]):
                    if value > 0:
                        output_row[x] = 255
            continue

        x1, y1, x2, y2 = expand_box(detection.box, image_size=image_size, padding=mask_padding)
        for y in range(y1, y2):
            row = mask[y]
            for x in range(x1, x2):
                row[x] = 255

    if mask_padding > 0 and any(detection.mask is not None for detection in detections):
        mask = dilate_mask(mask, mask_padding)

    return mask


def dilate_mask(mask: Sequence[Sequence[int]], radius: int) -> list[list[int]]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    if radius <= 0 or width == 0:
        return [list(row) for row in mask]

    dilated = [[0 for _ in range(width)] for _ in range(height)]
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value <= 0:
                continue
            y1 = _clamp(y - radius, 0, height)
            y2 = _clamp(y + radius + 1, 0, height)
            x1 = _clamp(x - radius, 0, width)
            x2 = _clamp(x + radius + 1, 0, width)
            for yy in range(y1, y2):
                for xx in range(x1, x2):
                    dilated[yy][xx] = 255
    return dilated


def translate_mask(mask: Sequence[Sequence[int]], offset_x: int, offset_y: int) -> list[list[int]]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    translated = [[0 for _ in range(width)] for _ in range(height)]
    for y, row in enumerate(mask):
        target_y = y + offset_y
        if target_y < 0 or target_y >= height:
            continue
        for x, value in enumerate(row):
            target_x = x + offset_x
            if value <= 0 or target_x < 0 or target_x >= width:
                continue
            translated[target_y][target_x] = 255
    return translated


def erode_mask(mask: Sequence[Sequence[int]], radius: int) -> list[list[int]]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    if radius <= 0 or width == 0:
        return [list(row) for row in mask]

    eroded = [[0 for _ in range(width)] for _ in range(height)]
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value <= 0:
                continue
            y1 = y - radius
            y2 = y + radius + 1
            x1 = x - radius
            x2 = x + radius + 1
            if y1 < 0 or x1 < 0 or y2 > height or x2 > width:
                continue
            if all(mask[yy][xx] > 0 for yy in range(y1, y2) for xx in range(x1, x2)):
                eroded[y][x] = 255
    return eroded


def plan_detail_regions(
    detections: Sequence[Detection],
    image_size: tuple[int, int],
    bbox_padding: int,
    mask_padding: int,
    crop_factor: float,
) -> list[DetailRegion]:
    regions: list[DetailRegion] = []
    for detection in detections:
        crop_box = expand_box(
            box=detection.box,
            image_size=image_size,
            padding=bbox_padding,
            crop_factor=crop_factor,
        )
        mask_box = expand_box(
            box=detection.box,
            image_size=image_size,
            padding=mask_padding,
            crop_factor=1.0,
        )
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            continue
        if mask_box[2] <= mask_box[0] or mask_box[3] <= mask_box[1]:
            continue
        regions.append(DetailRegion(detection=detection, crop_box=crop_box, mask_box=mask_box))
    return regions


def detail_inference_size(
    crop_size: tuple[int, int],
    bbox_size: tuple[int, int],
    guide_size: int,
    guide_size_for: GuideSizeFor,
    max_size: int,
    multiple: int = 8,
) -> tuple[int, int]:
    crop_width, crop_height = crop_size
    bbox_width, bbox_height = bbox_size
    crop_long = max(crop_width, crop_height, 1)
    reference_long = max(bbox_width, bbox_height, 1) if guide_size_for == "bbox" else crop_long

    scale = max(1.0, guide_size / reference_long) if guide_size > 0 else 1.0
    target_width = crop_width * scale
    target_height = crop_height * scale

    target_long = max(target_width, target_height, 1.0)
    if max_size > 0 and target_long > max_size:
        max_scale = max_size / target_long
        target_width *= max_scale
        target_height *= max_scale

    return (
        max(multiple, _round_to_multiple(target_width, multiple)),
        max(multiple, _round_to_multiple(target_height, multiple)),
    )


def mask_bounds(mask: Sequence[Sequence[int]]) -> Box | None:
    min_x: int | None = None
    min_y: int | None = None
    max_x: int | None = None
    max_y: int | None = None

    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value <= 0:
                continue
            min_x = x if min_x is None else min(min_x, x)
            min_y = y if min_y is None else min(min_y, y)
            max_x = x + 1 if max_x is None else max(max_x, x + 1)
            max_y = y + 1 if max_y is None else max(max_y, y + 1)

    if min_x is None or min_y is None or max_x is None or max_y is None:
        return None
    return (min_x, min_y, max_x, max_y)


def _sort_key(sort_by: SortMode):
    if sort_by == "area":
        return lambda detection: _area(detection.box)
    if sort_by == "left_to_right":
        return lambda detection: detection.box[0]
    if sort_by == "top_to_bottom":
        return lambda detection: detection.box[1]
    return lambda detection: detection.confidence


def _area(box: Box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _is_area_ratio_allowed(
    box: Box,
    image_area: int | None,
    min_area_ratio: float,
    max_area_ratio: float,
) -> bool:
    if image_area is None or image_area <= 0:
        return True
    area_ratio = _area(box) / image_area
    return area_ratio >= min_area_ratio and area_ratio <= max_area_ratio


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _round_to_multiple(value: float, multiple: int) -> int:
    return int(value // multiple) * multiple
