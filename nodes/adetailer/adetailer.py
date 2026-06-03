import sys
from pathlib import Path
from typing import Literal

try:
    from invokeai.invocation_api import BaseInvocation, ImageField, InputField, InvocationContext, invocation
except ModuleNotFoundError:
    from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
    from invokeai.app.invocations.fields import ImageField, InputField
    from invokeai.app.services.shared.invocation_context import InvocationContext

try:
    from invokeai.app.invocations.primitives import ImageOutput
except ImportError:
    from invokeai.app.invocations.image import ImageOutput

from .utils import (
    Detection,
    build_combined_mask,
    detail_inference_size,
    erode_mask,
    filter_detections,
    plan_detail_regions,
    resolve_diffusers_model,
    resolve_detector_model,
    resolve_detector_weight_path,
    translate_mask,
)


DetectionSort = Literal["confidence", "area", "left_to_right", "top_to_bottom"]
GuideSizeFor = Literal["bbox", "crop"]
DetailBackend = Literal["diffusers_inpaint", "enhance", "cv2_inpaint", "blur_fill"]
DetailScheduler = Literal["default", "euler", "euler_a", "ddim", "dpmpp_2m"]
_PIPELINE_CACHE = {}
DETECTOR = None
DETAIL_BACKEND = None


@invocation(
    "adetailer",
    title="ADetailer",
    tags=["adetailer", "detailer", "face", "hand", "inpaint", "ultralytics", "yolo"],
    category="ADetailer",
    version="0.1.0",
)
class ADetailerInvocation(BaseInvocation):
    """Automatically detects targets, details them, and returns the edited image."""

    image: ImageField = InputField(description="Input image to detail")
    prompt: str = InputField(
        default="",
        description="Positive detail prompt, for example 'detailed eyes, natural face, sharp focus'.",
    )
    negative_prompt: str = InputField(
        default="",
        description="Negative prompt for future diffusion backends and metadata.",
    )
    target: str = InputField(
        default="face",
        description="What to detail. This is also used as the class filter unless class_filter is set.",
    )
    model_path: str = InputField(
        default="auto",
        description="Ultralytics model name/path. Use auto to choose from target presets.",
    )
    confidence: float = InputField(default=0.3, ge=0.0, le=1.0, description="Detection confidence threshold")
    iou: float = InputField(default=0.5, ge=0.0, le=1.0, description="NMS IoU threshold")
    class_filter: str = InputField(default="", description="Comma-separated class names to keep. Defaults to target.")
    max_detections: int = InputField(default=8, ge=0, le=128, description="Maximum detections. Use 0 for no limit.")
    sort_by: DetectionSort = InputField(default="confidence", description="Detection processing order")
    min_area_ratio: float = InputField(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Reject detections smaller than this fraction of the image.",
    )
    max_area_ratio: float = InputField(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Reject detections larger than this fraction of the image.",
    )
    bbox_padding: int = InputField(default=32, ge=0, le=1024, description="Pixels added to each inpaint crop")
    mask_padding: int = InputField(default=16, ge=0, le=1024, description="Pixels added to each detected mask")
    mask_erosion: int = InputField(default=0, ge=0, le=1024, description="Pixels to erode the mask after padding")
    mask_offset_x: int = InputField(default=0, ge=-1024, le=1024, description="Horizontal mask offset in pixels")
    mask_offset_y: int = InputField(default=0, ge=-1024, le=1024, description="Vertical mask offset in pixels")
    mask_blur: int = InputField(default=4, ge=0, le=256, description="Blur radius for blended mask edges")
    crop_factor: float = InputField(default=1.5, ge=1.0, le=4.0, description="Scale each crop around its center")
    guide_size: int = InputField(
        default=512,
        ge=0,
        le=4096,
        description="Target detail resolution for the selected crop or bbox. Use 0 to keep crop size.",
    )
    guide_size_for: GuideSizeFor = InputField(
        default="bbox",
        description="Use the detected bbox or full crop as the reference for guide_size.",
    )
    max_size: int = InputField(
        default=1024,
        ge=0,
        le=4096,
        description="Maximum long edge for each detail pass. Use 0 for no cap.",
    )
    denoise_strength: float = InputField(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Detail strength. Maps to inpaint strength for diffusion backends and intensity for local backends.",
    )
    steps: int = InputField(default=20, ge=1, le=150, description="Diffusion steps for the detail pass")
    cfg_scale: float = InputField(default=7.0, ge=1.0, le=30.0, description="CFG scale for the detail pass")
    scheduler: DetailScheduler = InputField(
        default="default",
        description="Diffusion scheduler for the detail pass.",
    )
    seed: int = InputField(default=0, ge=0, description="Detail seed. Use 0 for pipeline randomness.")
    detail_iterations: int = InputField(
        default=1,
        ge=1,
        le=10,
        description="Number of detail passes to run for each detected region.",
    )
    detail_opacity: float = InputField(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Opacity of the detailed crop when blended back into the image.",
    )
    detail_backend: DetailBackend = InputField(
        default="diffusers_inpaint",
        description="Detail backend. diffusers_inpaint is prompt-driven; local backends are useful for testing.",
    )
    diffusers_model: str = InputField(
        default="auto",
        description="Diffusers inpaint model id/path. Use auto for ADETAILER_DIFFUSERS_MODEL or the built-in default.",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        try:
            from PIL import Image, ImageFilter
        except ImportError as error:
            raise RuntimeError("ADetailer requires Pillow to be installed in the InvokeAI environment.") from error

        image = context.images.get_pil(self.image.image_name).convert("RGB")
        detector = DETECTOR or _run_ultralytics_detection
        detail_backend_runner = DETAIL_BACKEND or _run_detail_backend
        detections = detector(
            image=image,
            model_path=resolve_detector_model(self.target, self.model_path),
            confidence=self.confidence,
            iou=self.iou,
        )
        detections = filter_detections(
            detections=detections,
            confidence_threshold=self.confidence,
            class_filter=self.class_filter or self.target,
            max_detections=self.max_detections,
            sort_by=self.sort_by,
            image_size=image.size,
            min_area_ratio=self.min_area_ratio,
            max_area_ratio=self.max_area_ratio,
        )
        if not detections:
            context.logger.info("ADetailer found no matching detections; returning the original image.")
            return ImageOutput.build(context.images.get_dto(self.image.image_name))

        result = image.copy()
        regions = plan_detail_regions(
            detections=detections,
            image_size=image.size,
            bbox_padding=self.bbox_padding,
            mask_padding=self.mask_padding,
            crop_factor=self.crop_factor,
        )

        for index, region in enumerate(regions):
            crop = result.crop(region.crop_box)
            mask = _region_mask_image(
                Image,
                image.size,
                region,
                self.mask_padding,
                self.mask_erosion,
                self.mask_offset_x,
                self.mask_offset_y,
            ).crop(region.crop_box)
            if self.mask_blur > 0:
                mask = mask.filter(ImageFilter.GaussianBlur(self.mask_blur))
            edited_crop = crop
            inference_size = detail_inference_size(
                crop_size=crop.size,
                bbox_size=(
                    region.detection.box[2] - region.detection.box[0],
                    region.detection.box[3] - region.detection.box[1],
                ),
                guide_size=self.guide_size,
                guide_size_for=self.guide_size_for,
                max_size=self.max_size,
            )
            for pass_index in range(self.detail_iterations):
                edited_crop = detail_backend_runner(
                    crop=edited_crop,
                    mask=mask,
                    backend=self.detail_backend,
                    strength=self.denoise_strength,
                    prompt=self.prompt,
                    negative_prompt=self.negative_prompt,
                    steps=self.steps,
                    cfg_scale=self.cfg_scale,
                    scheduler=self.scheduler,
                    seed=_region_pass_seed(self.seed, index, pass_index, self.detail_iterations),
                    diffusers_model=self.diffusers_model,
                    inference_size=inference_size,
                )
                edited_crop = _replace_blank_detail_output(
                    original_crop=crop,
                    edited_crop=edited_crop,
                    strength=self.denoise_strength,
                    prompt=self.prompt,
                    logger=context.logger,
                )
            edited_crop = _apply_detail_opacity(crop, edited_crop, self.detail_opacity)
            result.paste(edited_crop, region.crop_box, mask)

        context.logger.info(
            f"ADetailer processed {len(regions)} {self.target!r} region(s) with prompt={self.prompt!r}."
        )
        image_dto = context.images.save(image=result)
        return ImageOutput.build(image_dto)


@invocation(
    "adetailer_mask",
    title="ADetailer Mask",
    tags=["adetailer", "detailer", "mask", "ultralytics", "yolo"],
    category="ADetailer",
    version="0.1.0",
)
class ADetailerMaskInvocation(BaseInvocation):
    """Detects detail targets and outputs a binary inpaint mask."""

    image: ImageField = InputField(description="Input image")
    target: str = InputField(
        default="face",
        description="Detection target used when model_path is auto.",
    )
    model_path: str = InputField(
        default="auto",
        description="Ultralytics model name/path. Use auto to choose from target presets.",
    )
    confidence: float = InputField(default=0.3, ge=0.0, le=1.0, description="Detection confidence threshold")
    iou: float = InputField(default=0.5, ge=0.0, le=1.0, description="NMS IoU threshold")
    class_filter: str = InputField(
        default="",
        description="Comma-separated class names to keep. Leave empty to keep all model classes.",
    )
    max_detections: int = InputField(default=8, ge=0, le=128, description="Maximum detections. Use 0 for no limit.")
    sort_by: DetectionSort = InputField(
        default="confidence",
        description="Order detections before applying the maximum detection limit.",
    )
    min_area_ratio: float = InputField(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Reject detections smaller than this fraction of the image.",
    )
    max_area_ratio: float = InputField(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Reject detections larger than this fraction of the image.",
    )
    mask_padding: int = InputField(default=32, ge=0, le=1024, description="Pixels to grow each detected mask/box")
    mask_erosion: int = InputField(default=0, ge=0, le=1024, description="Pixels to erode the mask after padding")
    mask_offset_x: int = InputField(default=0, ge=-1024, le=1024, description="Horizontal mask offset in pixels")
    mask_offset_y: int = InputField(default=0, ge=-1024, le=1024, description="Vertical mask offset in pixels")
    bbox_padding: int = InputField(default=0, ge=0, le=1024, description="Extra box padding before mask creation")
    crop_factor: float = InputField(default=1.0, ge=1.0, le=4.0, description="Scale each detection box around its center")
    invert_mask: bool = InputField(default=False, description="Invert the generated mask")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        try:
            from PIL import Image, ImageOps
        except ImportError as error:
            raise RuntimeError("ADetailer Mask requires Pillow to be installed in the InvokeAI environment.") from error

        image = context.images.get_pil(self.image.image_name).convert("RGB")
        detections = _run_ultralytics_detection(
            image=image,
            model_path=resolve_detector_model(self.target, self.model_path),
            confidence=self.confidence,
            iou=self.iou,
        )
        detections = filter_detections(
            detections=detections,
            confidence_threshold=self.confidence,
            class_filter=self.class_filter,
            max_detections=self.max_detections,
            sort_by=self.sort_by,
            image_size=image.size,
            min_area_ratio=self.min_area_ratio,
            max_area_ratio=self.max_area_ratio,
        )
        padded = _pad_detections_for_mask(detections, image.size, self.bbox_padding, self.crop_factor)
        raw_mask = build_combined_mask(padded, image_size=image.size, mask_padding=self.mask_padding)
        raw_mask = erode_mask(raw_mask, radius=self.mask_erosion)
        raw_mask = translate_mask(raw_mask, offset_x=self.mask_offset_x, offset_y=self.mask_offset_y)
        mask = Image.new("L", image.size, 0)
        mask.putdata([value for row in raw_mask for value in row])
        if self.invert_mask:
            mask = ImageOps.invert(mask)

        image_dto = context.images.save(image=mask)
        return ImageOutput.build(image_dto)


@invocation(
    "adetailer_composite",
    title="ADetailer Composite",
    tags=["adetailer", "detailer", "mask", "composite"],
    category="ADetailer",
    version="0.1.0",
)
class ADetailerCompositeInvocation(BaseInvocation):
    """Blends a detailed image over a source image through a mask."""

    base_image: ImageField = InputField(description="Original image")
    detail_image: ImageField = InputField(description="Detailed or inpainted image")
    mask: ImageField = InputField(description="White areas receive detail image pixels")
    mask_blur: int = InputField(default=4, ge=0, le=256, description="Gaussian blur radius for the mask edge")
    opacity: float = InputField(default=1.0, ge=0.0, le=1.0, description="Detail image opacity inside the mask")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        try:
            from PIL import Image, ImageChops, ImageFilter
        except ImportError as error:
            raise RuntimeError("ADetailer Composite requires Pillow to be installed in the InvokeAI environment.") from error

        base_image = context.images.get_pil(self.base_image.image_name).convert("RGBA")
        detail_image = context.images.get_pil(self.detail_image.image_name).convert("RGBA").resize(base_image.size)
        mask = context.images.get_pil(self.mask.image_name).convert("L").resize(base_image.size)

        if self.mask_blur > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(self.mask_blur))
        if self.opacity < 1.0:
            mask = ImageChops.multiply(mask, Image.new("L", mask.size, round(255 * self.opacity)))

        result = Image.composite(detail_image, base_image, mask).convert("RGB")
        image_dto = context.images.save(image=result)
        return ImageOutput.build(image_dto)


def _run_ultralytics_detection(image, model_path: str, confidence: float, iou: float) -> list[Detection]:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(_ultralytics_import_error_message(error)) from error

    model_ref = resolve_detector_weight_path(str(Path(model_path).expanduser()))
    model = YOLO(model_ref)
    results = model.predict(source=image, conf=confidence, iou=iou, verbose=False)
    if not results:
        return []

    names = getattr(results[0], "names", {}) or getattr(model, "names", {})
    detections: list[Detection] = []

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        result_masks = getattr(result, "masks", None)
        for index, box in enumerate(boxes):
            xyxy = box.xyxy[0].tolist()
            class_index = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else -1
            class_name = str(names.get(class_index, class_index))
            score = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 1.0
            detections.append(
                Detection(
                    box=tuple(round(value) for value in xyxy),  # type: ignore[arg-type]
                    confidence=score,
                    class_name=class_name,
                    mask=_extract_ultralytics_mask(result_masks, index=index, image_size=image.size),
                )
            )

    return detections


def _ultralytics_import_error_message(error: ImportError) -> str:
    python_hint = f"InvokeAI Python executable: {sys.executable}"
    if isinstance(error, ModuleNotFoundError) and error.name == "ultralytics":
        return (
            "ADetailer requires ultralytics, but ultralytics is not installed in the Python environment "
            f"running InvokeAI. Install it into that environment. {python_hint}"
        )
    return (
        "ADetailer found ultralytics import code, but a dependency failed to import while loading it: "
        f"{error}. {python_hint}"
    )


def _extract_ultralytics_mask(result_masks, index: int, image_size: tuple[int, int]):
    if result_masks is None or getattr(result_masks, "data", None) is None:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None

    data = result_masks.data[index]
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    nearest = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
    mask = Image.fromarray((data > 0.5).astype("uint8") * 255, mode="L").resize(image_size, resample=nearest)
    width, height = image_size
    values = list(mask.getdata())
    return [values[y * width : (y + 1) * width] for y in range(height)]


def _region_mask_image(
    image_module,
    image_size: tuple[int, int],
    region,
    mask_padding: int,
    mask_erosion: int,
    mask_offset_x: int = 0,
    mask_offset_y: int = 0,
):
    mask = image_module.new("L", image_size, 0)
    if region.detection.mask is None:
        raw_mask = build_combined_mask(
            [
                Detection(
                    box=region.mask_box,
                    confidence=region.detection.confidence,
                    class_name=region.detection.class_name,
                )
            ],
            image_size=image_size,
            mask_padding=0,
        )
        raw_mask = erode_mask(raw_mask, radius=mask_erosion)
        raw_mask = translate_mask(raw_mask, offset_x=mask_offset_x, offset_y=mask_offset_y)
        width, height = image_size
        mask.putdata([value for row in raw_mask for value in row][: width * height])
        return mask

    width, height = image_size
    raw_mask = build_combined_mask([region.detection], image_size=image_size, mask_padding=mask_padding)
    raw_mask = erode_mask(raw_mask, radius=mask_erosion)
    raw_mask = translate_mask(raw_mask, offset_x=mask_offset_x, offset_y=mask_offset_y)
    values = [value for row in raw_mask for value in row]
    mask.putdata(values[: width * height])
    return mask


def _run_detail_backend(
    crop,
    mask,
    backend: DetailBackend,
    strength: float,
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg_scale: float,
    scheduler: DetailScheduler,
    seed: int,
    diffusers_model: str,
    inference_size: tuple[int, int],
):
    from PIL import ImageFilter

    if backend == "diffusers_inpaint":
        return _run_diffusers_inpaint(
            crop=crop,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            steps=steps,
            cfg_scale=cfg_scale,
            scheduler=scheduler,
            seed=seed,
            model=resolve_diffusers_model(diffusers_model),
            inference_size=inference_size,
        )
    if backend == "cv2_inpaint":
        try:
            from invokeai.backend.image_util.infill_methods.cv2_inpaint import cv2_inpaint
        except ImportError:
            return _enhance_crop(crop, strength=strength, prompt=prompt)
        rgba = crop.convert("RGBA")
        return cv2_inpaint(rgba).convert("RGB")
    if backend == "blur_fill":
        radius = max(1, round(2 + 12 * strength))
        return crop.filter(ImageFilter.GaussianBlur(radius)).convert("RGB")
    return _enhance_crop(crop, strength=strength, prompt=prompt)


def _region_pass_seed(seed: int, region_index: int, pass_index: int, detail_iterations: int) -> int:
    if seed <= 0:
        return 0
    return seed + region_index * detail_iterations + pass_index


def _apply_detail_opacity(original_crop, edited_crop, opacity: float):
    if opacity >= 1.0:
        return edited_crop
    if opacity <= 0.0:
        return original_crop
    try:
        from PIL import Image
    except ImportError:
        return edited_crop
    return Image.blend(original_crop.convert("RGB"), edited_crop.convert("RGB"), opacity)


def _replace_blank_detail_output(original_crop, edited_crop, strength: float, prompt: str, logger):
    if not _is_blank_placeholder_output(original_crop, edited_crop):
        return edited_crop
    if hasattr(logger, "warning"):
        logger.warning("ADetailer detail backend returned blank detail output; using local enhance fallback.")
    return _enhance_crop(original_crop, strength=strength, prompt=prompt)


def _is_blank_placeholder_output(original_crop, edited_crop) -> bool:
    return _max_pixel_value(edited_crop) <= 2 and _max_pixel_value(original_crop) > 16


def _max_pixel_value(image) -> int:
    if hasattr(image, "getextrema"):
        extrema = image.convert("RGB").getextrema()
        return max(channel_max for _channel_min, channel_max in extrema)
    values = image.getdata()
    max_value = 0
    for value in values:
        if isinstance(value, tuple):
            max_value = max(max_value, *(int(channel) for channel in value[:3]))
        else:
            max_value = max(max_value, int(value))
    return max_value


def _run_diffusers_inpaint(
    crop,
    mask,
    prompt: str,
    negative_prompt: str,
    strength: float,
    steps: int,
    cfg_scale: float,
    scheduler: DetailScheduler,
    seed: int,
    model: str,
    inference_size: tuple[int, int],
):
    if not model.strip():
        raise RuntimeError("ADetailer diffusers_inpaint backend resolved to an empty model id/path.")
    try:
        import torch
        from diffusers import AutoPipelineForInpainting
    except ImportError as error:
        raise RuntimeError(
            "ADetailer diffusers_inpaint backend requires diffusers and torch in the InvokeAI environment."
        ) from error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    cache_key = (model, device, str(dtype))
    pipe = _PIPELINE_CACHE.get(cache_key)
    if pipe is None:
        pipe = _load_inpaint_pipeline(AutoPipelineForInpainting, model, dtype)
        pipe = pipe.to(device)
        _PIPELINE_CACHE[cache_key] = pipe
    _apply_diffusers_scheduler(pipe, scheduler)
    generator = torch.Generator(device=device).manual_seed(seed) if seed > 0 else None
    original_size = crop.size
    from PIL import Image

    lanczos = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    nearest = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
    inference_crop = crop.convert("RGB").resize(inference_size, resample=lanczos)
    inference_mask = mask.convert("L").resize(inference_size, resample=nearest)
    output = pipe(
        prompt=prompt or "highly detailed, natural, clean",
        negative_prompt=negative_prompt or None,
        image=inference_crop,
        mask_image=inference_mask,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        generator=generator,
    )
    return output.images[0].convert("RGB").resize(original_size, resample=lanczos)


def _load_inpaint_pipeline(pipeline_cls, model: str, dtype):
    try:
        return pipeline_cls.from_pretrained(
            model,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
    except TypeError:
        return pipeline_cls.from_pretrained(model, torch_dtype=dtype)


def _apply_diffusers_scheduler(pipe, scheduler: DetailScheduler) -> None:
    if scheduler == "default":
        return
    try:
        if scheduler == "euler":
            from diffusers import EulerDiscreteScheduler

            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        elif scheduler == "euler_a":
            from diffusers import EulerAncestralDiscreteScheduler

            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif scheduler == "ddim":
            from diffusers import DDIMScheduler

            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        elif scheduler == "dpmpp_2m":
            from diffusers import DPMSolverMultistepScheduler

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    except Exception as error:
        raise RuntimeError(f"Failed to configure ADetailer scheduler {scheduler!r}.") from error


def _enhance_crop(crop, strength: float, prompt: str):
    from PIL import ImageEnhance, ImageFilter

    detail = crop.filter(ImageFilter.UnsharpMask(radius=1.2, percent=round(80 + 180 * strength), threshold=3))
    detail = ImageEnhance.Sharpness(detail).enhance(1.0 + strength)
    detail = ImageEnhance.Contrast(detail).enhance(1.0 + 0.25 * strength)
    if any(word in prompt.lower() for word in ("bright", "clear", "vivid", "sharp")):
        detail = ImageEnhance.Brightness(detail).enhance(1.0 + 0.08 * strength)
    return detail.convert("RGB")


def _expand_for_runtime(box, image_size, padding: int, crop_factor: float):
    from .utils import expand_box

    return expand_box(box=box, image_size=image_size, padding=padding, crop_factor=crop_factor)


def _pad_detections_for_mask(
    detections: list[Detection],
    image_size: tuple[int, int],
    bbox_padding: int,
    crop_factor: float,
) -> list[Detection]:
    return [
        Detection(
            box=_expand_for_runtime(
                detection.box,
                image_size,
                padding=bbox_padding,
                crop_factor=crop_factor,
            ),
            confidence=detection.confidence,
            class_name=detection.class_name,
            mask=detection.mask,
        )
        for detection in detections
    ]
