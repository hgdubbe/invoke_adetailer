# InvokeAI ADetailer Node Pack

Custom InvokeAI workflow nodes for ADetailer-style automatic detail repair.

The main node is `ADetailer`: give it an image, a target such as `face` or `hand`, a detector model, a prompt, and detail settings. It detects matching regions, plans crops and masks automatically, inpaints each crop, blends the edited crops back into the original image, and outputs a single edited image that can continue through the workflow.

## Nodes

- `ADetailer`: one-node edited-image output. This is the node intended for normal workflows.
- `ADetailer Mask`: advanced helper that outputs the detected mask.
- `ADetailer Composite`: advanced helper that blends a detailed image through a mask.

## Install

Copy this repository's `nodes/adetailer` folder into the root `nodes` folder of your InvokeAI install:

```text
InvokeAI/
  nodes/
    adetailer/
      __init__.py
      adetailer.py
      utils.py
```

Install runtime dependencies into InvokeAI's Python environment:

```bash
pip install -r requirements.txt
```

Restart InvokeAI after copying the files.

The common ADetailer detector presets are downloaded from [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) on first use and cached in `~/.cache/invokeai-adetailer`. The node uses `huggingface_hub` when available and falls back to a direct Hugging Face URL if hub download fails. Set `ADETAILER_MODEL_CACHE` to use a different cache directory.

To validate an install inside InvokeAI's Python environment:

```bash
python scripts/validate_install.py
```

To also resolve/cache the built-in detector presets:

```bash
python scripts/validate_install.py --download-detectors
```

## Basic Usage

Add `ADetailer` to a workflow after the image you want to improve.

Typical face settings:

- `target`: `face`
- `model_path`: `auto`
- `prompt`: `detailed eyes, natural face, sharp skin texture`
- `negative_prompt`: `deformed, blurry, extra teeth, bad eyes`
- `confidence`: `0.3`
- `min_area_ratio`: `0.0`
- `max_area_ratio`: `1.0`
- `bbox_padding`: `32`
- `mask_padding`: `16`
- `mask_erosion`: `0`
- `mask_offset_x`: `0`
- `mask_offset_y`: `0`
- `mask_blur`: `4`
- `crop_factor`: `1.5`
- `guide_size`: `512`
- `guide_size_for`: `bbox`
- `max_size`: `1024`
- `denoise_strength`: `0.35`
- `steps`: `20`
- `cfg_scale`: `7.0`
- `scheduler`: `default`
- `seed`: `0` for random, or a fixed value. Detail passes use `seed`, `seed + 1`, `seed + 2`, and so on across regions and iterations.
- `detail_iterations`: `1`
- `detail_opacity`: `1.0`
- `detail_backend`: `diffusers_inpaint`
- `diffusers_model`: `auto`

Typical hand settings:

- `target`: `hand`
- `model_path`: `auto`
- `prompt`: `natural detailed hands, correct fingers`
- `confidence`: `0.25`
- `min_area_ratio`: `0.0`
- `max_area_ratio`: `1.0`
- `bbox_padding`: `48`
- `mask_padding`: `24`
- `mask_erosion`: `0`
- `mask_offset_x`: `0`
- `mask_offset_y`: `0`
- `crop_factor`: `1.8`
- `guide_size`: `512`
- `guide_size_for`: `bbox`
- `max_size`: `1024`

## Current Backend

This version provides the complete InvokeAI node shape and automatic detect-mask-crop-inpaint-blend workflow. The `ADetailer` edit backend options are:

- `diffusers_inpaint`: prompt-driven inpainting using `diffusers` and `torch`
- `enhance`: sharpness and contrast detail pass
- `cv2_inpaint`: uses InvokeAI's OpenCV inpaint helper if available
- `blur_fill`: simple soft fill for testing masks and blends

`diffusers_inpaint` currently loads the configured pipeline directly instead of reusing InvokeAI's loaded model graph. That makes the node work as a self-contained edited-image node, but it can use more VRAM than a future direct InvokeAI model-manager backend.

When `diffusers_model` is `auto`, the node uses `ADETAILER_DIFFUSERS_MODEL` from the environment if it is set. Otherwise it falls back to `runwayml/stable-diffusion-inpainting`. Set `diffusers_model` to a local path or another Hugging Face inpaint pipeline id to override this per node.

When `model_path` is `auto`, the node selects a detector from `target`: `face` uses `face_yolov8n.pt`, `hand`/`hands` uses `hand_yolov8n.pt`, and `person`/`people`/`body` uses `person_yolov8n-seg.pt`. Set `model_path` to a local path or model filename to override this.
Unknown targets fall back to Ultralytics `yolov8n.pt`; use `class_filter` to select the classes you want in that case.

Detection models with segmentation output use the model mask for blending and still respect `mask_padding` and `mask_erosion`. BBox-only models use the detected rectangle expanded by `mask_padding` and then optionally eroded by `mask_erosion`. Each crop is resized for the detail pass with `guide_size`, `guide_size_for`, and `max_size`, then resized back and blended into the original image.

Use `min_area_ratio` and `max_area_ratio` to ignore detections that are too small or too large relative to the image. Use `mask_offset_x` and `mask_offset_y` to shift masks when a detector consistently lands slightly off target.

Supported diffusion scheduler values are `default`, `euler`, `euler_a`, `ddim`, and `dpmpp_2m`.

## Development

The pure geometry and mask helpers are testable without InvokeAI installed:

```bash
python3 -m unittest tests/test_adetailer_utils.py
```
