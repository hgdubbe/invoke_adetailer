#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUIRED_FOR_NODE_LOAD = ["invokeai"]
REQUIRED_FOR_IMAGE_EDIT = ["PIL", "ultralytics"]
REQUIRED_FOR_DIFFUSERS = ["diffusers", "torch", "transformers", "accelerate", "safetensors"]
OPTIONAL_DOWNLOAD = ["huggingface_hub"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate InvokeAI ADetailer node pack installation.")
    parser.add_argument(
        "--download-detectors",
        action="store_true",
        help="Resolve and download/cache the built-in ADetailer detector presets.",
    )
    args = parser.parse_args()

    print("InvokeAI ADetailer validation")
    print("============================")

    ok = True
    ok &= _check_modules("InvokeAI node loading", REQUIRED_FOR_NODE_LOAD)
    ok &= _check_modules("Image detection/editing", REQUIRED_FOR_IMAGE_EDIT)
    ok &= _check_import_object("Ultralytics YOLO import", "ultralytics", "YOLO")
    ok &= _check_modules("Prompt-driven diffusers backend", REQUIRED_FOR_DIFFUSERS)
    _check_modules("Detector download helper", OPTIONAL_DOWNLOAD, required=False)

    try:
        from nodes.adetailer import ADetailerCompositeInvocation, ADetailerInvocation, ADetailerMaskInvocation
    except Exception as error:
        ok = False
        print(f"[FAIL] Node pack import: {error}")
    else:
        names = [ADetailerInvocation, ADetailerMaskInvocation, ADetailerCompositeInvocation]
        if any(name is None for name in names):
            ok = False
            print("[FAIL] Node pack import: one or more invocation classes were not loaded")
        else:
            print("[ OK ] Node pack import")

    if args.download_detectors:
        ok &= _download_detector_presets()

    print()
    if ok:
        print("Validation passed.")
        return 0
    print("Validation failed. Install missing dependencies in the InvokeAI Python environment.")
    return 1


def _check_modules(title: str, module_names: list[str], required: bool = True) -> bool:
    missing = [name for name in module_names if importlib.util.find_spec(name) is None]
    if missing:
        status = "FAIL" if required else "WARN"
        print(f"[{status}] {title}: missing {', '.join(missing)}")
        return not required
    print(f"[ OK ] {title}")
    return True


def _check_import_object(title: str, module_name: str, object_name: str, required: bool = True) -> bool:
    try:
        module = importlib.import_module(module_name)
        getattr(module, object_name)
    except Exception as error:
        status = "FAIL" if required else "WARN"
        print(f"[{status}] {title}: {error}")
        return not required
    print(f"[ OK ] {title}")
    return True


def _download_detector_presets() -> bool:
    try:
        from nodes.adetailer.utils import TARGET_MODEL_PRESETS, resolve_detector_weight_path
    except Exception as error:
        print(f"[FAIL] Detector preset resolver import: {error}")
        return False

    ok = True
    for target, filename in sorted(TARGET_MODEL_PRESETS.items()):
        if target.endswith("s") or target in {"people", "body"}:
            continue
        try:
            path = resolve_detector_weight_path(filename)
        except Exception as error:
            ok = False
            print(f"[FAIL] Detector preset {target}: {error}")
        else:
            exists = Path(path).exists()
            status = " OK " if exists else "WARN"
            print(f"[{status}] Detector preset {target}: {path}")
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
