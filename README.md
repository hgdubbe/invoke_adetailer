import importlib
import sys
import types
import unittest

from nodes.adetailer.utils import Detection, DetailRegion


class FakeImageModule:
    @staticmethod
    def new(mode, size, color):
        return FakeMask(size=size, color=color)


class FakeMask:
    def __init__(self, size, color):
        self.size = size
        self.values = [color] * (size[0] * size[1])

    def paste(self, value, box):
        width, _height = self.size
        x1, y1, x2, y2 = box
        for y in range(y1, y2):
            for x in range(x1, x2):
                self.values[y * width + x] = value

    def putdata(self, values):
        self.values = list(values)


class ADetailerRuntimeHelperTests(unittest.TestCase):
    def test_region_mask_image_honors_mask_padding_for_segmentation_masks(self):
        with stubbed_adetailer_module() as adetailer:
            region = DetailRegion(
                detection=Detection(
                    box=(2, 2, 3, 3),
                    confidence=0.9,
                    class_name="face",
                    mask=[
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 255, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                ),
                crop_box=(0, 0, 5, 5),
                mask_box=(1, 1, 4, 4),
            )

            mask = adetailer._region_mask_image(FakeImageModule, (5, 5), region, mask_padding=1, mask_erosion=0)

            self.assertEqual(mask.values[1 * 5 + 1], 255)
            self.assertEqual(mask.values[3 * 5 + 3], 255)
            self.assertEqual(mask.values[0], 0)

    def test_pad_detections_for_mask_preserves_segmentation_mask(self):
        with stubbed_adetailer_module() as adetailer:
            source_mask = [[0, 255], [0, 0]]
            detections = [Detection(box=(1, 1, 2, 2), confidence=0.9, class_name="face", mask=source_mask)]

            padded = adetailer._pad_detections_for_mask(
                detections=detections,
                image_size=(8, 8),
                bbox_padding=1,
                crop_factor=1.0,
            )

            self.assertEqual(padded[0].mask, source_mask)
            self.assertEqual(padded[0].box, (0, 0, 3, 3))

    def test_apply_detail_opacity_returns_original_or_edited_at_extremes(self):
        with stubbed_adetailer_module() as adetailer:
            original = object()
            edited = object()

            self.assertIs(adetailer._apply_detail_opacity(original, edited, 0.0), original)
            self.assertIs(adetailer._apply_detail_opacity(original, edited, 1.0), edited)

    def test_ultralytics_import_error_message_identifies_missing_package(self):
        with stubbed_adetailer_module() as adetailer:
            error = ModuleNotFoundError("No module named 'ultralytics'", name="ultralytics")

            message = adetailer._ultralytics_import_error_message(error)

            self.assertIn("ultralytics is not installed", message)
            self.assertIn(sys.executable, message)

    def test_ultralytics_import_error_message_preserves_dependency_failure(self):
        with stubbed_adetailer_module() as adetailer:
            error = ModuleNotFoundError("No module named 'cv2'", name="cv2")

            message = adetailer._ultralytics_import_error_message(error)

            self.assertIn("dependency failed to import", message)
            self.assertIn("cv2", message)
            self.assertIn(sys.executable, message)


class stubbed_adetailer_module:
    def __enter__(self):
        self.module_names = [
            "invokeai",
            "invokeai.invocation_api",
            "invokeai.app",
            "invokeai.app.invocations",
            "invokeai.app.invocations.primitives",
            "nodes.adetailer",
            "nodes.adetailer.adetailer",
        ]
        self.previous = {name: sys.modules.get(name) for name in self.module_names}
        return _import_adetailer_with_stubs()

    def __exit__(self, exc_type, exc_value, traceback):
        for name, module in self.previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _import_adetailer_with_stubs():
    module_names = [
        "invokeai",
        "invokeai.invocation_api",
        "invokeai.app",
        "invokeai.app.invocations",
        "invokeai.app.invocations.primitives",
    ]
    for name in module_names:
        sys.modules.pop(name, None)

    invocation_api = types.ModuleType("invokeai.invocation_api")
    primitives = types.ModuleType("invokeai.app.invocations.primitives")

    class BaseInvocation:
        pass

    class ImageField:
        pass

    class InvocationContext:
        pass

    class ImageOutput:
        pass

    def invocation(*args, **kwargs):
        return lambda cls: cls

    invocation_api.BaseInvocation = BaseInvocation
    invocation_api.ImageField = ImageField
    invocation_api.InputField = lambda *args, **kwargs: None
    invocation_api.InvocationContext = InvocationContext
    invocation_api.invocation = invocation
    primitives.ImageOutput = ImageOutput

    sys.modules["invokeai"] = types.ModuleType("invokeai")
    sys.modules["invokeai.invocation_api"] = invocation_api
    sys.modules["invokeai.app"] = types.ModuleType("invokeai.app")
    sys.modules["invokeai.app.invocations"] = types.ModuleType("invokeai.app.invocations")
    sys.modules["invokeai.app.invocations.primitives"] = primitives
    sys.modules.pop("nodes.adetailer.adetailer", None)
    return importlib.import_module("nodes.adetailer.adetailer")


if __name__ == "__main__":
    unittest.main()
