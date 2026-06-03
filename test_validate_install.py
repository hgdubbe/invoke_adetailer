import importlib
import importlib.util
import sys
import types
import unittest
from types import SimpleNamespace

from nodes.adetailer.utils import Detection


@unittest.skipIf(importlib.util.find_spec("PIL") is None, "Pillow is not installed")
class ADetailerPillowSmokeTests(unittest.TestCase):
    def test_adetailer_enhance_backend_edits_and_saves_real_pil_image(self):
        with stubbed_invokeai_module() as adetailer:
            from PIL import Image

            adetailer.DETECTOR = lambda image, model_path, confidence, iou: [
                Detection(box=(1, 1, 5, 5), confidence=0.9, class_name="face")
            ]
            adetailer.DETAIL_BACKEND = None

            node = adetailer.ADetailerInvocation()
            node.image = SimpleNamespace(image_name="input-image")
            node.prompt = "bright sharp face"
            node.negative_prompt = ""
            node.target = "face"
            node.model_path = "auto"
            node.confidence = 0.3
            node.iou = 0.5
            node.class_filter = ""
            node.max_detections = 8
            node.sort_by = "confidence"
            node.min_area_ratio = 0.0
            node.max_area_ratio = 1.0
            node.bbox_padding = 0
            node.mask_padding = 0
            node.mask_erosion = 0
            node.mask_offset_x = 0
            node.mask_offset_y = 0
            node.mask_blur = 0
            node.crop_factor = 1.0
            node.guide_size = 0
            node.guide_size_for = "bbox"
            node.max_size = 0
            node.denoise_strength = 1.0
            node.steps = 20
            node.cfg_scale = 7.0
            node.scheduler = "default"
            node.seed = 0
            node.detail_iterations = 1
            node.detail_opacity = 1.0
            node.detail_backend = "enhance"
            node.diffusers_model = "auto"

            source = Image.new("RGB", (8, 8), (100, 100, 100))
            context = SimpleNamespace(images=FakeImagesService(source), logger=FakeLogger())

            output = node.invoke(context)

            self.assertEqual(output.image_name, "adetailer-output")
            self.assertEqual(len(context.images.saved), 1)
            self.assertGreater(context.images.saved[0].getpixel((2, 2))[0], 100)
            self.assertEqual(context.images.saved[0].getpixel((7, 7)), (100, 100, 100))

    def test_adetailer_replaces_blank_backend_output_with_enhanced_crop(self):
        with stubbed_invokeai_module() as adetailer:
            from PIL import Image

            adetailer.DETECTOR = lambda image, model_path, confidence, iou: [
                Detection(box=(1, 1, 5, 5), confidence=0.9, class_name="face")
            ]
            adetailer.DETAIL_BACKEND = lambda **kwargs: Image.new("RGB", kwargs["crop"].size, (0, 0, 0))

            node = adetailer.ADetailerInvocation()
            node.image = SimpleNamespace(image_name="input-image")
            node.prompt = "bright sharp face"
            node.negative_prompt = ""
            node.target = "face"
            node.model_path = "auto"
            node.confidence = 0.3
            node.iou = 0.5
            node.class_filter = ""
            node.max_detections = 8
            node.sort_by = "confidence"
            node.min_area_ratio = 0.0
            node.max_area_ratio = 1.0
            node.bbox_padding = 0
            node.mask_padding = 0
            node.mask_erosion = 0
            node.mask_offset_x = 0
            node.mask_offset_y = 0
            node.mask_blur = 0
            node.crop_factor = 1.0
            node.guide_size = 0
            node.guide_size_for = "bbox"
            node.max_size = 0
            node.denoise_strength = 1.0
            node.steps = 20
            node.cfg_scale = 7.0
            node.scheduler = "default"
            node.seed = 0
            node.detail_iterations = 1
            node.detail_opacity = 1.0
            node.detail_backend = "diffusers_inpaint"
            node.diffusers_model = "auto"

            source = Image.new("RGB", (8, 8), (100, 100, 100))
            context = SimpleNamespace(images=FakeImagesService(source), logger=FakeLogger())

            node.invoke(context)

            self.assertGreater(context.images.saved[0].getpixel((2, 2))[0], 100)
            self.assertNotEqual(context.images.saved[0].getpixel((2, 2)), (0, 0, 0))
            self.assertIn("blank detail output", context.logger.warning_message)


class FakeImagesService:
    def __init__(self, image):
        self.image = image
        self.saved = []

    def get_pil(self, image_name):
        return self.image.copy()

    def get_dto(self, image_name):
        return SimpleNamespace(image_name=image_name)

    def save(self, image):
        self.saved.append(image)
        return SimpleNamespace(image_name="adetailer-output")


class FakeLogger:
    def info(self, message):
        self.last_message = message

    def warning(self, message):
        self.warning_message = message


class stubbed_invokeai_module:
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
        _install_invokeai_stubs()
        return importlib.import_module("nodes.adetailer.adetailer")

    def __exit__(self, exc_type, exc_value, traceback):
        for name, module in self.previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _install_invokeai_stubs():
    invocation_api = types.ModuleType("invokeai.invocation_api")
    primitives = types.ModuleType("invokeai.app.invocations.primitives")

    class BaseInvocation:
        pass

    class ImageField:
        pass

    class InvocationContext:
        pass

    class ImageOutput:
        @classmethod
        def build(cls, image_dto):
            return image_dto

    invocation_api.BaseInvocation = BaseInvocation
    invocation_api.ImageField = ImageField
    invocation_api.InputField = lambda *args, **kwargs: None
    invocation_api.InvocationContext = InvocationContext
    invocation_api.invocation = lambda *args, **kwargs: lambda cls: cls
    primitives.ImageOutput = ImageOutput

    sys.modules["invokeai"] = types.ModuleType("invokeai")
    sys.modules["invokeai.invocation_api"] = invocation_api
    sys.modules["invokeai.app"] = types.ModuleType("invokeai.app")
    sys.modules["invokeai.app.invocations"] = types.ModuleType("invokeai.app.invocations")
    sys.modules["invokeai.app.invocations.primitives"] = primitives
    sys.modules.pop("nodes.adetailer", None)
    sys.modules.pop("nodes.adetailer.adetailer", None)


if __name__ == "__main__":
    unittest.main()
