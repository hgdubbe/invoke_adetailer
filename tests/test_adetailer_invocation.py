import importlib
import sys
import types
import unittest
from types import SimpleNamespace

from nodes.adetailer.utils import Detection


class FakeImage:
    def __init__(self, size, fill=0, mode="RGB"):
        self.size = size
        self.mode = mode
        self.width = size[0]
        self.height = size[1]
        self.pixels = [fill] * (self.width * self.height)

    def convert(self, mode):
        image = self.copy()
        image.mode = mode
        return image

    def copy(self):
        image = FakeImage(self.size, mode=self.mode)
        image.pixels = list(self.pixels)
        return image

    def crop(self, box):
        x1, y1, x2, y2 = box
        image = FakeImage((x2 - x1, y2 - y1), mode=self.mode)
        image.pixels = [
            self.pixels[y * self.width + x]
            for y in range(y1, y2)
            for x in range(x1, x2)
        ]
        return image

    def paste(self, source, box, mask=None):
        x1, y1, x2, y2 = box
        for yy, y in enumerate(range(y1, y2)):
            for xx, x in enumerate(range(x1, x2)):
                source_value = source if isinstance(source, int) else source.pixels[yy * source.width + xx]
                mask_value = 255 if mask is None else mask.pixels[yy * source.width + xx]
                if mask_value > 0:
                    self.pixels[y * self.width + x] = source_value

    def filter(self, _filter):
        return self

    def resize(self, size, resample=None):
        return FakeImage(size, fill=self.pixels[0] if self.pixels else 0, mode=self.mode)

    def putdata(self, values):
        self.pixels = list(values)

    def getdata(self):
        return list(self.pixels)


class FakeImageModule:
    @staticmethod
    def new(mode, size, color):
        return FakeImage(size=size, fill=color, mode=mode)


class FakeImageFilter:
    @staticmethod
    def GaussianBlur(radius):
        return ("GaussianBlur", radius)


class FakeImagesService:
    def __init__(self, image):
        self.image = image
        self.saved = []

    def get_pil(self, image_name):
        self.requested_name = image_name
        return self.image.copy()

    def get_dto(self, image_name):
        return SimpleNamespace(image_name=image_name)

    def save(self, image):
        self.saved.append(image)
        return SimpleNamespace(image_name="adetailer-output")


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


class ADetailerInvocationTests(unittest.TestCase):
    def test_adetailer_invocation_detects_edits_saves_and_returns_image(self):
        with stubbed_runtime_module() as adetailer:
            calls = []
            detector_calls = []

            def fake_detector(image, model_path, confidence, iou):
                detector_calls.append(model_path)
                return [
                    Detection(box=(1, 1, 3, 3), confidence=0.9, class_name="face"),
                    Detection(box=(4, 4, 6, 6), confidence=0.8, class_name="hand"),
                ]

            def fake_backend(**kwargs):
                calls.append(kwargs)
                edited = kwargs["crop"].copy()
                edited.pixels = [100 + len(calls)] * len(edited.pixels)
                return edited

            adetailer.DETECTOR = fake_detector
            adetailer.DETAIL_BACKEND = fake_backend
            node = adetailer.ADetailerInvocation()
            node.image = SimpleNamespace(image_name="input-image")
            node.prompt = "fix face"
            node.negative_prompt = "bad"
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
            node.denoise_strength = 0.35
            node.steps = 20
            node.cfg_scale = 7.0
            node.scheduler = "default"
            node.seed = 123
            node.detail_iterations = 1
            node.detail_opacity = 1.0
            node.detail_backend = "enhance"
            node.diffusers_model = ""

            source = FakeImage((8, 8), fill=0)
            context = SimpleNamespace(images=FakeImagesService(source), logger=FakeLogger())

            output = node.invoke(context)

            self.assertEqual(output.image_name, "adetailer-output")
            self.assertEqual(len(context.images.saved), 1)
            self.assertEqual(detector_calls, ["face_yolov8n.pt"])
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["seed"], 123)
            self.assertEqual(context.images.saved[0].pixels[1 * 8 + 1], 101)
            self.assertEqual(context.images.saved[0].pixels[4 * 8 + 4], 0)
            self.assertIn("processed 1", context.logger.messages[-1])

    def test_adetailer_invocation_increments_seed_for_multiple_regions(self):
        with stubbed_runtime_module() as adetailer:
            calls = []
            detector_calls = []

            def fake_detector(image, model_path, confidence, iou):
                detector_calls.append(model_path)
                return [
                    Detection(box=(1, 1, 3, 3), confidence=0.9, class_name="face"),
                    Detection(box=(4, 4, 6, 6), confidence=0.8, class_name="face"),
                ]

            def fake_backend(**kwargs):
                calls.append(kwargs)
                edited = kwargs["crop"].copy()
                edited.pixels = [kwargs["seed"] - 200] * len(edited.pixels)
                return edited

            adetailer.DETECTOR = fake_detector
            adetailer.DETAIL_BACKEND = fake_backend
            node = adetailer.ADetailerInvocation()
            node.image = SimpleNamespace(image_name="input-image")
            node.prompt = "fix face"
            node.negative_prompt = ""
            node.target = "face"
            node.model_path = "/models/face.pt"
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
            node.denoise_strength = 0.35
            node.steps = 20
            node.cfg_scale = 7.0
            node.scheduler = "default"
            node.seed = 201
            node.detail_iterations = 1
            node.detail_opacity = 1.0
            node.detail_backend = "enhance"
            node.diffusers_model = ""

            context = SimpleNamespace(images=FakeImagesService(FakeImage((8, 8), fill=0)), logger=FakeLogger())

            node.invoke(context)

            self.assertEqual(detector_calls, ["/models/face.pt"])
            self.assertEqual([call["seed"] for call in calls], [201, 202])
            self.assertEqual(context.images.saved[0].pixels[1 * 8 + 1], 1)
            self.assertEqual(context.images.saved[0].pixels[4 * 8 + 4], 2)

    def test_adetailer_invocation_increments_seed_for_multiple_passes(self):
        with stubbed_runtime_module() as adetailer:
            calls = []

            def fake_detector(image, model_path, confidence, iou):
                return [
                    Detection(box=(1, 1, 3, 3), confidence=0.9, class_name="face"),
                    Detection(box=(4, 4, 6, 6), confidence=0.8, class_name="face"),
                ]

            def fake_backend(**kwargs):
                calls.append(kwargs)
                edited = kwargs["crop"].copy()
                edited.pixels = [kwargs["seed"] - 300] * len(edited.pixels)
                return edited

            adetailer.DETECTOR = fake_detector
            adetailer.DETAIL_BACKEND = fake_backend
            node = adetailer.ADetailerInvocation()
            node.image = SimpleNamespace(image_name="input-image")
            node.prompt = "fix face"
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
            node.denoise_strength = 0.35
            node.steps = 20
            node.cfg_scale = 7.0
            node.scheduler = "default"
            node.seed = 301
            node.detail_iterations = 2
            node.detail_opacity = 1.0
            node.detail_backend = "enhance"
            node.diffusers_model = ""

            context = SimpleNamespace(images=FakeImagesService(FakeImage((8, 8), fill=0)), logger=FakeLogger())

            node.invoke(context)

            self.assertEqual([call["seed"] for call in calls], [301, 302, 303, 304])
            self.assertEqual(context.images.saved[0].pixels[1 * 8 + 1], 2)
            self.assertEqual(context.images.saved[0].pixels[4 * 8 + 4], 4)


class stubbed_runtime_module:
    def __enter__(self):
        self.module_names = [
            "PIL",
            "PIL.Image",
            "PIL.ImageFilter",
            "invokeai",
            "invokeai.invocation_api",
            "invokeai.app",
            "invokeai.app.invocations",
            "invokeai.app.invocations.primitives",
            "nodes.adetailer",
            "nodes.adetailer.adetailer",
        ]
        self.previous = {name: sys.modules.get(name) for name in self.module_names}
        _install_stubs()
        return importlib.import_module("nodes.adetailer.adetailer")

    def __exit__(self, exc_type, exc_value, traceback):
        for name, module in self.previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _install_stubs():
    pil = types.ModuleType("PIL")
    pil_image = types.ModuleType("PIL.Image")
    pil_filter = types.ModuleType("PIL.ImageFilter")
    pil_image.new = FakeImageModule.new
    pil_image.Image = FakeImage
    pil_filter.GaussianBlur = FakeImageFilter.GaussianBlur

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

    def invocation(*args, **kwargs):
        return lambda cls: cls

    invocation_api.BaseInvocation = BaseInvocation
    invocation_api.ImageField = ImageField
    invocation_api.InputField = lambda *args, **kwargs: None
    invocation_api.InvocationContext = InvocationContext
    invocation_api.invocation = invocation
    primitives.ImageOutput = ImageOutput

    sys.modules["PIL"] = pil
    sys.modules["PIL.Image"] = pil_image
    sys.modules["PIL.ImageFilter"] = pil_filter
    sys.modules["invokeai"] = types.ModuleType("invokeai")
    sys.modules["invokeai.invocation_api"] = invocation_api
    sys.modules["invokeai.app"] = types.ModuleType("invokeai.app")
    sys.modules["invokeai.app.invocations"] = types.ModuleType("invokeai.app.invocations")
    sys.modules["invokeai.app.invocations.primitives"] = primitives
    sys.modules.pop("nodes.adetailer", None)
    sys.modules.pop("nodes.adetailer.adetailer", None)


if __name__ == "__main__":
    unittest.main()
