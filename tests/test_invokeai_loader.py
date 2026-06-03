import importlib
import sys
import types
import unittest


class InvokeAILoaderTests(unittest.TestCase):
    def test_node_package_exports_invocations_when_invokeai_api_exists(self):
        module_names = [
            "invokeai",
            "invokeai.invocation_api",
            "invokeai.app",
            "invokeai.app.invocations",
            "invokeai.app.invocations.primitives",
        ]
        previous = {name: sys.modules.get(name) for name in module_names}
        try:
            for name in module_names:
                sys.modules.pop(name, None)

            invokeai = types.ModuleType("invokeai")
            invocation_api = types.ModuleType("invokeai.invocation_api")
            app = types.ModuleType("invokeai.app")
            invocations = types.ModuleType("invokeai.app.invocations")
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
                def decorator(cls):
                    cls._invokeai_invocation = {"args": args, "kwargs": kwargs}
                    return cls

                return decorator

            def input_field(*args, **kwargs):
                return None

            invocation_api.BaseInvocation = BaseInvocation
            invocation_api.ImageField = ImageField
            invocation_api.InputField = input_field
            invocation_api.InvocationContext = InvocationContext
            invocation_api.invocation = invocation
            primitives.ImageOutput = ImageOutput

            sys.modules["invokeai"] = invokeai
            sys.modules["invokeai.invocation_api"] = invocation_api
            sys.modules["invokeai.app"] = app
            sys.modules["invokeai.app.invocations"] = invocations
            sys.modules["invokeai.app.invocations.primitives"] = primitives

            sys.modules.pop("nodes.adetailer", None)
            sys.modules.pop("nodes.adetailer.adetailer", None)
            package = importlib.import_module("nodes.adetailer")

            self.assertEqual(package.ADetailerInvocation._invokeai_invocation["args"][0], "adetailer")
            self.assertEqual(package.ADetailerMaskInvocation._invokeai_invocation["args"][0], "adetailer_mask")
            self.assertEqual(
                package.ADetailerCompositeInvocation._invokeai_invocation["args"][0],
                "adetailer_composite",
            )
            self.assertIs(package.ADetailerInvocation.invoke.__annotations__["return"], ImageOutput)
            self.assertIs(package.ADetailerMaskInvocation.invoke.__annotations__["return"], ImageOutput)
            self.assertIs(package.ADetailerCompositeInvocation.invoke.__annotations__["return"], ImageOutput)
        finally:
            sys.modules.pop("nodes.adetailer", None)
            sys.modules.pop("nodes.adetailer.adetailer", None)
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_node_package_loads_when_image_output_moved_from_primitives_to_image(self):
        module_names = [
            "invokeai",
            "invokeai.invocation_api",
            "invokeai.app",
            "invokeai.app.invocations",
            "invokeai.app.invocations.primitives",
            "invokeai.app.invocations.image",
        ]
        previous = {name: sys.modules.get(name) for name in module_names}
        try:
            for name in module_names:
                sys.modules.pop(name, None)

            invokeai = types.ModuleType("invokeai")
            invocation_api = types.ModuleType("invokeai.invocation_api")
            app = types.ModuleType("invokeai.app")
            invocations = types.ModuleType("invokeai.app.invocations")
            primitives = types.ModuleType("invokeai.app.invocations.primitives")
            image = types.ModuleType("invokeai.app.invocations.image")

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
                def decorator(cls):
                    cls._invokeai_invocation = {"args": args, "kwargs": kwargs}
                    return cls

                return decorator

            def input_field(*args, **kwargs):
                return None

            invocation_api.BaseInvocation = BaseInvocation
            invocation_api.ImageField = ImageField
            invocation_api.InputField = input_field
            invocation_api.InvocationContext = InvocationContext
            invocation_api.invocation = invocation
            image.ImageOutput = ImageOutput

            sys.modules["invokeai"] = invokeai
            sys.modules["invokeai.invocation_api"] = invocation_api
            sys.modules["invokeai.app"] = app
            sys.modules["invokeai.app.invocations"] = invocations
            sys.modules["invokeai.app.invocations.primitives"] = primitives
            sys.modules["invokeai.app.invocations.image"] = image

            sys.modules.pop("nodes.adetailer", None)
            sys.modules.pop("nodes.adetailer.adetailer", None)
            package = importlib.import_module("nodes.adetailer")

            self.assertEqual(package.ADetailerInvocation._invokeai_invocation["args"][0], "adetailer")
            self.assertIs(package.ADetailerInvocation.invoke.__annotations__["return"], ImageOutput)
        finally:
            sys.modules.pop("nodes.adetailer", None)
            sys.modules.pop("nodes.adetailer.adetailer", None)
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


if __name__ == "__main__":
    unittest.main()
