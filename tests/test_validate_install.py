import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from scripts import validate_install


class ValidateInstallTests(unittest.TestCase):
    def test_check_import_object_passes_when_object_imports(self):
        with patch.object(validate_install.importlib, "import_module", return_value=SimpleNamespace(YOLO=object)):
            output = io.StringIO()
            with redirect_stdout(output):
                result = validate_install._check_import_object("Ultralytics YOLO import", "ultralytics", "YOLO")

        self.assertTrue(result)
        self.assertIn("[ OK ] Ultralytics YOLO import", output.getvalue())

    def test_check_import_object_reports_import_error(self):
        error = ModuleNotFoundError("No module named 'cv2'", name="cv2")
        with patch.object(validate_install.importlib, "import_module", side_effect=error):
            output = io.StringIO()
            with redirect_stdout(output):
                result = validate_install._check_import_object("Ultralytics YOLO import", "ultralytics", "YOLO")

        self.assertFalse(result)
        self.assertIn("[FAIL] Ultralytics YOLO import", output.getvalue())
        self.assertIn("cv2", output.getvalue())


if __name__ == "__main__":
    unittest.main()
