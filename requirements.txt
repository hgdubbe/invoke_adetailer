import unittest
from unittest.mock import patch

from nodes.adetailer.utils import (
    ADETAILER_HF_REPO,
    Detection,
    build_combined_mask,
    detail_inference_size,
    erode_mask,
    expand_box,
    filter_detections,
    plan_detail_regions,
    resolve_diffusers_model,
    resolve_detector_model,
    resolve_detector_weight_path,
    translate_mask,
)


class ADetailerUtilsTests(unittest.TestCase):
    def test_filter_detections_keeps_confident_allowed_classes_and_limits_count(self):
        detections = [
            Detection(box=(0, 0, 10, 10), confidence=0.91, class_name="face"),
            Detection(box=(5, 5, 15, 15), confidence=0.32, class_name="face"),
            Detection(box=(1, 1, 8, 8), confidence=0.84, class_name="hand"),
            Detection(box=(2, 2, 9, 9), confidence=0.99, class_name="person"),
        ]

        kept = filter_detections(
            detections,
            confidence_threshold=0.5,
            class_filter="face, hand",
            max_detections=1,
        )

        self.assertEqual([d.class_name for d in kept], ["face"])
        self.assertEqual(kept[0].confidence, 0.91)

    def test_filter_detections_rejects_boxes_outside_area_ratio_limits(self):
        detections = [
            Detection(box=(0, 0, 2, 2), confidence=0.9, class_name="face"),
            Detection(box=(0, 0, 5, 5), confidence=0.9, class_name="face"),
            Detection(box=(0, 0, 10, 10), confidence=0.9, class_name="face"),
        ]

        kept = filter_detections(
            detections,
            confidence_threshold=0.5,
            image_size=(10, 10),
            min_area_ratio=0.1,
            max_area_ratio=0.8,
        )

        self.assertEqual([d.box for d in kept], [(0, 0, 5, 5)])

    def test_expand_box_adds_padding_and_clamps_to_image_bounds(self):
        box = expand_box(
            box=(10, 12, 30, 42),
            image_size=(64, 48),
            padding=8,
            crop_factor=1.5,
        )

        self.assertEqual(box, (0, 0, 43, 48))

    def test_build_combined_mask_fills_boxes_and_dilates_edges(self):
        detections = [
            Detection(box=(2, 2, 5, 5), confidence=0.9, class_name="face"),
            Detection(box=(7, 1, 9, 3), confidence=0.8, class_name="hand"),
        ]

        mask = build_combined_mask(
            detections=detections,
            image_size=(10, 8),
            mask_padding=1,
        )

        self.assertEqual(len(mask), 8)
        self.assertEqual(len(mask[0]), 10)
        self.assertEqual(mask[1][1], 255)
        self.assertEqual(mask[5][5], 255)
        self.assertEqual(mask[0][6], 255)
        self.assertEqual(mask[3][9], 255)
        self.assertEqual(mask[7][0], 0)

    def test_build_combined_mask_uses_detection_mask_when_available(self):
        detection_mask = [
            [0, 0, 0, 0],
            [0, 255, 255, 0],
            [0, 0, 255, 0],
            [0, 0, 0, 0],
        ]
        detections = [
            Detection(box=(0, 0, 4, 4), confidence=0.9, class_name="face", mask=detection_mask),
        ]

        mask = build_combined_mask(detections=detections, image_size=(4, 4), mask_padding=0)

        self.assertEqual(mask[1][1], 255)
        self.assertEqual(mask[1][2], 255)
        self.assertEqual(mask[2][2], 255)
        self.assertEqual(mask[0][0], 0)
        self.assertEqual(mask[3][3], 0)

    def test_build_combined_mask_dilates_detection_mask(self):
        detection_mask = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 255, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        detections = [
            Detection(box=(2, 2, 3, 3), confidence=0.9, class_name="face", mask=detection_mask),
        ]

        mask = build_combined_mask(detections=detections, image_size=(5, 5), mask_padding=1)

        self.assertEqual(mask[1][1], 255)
        self.assertEqual(mask[1][3], 255)
        self.assertEqual(mask[3][1], 255)
        self.assertEqual(mask[3][3], 255)
        self.assertEqual(mask[0][0], 0)

    def test_translate_mask_moves_pixels_and_clips_at_edges(self):
        mask = [
            [0, 255, 0],
            [0, 0, 255],
            [0, 0, 0],
        ]

        translated = translate_mask(mask, offset_x=1, offset_y=1)

        self.assertEqual(translated, [[0, 0, 0], [0, 0, 255], [0, 0, 0]])

    def test_erode_mask_removes_edge_pixels(self):
        mask = [
            [0, 0, 0, 0, 0],
            [0, 255, 255, 255, 0],
            [0, 255, 255, 255, 0],
            [0, 255, 255, 255, 0],
            [0, 0, 0, 0, 0],
        ]

        eroded = erode_mask(mask, radius=1)

        self.assertEqual(
            eroded,
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 255, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        )

    def test_erode_mask_treats_image_bounds_as_empty(self):
        mask = [
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
        ]

        eroded = erode_mask(mask, radius=1)

        self.assertEqual(
            eroded,
            [
                [0, 0, 0],
                [0, 255, 0],
                [0, 0, 0],
            ],
        )

    def test_plan_detail_regions_creates_crop_and_mask_boxes(self):
        detections = [Detection(box=(20, 20, 40, 50), confidence=0.95, class_name="face")]

        regions = plan_detail_regions(
            detections=detections,
            image_size=(96, 96),
            bbox_padding=8,
            mask_padding=4,
            crop_factor=2.0,
        )

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].crop_box, (2, 0, 58, 73))
        self.assertEqual(regions[0].mask_box, (16, 16, 44, 54))

    def test_detail_inference_size_scales_small_crop_to_guide_size(self):
        size = detail_inference_size(
            crop_size=(160, 96),
            bbox_size=(80, 48),
            guide_size=512,
            guide_size_for="bbox",
            max_size=768,
            multiple=8,
        )

        self.assertEqual(size, (768, 456))

    def test_detail_inference_size_respects_max_size(self):
        size = detail_inference_size(
            crop_size=(640, 384),
            bbox_size=(640, 384),
            guide_size=1024,
            guide_size_for="crop",
            max_size=512,
            multiple=8,
        )

        self.assertEqual(size, (512, 304))

    def test_resolve_detector_model_uses_target_preset_when_model_path_is_auto(self):
        self.assertEqual(resolve_detector_model("face", "auto"), "face_yolov8n.pt")
        self.assertEqual(resolve_detector_model("hands", ""), "hand_yolov8n.pt")
        self.assertEqual(resolve_detector_model("person", "AUTO"), "person_yolov8n-seg.pt")

    def test_resolve_detector_model_keeps_explicit_model_path(self):
        self.assertEqual(resolve_detector_model("face", "/models/custom.pt"), "/models/custom.pt")

    def test_resolve_detector_weight_path_keeps_existing_or_non_preset_paths(self):
        with patch("nodes.adetailer.utils.Path.exists", return_value=True):
            self.assertEqual(resolve_detector_weight_path("/models/custom.pt"), "/models/custom.pt")
        self.assertEqual(resolve_detector_weight_path("custom.pt"), "custom.pt")

    def test_resolve_detector_weight_path_downloads_known_preset_with_hf_hub(self):
        calls = []

        def fake_hf_download(repo_id, filename, cache_dir):
            calls.append((repo_id, filename, cache_dir))
            return "/cache/face_yolov8n.pt"

        with patch("nodes.adetailer.utils.Path.exists", return_value=False):
            with patch("nodes.adetailer.utils._hf_hub_download", fake_hf_download):
                self.assertEqual(resolve_detector_weight_path("face_yolov8n.pt"), "/cache/face_yolov8n.pt")

        self.assertEqual(calls[0][0], ADETAILER_HF_REPO)
        self.assertEqual(calls[0][1], "face_yolov8n.pt")

    def test_resolve_detector_weight_path_falls_back_to_direct_download_when_hub_fails(self):
        direct_calls = []

        def fake_hf_download(repo_id, filename, cache_dir):
            raise RuntimeError("hub unavailable")

        def fake_direct_download(filename):
            direct_calls.append(filename)
            return "/cache/hand_yolov8n.pt"

        with patch("nodes.adetailer.utils.Path.exists", return_value=False):
            with patch("nodes.adetailer.utils._hf_hub_download", fake_hf_download):
                with patch("nodes.adetailer.utils._download_adetailer_weight_direct", fake_direct_download):
                    self.assertEqual(resolve_detector_weight_path("hand_yolov8n.pt"), "/cache/hand_yolov8n.pt")

        self.assertEqual(direct_calls, ["hand_yolov8n.pt"])

    def test_resolve_diffusers_model_uses_env_override_for_auto(self):
        with patch.dict("os.environ", {"ADETAILER_DIFFUSERS_MODEL": "/models/inpaint"}, clear=False):
            self.assertEqual(resolve_diffusers_model("auto"), "/models/inpaint")
            self.assertEqual(resolve_diffusers_model(""), "/models/inpaint")

    def test_resolve_diffusers_model_uses_default_for_auto(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolve_diffusers_model("auto"), "runwayml/stable-diffusion-inpainting")

    def test_resolve_diffusers_model_ignores_empty_env_override(self):
        with patch.dict("os.environ", {"ADETAILER_DIFFUSERS_MODEL": " "}, clear=True):
            self.assertEqual(resolve_diffusers_model("auto"), "runwayml/stable-diffusion-inpainting")

    def test_resolve_diffusers_model_keeps_explicit_model(self):
        self.assertEqual(resolve_diffusers_model("/models/custom-inpaint"), "/models/custom-inpaint")


if __name__ == "__main__":
    unittest.main()
