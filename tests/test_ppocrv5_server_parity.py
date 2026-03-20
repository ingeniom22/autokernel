from __future__ import annotations

import importlib
import os
import unittest
from pathlib import Path
from typing import List


def _missing_modules() -> List[str]:
    required = ["torch", "numpy", "cv2", "PIL", "yaml", "shapely", "pyclipper"]
    missing = []
    for module_name in required:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    return missing


MISSING_MODULES = _missing_modules()
IMPORT_ERROR = None

if not MISSING_MODULES:
    try:
        import cv2
        import numpy as np
        import torch

        from models.ppocrv5_server import (
            DET_PTH_ENV,
            DET_YAML_RELATIVE,
            REC_PTH_ENV,
            REC_YAML_RELATIVE,
            PPOCRv5ServerDetModel,
            PPOCRv5ServerRecModel,
            default_server_det_weights_path,
            default_server_rec_weights_path,
            ensure_ppocr_on_path,
        )

        PPOCR_ROOT = ensure_ppocr_on_path()
        from pytorchocr.data import transform
        from tools.infer import predict_det, predict_rec, predict_system, pytorchocr_utility
    except Exception as exc:  # pragma: no cover - import-time environment guard
        IMPORT_ERROR = exc


def _build_server_args(det_weights: Path, rec_weights: Path):
    parser = pytorchocr_utility.init_args()
    args = parser.parse_args([])
    args.use_gpu = False
    args.use_angle_cls = False
    args.det_algorithm = "DB"
    args.det_yaml_path = str(PPOCR_ROOT / DET_YAML_RELATIVE)
    args.det_model_path = str(det_weights)
    args.rec_yaml_path = str(PPOCR_ROOT / REC_YAML_RELATIVE)
    args.rec_model_path = str(rec_weights)
    args.rec_image_shape = "3,48,320"
    args.rec_char_dict_path = str(PPOCR_ROOT / "pytorchocr/utils/dict/ppocrv5_dict.txt")
    args.image_dir = str(PPOCR_ROOT / "doc/imgs/1.jpg")
    return args


class TestPPOCRv5ServerParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if MISSING_MODULES:
            raise unittest.SkipTest(
                "Missing PP-OCR runtime dependencies: " + ", ".join(MISSING_MODULES)
            )
        if IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"Could not import PP-OCR integration stack: {IMPORT_ERROR}")

        cls.det_weights = Path(
            os.getenv(DET_PTH_ENV, str(default_server_det_weights_path()))
        ).expanduser()
        cls.rec_weights = Path(
            os.getenv(REC_PTH_ENV, str(default_server_rec_weights_path()))
        ).expanduser()
        if not cls.det_weights.exists() or not cls.rec_weights.exists():
            raise unittest.SkipTest(
                "Converted PP-OCRv5 server weights are missing. Run "
                "scripts/prepare_ppocrv5_server.py first."
            )

        cls.args = _build_server_args(cls.det_weights, cls.rec_weights)
        cls.det_wrapper = PPOCRv5ServerDetModel()
        cls.rec_wrapper = PPOCRv5ServerRecModel()
        cls.reference_detector = predict_det.TextDetector(cls.args)
        cls.reference_recognizer = predict_rec.TextRecognizer(cls.args)
        cls.reference_system = predict_system.TextSystem(cls.args)
        cls.autokernel_system = predict_system.TextSystem(cls.args)
        cls.autokernel_system.text_detector.net = cls.det_wrapper.net
        cls.autokernel_system.text_recognizer.net = cls.rec_wrapper.net

        cls.det_images = [
            PPOCR_ROOT / "doc/imgs_en/img_10.jpg",
            PPOCR_ROOT / "doc/imgs/1.jpg",
        ]
        cls.rec_images = [
            PPOCR_ROOT / "doc/imgs_words/ch/word_1.jpg",
            PPOCR_ROOT / "doc/imgs_words/en/word_1.png",
        ]

    def _load_image(self, path: Path):
        image = cv2.imread(str(path))
        self.assertIsNotNone(image, f"Failed to load fixture: {path}")
        return image

    def test_detector_tensor_parity(self) -> None:
        for image_path in self.det_images:
            with self.subTest(image=image_path.name):
                image = self._load_image(image_path)
                data = {"image": image}
                processed = transform(data, self.reference_detector.preprocess_op)
                norm_img, _shape_list = processed
                self.assertIsNotNone(norm_img)
                batch = np.expand_dims(norm_img, axis=0).copy()
                inp = torch.from_numpy(batch)

                with torch.no_grad():
                    reference_maps = self.reference_detector.net(inp)["maps"]
                    autokernel_maps = self.det_wrapper(inp)

                torch.testing.assert_close(reference_maps, autokernel_maps)

    def test_recognizer_tensor_parity(self) -> None:
        for image_path in self.rec_images:
            with self.subTest(image=image_path.name):
                image = self._load_image(image_path)
                wh_ratio = image.shape[1] / float(image.shape[0])
                norm_img = self.reference_recognizer.resize_norm_img(image, wh_ratio)
                batch = np.expand_dims(norm_img, axis=0).copy()
                inp = torch.from_numpy(batch)

                with torch.no_grad():
                    reference_probs = self.reference_recognizer.net(inp)
                    autokernel_probs = self.rec_wrapper(inp)

                torch.testing.assert_close(reference_probs, autokernel_probs)

    def test_end_to_end_image_parity(self) -> None:
        for image_path in self.det_images:
            with self.subTest(image=image_path.name):
                image = self._load_image(image_path)
                ref_boxes, ref_rec, _ref_times = self.reference_system(image.copy())
                ak_boxes, ak_rec, _ak_times = self.autokernel_system(image.copy())

                self.assertEqual(len(ref_boxes), len(ak_boxes))
                self.assertEqual(len(ref_rec), len(ak_rec))

                for ref_box, ak_box in zip(ref_boxes, ak_boxes):
                    ref_array = np.asarray(ref_box, dtype=np.float32)
                    ak_array = np.asarray(ak_box, dtype=np.float32)
                    self.assertEqual(ref_array.shape, ak_array.shape)
                    self.assertLessEqual(np.max(np.abs(ref_array - ak_array)), 2.0)

                for (ref_text, ref_score), (ak_text, ak_score) in zip(ref_rec, ak_rec):
                    self.assertEqual(ref_text, ak_text)
                    self.assertAlmostEqual(float(ref_score), float(ak_score), delta=1e-3)
