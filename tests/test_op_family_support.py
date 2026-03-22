from __future__ import annotations

import unittest

import extract
import profile
from support import build_support_stage


class TestOpFamilySupport(unittest.TestCase):
    def test_profile_classifies_batchnorm(self) -> None:
        self.assertEqual(profile.classify_kernel("aten::native_batch_norm"), "batchnorm")

    def test_profile_classifies_layout_transform(self) -> None:
        kernel_name = (
            "void cudnn::engines_precompiled::nchwToNhwcKernel<__half, __half, float>"
        )
        self.assertEqual(profile.classify_kernel(kernel_name), "layout_transform")

    def test_parse_conv2d_shape(self) -> None:
        parsed = extract.parse_shape_info(
            "[[1, 4096, 1, 40], [256, 4096, 3, 3], [], [], [], [], [], [], []]",
            "conv2d",
        )
        self.assertEqual(
            parsed,
            {
                "N": 1,
                "C_in": 4096,
                "C_out": 256,
                "H": 1,
                "W": 40,
                "KH": 3,
                "KW": 3,
                "stride_h": 1,
                "stride_w": 1,
                "pad_h": 1,
                "pad_w": 1,
                "dil_h": 1,
                "dil_w": 1,
                "groups": 1,
            },
        )

    def test_parse_batchnorm_shape(self) -> None:
        parsed = extract.parse_shape_info(
            "[[1, 192, 6, 80], [192], [192], [192], [192], [], [], []]",
            "batchnorm",
        )
        self.assertEqual(parsed, {"N": 1, "C": 192, "H": 6, "W": 80})

    def test_parse_layout_transform_shape(self) -> None:
        parsed = extract.parse_shape_info(
            "[[1, 192, 6, 80], [1, 192, 6, 80]]",
            "layout_transform",
        )
        self.assertEqual(parsed, {"N": 1, "C": 192, "H": 6, "W": 80})

    def test_conv2d_shape_scaling_preserves_hyperparameters(self) -> None:
        scaled = extract.scale_shape_for_op(
            "conv2d",
            {
                "N": 1,
                "C_in": 192,
                "C_out": 192,
                "H": 6,
                "W": 80,
                "KH": 3,
                "KW": 3,
                "stride_h": 1,
                "stride_w": 1,
                "pad_h": 1,
                "pad_w": 1,
                "dil_h": 1,
                "dil_w": 1,
                "groups": 192,
            },
            0.5,
        )
        self.assertEqual(scaled["KH"], 3)
        self.assertEqual(scaled["KW"], 3)
        self.assertEqual(scaled["groups"], scaled["C_in"])

    def test_support_stage_flags_conv2d_for_reinsertion(self) -> None:
        support = build_support_stage("conv2d", {"conv2d"})
        self.assertTrue(support["profile_supported"])
        self.assertTrue(support["extract_supported"])
        self.assertTrue(support["reinsert_supported"])

    def test_support_stage_keeps_layout_transform_non_reinsertable(self) -> None:
        support = build_support_stage("layout_transform", {"layout_transform"})
        self.assertTrue(support["profile_supported"])
        self.assertTrue(support["extract_supported"])
        self.assertFalse(support["reinsert_supported"])


if __name__ == "__main__":
    unittest.main()
