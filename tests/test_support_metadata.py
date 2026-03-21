from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(module_name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


profile = _load_module("autokernel_profile", "profile.py")
support = _load_module("autokernel_support", "support.py")
verify = _load_module("autokernel_verify", "verify.py")
extract = _load_module("autokernel_extract", "extract.py")


class TestKernelClassification(unittest.TestCase):
    def test_ppocr_vision_op_families_are_classified(self) -> None:
        cases = {
            "aten::cudnn_convolution": "conv2d",
            "aten::depthwise_conv2d": "depthwise_conv2d",
            "aten::conv_transpose2d": "conv_transpose2d",
            "aten::upsample_nearest2d": "interpolate",
            "aten::cat": "concat",
            "aten::avg_pool2d": "pooling",
            "aten::relu_": "elementwise",
            "aten::addmm": "matmul",
        }
        for kernel_name, expected in cases.items():
            with self.subTest(kernel_name=kernel_name):
                self.assertEqual(profile.classify_kernel(kernel_name), expected)


class TestExtractionShapeParsing(unittest.TestCase):
    def test_parse_shape_info_handles_profiler_shape_lists(self) -> None:
        self.assertEqual(
            extract.parse_shape_info("[[40, 120], [120, 18385]]", "matmul"),
            {"M": 40, "N": 18385, "K": 120},
        )
        self.assertEqual(
            extract.parse_shape_info("[[120], [40, 240], [240, 120], [], []]", "matmul"),
            {"M": 40, "N": 120, "K": 240},
        )
        self.assertEqual(
            extract.parse_shape_info("[[1, 40, 120], [], [120], [120], []]", "layernorm"),
            {"batch": 40, "dim": 120},
        )
        self.assertEqual(
            extract.parse_shape_info("[[1, 40, 18385], [], []]", "softmax"),
            {"rows": 40, "cols": 18385},
        )

    def test_get_supported_kernels_can_prefer_operator_level_shape_resolved_entries(self) -> None:
        report = {
            "top_kernels": [
                {
                    "rank": 5,
                    "name": "turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn",
                    "op_type": "matmul",
                    "autokernel_supported": True,
                    "shape_info": "",
                },
                {
                    "rank": 34,
                    "name": "aten::mm",
                    "op_type": "matmul",
                    "autokernel_supported": True,
                    "shape_info": "[[40, 120], [120, 18385]]",
                },
                {
                    "rank": 53,
                    "name": "aten::native_layer_norm",
                    "op_type": "layernorm",
                    "autokernel_supported": True,
                    "shape_info": "[[1, 40, 120], [], [120], [120], []]",
                },
            ]
        }

        selected = extract.get_supported_kernels(
            report,
            prefer_operator_level=True,
            require_shape=True,
        )
        self.assertEqual([kernel["name"] for kernel in selected], ["aten::mm", "aten::native_layer_norm"])


class TestSupportStages(unittest.TestCase):
    def test_support_stages_distinguish_extract_and_reinsert(self) -> None:
        matmul = support.build_support_stage("matmul", {"matmul"})
        self.assertEqual(
            matmul,
            {
                "profile_supported": True,
                "extract_supported": True,
                "reinsert_supported": True,
            },
        )

        flash = support.build_support_stage("flash_attention", {"flash_attention"})
        self.assertEqual(
            flash,
            {
                "profile_supported": True,
                "extract_supported": True,
                "reinsert_supported": False,
            },
        )

        conv = support.build_support_stage("conv2d", {"matmul"})
        self.assertEqual(
            conv,
            {
                "profile_supported": True,
                "extract_supported": False,
                "reinsert_supported": False,
            },
        )

    def test_verification_report_includes_skipped_kernels(self) -> None:
        result = verify.VerificationResult(
            model_name="PPOCRv5ServerRecModel",
            input_shape="1,3,48,320",
            dtype_str="float32",
            gpu_name="Test GPU",
            ref_output_shape="[1, 80, 6625]",
            ref_latency_ms=10.0,
            opt_output_shape="[1, 80, 6625]",
            opt_latency_ms=8.0,
            kernels_replaced=[
                {
                    "type": "matmul",
                    "rank": 1,
                    "speedup": 1.5,
                    "path": "workspace/kernel_matmul_1_optimized.py",
                    "modules_replaced": 12,
                }
            ],
            kernels_skipped=[
                {
                    "type": "conv2d",
                    "rank": 2,
                    "speedup": 0.0,
                    "path": "workspace/kernel_conv2d_2_optimized.py",
                    "reason": "No reinsertion strategy for this operator family.",
                }
            ],
            correctness="PASS",
            end_to_end_speedup=1.25,
        )

        report = verify.format_report(result)
        self.assertIn("Kernels skipped:", report)
        self.assertIn("conv2d (rank 2)", report)
        self.assertIn("modules", report)
