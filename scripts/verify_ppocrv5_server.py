#!/usr/bin/env python3
"""
Run PP-OCRv5 server parity checks.

Examples:
    uv run scripts/verify_ppocrv5_server.py
    uv run scripts/verify_ppocrv5_server.py --suite det
    uv run scripts/verify_ppocrv5_server.py --suite rec
    uv run scripts/verify_ppocrv5_server.py --suite e2e
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
AUTOKERNEL_ROOT = SCRIPT_DIR.parent
if str(AUTOKERNEL_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOKERNEL_ROOT))


SUITES = {
    "all": "tests.test_ppocrv5_server_parity",
    "det": "tests.test_ppocrv5_server_parity.TestPPOCRv5ServerParity.test_detector_tensor_parity",
    "rec": "tests.test_ppocrv5_server_parity.TestPPOCRv5ServerParity.test_recognizer_tensor_parity",
    "e2e": "tests.test_ppocrv5_server_parity.TestPPOCRv5ServerParity.test_end_to_end_image_parity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PP-OCRv5 server parity tests.")
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="all",
        help="Subset of PP-OCRv5 parity checks to run.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=2,
        help="unittest verbosity level (default: 2).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite_name = SUITES[args.suite]
    suite = unittest.defaultTestLoader.loadTestsFromName(suite_name)
    runner = unittest.TextTestRunner(verbosity=args.verbosity)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
