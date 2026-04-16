"""
Validation tests for the certificate OCR pipeline.

Run from project root:
    conda activate cert_ocr
    python tests/test_pipeline.py

Place test images in tests/sample_images/ before running:
    low_res.jpg   — 640×480  (Test A)
    high_res.jpg  — 2000×2000 (Test B)
    artistic.jpg  — script/calligraphy font cert (Test C)
"""
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cert_ocr.model import load_model
from cert_ocr.pipeline import extract_certificate_data
from cert_ocr.utils import validate_result, REQUIRED_KEYS

PASS = "\033[92m✓ PASSED\033[0m"
FAIL = "\033[91m✗ FAILED\033[0m"

TESTS = [
    {
        "id": "A",
        "label": "Low Res (640×480)",
        "image": "tests/sample_images/low_res.jpg",
        "max_time_s": 30,
        "max_pixels": 1280 * 28 * 28,  # safe default
    },
    {
        "id": "B",
        "label": "High Res (2000×2000)",
        "image": "tests/sample_images/high_res.jpg",
        "max_time_s": 90,
        "max_pixels": None,  # full native resolution; comment out max_pixels if OOM
    },
    {
        "id": "C",
        "label": "Artistic / Script Font",
        "image": "tests/sample_images/artistic.jpg",
        "max_time_s": 60,
        "max_pixels": 1280 * 28 * 28,
    },
]


def run_test(test: dict) -> bool:
    image_path = test["image"]
    label = f"Test {test['id']}: {test['label']}"

    if not os.path.exists(image_path):
        print(f"[{label}] SKIPPED — image not found: {image_path}")
        return True  # Don't fail on missing sample images

    print(f"\n[{label}] Loading model…")
    model, processor = load_model(max_pixels=test["max_pixels"])

    print(f"[{label}] Running inference…")
    t0 = time.time()
    result = extract_certificate_data(image_path, model, processor)
    elapsed = time.time() - t0

    print(f"  Inference: {elapsed:.1f}s")
    print(f"  Output:    {json.dumps(result, ensure_ascii=False)}")

    is_valid, missing = validate_result(result)

    if not is_valid:
        print(f"  {FAIL} — missing keys: {missing}")
        return False

    if elapsed > test["max_time_s"]:
        print(f"  {FAIL} — took {elapsed:.1f}s, limit is {test['max_time_s']}s")
        return False

    print(f"  {PASS}")
    return True


def main():
    print("Running all tests (each loads model independently)…")

    results = []
    for test in TESTS:
        results.append(run_test(test))

    ran = [t for t in TESTS if os.path.exists(t["image"])]
    passed = sum(results)
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{len(ran)} tests passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
