"""
MVP entry point: extract structured data from a marathon certificate image.

Usage:
    conda activate cert_ocr
    python run_mvp.py <image_path>

Example:
    python run_mvp.py tests/sample_images/low_res.jpg
"""
import sys
import json
import time

from cert_ocr.model import load_model
from cert_ocr.pipeline import extract_certificate_data
from cert_ocr.utils import validate_result


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "tests/sample_images/low_res.jpg"

    print(f"Image: {image_path}")
    print("Loading model…")
    t0 = time.time()
    model, processor = load_model()
    print(f"Model ready in {time.time() - t0:.1f}s\n")

    print("Running inference…")
    t1 = time.time()
    result = extract_certificate_data(image_path, model, processor)
    elapsed = time.time() - t1
    print(f"Inference done in {elapsed:.1f}s\n")

    print("--- Result ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    is_valid, missing = validate_result(result)
    if is_valid:
        print("\n✓ All required fields present.")
    else:
        print(f"\n⚠ Missing fields: {missing}")


if __name__ == "__main__":
    main()
