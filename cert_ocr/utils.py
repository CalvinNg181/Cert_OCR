import json
import re
from pathlib import Path
from PIL import Image
import fitz  # pymupdf

REQUIRED_KEYS = {"runner_name", "race_category", "finish_time", "time_type"}


def pdf_to_image(pdf_path: str, dpi: int = 200) -> Image.Image:
    """Render the first page of a PDF to a PIL Image at the given DPI."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 is PDF default DPI
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def load_image(path: str) -> tuple[Image.Image, str]:
    """
    Load an image from a file path, auto-converting PDF if needed.
    Returns (PIL Image, source_type) where source_type is 'pdf' or 'image'.
    """
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return pdf_to_image(path), "pdf"
    return Image.open(path).convert("RGB"), "image"


def parse_json_output(raw: str) -> dict:
    """
    Best-effort JSON extraction from model output.
    Handles cases where the model wraps JSON in markdown code fences.
    """
    # Strip markdown code fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1)

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fall back: find the first {...} block in the string
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    return {"raw_output": raw, "parse_error": True}


def validate_result(result: dict) -> tuple[bool, list[str]]:
    """Returns (is_valid, list_of_missing_keys)."""
    missing = [k for k in REQUIRED_KEYS if k not in result]
    return len(missing) == 0, missing
