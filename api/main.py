from contextlib import asynccontextmanager
import io
import tempfile
import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cert_ocr.model import load_model
from cert_ocr.pipeline import extract_certificate_data
from cert_ocr.utils import validate_result


# --- Shared model state -------------------------------------------------

_model = None
_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once on startup; release on shutdown."""
    global _model, _processor
    print("Loading Qwen2.5-VL model…")
    _model, _processor = load_model()
    print("Model ready. Service is up.")
    yield
    # Cleanup on shutdown
    del _model, _processor
    print("Model unloaded.")


# --- App ----------------------------------------------------------------

app = FastAPI(
    title="Certificate OCR Service",
    description="Extract runner data from marathon/running certificate PDFs.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Schemas ------------------------------------------------------------

class CertificateResult(BaseModel):
    runner_name: str | None
    race_category: str | None
    finish_time: str | None
    time_type: str | None
    parse_error: bool = False
    raw_output: str | None = None


# --- Endpoints ----------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/extract", response_model=CertificateResult)
async def extract(file: UploadFile = File(...)):
    """
    Upload a certificate PDF (or image) and receive structured JSON back.

    Accepted formats: .pdf, .jpg, .jpeg, .png
    """
    suffix = os.path.splitext(file.filename)[-1].lower()
    if suffix not in {".pdf", ".jpg", ".jpeg", ".png"}:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Use .pdf, .jpg, .jpeg or .png.",
        )

    if _model is None or _processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Write upload to a temp file so pipeline can read it normally
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = extract_certificate_data(tmp_path, _model, _processor)
    finally:
        os.unlink(tmp_path)

    return CertificateResult(**{
        "runner_name":   result.get("runner_name"),
        "race_category": result.get("race_category"),
        "finish_time":   result.get("finish_time"),
        "time_type":     result.get("time_type"),
        "parse_error":   result.get("parse_error", False),
        "raw_output":    result.get("raw_output"),
    })
