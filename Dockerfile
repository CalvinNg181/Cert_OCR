# -----------------------------------------------------------------------
# Base: Official PyTorch 2.11 + CUDA 12.8 + cuDNN9 runtime
# Chosen over nvidia/cuda (too bare) and NGC pytorch (too large ~20GB).
# pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime is ~8GB with PyTorch
# pre-installed and full Blackwell (sm_120) support out of the box.
# -----------------------------------------------------------------------
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# -----------------------------------------------------------------------
# Install Python dependencies
# PyTorch already included in base image — skip re-installing it.
# -----------------------------------------------------------------------
RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    bitsandbytes \
    qwen-vl-utils \
    pymupdf \
    pillow \
    fastapi \
    "uvicorn[standard]" \
    python-multipart

# -----------------------------------------------------------------------
# Copy application code
# -----------------------------------------------------------------------
COPY cert_ocr/ ./cert_ocr/
COPY api/      ./api/

# -----------------------------------------------------------------------
# Runtime config
# -----------------------------------------------------------------------
# Model cache lives in a mounted volume (see docker-compose.yml)
ENV HF_HOME=/cache/huggingface
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
