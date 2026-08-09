# CUDA image. Requires the NVIDIA Container Toolkit on the host.
#
#   docker build -f docker/Dockerfile -t vggt-reconstruction-service .
#   docker run --gpus all -p 8000:8000 vggt-reconstruction-service
#
# There is no CPU variant. VGGT-1B on a CPU takes minutes per job, which is
# not a service anyone would deploy - see the benchmark section of README.md.

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# The CUDA build must come from the PyTorch index. PyPI serves a CPU-only
# wheel, which would silently ignore the GPU this image exists for.
RUN pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# --no-deps: VGGT's metadata pins numpy<2, which is stale. It runs correctly
# on numpy 2.x, and honouring the pin would downgrade the whole image.
RUN pip install --no-deps git+https://github.com/facebookresearch/vggt.git

RUN useradd --create-home --uid 10001 vision
COPY --chown=vision:vision src/ ./src/

ENV PYTHONPATH=/app/src \
    VGGT_ARTIFACT_DIR=/data/artifacts \
    HF_HOME=/home/vision/.cache/huggingface

RUN mkdir -p /data/artifacts && chown -R vision:vision /data
USER vision

# Bake the weights in. Without this a cold start downloads about 5 GB, which
# fails in an air-gapped cluster and makes startup depend on a third party.
# From a warm cache the model loads in roughly 10 seconds; the 5 GB download
# is what makes a cold first run take minutes.
RUN python -c "from vggt.models.vggt import VGGT; VGGT.from_pretrained('facebook/VGGT-1B')"

EXPOSE 8000

# Readiness, not liveness. The container answers /healthz immediately but
# cannot serve a reconstruction until the model is resident.
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/readyz || exit 1

CMD ["uvicorn", "reconstruction.service:app", "--host", "0.0.0.0", "--port", "8000"]
