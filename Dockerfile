FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

# Copy project metadata
COPY pyproject.toml uv.lock* ./

# Copy CLI application
COPY embeddings ./embeddings

# Install package into system python
RUN uv pip install --system .

# Download the model and include in the Docker image
# NOTE: The env vars "TE_MODEL_URI" and "TE_MODEL_DOWNLOAD_PATH" are set here to support
#  the downloading of the model into this image build, but persist in the container and
#  effectively also set this as the default model.
ENV HF_HUB_DISABLE_PROGRESS_BARS=true
ENV TE_MODEL_URI=opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte
ENV TE_MODEL_DOWNLOAD_PATH=/model
RUN python -m embeddings.cli --verbose download-model

ENTRYPOINT ["python", "-m", "embeddings.cli"]
