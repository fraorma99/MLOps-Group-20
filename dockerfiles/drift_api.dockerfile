# dockerfiles/drift_api.dockerfile
FROM python:3.12-slim

# Copy 'uv' binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Prevent uv from downloading other Python versions
ENV UV_PYTHON_PREFERENCE=only-system
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy dependency files first (better caching)
COPY pyproject.toml uv.lock ./

# Install runtime deps (no dev deps) into /app/.venv
RUN uv sync --frozen --no-dev --no-install-project

# Copy source + reference data needed by drift API
COPY src/ src/
COPY data/raw/ data/raw/

# Metadata files required by the project build (pyproject references them)
COPY README.md LICENSE ./

# Install the project so `mlops_group_20` is importable
RUN uv pip install -e .

# Use the venv binaries directly (avoid re-resolving deps at runtime)
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Cloud Run listens on $PORT (typically 8080 for this service)
EXPOSE 8080

# Start the drift monitoring API
CMD ["sh", "-c", "uvicorn mlops_group_20.drift_api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
