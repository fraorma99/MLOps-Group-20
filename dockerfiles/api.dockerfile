# Use a Python 3.12 image based on Debian Slim
FROM python:3.12-slim

# Install 'uv' by copying the binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Prevent 'uv' from downloading other Python versions and use the system one
ENV UV_PYTHON_PREFERENCE=only-system

# Set the working directory inside the container
WORKDIR /app

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Synchronize dependencies (excluding dev tools) without installing the project yet
RUN uv sync --frozen --no-dev --no-install-project

# Copy configuration (Required for Hydra - Milestone M11)
COPY configs/ configs/

# Copy the source code and necessary artifacts
COPY src/ src/
COPY models/ models/
COPY data/ data/

# Metadata files are required for the local package installation
COPY README.md LICENSE ./

# Install the project in editable mode so 'mlops_group_20' is available as a module
RUN uv pip install -e .

# Expose the port that FastAPI will use
EXPOSE 8000

# Start the API with uvicorn
ENTRYPOINT ["uv", "run", "uvicorn", "mlops_group_20.api:app", "--host", "0.0.0.0", "--port", "8000"]
