# Use a Python 3.12 image based on Debian Slim
FROM python:3.12-slim

# Install 'uv' from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Ensure 'uv' uses the system Python
ENV UV_PYTHON_PREFERENCE=only-system

# Set the working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (frozen to lockfile)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code, data, and metadata
COPY src/ src/
COPY data/ data/
COPY README.md LICENSE ./ 

# Install the local project module
RUN uv pip install -e .

# Create the models directory if it doesn't exist to avoid mounting issues
RUN mkdir -p models

# Execute the training script as the main process
ENTRYPOINT ["uv", "run", "python", "-m", "mlops_group_20.train"]