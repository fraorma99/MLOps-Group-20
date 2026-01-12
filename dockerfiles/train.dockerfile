# Use a lightweight Python base image
FROM python:3.12-slim

# Install uv directly from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set environment variable to prefer system Python
ENV UV_PYTHON_PREFERENCE=only-system

# Set the working directory inside the container 
WORKDIR /app

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies without installing the project itself yet
# This ensures that changing your code doesn't trigger a full reinstall of libraries
RUN uv sync --frozen --no-dev

# Copy the source code and necessary project files
COPY src/ src/
COPY data/ data/
COPY README.md ./
COPY LICENSE ./

# Install the project in editable mode so the 'mlops_group_20' module is recognized
RUN uv pip install -e .

# Set the default command to run the training script
ENTRYPOINT ["uv", "run", "python", "-m", "mlops_group_20.train"]
