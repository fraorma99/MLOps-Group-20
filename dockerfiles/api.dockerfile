# Use a lightweight Python base image
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# FORCE uv to use the system python (3.11) and NOT download others
ENV UV_PYTHON_PREFERENCE=only-system

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (but not the project itself yet)
# This will now correctly use Python 3.11 from the base image
RUN uv sync --frozen --no-dev --no-install-project

# Copy necessary source code and artifacts
COPY src/ src/
COPY models/ models/
COPY data/splits/ data/splits/

# Install the project locally
RUN uv pip install -e .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI server using uvicorn
ENTRYPOINT ["uv", "run", "uvicorn", "mlops_group_20.api:app", "--host", "0.0.0.0", "--port", "8000"]