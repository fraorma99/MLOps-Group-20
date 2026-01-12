# Use a lightweight Python base image
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install production dependencies only
RUN uv sync --frozen --no-dev

# Copy necessary source code and artifacts
# We need the source code, the trained model, and the data mappings
COPY src/ src/
COPY models/ models/
COPY data/splits/ data/splits/

# Install the project
RUN uv pip install -e .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI server using uvicorn
ENTRYPOINT ["uv", "run", "uvicorn", "mlops_group_20.api:app", "--host", "0.0.0.0", "--port", "8000"]