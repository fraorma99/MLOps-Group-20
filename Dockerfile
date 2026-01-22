FROM python:3.12-slim AS builder

# Install system deps (git for clone)
RUN apt-get update && apt-get install -y git \
  && rm -rf /var/lib/apt/lists/*

# Copy uv binary (fast installs)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Clone Kenzov3 branch
RUN git clone -b Kenzov3 https://github.com/fraorma99/MLOps-Group-20.git .

# Install deps + editable package
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# Run setup script
RUN ./scripts/setup.sh

# Expect CSV to already be present in repo
RUN test -f data/raw/language_detection.csv || (echo "Missing data/raw/language_detection.csv. Add the CSV before building." && exit 1)

# Process data
RUN uv run python src/mlops_group_20/data.py \
  data/raw/language_detection.csv data/processed/


# Runtime stage
FROM python:3.12-slim AS runtime
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Copy uv binary from builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=src

VOLUME ["/app/wandb", "/app/models"]

#Use python directly (venv has everything)
ENTRYPOINT ["/app/.venv/bin/python", "-m", "mlops_group_20.train"]
CMD ["--config-name", "sweep", "--multirun", \
     "optimizer.lr=0.001,0.0015,0.002", \
     "wandb.run_name=lr_${optimizer.lr}"]
