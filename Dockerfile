FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

#Clone respository
RUN git clone -b Kenzov3 https://github.com/fraorma99/MLOps-Group-20.git .

#Single sync + editable (uv sync handles -e automatically)
RUN --mount=type=cache,target=/root/.cache/uv uv sync

#Run setup, download, process data (matches your steps 1-4)
RUN ./scripts/setup.sh
RUN ./scripts/download_data.sh
RUN uv run python src/mlops_group_20/data.py data/raw/language_detection.csv data/processed/

FROM python:3.12-slim AS runtime
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=src

VOLUME ["/app/wandb", "/app/models"]

#Default: your exact train command with sweep
ENTRYPOINT ["uv", "run", "python", "-m", "mlops_group_20.train"]
CMD ["--config-name", "sweep", "--multirun", "optimizer.lr=0.001,0.0015,0.002", \
     "wandb.run_name=lr_${optimizer.lr}"]
