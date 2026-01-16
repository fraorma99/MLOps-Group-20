FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

#Install uv
RUN pip install 'uv>=0.4.0'

WORKDIR /app

#Copy uv files first (cache deps!)
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-isolated  # Installs ALL deps system-wide

#Copy rest + editable install
COPY . .
RUN uv pip install --system -e .

#DNeeds fixing - find solutionf or data part
#COPY scripts/download_data.sh .
#RUN chmod +x ./scripts/download_data.sh && ./scripts/download_data.sh

CMD ["uv", "run", "python", "-m", "mlops_group_20.train"]
