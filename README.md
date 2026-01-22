# Project Description

**1. Overall Goal of the Project**

The primary objective of this project is to develop and deploy a robust, scalable, and fully automated MLOps pipeline for Language Detection. Our goal is to implement a complete lifecycle management system, from data versioning and model training to containerized deployment and follow what have been taught in the MLOps course. We aim to provide an API where users can submit text fragments and receive the predicted language with high confidence.



## Project structure

The directory structure of the project looks like this:
```txt
MLOps-Group-20/
├── .devcontainer/           # Dev environment setup
│   ├── devcontainer.json
│   └── post_create.sh
├── .dvc/                    # Data versioning config
│   ├── .gitignore
│   └── config
├── .github/                 # Automation and CI/CD
│   ├── dependabot.yaml
│   └── workflows/
│       ├── data_changes.yaml
│       ├── linting.yaml
│       ├── model_changes.yaml
│       └── tests.yaml
├── configs/                 # Hydra and Sweeps
│   ├── config.yaml
│   └── sweep.yaml
├── data/                    # Data management
│   ├── processed/           # Processed data files
│   │   ├── processed.pkl
│   │   └── splits/          # Data split indices
│   │       ├── test_indices.pkl
│   │       ├── train_indices.pkl
│   │       └── val_indices.pkl
│   ├── raw/                 # Original dataset files
│   │   └── language_detection.csv
│   ├── splits/              # Metadata and vocab
│   │   ├── label_mappings.pkl
│   │   ├── split_info.pkl
│   │   └── vocab.pkl
│   └── raw.dvc              # DVC data pointer
├── dockerfiles/             # Container recipes
│   ├── api.dockerfile
│   ├── drift_api.dockerfile
│   └── train.dockerfile
├── docs/                    # Technical documentation
│   ├── mkdocs.yml
│   ├── README.md
│   └── source/
│       └── index.md
├── images/                  # Project visual assets
│   └── figures/             # Training plots (M14)
├── models/                  # Model artifacts
│   ├── best_model.pt
│   └── training_history.pt
├── monitoring/
│   ├── data_drifting.py/ 
├── multirun/                # Hydra multirun logs
├── notebooks/               # Exploratory Jupyter notebooks
├── onnx/                    # Create and test onnx model API
│   ├── export_onnx.py
│   └── test_onnx.py
├── outputs/                 # Hydra run logs
├── reports/                 # Compliance and Cloud reports
│   ├── reports.py
│   ├── data_drifting/
│   ├── README.md
│   └── figures/             # GCP and WandB screenshots
│       ├── bucket.png, build.png, overview.png, registry.png
│       └── wandb.png, wandb_train.png, wandb_val.png
├── scripts/                 # Automation utility scripts
│   ├── download_data.sh
│   └── setup.sh
├── src/mlops_group_20/      # Main Python package
│   ├── __init__.py
│   ├── api.py
│   ├── api_onnx.py
│   ├── data.py
│   ├── drift_api.py
│   ├── evaluate.py
│   ├── model.py
│   ├── train.py
│   └── visualize.py
├── tests/                   # Testing suite (Pytest)
│   ├── integrationtests/    # Tests original API
│       ├── tests_apis.py
│   ├── performancetests/    # Tests original API
│       ├── locusfile.py
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── wandb/                   # Local WandB cache
│
├── .dockerignore            # Docker exclusion rules
├── .dvcignore               # DVC exclusion rules
├── .env.example             # Template for secrets
├── .env 2.example           # Alternative secret template
├── .gitignore               # Git exclusion rules
├── .pre-commit-config.yaml  # Pre-push code quality
├── .python-version          # Project Python version
├── Dockerfile               # Root Docker recipe
├── LICENSE                  # Project legal license
├── README.md                # Project main documentation
├── cloudbuild.yaml          # GCP build pipeline
├── docker-compose.yml       # Service orchestration
├── profile.prof             # Training profile data
├── prometheus/              # Prometheus database and storage files
├── prometheus.single.yml    # Simplified Prometheus config for single-service testing
├── prometheus.yml           # Main Prometheus configuration (scrape jobs and targets)
├── pyproject.toml           # Project dependencies (UV)
├── report_dynamo_export.sarif # SARIF report for PyTorch Dynamo ONNX export analysis
├── tasks.py                 # Project task runner
└── uv.lock                  # Deterministic dependency lock
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
Data from: https://www.kaggle.com/datasets/basilb2s/language-detection

## Getting Started

Clone the repository:
```bash
git clone https://github.com/fraorma99/MLOps-Group-20.git
cd MLOps-Group-20
```

Then head to the **Docker Deployment** section below to get started!

## 🐳 Docker Deployment

### Quick Start (API Only)

If you just want to test the language detection API without running training:

**1. Run the API container** (no setup needed, clone repository):
```bash
docker login
docker compose pull
docker compose up
```

**2. Open the web UI**:
Visit [http://localhost:8000/ui](http://localhost:8000/ui) for FastAPI
Visit [http://localhost:9090](http://localhost:9090) for Prometheus metrics

Type any text and get instant language detection

---

### Training locally with processed data with wandb implementation
```
uv sync
uv pip install -e
uv run wandb login
uv run python -m mlops_group_20.train

```



