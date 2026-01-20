# Project Description

**1. Overall Goal of the Project**

The primary objective of this project is to develop and deploy a robust, scalable, and fully automated MLOps pipeline for Language Detection. Our goal is to implement a complete lifecycle management system, from data versioning and model training to containerized deployment and follow what have been taught in the MLOps course. We aim to provide an API where users can submit text fragments and receive the predicted language with high confidence.

**2. Frameworks**

We will integrate these key frameworks:
*PyTorch / PyTorch Lightning:* We will use PyTorch for the core modeling. PyTorch Lightning will be integrated to standardize the training loop, making it easier to handle device placement (CPU/GPU).

*Hydra:* For configuration management. We will use Hydra to manage hyperparameters, data paths, and model settings, allowing to run different experiments without modifying the source code.

*DVC (Data Version Control):* To manage our datasets. Since GitHub is not designed for large files, DVC will allow us to version our data and models, ensuring reproducibility across all team members.

*Docker:* We will containerize both the training environment and the inference API to ensure it works on all machines.

*FastAPI:* This will be our web framework to serve the model as a REST API, integrated within the api.py module.

**3. Data**

We will use the "Language Detection" dataset available on Kaggle. 

https://www.kaggle.com/datasets/basilb2s/language-detection

This dataset contains over 10,000 rows of text samples across 17 different languages (including English, Danish, Italian, Arabic, etc.).

Preprocessing: Our pipeline in data.py will handle text cleaning (removing special characters and numbers), tokenization, and converting text into numerical representations (TF-IDF or Word Embeddings).

Evolution: As the project progresses, we may investigate "data drifting" by introducing noisier text (e.g., tweets or slang) to see how the model performance degrades.

**4. Models**

*Baseline Model:* We will start with a straightforward classification model that identifies languages by analyzing the frequency and patterns of common words and characters. This approach allows us to establish and test our entire MLOps infrastructure (the "pipes" of the project) without getting bogged down in complex AI architecture on day one.

*Main Model:* When the infrastructure is stable, we could implement a Neural Network specifically designed for text. Instead of just counting words, this model will learn to recognize the "shape" and sequence of sentences.


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
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

**1. Run the API container** (no setup needed):
```bash
docker login
docker pull --platform linux/amd64 kenzodtu/mlops-api:latest
docker run -d --name mlops-api -p 8000:8000 kenzodtu/mlops-api:latest
```

**2. Wait for startup** (takes ~30 seconds on first run):
The container will download dependencies on first launch. Check it's ready:
```bash
docker logs mlops-api -f  # Press Ctrl+C when you see "Application startup complete"
```

**3. Open the web UI**:
Visit [http://localhost:8000/ui](http://localhost:8000/ui)

Type any text and get instant language detection with confidence scores! 🎉

---

### Full Setup with Training (docker-compose)

If you want to run training + API together on your machine:

**Prerequisites**:
- Docker Desktop installed ([download here](https://www.docker.com/products/docker-desktop))
- `.env` file created (see above)
- At least 8GB free disk space

**1. Create `.env` file** (same as above)

**2. Start the full stack**:
```bash
docker compose up -d
```

This will:
- Start the trainer container to process data and train the model
- Save trained models to `./models/`
- Start the API container that uses the trained models
- Logs are visible with `docker compose logs -f`

**3. Access the API**:
- Web UI: [http://localhost:8000/ui](http://localhost:8000/ui)
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health check: `curl http://localhost:8000/health`

**4. Make predictions** (command line):
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour, comment allez-vous?"}'
```

Response:
```json
{
  "input_text": "Bonjour, comment allez-vous?",
  "predicted_language": "French",
  "status": "success"
}
```

---

### Stopping Services

```bash
# Stop all services
docker compose down

# Stop and remove volumes (clean slate)
docker compose down -v
```

---

### Troubleshooting

**API won't start**: Check if port 8000 is already in use
```bash
lsof -i :8000  # List what's using port 8000
kill -9 <PID>  # Kill the process
```

**Training still running**: Check logs
```bash
docker compose logs -f trainer
```

**Images not found**: Pull latest versions
```bash
docker pull kenzodtu/mlops-api:latest
docker pull kenzodtu/mlops-group-20-trainer:latest
```

---

### Manual Docker Commands (if not using docker-compose)

Build trainer image locally:
```bash
docker build --platform linux/amd64 -t mlops-kenzov3 .
```

Run trainer:
```bash
docker run -it --name mlops-trainer \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/wandb:/app/wandb \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e WANDB_ENTITY=$WANDB_ENTITY \
  -e WANDB_PROJECT=$WANDB_PROJECT \
  mlops-kenzov3
```

Build API image:
```bash
docker build --platform linux/amd64 -f dockerfiles/api.dockerfile -t mlops-api:latest .
```

Run API:
```bash
docker run -d --name mlops-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data/splits:/app/data/splits:ro \
  mlops-api:latest
```
