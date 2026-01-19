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
```bash
git clone https://github.com/fraorma99/MLOps-Group-20.git
cd MLOps-Group-20
```
to clone Fran_V1:
```bash
git clone -b Fran_V1 https://github.com/fraorma99/MLOps-Group-20.git
cd MLOps-Group-20
```

**1. Install dependencies**
```
uv sync
./scripts/setup.sh
```
**2. Install the package in "editable" mode (-e)**
```
uv pip install -e .
```
**3. Download dataset**
```
./scripts/download_data.sh
```
**4. Process data**
```
uv run python src/mlops_group_20/data.py \
  data/raw/language_detection.csv \
  data/processed/
```

**5. wandb login**
```
uv run wandb login
```
paste your api key from wandb account

**6. Train the model on the kaggle data**
```
PYTHONPATH=src uv run python -m mlops_group_20.train \
  wandb.entity=fra-orma99-danmarks-tekniske-universitet-dtu \               
  wandb.project=mlops_group_20
```
**bonus. Evaluate the model on the testing**
```
uv run python src/mlops_group_20/evaluate.py
```
**bonus. Visualize performance**
```
uv run python src/mlops_group_20/visualize.py
```

## Docker - update to main once merged and test to see if .env is needed since login manually is avaiable
```

docker build --platform linux/amd64 -t mlops-kenzov3 .
docker run -it --name mlops-trainer -v $(pwd)/models:/app/models -v $(pwd)/outputs:/app/outputs -v $(pwd)/wandb:/app/wandb -e WANDB_API_KEY=$WANDB_API_KEY -e WANDB_ENTITY=$WANDB_ENTITY -e WANDB_PROJECT=$WANDB_PROJECT mlops-kenzov3
```
