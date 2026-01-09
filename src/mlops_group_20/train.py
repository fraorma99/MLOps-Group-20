import pickle
from pathlib import Path

import pandas as pd

from mlops_group_20.model import Model
from mlops_group_20.data import LanguageDataset


def train():
    # Load the processed dataset
    data_path = Path("data/processed/processed.pkl")
    data = pd.read_pickle(data_path)
    
    print(f"Dataset loaded: {len(data)} samples")
    print(f"Columns: {data.columns.tolist()}")
    print(f"Languages: {data['Language'].unique()}")
    
    model = Model()
    # add rest of your training code here


if __name__ == "__main__":
    train()
