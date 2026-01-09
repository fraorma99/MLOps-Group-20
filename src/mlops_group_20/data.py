from pathlib import Path
import pandas as pd
import typer
from torch.utils.data import Dataset
from typing import Any

class LanguageDataset(Dataset):
    """Language detection dataset."""
    
    def __init__(self, data_path: Path) -> None:
        self.data = pd.read_csv(data_path)
        self.texts = self.data['Text'].tolist()
        self.labels = self.data['Language'].tolist()
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            'text': self.texts[index],
            'label': self.labels[index]
        }
    
    def preprocess(self, output_folder: Path) -> None:
        """Save processed data."""
        output_folder.mkdir(exist_ok=True)
        self.data.to_pickle(output_folder / 'processed.pkl')
        print(f"Processed data saved to: {output_folder / 'processed.pkl'}")

def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing language data...")
    dataset = LanguageDataset(data_path)
    dataset.preprocess(output_folder)

if __name__ == "__main__":
    typer.run(preprocess)
