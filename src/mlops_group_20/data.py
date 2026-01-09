from pathlib import Path
import pandas as pd
import typer
from torch.utils.data import Dataset
from typing import Any
from collections import Counter
import torch

class Vocabulary:
    """Simple vocabulary class to replace torchtext vocab."""
    def __init__(self, min_freq=2):
        self.token2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2token = {0: '<pad>', 1: '<unk>'}
        self.min_freq = min_freq
        self.default_index = 1  # <unk>
    
    def build(self, token_lists):
        """Build vocabulary from list of token lists."""
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
    
    def __len__(self):
        return len(self.token2idx)
    
    def __getitem__(self, token):
        return self.token2idx.get(token, self.default_index)

def simple_tokenizer(text):
    """Simple character-level tokenizer - works well for language detection."""
    return list(text.lower())

class TextDataset(Dataset):
    """PyTorch adapter for language detection, tokenizes, converts the index and pads."""
    def __init__(self, texts, labels, vocab, tokenizer, max_len=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize and convert to indices
        tokens = self.tokenizer(text)[:self.max_len]
        indices = [self.vocab[token] for token in tokens]
        
        # Pad or truncate to max_len
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))  # 0 is padding
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

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
        output_folder.mkdir(parents=True, exist_ok=True)
        self.data.to_pickle(output_folder / 'processed.pkl')
        print(f"Processed data saved to: {output_folder / 'processed.pkl'}")

def preprocess_fn(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing language data...")
    dataset = LanguageDataset(data_path)
    dataset.preprocess(output_folder)

if __name__ == "__main__":
    typer.run(preprocess_fn)