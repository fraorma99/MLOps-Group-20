import pickle
from pathlib import Path
from collections import Counter

import pandas as pd


from mlops_group_20.model import LanguageClassifier
from mlops_group_20.data import LanguageDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split


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
    #PyTorch adapter for language detection, tokenizes, 
    #converts the index and pads.
    
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


def train():
    # Load the processed dataset
    data_path = Path("data/processed/processed.pkl")
    data = pd.read_pickle(data_path)
    
    print(f"Dataset loaded: {len(data)} samples")
    
    # Create label mapping
    languages = sorted(data['Language'].unique())
    label2idx = {lang: idx for idx, lang in enumerate(languages)}
    idx2label = {idx: lang for lang, idx in label2idx.items()}
    print(f"Languages ({len(languages)}): {languages}")
    
    # Save label mappings for later use
    Path("data/splits").mkdir(exist_ok=True)
    pd.to_pickle({'label2idx': label2idx, 'idx2label': idx2label}, "data/splits/label_mappings.pkl")

    torch.manual_seed(42)  # For reproducibility
    
    train_idx, temp_idx = train_test_split(
        range(len(data)), 
        test_size=0.3, 
        stratify=data['Language'],
        random_state=42
    )
    
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=data.iloc[temp_idx]['Language'],
        random_state=42
    )
    
    # Save indices for evaluate/visualize
    split_info = {'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}
    pd.to_pickle(split_info, "data/splits/split_info.pkl")
    print(f"Saved splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    # Create data splits
    train_data = data.iloc[train_idx].reset_index(drop=True)
    val_data = data.iloc[val_idx].reset_index(drop=True)
    test_data = data.iloc[test_idx].reset_index(drop=True)

    # Setup tokenizer (character-level works well for language detection)
    tokenizer = simple_tokenizer
    
    # Build vocabulary from training data only
    print("Building vocabulary...")
    vocab = Vocabulary(min_freq=2)
    vocab.build([tokenizer(text) for text in train_data['Text'].tolist()])
    print(f"Vocabulary size: {len(vocab)}")
    
    # Save vocabulary for later use
    pd.to_pickle(vocab, "data/splits/vocab.pkl")
    
    # Convert labels to indices
    train_labels = [label2idx[lang] for lang in train_data['Language']]
    val_labels = [label2idx[lang] for lang in val_data['Language']]
    test_labels = [label2idx[lang] for lang in test_data['Language']]
    
    # Create datasets
    train_dataset = TextDataset(train_data['Text'].tolist(), train_labels, vocab, tokenizer)
    val_dataset = TextDataset(val_data['Text'].tolist(), val_labels, vocab, tokenizer)
    test_dataset = TextDataset(test_data['Text'].tolist(), test_labels, vocab, tokenizer)
    
    # Create dataloaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LanguageClassifier(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=256,
        num_classes=len(languages),
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            Path("models").mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'vocab_size': len(vocab),
                'num_classes': len(languages)
            }, "models/best_model.pt")
            print(f"  -> Saved best model with val_acc: {val_acc:.2f}%")
    
    print("\nTraining complete!")
    print(f"Best validation accuracy: {val_acc:.2f}%")


if __name__ == "__main__":
    train()
