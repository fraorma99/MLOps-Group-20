import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from mlops_group_20.model import LanguageClassifier
from mlops_group_20.data import Vocabulary, TextDataset, simple_tokenizer

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
    Path("data/splits").mkdir(parents=True, exist_ok=True)
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

    # Setup tokenizer and Build vocabulary
    tokenizer = simple_tokenizer
    print("Building vocabulary...")
    vocab = Vocabulary(min_freq=2)
    vocab.build([tokenizer(text) for text in train_data['Text'].tolist()])
    print(f"Vocabulary size: {len(vocab)}")
    
    # Save vocabulary for later use
    pd.to_pickle(vocab, "data/splits/vocab.pkl")
    
    # Create datasets
    train_dataset = TextDataset(train_data['Text'].tolist(), [label2idx[l] for l in train_data['Language']], vocab, tokenizer)
    val_dataset = TextDataset(val_data['Text'].tolist(), [label2idx[l] for l in val_data['Language']], vocab, tokenizer)
    
    # Create dataloaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LanguageClassifier(
        vocab_size=len(vocab),
        num_classes=len(languages)
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
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
        val_loss, val_correct, val_total = 0, 0, 0
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

        train_losses.append(avg_train_loss); val_losses.append(avg_val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            Path("models").mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': len(vocab),
                'num_classes': len(languages)
            }, "models/best_model.pt")

    # Save training history
    torch.save({
        'epochs': list(range(1, num_epochs + 1)),
        'train_losses': train_losses, 'val_losses': val_losses,
        'train_accs': train_accs, 'val_accs': val_accs,
        'final_train_acc': train_accs[-1], 'final_val_acc': val_accs[-1]
    }, "models/training_history.pt")

if __name__ == "__main__":
    train()