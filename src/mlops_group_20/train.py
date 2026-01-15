import logging
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig, OmegaConf

from mlops_group_20.model import LanguageClassifier
from mlops_group_20.data import Vocabulary, TextDataset, simple_tokenizer

# Initialize logger (Standard Hydra/DTU MLOps practice)
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig):
    # 1. Log the loaded configuration for reproducibility
    log.info(f"Configuration used:\n{OmegaConf.to_yaml(cfg)}")

    # 2. Set seeds for statistical reproducibility
    torch.manual_seed(cfg.training.seed)
    log.info(f"Reproducibility seed set to: {cfg.training.seed}")

    # 3. Load the processed dataset using paths from Hydra config
    data_path = Path(f"{cfg.paths.processed_dir}/processed.pkl")
    if not data_path.exists():
        log.error(f"Dataset not found at {data_path}. Please run preprocessing first.")
        return
        
    data = pd.read_pickle(data_path)
    log.info(f"Dataset loaded: {len(data)} samples")
    
    # Create label mapping
    languages = sorted(data['Language'].unique())
    label2idx = {lang: idx for idx, lang in enumerate(languages)}
    idx2label = {idx: lang for lang, idx in label2idx.items()}
    log.info(f"Languages ({len(languages)}): {languages}")
    
    # Save label mappings to the splits directory defined in config
    splits_path = Path(cfg.paths.splits_dir)
    splits_path.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({'label2idx': label2idx, 'idx2label': idx2label}, splits_path / "label_mappings.pkl")

    # Train/Val/Test splits
    train_idx, temp_idx = train_test_split(
        range(len(data)), 
        test_size=0.3, 
        stratify=data['Language'],
        random_state=cfg.training.seed
    )
    
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=data.iloc[temp_idx]['Language'],
        random_state=cfg.training.seed
    )
    
    # Save indices
    split_info = {'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}
    pd.to_pickle(split_info, splits_path / "split_info.pkl")
    log.info(f"Saved splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    train_data = data.iloc[train_idx].reset_index(drop=True)
    val_data = data.iloc[val_idx].reset_index(drop=True)

    # Setup tokenizer and Build vocabulary
    tokenizer = simple_tokenizer
    log.info("Building vocabulary...")
    vocab = Vocabulary(min_freq=2)
    vocab.build([tokenizer(text) for text in train_data['Text'].tolist()])
    log.info(f"Vocabulary size: {len(vocab)}")
    
    # Save vocabulary
    pd.to_pickle(vocab, splits_path / "vocab.pkl")
    
    # Create datasets
    train_dataset = TextDataset(train_data['Text'].tolist(), [label2idx[l] for l in train_data['Language']], vocab, tokenizer)
    val_dataset = TextDataset(val_data['Text'].tolist(), [label2idx[l] for l in val_data['Language']], vocab, tokenizer)
    
    # Create dataloaders using batch_size from config
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    log.info(f"Using device: {device}")
    
    # Initialize model using parameters from model/default.yaml
    model = LanguageClassifier(
        vocab_size=len(vocab),
        num_classes=len(languages),
        hidden_dim=cfg.model.hidden_dim
    ).to(device)
    
    # Loss and optimizer using learning rate from config
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    # Training loop using epochs from config
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(cfg.training.epochs):
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

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        log.info(f"Epoch {epoch+1}/{cfg.training.epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model to the path specified in config
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            Path("models").mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': len(vocab),
                'num_classes': len(languages)
            }, cfg.paths.model_save_path)

    # Save training history
    torch.save({
        'epochs': list(range(1, cfg.training.epochs + 1)),
        'train_losses': train_losses, 'val_losses': val_losses,
        'train_accs': train_accs, 'val_accs': val_accs,
        'final_train_acc': train_accs[-1], 'final_val_acc': val_accs[-1]
    }, "models/training_history.pt")
    log.info("Training history saved.")

if __name__ == "__main__":
    train()