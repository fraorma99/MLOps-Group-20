import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from torchmetrics import Accuracy, Precision, Recall, F1Score
#import cProfile - done profiling

from mlops_group_20.model import LanguageClassifier
from mlops_group_20.data import Vocabulary, TextDataset, simple_tokenizer

#pr = cProfile.Profile()

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig):
    #pr.enable()
    # Load the processed dataset
    data_path = Path(cfg.data.path)
    data = pd.read_pickle(data_path)
    print(f"Dataset loaded: {len(data)} samples")

    # Optional: initialize Weights & Biases
    if cfg.wandb.enabled:
        wandb.init(
            project=str(cfg.wandb.project),
            entity=cfg.wandb.entity if cfg.wandb.entity else None,
            name=cfg.wandb.run_name if cfg.wandb.run_name else None,
            notes=cfg.wandb.notes if cfg.wandb.notes else None,
            tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    
    # Create label mapping
    languages = sorted(data['Language'].unique())
    label2idx = {lang: idx for idx, lang in enumerate(languages)}
    idx2label = {idx: lang for lang, idx in label2idx.items()}
    print(f"Languages ({len(languages)}): {languages}")
    
    # Save label mappings for later use
    splits_dir = Path(cfg.data.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({'label2idx': label2idx, 'idx2label': idx2label}, splits_dir / "label_mappings.pkl")

    torch.manual_seed(cfg.data.seed)  # For reproducibility
    
    # Compute split sizes from config
    train_size = float(cfg.data.split.train_size)
    val_size = float(cfg.data.split.val_size)
    test_size = float(cfg.data.split.test_size)

    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "train/val/test sizes must sum to 1"

    # First split: train vs temp (val+test)
    train_idx, temp_idx = train_test_split(
        range(len(data)),
        test_size=(1.0 - train_size),
        stratify=data[cfg.data.split.stratify_column],
        random_state=cfg.data.seed,
    )
    
    # Second split: val vs test within temp
    # Determine ratio for test within temp
    temp_total = val_size + test_size
    test_ratio_in_temp = test_size / temp_total if temp_total > 0 else 0.5

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_ratio_in_temp,
        stratify=data.iloc[temp_idx][cfg.data.split.stratify_column],
        random_state=cfg.data.seed,
    )
    
    # Save indices for evaluate/visualize
    split_info = {'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}
    pd.to_pickle(split_info, splits_dir / "split_info.pkl")
    print(f"Saved splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    # Create data splits
    train_data = data.iloc[train_idx].reset_index(drop=True)
    val_data = data.iloc[val_idx].reset_index(drop=True)

    # Setup tokenizer and Build vocabulary
    tokenizer = simple_tokenizer
    print("Building vocabulary...")
    vocab = Vocabulary(min_freq=cfg.data.vocab_min_freq)
    vocab.build([tokenizer(text) for text in train_data['Text'].tolist()])
    print(f"Vocabulary size: {len(vocab)}")
    
    # Save vocabulary for later use
    pd.to_pickle(vocab, splits_dir / "vocab.pkl")
    
    # Create datasets
    train_dataset = TextDataset(
        train_data['Text'].tolist(),
        [label2idx[l] for l in train_data['Language']],
        vocab,
        tokenizer,
        max_len=cfg.data.max_len,
    )
    val_dataset = TextDataset(
        val_data['Text'].tolist(),
        [label2idx[l] for l in val_data['Language']],
        vocab,
        tokenizer,
        max_len=cfg.data.max_len,
    )
    
    # Create dataloaders
    batch_size = int(cfg.training.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    # Device selection
    if cfg.training.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    elif cfg.training.device == 'cuda':
        device = torch.device('cuda')
    elif cfg.training.device == 'mps':
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    model = LanguageClassifier(
        vocab_size=len(vocab),
        embed_dim=int(cfg.model.embed_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        num_layers=int(cfg.model.num_layers),
        dropout=float(cfg.model.dropout),
        num_classes=len(languages),
    ).to(device)

    if cfg.wandb.enabled and cfg.wandb.watch_model:
        wandb.watch(model, log="all", log_freq=int(cfg.wandb.log_freq))
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    if cfg.optimizer.name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.optimizer.lr), weight_decay=float(cfg.optimizer.weight_decay))
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")

    # Scheduler
    if cfg.scheduler.name.lower() == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(cfg.scheduler.mode),
            patience=int(cfg.scheduler.patience),
            factor=float(cfg.scheduler.factor),
        )
    else:
        scheduler = None
    
    # Initialize torchmetrics
    num_classes = len(languages)
    train_acc_metric = Accuracy(task='multiclass', num_classes=num_classes).to(device)
    train_precision_metric = Precision(task='multiclass', num_classes=num_classes, average='weighted').to(device)
    train_recall_metric = Recall(task='multiclass', num_classes=num_classes, average='weighted').to(device)
    train_f1_metric = F1Score(task='multiclass', num_classes=num_classes, average='weighted').to(device)
    
    val_acc_metric = Accuracy(task='multiclass', num_classes=num_classes).to(device)
    val_precision_metric = Precision(task='multiclass', num_classes=num_classes, average='weighted').to(device)
    val_recall_metric = Recall(task='multiclass', num_classes=num_classes, average='weighted').to(device)
    val_f1_metric = F1Score(task='multiclass', num_classes=num_classes, average='weighted').to(device)
    
    # Training loop
    num_epochs = int(cfg.training.num_epochs)
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_acc_metric.reset()
        train_precision_metric.reset()
        train_recall_metric.reset()
        train_f1_metric.reset()
        
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_acc_metric(predicted, labels)
            train_precision_metric(predicted, labels)
            train_recall_metric(predicted, labels)
            train_f1_metric(predicted, labels)
        
        train_acc = train_acc_metric.compute().item() * 100
        train_precision = train_precision_metric.compute().item() * 100
        train_recall = train_recall_metric.compute().item() * 100
        train_f1 = train_f1_metric.compute().item() * 100
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_acc_metric.reset()
        val_precision_metric.reset()
        val_recall_metric.reset()
        val_f1_metric.reset()
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_acc_metric(predicted, labels)
                val_precision_metric(predicted, labels)
                val_recall_metric(predicted, labels)
                val_f1_metric(predicted, labels)
        
        val_acc = val_acc_metric.compute().item() * 100
        val_precision = val_precision_metric.compute().item() * 100
        val_recall = val_recall_metric.compute().item() * 100
        val_f1 = val_f1_metric.compute().item() * 100
        avg_val_loss = val_loss / len(val_loader)
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        train_losses.append(avg_train_loss); val_losses.append(avg_val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        # W&B logging
        if cfg.wandb.enabled:
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "train/acc": train_acc,
                "train/precision": train_precision,
                "train/recall": train_recall,
                "train/f1": train_f1,
                "val/loss": avg_val_loss,
                "val/acc": val_acc,
                "val/precision": val_precision,
                "val/recall": val_recall,
                "val/f1": val_f1,
                "lr": current_lr,
            })
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Ensure directory exists for best model
            best_model_path = Path(cfg.training.save_best_to)
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            saved_obj = {
                'model_state_dict': model.state_dict(),
                'vocab_size': len(vocab),
                'num_classes': len(languages)
            }
            torch.save(saved_obj, best_model_path)
            if cfg.wandb.enabled:
                artifact = wandb.Artifact("best_model", type="model")
                artifact.add_file(str(best_model_path))
                wandb.log_artifact(artifact)

    # Save training history
    # Save training history
    history_path = Path(cfg.training.save_history_to)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epochs': list(range(1, num_epochs + 1)),
        'train_losses': train_losses, 'val_losses': val_losses,
        'train_accs': train_accs, 'val_accs': val_accs,
        'final_train_acc': train_accs[-1], 'final_val_acc': val_accs[-1]
    }, history_path)

    if cfg.wandb.enabled:
        wandb.finish()
    
    #Profiling finished
    #pr.disable()
    #pr.print_stats(sort='cumulative')

if __name__ == "__main__":
    train()