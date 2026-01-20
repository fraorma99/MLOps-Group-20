import pickle
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from mlops_group_20.model import LanguageClassifier
from mlops_group_20.data import Vocabulary, TextDataset, simple_tokenizer

def evaluate():
    # Load the data the same way as train such that we can test on unseen test data
    data_path = Path("data/processed/processed.pkl")
    data = pd.read_pickle(data_path)

    split_info = pd.read_pickle("data/splits/split_info.pkl")
    label_info = pd.read_pickle("data/splits/label_mappings.pkl")
    vocab = pd.read_pickle("data/splits/vocab.pkl")

    label2idx = label_info['label2idx']
    idx2label = label_info['idx2label']

    # Test set (EXACT same split as training)
    test_data = data.iloc[split_info['test_idx']].reset_index(drop=True)
    test_labels = [label2idx[lang] for lang in test_data['Language']]

    test_dataset = TextDataset(test_data['Text'].tolist(), test_labels, vocab, simple_tokenizer)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    # Device management - compatible with MPS (Mac)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    checkpoint = torch.load("models/best_model.pt", map_location=device)

    model = LanguageClassifier(
        vocab_size=checkpoint['vocab_size'],
        num_classes=checkpoint['num_classes']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Evaluate
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics and Confusion Matrix (Keeping your original logic)
    accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"Test Accuracy: {accuracy:.2f}%")

    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=idx2label.values(), yticklabels=idx2label.values(), cmap='Blues')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)')
    plt.savefig(Path('reports/figures/confusion_matrix.png'), dpi=300)
    plt.close()
    print("✓ Confusion matrix saved: reports/figures/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
