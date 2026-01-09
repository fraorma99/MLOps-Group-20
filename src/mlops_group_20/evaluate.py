import pickle
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from mlops_group_20.model import LanguageClassifier

from collections import Counter

class Vocabulary:
    """Simple vocabulary class to replace torchtext vocab."""  #Copy entire class from train.py
    def __init__(self, min_freq=2):
        self.token2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2token = {0: '<pad>', 1: '<unk>'}
        self.min_freq = min_freq
        self.default_index = 1
    def build(self, token_lists):
        counter = Counter()
        for tokens in token_lists: counter.update(tokens)
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
    def __len__(self): return len(self.token2idx)
    def __getitem__(self, token): return self.token2idx.get(token, self.default_index)

def simple_tokenizer(text):
    """Simple character-level tokenizer."""
    return list(text.lower())

class TextDataset(torch.utils.data.Dataset):  # Full class copy from train.py
    def __init__(self, texts, labels, vocab, tokenizer, max_len=200):
        self.texts = texts; self.labels = labels
        self.vocab = vocab; self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text, label = self.texts[idx], self.labels[idx]
        tokens = self.tokenizer(text)[:self.max_len]
        indices = [self.vocab[token] for token in tokens]
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)



def evaluate():
    #Load the data the same way as train such that we can test on unseen test data
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
    
    from train import simple_tokenizer, TextDataset  # Copy classes/functions
    tokenizer = simple_tokenizer
    
    test_dataset = TextDataset(
        test_data['Text'].tolist(), 
        test_labels, 
        vocab, tokenizer
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    
    #Loads the best model found during the training stage in train.py
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    # Metrics
    accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=idx2label.values()))
    
    # Confusion Matrix Visualization
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=idx2label.values(), 
                yticklabels=idx2label.values(), cmap='Blues')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(Path('reports/figures/confusion_matrix.png'), dpi=300, bbox_inches='tight')
    
    #plt.show()
    plt.close()
    print("✓ Confusion matrix saved: reports/figures/confusion_matrix.png")
    
    # Per-language accuracy
    lang_acc = {}
    for lang_idx, lang in idx2label.items():
        lang_mask = [l == lang_idx for l in all_labels]  # True for this language
        lang_correct = sum(1 for i, (p, l) in enumerate(zip(all_preds, all_labels)) 
                        if lang_mask[i] and p == l)
        lang_total = sum(lang_mask)
        if lang_total > 0:
            acc = 100 * lang_correct / lang_total
            lang_acc[lang] = acc

    print("\nPer-language accuracy:")
    for lang, acc in sorted(lang_acc.items(), key=lambda x: x[1], reverse=True):
        print(f"{lang}: {acc:.1f}%")    

if __name__ == "__main__":
    evaluate()
