# export_onnx.py - Completo con TU base ResNet
import torch
import torchvision
import onnx
import pickle
from pathlib import Path
from mlops_group_20.model import LanguageClassifier

# 1. Carga TU best_model.pt
checkpoint = torch.load("models/best_model.pt", map_location='cpu')
vocab = pickle.load(open("data/splits/vocab.pkl", "rb"))

# 2. Recrear TU modelo exacto
model = LanguageClassifier(
    vocab_size=len(vocab),
    embed_dim=128,  # ← De tu config.yaml
    hidden_dim=256,
    num_layers=2,
    dropout=0.3,
    num_classes=checkpoint.get('num_classes', 10)
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 3. Export TU modelo
dummy_input = torch.randint(0, len(vocab), (1, 50))

torch.onnx.export(
    model,
    dummy_input,
    "models/best_model_fixed.onnx",
    input_names=['input_ids'],
    output_names=['output'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'seq_len'},
        'output': {0: 'batch_size'}
    },
    opset_version=12,
    do_constant_folding=True
)
