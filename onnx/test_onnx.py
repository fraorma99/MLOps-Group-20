import onnxruntime as ort
import numpy as np
import torch
import pickle
from pathlib import Path

# Carga tu ONNX
onnx_path = Path("models/best_model.onnx")
session = ort.InferenceSession(str(onnx_path))

print("✅ Session OK")
print("Inputs:", [inp.name for inp in session.get_inputs()])
print("Outputs:", [out.name for out in session.get_outputs()])

# Load vocab para saber el tamaño
vocab = pickle.load(open("data/splits/vocab.pkl", "rb"))

# Test input (batch=1, seq_len=50)
dummy_input = np.random.randint(0, len(vocab), (1, 50)).astype(np.int64)
outputs = session.run(None, {"input_ids": dummy_input})

print(f"✅ Output shape: {outputs[0].shape}")
