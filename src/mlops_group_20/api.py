from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from pathlib import Path
import pandas as pd
from mlops_group_20.model import LanguageClassifier
from mlops_group_20.data import simple_tokenizer

app = FastAPI(title="Language Detection API")

# Define the request body structure
class TextRequest(BaseModel):
    text: str

# Global variables to store model and mappings
model = None
vocab = None
idx2label = None
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

@app.on_event("startup")
def load_artifacts():
    """Load model and metadata on startup."""
    global model, vocab, idx2label
    
    try:
        # Load mappings and vocab
        label_info = pd.read_pickle("data/splits/label_mappings.pkl")
        idx2label = label_info['idx2label']
        vocab = pd.read_pickle("data/splits/vocab.pkl")
        
        # Load model checkpoint
        checkpoint = torch.load("models/best_model.pt", map_location=device)
        
        # Initialize and load model state
        model = LanguageClassifier(
            vocab_size=checkpoint['vocab_size'],
            num_classes=checkpoint['num_classes']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✓ Model and artifacts loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError("Model artifacts not found. Please run training first.")

@app.get("/")
def root():
    return {"message": "Language Detection API is running. Use /predict to detect language."}

@app.post("/predict")
def predict(request: TextRequest):
    """Predict the language of the input text."""
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is empty")
    
    # Preprocess the input text
    tokens = simple_tokenizer(request.text)[:200]
    indices = [vocab[token] for token in tokens]
    
    # Padding
    if len(indices) < 200:
        indices += [0] * (200 - len(indices))
        
    input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_idx = outputs.max(1)
        prediction = idx2label[predicted_idx.item()]
    
    return {
        "input_text": request.text,
        "predicted_language": prediction,
        "status": "success"
    }