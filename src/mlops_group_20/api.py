import logging
import torch
import hydra
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from omegaconf import DictConfig
from mlops_group_20.model import LanguageClassifier
from mlops_group_20.data import simple_tokenizer

# Standard Logger setup (M14)
log = logging.getLogger(__name__)

app = FastAPI(title="Language Detection API")

# Global variables to store model and mappings
model = None
vocab = None
idx2label = None
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

class TextRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Language Detection API is running. Use /predict to detect language."}

@app.post("/predict")
def predict(request: TextRequest):
    """Predict the language of the input text."""
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is empty")
    
    # Preprocess the input text using global vocab
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

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Initialize the API by loading artifacts using paths from the Hydra config.
    """
    global model, vocab, idx2label
    
    log.info("Starting API initialization via Hydra...")
    
    try:
        # Load mappings and vocab using paths from cfg.paths
        log.info(f"Loading vocabulary and mappings from: {cfg.paths.splits_dir}")
        label_info = pd.read_pickle(f"{cfg.paths.splits_dir}/label_mappings.pkl")
        idx2label = label_info['idx2label']
        vocab = pd.read_pickle(f"{cfg.paths.splits_dir}/vocab.pkl")
        
        # Load model checkpoint using path from cfg.paths
        log.info(f"Loading model from: {cfg.paths.model_save_path}")
        checkpoint = torch.load(cfg.paths.model_save_path, map_location=device)
        
        # Initialize and load model state using model config parameters
        model = LanguageClassifier(
            vocab_size=checkpoint['vocab_size'],
            num_classes=checkpoint['num_classes'],
            hidden_dim=cfg.model.hidden_dim
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        log.info(f"✓ Model and artifacts loaded successfully on {device}")
        
    except Exception as e:
        log.error(f"Error loading model: {e}")
        raise RuntimeError("Model artifacts not found. Please run training first.")

    # Start the server using parameters from cfg.api
    import uvicorn
    log.info(f"Starting server on {cfg.api.host}:{cfg.api.port}")
    uvicorn.run(app, host=cfg.api.host, port=cfg.api.port)

if __name__ == "__main__":
    main()