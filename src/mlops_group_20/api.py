import os
import time
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
from pathlib import Path
import pandas as pd
from mlops_group_20.model import LanguageClassifier
from mlops_group_20.data import simple_tokenizer
from typing import Optional

# Optional Prometheus instrumentation (enabled via ENABLE_METRICS=true)
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "false").lower() == "true"
if ENABLE_METRICS:
    try:
        from prometheus_client import (
            Counter,
            Gauge,
            Histogram,
            CONTENT_TYPE_LATEST,
            generate_latest,
        )

        REQUESTS_TOTAL = Counter(
            "api_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "http_status"],
        )
        REQUEST_LATENCY = Histogram(
            "api_request_latency_seconds",
            "Latency of HTTP requests in seconds",
            ["endpoint"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )
        PREDICTIONS_TOTAL = Counter(
            "api_predictions_total",
            "Total number of predictions made",
            ["language"],
        )
        INFERENCE_LATENCY = Histogram(
            "api_inference_latency_seconds",
            "Model inference latency in seconds",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )
        INPUT_TEXT_LENGTH = Histogram(
            "api_input_text_length",
            "Length of input text (characters)",
            buckets=(0, 20, 50, 100, 200, 400, 800, 1600),
        )
        MODEL_LOADED = Gauge(
            "api_model_loaded",
            "Whether the model is loaded successfully (1 yes, 0 no)",
        )
    except Exception:
        # If prometheus_client isn't available, disable metrics gracefully
        ENABLE_METRICS = False

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
        # Resolve paths relative to project root
        project_root = Path(__file__).parent.parent.parent
        splits_dir = project_root / "data" / "splits"
        models_dir = project_root / "models"
        
        # Load mappings and vocab
        label_mappings_path = splits_dir / "label_mappings.pkl"
        vocab_path = splits_dir / "vocab.pkl"
        
        if not label_mappings_path.exists():
            raise FileNotFoundError(f"Label mappings not found at {label_mappings_path}")
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary not found at {vocab_path}")
        
        label_info = pd.read_pickle(label_mappings_path)
        idx2label = label_info['idx2label']
        vocab = pd.read_pickle(vocab_path)
        
        # Load model checkpoint
        model_path = models_dir / "best_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
        except RuntimeError as e:
            # If standard load fails, try with weights_only=False (older PyTorch format)
            print(f"Standard torch.load failed: {e}. Trying with weights_only=False...")
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Initialize and load model state
        model = LanguageClassifier(
            vocab_size=checkpoint['vocab_size'],
            num_classes=checkpoint['num_classes']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✓ Model and artifacts loaded successfully on {device}")
        print(f"✓ Supported languages: {list(idx2label.values())}")
        if ENABLE_METRICS:
            try:
                MODEL_LOADED.set(1)
            except Exception:
                pass
    except Exception as e:
        print(f"Error loading model: {e}")
        if ENABLE_METRICS:
            try:
                MODEL_LOADED.set(0)
            except Exception:
                pass
        raise RuntimeError("Model artifacts not found. Please run training first.")

# Lightweight HTTP metrics middleware (only when enabled)
if ENABLE_METRICS:
    @app.middleware("http")
    async def metrics_middleware(request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        try:
            elapsed = time.perf_counter() - start
            endpoint = request.url.path
            REQUESTS_TOTAL.labels(
                method=request.method, endpoint=endpoint, http_status=str(response.status_code)
            ).inc()
            REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
        except Exception:
            # Never let metrics break the API
            pass
        return response

@app.get("/")
def root():
    return {"message": "Language Detection API is running. Use /predict to detect language."}

@app.get("/languages")
def get_languages():
    """Return the list of supported languages."""
    if idx2label is None:
        raise HTTPException(status_code=500, detail="Model not loaded yet")
    languages = sorted(list(idx2label.values()))
    return {"supported_languages": languages}

@app.get("/ui", response_class=HTMLResponse)
def ui():
    """Serve a simple web UI for language detection."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Language Detector</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                color: #333;
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: #ffffff;
                border: 1px solid #e1e5e9;
                padding: 40px;
                max-width: 600px;
                width: 100%;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            }
            h1 {
                font-family: 'Inter', sans-serif;
                font-weight: 700;
                color: #4a5568;
                text-align: center;
                margin-bottom: 10px;
                font-size: 2.5em;
                letter-spacing: -0.02em;
            }
            .subtitle {
                text-align: center;
                color: #718096;
                margin-bottom: 30px;
                font-size: 1.1em;
                font-weight: 400;
            }
            textarea {
                width: 100%;
                padding: 16px;
                border: 2px solid #e2e8f0;
                background: #fafbff;
                color: #2d3748;
                font-size: 16px;
                font-family: inherit;
                resize: vertical;
                min-height: 150px;
                transition: all 0.2s;
                border-radius: 0;
            }
            textarea:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
            }
            button {
                width: 100%;
                padding: 16px;
                background: #edf2f7;
                color: #4a5568;
                border: 2px solid #e2e8f0;
                font-size: 16px;
                font-weight: 600;
                font-family: inherit;
                cursor: pointer;
                margin-top: 20px;
                transition: all 0.2s;
                border-radius: 0;
            }
            button:hover {
                background: #e6fffa;
                border-color: #38b2ac;
                color: #285e61;
            }
            button:active {
                background: #c7f3e9;
            }
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                background: #f7fafc;
            }
            .result {
                margin-top: 20px;
                padding: 24px;
                background: #f0fff4;
                border: 2px solid #c6f6d5;
                text-align: center;
                display: none;
            }
            .result.show {
                display: block;
                animation: fadeIn 0.3s;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .language {
                font-size: 2.8em;
                font-weight: 700;
                color: #22543d;
                margin-bottom: 8px;
                font-family: 'Inter', sans-serif;
                letter-spacing: -0.03em;
            }
            .confidence {
                color: #4a5568;
                font-size: 1.1em;
                font-weight: 500;
            }
            .error {
                background: #fed7d7;
                color: #742a2a;
                border: 2px solid #fc8181;
                padding: 16px;
                margin-top: 20px;
                display: none;
            }
            .error.show {
                display: block;
            }
            .example {
                color: #a0aec0;
                font-size: 0.95em;
                margin-top: 12px;
                text-align: center;
                font-style: italic;
            }
            .languages {
                margin-top: 20px;
                padding: 16px;
                background: #ebf8ff;
                border: 2px solid #bee3f8;
            }
            .languages-title {
                font-weight: 600;
                color: #2b6cb0;
                margin-bottom: 12px;
                font-size: 1em;
            }
            .languages-list {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }
            .language-tag {
                background: #ffffff;
                color: #2c5282;
                padding: 8px 14px;
                border: 1px solid #bee3f8;
                font-size: 0.9em;
                font-weight: 500;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Language Detector</h1>
            <p class="subtitle">Enter text below to detect its language</p>
            
            <textarea id="textInput" placeholder="Type or paste any text here..."></textarea>
            <p class="example">Try: "Hello, how are you?" or "Bonjour, comment allez-vous?"</p>
            
            <div class="languages">
                <div class="languages-title">Supported Languages:</div>
                <div class="languages-list" id="languagesList">Loading...</div>
            </div>
            
            <button onclick="detectLanguage()">Detect Language</button>
            
            <div id="result" class="result">
                <div class="language" id="language"></div>
                <div class="confidence" id="confidence"></div>
            </div>
            
            <div id="error" class="error"></div>
        </div>

        <script>
            async function loadLanguages() {
                try {
                    const response = await fetch('/languages');
                    const data = await response.json();
                    const languages = data.supported_languages;
                    const listDiv = document.getElementById('languagesList');
                    
                    listDiv.innerHTML = languages.map(lang => 
                        `<span class="language-tag">${lang}</span>`
                    ).join('');
                } catch (error) {
                    console.error('Failed to load languages:', error);
                    document.getElementById('languagesList').textContent = 'Failed to load';
                }
            }
            
            window.addEventListener('load', loadLanguages);
            
            async function detectLanguage() {
                const text = document.getElementById('textInput').value.trim();
                const resultDiv = document.getElementById('result');
                const errorDiv = document.getElementById('error');
                const button = document.querySelector('button');
                
                resultDiv.classList.remove('show');
                errorDiv.classList.remove('show');
                
                if (!text) {
                    errorDiv.textContent = 'Please enter some text first.';
                    errorDiv.classList.add('show');
                    return;
                }
                
                button.disabled = true;
                button.textContent = 'Detecting...';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text: text })
                    });
                    
                    if (!response.ok) {
                        throw new Error('Detection failed');
                    }
                    
                    const data = await response.json();
                    
                    document.getElementById('language').textContent = data.predicted_language;
                    document.getElementById('confidence').textContent = 
                        data.confidence ? `Confidence: ${(data.confidence * 100).toFixed(1)}%` : '';
                    resultDiv.classList.add('show');
                    
                } catch (error) {
                    errorDiv.textContent = 'Error: ' + error.message;
                    errorDiv.classList.add('show');
                } finally {
                    button.disabled = false;
                    button.textContent = 'Detect Language';
                }
            }
            
            document.getElementById('textInput').addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    detectLanguage();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)



# Expose Prometheus metrics endpoint when enabled
if ENABLE_METRICS:
    @app.get("/metrics")
    def metrics() -> Response:
        try:
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
        except Exception:
            # Fallback empty response if something goes wrong
            return Response(status_code=500)

@app.post("/predict")
def predict(request: TextRequest):
    """Predict the language of the input text."""
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is empty")
    
    # Preprocess the input text
    if ENABLE_METRICS:
        try:
            INPUT_TEXT_LENGTH.observe(len(request.text))
        except Exception:
            pass

    tokens = simple_tokenizer(request.text)[:200]
    indices = [vocab[token] for token in tokens]
    
    # Padding
    if len(indices) < 200:
        indices += [0] * (200 - len(indices))
        
    input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
    
    # Inference
    if ENABLE_METRICS:
        inf_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_idx = outputs.max(1)
        prediction = idx2label[predicted_idx.item()]
    if ENABLE_METRICS:
        try:
            INFERENCE_LATENCY.observe(time.perf_counter() - inf_start)
            PREDICTIONS_TOTAL.labels(language=str(prediction)).inc()
        except Exception:
            pass
    
    return {
        "input_text": request.text,
        "predicted_language": prediction,
        "status": "success"
    }
