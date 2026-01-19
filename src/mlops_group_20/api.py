from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize and load model state
        model = LanguageClassifier(
            vocab_size=checkpoint['vocab_size'],
            num_classes=checkpoint['num_classes']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✓ Model and artifacts loaded successfully on {device}")
        print(f"✓ Supported languages: {list(idx2label.values())}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError("Model artifacts not found. Please run training first.")

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
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                max-width: 600px;
                width: 100%;
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 10px;
                font-size: 2em;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
                font-size: 0.95em;
            }
            textarea {
                width: 100%;
                padding: 15px;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 16px;
                resize: vertical;
                min-height: 150px;
                font-family: inherit;
                transition: border-color 0.3s;
            }
            textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            button {
                width: 100%;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                margin-top: 20px;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
            }
            button:active {
                transform: translateY(0);
            }
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            .result {
                margin-top: 20px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                text-align: center;
                display: none;
            }
            .result.show {
                display: block;
                animation: fadeIn 0.5s;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .language {
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
                margin-bottom: 5px;
            }
            .confidence {
                color: #666;
                font-size: 0.9em;
            }
            .error {
                background: #fee;
                color: #c33;
                padding: 15px;
                border-radius: 10px;
                margin-top: 20px;
                display: none;
            }
            .error.show {
                display: block;
            }
            .example {
                color: #999;
                font-size: 0.85em;
                margin-top: 10px;
                text-align: center;
            }
            .languages {
                margin-top: 20px;
                padding: 15px;
                background: #f0f4ff;
                border-radius: 10px;
                border-left: 4px solid #667eea;
            }
            .languages-title {
                font-weight: 600;
                color: #333;
                margin-bottom: 10px;
                font-size: 0.9em;
            }
            .languages-list {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }
            .language-tag {
                background: white;
                color: #667eea;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 0.85em;
                border: 1px solid #667eea;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌍 Language Detector</h1>
            <p class="subtitle">Enter text below to detect its language</p>
            
            <textarea id="textInput" placeholder="Type or paste any text here..."></textarea>
            <p class="example">Try: "Hello, how are you?" or "Bonjour, comment allez-vous?"</p>
            
            <div class="languages">
                <div class="languages-title">✨ Supported Languages:</div>
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
            // Load supported languages on page load
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
                    document.getElementById('languagesList').textContent = 'Failed to load languages';
                }
            }
            
            // Load languages when page loads
            window.addEventListener('load', loadLanguages);
            
            async function detectLanguage() {
                const text = document.getElementById('textInput').value.trim();
                const resultDiv = document.getElementById('result');
                const errorDiv = document.getElementById('error');
                const button = document.querySelector('button');
                
                // Hide previous results
                resultDiv.classList.remove('show');
                errorDiv.classList.remove('show');
                
                if (!text) {
                    errorDiv.textContent = 'Please enter some text first!';
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
                        throw new Error('Failed to detect language');
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
            
            // Allow Enter key to submit (with Shift+Enter for new line)
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