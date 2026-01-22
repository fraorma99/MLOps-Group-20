import pytest
import torch
from fastapi.testclient import TestClient
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from src.mlops_group_20.api import app, model, vocab, idx2label, device  # Ajusta import

client = TestClient(app)

class TestLanguageDetectionAPI:
    """Tests completos Language Detection API."""

    @pytest.fixture(autouse=True)
    def setup_model(self):
        """Mock model to test faster"""
        global model, vocab, idx2label

        idx2label = {0: 'Spanish', 1: 'English', 2: 'French'}
        vocab = {'hola que tal estás': 10, 'hello how are you': 20, 'bonjour comment allez-vous': 30, '<pad>': 0}

        # Mock model simple
        model = MagicMock()
        model.return_value = torch.tensor([[100, -50, -20]])

    def test_root_endpoint(self):
        """Returns a valid output"""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["message"] == "Language Detection API is running."

    def test_get_languages(self):
        """Import supported languages."""
        response = client.get("/languages")
        assert response.status_code == 200
        data = response.json()
        assert "supported_languages" in data
        assert isinstance(data["supported_languages"], list)

    def test_get_ui(self):
        """ui returns valid HTML page."""
        response = client.get("/ui")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert "<!DOCTYPE html>" in response.text

    @patch('src.mlops_group_20.api.simple_tokenizer')

    def test_predict_language_spanish(self, mock_tokenizer):
        """Predict a spanish text"""
        mock_tokenizer.return_value = ['hola', 'mundo']

        payload = {"text": "Hola mundo, esto es español."}
        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["predicted_language"] == "Spanish"
        assert data["status"] == "success"
        assert "input_text" in data

    @patch('src.mlops_group_20.api.simple_tokenizer')

    def test_predict_language_english(self, mock_tokenizer):
        """Predict a english text"""
        mock_tokenizer.return_value = ['hello', 'world']
        with patch.object(model, 'return_value', torch.tensor([[ -50, 100, -20]])):  # Predice idx 1
            payload = {"text": "Hello world this is English."}
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            assert response.json()["predicted_language"] == "English"

    def test_predict_empty_text(self):
        """API response with an empty text"""
        payload = {"text": ""}
        response = client.post("/predict", json=payload)
        assert response.status_code == 400
        assert "Text is empty" in response.json()["detail"]

    def test_predict_no_text(self):
        """API without 'text' field returns 422 validation error."""
        payload = {"foo": "bar"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    @patch('src.mlops_group_20.api.simple_tokenizer')
    def test_predict_long_text(self, mock_tokenizer):
        """API response with a very long text"""
        mock_tokenizer.return_value = ['token'] * 300  # >200
        payload = {"text": "a" * 10000}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_model_startup_error(self):
        """API model loading failure on startup."""
        with patch('src.mlops_group_20.api.load_artifacts', side_effect=RuntimeError("No model")):
            response = client.get("/languages")
            assert response.status_code == 500
