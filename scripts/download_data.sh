#!/bin/bash
set -e
echo "Downloading language dataset..."
mkdir -p data/raw
uv run kaggle datasets download -d basilb2s/language-detection -p data/raw/
cd data/raw/
unzip -o language-detection.zip
rm language-detection.zip
mv "Language Detection.csv" language_detection.csv  # Fix Kaggle spaces
cd ../..
echo "Dataset ready: data/raw/language_detection.csv"
