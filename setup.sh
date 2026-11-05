#!/usr/bin/env bash
# One-time setup for the Document Q&A API (Linux / Debian-Ubuntu, also works on macOS with brew).
set -e

echo "==> Creating data folder"
mkdir -p data

echo "==> Installing Tesseract OCR (system package)"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y tesseract-ocr python3-venv python3-pip
  # For Hindi OCR also run: sudo apt-get install -y tesseract-ocr-hin
elif command -v brew >/dev/null 2>&1; then
  brew install tesseract
else
  echo "Install Tesseract manually: https://tesseract-ocr.github.io/tessdoc/Installation.html"
fi

echo "==> Creating virtual environment (.venv)"
python3 -m venv .venv
source .venv/bin/activate

echo "==> Installing Python packages"
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Pulling Ollama models (Ollama must be installed: https://ollama.com/download)"
if command -v ollama >/dev/null 2>&1; then
  ollama pull nomic-embed-text
  ollama pull llama3.2:3b
else
  echo "Ollama not found. Install it, then run:"
  echo "  ollama pull nomic-embed-text"
  echo "  ollama pull llama3.2:3b"
fi

echo
echo "Setup done. Put your documents in ./data and start the server with:  ./run.sh"