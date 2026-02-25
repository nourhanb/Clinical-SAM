#!/bin/bash
# Setup script to recreate environment with Python 3.8

set -e

echo "=== Setting up Clinical-SAM with Python 3.8 ==="
echo ""

# Check if Python 3.8 is available
if ! command -v python3.8 &> /dev/null; then
    echo "ERROR: Python 3.8 is not installed."
    echo "Please run first:"
    echo "  sudo apt update"
    echo "  sudo apt install -y python3.8 python3.8-venv python3.8-dev"
    exit 1
fi

echo "✓ Python 3.8 found: $(python3.8 --version)"
echo ""

# Remove old venv
if [ -d "venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf venv
fi

# Create new venv with Python 3.8
echo "Creating virtual environment with Python 3.8..."
python3.8 -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate and upgrade pip
echo "Activating virtual environment and upgrading pip..."
source venv/bin/activate
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"
echo ""

# Install Python requirements
echo "Installing Python requirements (this may take a while)..."
pip install --find-links https://download.pytorch.org/whl/torch_stable.html -r requirements.txt
echo "✓ Python requirements installed"
echo ""

# Install RepViT SAM package
if [ -d "RepViT/sam" ]; then
    echo "Installing RepViT SAM package..."
    cd RepViT/sam
    pip install -e .
    cd ../..
    echo "✓ RepViT SAM package installed"
fi
echo ""

echo "=== Setup Complete! ==="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""

