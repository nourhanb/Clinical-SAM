#!/bin/bash

# Install dependencies for the project
sudo apt install -y python3-dev python3-setuptools libjpeg-dev zlib1g-dev wget

# Install requirements
pip install -r requirements.txt

# Clone RepViT (provides the RepViT-SAM backbone) if it is not already present
if [ ! -d RepViT ]; then
    git clone https://github.com/THU-MIG/RepViT.git RepViT
fi

# Navigate to RepViT and install the SAM package
cd RepViT/sam
pip install --user -e . 

# Dowload the sam weights using wget
mkdir weights
cd weights

if [ ! -f repvit_sam.pt ]; then
    wget https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_sam.pt
else
    echo "repvit_sam.pt already exists, skipping download."
fi

echo "Dependencies installed successfully"


