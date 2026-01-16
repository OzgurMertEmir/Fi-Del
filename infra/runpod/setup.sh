#!/bin/bash
# RunPod Setup Script for FiDel Training
#
# Run this on a fresh RunPod instance to set up the environment
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/YOUR_USER/FiDel/main/infra/runpod/setup.sh | bash
#   OR
#   bash setup.sh

set -e

echo "=============================================="
echo "FiDel RunPod Setup"
echo "=============================================="

# Update system
apt-get update -qq

# Install system dependencies
apt-get install -y -qq git curl wget

# Upgrade pip
pip install --upgrade pip -q

# Install ML dependencies
pip install -q \
    boto3 \
    pandas \
    pyarrow \
    torch \
    scikit-learn \
    xgboost \
    lightgbm \
    numpy

# Clone repo (or you can upload manually)
if [ ! -d "/workspace/FiDel" ]; then
    echo "Cloning FiDel repository..."
    cd /workspace
    git clone https://github.com/YOUR_USER/FiDel.git 2>/dev/null || echo "Clone failed - upload manually"
fi

# Create AWS credentials directory
mkdir -p ~/.aws

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Configure AWS credentials:"
echo "   export AWS_ACCESS_KEY_ID=your_key"
echo "   export AWS_SECRET_ACCESS_KEY=your_secret"
echo "   export AWS_DEFAULT_REGION=us-east-1"
echo ""
echo "2. Run training:"
echo "   cd /workspace/FiDel"
echo "   python scripts/runpod_train.py --model transformer --days 7 --epochs 50"
echo ""
echo "Available models: xgboost, lightgbm, lstm, gru, transformer"
echo ""
