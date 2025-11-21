#!/bin/bash

# AETHER-1 Turkish AGI Training Script for Cloud (Tesla T4)
# Usage: bash run_cloud_training.sh

set -e # Exit on error

echo "🚀 Starting AETHER-1 Cloud Training Setup..."

# 1. Install Dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt
pip install sentencepiece datasets wandb mamba-ssm causal-conv1d

# 2. Login to WandB (Optional, requires API key)
# echo "🔑 Logging into WandB..."
# wandb login

# 3. Prepare Data & Tokenizer
echo "morphology 🧠 Preparing Turkish Corpus & Tokenizer..."
python src/data/prepare_tr_corpus.py --output_dir data/corpus_v1 --vocab_size 50257

# 4. Run Training
echo "🔥 Starting Mamba Training..."
# We use the turkish_base.yaml config we created
python train.py --config configs/turkish_base.yaml

echo "✅ Training Complete!"
