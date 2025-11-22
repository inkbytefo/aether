#!/bin/bash
## Developer: inkbytefo
## Modified: 2025-11-22

# AETHER Phase 1 Training Launcher
# Usage: bash START_TRAINING.sh

set -e

echo "=== AETHER Phase 1 Training Launcher ==="

# 1. Environment Check
echo "[1/4] Checking environment..."
conda activate aether 2>/dev/null || source activate aether
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'"
echo "✅ CUDA available"

# 2. Verify Data Files
echo "[2/4] Verifying data files..."
if [ ! -f "data/corpus_v1/tokenizer.model" ]; then
    echo "❌ Tokenizer not found! Run: python3 src/data/prepare_phase1_tr.py"
    exit 1
fi

if [ ! -f "data/corpus_v1/train.bin" ]; then
    echo "❌ Training data not found! Run: python3 src/data/prepare_phase1_tr.py"
    exit 1
fi

echo "✅ Data files verified"

# 3. WandB Setup
echo "[3/4] WandB setup..."
if ! wandb status &>/dev/null; then
    echo "⚠️  WandB not logged in. Using offline mode."
    export WANDB_MODE=offline
fi

# 4. Start Training
echo "[4/4] Starting training..."
mkdir -p logs

nohup python3 train.py --config configs/phase1_tr.yaml > logs/phase1_training.log 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > logs/phase1_training.pid

echo ""
echo "✅ Training started!"
echo "   PID: $TRAIN_PID"
echo "   Config: configs/phase1_tr.yaml"
echo "   Log: logs/phase1_training.log"
echo ""
echo "Monitor with:"
echo "  tail -f logs/phase1_training.log"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Stop with:"
echo "  kill $TRAIN_PID"
