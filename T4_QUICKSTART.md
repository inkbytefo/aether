## Developer: inkbytefo
## Modified: 2025-11-22

# AETHER-1 T4 GPU Cluster - Hızlı Başlangıç

## 1. Environment Setup (Brev.dev T4)

```bash
# GPU kontrolü
nvidia-smi

# PyTorch + CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Mamba dependencies
pip install mamba-ssm causal-conv1d>=1.1.0

# Other deps
pip install transformers datasets sentencepiece wandb PyYAML tqdm

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from mamba_ssm.modules.mamba_simple import Mamba; print('✅ Mamba OK')"
```

## 2. Data Preparation

```bash
cd AETHER

# Run optimized data prep (50M tokens, ~10 min on T4)
python src/data/prepare_phase1_optimized.py
```

**Output:**
- `data/corpus_v1/tokenizer.model`
- `data/corpus_v1/train.bin` (~45M tokens)
- `data/corpus_v1/val.bin` (~5M tokens)

## 3. Training

### Option A: Standard Model (~500M params)
```bash
# WandB setup (optional)
export WANDB_API_KEY="your_key"

# Start training
python train.py --config configs/hybrid_phase1.yaml
```

### Option B: Safe Mode (~150M params)
Eğer CUDA OOM alırsan:
```bash
python train.py --config configs/hybrid_phase1_t4_safe.yaml
```

## 4. Background Training

Bağlantı kesilse bile devam etsin:

```bash
nohup python train.py --config configs/hybrid_phase1.yaml > training.log 2>&1 &
echo $! > train.pid
tail -f training.log
```

**Stop:**
```bash
kill $(cat train.pid)
```

## 5. Monitoring

```bash
# GPU kullanımı (ayrı terminal)
watch -n 1 nvidia-smi

# Logs
tail -f training.log

# WandB dashboard
# https://wandb.ai/your-project/AETHER-1
```

## 6. Resume Training

Eğitim kesilirse:

```bash
python train.py \
  --config configs/hybrid_phase1.yaml \
  --resume_from models/saved/aether_phase1.pt
```

## Troubleshooting

### CUDA OOM
```bash
# 1. Safe mode config kullan
python train.py --config configs/hybrid_phase1_t4_safe.yaml

# 2. Veya manuel olarak batch size düşür
# configs/hybrid_phase1.yaml edit:
# batch_size: 10 -> 6
# seq_length: 512 -> 256
```

### Triton Error
```bash
pip uninstall triton
pip install triton==2.1.0
```

### Dataset Loading Slow
```bash
# data prep script'te token sayısını düşür:
# max_tokens = 50_000_000 -> 25_000_000
```

## Expected Timeline (T4 GPU)

- Data Prep: ~10 minutes (50M tokens)
- Training Config A: ~30 hours (100K steps)
- Training Config B: ~20 hours (150K steps)

## Model Checkpoints

Otomatik kaydedilir:
- `models/saved/aether_phase1.pt` (final)
- Her 2000 step'te backup

## Validation

Training sırasında her 500 step'te otomatik validation yapılır.
WandB'de `val_loss` grafiğini izle.

---

## Quick Commands Cheat Sheet

```bash
# Setup
nvidia-smi && python -c "import torch; print(torch.cuda.is_available())"

# Prepare
python src/data/prepare_phase1_optimized.py

# Train
nohup python train.py --config configs/hybrid_phase1.yaml > train.log 2>&1 &

# Monitor
tail -f train.log

# Resume
python train.py --config configs/hybrid_phase1.yaml --resume_from models/saved/aether_phase1.pt
```
