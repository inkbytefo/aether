# 🚀 AETHER-1 Quickstart Guide

This guide will help you set up the environment, prepare the Turkish corpus, and start training the **AETHER-1** model (Mamba architecture) on a GPU cluster (Tesla T4 or better).

## 1. Environment Setup

Ensure you have Python 3.10+ and CUDA installed.

```bash
# 1. Clone the repository (if not already done)
# git clone ...
# cd AETHER

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install sentencepiece datasets wandb mamba-ssm causal-conv1d
```

## 2. Data Preparation (Critical)

Before training, you must download the datasets, train the tokenizer, and generate binary files.

**Run this command:**
```bash
# Generates 60% TR Wiki, 30% Code, 10% Math mix
# Trains Unigram Tokenizer (Vocab: 50,257)
# Outputs binary files to data/corpus_v1/
python src/data/prepare_tr_corpus.py --output_dir data/corpus_v1 --vocab_size 50257
```
*Note: This process is CPU-intensive and may take 10-20 minutes depending on your internet connection and CPU.*

## 3. Training (Phase 1)

Start the training loop using the Turkish-optimized Mamba configuration.

**Run on GPU:**
```bash
python train.py --config configs/turkish_base.yaml
```

### Configuration Highlights (`configs/turkish_base.yaml`)
- **Model:** Mamba (SSM), `d_model=768`, `n_layer=24`, `d_state=32` (optimized for Turkish morphology).
- **Context:** 2048 tokens.
- **Hardware:** Designed to fit comfortably on a **Tesla T4 (16GB)**.

## 4. Monitoring

Training metrics are logged to **Weights & Biases (WandB)**.
- **Train Loss:** Should decrease steadily.
- **Aux Loss (Z-Loss):** Keeps logits stable.

## 5. Troubleshooting

- **OOM (Out of Memory):** Reduce `batch_size` in `configs/turkish_base.yaml` (e.g., from 32 to 16 or 8).
- **Tokenizer Mismatch:** The script automatically syncs config vocab size with the tokenizer, but ensure you ran Step 2 correctly.
