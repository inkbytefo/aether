# AETHER-1 Training Guide

## Phase 1: Foundation (The Infant)
**Goal:** Train the Mamba backbone on TinyStories to learn basic language structure.

### 1. Setup (Cloud/Local GPU)
Ensure you are in the project root and virtual environment is active.
```bash
pip install -r requirements.txt
```

### 2. Start Training
```bash
python train.py --config configs/config.yaml
```
*   **Expected Output:** Loss should decrease from ~10.0 to ~2.0 over 1000 steps.
*   **Duration:** ~15-30 mins on a T4/A100 GPU.
*   **Artifacts:** Model saved to `models/saved/aether_phase1.pt`.

---

## Phase 2: Structural Thinking (The Child)
**Goal:** Train on Code (MBPP) and Math (GSM8K) to learn logic.

### 1. Prepare Data
Run the data preparation scripts (if not already done):
```bash
python src/data/prepare_code.py
python src/data/prepare_math.py
```

### 2. Update Config (Optional)
Create a new config for Phase 2 if needed (e.g., `configs/phase2.yaml`), or use the default.
*   *Note: We will implement specific Phase 2 training logic in the next steps, but you can run the current `train.py` on the new data by updating the dataset path in config.*

### 3. Start Training
(Command will be updated once Phase 2 specific training script is ready. For now, focus on Phase 1).
