## Developer: inkbytefo
## Modified: 2025-11-22

# AETHER-1 Phase 1 Training Guide (Cloud Linux)

## Ön Koşullar
- ✅ CUDA 12.1 Toolkit kurulu
- ✅ mamba-ssm kurulu
- ✅ Tesla T4 GPU (15GB VRAM)
- ✅ Conda environment: `aether`

---

## Adım 1: Tokenizer Eğitimi

### 1.1 Veri Hazırlığı ve Tokenizer Eğitimi
```bash
cd ~/AETHER  # veya projenin bulunduğu dizin
conda activate aether

# Data preparation script çalıştır (tokenizer + data)
python3 src/data/prepare_phase1_tr.py
```

**Bu script:**
- Turkish Wikipedia + TinyStories (English) yükler
- SentencePiece tokenizer eğitir (vocab_size=50257, unigram model)
- Tokenize edilmiş binary data oluşturur (`train.bin`, `val.bin`)
- Output: `data/corpus_v1/tokenizer.model`

**Beklenen Süre:** ~15-30 dakika (dataset download + tokenization)

### 1.2 Tokenizer Doğrulama
```bash
python3 -c "
from src.data.tokenizer import Tokenizer
tok = Tokenizer('data/corpus_v1/tokenizer.model')
print(f'Vocab Size: {len(tok)}')
print(f'Test: {tok.decode(tok.encode(\"Merhaba dünya!\")[\"input_ids\"])}')
"
```

**Beklenen Çıktı:**
```
Vocab Size: 50257
Test: Merhaba dünya!
```

---

## Adım 2: Config Kontrolü

### 2.1 Phase 1 Config Görüntüle
```bash
cat configs/phase1_tr.yaml
```

**Kritik Parametreler:**
- `vocab_size: 32000` → **HATA!** Tokenizer 50257, config 32000
- `max_steps: 10000` → İlk deney için uygun
- `batch_size: 32` → Tesla T4 için optimize

### 2.2 Config Güncelleme (Gerekli)
```bash
# vocab_size'ı tokenizer ile eşleştir
sed -i 's/vocab_size: 32000/vocab_size: 50257/' configs/phase1_tr.yaml

# Doğrulama
grep vocab_size configs/phase1_tr.yaml
```

**Beklenen:** `vocab_size: 50257`

---

## Adım 3: WandB Kurulumu (Opsiyonel ama Önerilen)

```bash
# WandB kurulumu
pip install wandb

# Login (API key gerekli: https://wandb.ai/authorize)
wandb login

# Offline mode (internet yoksa)
export WANDB_MODE=offline
```

---

## Adım 4: Training Başlatma

### 4.1 Hızlı Test (100 step)
```bash
# Test run (config'i geçici override)
python3 train.py \
  --config configs/phase1_tr.yaml \
  2>&1 | tee logs/phase1_test.log
```

**İlk 10 step'te kontrol edilecekler:**
- ✅ GPU kullanımı (`nvidia-smi` ile)
- ✅ Loss azalıyor mu?
- ✅ VRAM kullanımı (~8-10GB olmalı)
- ✅ OOM hatası yok

### 4.2 Full Training (10K steps)
```bash
# Logs dizini oluştur
mkdir -p logs

# Training başlat (background + log)
nohup python3 train.py \
  --config configs/phase1_tr.yaml \
  > logs/phase1_training.log 2>&1 &

# Process ID kaydet
echo $! > logs/phase1_training.pid
```

### 4.3 Training Monitoring
```bash
# Log takibi (real-time)
tail -f logs/phase1_training.log

# GPU monitoring
watch -n 1 nvidia-smi

# Process kontrolü
ps aux | grep train.py

# Training durdurma (gerekirse)
kill $(cat logs/phase1_training.pid)
```

---

## Adım 5: Checkpoint Doğrulama

### 5.1 Training Tamamlandıktan Sonra
```bash
# Model checkpoint kontrolü
ls -lh models/saved/aether_phase1.pt

# Checkpoint yükleme testi
python3 -c "
import torch
ckpt = torch.load('models/saved/aether_phase1.pt', map_location='cpu')
print(f'Checkpoint keys: {list(ckpt.keys())[:5]}')
print(f'Model size: {sum(p.numel() for p in ckpt.values()) / 1e6:.2f}M params')
"
```

### 5.2 Inference Testi
```bash
python3 inference.py \
  --checkpoint models/saved/aether_phase1.pt \
  --config configs/phase1_tr.yaml \
  --prompt "Merhaba, benim adım" \
  --max_length 50
```

---

## Beklenen Training Metrikleri

| Metric | Initial | 1K steps | 5K steps | 10K steps |
|--------|---------|----------|----------|-----------|
| Train Loss | ~10.5 | ~4.5 | ~3.2 | ~2.8 |
| Val Loss | ~10.8 | ~5.0 | ~3.8 | ~3.5 |
| GPU Util | 95-100% | 95-100% | 95-100% | 95-100% |
| VRAM | ~9GB | ~9GB | ~9GB | ~9GB |
| Step/sec | ~2.5 | ~2.5 | ~2.5 | ~2.5 |

**Total Training Time:** ~1.1 saat (10K steps × 0.4s/step)

---

## Sorun Giderme

### Hata: "RuntimeError: CUDA out of memory"
```bash
# Batch size azalt
sed -i 's/batch_size: 32/batch_size: 16/' configs/phase1_tr.yaml
```

### Hata: "FileNotFoundError: data/corpus_v1/train.bin"
```bash
# Data preparation tekrar çalıştır
python3 src/data/prepare_phase1_tr.py
```

### Hata: "vocab_size mismatch"
```bash
# Config'i tokenizer ile senkronize et
python3 -c "
from src.data.tokenizer import Tokenizer
tok = Tokenizer('data/corpus_v1/tokenizer.model')
print(f'Update config vocab_size to: {len(tok)}')
"
```

### Hata: "Triton CUDA assert triggered"
```bash
# Token ID range kontrolü
python3 diagnose_tokenizer.py
```

---

## Hızlı Başlangıç (Tek Komut)

```bash
#!/bin/bash
# quick_start_phase1.sh

set -e

echo "=== AETHER Phase 1 Quick Start ==="

# 1. Data Preparation
echo "[1/4] Preparing data..."
python3 src/data/prepare_phase1_tr.py

# 2. Config Fix
echo "[2/4] Fixing config..."
sed -i 's/vocab_size: 32000/vocab_size: 50257/' configs/phase1_tr.yaml

# 3. WandB Setup
echo "[3/4] WandB setup..."
export WANDB_MODE=offline

# 4. Training
echo "[4/4] Starting training..."
mkdir -p logs
nohup python3 train.py --config configs/phase1_tr.yaml > logs/phase1.log 2>&1 &
echo $! > logs/phase1.pid

echo "✅ Training started! Monitor with: tail -f logs/phase1.log"
```

**Kullanım:**
```bash
chmod +x quick_start_phase1.sh
./quick_start_phase1.sh
```

---

## Notlar

1. **Data Persistence:** Brev.dev instance geçiciyse, `data/` ve `models/` dizinlerini backup alın
2. **Gradient Accumulation:** Config'te `gradient_accumulation_steps: 4` → effective batch = 32×4 = 128
3. **Mixed Precision:** AMP (float16) otomatik aktif → 2x hız artışı
4. **Validation:** Her 100 step'te otomatik validation
5. **Checkpoint:** Training sonunda `models/saved/aether_phase1.pt` kaydedilir

---

## Sonraki Adımlar (Phase 2)

Phase 1 tamamlandıktan sonra:
1. Checkpoint'i `models/saved/aether_phase1.pt` olarak kaydet
2. `configs/phase2_tr.yaml` oluştur (Hebbian plasticity ekle)
3. `train.py --config configs/phase2_tr.yaml --resume_from models/saved/aether_phase1.pt`
