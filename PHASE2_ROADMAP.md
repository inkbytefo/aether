## Developer: inkbytefo
## Modified: 2025-11-22

# AETHER Phase 2: Hebbian Plasticity & Adaptive Memory

## Genel Bakış

**Amaç:** Phase 1'deki statik Mamba modelini, **Hebbian Learning** (Fast Weights) ile dinamik hafıza yeteneğine sahip bir sisteme dönüştürmek.

**Metafor:** Phase 1 = Bebek (temel dil), Phase 2 = Çocuk (öğrenmeyi öğrenme, context içinde adaptasyon)

---

## Teorik Temel: Hebbian Plasticity

### Neden Gerekli?

Standart transformatörler ve Mamba modelleri **statik weights** kullanır:
- Eğitim sırasında öğrenilen bilgi sabittir
- Inference sırasında yeni bilgi öğrenemez
- Uzun context'te "unutma" problemi var

**Hebbian Learning** çözümü:
- **Fast Weights:** Context içinde dinamik olarak güncellenen ağırlıklar
- **Associative Memory:** "Neurons that fire together, wire together"
- **Zero-shot Adaptation:** Yeni pattern'leri inference sırasında öğrenme

### Matematiksel Model

```
A_{t+1} = λ * A_t + η * (k_t ⊗ v_t^T)

Output = x @ W_static + (x @ A_t) @ W_read
```

**Parametreler:**
- `A_t`: Dinamik hafıza matrisi (d_model × d_model)
- `λ`: Decay rate (0.9) - eski hafızayı ne kadar koruyacağız
- `η`: Learning rate (0.1) - yeni bilgiyi ne kadar hızlı öğreneceğiz
- `k_t, v_t`: Key-Value projections (attention benzeri)

---

## Mimari Değişiklikler

### 1. PlasticMambaBlock Yapısı

```
Input
  ↓
[RMSNorm]
  ↓
[Mamba Mixer] ← Standart SSM (temporal dependencies)
  ↓
[Residual +]
  ↓
[RMSNorm]
  ↓
[HebbianMemory] ← YENİ! (associative recall)
  ↓
[Residual +]
  ↓
Output
```

**Özellikler:**
- Her layer'da **iki aşamalı işlem**: Mamba (sequence) + Hebbian (memory)
- Hebbian state her layer için ayrı tutulur (`hebbian_0`, `hebbian_1`, ...)
- Inference sırasında state korunur (autoregressive generation)

### 2. HebbianMemory Layer

**Bileşenler:**
- `w_key`: Input → Key projection
- `w_value`: Input → Value projection  
- `w_query`: Input → Query projection
- `out_proj`: Memory output → Final output

**Forward Pass:**
1. **Read:** `memory_out = query @ A_t`
2. **Update:** `A_{t+1} = λ * A_t + η * (key ⊗ value^T)`
3. **Output:** `out_proj(memory_out)`

---

## Eğitim Stratejisi

### Data Composition (Phase 2)

| Dataset | Oran | Amaç |
|---------|------|------|
| Turkish Wikipedia | 40% | Genel bilgi, Turkish syntax |
| TinyStories (EN) | 20% | Narrative structure |
| Python Code | 25% | Logical reasoning, syntax |
| GSM8K (Math) | 15% | Arithmetic, problem solving |

**Total Tokens:** ~500M (Phase 1'den 5x artış)

### Hyperparameters

```yaml
model:
  d_model: 512  # 256'dan artırıldı (daha fazla capacity)
  n_layer: 12   # 6'dan artırıldı (daha derin reasoning)
  vocab_size: 50257
  ssm_cfg:
    d_state: 16
    d_conv: 4
    expand: 2
  use_plasticity: true  # ← KRITIK!
  hebbian_cfg:
    learning_rate: 0.1
    decay_rate: 0.9

training:
  batch_size: 8
  learning_rate: 0.0003  # Daha düşük (plasticity için stable)
  max_steps: 50000  # 10K'dan artırıldı
  gradient_accumulation_steps: 16
  warmup_steps: 1000
```

### Curriculum Learning

**Stage 1 (Steps 0-10K):** Foundation
- Turkish Wiki + TinyStories
- Hebbian layers **frozen** (sadece Mamba eğitilir)
- Phase 1 checkpoint'ten başla

**Stage 2 (Steps 10K-30K):** Plasticity Activation
- Tüm datasets karışık
- Hebbian layers **unfreeze**
- Learning rate 0.0003 → 0.0001 (cosine decay)

**Stage 3 (Steps 30K-50K):** Refinement
- Code + Math ağırlıklı (60%)
- Hebbian learning rate decay: 0.1 → 0.05

---

## Beklenen Yetenekler

### 1. In-Context Learning (Few-Shot)

**Örnek:**
```
Prompt:
"İngilizce: cat → Türkçe: kedi
İngilizce: dog → Türkçe: köpek
İngilizce: bird → Türkçe: ?"

Expected Output: "kuş"
```

**Mekanizma:** Hebbian layer, prompt içindeki pattern'i öğrenir ve genelleştirir.

### 2. Associative Recall

**Örnek:**
```
Context:
"Einstein doğum tarihi: 1879
Newton doğum tarihi: 1643
Galileo doğum tarihi: 1564"

Query: "Einstein ne zaman doğdu?"
Expected: "1879"
```

### 3. Code Completion (Syntax Awareness)

**Örnek:**
```python
def fibonacci(n):
    if n <= 1:
        return n
    # Model buradan devam etmeli
```

**Expected:**
```python
    return fibonacci(n-1) + fibonacci(n-2)
```

---

## Implementation Checklist

### Hazırlık (Phase 1 → Phase 2 Geçiş)

- [x] Phase 1 checkpoint kaydedildi (`aether_phase1.pt`)
- [ ] Phase 2 config oluştur (`configs/phase2_tr.yaml`)
- [ ] Data preparation script güncelle (Code + Math datasets ekle)
- [ ] PlasticMambaLLM test et (forward pass, gradient flow)

### Eğitim

- [ ] Stage 1: Hebbian frozen training (10K steps)
- [ ] Stage 2: Full plasticity training (20K steps)
- [ ] Stage 3: Refinement (20K steps)
- [ ] Checkpoint her 5K step'te kaydet

### Validation

- [ ] In-context learning test (few-shot translation)
- [ ] Associative recall test (needle-in-haystack)
- [ ] Code completion benchmark (HumanEval subset)
- [ ] Math reasoning (GSM8K validation set)

---

## Teknik Detaylar

### 1. Checkpoint Loading (Phase 1 → Phase 2)

```python
# Phase 1 checkpoint yükle
checkpoint = torch.load("models/saved/aether_phase1.pt")

# PlasticMambaLLM oluştur
model = PlasticMambaLLM(config)

# Partial loading (Mamba layers eşleşir, Hebbian yeni)
model.load_state_dict(checkpoint, strict=False)

# Hebbian layers random init olacak (expected)
```

### 2. Hebbian State Management (Inference)

```python
# Inference başlangıcı
inference_params = {}

for token_id in input_ids:
    output = model(token_id, inference_params=inference_params)
    # inference_params otomatik güncellenir:
    # {"hebbian_0": A_0, "hebbian_1": A_1, ...}
```

### 3. Memory Optimization

**Problem:** Hebbian state (batch × d_model × d_model) çok büyük
- Batch=8, d_model=512 → 8 × 512 × 512 × 4 bytes = **8MB per layer**
- 12 layer → **96MB** ek VRAM

**Çözüm:**
- Training: Gradient checkpointing kullan
- Inference: Batch size = 1 (generation için yeterli)

---

## Beklenen Metrikler

### Training Metrics

| Metric | Phase 1 (Final) | Phase 2 (Target) |
|--------|-----------------|------------------|
| Train Loss | ~2.5 | ~1.8 |
| Val Loss | ~3.0 | ~2.2 |
| Perplexity | ~20 | ~9 |
| Code Accuracy | N/A | >40% (HumanEval) |
| Math Accuracy | N/A | >25% (GSM8K) |

### Plasticity Metrics (Yeni!)

- **Hebbian Activation:** Ortalama `||A_t||_F` (Frobenius norm)
- **Memory Utilization:** Non-zero entries in A_t
- **Adaptation Speed:** Few-shot accuracy vs. num examples

---

## Sorun Giderme

### 1. "Hebbian layers not learning"

**Belirti:** Hebbian activation norm çok düşük (<0.01)
**Çözüm:**
- Learning rate artır: 0.1 → 0.3
- Decay rate azalt: 0.9 → 0.7 (daha agresif update)

### 2. "Training unstable (loss spikes)"

**Belirti:** Loss ani artışlar gösteriyor
**Çözüm:**
- Gradient clipping: `max_grad_norm: 1.0` → `0.5`
- Hebbian learning rate azalt: 0.1 → 0.05
- Warmup steps artır: 1000 → 2000

### 3. "OOM during training"

**Çözüm:**
- Batch size azalt: 8 → 4
- Gradient accumulation artır: 16 → 32
- Gradient checkpointing aktif et

---

## Sonraki Adımlar (Phase 3)

Phase 2 tamamlandıktan sonra:

1. **Neuro-Symbolic Integration:** Hebbian memory + symbolic reasoning
2. **Multi-Modal:** Vision encoder ekleme (CLIP-style)
3. **Reinforcement Learning:** RLHF ile alignment
4. **Tree-of-Thoughts:** System 2 reasoning engine

---

## Kaynaklar

- **Fast Weights:** [Schmidhuber, 1992](https://people.idsia.ch/~juergen/fastweights.html)
- **Hebbian Learning:** [Ba et al., 2016 - Using Fast Weights to Attend to the Recent Past](https://arxiv.org/abs/1610.06258)
- **Mamba Architecture:** [Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)
- **Turkish NLP:** [BERTurk](https://github.com/stefan-it/turkish-bert)

---

## Özet

**Phase 2 = Static Model → Adaptive Agent**

- ✅ Hebbian plasticity ile in-context learning
- ✅ Code + Math datasets ile reasoning
- ✅ 50K steps, ~500M tokens
- ✅ Beklenen süre: ~6-8 saat (Tesla T4)

**Başarı Kriteri:** Model, prompt içindeki pattern'leri öğrenip genelleştirebilmeli (few-shot >70% accuracy).
