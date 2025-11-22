## Developer: inkbytefo
## Modified: 2025-11-22

# AETHER Phase 4: Tree-of-Thoughts & System 2 Reasoning

## Genel Bakış

**Amaç:** Phase 3'teki bilgi tabanlı reasoning'i, **deliberate problem solving** ve **self-correction** yetenekleriyle genişletmek.

**Metafor:** Phase 3 = Genç (bilgi + mantık), Phase 4 = Yetişkin (stratejik düşünme, planlama)

---

## Teorik Temel: Tree-of-Thoughts (ToT)

### Kahneman's Dual Process Theory

**System 1:** Hızlı, otomatik, sezgisel
- Mamba + Hebbian → Pattern recognition

**System 2:** Yavaş, kontrollü, analitik
- ToT Search → Deliberate reasoning

### ToT vs. Chain-of-Thought (CoT)

| Özellik | CoT | ToT |
|---------|-----|-----|
| Yapı | Linear chain | Tree (branching) |
| Backtracking | ❌ | ✅ |
| Self-evaluation | ❌ | ✅ |
| Compute | O(n) | O(b^d) |

**Örnek:**

**CoT:**
```
Problem: 24 oyunu (4, 6, 8, 2) → 24
Thought: 4 + 6 = 10, 10 + 8 = 18, 18 + 2 = 20 ❌
(Stuck, no backtracking)
```

**ToT:**
```
Problem: 24 oyunu (4, 6, 8, 2) → 24

Root
├─ Branch 1: 4 + 6 = 10
│  ├─ 10 + 8 = 18 → 18 + 2 = 20 ❌ (dead end)
│  └─ 10 * 2 = 20 → 20 + 8 = 28 ❌ (dead end)
├─ Branch 2: 4 * 6 = 24 ✅ (solution found!)
└─ (prune other branches)
```

---

## Mimari Bileşenler

### 1. Thought Generator

**Görev:** Her node'da olası "thoughts" (düşünce adımları) üret

```python
class ThoughtGenerator(nn.Module):
    def generate_thoughts(self, state, num_candidates=5):
        # state: Current problem state
        # Returns: List of candidate next steps
        
        thoughts = []
        for _ in range(num_candidates):
            thought = self.model.generate(
                prompt=f"<state>{state}</state><think>",
                max_tokens=50,
                temperature=0.8  # Diversity
            )
            thoughts.append(thought)
        return thoughts
```

### 2. Evaluator (Value Function)

**Görev:** Her thought'ı değerlendir (얼마나 promising?)

```python
class ThoughtEvaluator(nn.Module):
    def evaluate(self, thought, goal):
        # Returns: Score ∈ [0, 1]
        
        # Method 1: Model-based (neural)
        score_neural = self.value_head(thought_embedding)
        
        # Method 2: Symbolic (rule-based)
        score_symbolic = self.symbolic_checker(thought, goal)
        
        # Hybrid
        return 0.7 * score_neural + 0.3 * score_symbolic
```

### 3. Search Algorithm

**Beam Search (Greedy):**
```python
def beam_search(problem, beam_width=3, max_depth=5):
    beam = [initial_state]
    
    for depth in range(max_depth):
        candidates = []
        for state in beam:
            thoughts = generator.generate_thoughts(state)
            for thought in thoughts:
                score = evaluator.evaluate(thought)
                candidates.append((thought, score))
        
        # Keep top-k
        beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Check for solution
        if any(is_solution(t) for t, _ in beam):
            return best_solution(beam)
    
    return beam[0]  # Best effort
```

**Monte Carlo Tree Search (MCTS):**
- Selection: UCB1 (exploration vs. exploitation)
- Expansion: Generate new thoughts
- Simulation: Rollout to terminal state
- Backpropagation: Update node values

---

## Eğitim Stratejisi

### Data Composition

| Dataset | Oran | Amaç |
|---------|------|------|
| Math Reasoning (GSM8K, MATH) | 35% | Multi-step problem solving |
| Code Debugging | 25% | Error detection + correction |
| Logic Puzzles (ARC, LSAT) | 20% | Abstract reasoning |
| Planning Tasks (ALFWorld) | 10% | Sequential decision making |
| Adversarial QA | 10% | Robustness |

**Total Tokens:** ~2B (Phase 3'ten 2x artış)

### Synthetic Data Generation

**Chain-of-Thought Traces:**
```python
def generate_cot_data(problem):
    # 1. Solve problem with ToT
    solution, tree = tot_solve(problem)
    
    # 2. Extract successful path
    path = extract_path(tree, solution)
    
    # 3. Format as CoT
    cot = format_cot(path)
    
    # 4. Create training example
    return {
        "input": problem,
        "output": f"<think>{cot}</think>{solution}"
    }
```

**Contrastive Examples:**
```python
# Positive: Correct reasoning
positive = {
    "problem": "2 + 2 = ?",
    "thought": "2 + 2 = 4",
    "label": 1.0
}

# Negative: Incorrect reasoning
negative = {
    "problem": "2 + 2 = ?",
    "thought": "2 + 2 = 5",
    "label": 0.0
}
```

### Training Stages

**Stage 1 (Steps 0-30K): Thought Generation**
- Phase 3 checkpoint'ten başla
- Train generator on CoT traces
- Loss: Cross-entropy (next token prediction)

**Stage 2 (Steps 30K-70K): Value Learning**
- Train evaluator on (thought, score) pairs
- Loss: MSE (predicted vs. true value)
- Multi-task: Generation + Evaluation

**Stage 3 (Steps 70K-100K): End-to-End RL**
- Reinforcement learning (PPO / DPO)
- Reward: Solution correctness + efficiency
- Self-play: Model generates own training data

### Hyperparameters

```yaml
model:
  d_model: 1024  # 768'den artırıldı
  n_layer: 20    # 16'dan artırıldı
  vocab_size: 50257
  use_plasticity: true
  use_kg_integration: true
  use_tot: true  # ← YENİ!
  tot_cfg:
    beam_width: 5
    max_depth: 8
    value_head_dim: 256

training:
  batch_size: 2  # ToT overhead nedeniyle azaltıldı
  learning_rate: 0.0001
  max_steps: 100000
  gradient_accumulation_steps: 64  # Effective batch = 128
  rl_cfg:
    algorithm: "PPO"
    gamma: 0.99
    lambda_gae: 0.95
```

---

## Beklenen Yetenekler

### 1. Multi-Step Math Reasoning

**Örnek:**
```
Problem: "Bir mağazada 3 elma 5 TL. 7 elma kaç TL?"

ToT Reasoning:
<think>
Step 1: 3 elma = 5 TL → 1 elma = 5/3 TL
Step 2: 7 elma = 7 * (5/3) TL
Step 3: 7 * 5 = 35, 35/3 = 11.67 TL
</think>
Answer: 11.67 TL
```

### 2. Code Debugging

**Örnek:**
```python
# Buggy code
def fibonacci(n):
    if n <= 1:
        return 1  # BUG: should be 'return n'
    return fibonacci(n-1) + fibonacci(n-2)

# ToT Debugging
<think>
Hypothesis 1: Base case wrong
  Test: fib(0) = ? Expected: 0, Got: 1 ✅ BUG FOUND
  Fix: return n
Hypothesis 2: Recursion wrong
  Test: fib(5) = ? Expected: 5, Got: 8 ❌ (after fix 1)
  No further bugs
</think>
Fixed code: return n (line 3)
```

### 3. Planning & Strategy

**Örnek:**
```
Goal: "Kahve yap"

ToT Plan:
<think>
Option 1: Kahve makinesi
  ├─ Check: Makine var mı? → Evet
  ├─ Check: Kahve var mı? → Evet
  └─ Steps: Su ekle → Kahve ekle → Başlat ✅
Option 2: Türk kahvesi
  ├─ Check: Cezve var mı? → Hayır ❌
  └─ (prune)
</think>
Plan: Kahve makinesini kullan
```

### 4. Self-Correction

**Örnek:**
```
Q: "Paris'in başkenti neresi?"
Initial: "Fransa" ❌

<think>
Wait, "başkent" ilişkisi ters.
Paris bir şehir, başkent olamaz.
Doğru soru: "Fransa'nın başkenti neresi?"
</think>
Corrected: "Soru hatalı. Paris zaten bir başkent (Fransa'nın)."
```

---

## Implementation Checklist

### Hazırlık

- [ ] Phase 3 checkpoint kaydedildi (`aether_phase3.pt`)
- [ ] ToT search engine implement
- [ ] Value head ekleme (model architecture)
- [ ] RL trainer setup (PPO / DPO)

### Eğitim

- [ ] Stage 1: CoT generation (30K steps)
- [ ] Stage 2: Value learning (40K steps)
- [ ] Stage 3: RL fine-tuning (30K steps)
- [ ] Checkpoint her 10K step'te kaydet

### Validation

- [ ] Math accuracy (GSM8K, MATH)
- [ ] Code debugging (HumanEval-Debug)
- [ ] Logic puzzles (ARC-Challenge)
- [ ] Planning tasks (ALFWorld)

---

## Teknik Detaylar

### 1. Value Head Architecture

```python
class MambaWithValueHead(nn.Module):
    def __init__(self, base_model, value_dim=256):
        self.base = base_model
        self.value_head = nn.Sequential(
            nn.Linear(base_model.d_model, value_dim),
            nn.ReLU(),
            nn.Linear(value_dim, 1),
            nn.Sigmoid()  # Output ∈ [0, 1]
        )
    
    def forward(self, x, return_value=False):
        hidden = self.base(x)
        logits = self.base.lm_head(hidden)
        
        if return_value:
            value = self.value_head(hidden[:, -1, :])  # Last token
            return logits, value
        return logits
```

### 2. PPO Training Loop

```python
def ppo_step(model, problem):
    # 1. Generate trajectory with ToT
    trajectory, rewards = tot_solve_with_rewards(problem)
    
    # 2. Compute advantages (GAE)
    advantages = compute_gae(rewards, values, gamma=0.99)
    
    # 3. PPO update
    for epoch in range(ppo_epochs):
        logprobs_new, values_new = model(trajectory)
        ratio = torch.exp(logprobs_new - logprobs_old)
        
        # Clipped objective
        loss_policy = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
        ).mean()
        
        loss_value = F.mse_loss(values_new, returns)
        loss = loss_policy + 0.5 * loss_value
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 3. Thought Pruning (Efficiency)

```python
def prune_thoughts(thoughts, threshold=0.3):
    # Remove low-value thoughts early
    scored = [(t, evaluator(t)) for t in thoughts]
    return [t for t, s in scored if s > threshold]
```

---

## Beklenen Metrikler

| Metric | Phase 3 (Final) | Phase 4 (Target) |
|--------|-----------------|------------------|
| GSM8K Accuracy | ~40% | >75% |
| MATH Accuracy | ~15% | >35% |
| HumanEval (Debug) | N/A | >60% |
| ARC-Challenge | ~30% | >55% |
| Planning Success | N/A | >70% |

---

## Sorun Giderme

### 1. "ToT search too slow"

**Çözüm:**
- Beam width azalt: 5 → 3
- Max depth azalt: 8 → 5
- Parallel beam search (multi-GPU)

### 2. "Value head not learning"

**Çözüm:**
- Separate learning rate (2x base LR)
- Pretrain on synthetic (correct/incorrect) pairs
- Auxiliary loss: Predict solution correctness

### 3. "RL training unstable"

**Çözüm:**
- Smaller PPO clip: 0.2 → 0.1
- Entropy bonus: Encourage exploration
- KL penalty: Prevent policy collapse

---

## Kaynaklar

- **Tree-of-Thoughts:** [Yao et al., 2023](https://arxiv.org/abs/2305.10601)
- **PPO:** [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
- **MCTS:** [Silver et al., 2016 - AlphaGo](https://www.nature.com/articles/nature16961)
- **Value Learning:** [OpenAI - Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)

---

## Özet

**Phase 4 = Reasoning Agent → Strategic Agent**

- ✅ Tree-of-Thoughts search (deliberate reasoning)
- ✅ Value function (thought evaluation)
- ✅ Self-correction (backtracking)
- ✅ RL fine-tuning (PPO)
- ✅ 100K steps, ~2B tokens
- ✅ Beklenen süre: ~15-18 saat (Tesla T4)

**Başarı Kriteri:** GSM8K >75%, MATH >35%, self-correction rate >60%
