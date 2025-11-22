## Developer: inkbytefo
## Modified: 2025-11-22

# AETHER Phase 3: Neuro-Symbolic Reasoning & Knowledge Integration

## Genel Bakış

**Amaç:** Phase 2'deki adaptif hafıza sistemini, **sembolik reasoning** ve **bilgi grafı entegrasyonu** ile birleştirerek hibrit bir zeka sistemi oluşturmak.

**Metafor:** Phase 2 = Çocuk (öğrenme), Phase 3 = Genç (mantıksal düşünme, bilgi bağlama)

---

## Teorik Temel: Neuro-Symbolic AI

### Dual-Process Theory

**System 1 (Neural):** Hızlı, sezgisel, pattern-based
- Mamba + Hebbian → Language understanding, pattern recognition

**System 2 (Symbolic):** Yavaş, mantıksal, rule-based
- Knowledge Graph + Logic Engine → Reasoning, verification

### Hibrit Mimari Avantajları

| Özellik | Pure Neural | Pure Symbolic | Hibrit |
|---------|-------------|---------------|--------|
| Generalization | ✅ | ❌ | ✅ |
| Interpretability | ❌ | ✅ | ✅ |
| Reasoning | ⚠️ | ✅ | ✅ |
| Learning | ✅ | ❌ | ✅ |

---

## Mimari Bileşenler

### 1. Knowledge Graph (Bilgi Grafı)

**Yapı:**
```
(Entity) --[Relation]--> (Entity)

Örnek:
(Einstein) --[doğum_tarihi]--> (1879)
(Einstein) --[meslek]--> (Fizikçi)
(Fizikçi) --[alt_kategori]--> (Bilim_İnsanı)
```

**Kaynak:**
- Wikidata (Turkish subset)
- ConceptNet (common sense)
- Custom domain knowledge (Math, Code)

**Storage:** Neo4j veya RDF triple store

### 2. Symbolic Reasoner

**Bileşenler:**
- **Query Planner:** Natural language → SPARQL/Cypher
- **Inference Engine:** Rule-based deduction (Prolog-style)
- **Verifier:** Neural output'u symbolic rules ile doğrulama

**Örnek Reasoning Chain:**
```
Query: "Einstein'ın doğduğu yüzyıl?"

1. Neural: "Einstein" → Entity extraction
2. KG Query: (Einstein) --[doğum_tarihi]--> ?year
3. Result: 1879
4. Symbolic: 1879 ∈ [1800, 1900) → 19. yüzyıl
5. Output: "19. yüzyıl"
```

### 3. Neural-Symbolic Bridge

**Mekanizma:**
```python
class NeuroSymbolicLayer(nn.Module):
    def forward(self, neural_embedding, kg_context):
        # Neural embedding: (batch, seq, d_model)
        # KG context: Retrieved entities + relations
        
        # 1. Entity linking
        entities = self.entity_linker(neural_embedding)
        
        # 2. KG retrieval
        kg_facts = self.kg_query(entities)
        
        # 3. Fact encoding
        fact_embeddings = self.fact_encoder(kg_facts)
        
        # 4. Fusion
        output = self.fusion(neural_embedding, fact_embeddings)
        
        return output
```

---

## Eğitim Stratejisi

### Data Composition

| Dataset | Oran | Amaç |
|---------|------|------|
| Wikipedia + KG | 30% | Factual knowledge |
| Code + Docstrings | 25% | Logical reasoning |
| Math + Proofs | 20% | Symbolic manipulation |
| QA Pairs (SQuAD-TR) | 15% | Question answering |
| Common Sense (COPA) | 10% | Implicit reasoning |

**Total Tokens:** ~1B (Phase 2'den 2x artış)

### Training Stages

**Stage 1 (Steps 0-20K): Knowledge Grounding**
- Phase 2 checkpoint'ten başla
- KG embeddings freeze, neural model train
- Loss: LM loss + Entity linking loss

**Stage 2 (Steps 20K-60K): Joint Training**
- KG embeddings unfreeze
- Multi-task learning:
  - Language modeling (60%)
  - QA (20%)
  - Fact verification (10%)
  - Logical inference (10%)

**Stage 3 (Steps 60K-80K): Reasoning Refinement**
- Synthetic reasoning chains (CoT-style)
- Adversarial examples (hallucination detection)
- Symbolic verifier feedback loop

### Hyperparameters

```yaml
model:
  d_model: 768  # 512'den artırıldı
  n_layer: 16   # 12'den artırıldı
  vocab_size: 50257
  use_plasticity: true
  use_kg_integration: true  # ← YENİ!
  kg_cfg:
    entity_dim: 256
    relation_dim: 128
    num_hops: 2  # KG traversal depth

training:
  batch_size: 4  # KG retrieval overhead nedeniyle azaltıldı
  learning_rate: 0.0002
  max_steps: 80000
  gradient_accumulation_steps: 32  # Effective batch = 128
```

---

## Beklenen Yetenekler

### 1. Factual Question Answering

**Örnek:**
```
Q: "Türkiye'nin başkenti neresidir?"
A: "Ankara"

Reasoning:
1. Entity: "Türkiye" → Wikidata Q43
2. KG Query: (Q43) --[başkent]--> ?
3. Result: Ankara (Q3640)
```

### 2. Multi-Hop Reasoning

**Örnek:**
```
Q: "Einstein'ın doğduğu ülkenin başkenti?"
A: "Berlin" (1879'da Almanya İmparatorluğu)

Reasoning:
1. (Einstein) --[doğum_yeri]--> (Ulm)
2. (Ulm) --[ülke]--> (Almanya)
3. (Almanya) --[başkent]--> (Berlin)
```

### 3. Logical Inference

**Örnek:**
```
Fact 1: "Tüm insanlar ölümlüdür"
Fact 2: "Sokrates bir insandır"
Query: "Sokrates ölümlü müdür?"
A: "Evet"

Reasoning (Syllogism):
∀x (İnsan(x) → Ölümlü(x))
İnsan(Sokrates)
∴ Ölümlü(Sokrates)
```

### 4. Hallucination Detection

**Örnek:**
```
Model Output: "Einstein 1955'te Mars'ta öldü"

Verifier:
1. KG Check: (Einstein) --[ölüm_yeri]--> (Princeton)
2. Conflict detected: Mars ≠ Princeton
3. Flag: HALLUCINATION
4. Correction: "Einstein 1955'te Princeton'da öldü"
```

---

## Implementation Checklist

### Hazırlık

- [ ] Phase 2 checkpoint kaydedildi (`aether_phase2.pt`)
- [ ] Knowledge Graph setup (Neo4j / RDFLib)
- [ ] Wikidata Turkish subset download (~500MB)
- [ ] Entity linking model (mGENRE veya custom)

### Eğitim

- [ ] Stage 1: Knowledge grounding (20K steps)
- [ ] Stage 2: Joint training (40K steps)
- [ ] Stage 3: Reasoning refinement (20K steps)
- [ ] Checkpoint her 10K step'te kaydet

### Validation

- [ ] QA accuracy (SQuAD-TR test set)
- [ ] Multi-hop reasoning (HotpotQA-TR)
- [ ] Fact verification (FEVER-TR)
- [ ] Hallucination rate (<5% target)

---

## Teknik Detaylar

### 1. Knowledge Graph Integration

**Retrieval Pipeline:**
```python
def kg_retrieve(entity, num_hops=2):
    # 1. Entity linking
    entity_id = entity_linker.link(entity)
    
    # 2. Subgraph extraction
    query = f"""
    MATCH (e:Entity {{id: '{entity_id}'}})-[r*1..{num_hops}]-(n)
    RETURN e, r, n
    """
    subgraph = neo4j.run(query)
    
    # 3. Embedding
    facts = [encode_triple(s, r, o) for s, r, o in subgraph]
    return facts
```

### 2. Multi-Task Loss

```python
loss_total = (
    0.6 * loss_lm +        # Language modeling
    0.2 * loss_qa +        # Question answering
    0.1 * loss_verify +    # Fact verification
    0.1 * loss_inference   # Logical inference
)
```

### 3. Symbolic Verifier

```python
class SymbolicVerifier:
    def verify(self, claim, kg):
        # 1. Parse claim → (subject, predicate, object)
        triple = self.parser.parse(claim)
        
        # 2. KG lookup
        kg_fact = kg.query(triple.subject, triple.predicate)
        
        # 3. Compare
        if kg_fact == triple.object:
            return True, 1.0  # Verified
        elif kg_fact is None:
            return None, 0.5  # Unknown
        else:
            return False, 0.0  # Contradiction
```

---

## Beklenen Metrikler

| Metric | Phase 2 (Final) | Phase 3 (Target) |
|--------|-----------------|------------------|
| Train Loss | ~1.8 | ~1.3 |
| QA Accuracy | ~30% | >70% |
| Multi-Hop Acc | N/A | >50% |
| Hallucination Rate | ~20% | <5% |
| Fact Verification | N/A | >85% |

---

## Sorun Giderme

### 1. "KG retrieval too slow"

**Çözüm:**
- KG embedding cache (Redis)
- Batch retrieval
- Approximate nearest neighbor (FAISS)

### 2. "Entity linking errors"

**Çözüm:**
- Fine-tune entity linker on Turkish data
- Ensemble: mGENRE + fuzzy matching
- Confidence threshold: >0.8

### 3. "Symbolic-neural mismatch"

**Belirti:** Model KG facts'i ignore ediyor
**Çözüm:**
- KG attention weight artır
- Fact encoder learning rate 2x
- Contrastive loss ekle (correct vs. incorrect facts)

---

## Kaynaklar

- **Neuro-Symbolic AI:** [Garcez et al., 2019](https://arxiv.org/abs/1905.12389)
- **Knowledge Graphs:** [Wikidata](https://www.wikidata.org/)
- **Entity Linking:** [mGENRE](https://github.com/facebookresearch/GENRE)
- **Logical Reasoning:** [LogiQA](https://github.com/lgw863/LogiQA-dataset)

---

## Özet

**Phase 3 = Adaptive Agent → Reasoning Agent**

- ✅ Knowledge Graph integration (Wikidata)
- ✅ Symbolic reasoning (logic engine)
- ✅ Hallucination detection (verifier)
- ✅ 80K steps, ~1B tokens
- ✅ Beklenen süre: ~10-12 saat (Tesla T4)

**Başarı Kriteri:** QA accuracy >70%, hallucination rate <5%
