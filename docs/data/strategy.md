# Data Strategy and Processing Protocol (V 1.0)

**Project:** AETHER-1
**Focus:** Curriculum Learning & Data Pipeline

---

## 1. Data Pipeline Strategy
Quality over Quantity. Since we are training a "Reasoning" model, we need high-signal, low-noise data.

### Pipeline Steps
1.  **Ingestion:** Raw data collection (Text, Code, Math).
2.  **Cleaning:** Deduplication, PII removal, and formatting.
3.  **Tokenization:** Converting text to inputs for the Mamba/Transformer model.
4.  **Embedding:** Generating vector representations for the Memory module.

---

## 2. Curriculum Learning Phases
We will train the model in distinct phases, mimicking human development.

### Phase 1: Foundation (The Infant)
*   **Data:** `TinyStories` dataset.
*   **Duration:** 20% of total training.
*   **Goal:** Minimize loss, learn basic grammar and sentence structure.
*   **Mechanism:** Backbone training only (Mamba), Plastic Memory disabled.

### Phase 2: Structural Thinking (The Child)
*   **Data:** `Python Code` + `Math Datasets`.
*   **Duration:** 40% of total training.
*   **Goal:** Learn syntax, logic, and step-by-step reasoning.
*   **Special Config:** Higher weight on "Next State Prediction" loss for code blocks.

### Phase 3: Hybrid Integration (The Teenager)
*   **Data:** Mixed dataset (Stories + Code + Wiki).
*   **Duration:** 30% of total training.
*   **Innovation:** **Plastic Memory (Fast Weights) Layers Activated.**
*   **Goal:** Adaptation to changing contexts (switching between code and narrative).

### Phase 4: Alignment & System 2 (The Adult)
*   **Data:** Synthetic "Hard Logic" questions and `<think>` tagged data.
*   **Duration:** 10% of total training.
*   **Goal:** Trigger the `reasoning` module before answering. Penalize hallucination heavily.

---

## 3. Test & Quality Control Plan
**Goal:** Measure "Intelligence," not just memorization.

### 3.1. Automated Metrics
*   **Perplexity (PPL):** Surprise factor in prediction (Lower is better).
*   **Validation Loss:** Error rate on unseen data.

### 3.2. AGI Reference Tests
*   **ARC-Easy (Abstraction and Reasoning Corpus):** Visual/Logical pattern recognition. The gold standard for AGI.
*   **Hellaswag:** Common sense completion.
*   **Needle In A Haystack:** Long context retrieval test. Can the model find specific info ("Password: Blue") hidden in 100 pages of text?
