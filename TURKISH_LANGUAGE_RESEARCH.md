# Turkish Language & AI Research Report

**Date:** November 21, 2025
**Project:** AETHER-1
**Topic:** Turkish Language Integration & Retraining Strategy

## 1. Turkish Language Characteristics & AI Implications

### 1.1. Agglutinative Morphology
Turkish is an agglutinative language, meaning words are formed by stringing together morphemes (root + suffixes).
*   **Example:** "Avrupalılaştıramadıklarımızdanmışsınızcasına" (As if you were one of those whom we could not make European).
*   **AI Challenge:** Standard tokenizers (BPE/WordPiece) trained on English often shatter Turkish words into many meaningless fragments (e.g., "Av", "##ru", "##pa", ...). This increases the sequence length significantly (2.5x vs English) and dilutes semantic meaning.
*   **AI Opportunity:** A morphology-aware tokenizer or a character-level/byte-level model can capture the regular structure of Turkish better than word-level models.

### 1.2. Vowel Harmony & Syntax
*   **Vowel Harmony:** Suffixes change vowels to match the root (e.g., "ev-ler", "masa-lar"). This is a strict rule that models must learn.
*   **SOV Word Order:** Subject-Object-Verb is the standard, but word order is flexible (scrambling) due to case marking. This requires the model to rely on case markers rather than position for role assignment.

---

## 2. Mamba (SSM) Suitability for Turkish

**Verdict: Highly Suitable**

1.  **Linear Scaling vs. Sequence Length:**
    *   Since Turkish tokenization results in longer sequences (more tokens per sentence), Transformers ($O(L^2)$) suffer from quadratic cost.
    *   Mamba ($O(L)$) is ideal for this. It can handle the increased token count of Turkish without the massive compute penalty of Attention.

2.  **Morphological Modeling:**
    *   SSMs are continuous-time systems discretized. They are theoretically better at modeling the "state" evolution of a word as suffixes are added, compared to the discrete "bag-of-words" attention mechanism.

3.  **Token-Free Potential:**
    *   MambaByte (byte-level Mamba) is a strong candidate for Turkish, as it bypasses the tokenizer entirely, learning morphology directly from characters/bytes.

---

## 3. Datasets for Training

We have identified high-quality datasets to train a Turkish-first or Multilingual model:

*   **General Corpora:**
    *   **TS Corpus:** 1.3B tokens (Newspapers, Social Media).
    *   **OSCAR (Turkish):** Massive web crawl data (cleaned).
    *   **C4 (Turkish subset):** Common Crawl.
*   **Instruction Tuning:**
    *   `merve/turkish_instructions`
    *   `Bactrian-X` (Turkish subset)
*   **Reasoning (Phase 2/4):**
    *   `GSM8K-TR` (Translated Math)
    *   `OpenOrca-TR` (Translated Reasoning)

---

## 4. Integration Plan for AETHER

### 4.1. Tokenizer Strategy
*   **Option A (Standard):** Train a new BPE tokenizer *specifically* on Turkish + English data (50k vocab). Do not use the default GPT-2 tokenizer.
*   **Option B (Experimental):** Use a Byte-Level tokenizer (UTF-8 bytes) to fully leverage Mamba's strengths on morphology.

### 4.2. Training Phases (Revised)
*   **Phase 1 (Foundation):** Train on **OSCAR (TR) + TinyStories (EN)**. Goal: Learn Turkish grammar and basic narrative structure.
*   **Phase 2 (Structure):** Train on **Python Code + GSM8K-TR**. Goal: Logic and syntax.
*   **Phase 3 (Plasticity):** Enable Hebbian layers. Train on mixed context.
*   **Phase 4 (Reasoning):** Finetune on **Turkish Instructions + Chain-of-Thought**.

### 4.3. Recommendation
We should proceed with **Option A (Custom BPE)** for stability, but keep the architecture ready for MambaByte experiments later. We will retrain the model from scratch using a mixed English-Turkish dataset to ensure it retains coding ability (English-heavy) while mastering Turkish.
