# AETHER-1 Development Roadmap

This roadmap outlines the development phases for the AETHER-1 cognitive architecture, moving from a basic language model to a reasoning agent with plastic memory.

## Phase 1: Foundation (The Infant) - [CURRENT]
**Goal:** Establish the Mamba backbone and validate the training pipeline.
*   [x] **Infrastructure:** Project setup, directory structure, config management.
*   [x] **Data:** TinyStories dataset integration, GPT-2 Tokenizer.
*   [x] **Model:** Mamba-SSM backbone implementation (`MambaLLM`).
*   [x] **Training:** Basic training loop with WandB logging.
*   [ ] **Validation:** Proof-of-Concept training run (User executing on Cloud).

## Phase 2: Structural Thinking (The Child)
**Goal:** Teach the model syntax, logic, and structured thinking.
*   **Data Expansion:**
    *   Integrate Python Code datasets (e.g., The Stack, MBPP).
    *   Integrate Math datasets (e.g., GSM8K, OpenWebMath).
*   **Architecture Refinement:**
    *   Optimize Mamba config for code/logic (longer context).
    *   Implement "Next State Prediction" loss weighting (focus on logic tokens).
*   **Training:**
    *   Curriculum learning: Shift from stories to code/math.

## Phase 3: Plasticity & Memory (The Teenager)
**Goal:** Enable dynamic learning (Fast Weights) to adapt to context without retraining.
*   **Research & Design:**
    *   Design Hebbian Learning layer ($A_{fast}$ matrix update rule).
*   **Implementation:**
    *   Create `PlasticMambaBlock` (Standard Mamba + Fast Weights).
    *   Implement "Associative Recall" tasks to test memory.
*   **Training:**
    *   Train on mixed datasets (Stories + Code + Wiki).
    *   **Objective:** Minimize loss on "Needle in a Haystack" type tasks within the context.

## Phase 4: System 2 Reasoning (The Adult)
**Goal:** Implement the "Tree-of-Thoughts" reasoning engine for complex problem solving.
*   **Data Engineering:**
    *   Generate synthetic data with `<think>` tags (Chain-of-Thought traces).
*   **Architecture:**
    *   Implement `ReasoningEngine` (ToT Search).
    *   Implement `Evaluator` (Value function for thought paths).
*   **Inference:**
    *   Develop `generate_with_reasoning()` method.
    *   Implement dynamic compute allocation (spend more time on hard problems).

## Phase 5: Alignment & AGI Testing
**Goal:** Verify "Intelligence" against AGI benchmarks.
*   **Benchmarks:**
    *   ARC-Easy & ARC-Challenge (Visual/Abstract reasoning).
    *   Hellaswag (Common sense).
*   **Fine-tuning:**
    *   RLHF or DPO (Direct Preference Optimization) for safety and instruction following.
