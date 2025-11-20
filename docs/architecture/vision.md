# Project Vision Document (V 1.0)

**Project Codename:** AETHER-1 (Autonomous Episodic Thinking & Heuristic Reasoning)
**Date:** November 20, 2025
**Status:** Inception
**Goal:** AGI Fundamental Architecture Research

---

## 1. Executive Summary
**AETHER-1** is an experimental artificial intelligence architecture designed to address the fundamental limitations of current Large Language Models (LLMs), specifically "static weights," "hallucination," and "lack of reasoning."

This project proposes a hybrid structure that focuses on **Next State Prediction** rather than just *Next Token Prediction*, and incorporates **Plasticity** to allow for continued learning after training. Our goal is not to build a commercial chatbot, but to create a **"Cognitive Core"** that mimics the working principles of biological intelligence.

---

## 2. The Problem
Current State-of-the-Art (SOTA) Transformer architectures (GPT-4, Claude, Llama) face three main bottlenecks:

1.  **Static Memory:** Learning stops the moment training ends. The model does not remember anything outside its context window.
2.  **System 1 Thinking:** Models do not "think"; they only perform reflexive statistical matching. This leads to errors in complex logic questions.
3.  **Lack of World Model:** They process text but do not simulate the physical or causal rules of the world.

---

## 3. The Solution: AETHER-1 Architecture
We are moving away from the standard Transformer architecture.

### 3.1. Core Philosophy
*   **From Static to Plastic:** The model's weights must be updateable during inference (Fast Weights).
*   **From Token to State:** Instead of predicting the next word, it should predict the next "mental state."
*   **From Chat to Agent:** It should not just speak, but act and plan.

### 3.2. Technical Stack (Draft)
*   **Backbone:** Mamba (SSM - State Space Models) or Hybrid (Attention + SSM). *Reason: Infinite context and low inference cost.*
*   **Memory:** Vector Database (ChromaDB / Faiss) + Ephemeral Short-Term Memory.
*   **Optimizer:** Local SGD (to update weights during runtime).

---

## 4. Roadmap

### Phase 1: The Seed (Current)
*   Literature review (DeepMind, OpenAI, LeCun papers).
*   Setting up the "Small Scale" prototype environment.
*   Testing the Mamba architecture.

### Phase 2: The Toddler
*   Training with simple logic and causal datasets (TinyStories, ARC).
*   First tests of the "Plasticity" module.

### Phase 3: The Student
*   Integration of coding and math capabilities.
*   Recursive self-improvement tests.
