# AETHER-1: Autonomous Episodic Thinking & Heuristic Reasoning

**Status:** Phase 2 (Plasticity) - In Progress 🔄

AETHER-1 is an experimental cognitive architecture aiming to move beyond static LLMs towards agents with plastic memory and reasoning capabilities. It combines the efficiency of Mamba (SSM) with Hebbian Learning (Fast Weights) for dynamic in-context adaptation.

## 🚀 Current Status
- **Phase 1 (Foundation):** ✅ Completed (Turkish + English Base)
- **Phase 2 (Plasticity):** 🔄 In Progress (Hebbian Learning Training)
- **Phase 3 (Reasoning):** ⏳ Planned
- **Phase 4 (Agentic):** ⏳ Planned

## Documentation
Please refer to the `docs/` directory and root markdown files for detailed documentation:
- [Phase 1 Guide](PHASE1_TRAINING_GUIDE.md)
- [Phase 2 Roadmap](PHASE2_ROADMAP.md)
- [Phase 2 Guide](PHASE2_TRAINING_GUIDE.md)

## Setup

1.  **Clone and Setup:**
    ```bash
    git clone https://github.com/inkbytefo/AETHER-1.git
    cd AETHER-1
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Tests:**
    ```bash
    pytest
    ```
