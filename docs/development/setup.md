# Setup and Installation Guide

## Prerequisites
*   Python 3.10+
*   CUDA-compatible GPU (recommended for training)
*   Git

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/inkbytefo/AETHER-1.git
    cd AETHER-1
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration
*   Copy `.env.example` to `.env` and configure your API keys (if applicable).
*   Adjust `config.yaml` for model parameters.

## Running the Project
*   **Training:** `python train.py`
*   **Inference:** `python main.py`
