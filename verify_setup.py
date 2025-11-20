import sys
import os

def check_setup():
    print("Checking AETHER-1 Setup...")
    
    # Check Python version
    print(f"Python: {sys.version.split()[0]}")
    
    # Check Directories
    required_dirs = ['src/models', 'src/data', 'src/utils', 'tests', 'configs', 'notebooks']
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
    else:
        print("✅ Directory structure verified.")
        
    # Check Dependencies (Try import)
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except ImportError:
        print("⚠️  PyTorch not installed or not found in environment.")
        
    try:
        import mamba_ssm
        print("✅ Mamba-SSM installed.")
    except ImportError:
        print("⚠️  Mamba-SSM not installed.")
        
    print("\nSetup check complete.")

if __name__ == "__main__":
    check_setup()
