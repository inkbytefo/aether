## Developer: inkbytefo
## Modified: 2025-11-22

# CUDA Toolkit Kurulum Kılavuzu (mamba-ssm için)

## Durum Tespiti
```bash
# GPU ve Driver Kontrolü
nvidia-smi
# ✅ CUDA Version: 12.8 (Driver desteği)
# ✅ GPU: Tesla T4 (15GB VRAM)

# NVCC Kontrolü (Conda Env İçinde)
nvcc --version
# ❌ Beklenen Hata: command not found
```

---

## Çözüm 1: CUDA Toolkit Kurulumu (Önerilen)

### Adım 1: CUDA Toolkit Kurulumu
```bash
conda activate aether
conda install -c nvidia cuda-toolkit=12.8 cuda-nvcc=12.8 -y
```

### Adım 2: Environment Variables (Kalıcı)
```bash
# ~/.bashrc dosyasına ekleyin
cat >> ~/.bashrc << 'EOF'

# CUDA Environment for Aether
if [[ "$CONDA_DEFAULT_ENV" == "aether" ]]; then
    export CUDA_HOME=$CONDA_PREFIX
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export CUDA_VISIBLE_DEVICES=0
fi
EOF

# Aktivasyon
source ~/.bashrc
conda deactivate && conda activate aether
```

### Adım 3: NVCC Doğrulama
```bash
nvcc --version
# Beklenen: Cuda compilation tools, release 12.8
```

### Adım 4: mamba-ssm Kurulumu
```bash
pip install mamba-ssm --no-build-isolation -v
```

**Flags Açıklaması:**
- `--no-build-isolation`: Pip'in kendi build env yerine conda env kullanır
- `-v`: Verbose output (hata durumunda debug için)

---

## Çözüm 2: Pre-compiled Wheel (Hızlı Alternatif)

Eğer derleme sorunları devam ederse:

```bash
# Causal-Conv1D bağımlılığı
pip install causal-conv1d>=1.4.0

# Mamba-SSM (pre-built)
pip install mamba-ssm --find-links https://github.com/state-spaces/mamba/releases
```

---

## Doğrulama

```bash
python -c "
import torch
import mamba_ssm
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'Mamba-SSM: {mamba_ssm.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

**Beklenen Çıktı:**
```
PyTorch: 2.9.1+cu128
CUDA Available: True
Mamba-SSM: 2.2.6
GPU: Tesla T4
```

---

## Sorun Giderme

### Hata: "CUDA mismatch"
```bash
# PyTorch CUDA version kontrolü
python -c "import torch; print(torch.version.cuda)"

# Toolkit version eşleştirme
conda install cuda-toolkit=$(python -c "import torch; print(torch.version.cuda)")
```

### Hata: "ninja: build stopped"
```bash
# Ninja build system kurulumu
conda install ninja -y
pip install mamba-ssm --no-build-isolation
```

### Hata: "causal_conv1d not found"
```bash
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

---

## Notlar

1. **Conda vs System CUDA:** Conda env içi kurulum → izolasyon + reproducibility
2. **Build Time:** İlk kurulum ~5-10 dakika sürebilir (CUDA kernels derleniyor)
3. **Memory:** Build sırasında ~2GB RAM kullanımı normal
4. **Brev.dev Persistence:** Instance geçiciyse, bu adımları `setup.sh` olarak kaydedin

---

## Hızlı Kurulum Scripti

```bash
#!/bin/bash
# setup_cuda.sh

set -e

echo "=== CUDA Toolkit Kurulumu ==="
conda install -c nvidia cuda-toolkit=12.8 cuda-nvcc=12.8 -y

echo "=== Environment Variables ==="
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "=== NVCC Doğrulama ==="
nvcc --version

echo "=== Mamba-SSM Kurulumu ==="
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation -v

echo "=== Test ==="
python -c "import mamba_ssm; print('✅ Mamba-SSM kuruldu!')"
```

**Kullanım:**
```bash
chmod +x setup_cuda.sh
./setup_cuda.sh
```
