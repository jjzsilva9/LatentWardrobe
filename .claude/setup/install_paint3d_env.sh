#!/usr/bin/env bash
# Recreates the Paint3D conda environment inside WSL2 Ubuntu.
#
# Run from inside WSL, from anywhere in the repo:
#   bash .claude/setup/install_paint3d_env.sh
#
# Prerequisites (not handled by this script):
#   - WSL2 with an Ubuntu distro installed, GPU/CUDA passthrough working.
#     Verify first with: wsl -e nvidia-smi  (should show the GPU, not error)
#   - See .claude/setup/README.md for context and troubleshooting.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PAINT3D_DIR="$REPO_ROOT/external/paint3d"
CONDA_DIR="$HOME/miniconda3"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

# 1. Miniconda
if [ ! -d "$CONDA_DIR" ]; then
  curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

# 2. Accept Anaconda default-channel ToS -- required by newer conda versions
#    before they'll solve against repo.anaconda.com/pkgs/main or /pkgs/r.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# 3. Create the env from Paint3D's own environment.yaml (python 3.8.5,
#    cudatoolkit 11.3, pytorch 1.12.1, plus its pip requirements).
if conda env list | grep -q '^paint3d '; then
  echo "paint3d env already exists, skipping create (remove with 'conda env remove -n paint3d' to rebuild)"
else
  conda env create -f "$PAINT3D_DIR/environment.yaml" -n paint3d
fi
conda activate paint3d

# 4. System packages missing from this minimal WSL Ubuntu image.
#    libgomp1: OpenMP runtime torch's compiled libs link against.
#    patchelf: used below to work around the execstack issue.
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq libgomp1 patchelf

# 5. Some WSL Ubuntu builds have hardened glibc/kernel that refuse to load
#    old (2022-era) shared libs requesting an executable stack. PyTorch
#    1.12.1's .so files are flagged that way as a harmless build artifact --
#    clearing the flag is safe and doesn't change behavior. This step is a
#    no-op (and harmless) if your distro doesn't have the issue.
find "$CONDA_DIR/envs/paint3d/lib/python3.8/site-packages/torch" -name '*.so*' \
  -exec patchelf --clear-execstack {} \; 2>/dev/null || true

# 6. kaolin -- version must match torch/cuda exactly.
pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu113.html

# 7. pytorch3d -- prebuilt wheel matched to py38_cu113_pyt1121.
#    Install fvcore/iopath from PyPI FIRST: the fbaipublicfiles link-only
#    index doesn't host them, so `pip install --no-index` for pytorch3d
#    fails to resolve them if skipped.
pip install fvcore iopath
pip install --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1121/download.html

# 8. huggingface_hub/diffusers version skew: environment.yaml pins
#    diffusers==0.25.0 but not huggingface_hub, so pip installs whatever is
#    newest -- which has since dropped `cached_download`, a name diffusers
#    0.25.0 still imports. Pin to a version from the same era.
pip install 'huggingface_hub==0.20.3'

# 9. Verify
python -c "
import torch, kaolin, pytorch3d
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
print('kaolin', kaolin.__version__)
print('pytorch3d', pytorch3d.__version__)
"

echo ""
echo "Done. Activate with: conda activate paint3d"
echo "Before running any pipeline_paint3d_*.py script, also export:"
echo "  export LD_LIBRARY_PATH=/usr/lib/wsl/lib:\$LD_LIBRARY_PATH"
echo "(see README.md for why -- cuDNN dlopens the unversioned libcuda.so, which WSL's ldconfig cache doesn't index)"
