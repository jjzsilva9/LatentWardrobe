#!/usr/bin/env bash
# Creates the "eval" conda environment inside WSL2 Ubuntu, used only by
# scripts/evaluate_crowd.py. Kept separate from the "paint3d" env
# (Python 3.8.5 / torch 1.12.1+cu113 / kaolin / pytorch3d) so that upgrading
# eval-only packages (torch.hub DINOv2, dreamsim, open_clip) can never break
# the generation pipeline.
#
# Run from inside WSL, from anywhere in the repo:
#   bash .claude/setup/install_eval_env.sh
#
# Prerequisites: same WSL2 + GPU passthrough as install_paint3d_env.sh
# (see .claude/setup/README.md). This script does not touch that env.
set -euo pipefail

CONDA_DIR="$HOME/miniconda3"
source "$CONDA_DIR/etc/profile.d/conda.sh"

if conda env list | grep -q '^eval '; then
  echo "eval env already exists, skipping create (remove with 'conda env remove -n eval' to rebuild)"
else
  conda create -y -n eval python=3.10
fi
conda activate eval

# Recent torch/torchvision (cu121 wheels) -- independent of Paint3D's pinned
# 1.12.1+cu113. The host driver (581.15, CUDA 13.0-capable per README.md) is
# backward compatible with cu121.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# DINOv2 is loaded via torch.hub (facebookresearch/dinov2) at eval-script
# runtime, not pip-installed -- no extra package needed here beyond torch/
# torchvision. First run of evaluate_crowd.py downloads the hub repo + weights
# and caches them under ~/.cache/torch/hub.

# CLIPScore coherence + (legacy) CLIP diversity metric.
pip install open_clip_torch

# Perceptual distances.
pip install lpips dreamsim

# Frame extraction / color histograms / low-res pixel L2.
pip install opencv-python-headless pillow numpy scipy

# scripts/plot_crowd_comparison.py.
pip install matplotlib

echo ""
echo "Done. Activate with: conda activate eval"
echo "Run evaluate_crowd.py from this env, e.g.:"
echo "  conda activate eval && python scripts/evaluate_crowd.py --crowd-dir outputs/<batch>/<prompt_slug>"
