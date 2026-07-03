#!/bin/bash
# Daytime shakedown batch (2026-07-03): small varied run to surface quality
# issues (low-poly faceting, fullbody garments, prompt phrasing) before
# committing tonight's big batch. Launch from PowerShell (VISIBLE window):
#   Start-Process wsl.exe -ArgumentList '-u','root','-e','bash','/mnt/c/Users/silvajo/LatentWardrobe/scripts/run_shakedown.sh'
source /root/miniconda3/etc/profile.d/conda.sh
conda activate paint3d
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd /mnt/c/Users/silvajo/LatentWardrobe
mkdir -p outputs/shakedown_20260703
python -u scripts/generate_crowd_baseline.py \
  --prompt 'superhero costume' \
  --prompt 'aliens on the moon' \
  --prompt 'medieval villagers' \
  --prompt 'business casual office workers' \
  --num-members 3 \
  --outdir outputs/shakedown_20260703 \
  --resume 2>&1 | tee -a outputs/shakedown_20260703/run.log
