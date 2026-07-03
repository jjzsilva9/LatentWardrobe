#!/bin/bash
# Overnight batch 2026-07-03: baseline vs Method 1a (CLIP mesh sampling).
# 3 themes x 2 arms x N=30 = 180 members, interleaved theme-by-theme so a
# partial night still yields complete baseline-vs-clip pairs per theme.
# Expected ~20-24h at ~7-8 min/member: --resume makes spillover into the next
# day safe (relaunch the same command; finished members are skipped).
#
# Launch from Windows PowerShell with a VISIBLE console window (full live log,
# also tee'd to run.log):
#   Start-Process wsl.exe -ArgumentList '-u','root','-e','bash','/mnt/c/Users/silvajo/LatentWardrobe/scripts/run_overnight.sh'
#
# Closing that window kills the run. Do NOT launch via `wsl -e bash -c
# "nohup ... &"` -- WSL2 kills the VM when the launching wsl.exe exits.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate paint3d
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd /mnt/c/Users/silvajo/LatentWardrobe

OUT=outputs/overnight_20260703
mkdir -p "$OUT"
N=30
TAU=0.02

run () {  # run <arm-outdir> <prompt> [extra driver args...]
  local arm="$1"; local prompt="$2"; shift 2
  python -u scripts/generate_crowd_baseline.py \
    --prompt "$prompt" --num-members "$N" \
    --outdir "$OUT/$arm" --resume "$@" 2>&1 | tee -a "$OUT/run.log"
}

for prompt in 'medieval villagers' 'business casual office workers' 'aliens on the moon'; do
  run baseline     "$prompt"
  run clip_tau002  "$prompt" --mesh-sampling clip --tau "$TAU"
done

echo "OVERNIGHT BATCH COMPLETE" | tee -a "$OUT/run.log"
