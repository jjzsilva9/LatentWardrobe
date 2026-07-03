#!/bin/bash
# Over-WEEKEND batch (launched Fri 2026-07-03 evening): the dissertation's
# first real method comparison. Method 1 (baseline, uniform mesh sampling) vs
# Method 2 (CLIP mesh sampling, tau=0.02), 3 themes x N=30 x 2 arms = 180
# members at ~24 min/member on the A4000 => ~72h, i.e. Fri evening -> Mon.
# Interleaved theme-by-theme so a partial weekend still yields complete
# baseline-vs-clip pairs per theme; --resume makes stopping/relaunching safe
# at any point (finished members are skipped).
#
# Launch from Windows PowerShell with a VISIBLE console window (closing the
# window kills the run; do NOT use `nohup ... &` -- WSL2 tears down the VM
# when the launching wsl.exe exits, see outputs/overnight_20260702/README.md):
#
#   Start-Process wsl.exe -ArgumentList '-u root bash -c "mkdir -p /mnt/u; mountpoint -q /mnt/u || mount -t drvfs U: /mnt/u; bash \"/mnt/u/My Documents/LatentWardrobe/scripts/run_overnight.sh\""'
#
# NOTE: no `set -u` here -- conda's activate scripts reference unset vars and
# would kill the shell before anything is logged (learned 2026-07-03).
source /root/miniconda3/etc/profile.d/conda.sh
conda activate paint3d
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd "/mnt/u/My Documents/LatentWardrobe"

OUT=outputs/weekend_20260703
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

echo "WEEKEND BATCH COMPLETE" | tee -a "$OUT/run.log"
