#!/bin/bash
# Post-wipe shakedown (2026-07-03, wipe #2): small end-to-end rehearsal of the
# generate -> evaluate loop on the rebuilt environment. Baseline method only:
# 2 prompts x 2 members, then evaluate_crowd.py on each crowd.
# (Superseded same day by the over-weekend batch in run_overnight.sh once
# member_00 validated the rebuilt pipeline; kept for future env rehearsals.)
#
# The repo lives on the U: network share (survives lab-machine wipes). WSL
# does NOT automount network drives, so the launcher must mount U: first.
# Launch from Windows PowerShell with a VISIBLE window (closing the window
# kills the run; do NOT use `nohup ... &` -- WSL tears down the VM when the
# launching wsl.exe exits, see outputs/overnight_20260702/README.md):
#
#   Start-Process wsl.exe -ArgumentList '-u root bash -c "mkdir -p /mnt/u; mountpoint -q /mnt/u || mount -t drvfs U: /mnt/u; bash \"/mnt/u/My Documents/LatentWardrobe/scripts/run_shakedown.sh\""'
#
# NOTE: no `set -u` here -- conda's activate scripts reference unset vars and
# would kill the shell before anything is logged (learned 2026-07-03).
source /root/miniconda3/etc/profile.d/conda.sh
conda activate paint3d
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd "/mnt/u/My Documents/LatentWardrobe"

OUT=outputs/shakedown2_20260703
mkdir -p "$OUT"

python -u scripts/generate_crowd_baseline.py \
  --prompt 'medieval villagers' \
  --prompt 'aliens on the moon' \
  --num-members 2 \
  --outdir "$OUT" \
  --resume 2>&1 | tee -a "$OUT/run.log"

echo "GENERATION DONE -- evaluating" | tee -a "$OUT/run.log"

for slug in medieval_villagers aliens_on_the_moon; do
  python -u scripts/evaluate_crowd.py --crowd-dir "$OUT/$slug" \
    2>&1 | tee -a "$OUT/run.log"
done

echo "SHAKEDOWN COMPLETE" | tee -a "$OUT/run.log"
