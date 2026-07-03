#!/bin/bash
# Overnight baseline batch. Launch from Windows PowerShell with a VISIBLE
# console window streaming the full, unfiltered log (and tee'd to run.log):
#
#   Start-Process wsl.exe -ArgumentList '-u','root','-e','bash','/mnt/c/Users/silvajo/LatentWardrobe/scripts/run_overnight.sh'
#
# - The console window shows everything live; run.log keeps the permanent copy.
# - CLOSING that window kills the run (resume with the same command; the
#   driver's --resume skips finished members).
# - Do NOT launch with `wsl -e bash -c "nohup ... &"` -- WSL2 kills the whole
#   VM ~8s after the launching wsl.exe exits; nohup does not survive that.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate paint3d
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd /mnt/c/Users/silvajo/LatentWardrobe
python -u scripts/generate_crowd_baseline.py \
  --prompt 'superhero costume' \
  --prompt 'aliens on the moon' \
  --prompt 'medieval villagers' \
  --prompt 'business casual office workers' \
  --num-members 25 \
  --outdir outputs/overnight_20260702 \
  --resume 2>&1 | tee -a outputs/overnight_20260702/run.log
