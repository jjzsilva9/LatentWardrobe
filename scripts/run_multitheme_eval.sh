#!/bin/bash
# Evaluation pass for the multitheme batch -- same structure as
# run_factorial7_eval.sh (see its header), but looping cells x prompt slugs.
# Safe to re-run: skips any crowd dir that already has metrics.json.
#
# Launch (visible window, from Windows PowerShell):
#   Start-Process wsl.exe -ArgumentList '-u root bash -c "mkdir -p /mnt/u; mountpoint -q /mnt/u || mount -t drvfs U: /mnt/u; bash \"/mnt/u/My Documents/LatentWardrobe/scripts/run_multitheme_eval.sh\""'
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eval
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd "/mnt/u/My Documents/LatentWardrobe"

OUT=outputs/multitheme_20260710
SLUGS=(superheroes business_formalwear)  # must match generate_crowd_baseline.py's slugify()
CELLS=(1_random_white_plain 2_clip_white_plain 3_clip_pink_plain 4_clip_white_color \
       5_clip_pink_color 8_clip_white_llm6 9_clip_pink_llm6)

for cell in "${CELLS[@]}"; do
  for slug in "${SLUGS[@]}"; do
    crowd_dir="$OUT/$cell/$slug"
    metrics="$crowd_dir/metrics.json"
    if [ -f "$metrics" ]; then
      echo "[run_multitheme_eval] cell=$cell slug=$slug metrics.json already exists, skipping" | tee -a "$OUT/eval.log"
      continue
    fi
    if [ ! -d "$crowd_dir" ]; then
      echo "[run_multitheme_eval] cell=$cell slug=$slug has no crowd dir yet -- skipping" | tee -a "$OUT/eval.log"
      continue
    fi
    python -u scripts/evaluate_crowd.py --crowd-dir "$crowd_dir" --device cpu 2>&1 | tee -a "$OUT/eval.log"
    rc=${PIPESTATUS[0]}
    echo "[run_multitheme_eval] cell=$cell slug=$slug evaluate_exit=$rc" | tee -a "$OUT/eval.log"
  done
done

echo "MULTITHEME EVAL PASS COMPLETE" | tee -a "$OUT/eval.log"
