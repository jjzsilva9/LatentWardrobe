#!/bin/bash
# Evaluation pass for the factorial7 batch -- split out from run_factorial7.sh
# on 2026-07-07 so generation (paint3d env, no network) and evaluation (eval
# env, loads DreamSim/DINOv2/open_clip and makes live HuggingFace Hub calls)
# never run in the same process. Run this any time after (some of)
# run_factorial7.sh's cells have finished generating -- safe to re-run:
# each cell is skipped if outputs/factorial7_20260706/<cell>/medieval_villagers/
# metrics.json already exists (see evaluate_crowd.py), and a cell with no
# renders yet just fails that one crowd-dir and continues to the next.
#
# Launch the same way as run_factorial7.sh (visible window):
#   Start-Process wsl.exe -ArgumentList '-u root bash -c "mkdir -p /mnt/u; mountpoint -q /mnt/u || mount -t drvfs U: /mnt/u; bash \"/mnt/u/My Documents/LatentWardrobe/scripts/run_factorial7_eval.sh\""'
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eval
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd "/mnt/u/My Documents/LatentWardrobe"

OUT=outputs/factorial7_20260706
PROMPT_SLUG=medieval_villagers  # must match generate_crowd_baseline.py's slugify(PROMPT)
CELLS=(1_random_white_plain 2_clip_white_plain 3_clip_pink_plain 4_clip_white_color \
       5_clip_pink_color 6_clip_white_llm 7_clip_pink_llm 8_clip_white_llm6 9_clip_pink_llm6)

for cell in "${CELLS[@]}"; do
  crowd_dir="$OUT/$cell/$PROMPT_SLUG"
  metrics="$crowd_dir/metrics.json"
  if [ -f "$metrics" ]; then
    echo "[run_factorial7_eval] cell=$cell metrics.json already exists, skipping ($metrics)" | tee -a "$OUT/eval.log"
    continue
  fi
  if [ ! -d "$crowd_dir" ]; then
    echo "[run_factorial7_eval] cell=$cell has no crowd dir yet ($crowd_dir) -- generation hasn't reached it, skipping" | tee -a "$OUT/eval.log"
    continue
  fi
  python -u scripts/evaluate_crowd.py --crowd-dir "$crowd_dir" 2>&1 | tee -a "$OUT/eval.log"
  rc=${PIPESTATUS[0]}
  echo "[run_factorial7_eval] cell=$cell evaluate_exit=$rc" | tee -a "$OUT/eval.log"
done

echo "FACTORIAL7 EVAL PASS COMPLETE" | tee -a "$OUT/eval.log"
