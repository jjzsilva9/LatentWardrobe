#!/bin/bash
# Evaluation pass for the crowd library (outputs/crowd_library_20260713) --
# same structure as run_multitheme_eval.sh, but discovers prompt dirs from
# disk per cell instead of a hardcoded SLUGS list, since the theme list here
# keeps growing (scripts/crowd_library_prompts.txt). Safe to re-run: skips
# any crowd dir that already has metrics.json. Runs on CPU (--device cpu) so
# it never contends with generation for the GPU -- see 2026-07-13 incident
# where a GPU-bound eval pass stalled ~47 min behind a concurrent Paint3D run.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eval
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd "/mnt/u/My Documents/LatentWardrobe"

OUT="outputs/${CROWD_LIBRARY_OUT:-crowd_library_20260713}"
CELLS=(1_random_white_plain 2_clip_white_plain 3_clip_pink_plain 4_clip_white_axes \
       5_clip_pink_axes 8_clip_white_llm6 9_clip_pink_llm6)

for cell in "${CELLS[@]}"; do
  cell_dir="$OUT/$cell"
  [ -d "$cell_dir" ] || { echo "[run_crowd_library_eval] cell=$cell has no dir yet -- skipping" | tee -a "$OUT/eval.log"; continue; }
  for prompt_dir in "$cell_dir"/*/; do
    [ -d "$prompt_dir" ] || continue
    slug="$(basename "$prompt_dir")"
    metrics="$prompt_dir/metrics.json"
    if [ -f "$metrics" ]; then
      echo "[run_crowd_library_eval] cell=$cell slug=$slug metrics.json already exists, skipping" | tee -a "$OUT/eval.log"
      continue
    fi
    python -u scripts/evaluate_crowd.py --crowd-dir "$prompt_dir" --device cpu 2>&1 | tee -a "$OUT/eval.log"
    rc=${PIPESTATUS[0]}
    echo "[run_crowd_library_eval] cell=$cell slug=$slug evaluate_exit=$rc" | tee -a "$OUT/eval.log"
  done
done

echo "CROWD LIBRARY EVAL PASS COMPLETE" | tee -a "$OUT/eval.log"
