#!/bin/bash
# Comparison plots for the crowd library -- thin wrapper around
# plot_crowd_comparison.py, single batch dir (everything is consolidated
# under outputs/crowd_library_20260713 now, so no --extra-batch-dir needed).
# Run after run_crowd_library_eval.sh; prompts are discovered from whatever
# metrics.json files exist, so this naturally grows as more themes finish.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eval
cd "/mnt/u/My Documents/LatentWardrobe"

OUT=outputs/crowd_library_20260713
CELLS=(1_random_white_plain 2_clip_white_plain 3_clip_pink_plain 4_clip_white_axes \
       5_clip_pink_axes 8_clip_white_llm6 9_clip_pink_llm6)

python -u scripts/plot_crowd_comparison.py --batch-dir "$OUT" --methods "${CELLS[@]}" \
  --out-dir "$OUT/plots" 2>&1 | tee -a "$OUT/plot.log"
rc=${PIPESTATUS[0]}
echo "[run_crowd_library_plot] plot_exit=$rc" | tee -a "$OUT/plot.log"
