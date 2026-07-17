#!/bin/bash
# Comparison plots for the crowd library -- thin wrapper around
# plot_crowd_comparison.py, single batch dir (everything is consolidated
# under outputs/crowd_library_20260713 now, so no --extra-batch-dir needed).
# Run after run_crowd_library_eval.sh; prompts are discovered from whatever
# metrics.json files exist, so this naturally grows as more themes finish.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eval
cd "/mnt/u/My Documents/LatentWardrobe"

OUT="outputs/${CROWD_LIBRARY_OUT:-crowd_library_20260713}"
MESHFIXED_OUT="outputs/${CROWD_LIBRARY_MESHFIXED_OUT:-crowd_library_20260713_meshfixed}"
CELLS=(1_random_white_plain 2_clip_white_plain 3_clip_pink_plain 4_clip_white_axes \
       5_clip_pink_axes 8_clip_white_llm6 9_clip_pink_llm6)
# shape eval only runs on the white-noise cells -- see run_shape_eval.sh
SHAPE_CELLS=(1_random_white_plain 2_clip_white_plain 4_clip_white_axes 8_clip_white_llm6)

# Charts are grouped by EVALUATION STAGE, all under $OUT/plots/:
#   mesh/       -- mesh-selection-only eval (shape_metrics.json on $OUT):
#                  what does mesh sampling alone change?
#   texture/    -- texture-only eval (metrics.json on $MESHFIXED_OUT, the
#                  mesh-fixed replay batch: meshes+seeds pinned to the
#                  reference cell): what do prompt strategy and noise change
#                  when the meshes can't?
#   end_to_end/ -- full-pipeline eval (metrics.json on $OUT): mesh selection
#                  AND texturing both vary, the as-deployed comparison.

python -u scripts/plot_crowd_comparison.py --batch-dir "$OUT" --methods "${CELLS[@]}" \
  --out-dir "$OUT/plots/end_to_end" 2>&1 | tee -a "$OUT/plot.log"
rc=${PIPESTATUS[0]}
echo "[run_crowd_library_plot] end_to_end_plot_exit=$rc" | tee -a "$OUT/plot.log"

python -u scripts/plot_crowd_comparison.py --batch-dir "$OUT" --methods "${SHAPE_CELLS[@]}" \
  --shape --out-dir "$OUT/plots/mesh" 2>&1 | tee -a "$OUT/plot.log"
rc=${PIPESTATUS[0]}
echo "[run_crowd_library_plot] mesh_plot_exit=$rc" | tee -a "$OUT/plot.log"

# cell 1 is dropped for the texture stage: with meshes+seeds pinned it is a
# replicate of cell 2 (plain prompt, white noise) -- see TEXTURE_METHOD_LABELS
TEXTURE_CELLS=(2_clip_white_plain 3_clip_pink_plain 4_clip_white_axes \
               5_clip_pink_axes 8_clip_white_llm6 9_clip_pink_llm6)
if [ -d "$MESHFIXED_OUT" ]; then
  python -u scripts/plot_crowd_comparison.py --batch-dir "$MESHFIXED_OUT" --methods "${TEXTURE_CELLS[@]}" \
    --texture-labels --out-dir "$OUT/plots/texture" 2>&1 | tee -a "$OUT/plot.log"
  rc=${PIPESTATUS[0]}
  echo "[run_crowd_library_plot] texture_plot_exit=$rc" | tee -a "$OUT/plot.log"
else
  echo "[run_crowd_library_plot] texture stage skipped: $MESHFIXED_OUT does not exist yet (run scripts/run_crowd_library_mesh_fixed.sh + its eval first)" | tee -a "$OUT/plot.log"
fi
