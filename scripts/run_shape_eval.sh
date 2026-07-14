#!/bin/bash
# Eval Part 1 (silhouette/mesh-selection only) for the crowd library. Two
# passes: render untextured turntables (GPU, paint3d env), then score them
# (CPU, eval env -- see run_crowd_library_eval.sh for why eval always runs on
# CPU: a GPU-bound eval pass stalled ~47 min behind a concurrent Paint3D run
# on 2026-07-13). Safe to re-run: both generate_shape_eval.py --resume and the
# metrics.json-exists check below skip completed work.
#
# NOTE: the render pass IS GPU-bound (Paint3D's differentiable renderer, no
# diffusion). If scripts/run_crowd_library.sh is generating at the same time,
# either wait for it to finish or accept some slowdown on both -- don't expect
# this to be free like the CPU-only eval pass is.
source /root/miniconda3/etc/profile.d/conda.sh
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd "/mnt/u/My Documents/LatentWardrobe"

OUT=outputs/crowd_library_20260713

conda activate paint3d
python -u scripts/generate_shape_eval.py --library-dir "$OUT" --resume 2>&1 | tee -a "$OUT/shape_render.log"

conda activate eval
CELLS=(1_random_white_plain 2_clip_white_plain 4_clip_white_axes 8_clip_white_llm6)
for cell in "${CELLS[@]}"; do
  cell_dir="$OUT/$cell"
  [ -d "$cell_dir" ] || continue
  for prompt_dir in "$cell_dir"/*/; do
    [ -d "$prompt_dir" ] || continue
    slug="$(basename "$prompt_dir")"
    metrics="$prompt_dir/shape_metrics.json"
    if [ -f "$metrics" ]; then
      echo "[run_shape_eval] cell=$cell slug=$slug shape_metrics.json already exists, skipping" | tee -a "$OUT/shape_eval.log"
      continue
    fi
    python -u scripts/evaluate_shape.py --crowd-dir "$prompt_dir" --device cpu 2>&1 | tee -a "$OUT/shape_eval.log"
    rc=${PIPESTATUS[0]}
    echo "[run_shape_eval] cell=$cell slug=$slug evaluate_exit=$rc" | tee -a "$OUT/shape_eval.log"
  done
done

echo "SHAPE EVAL PASS COMPLETE" | tee -a "$OUT/shape_eval.log"
