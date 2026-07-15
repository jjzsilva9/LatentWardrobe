#!/bin/bash
# Eval pass for the Part 2 (texturing-only, MESH-FIXED) batch -- same
# script as run_crowd_library_eval.sh, pointed at the mesh-fixed output
# dir. Reuses evaluate_crowd.py UNCHANGED: the mesh-fixed batch's dirs have
# the same manifest.json/member_XX shape as any other crowd, so nothing
# about scoring needs to know meshes were pinned. Runs on CPU, same reason
# as run_crowd_library_eval.sh (never contend with a GPU-bound generation
# pass).
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eval
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd "/mnt/u/My Documents/LatentWardrobe"

OUT="outputs/${CROWD_LIBRARY_MESHFIXED_OUT:-crowd_library_20260713_meshfixed}"
CELLS=(1_random_white_plain 2_clip_white_plain 3_clip_pink_plain 4_clip_white_axes \
       5_clip_pink_axes 8_clip_white_llm6 9_clip_pink_llm6)

for cell in "${CELLS[@]}"; do
  cell_dir="$OUT/$cell"
  [ -d "$cell_dir" ] || { echo "[run_crowd_library_mesh_fixed_eval] cell=$cell has no dir yet -- skipping" | tee -a "$OUT/eval.log"; continue; }
  for prompt_dir in "$cell_dir"/*/; do
    [ -d "$prompt_dir" ] || continue
    slug="$(basename "$prompt_dir")"
    metrics="$prompt_dir/metrics.json"
    if [ -f "$metrics" ]; then
      echo "[run_crowd_library_mesh_fixed_eval] cell=$cell slug=$slug metrics.json already exists, skipping" | tee -a "$OUT/eval.log"
      continue
    fi
    python -u scripts/evaluate_crowd.py --crowd-dir "$prompt_dir" --device cpu 2>&1 | tee -a "$OUT/eval.log"
    rc=${PIPESTATUS[0]}
    echo "[run_crowd_library_mesh_fixed_eval] cell=$cell slug=$slug evaluate_exit=$rc" | tee -a "$OUT/eval.log"
  done
done

echo "MESH-FIXED EVAL PASS COMPLETE" | tee -a "$OUT/eval.log"
