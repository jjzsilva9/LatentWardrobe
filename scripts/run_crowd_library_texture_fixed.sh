#!/bin/bash
# Eval Part 2 (texturing methods only, matched meshes): every method that can
# affect the TEXTURE in any way (cells 1, 2, 3, 4, 5, 8, 9 -- everything
# except the legacy 6/7 slots) replayed against ONE shared set of garment
# meshes + diffusion seeds, so cross-method differences can only come from
# prompt strategy (plain / CLIP-axes / WANDER-LLM) and noise type
# (white/pink) -- never from mesh selection. See generate_crowd_baseline.py's
# --fixed-members-from for the mechanism: each cell still runs its own
# mesh-sampling/axis/LLM logic (so its full_desc prompt text is exactly what
# that method would produce), then garments+outfit_type+sd_seed get
# overwritten from the reference cell's manifest.json, per member index,
# before Paint3D runs.
#
# Reference mesh set: 2_clip_white_plain from the original crowd_library run
# (outputs/crowd_library_20260713) -- CLIP-weighted theme-relevant sampling,
# uncomplicated by axis/LLM machinery, so it's a neutral pick. Cell 2 itself
# needs no regeneration under this scheme (it'd just reproduce itself
# bit-for-bit at the cost of a full Paint3D run) -- copied straight over
# instead.
#
# Compute cost: this is 6 cells' worth of FULL Paint3D generation (stage1 +
# stage2, same cost per member as the original run), not a cheap eval pass --
# don't launch this casually alongside scripts/run_crowd_library.sh without
# checking GPU contention first.
#
# Same crash-hardened structure as run_crowd_library.sh (see its header) --
# generate to LOCAL WSL disk, --sync-to mirrors every member to U: as it
# finishes, every cell || exit 1.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate paint3d
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
REPO="/mnt/u/My Documents/LatentWardrobe"

if ! timeout 20 stat "$REPO" >/dev/null 2>&1; then
  echo "[run_crowd_library_texture_fixed] FATAL: \"$REPO\" not responding within 20s -- U: mount is not healthy, aborting before launch" >&2
  exit 1
fi
cd "$REPO"

REFERENCE_CELL="$REPO/outputs/${CROWD_LIBRARY_OUT:-crowd_library_20260713}/2_clip_white_plain"
REMOTE_OUT="$REPO/outputs/${CROWD_LIBRARY_TEXFIXED_OUT:-crowd_library_20260713_texturefixed}"
LOCAL_OUT="/root/${CROWD_LIBRARY_TEXFIXED_LOCAL:-crowd_library_texturefixed_local}"
PROMPTS_FILE="scripts/crowd_library_prompts.txt"
mkdir -p "$LOCAL_OUT" "$REMOTE_OUT"

if ! timeout 600 rsync -a "$REMOTE_OUT/" "$LOCAL_OUT/"; then
  echo "[run_crowd_library_texture_fixed] FATAL: initial rsync from $REMOTE_OUT timed out/failed -- check the U: mount before relaunching" >&2
  exit 1
fi

N=5
TAU=0.02
AXIS_TAU=0.05

# --skip-stage2 (2026-07-14): the prompt-strategy/noise differences this
# batch isolates are already expressed at stage1 (both stages' prompts are
# built from the same prompt/full_desc/category_phrase -- see
# generate_crowd_baseline.py's build_stage1_cfg/build_stage2_cfg), so scoring
# stage1's own turntable is a reasonable proxy at ~1/9th the per-member cost.
# See run_crowd_library.sh's own --skip-stage2 comment for what this trades
# away (final polish, color-diversity metric).
run () {  # run <prompt> <cell-outdir> [extra driver args...]
  local prompt="$1"; local cell="$2"; shift 2
  python -u scripts/generate_crowd_baseline.py \
    --prompt "$prompt" --num-members "$N" --skip-stage2 \
    --outdir "$LOCAL_OUT/$cell" --sync-to "$REMOTE_OUT/$cell" --resume \
    --fixed-members-from "$REFERENCE_CELL" "$@" \
    2>&1 | tee -a "$LOCAL_OUT/run.log"
  local rc=${PIPESTATUS[0]}
  echo "[run_crowd_library_texture_fixed] prompt='$prompt' cell=$cell python_exit=$rc" | tee -a "$LOCAL_OUT/run.log"
  timeout 120 rsync -a "$LOCAL_OUT/$cell/" "$REMOTE_OUT/$cell/" \
    || echo "[run_crowd_library_texture_fixed] prompt='$prompt' cell=$cell end-of-cell rsync timed out/failed (non-fatal)" | tee -a "$LOCAL_OUT/run.log"
  cp "$LOCAL_OUT/run.log" "$REMOTE_OUT/run.log" 2>/dev/null || true
  return "$rc"
}

while IFS= read -r prompt || [ -n "$prompt" ]; do
  prompt="${prompt%%#*}"
  prompt="$(echo -n "$prompt" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  [ -z "$prompt" ] && continue

  echo "[run_crowd_library_texture_fixed] === theme: $prompt ===" | tee -a "$LOCAL_OUT/run.log"

  # cell 2 = the reference itself: copy, don't regenerate.
  slug="$(python -c "import sys; sys.path.insert(0, 'scripts'); from generate_crowd_baseline import slugify; print(slugify(sys.argv[1]))" "$prompt")"
  if [ -d "$REFERENCE_CELL/$slug" ] && [ ! -d "$LOCAL_OUT/2_clip_white_plain/$slug" ]; then
    mkdir -p "$LOCAL_OUT/2_clip_white_plain"
    cp -r "$REFERENCE_CELL/$slug" "$LOCAL_OUT/2_clip_white_plain/$slug"
    echo "[run_crowd_library_texture_fixed] prompt='$prompt' cell=2_clip_white_plain copied from reference (no regeneration)" | tee -a "$LOCAL_OUT/run.log"
    timeout 120 rsync -a "$LOCAL_OUT/2_clip_white_plain/$slug/" "$REMOTE_OUT/2_clip_white_plain/$slug/" \
      || echo "[run_crowd_library_texture_fixed] prompt='$prompt' cell=2_clip_white_plain copy-sync timed out/failed" | tee -a "$LOCAL_OUT/run.log"
  elif [ ! -d "$REFERENCE_CELL/$slug" ]; then
    echo "[run_crowd_library_texture_fixed] FATAL: reference cell has no data for '$prompt' ($REFERENCE_CELL/$slug missing) -- run the original crowd_library batch for this theme first" >&2
    exit 1
  fi

  run "$prompt" 1_random_white_plain || exit 1
  run "$prompt" 3_clip_pink_plain --mesh-sampling clip --tau "$TAU" --pink-noise || exit 1
  run "$prompt" 4_clip_white_axes --mesh-sampling clip --tau "$TAU" \
      --clip-axes --axis-tau "$AXIS_TAU" || exit 1
  run "$prompt" 5_clip_pink_axes --mesh-sampling clip --tau "$TAU" --pink-noise \
      --clip-axes --axis-tau "$AXIS_TAU" || exit 1
  run "$prompt" 8_clip_white_llm6 --mesh-sampling clip --tau "$TAU" \
      --llm-emitters --llm-axes clothing --llm-novelty clothing || exit 1
  run "$prompt" 9_clip_pink_llm6 --mesh-sampling clip --tau "$TAU" --pink-noise \
      --llm-emitters --llm-axes clothing --llm-novelty clothing || exit 1

  echo "[run_crowd_library_texture_fixed] === theme complete: $prompt ===" | tee -a "$LOCAL_OUT/run.log"
done < "$PROMPTS_FILE"

echo "TEXTURE-FIXED PASS COMPLETE" | tee -a "$LOCAL_OUT/run.log"
cp "$LOCAL_OUT/run.log" "$REMOTE_OUT/run.log" 2>/dev/null || true
