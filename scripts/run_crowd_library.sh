#!/bin/bash
# Crowd Library (2026-07-13): open-ended, ever-growing multi-theme x
# multi-method dataset -- successor to run_multitheme.sh's fixed 2-theme
# batch. Restructured theme-major (all 7 methods for theme N before starting
# theme N+1), so an interrupted run always leaves complete, chart-ready data
# behind -- never a theme with some methods done and others missing. Reads
# scripts/crowd_library_prompts.txt fresh every launch: append more themes
# and relaunch any time, --resume makes every (theme, method) combo
# idempotent so already-finished work is never redone.
#
# Consolidates the 3 themes generated in earlier batches (business
# formalwear / superheroes from outputs/multitheme_20260710, medieval
# villagers from outputs/factorial7_20260706) into this one ongoing
# directory. Cells 1/2/3/8/9 for those 3 themes are copied over unchanged
# (see the one-time copy step run before this script's first launch) --
# only cells 4/5 regenerate for them, now as Method 6 (--clip-axes,
# 2026-07-13): CLIP-weighted independent sampling over all six clothing
# emitter axes (not just colour) -- see generate_crowd_baseline.py's
# --clip-axes help and mesh_sampling.axis_weights. Supersedes the
# colour-only --color-prompt that used to live in cells 4/5 (methodically
# the same idea, --clip-axes just does it for every axis, so it wasn't
# worth the compute to run both). Directories renamed _color -> _axes to
# match; the old _color dirs in multitheme_20260710/factorial7_20260706 are
# untouched historical data, not part of this batch.
#
# Same crash-hardened structure as run_multitheme.sh (see its header for the
# full war-story rationale): generate to LOCAL WSL disk, --sync-to mirrors
# every member to U: as it finishes, per-cell rsync as a consistency check,
# every cell || exit 1 so a broken machine aborts instead of burning members.
#
# DO NOT launch directly -- launch via a supervisor (visible window, from
# Windows PowerShell):
#   Start-Process wsl.exe -ArgumentList '-u root bash -c "mkdir -p /mnt/u; mountpoint -q /mnt/u || mount -t drvfs U: /mnt/u; bash \"/mnt/u/My Documents/LatentWardrobe/scripts/run_crowd_library.sh\""'
#
# NOTE: no `set -u` -- conda's activate scripts reference unset vars.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate paint3d
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
REPO="/mnt/u/My Documents/LatentWardrobe"

if ! timeout 20 stat "$REPO" >/dev/null 2>&1; then
  echo "[run_crowd_library] FATAL: \"$REPO\" not responding within 20s -- U: mount is not healthy, aborting before launch" >&2
  exit 1
fi
cd "$REPO"

REMOTE_OUT="$REPO/outputs/crowd_library_20260713"
LOCAL_OUT=/root/crowd_library_local
PROMPTS_FILE="scripts/crowd_library_prompts.txt"
mkdir -p "$LOCAL_OUT" "$REMOTE_OUT"

# pull down anything already synced (resume after crash/wipe, or the 3
# pre-seeded themes' copied-over cells 1/2/3/8/9) -- 600s budget, multi-GB
# over drvfs after a distro wipe.
if ! timeout 600 rsync -a "$REMOTE_OUT/" "$LOCAL_OUT/"; then
  echo "[run_crowd_library] FATAL: initial rsync from $REMOTE_OUT timed out/failed -- check the U: mount before relaunching" >&2
  exit 1
fi

N=5
TAU=0.02
AXIS_TAU=0.05

run () {  # run <prompt> <cell-outdir> [extra driver args...]
  local prompt="$1"; local cell="$2"; shift 2
  python -u scripts/generate_crowd_baseline.py \
    --prompt "$prompt" --num-members "$N" \
    --outdir "$LOCAL_OUT/$cell" --sync-to "$REMOTE_OUT/$cell" --resume "$@" \
    2>&1 | tee -a "$LOCAL_OUT/run.log"
  local rc=${PIPESTATUS[0]}
  echo "[run_crowd_library] prompt='$prompt' cell=$cell python_exit=$rc" | tee -a "$LOCAL_OUT/run.log"
  timeout 120 rsync -a "$LOCAL_OUT/$cell/" "$REMOTE_OUT/$cell/" \
    || echo "[run_crowd_library] prompt='$prompt' cell=$cell end-of-cell rsync timed out/failed (non-fatal, --sync-to already covers most of it)" | tee -a "$LOCAL_OUT/run.log"
  cp "$LOCAL_OUT/run.log" "$REMOTE_OUT/run.log" 2>/dev/null || true
  return "$rc"
}

while IFS= read -r prompt || [ -n "$prompt" ]; do
  prompt="${prompt%%#*}"    # strip trailing comments
  prompt="$(echo -n "$prompt" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  [ -z "$prompt" ] && continue

  echo "[run_crowd_library] === theme: $prompt ===" | tee -a "$LOCAL_OUT/run.log"

  run "$prompt" 1_random_white_plain || exit 1
  run "$prompt" 2_clip_white_plain --mesh-sampling clip --tau "$TAU" || exit 1
  run "$prompt" 3_clip_pink_plain --mesh-sampling clip --tau "$TAU" --pink-noise || exit 1
  run "$prompt" 4_clip_white_axes --mesh-sampling clip --tau "$TAU" \
      --clip-axes --axis-tau "$AXIS_TAU" || exit 1
  run "$prompt" 5_clip_pink_axes --mesh-sampling clip --tau "$TAU" --pink-noise \
      --clip-axes --axis-tau "$AXIS_TAU" || exit 1
  run "$prompt" 8_clip_white_llm6 --mesh-sampling clip --tau "$TAU" \
      --llm-emitters --llm-axes clothing --llm-novelty clothing || exit 1
  run "$prompt" 9_clip_pink_llm6 --mesh-sampling clip --tau "$TAU" --pink-noise \
      --llm-emitters --llm-axes clothing --llm-novelty clothing || exit 1

  echo "[run_crowd_library] === theme complete: $prompt ===" | tee -a "$LOCAL_OUT/run.log"
done < "$PROMPTS_FILE"

echo "CROWD LIBRARY PASS COMPLETE (all prompts in $PROMPTS_FILE done) -- append more to the file and relaunch any time" | tee -a "$LOCAL_OUT/run.log"
cp "$LOCAL_OUT/run.log" "$REMOTE_OUT/run.log" 2>/dev/null || true
