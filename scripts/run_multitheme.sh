#!/bin/bash
# Multi-theme generalization batch (2026-07-10): do factorial7's conclusions
# hold beyond "medieval villagers"? The 7 keeper methods (cells 6/7 dropped --
# superseded by the 6-axis WANDER cells 8/9, see run_factorial7.sh) x 2 new
# themes x N=5 = 70 members at ~21 min/member => ~24-25 h.
#
# Same crash-hardened structure as run_factorial7.sh (see its header for the
# full war-story rationale): generate to LOCAL WSL disk, --sync-to mirrors
# every member to U: as it finishes, per-cell rsync as a consistency check,
# every cell || exit 1 so a broken machine aborts instead of burning members.
#
# DO NOT launch directly -- launch run_multitheme_supervised.sh (visible
# window, from Windows PowerShell):
#   Start-Process wsl.exe -ArgumentList '-u root bash -c "mkdir -p /mnt/u; mountpoint -q /mnt/u || mount -t drvfs U: /mnt/u; bash \"/mnt/u/My Documents/LatentWardrobe/scripts/run_multitheme_supervised.sh\""'
#
# NOTE: no `set -u` -- conda's activate scripts reference unset vars.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate paint3d
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
REPO="/mnt/u/My Documents/LatentWardrobe"

if ! timeout 20 stat "$REPO" >/dev/null 2>&1; then
  echo "[run_multitheme] FATAL: \"$REPO\" not responding within 20s -- U: mount is not healthy, aborting before launch" >&2
  exit 1
fi
cd "$REPO"

REMOTE_OUT="$REPO/outputs/multitheme_20260710"
LOCAL_OUT=/root/multitheme_local
mkdir -p "$LOCAL_OUT" "$REMOTE_OUT"

# pull down anything already synced (resume after crash/wipe) -- 600s budget,
# multi-GB over drvfs after a distro wipe.
if ! timeout 600 rsync -a "$REMOTE_OUT/" "$LOCAL_OUT/"; then
  echo "[run_multitheme] FATAL: initial rsync from $REMOTE_OUT timed out/failed -- check the U: mount before relaunching" >&2
  exit 1
fi

PROMPT_A='superheroes'
PROMPT_B='business formalwear'
N=5
TAU=0.02

run () {  # run <cell-outdir> [extra driver args...]
  local cell="$1"; shift
  python -u scripts/generate_crowd_baseline.py \
    --prompt "$PROMPT_A" --prompt "$PROMPT_B" --num-members "$N" \
    --outdir "$LOCAL_OUT/$cell" --sync-to "$REMOTE_OUT/$cell" --resume "$@" \
    2>&1 | tee -a "$LOCAL_OUT/run.log"
  local rc=${PIPESTATUS[0]}
  echo "[run_multitheme] cell=$cell python_exit=$rc" | tee -a "$LOCAL_OUT/run.log"
  timeout 120 rsync -a "$LOCAL_OUT/$cell/" "$REMOTE_OUT/$cell/" \
    || echo "[run_multitheme] cell=$cell end-of-cell rsync timed out/failed (non-fatal, --sync-to already covers most of it)" | tee -a "$LOCAL_OUT/run.log"
  cp "$LOCAL_OUT/run.log" "$REMOTE_OUT/run.log" 2>/dev/null || true
  return "$rc"
}

# 1) random mesh, white noise, plain prompt -- the baseline
run 1_random_white_plain || exit 1

# 2) sampled mesh, white noise, plain prompt
run 2_clip_white_plain --mesh-sampling clip --tau "$TAU" || exit 1

# 3) sampled mesh, pink noise, plain prompt
run 3_clip_pink_plain --mesh-sampling clip --tau "$TAU" --pink-noise || exit 1

# 4) sampled mesh, white noise, random-colour prompt
run 4_clip_white_color --mesh-sampling clip --tau "$TAU" --color-prompt || exit 1

# 5) sampled mesh, pink noise, random-colour prompt
run 5_clip_pink_color --mesh-sampling clip --tau "$TAU" --pink-noise --color-prompt || exit 1

# 8/9) 6-axis WANDER (Method 5b -- the factorial7 winner); cell numbers kept
#      from factorial7 so plot labels/colors stay consistent across batches.
run 8_clip_white_llm6 --mesh-sampling clip --tau "$TAU" --llm-emitters --llm-axes clothing --llm-novelty clothing || exit 1
run 9_clip_pink_llm6 --mesh-sampling clip --tau "$TAU" --pink-noise --llm-emitters --llm-axes clothing --llm-novelty clothing || exit 1

echo "MULTITHEME GENERATION COMPLETE -- run scripts/run_multitheme_eval.sh next" | tee -a "$LOCAL_OUT/run.log"
cp "$LOCAL_OUT/run.log" "$REMOTE_OUT/run.log" 2>/dev/null || true
