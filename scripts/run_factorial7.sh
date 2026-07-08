#!/bin/bash
# Overnight batch (launched 2026-07-06 evening): 7-method reduced factorial
# over mesh selection (random/CLIP-sampled) x noise init (white/pink) x
# prompt mutation (plain/random-colour/WANDER-inspired), 1 theme x N=10.
# The full 2x2x3=12 factorial was reduced to 7 cells at the user's request:
# random-mesh is only tested once (as the baseline, cell 1) since the
# mesh-sampling x mutation interaction only matters once mesh selection is
# actually CLIP-anchored -- see .claude/dissertation-plan.md Methods 1/2/4/5.
#
# 70 members total at ~23 min/member (measured from outputs/weekend_20260703/
# run.log: stage1 ~130-165s + stage2 ~1130-1175s) => ~27h. User explicitly
# accepted this runs past their original ~11am checkpoint.
#
# Launch from Windows PowerShell with a VISIBLE console window (closing the
# window kills the run; do NOT use `nohup ... &` -- WSL2 tears down the VM
# when the launching wsl.exe exits, see outputs/overnight_20260702/README.md):
#
#   Start-Process wsl.exe -ArgumentList '-u root bash -c "mkdir -p /mnt/u; mountpoint -q /mnt/u || mount -t drvfs U: /mnt/u; bash \"/mnt/u/My Documents/LatentWardrobe/scripts/run_factorial7.sh\""'
#
# NOTE: no `set -u` here -- conda's activate scripts reference unset vars and
# would kill the shell before anything is logged (learned 2026-07-03).
#
# 2026-07-07: generation-only now (evaluate_crowd.py split out to
# run_factorial7_eval.sh) -- see that file's header for why.
#
# 2026-07-08: generation now writes to LOCAL WSL disk (/root/factorial7_local,
# ext4, ~900GB free), not directly to the U: network share, and each cell is
# rsync'd to the network share only once, right after that cell finishes.
# Cause: the run went silent for hours with no crash/error, twice, each time
# right after a cell had just finished cleanly (python_exit=0) -- identical
# signature to the very first 2026-07-06 failure, which was `tee` hanging
# mid-write to run.log on the U: drvfs mount. Multi-hour continuous writes
# (run.log via tee, and every stage1/stage2 output file) to a network share
# over an unattended run is just fragile on this machine; generating locally
# and syncing at clean per-cell checkpoints removes the network mount from
# the hot path entirely, so a share hiccup can at worst fail one short rsync
# (retryable) instead of silently freezing GPU generation for hours.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate paint3d
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
REPO="/mnt/u/My Documents/LatentWardrobe"
cd "$REPO"

REMOTE_OUT="$REPO/outputs/factorial7_20260706"
LOCAL_OUT=/root/factorial7_local
mkdir -p "$LOCAL_OUT" "$REMOTE_OUT"

# pull down anything already on the network share (e.g. cells generated
# before this local/sync split existed) so --resume recognizes it and
# doesn't regenerate already-completed members.
rsync -a "$REMOTE_OUT/" "$LOCAL_OUT/"

PROMPT='medieval villagers'
N=10
TAU=0.02

run () {  # run <cell-outdir> [extra driver args...]
  local cell="$1"; shift
  python -u scripts/generate_crowd_baseline.py \
    --prompt "$PROMPT" --num-members "$N" \
    --outdir "$LOCAL_OUT/$cell" --resume "$@" 2>&1 | tee -a "$LOCAL_OUT/run.log"
  local rc=${PIPESTATUS[0]}
  echo "[run_factorial7] cell=$cell python_exit=$rc" | tee -a "$LOCAL_OUT/run.log"
  # sync this cell + the log back to the network share now, at a clean
  # checkpoint -- short and retryable, unlike writing continuously for hours.
  rsync -a "$LOCAL_OUT/$cell/" "$REMOTE_OUT/$cell/"
  cp "$LOCAL_OUT/run.log" "$REMOTE_OUT/run.log"
  return "$rc"
}

# 1) random mesh, white noise, plain prompt -- the baseline
run 1_random_white_plain

# 2) sampled mesh, white noise, plain prompt
run 2_clip_white_plain --mesh-sampling clip --tau "$TAU"

# 3) sampled mesh, pink noise, plain prompt
run 3_clip_pink_plain --mesh-sampling clip --tau "$TAU" --pink-noise

# 4) sampled mesh, white noise, random-colour prompt
run 4_clip_white_color --mesh-sampling clip --tau "$TAU" --color-prompt

# 5) sampled mesh, pink noise, random-colour prompt
run 5_clip_pink_color --mesh-sampling clip --tau "$TAU" --pink-noise --color-prompt

# 6) sampled mesh, white noise, WANDER-inspired prompt
run 6_clip_white_llm --mesh-sampling clip --tau "$TAU" --llm-emitters

# 7) sampled mesh, pink noise, WANDER-inspired prompt
run 7_clip_pink_llm --mesh-sampling clip --tau "$TAU" --pink-noise --llm-emitters

echo "FACTORIAL7 GENERATION COMPLETE -- run scripts/run_factorial7_eval.sh next" | tee -a "$LOCAL_OUT/run.log"
cp "$LOCAL_OUT/run.log" "$REMOTE_OUT/run.log"
