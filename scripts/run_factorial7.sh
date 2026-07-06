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
source /root/miniconda3/etc/profile.d/conda.sh
conda activate paint3d
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd "/mnt/u/My Documents/LatentWardrobe"

OUT=outputs/factorial7_20260706
mkdir -p "$OUT"
PROMPT='medieval villagers'
PROMPT_SLUG=medieval_villagers  # must match generate_crowd_baseline.py's slugify(PROMPT)
N=10
TAU=0.02

run () {  # run <cell-outdir> [extra driver args...]
  local cell="$1"; shift
  python -u scripts/generate_crowd_baseline.py \
    --prompt "$PROMPT" --num-members "$N" \
    --outdir "$OUT/$cell" --resume "$@" 2>&1 | tee -a "$OUT/run.log"
  local rc=${PIPESTATUS[0]}
  echo "[run_factorial7] cell=$cell python_exit=$rc" | tee -a "$OUT/run.log"
  python -u scripts/evaluate_crowd.py --crowd-dir "$OUT/$cell/$PROMPT_SLUG" \
    2>&1 | tee -a "$OUT/run.log"
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

echo "FACTORIAL7 BATCH COMPLETE" | tee -a "$OUT/run.log"
