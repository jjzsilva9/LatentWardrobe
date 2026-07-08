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
# 2026-07-08 (evening): DO NOT launch this file directly anymore -- launch
# run_factorial7_supervised.sh instead (see its header). This script died
# silently twice in one day (mount wedge at 16:48 killed cell 4 mid-member,
# then cells 5-7 each burned their first member against the broken mount);
# the supervisor auto-restarts it with health gates + a hang watchdog, and
# --resume makes every restart nearly free.
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
#
# 2026-07-08 (later same day): per-cell sync wasn't enough -- the lab
# machine hard-crashed mid-cell-4 and, because this machine wipes WSL's
# local disk on any crash/reboot, everything in /root/factorial7_local for
# the in-progress cell was lost (cells 1-3, already rsync'd, survived on
# U:). Root-caused the underlying drvfs hangs too: WSL2's drvfs/9P layer is
# documented (multiple open microsoft/WSL issues) to silently stall network
# I/O in uninterruptible sleep with no error -- a real, still-unresolved
# bug, not something fixable here. Generation therefore still never writes
# straight to U: (that's what caused the original multi-hour freezes).
# Instead --sync-to now makes generate_crowd_baseline.py mirror EVERY
# member to the network share as soon as it finishes (backgrounded,
# timeout-bounded -- see sync_pdir() there), not just every cell, so a
# crash now loses at most ~one member (~20-25 min) instead of up to a
# whole cell (~4h at N=10). The per-cell rsync below stays as a cheap,
# already-mostly-redundant final consistency check.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate paint3d
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
REPO="/mnt/u/My Documents/LatentWardrobe"

# fail fast (not a multi-hour silent hang) if the network mount is already
# unhealthy before we commit a GPU to an unattended multi-hour run.
if ! timeout 20 stat "$REPO" >/dev/null 2>&1; then
  echo "[run_factorial7] FATAL: \"$REPO\" not responding within 20s -- U: mount is not healthy, aborting before launch" >&2
  exit 1
fi
cd "$REPO"

REMOTE_OUT="$REPO/outputs/factorial7_20260706"
LOCAL_OUT=/root/factorial7_local
mkdir -p "$LOCAL_OUT" "$REMOTE_OUT"

# pull down anything already on the network share (e.g. cells/members
# generated before this run, or synced by --sync-to mid-cell before a
# crash) so --resume recognizes it and doesn't regenerate completed work.
# timeout-bounded: a wedged mount here should fail loudly, not hang forever.
# 600s not 120s: after a distro wipe LOCAL_OUT starts empty and this pull is
# multi-GB over drvfs -- 120s would false-abort exactly when resume matters most.
if ! timeout 600 rsync -a "$REMOTE_OUT/" "$LOCAL_OUT/"; then
  echo "[run_factorial7] FATAL: initial rsync from $REMOTE_OUT timed out/failed -- check the U: mount before relaunching" >&2
  exit 1
fi

PROMPT='medieval villagers'
N=10
TAU=0.02

run () {  # run <cell-outdir> [extra driver args...]
  local cell="$1"; shift
  python -u scripts/generate_crowd_baseline.py \
    --prompt "$PROMPT" --num-members "$N" \
    --outdir "$LOCAL_OUT/$cell" --sync-to "$REMOTE_OUT/$cell" --resume "$@" \
    2>&1 | tee -a "$LOCAL_OUT/run.log"
  local rc=${PIPESTATUS[0]}
  echo "[run_factorial7] cell=$cell python_exit=$rc" | tee -a "$LOCAL_OUT/run.log"
  # final consistency check now that --sync-to already mirrored every member
  # as it finished -- should be near-instant (rsync skips unchanged files).
  # Non-fatal and timeout-bounded: a hiccup here just means the next cell's
  # own per-member syncs catch up, not that the whole batch stalls.
  timeout 120 rsync -a "$LOCAL_OUT/$cell/" "$REMOTE_OUT/$cell/" \
    || echo "[run_factorial7] cell=$cell end-of-cell rsync timed out/failed (non-fatal, --sync-to already covers most of it)" | tee -a "$LOCAL_OUT/run.log"
  cp "$LOCAL_OUT/run.log" "$REMOTE_OUT/run.log" 2>/dev/null || true
  return "$rc"
}

# || exit 1 on every cell (added 2026-07-08): a cell dying means something
# systemic (GPU/dxg event, wedged mount, killed subprocess), not a bad cell --
# on 2026-07-08 the mount wedged mid-cell-4 and this script marched on through
# cells 5-7, burning each one's first member against a broken machine in ~3
# minutes apiece, then printed COMPLETE. Abort instead; the supervisor
# (run_factorial7_supervised.sh) relaunches when the machine probes healthy.

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

# 6) sampled mesh, white noise, WANDER-inspired prompt
run 6_clip_white_llm --mesh-sampling clip --tau "$TAU" --llm-emitters || exit 1

# 7) sampled mesh, pink noise, WANDER-inspired prompt
run 7_clip_pink_llm --mesh-sampling clip --tau "$TAU" --pink-noise --llm-emitters || exit 1

echo "FACTORIAL7 GENERATION COMPLETE -- run scripts/run_factorial7_eval.sh next" | tee -a "$LOCAL_OUT/run.log"
cp "$LOCAL_OUT/run.log" "$REMOTE_OUT/run.log" 2>/dev/null || true
