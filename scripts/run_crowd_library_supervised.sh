#!/bin/bash
# Supervisor for run_crowd_library.sh -- a direct adaptation of
# run_multitheme_supervised.sh (see run_factorial7_supervised.sh for the
# design rationale: counts members ON DISK rather than trusting exit codes,
# health-gates the mount + GPU before every launch, 2h watchdog
# force-restart, halts after 3 zero-progress attempts). Differences here:
# - Batch identity: run_crowd_library.sh's cells/paths, honouring the same
#   CROWD_LIBRARY_OUT / CROWD_LIBRARY_LOCAL env overrides it takes (so the
#   stage1eval variant is supervised with the same script).
# - CROWD_LIBRARY_BATCH_SCRIPT picks which batch script gets supervised
#   (default run_crowd_library.sh; set run_crowd_library_mesh_fixed.sh for
#   the mesh-fixed replay batch -- its own MESHFIXED_* out/local dirs must
#   then match the CROWD_LIBRARY_OUT/LOCAL this supervisor monitors).
# - TARGET is computed from scripts/crowd_library_prompts.txt at launch
#   (the theme list is ever-growing) instead of a hardcoded prompt count.
# - Done-marker is stage1_render/render_rgb.mp4, the final per-member
#   artifact under --skip-stage2 (see generate_crowd_baseline.py's resume
#   check), not stage2/tile_res_0/albedo.png -- run_crowd_library.sh runs
#   stage1-only as of 2026-07-14.
#
# Launch (visible window, from Windows PowerShell; add the CROWD_LIBRARY_*
# exports before `bash` for the stage1eval variant):
#   Start-Process wsl.exe -ArgumentList '-u root bash -c "mkdir -p /mnt/u; mountpoint -q /mnt/u || mount -t drvfs U: /mnt/u; bash \"/mnt/u/My Documents/LatentWardrobe/scripts/run_crowd_library_supervised.sh\""'
#
# NOTE: no `set -u`/`set -e` -- see run_factorial7_supervised.sh.

main() {
  local REPO="/mnt/u/My Documents/LatentWardrobe"
  local BATCH_SCRIPT="$REPO/scripts/${CROWD_LIBRARY_BATCH_SCRIPT:-run_crowd_library.sh}"
  local LOCAL_OUT="/root/${CROWD_LIBRARY_LOCAL:-crowd_library_local}"
  local REMOTE_OUT="$REPO/outputs/${CROWD_LIBRARY_OUT:-crowd_library_20260713}"
  local PY=/root/miniconda3/envs/paint3d/bin/python
  local SLOG="$LOCAL_OUT/supervisor.log"
  local CELLS=(1_random_white_plain 2_clip_white_plain 3_clip_pink_plain
               4_clip_white_axes 5_clip_pink_axes 8_clip_white_llm6 9_clip_pink_llm6)
  local N=5
  local BATCH_WINDOW=7200   # max seconds per attempt before forced restart
  export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
  mkdir -p "$LOCAL_OUT"

  # themes currently queued: non-blank lines with comments stripped
  local PROMPTS
  PROMPTS=$(sed -e 's/#.*//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
              "$REPO/scripts/crowd_library_prompts.txt" | grep -c .)
  local TARGET=$(( ${#CELLS[@]} * N * PROMPTS ))

  log() { echo "[supervisor] $(date '+%F %T') $*" >> "$SLOG"; }
  push_status() {
    timeout 30 cp "$SLOG" "$REMOTE_OUT/supervisor.log" 2>/dev/null
    timeout 30 cp "$LOCAL_OUT/run.log" "$REMOTE_OUT/run.log" 2>/dev/null
    return 0
  }
  mount_ok() { timeout 20 stat "$BATCH_SCRIPT" >/dev/null 2>&1; }
  gpu_ok()   { timeout 180 "$PY" -c "import torch; x=torch.randn(512,512,device='cuda'); (x@x).sum().item()" >/dev/null 2>&1; }
  batch_alive() { pgrep -f 'generate_crowd_baseline\.py|pipeline_paint3d_stage' >/dev/null 2>&1; }
  count_done() {  # members with a finished stage1 turntable on local disk
    local n=0 c f
    for c in "${CELLS[@]}"; do
      for f in "$LOCAL_OUT/$c"/*/member_*/stage1_render/render_rgb.mp4; do
        [ -e "$f" ] && n=$((n+1))
      done
    done
    echo "$n"
  }

  if [ ! -x "$PY" ]; then
    echo "FATAL: paint3d env missing at $PY -- run .claude/setup/install_paint3d_env.sh (and install_ollama.sh for the llm6 cells), then relaunch."
    exit 1
  fi

  echo "[supervisor] started $(date '+%F %T') -- output from here on: $SLOG (mirrored to $REMOTE_OUT/supervisor.log)"
  log "=== supervisor start: target $TARGET members over ${#CELLS[@]} cells x $PROMPTS prompts, ${BATCH_WINDOW}s watchdog, out=$REMOTE_OUT ==="

  local CUR_PG=""
  trap 'log "SIGHUP received (console window died?) -- batch is detached, continuing"' HUP
  trap 'log "INT/TERM received -- killing batch group and exiting"; [ -n "$CUR_PG" ] && kill -TERM -- "-$CUR_PG" 2>/dev/null; exit 130' INT TERM

  local stall=0 last_done=-1 attempt=0 done_n rc start c
  while :; do
    done_n=$(count_done)
    log "progress: $done_n/$TARGET members complete (attempts so far: $attempt, stalled attempts: $stall)"
    push_status

    if [ "$done_n" -ge "$TARGET" ]; then
      log "all $TARGET members complete -- running final verified sync to U:"
      until mount_ok; do log "U: mount not responding; retrying final sync in 300s"; sleep 300; done
      for c in "${CELLS[@]}"; do
        if timeout 900 rsync -a "$LOCAL_OUT/$c/" "$REMOTE_OUT/$c/"; then
          log "final sync $c: ok"
        else
          log "final sync $c: FAILED/timed out -- rerun 'rsync -a $LOCAL_OUT/$c/ <remote>/$c/' by hand"
        fi
      done
      date > "$REMOTE_OUT/CROWD_LIBRARY_COMPLETE.txt" 2>/dev/null
      log "DONE -- next step: scripts/run_shape_eval.sh / scripts/run_crowd_library_eval.sh"
      push_status
      break
    fi

    if [ "$done_n" -le "$last_done" ]; then stall=$((stall+1)); else stall=0; fi
    last_done=$done_n
    if [ "$stall" -ge 3 ]; then
      log "HALT: 3 consecutive attempts added zero members -- deterministic failure, needs a human (check run.log tail)"
      { date; echo "supervisor halted at $done_n/$TARGET: no progress across 3 attempts"; } \
        > "$REMOTE_OUT/ATTENTION_SUPERVISOR_HALTED.txt" 2>/dev/null
      push_status
      exit 1
    fi

    # gates: never launch onto a machine that will just kill the attempt
    while batch_alive; do
      log "leftover batch process still alive (unkillable D-state from a wedged mount?) -- waiting 300s; will not double-launch onto the GPU"
      push_status; sleep 300
    done
    until mount_ok; do log "U: mount not responding -- waiting 300s"; push_status; sleep 300; done
    until gpu_ok;   do log "GPU probe failed -- waiting 300s"; push_status; sleep 300; done

    attempt=$((attempt+1))
    log "launching batch attempt $attempt"
    # own session: immune to console death, and killable as one process group
    setsid bash "$BATCH_SCRIPT" >/dev/null 2>&1 &
    CUR_PG=$!
    start=$(date +%s)
    while kill -0 "$CUR_PG" 2>/dev/null; do
      if [ $(( $(date +%s) - start )) -ge "$BATCH_WINDOW" ]; then
        log "watchdog: attempt $attempt exceeded ${BATCH_WINDOW}s -- force-restarting (a wedged drvfs/GPU call can hang unkillably; --resume makes the retry cheap)"
        kill -TERM -- "-$CUR_PG" 2>/dev/null
        sleep 45
        kill -KILL -- "-$CUR_PG" 2>/dev/null
        break
      fi
      sleep 60
    done
    # poll instead of `wait`: an unkillable D-state child would block `wait`
    # forever; the batch_alive gate above handles any survivor next loop.
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
      kill -0 "$CUR_PG" 2>/dev/null || break
      sleep 5
    done
    if kill -0 "$CUR_PG" 2>/dev/null; then
      rc=125
      log "attempt $attempt: process group $CUR_PG refuses to die (D-state); not waiting on it"
    else
      wait "$CUR_PG" 2>/dev/null; rc=$?
    fi
    CUR_PG=""
    # strays outside the group (shouldn't exist, but a GPU-holding orphan
    # would poison every following attempt with OOM)
    pkill -KILL -f 'generate_crowd_baseline\.py|pipeline_paint3d_stage' 2>/dev/null
    log "attempt $attempt ended rc=$rc"
    push_status
    sleep 30
  done
}

main
