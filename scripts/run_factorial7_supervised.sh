#!/bin/bash
# Self-healing launcher for the factorial7 batch. Launch THIS, not
# run_factorial7.sh directly.
#
# Why (2026-07-08): the batch died silently twice in one day. Post-mortem of
# the second death: the U: drvfs mount wedged at 16:48 (documented WSL2 9P
# bug, see run_factorial7.sh header), killing cell 4's in-flight member; the
# batch script then marched through cells 5-7, each dying within seconds on
# its first member against the broken mount, printed "COMPLETE", and exited.
# GPU (dxg) errors appeared in dmesg through 17:00; by 17:05 GPU, mount and
# ollama all probed healthy again. i.e. the machine has transient multi-minute
# bad spells, and one-shot launches lose hours to them. This wrapper:
#
#   - loops until all cells are complete ON DISK (counts stage2 final
#     textures; never trusts exit codes or "COMPLETE" banners),
#   - health-gates every (re)launch: GPU probe (real CUDA matmul in the
#     paint3d env), U: mount probe, and no leftover batch processes,
#   - runs each attempt under a 2h watchdog so a hung drvfs/GPU call can
#     never stall the night away (--resume makes restarts nearly free; the
#     llm_members.json persistence in generate_crowd_baseline.py makes them
#     safe for the LLM cells too),
#   - halts loudly (ATTENTION file on U:) only if 3 consecutive attempts add
#     zero new members -- that means something deterministic is broken and a
#     human needs to look,
#   - mirrors its log + run.log to the network share after every event, so
#     progress is visible from home even mid-cell,
#   - runs the batch in its own session with stdout detached from the
#     console, so a dying/frozen console window can no longer kill or block
#     generation (the 16:48 event froze run.log mid-line, which only a
#     console/pipe death explains). Closing the window still risks WSL VM
#     teardown if it's the last wsl.exe -- keep the window open anyway.
#
# Launch from Windows PowerShell (visible window, same rule as before):
#
#   Start-Process wsl.exe -ArgumentList '-u root bash -c "mkdir -p /mnt/u; mountpoint -q /mnt/u || mount -t drvfs U: /mnt/u; bash \"/mnt/u/My Documents/LatentWardrobe/scripts/run_factorial7_supervised.sh\""'
#
# Watch progress from anywhere: outputs/factorial7_20260706/supervisor.log
# (and run.log) on U:, refreshed after every event.
#
# After a machine wipe: reinstall per .claude/setup/ (install_paint3d_env.sh,
# install_ollama.sh), then relaunch this -- the batch's own pull-down rsync +
# --resume skip everything already synced to U:.
#
# NOTE: no `set -u`/`set -e` -- conda activate inside run_factorial7.sh
# references unset vars, and this script's whole job is handling failures
# itself rather than dying on them.

main() {
  local REPO="/mnt/u/My Documents/LatentWardrobe"
  local LOCAL_OUT=/root/factorial7_local
  local REMOTE_OUT="$REPO/outputs/factorial7_20260706"
  local PY=/root/miniconda3/envs/paint3d/bin/python
  local SLOG="$LOCAL_OUT/supervisor.log"
  local CELLS=(1_random_white_plain 2_clip_white_plain 3_clip_pink_plain
               4_clip_white_color 5_clip_pink_color 6_clip_white_llm 7_clip_pink_llm)
  local N=10
  local TARGET=$(( ${#CELLS[@]} * N ))
  local BATCH_WINDOW=7200   # max seconds per attempt before forced restart
  export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
  mkdir -p "$LOCAL_OUT"

  log() { echo "[supervisor] $(date '+%F %T') $*" >> "$SLOG"; }
  push_status() {  # best-effort, never blocks long, never fails the caller
    timeout 30 cp "$SLOG" "$REMOTE_OUT/supervisor.log" 2>/dev/null
    timeout 30 cp "$LOCAL_OUT/run.log" "$REMOTE_OUT/run.log" 2>/dev/null
    return 0
  }
  mount_ok() { timeout 20 stat "$REPO/scripts/run_factorial7.sh" >/dev/null 2>&1; }
  gpu_ok()   { timeout 180 "$PY" -c "import torch; x=torch.randn(512,512,device='cuda'); (x@x).sum().item()" >/dev/null 2>&1; }
  batch_alive() { pgrep -f 'generate_crowd_baseline\.py|pipeline_paint3d_stage' >/dev/null 2>&1; }
  count_done() {  # members with a stage2 final texture on local disk
    local n=0 c f
    for c in "${CELLS[@]}"; do
      for f in "$LOCAL_OUT/$c"/*/member_*/stage2/tile_res_0/albedo.png; do
        [ -e "$f" ] && n=$((n+1))
      done
    done
    echo "$n"
  }

  if [ ! -x "$PY" ]; then
    echo "FATAL: paint3d env missing at $PY -- run .claude/setup/install_paint3d_env.sh (and install_ollama.sh for cells 6/7), then relaunch."
    exit 1
  fi

  # The only console output; everything after goes to $SLOG so a frozen/dead
  # console can never block the supervisor (that's how run.log froze at 16:48).
  echo "[supervisor] started $(date '+%F %T') -- output from here on: $SLOG (mirrored to $REMOTE_OUT/supervisor.log)"
  log "=== supervisor start: target $TARGET members over ${#CELLS[@]} cells, ${BATCH_WINDOW}s watchdog ==="

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
      date > "$REMOTE_OUT/FACTORIAL7_COMPLETE.txt" 2>/dev/null
      log "DONE -- next step: scripts/run_factorial7_eval.sh"
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
    setsid bash "$REPO/scripts/run_factorial7.sh" >/dev/null 2>&1 &
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
