#!/bin/bash
# Pipeline watcher: chains the full crowd-library pipeline unattended --
# 2 generations + 3 evaluations + plots/contact sheets -- launching each
# phase the moment its dependency completes:
#
#   phase A  wait: end-to-end generation (already running under
#            run_crowd_library_supervised.sh when this watcher starts;
#            relaunched here if its supervisor vanishes without a marker)
#   phase B  launch: mesh-fixed generation (GPU, supervised, reference =
#            the end-to-end batch's 2_clip_white_plain) AND the end-to-end
#            eval (CPU) in parallel; wait for both
#   phase C  launch: shape eval (GPU render + CPU scoring) AND the
#            mesh-fixed texture eval (CPU) in parallel; wait for both
#   phase D  run: run_crowd_library_plot.sh (fills plots/{mesh,texture,
#            end_to_end}) + contact sheets per theme for both batches;
#            write PIPELINE_COMPLETE.txt
#
# Completion detection reuses what each piece already writes: the
# supervisors' CROWD_LIBRARY_COMPLETE.txt, the eval runners' "* PASS
# COMPLETE" log banners. If a supervisor writes
# ATTENTION_SUPERVISOR_HALTED.txt this watcher stops the chain and writes
# PIPELINE_HALTED.txt rather than evaluating garbage. Eval runners are
# relaunched (capped) if they die without their banner -- all of them
# skip already-scored dirs, so relaunch is idempotent.
#
# Batch names come from env (REQUIRED -- no useful defaults across dates):
#   CROWD_LIBRARY_OUT           end-to-end batch, e.g. crowd_library_20260715_stage1eval
#   CROWD_LIBRARY_MESHFIXED_OUT mesh-fixed batch, e.g. crowd_library_20260715_meshfixed
#
# Launch (visible window, from Windows PowerShell):
#   Start-Process wsl.exe -ArgumentList '-u root bash -c "mkdir -p /mnt/u; mountpoint -q /mnt/u || mount -t drvfs U: /mnt/u; export CROWD_LIBRARY_OUT=<e2e> CROWD_LIBRARY_MESHFIXED_OUT=<meshfixed>; bash \"/mnt/u/My Documents/LatentWardrobe/scripts/run_crowd_library_pipeline.sh\""'
#
# NOTE: no `set -u`/`set -e` -- same conda-activate reasons as the run scripts.

REPO="/mnt/u/My Documents/LatentWardrobe"
E2E="$CROWD_LIBRARY_OUT"
MF="$CROWD_LIBRARY_MESHFIXED_OUT"
if [ -z "$E2E" ] || [ -z "$MF" ]; then
  echo "FATAL: export CROWD_LIBRARY_OUT (end-to-end batch) and CROWD_LIBRARY_MESHFIXED_OUT (mesh-fixed batch) before launching" >&2
  exit 1
fi
E2E_DIR="$REPO/outputs/$E2E"
MF_DIR="$REPO/outputs/$MF"
PLOG="$E2E_DIR/pipeline_watcher.log"
CELLS="1_random_white_plain 2_clip_white_plain 3_clip_pink_plain 4_clip_white_axes 5_clip_pink_axes 8_clip_white_llm6 9_clip_pink_llm6"
POLL=120

log() { echo "[pipeline] $(date '+%F %T') $*" | tee -a "$PLOG"; }
# console-window death must not kill the chain (same rationale as the
# supervisor's HUP trap). A WSL VM crash still kills everything inside it --
# if this log goes silent with no marker files appearing, run `wsl --shutdown`
# from Windows and relaunch this script; every phase re-detects completion
# from disk, so relaunch is always safe.
trap 'log "SIGHUP received (console window died?) -- continuing"' HUP
halt() {
  { date; echo "$1"; } > "$E2E_DIR/PIPELINE_HALTED.txt" 2>/dev/null
  log "HALTED: $1"
  exit 1
}

gen_done()   { [ -f "$1/CROWD_LIBRARY_COMPLETE.txt" ]; }
gen_halted() { [ -f "$1/ATTENTION_SUPERVISOR_HALTED.txt" ]; }

launch_e2e_supervisor() {
  CROWD_LIBRARY_OUT="$E2E" CROWD_LIBRARY_LOCAL="${E2E}_local" \
    setsid bash "$REPO/scripts/run_crowd_library_supervised.sh" >/dev/null 2>&1 &
}
launch_mf_supervisor() {
  CROWD_LIBRARY_OUT="$MF" CROWD_LIBRARY_LOCAL="${MF}_local" \
  CROWD_LIBRARY_MESHFIXED_OUT="$MF" CROWD_LIBRARY_MESHFIXED_LOCAL="${MF}_local" \
  CROWD_LIBRARY_BATCH_SCRIPT=run_crowd_library_mesh_fixed.sh \
  CROWD_LIBRARY_REF_OUT="$E2E" \
    setsid bash "$REPO/scripts/run_crowd_library_supervised.sh" >/dev/null 2>&1 &
}

log "=== pipeline watcher start: e2e=$E2E meshfixed=$MF ==="

# ---- phase A: end-to-end generation ---------------------------------------
relaunch=0
until gen_done "$E2E_DIR"; do
  gen_halted "$E2E_DIR" && halt "end-to-end supervisor halted (see $E2E_DIR/ATTENTION_SUPERVISOR_HALTED.txt)"
  if ! pgrep -f "[r]un_crowd_library_supervised.sh" >/dev/null; then
    relaunch=$((relaunch+1))
    [ "$relaunch" -gt 3 ] && halt "end-to-end supervisor vanished without completing ($relaunch relaunch attempts)"
    log "phase A: supervisor not running and no completion marker -- relaunching (#$relaunch)"
    launch_e2e_supervisor
  fi
  sleep "$POLL"
done
log "phase A complete: end-to-end generation done ($E2E)"
sleep 30  # let the finished supervisor fully exit before pattern-matching for the next one

# ---- phase B: mesh-fixed generation ONLY -----------------------------------
# Serialized 2026-07-16: this phase used to overlap the end-to-end eval (CPU)
# with generation (GPU), but the WSL VM died twice in one day under that
# combined load -- the eval now carries a ViT-H HPSv2 model, and the sum
# exceeded the VM's memory ceiling (see ~/.wslconfig). One memory-heavy
# workload at a time from here on.
log "phase B: launching mesh-fixed generation (GPU)"
launch_mf_supervisor
relaunch_mf=0
until gen_done "$MF_DIR"; do
  gen_halted "$MF_DIR" && halt "mesh-fixed supervisor halted (see $MF_DIR/ATTENTION_SUPERVISOR_HALTED.txt)"
  if ! pgrep -f "[r]un_crowd_library_supervised.sh" >/dev/null; then
    relaunch_mf=$((relaunch_mf+1))
    [ "$relaunch_mf" -gt 3 ] && halt "mesh-fixed supervisor vanished without completing ($relaunch_mf relaunch attempts)"
    log "phase B: mesh-fixed supervisor not running -- relaunching (#$relaunch_mf)"
    launch_mf_supervisor
  fi
  sleep "$POLL"
done
log "phase B complete: mesh-fixed generation done ($MF)"
sleep 30

# ---- phase C: the three evals, strictly sequential --------------------------
e3_done() { grep -q "CROWD LIBRARY EVAL PASS COMPLETE" "$E2E_DIR/eval.log" 2>/dev/null; }
e2_done() { grep -q "MESH-FIXED EVAL PASS COMPLETE" "$MF_DIR/eval.log" 2>/dev/null; }
e1_done() { grep -q "SHAPE EVAL PASS COMPLETE" "$E2E_DIR/shape_eval.log" 2>/dev/null; }

log "phase C1: end-to-end eval"
CROWD_LIBRARY_OUT="$E2E" setsid bash "$REPO/scripts/run_crowd_library_eval.sh" >/dev/null 2>&1 &
relaunch_e3=0
until e3_done; do
  if ! pgrep -f "[r]un_crowd_library_eval.sh" >/dev/null; then
    relaunch_e3=$((relaunch_e3+1))
    [ "$relaunch_e3" -gt 5 ] && halt "end-to-end eval keeps dying without its completion banner ($relaunch_e3 relaunches)"
    log "phase C1: end-to-end eval not running -- relaunching (#$relaunch_e3, idempotent)"
    CROWD_LIBRARY_OUT="$E2E" setsid bash "$REPO/scripts/run_crowd_library_eval.sh" >/dev/null 2>&1 &
  fi
  sleep "$POLL"
done
log "phase C1 complete: end-to-end eval scored"

log "phase C2: mesh-fixed eval"
CROWD_LIBRARY_MESHFIXED_OUT="$MF" setsid bash "$REPO/scripts/run_crowd_library_mesh_fixed_eval.sh" >/dev/null 2>&1 &
relaunch_e2=0
until e2_done; do
  if ! pgrep -f "[r]un_crowd_library_mesh_fixed_eval.sh" >/dev/null; then
    relaunch_e2=$((relaunch_e2+1))
    [ "$relaunch_e2" -gt 5 ] && halt "mesh-fixed eval keeps dying without its completion banner ($relaunch_e2 relaunches)"
    log "phase C2: mesh-fixed eval not running -- relaunching (#$relaunch_e2, idempotent)"
    CROWD_LIBRARY_MESHFIXED_OUT="$MF" setsid bash "$REPO/scripts/run_crowd_library_mesh_fixed_eval.sh" >/dev/null 2>&1 &
  fi
  sleep "$POLL"
done
log "phase C2 complete: mesh-fixed eval scored"

log "phase C3: shape eval"
CROWD_LIBRARY_OUT="$E2E" setsid bash "$REPO/scripts/run_shape_eval.sh" >/dev/null 2>&1 &
relaunch_e1=0
until e1_done; do
  if ! pgrep -f "[r]un_shape_eval.sh" >/dev/null; then
    relaunch_e1=$((relaunch_e1+1))
    [ "$relaunch_e1" -gt 5 ] && halt "shape eval keeps dying without its completion banner ($relaunch_e1 relaunches)"
    log "phase C3: shape eval not running -- relaunching (#$relaunch_e1, idempotent)"
    CROWD_LIBRARY_OUT="$E2E" setsid bash "$REPO/scripts/run_shape_eval.sh" >/dev/null 2>&1 &
  fi
  sleep "$POLL"
done
log "phase C complete: all three evals scored"

# ---- phase D: plots + contact sheets ---------------------------------------
log "phase D: plots + contact sheets"
cd "$REPO" || halt "repo not reachable for phase D"
CROWD_LIBRARY_OUT="$E2E" CROWD_LIBRARY_MESHFIXED_OUT="$MF" \
  bash "$REPO/scripts/run_crowd_library_plot.sh" >>"$PLOG" 2>&1

source /root/miniconda3/etc/profile.d/conda.sh
conda activate eval
sheets() {  # sheets <batch-name> <stage-subdir>
  local batch="$1" stage="$2" pdir slug
  for pdir in "$REPO/outputs/$batch/1_random_white_plain"/*/; do
    [ -f "$pdir/manifest.json" ] || continue
    slug="$(basename "$pdir")"
    python -u scripts/plot_crowd_contact_sheet.py --batch-dir "outputs/$batch" \
      --methods $CELLS --prompt-slug "$slug" \
      --out "outputs/$E2E/plots/$stage/contact_sheets/$slug.png" >>"$PLOG" 2>&1 \
      || log "phase D: contact sheet failed for $batch/$slug (non-fatal)"
  done
}
sheets "$E2E" end_to_end
sheets "$MF" texture

date > "$E2E_DIR/PIPELINE_COMPLETE.txt"
log "=== PIPELINE COMPLETE: plots in outputs/$E2E/plots/{mesh,texture,end_to_end} ==="
