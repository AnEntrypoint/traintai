#!/usr/bin/env bash
# v23 real kernel-level process isolation for round60pipelined8chip's eval.
#
# Real finding this exists to work around: process-level isolation for eval
# WITHIN one Kaggle TPU kernel run is architecturally impossible -- PJRT/
# libtpu locks the whole 8-chip board to a single owning OS process, and
# torch_xla has no supported release/reacquire call. The only real isolation
# boundary is a full Kaggle kernel restart. v22's live run proved 2 cycles/
# run is the proven-survivable eval envelope before per-call cost compounds
# fatally; the notebook's own MAX_CYCLES_PER_RUN=2 stops each run cleanly at
# that point and checkpoints student weights + optimizer state + cycle
# history flat to /kaggle/working/ root (kaggle kernels output only reliably
# retrieves root-level files, not nested ones -- traintai round 7 finding).
#
# This script closes the loop: wait for a run to finish -> pull its real
# checkpoint -> publish it as a new dataset version -> push a fresh kernel
# run that resumes from it -> repeat, bounded by MAX_RELAUNCHES.
set -uo pipefail
cd "$(dirname "$0")/../../round60-pipelined-8chip" || { echo "round60-pipelined-8chip dir not found next to traintai/" >&2; exit 1; }

KERNEL="heclgang/round60pipelined8chip"
CHECKPOINT_DATASET="heclgang/round60-checkpoint"
POLL_INTERVAL_S=60
MAX_RELAUNCHES=20   # real bound -- ~2 cycles/run * 20 = up to 40 real cycles across the whole chain before this script stops itself and reports

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

# First real run of the chain: push once before entering the wait loop below
# (every subsequent iteration's push happens at the end of its own loop body,
# after that iteration's checkpoint round-trip -- this is the one push with
# no prior iteration to follow).
log "=== initial push: starting the relaunch chain ==="
kaggle kernels push -p . 2>&1
log "waiting ${POLL_INTERVAL_S}s for the initial push to register before polling status"
sleep "$POLL_INTERVAL_S"

for i in $(seq 1 "$MAX_RELAUNCHES"); do
  log "=== relaunch iteration $i/$MAX_RELAUNCHES: waiting for $KERNEL to reach a terminal state ==="
  while true; do
    status=$(kaggle kernels status "$KERNEL" 2>&1)
    log "status: $status"
    if echo "$status" | grep -qE "COMPLETE|ERROR"; then
      break
    fi
    sleep "$POLL_INTERVAL_S"
  done

  if echo "$status" | grep -q "ERROR"; then
    log "!! kernel ERRORed this run -- pulling log for diagnosis, NOT auto-relaunching (a real crash needs a human/agent to read the log and decide the next fix, not blind retry)"
    kaggle kernels output "$KERNEL" -p ./out_relaunch_$i 2>&1
    log "log downloaded to ./out_relaunch_$i -- stopping the relaunch chain here"
    exit 1
  fi

  log "kernel COMPLETE -- pulling real checkpoint output"
  out_dir="./out_relaunch_$i"
  rm -rf "$out_dir"
  kaggle kernels output "$KERNEL" -p "$out_dir" 2>&1

  if [ ! -f "$out_dir/config.json" ]; then
    log "!! no config.json in this run's output -- student_load_ok was likely False or checkpoint save failed. Stopping (real, not silently retrying with no checkpoint to resume from)."
    exit 1
  fi

  cycles_total=$(python -c "import json; print(len(json.load(open('$out_dir/cycle_history.json'))))" 2>&1) || cycles_total="unknown"
  log "real checkpoint found: $cycles_total total cycle_history rows so far"

  # kaggle datasets version requires dataset-metadata.json inside the upload
  # dir -- kaggle kernels output never writes one (it downloads real kernel
  # output files only), so it must be written here each iteration.
  cat > "$out_dir/dataset-metadata.json" <<EOF
{
  "title": "round60 checkpoint",
  "id": "$CHECKPOINT_DATASET",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

  log "publishing checkpoint as new version of $CHECKPOINT_DATASET"
  kaggle datasets version -p "$out_dir" -m "round60 relaunch iteration $i -- $cycles_total real cycles total" -r skip 2>&1

  log "re-pushing $KERNEL to resume from the updated checkpoint"
  kaggle kernels push -p . 2>&1

  log "waiting ${POLL_INTERVAL_S}s for the new push to register before polling status"
  sleep "$POLL_INTERVAL_S"
done

log "=== MAX_RELAUNCHES ($MAX_RELAUNCHES) reached -- stopping the relaunch chain. Check the last out_relaunch_* dir's cycle_history.json for the real cumulative result. ==="
