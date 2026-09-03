#!/usr/bin/env bash
# Shared helpers for run_pipeline.sh. Sourced, not executed directly.

# --- Tunable defaults, all overridable via environment variable ---------
# BATCH_SIZE=42: search_mask.py hardcodes drop_last=False for papila, and
# utilities/utils.py's accuracy_by_gender/accuracy_by_age divide by
# per-subgroup batch counts with no zero-guard, so an uneven leftover batch
# can crash with ZeroDivisionError. 42 evenly divides the split sizes
# (train=294, val=84, test=42), avoiding the crash without touching repo code.
: "${BATCH_SIZE:=42}"

# STAGE1_EPOCHS/STAGE1_NUM_TRIALS: the Colab version kept these at 5/10
# specifically to fit inside a free-tier T4's session-time limit. That
# constraint doesn't exist on a local always-on machine, and it's the likely
# cause of "results too short" / Optuna's diagnostic plots failing (too few
# trials survived pruning with too little signal per trial to compare).
: "${STAGE1_EPOCHS:=15}"
: "${STAGE1_NUM_TRIALS:=20}"
: "${STAGE2_EPOCHS:=15}"

# WORKERS: PyTorch DataLoader with num_workers>0 has a documented history of
# silently deadlocking on ROCm. Default conservative; verify_env.py smoke-tests
# this exact value before the real run. Drop to 0 if you see a hang with no
# traceback.
: "${WORKERS:=4}"

: "${PRUNER:=SuccessiveHalving}"

banner() {
    local title="$1"
    local detail="${2:-}"
    echo ""
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] ${title}"
    if [[ -n "$detail" ]]; then
        echo "   ${detail}"
    fi
    echo "================================================================"
}

# run_cmd CMD...  — echoes the exact command, then runs it with output
# streamed live (no capture) so the recording shows real progress. Relies on
# `set -euo pipefail` in the caller to stop the whole pipeline on failure.
run_cmd() {
    echo "+ $*"
    "$@"
}

# discover_mask REPO_DIR ATTR — most-recently-modified .npy under
# Optuna_Masks/ATTR/, mirroring the notebook's glob-newest-file logic exactly.
# Prints ONLY the path to stdout (diagnostics go to stderr) so it's safe to
# capture with $(discover_mask ...).
discover_mask() {
    local repo_dir="$1"
    local attr="$2"
    local mask
    mask=$(find "$repo_dir" -path "*/Optuna_Masks/${attr}/*.npy" -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -1 | cut -d' ' -f2-)
    if [[ -z "$mask" ]]; then
        echo "ERROR: no mask found under ${repo_dir}/**/Optuna_Masks/${attr}/*.npy — did Stage 1 finish without error?" >&2
        return 1
    fi
    echo "$mask"
}
