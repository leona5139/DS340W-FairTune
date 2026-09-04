#!/usr/bin/env bash
# Shared helpers for run_pipeline.sh. Sourced, not executed directly.

# PY_BIN: resolve straight from $VIRTUAL_ENV rather than trusting PATH.
# Windows' `python -m venv` only ever creates Scripts/python.exe, never a
# python3 — so a plain `command -v python3` skips right past the activated
# venv and hits Windows' App Execution Alias stub (a shim that just errors
# out telling you to install Python from the Microsoft Store), even with the
# venv active. run_pipeline.sh already requires VIRTUAL_ENV to be set before
# sourcing this, so go directly to that interpreter; only fall back to a
# PATH search if this is ever sourced without an active venv.
PY_BIN=""
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    if [[ -x "${VIRTUAL_ENV}/Scripts/python.exe" ]]; then
        PY_BIN="${VIRTUAL_ENV}/Scripts/python.exe"
    elif [[ -x "${VIRTUAL_ENV}/bin/python" ]]; then
        PY_BIN="${VIRTUAL_ENV}/bin/python"
    fi
fi
[[ -z "$PY_BIN" ]] && PY_BIN="$(command -v python3 >/dev/null 2>&1 && echo python3 || echo python)"

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

# WORKERS: PyTorch DataLoader with num_workers>0 can occasionally hang with
# no traceback on some GPU/OS combinations. Default conservative;
# verify_env.py smoke-tests this exact value before the real run. Drop to 0
# if you see a hang with no traceback.
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
