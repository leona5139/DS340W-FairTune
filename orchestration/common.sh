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
# BATCH_SIZE has no single default here: search_mask.py hardcodes
# drop_last=False, and utilities/utils.py's per-subgroup accuracy/AUC
# functions divide by per-subgroup batch counts with no zero-guard, so an
# uneven leftover batch can crash with ZeroDivisionError. The safe value is
# a divisor of the *current dataset's* split sizes, which differ by dataset
# (PAPILA: 294/84/42, evenly divided by 42; glaucoma: see
# INTERSECTIONAL_FAIRNESS_PLAN.md for its split sizes and chosen divisor) —
# so run_pipeline.sh sets this per-dataset, after parsing --dataset, rather
# than a single value being set here.

# STAGE1_EPOCHS/STAGE1_NUM_TRIALS/STAGE2_EPOCHS: a real timed run on this
# machine measured ~10 min/trial at STAGE1_EPOCHS=15, which put the old
# 15/20 defaults at 20+ hours for a single attribute — nowhere near the
# assignment's 4-hour recording budget. Diagnostic-plot quality and mask
# quality are both explicitly not goals here (see CLAUDE.md); the only
# requirement is that Stage 1 produces *a* mask Stage 2 can load. These
# values trade search depth for wall-clock time accordingly — raise them
# back toward 15/20 only if there's time to spare.
: "${STAGE1_EPOCHS:=5}"
: "${STAGE1_NUM_TRIALS:=8}"
: "${STAGE2_EPOCHS:=5}"

# WORKERS: PyTorch DataLoader with num_workers>0 has a documented history of
# silently deadlocking on ROCm, and can occasionally hang on other GPU/OS
# combinations too. Default conservative; verify_env.py smoke-tests this
# exact value before the real run. Drop to 0 if you see a hang with no
# traceback.
: "${WORKERS:=4}"

: "${PRUNER:=SuccessiveHalving}"

# --search-method sensitivity tunables (search_mask_sensitivity.py, the
# SPT/GPS-style one-shot ranking that replaces Optuna/TPE -- see
# fairtune-speedup-plan.md). SENSITIVITY_NUM_BATCHES left empty by default
# means "exactly one full epoch of the train loader" (the Python script's
# own default; see search_mask_sensitivity.py's compute_sensitivity_scores).
# SENSITIVITY_LR exists because TPE searches --lr per trial
# (1e-5..1e-3 log-uniform) but the k-sweep has no per-trial search to do
# that job -- parse_args.py's shared --lr default (0.1) is a torchvision
# boilerplate value nowhere near right for AdamW ViT fine-tuning, so this
# stage passes an explicit, sane fixed value instead of relying on it.
: "${SENSITIVITY_K_SCHEDULE:=1 2 3 4 6 8 12 18 24 36}"
: "${SENSITIVITY_NUM_BATCHES:=}"
: "${RANK_METHOD:=worst_group_sensitivity}"
: "${SENSITIVITY_NORMALIZE:=param_count}"
: "${SENSITIVITY_SEED:=0}"
: "${SENSITIVITY_LR:=1e-4}"

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
