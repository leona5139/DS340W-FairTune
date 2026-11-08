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
# BATCH_SIZE: run_pipeline.sh sets a per-dataset default after parsing
# --dataset (PAPILA 42, Harvard-GF 30), kept from earlier runs so timings stay
# comparable. It no longer has to divide the split sizes: the per-group
# accuracy helpers are zero-guarded and the evaluators compute group
# accuracy/AUC once over the whole split (training_utils._pooled_fairness_metrics),
# so a short final batch or a batch missing a group is harmless.

# STAGE1_EPOCHS/STAGE2_EPOCHS: 30, the FairTune paper's fine-tuning budget
# (Sec. 4.1). Used for every TPE trial, every sensitivity k-sweep training and
# the Stage 2 fine-tune, so all methods share one per-training cost. (Earlier
# 5/8/5 defaults were a cut-down budget for the class's recorded run.)
# STAGE1_NUM_TRIALS: run_pipeline.sh's own TPE default; run_experiments.sh
# overrides it with TPE_SAMPLE_TRIALS for Experiment 1.
# TPE_SAMPLE_TRIALS: trials actually run per (dataset, attr) in Experiment 1.
# 5: every TPE trial trains the full epoch budget (pruning is only checked
# after the last epoch), so trial cost is ~constant and only its mean is
# needed; 5 trials (up from 3) steadies that mean for the 200-trial
# extrapolation (summarize_runs.py prints the std to confirm).
# TPE_FULL_TRIALS: budget the sample is extrapolated to (the paper's search
# "saturated within 200 trials", Appendix A.1) by summarize_runs.py.
: "${STAGE1_EPOCHS:=30}"
: "${STAGE1_NUM_TRIALS:=8}"
: "${STAGE2_EPOCHS:=30}"
: "${TPE_SAMPLE_TRIALS:=5}"
: "${TPE_FULL_TRIALS:=200}"

# WORKERS: PyTorch DataLoader with num_workers>0 has a documented history of
# silently deadlocking on ROCm, and can occasionally hang on other GPU/OS
# combinations too. Default conservative; verify_env.py smoke-tests this
# exact value before the real run. Drop to 0 if you see a hang with no
# traceback.
: "${WORKERS:=4}"

: "${PRUNER:=SuccessiveHalving}"

# AMP=1 (default) turns on autocast in every training stage, for TPE and
# sensitivity alike; AMP=0 for full fp32. AMP_DTYPE: bf16 (RDNA4, NVIDIA
# Ampere / RTX 30xx and newer) or fp16 (older NVIDIA cards).
: "${AMP:=1}"
: "${AMP_DTYPE:=bf16}"

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

# best_lr REPO_DIR DATASET SENS_ATTR — learning rate of the best Optuna trial
# (last column of the one data row in Best_Params<attr>_auto_peft1_vit_base_min_auc.csv).
# Used by Stage 2 so the finetune does not fall back to parse_args' default
# lr of 0.1 (which collapsed the earlier PAPILA run to AUC 0.5).
best_lr() {
    local repo_dir="$1" dataset="$2" attr="$3"
    local f="${repo_dir}/vit_base/${dataset}/Optuna Run Stats/Best_Params${attr}_auto_peft1_vit_base_min_auc.csv"
    if [[ ! -f "$f" ]]; then
        echo "ERROR: ${f} not found -- cannot read the best Optuna lr (pass --lr to override)" >&2
        return 1
    fi
    tail -n 1 "$f" | awk -F, '{print $NF}'
}
