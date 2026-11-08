#!/usr/bin/env bash
# Single entry point for the FairTune pipeline (PAPILA and Harvard-GF
# glaucoma). Replaces the old Colab notebook's click-every-cell workflow:
# this is the "press play and walk away" script meant to be screen-recorded
# directly.
#
# Usage:
#   ./run_pipeline.sh {all|preprocess|config|verify|stage1|stage2} [gender|age|race|intersectional|both] [options]
#
# Options:
#   --dataset {papila|glaucoma}  Which dataset to run (default: papila).
#   --search-method {optuna|sensitivity}  Stage 1 mask-search method (default: optuna).
#                          "sensitivity" runs search_mask_sensitivity.py -- a one-shot
#                          gradient-sensitivity ranking + nested k-sweep (see
#                          fairtune-speedup-plan.md) -- instead of search_mask.py's
#                          Optuna/TPE search. Stage 2 is unaffected either way: it just
#                          discovers whichever mask Stage 1 most recently wrote.
#   --smoke-test          Use tiny epochs/trials (2/2/2) for a fast dry run.
#   --repo-dir PATH        FairTune repo clone to run against (default: this script's own repo).
#   --papila-dir PATH      Raw Papila/ folder, containing ClinicalData/ + FundusImages/
#                          (default: $HOME/fairtune_papila/data/Papila). Only used when --dataset papila.
#   --harvard-dir PATH     Raw harvard-dataset/ folder, containing Bscan/ + RNFLT/
#                          (default: $HOME/fairtune_papila/data/harvard-dataset). Only used when --dataset glaucoma.
#   --splits-dir PATH      Where train.csv/val.csv/test.csv live or get written
#                          (default: REPO_DIR/data_splits_papila or REPO_DIR/data_splits_glaucoma,
#                          both already committed to the repo).
#   --force-preprocess     Regenerate the CSVs even if they already exist.
#   --mask-path PATH       Stage 2 only: use this mask instead of the most recent one under
#                          Optuna_Masks/<attr>/ (e.g. evaluate a gender-searched mask on the
#                          intersectional groups: stage2 intersectional --mask-path <gender mask>).
#   --lr LR                Stage 2 learning rate override (default: sensitivity -> $SENSITIVITY_LR,
#                          optuna -> best lr found by the Optuna search).
# Env: AMP=1 (default; AMP=0 for fp32) autocast in every training stage, AMP_DTYPE=bf16|fp16 (default bf16;
#      use fp16 on NVIDIA cards older than RTX 30xx). STAGE1_EPOCHS/STAGE2_EPOCHS (default 30)
#      and the other knobs in common.sh are env-overridable too. ATTR "race"
#      (glaucoma only) runs the 3-class race baseline through the N-group evaluator
#      (--sens_attribute intersectional --group_cols Race), so its outputs land under
#      Optuna_Masks/intersectional/ -- run_experiments.sh moves them out after every run.
#
# Examples:
#   ./run_pipeline.sh all                                     # the real, recorded PAPILA run
#   ./run_pipeline.sh all gender --smoke-test --repo-dir ~/fairtune_papila/FairTune_smoketest
#   ./run_pipeline.sh stage1 age                              # resume just one stage later
#   ./run_pipeline.sh all intersectional --dataset glaucoma   # Harvard-GF, joint age x sex x race
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

REPO_DIR="$(dirname "$SCRIPT_DIR")"
DATASET="papila"
SEARCH_METHOD="optuna"
PAPILA_DIR="${HOME}/fairtune_papila/data/Papila"
HARVARD_DIR="${HOME}/fairtune_papila/data/harvard-dataset"
SPLITS_DIR=""   # resolved after REPO_DIR/DATASET are finalized, see below
SMOKE_TEST=0
FORCE_PREPROCESS=0
MASK_PATH_OVERRIDE=""
LR_OVERRIDE=""
ATTR="both"

usage() {
    sed -n '2,44p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

[[ $# -ge 1 ]] || usage
SUBCOMMAND="$1"; shift

if [[ $# -ge 1 && "$1" != --* ]]; then
    ATTR="$1"; shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --search-method) SEARCH_METHOD="$2"; shift 2 ;;
        --smoke-test) SMOKE_TEST=1; shift ;;
        --repo-dir) REPO_DIR="$2"; shift 2 ;;
        --papila-dir) PAPILA_DIR="$2"; shift 2 ;;
        --harvard-dir) HARVARD_DIR="$2"; shift 2 ;;
        --splits-dir) SPLITS_DIR="$2"; shift 2 ;;
        --force-preprocess) FORCE_PREPROCESS=1; shift ;;
        --mask-path) MASK_PATH_OVERRIDE="$2"; shift 2 ;;
        --lr) LR_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

case "$DATASET" in
    papila|glaucoma) ;;
    *) echo "--dataset must be papila or glaucoma (got: $DATASET)" >&2; exit 1 ;;
esac

case "$SEARCH_METHOD" in
    optuna|sensitivity) ;;
    *) echo "--search-method must be optuna or sensitivity (got: $SEARCH_METHOD)" >&2; exit 1 ;;
esac

if [[ -z "$SPLITS_DIR" ]]; then
    case "$DATASET" in
        papila) SPLITS_DIR="${REPO_DIR}/data_splits_papila" ;;
        glaucoma) SPLITS_DIR="${REPO_DIR}/data_splits_glaucoma" ;;
    esac
fi

# Per-dataset BATCH_SIZE default (only applied if not already set via env —
# see common.sh's comment; no longer needs to divide the split sizes).
if [[ -z "${BATCH_SIZE:-}" ]]; then
    case "$DATASET" in
        papila) BATCH_SIZE=42 ;;
        glaucoma) BATCH_SIZE=30 ;;
    esac
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: no active Python virtualenv (\$VIRTUAL_ENV unset)." >&2
    echo "Activate the project venv first: source .venv/bin/activate   (Windows Git Bash: source .venv/Scripts/activate)" >&2
    exit 1
fi

if [[ "$SMOKE_TEST" -eq 1 ]]; then
    STAGE1_EPOCHS=2
    STAGE1_NUM_TRIALS=2
    STAGE2_EPOCHS=2
    SENSITIVITY_K_SCHEDULE="1 4 36"
    echo "--smoke-test: using STAGE1_EPOCHS=2 STAGE1_NUM_TRIALS=2 STAGE2_EPOCHS=2 SENSITIVITY_K_SCHEDULE='${SENSITIVITY_K_SCHEDULE}'"
fi

case "$ATTR" in
    gender|age|intersectional) ATTRS=("$ATTR") ;;
    race)
        [[ "$DATASET" == "glaucoma" ]] || { echo "ATTR race is only available with --dataset glaucoma" >&2; exit 1; }
        ATTRS=("$ATTR") ;;
    both) ATTRS=(gender age) ;;
    *) echo "ATTR must be gender, age, race, intersectional, or both (got: $ATTR)" >&2; exit 1 ;;
esac

# attr_args ATTR -> sets SENS_ATTR (the --sens_attribute value) and EXTRA_ATTR_ARGS.
attr_args() {
    EXTRA_ATTR_ARGS=()
    if [[ "$1" == "race" ]]; then
        SENS_ATTR="intersectional"
        EXTRA_ATTR_ARGS=(--group_cols Race)
    else
        SENS_ATTR="$1"
    fi
}

# Class weights are only defined for papila (see search_mask.py); passing
# --compute_cw on glaucoma crashes the search/finetune scripts.
CW_ARGS=()
[[ "$DATASET" == "papila" ]] && CW_ARGS=(--compute_cw)
AMP_ARGS=()
[[ "$AMP" == "1" ]] && AMP_ARGS=(--amp --amp_dtype "$AMP_DTYPE")

do_preprocess() {
    # Glaucoma's image input (RNFLT_png/) is generated by preprocess_glaucoma.py from the raw
    # .npz files, so the committed CSVs alone are not enough on a fresh machine. Rerunning it
    # rewrites the CSVs identically (it uses Harvard-GF's shipped split, no randomness).
    local need_png=0
    if [[ "$DATASET" == "glaucoma" ]] && ! compgen -G "${HARVARD_DIR}/RNFLT_png/*.png" >/dev/null; then
        need_png=1
    fi
    if [[ -f "${SPLITS_DIR}/train.csv" && -f "${SPLITS_DIR}/val.csv" && -f "${SPLITS_DIR}/test.csv" \
          && "$FORCE_PREPROCESS" -eq 0 && "$need_png" -eq 0 ]]; then
        echo "Splits already exist at ${SPLITS_DIR} (use --force-preprocess to regenerate). Skipping."
        return
    fi
    if [[ "$DATASET" == "papila" ]]; then
        banner "PREPROCESS — building train/val/test.csv from PAPILA" "papila_dir=${PAPILA_DIR} -> ${SPLITS_DIR}"
        run_cmd "$PY_BIN" "${SCRIPT_DIR}/preprocess_papila.py" --papila-dir "$PAPILA_DIR" --out-dir "$SPLITS_DIR"
    else
        banner "PREPROCESS — building train/val/test.csv from Harvard-GF" "harvard_dir=${HARVARD_DIR} -> ${SPLITS_DIR}"
        run_cmd "$PY_BIN" "${SCRIPT_DIR}/preprocess_glaucoma.py" --harvard-dir "$HARVARD_DIR" --out-dir "$SPLITS_DIR"
    fi
}

do_config() {
    banner "CONFIG — filling config.yaml" "repo_dir=${REPO_DIR} dataset=${DATASET}"
    if [[ "$DATASET" == "papila" ]]; then
        IMG_DIR="${PAPILA_DIR}/FundusImages"
    else
        IMG_DIR="${HARVARD_DIR}/RNFLT_png"
    fi
    run_cmd "$PY_BIN" "${SCRIPT_DIR}/configure_fairtune.py" --repo-dir "$REPO_DIR" --dataset "$DATASET" \
        --img-dir "$IMG_DIR" \
        --train-csv "${SPLITS_DIR}/train.csv" \
        --val-csv "${SPLITS_DIR}/val.csv" \
        --test-csv "${SPLITS_DIR}/test.csv"
}

do_verify() {
    run_cmd "$PY_BIN" "${SCRIPT_DIR}/configure_fairtune.py" --repo-dir "$REPO_DIR" --dataset "$DATASET" --verify-only
}

do_verify_env() {
    banner "VERIFY ENV — CUDA / GPU sanity check"
    run_cmd "$PY_BIN" "${SCRIPT_DIR}/verify_env.py" --workers "$WORKERS"
}

do_stage1() {
    local attr="$1"
    attr_args "$attr"
    do_verify
    if [[ "$SEARCH_METHOD" == "optuna" ]]; then
        banner "STAGE 1 — mask search, Optuna/TPE (sens_attribute=${attr})" \
            "epochs=${STAGE1_EPOCHS} trials=${STAGE1_NUM_TRIALS} pruner=${PRUNER} batch_size=${BATCH_SIZE} workers=${WORKERS}"
        (
            cd "$REPO_DIR"
            run_cmd "$PY_BIN" search_mask.py \
                --model vit_base \
                --dataset "$DATASET" \
                --sens_attribute "$SENS_ATTR" "${EXTRA_ATTR_ARGS[@]}" \
                --tuning_method auto_peft1 \
                --objective_metric min_auc \
                "${CW_ARGS[@]}" "${AMP_ARGS[@]}" \
                --num_trials "$STAGE1_NUM_TRIALS" \
                --epochs "$STAGE1_EPOCHS" \
                --batch-size "$BATCH_SIZE" \
                --workers "$WORKERS" \
                --pruner "$PRUNER" \
                --disable_storage \
                --disable_checkpointing \
                --disable_plotting \
                --device cuda
        )
    else
        banner "STAGE 1 — mask search, sensitivity ranking (sens_attribute=${attr})" \
            "epochs=${STAGE1_EPOCHS} k_schedule=${SENSITIVITY_K_SCHEDULE} rank_method=${RANK_METHOD} batch_size=${BATCH_SIZE} workers=${WORKERS}"
        (
            cd "$REPO_DIR"
            # shellcheck disable=SC2086  # SENSITIVITY_K_SCHEDULE/SENSITIVITY_NUM_BATCHES are
            # deliberately unquoted below: --k_schedule/--sensitivity_num_batches take nargs="+",
            # so a multi-value env var (e.g. "1 4 36") must word-split into separate argv entries.
            run_cmd "$PY_BIN" search_mask_sensitivity.py \
                --model vit_base \
                --dataset "$DATASET" \
                --sens_attribute "$SENS_ATTR" "${EXTRA_ATTR_ARGS[@]}" \
                --tuning_method auto_peft1 \
                --objective_metric min_auc \
                "${CW_ARGS[@]}" "${AMP_ARGS[@]}" \
                --rank_method "$RANK_METHOD" \
                --sensitivity_normalize "$SENSITIVITY_NORMALIZE" \
                --sensitivity_seed "$SENSITIVITY_SEED" \
                --k_schedule $SENSITIVITY_K_SCHEDULE \
                ${SENSITIVITY_NUM_BATCHES:+--sensitivity_num_batches $SENSITIVITY_NUM_BATCHES} \
                --lr "$SENSITIVITY_LR" \
                --epochs "$STAGE1_EPOCHS" \
                --batch-size "$BATCH_SIZE" \
                --workers "$WORKERS" \
                --disable_checkpointing \
                --device cuda
        )
    fi
    local mask
    mask=$(discover_mask "$REPO_DIR" "$SENS_ATTR")
    echo "Discovered mask for ${attr}: ${mask}"
}

do_stage2() {
    local attr="$1"
    attr_args "$attr"
    do_verify
    local mask lr
    if [[ -n "$MASK_PATH_OVERRIDE" ]]; then
        mask="$MASK_PATH_OVERRIDE"
    else
        mask=$(discover_mask "$REPO_DIR" "$SENS_ATTR")
    fi
    if [[ -n "$LR_OVERRIDE" ]]; then
        lr="$LR_OVERRIDE"
    elif [[ "$SEARCH_METHOD" == "sensitivity" ]]; then
        lr="$SENSITIVITY_LR"
    else
        lr=$(best_lr "$REPO_DIR" "$DATASET" "$SENS_ATTR")
    fi
    banner "STAGE 2 — fine-tune with mask (sens_attribute=${attr})" \
        "mask=${mask} lr=${lr} epochs=${STAGE2_EPOCHS} batch_size=${BATCH_SIZE} workers=${WORKERS}"
    (
        cd "$REPO_DIR"
        run_cmd "$PY_BIN" finetune_with_mask.py \
            --model vit_base \
            --dataset "$DATASET" \
            --sens_attribute "$SENS_ATTR" "${EXTRA_ATTR_ARGS[@]}" \
            --tuning_method auto_peft1 \
            --mask_path "$mask" \
            --lr "$lr" \
            --cal_equiodds \
            --use_metric auc \
            --objective_metric min_auc \
            "${CW_ARGS[@]}" "${AMP_ARGS[@]}" \
            --epochs "$STAGE2_EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --workers "$WORKERS" \
            --device cuda
    )
}

do_results() {
    banner "RESULTS"
    "$PY_BIN" - "$REPO_DIR" <<'PYEOF'
import glob
import sys
import pandas as pd

repo_dir = sys.argv[1]
files = glob.glob(f"{repo_dir}/**/RESULTS_*.csv", recursive=True)
if not files:
    print("No RESULTS_*.csv found yet.")
else:
    for rf in files:
        print(f"\n=== {rf} ===")
        print(pd.read_csv(rf).to_string(index=False))
PYEOF
}

case "$SUBCOMMAND" in
    preprocess)
        do_preprocess
        ;;
    config)
        do_preprocess
        do_config
        ;;
    verify)
        do_verify_env
        do_verify
        ;;
    stage1)
        for attr in "${ATTRS[@]}"; do do_stage1 "$attr"; done
        ;;
    stage2)
        for attr in "${ATTRS[@]}"; do do_stage2 "$attr"; done
        ;;
    all)
        do_verify_env
        do_preprocess
        do_config
        for attr in "${ATTRS[@]}"; do do_stage1 "$attr"; done
        for attr in "${ATTRS[@]}"; do do_stage2 "$attr"; done
        do_results
        banner "DONE" "All stages completed with no errors."
        ;;
    *)
        echo "Unknown subcommand: $SUBCOMMAND" >&2
        usage
        ;;
esac
