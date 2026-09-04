#!/usr/bin/env bash
# Single entry point for the FairTune/PAPILA pipeline. Replaces the old Colab
# notebook's click-every-cell workflow: this is the "press play and walk
# away" script meant to be screen-recorded directly.
#
# Usage:
#   ./run_pipeline.sh {all|preprocess|config|verify|stage1|stage2} [gender|age|both] [options]
#
# Options:
#   --smoke-test          Use tiny epochs/trials (2/2/2) for a fast dry run.
#   --repo-dir PATH        FairTune repo clone to run against (default: this script's own repo).
#   --papila-dir PATH      Raw Papila/ folder, containing ClinicalData/ + FundusImages/
#                          (default: $HOME/fairtune_papila/data/Papila).
#   --splits-dir PATH      Where train.csv/val.csv/test.csv live or get written
#                          (default: REPO_DIR/data_splits — already committed to the repo).
#   --force-preprocess     Regenerate the CSVs even if they already exist.
#
# Examples:
#   ./run_pipeline.sh all                                     # the real, recorded run
#   ./run_pipeline.sh all gender --smoke-test --repo-dir ~/fairtune_papila/FairTune_smoketest
#   ./run_pipeline.sh stage1 age                              # resume just one stage later
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

REPO_DIR="$(dirname "$SCRIPT_DIR")"
PAPILA_DIR="${HOME}/fairtune_papila/data/Papila"
SPLITS_DIR=""   # resolved after REPO_DIR is finalized, see below
SMOKE_TEST=0
FORCE_PREPROCESS=0
ATTR="both"

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

[[ $# -ge 1 ]] || usage
SUBCOMMAND="$1"; shift

if [[ $# -ge 1 && "$1" != --* ]]; then
    ATTR="$1"; shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test) SMOKE_TEST=1; shift ;;
        --repo-dir) REPO_DIR="$2"; shift 2 ;;
        --papila-dir) PAPILA_DIR="$2"; shift 2 ;;
        --splits-dir) SPLITS_DIR="$2"; shift 2 ;;
        --force-preprocess) FORCE_PREPROCESS=1; shift ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

[[ -z "$SPLITS_DIR" ]] && SPLITS_DIR="${REPO_DIR}/data_splits"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: no active Python virtualenv (\$VIRTUAL_ENV unset)." >&2
    echo "Activate the project venv first: source .venv/Scripts/activate" >&2
    exit 1
fi

if [[ "$SMOKE_TEST" -eq 1 ]]; then
    STAGE1_EPOCHS=2
    STAGE1_NUM_TRIALS=2
    STAGE2_EPOCHS=2
    echo "--smoke-test: using STAGE1_EPOCHS=2 STAGE1_NUM_TRIALS=2 STAGE2_EPOCHS=2"
fi

case "$ATTR" in
    gender|age) ATTRS=("$ATTR") ;;
    both) ATTRS=(gender age) ;;
    *) echo "ATTR must be gender, age, or both (got: $ATTR)" >&2; exit 1 ;;
esac

do_preprocess() {
    if [[ -f "${SPLITS_DIR}/train.csv" && -f "${SPLITS_DIR}/val.csv" && -f "${SPLITS_DIR}/test.csv" && "$FORCE_PREPROCESS" -eq 0 ]]; then
        echo "Splits already exist at ${SPLITS_DIR} (use --force-preprocess to regenerate). Skipping."
        return
    fi
    banner "PREPROCESS — building train/val/test.csv from PAPILA" "papila_dir=${PAPILA_DIR} -> ${SPLITS_DIR}"
    run_cmd "$PY_BIN" "${SCRIPT_DIR}/preprocess_papila.py" --papila-dir "$PAPILA_DIR" --out-dir "$SPLITS_DIR"
}

do_config() {
    banner "CONFIG — filling config.yaml" "repo_dir=${REPO_DIR}"
    run_cmd "$PY_BIN" "${SCRIPT_DIR}/configure_fairtune.py" --repo-dir "$REPO_DIR" \
        --img-dir "${PAPILA_DIR}/FundusImages" \
        --train-csv "${SPLITS_DIR}/train.csv" \
        --val-csv "${SPLITS_DIR}/val.csv" \
        --test-csv "${SPLITS_DIR}/test.csv"
}

do_verify() {
    run_cmd "$PY_BIN" "${SCRIPT_DIR}/configure_fairtune.py" --repo-dir "$REPO_DIR" --verify-only
}

do_verify_env() {
    banner "VERIFY ENV — CUDA / GPU sanity check"
    run_cmd "$PY_BIN" "${SCRIPT_DIR}/verify_env.py" --workers "$WORKERS"
}

do_stage1() {
    local attr="$1"
    do_verify
    banner "STAGE 1 — mask search (sens_attribute=${attr})" \
        "epochs=${STAGE1_EPOCHS} trials=${STAGE1_NUM_TRIALS} pruner=${PRUNER} batch_size=${BATCH_SIZE} workers=${WORKERS}"
    (
        cd "$REPO_DIR"
        run_cmd "$PY_BIN" search_mask.py \
            --model vit_base \
            --dataset papila \
            --sens_attribute "$attr" \
            --tuning_method auto_peft1 \
            --objective_metric min_auc \
            --compute_cw \
            --num_trials "$STAGE1_NUM_TRIALS" \
            --epochs "$STAGE1_EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --workers "$WORKERS" \
            --pruner "$PRUNER" \
            --disable_storage \
            --disable_checkpointing \
            --device cuda
    )
    local mask
    mask=$(discover_mask "$REPO_DIR" "$attr")
    echo "Discovered mask for ${attr}: ${mask}"
}

do_stage2() {
    local attr="$1"
    do_verify
    local mask
    mask=$(discover_mask "$REPO_DIR" "$attr")
    banner "STAGE 2 — fine-tune with mask (sens_attribute=${attr})" \
        "mask=${mask} epochs=${STAGE2_EPOCHS} batch_size=${BATCH_SIZE} workers=${WORKERS}"
    (
        cd "$REPO_DIR"
        run_cmd "$PY_BIN" finetune_with_mask.py \
            --model vit_base \
            --dataset papila \
            --sens_attribute "$attr" \
            --tuning_method auto_peft1 \
            --mask_path "$mask" \
            --cal_equiodds \
            --use_metric auc \
            --compute_cw \
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
