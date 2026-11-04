#!/usr/bin/env bash
# Experiment driver: runs the grid from EXPERIMENT_CHECKLIST.md with per-run
# wall-clock timing and isolated outputs, on top of run_pipeline.sh.
#
# Usage:
#   ./run_experiments.sh PHASE [options]
#     PHASE: phase1 | ablation | phase2 | calibration | all
#       phase1       budget-matched TPE (10 trials) vs sensitivity (worst_group_sensitivity),
#                    single attributes. papila: gender age; glaucoma: gender age race.
#       ablation     sensitivity with rank_method random / erm_sensitivity (seed 0).
#       phase2       intersectional sensitivity search, then the core ablation: Stage 2 on the
#                    intersectional groups with each Phase-1 single-attribute mask, plus
#                    full-fine-tune (all-ones) and head-only (all-zeros) mask baselines.
#       calibration  ONE full-budget TPE (50 trials) on gender, used to extrapolate
#                    full-budget cost for the other cells (labelled "extrapolated").
#   --dataset {papila|glaucoma|both}   (default both)
#   --seeds-papila "0 1 2"  --seeds-glaucoma "0 1"
#   --smoke-test           tiny epochs/trials (passed through to run_pipeline.sh)
#   --dry-run              print the commands, run nothing
#   --runs-dir DIR         default: <repo>/runs
#
# Layout: runs/<dataset>/<label>/<attr>/seed<N>/{outputs...,DONE}; every run also
# appends one row to runs/timings.csv. Finished runs (DONE present) are skipped,
# so the script is resumable. Timing rows are only trustworthy when one run is
# on the GPU at a time: do not run two copies of this script concurrently for
# the seed-0 (timing) runs.
#
# TPE (search_mask.py) has no seed flag: its "seeds" are independent repeats.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PIPE="${SCRIPT_DIR}/run_pipeline.sh"

PHASE="${1:-}"; [[ -n "$PHASE" ]] || { sed -n 2,30p "$0"; exit 1; }; shift
DATASETS="papila glaucoma"
SEEDS_PAPILA="0 1 2"
SEEDS_GLAUCOMA="0 1"
SMOKE=()
DRY=0
RUNS_DIR="${REPO_DIR}/runs"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) [[ "$2" == both ]] && DATASETS="papila glaucoma" || DATASETS="$2"; shift 2 ;;
        --seeds-papila) SEEDS_PAPILA="$2"; shift 2 ;;
        --seeds-glaucoma) SEEDS_GLAUCOMA="$2"; shift 2 ;;
        --smoke-test) SMOKE=(--smoke-test); shift ;;
        --dry-run) DRY=1; shift ;;
        --runs-dir) RUNS_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$RUNS_DIR"
TIMINGS="${RUNS_DIR}/timings.csv"
[[ -f "$TIMINGS" || $DRY -eq 1 ]] || echo "dataset,label,attr,seed,stage1_seconds,stage2_seconds,total_seconds,num_trials_or_k" > "$TIMINGS"

attrs_for() { [[ "$1" == papila ]] && echo "gender age" || echo "gender age race"; }
seeds_for() { [[ "$1" == papila ]] && echo "$SEEDS_PAPILA" || echo "$SEEDS_GLAUCOMA"; }
# Outputs live under vit_base/<dataset>/ and use the --sens_attribute value in their names
# ("race" runs as intersectional + --group_cols Race, see run_pipeline.sh).
sens_of() { [[ "$1" == race ]] && echo intersectional || echo "$1"; }

say() { echo "[$(date +%H:%M:%S)] $*"; }
run() { say "+ $*"; [[ $DRY -eq 1 ]] || "$@"; }

# collect DATASET ATTR DEST — move this run's outputs out of the repo tree so the
# next run starts clean (RESULTS csvs append, discover_mask picks the newest .npy).
collect() {
    local ds="$1" sens="$2" dest="$3" base="${REPO_DIR}/vit_base/${ds}"
    [[ $DRY -eq 1 ]] && { say "(collect ${base} -> ${dest})"; return; }
    mkdir -p "$dest"
    shopt -s nullglob
    for p in "$base"/RESULTS_*.csv "$base"/test_predictions_*.npz "$base/Optuna_Masks/$sens" "$base/Sensitivity_Scores/$sens" \
             "$base/Sensitivity_Results/$sens" "$base"/Optuna\ Results "$base"/Optuna\ Run\ Stats; do
        [[ -e "$p" ]] && mv "$p" "$dest/"
    done
    shopt -u nullglob
    touch "$dest/DONE"
}

# preclean DATASET ATTR — park leftovers from manual runs so they cannot leak in.
preclean() {
    local ds="$1" sens="$2" base="${REPO_DIR}/vit_base/${ds}" stale="${RUNS_DIR}/_stale/$(date +%s)"
    [[ $DRY -eq 1 ]] && return
    shopt -s nullglob
    for p in "$base/Optuna_Masks/$sens" "$base"/RESULTS_*.csv "$base"/test_predictions_*.npz; do
        [[ -e "$p" ]] && { mkdir -p "$stale"; mv "$p" "$stale/"; }
    done
    shopt -u nullglob
}

# search_run DATASET LABEL ATTR SEED METHOD RANK TRIALS
search_run() {
    local ds="$1" label="$2" attr="$3" seed="$4" method="$5" rank="$6" trials="$7"
    local sens dest; sens="$(sens_of "$attr")"
    dest="${RUNS_DIR}/${ds}/${label}/${attr}/seed${seed}"
    [[ -f "$dest/DONE" ]] && { say "skip (done): $dest"; return; }
    preclean "$ds" "$sens"
    local t0 t1 t2
    t0=$(date +%s)
    SENSITIVITY_SEED="$seed" RANK_METHOD="$rank" STAGE1_NUM_TRIALS="$trials" \
        run "$PIPE" stage1 "$attr" --dataset "$ds" --search-method "$method" "${SMOKE[@]}"
    t1=$(date +%s)
    SENSITIVITY_SEED="$seed" RANK_METHOD="$rank" \
        run "$PIPE" stage2 "$attr" --dataset "$ds" --search-method "$method" "${SMOKE[@]}"
    t2=$(date +%s)
    collect "$ds" "$sens" "$dest"
    [[ $DRY -eq 1 ]] || echo "${ds},${label},${attr},${seed},$((t1-t0)),$((t2-t1)),$((t2-t0)),${trials}" >> "$TIMINGS"
}

# stage2_with_mask DATASET LABEL ATTR SEED MASK  — Stage 2 on ATTR's groups with a given mask.
stage2_with_mask() {
    local ds="$1" label="$2" attr="$3" seed="$4" mask="$5"
    local sens dest; sens="$(sens_of "$attr")"
    dest="${RUNS_DIR}/${ds}/${label}/${attr}/seed${seed}"
    [[ -f "$dest/DONE" ]] && { say "skip (done): $dest"; return; }
    preclean "$ds" "$sens"
    local t0 t1; t0=$(date +%s)
    # sensitivity method => Stage 2 uses $SENSITIVITY_LR for every borrowed/baseline mask, so
    # all intersectional comparisons share one lr.
    run "$PIPE" stage2 "$attr" --dataset "$ds" --search-method sensitivity --mask-path "$mask" "${SMOKE[@]}"
    t1=$(date +%s)
    collect "$ds" "$sens" "$dest"
    [[ $DRY -eq 1 ]] || echo "${ds},${label},${attr},${seed},0,$((t1-t0)),$((t1-t0)),mask=$(basename "$mask")" >> "$TIMINGS"
}

make_const_mask() { # PATH VALUE
    [[ $DRY -eq 1 ]] && { say "(write const mask $1 = $2)"; return; }
    mkdir -p "$(dirname "$1")"
    python3 -c "import numpy as np,sys; np.save(sys.argv[1], np.full(36, int(sys.argv[2]), dtype=np.int8))" "$1" "$2"
}

phase1() {
    for ds in $DATASETS; do for attr in $(attrs_for "$ds"); do for seed in $(seeds_for "$ds"); do
        search_run "$ds" tpe10 "$attr" "$seed" optuna worst_group_sensitivity 10
        search_run "$ds" sens_wg "$attr" "$seed" sensitivity worst_group_sensitivity 10
    done; done; done
}

ablation() {
    for ds in $DATASETS; do for attr in $(attrs_for "$ds"); do
        search_run "$ds" sens_random "$attr" 0 sensitivity random 10
        search_run "$ds" sens_erm "$attr" 0 sensitivity erm_sensitivity 10
    done; done
}

calibration() {
    for ds in $DATASETS; do
        search_run "$ds" tpe50 gender 0 optuna worst_group_sensitivity 50
    done
}

phase2() {
    for ds in $DATASETS; do for seed in $(seeds_for "$ds"); do
        search_run "$ds" sens_wg intersectional "$seed" sensitivity worst_group_sensitivity 10
        # Core ablation: single-attribute-searched masks evaluated on the intersectional groups.
        for attr in $(attrs_for "$ds"); do
            local sens maskdir mask
            sens="$(sens_of "$attr")"
            maskdir="${RUNS_DIR}/${ds}/sens_wg/${attr}/seed${seed}/${sens}"
            mask=$(ls -t "$maskdir"/*.npy 2>/dev/null | head -1 || true)
            if [[ -z "$mask" && $DRY -eq 0 ]]; then
                echo "WARNING: no Phase-1 mask at $maskdir (run phase1 first); skipping xmask_${attr}" >&2; continue
            fi
            stage2_with_mask "$ds" "xmask_${attr}" intersectional "$seed" "${mask:-$maskdir/MASK.npy}"
        done
        make_const_mask "${RUNS_DIR}/_const_masks/ones.npy" 1
        make_const_mask "${RUNS_DIR}/_const_masks/zeros.npy" 0
        stage2_with_mask "$ds" base_full_ft intersectional "$seed" "${RUNS_DIR}/_const_masks/ones.npy"
        stage2_with_mask "$ds" base_head_only intersectional "$seed" "${RUNS_DIR}/_const_masks/zeros.npy"
    done; done
}

case "$PHASE" in
    phase1) phase1 ;;
    ablation) ablation ;;
    phase2) phase2 ;;
    calibration) calibration ;;
    all) phase1; ablation; phase2; calibration ;;
    *) echo "Unknown phase: $PHASE" >&2; exit 1 ;;
esac
say "done: $PHASE"
