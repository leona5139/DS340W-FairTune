#!/usr/bin/env bash
# Experiment driver (see RUN_GUIDE.md): per-run wall-clock timing and isolated
# outputs, on top of run_pipeline.sh.
#
# Usage:
#   ./run_experiments.sh PHASE [options]
#     PHASE:
#       exp1           Original FairTune (TPE, search_mask.py), Stage 1 only, a SAMPLE of
#                      $TPE_SAMPLE_TRIALS trials (default 5) per (dataset, attr), seed 0.
#                      summarize_runs.py extrapolates it to $TPE_FULL_TRIALS (default 200).
#                      papila: gender age intersectional; glaucoma: gender age race intersectional.
#       exp2           Sensitivity search (worst_group_sensitivity) + Stage 2, single attributes,
#                      every seed. papila: gender age; glaucoma: gender age race.
#       exp3           Sensitivity search + Stage 2 on intersectional groups, every seed
#                      (papila: age x sex = 4 groups; glaucoma: age x sex x race = 12).
#       all            exp1, then exp2+exp3 interleaved seed-major (all seed-0 runs first,
#                      so every runtime number is in before the extra seeds start).
#       exp3_ablation  (optional, needs exp2+exp3) Stage 2 on the intersectional groups with
#                      each exp2 single-attribute mask, plus full-FT (all-ones) and head-only
#                      (all-zeros) masks.
#       ablation       (optional) sensitivity with rank_method random / erm_sensitivity, seed 0.
#   --dataset {papila|glaucoma|both}   (default both)
#   --seeds-papila "0 1 2"  --seeds-glaucoma "0"   (defaults)
#   --smoke-test           tiny epochs/trials (passed through to run_pipeline.sh)
#   --dry-run              print the commands, run nothing
#   --runs-dir DIR         default: <repo>/runs
#
# Layout: runs/<dataset>/<label>/<attr>/seed<N>/{outputs...,DONE}; every run also
# appends one row to runs/timings.csv. Finished runs (DONE present) are skipped,
# so the script is resumable and seeds can be added later. Timing rows are only
# trustworthy when one run is on the GPU at a time: never run two copies of this
# script at once.
#
# TPE (search_mask.py) has no seed flag: its "seeds" would be independent repeats.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PIPE="${SCRIPT_DIR}/run_pipeline.sh"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"   # PY_BIN, TPE_SAMPLE_TRIALS

ORIG_ARGS=("$@")
PHASE="${1:-}"; [[ -n "$PHASE" ]] || { sed -n 2,35p "$0"; exit 1; }; shift

# Keep the machine awake for the whole run (Linux + systemd only; a no-op elsewhere,
# e.g. Windows Git Bash). A system suspend mid-training kills the job, and on AMD/ROCm
# the GPU reset that follows can leave compute unusable until a reboot.
# FAIRTUNE_NO_INHIBIT=1 skips this.
if [[ -z "${FAIRTUNE_INHIBITED:-}" && -z "${FAIRTUNE_NO_INHIBIT:-}" && " ${ORIG_ARGS[*]} " != *" --dry-run "* ]]; then
    INHIBIT_BIN="$(command -v systemd-inhibit || true)"
    [[ -z "$INHIBIT_BIN" && -x /run/current-system/sw/bin/systemd-inhibit ]] && INHIBIT_BIN=/run/current-system/sw/bin/systemd-inhibit
    if [[ -n "$INHIBIT_BIN" ]] && "$INHIBIT_BIN" --what=sleep:idle --who=FairTune --why=probe --mode=block true 2>/dev/null; then
        export FAIRTUNE_INHIBITED=1
        echo "[$(date +%H:%M:%S)] holding a sleep/idle inhibitor lock for this run (systemd-inhibit)"
        exec "$INHIBIT_BIN" --what=sleep:idle --who=FairTune --why="training run: $*" --mode=block "$0" "${ORIG_ARGS[@]}"
    fi
fi

DATASETS="papila glaucoma"
SEEDS_PAPILA="0 1 2"
SEEDS_GLAUCOMA="0"
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

# collect DATASET SENS DEST — move this run's outputs out of the repo tree so the
# next run starts clean (RESULTS csvs append, discover_mask picks the newest .npy).
collect() {
    local ds="$1" sens="$2" dest="$3" base="${REPO_DIR}/vit_base/${ds}"
    [[ $DRY -eq 1 ]] && { say "(collect ${base} -> ${dest})"; return; }
    mkdir -p "$dest"
    shopt -s nullglob
    for p in "$base"/RESULTS_*.csv "$base"/test_predictions_*.npz "$base"/Optuna\ Results "$base"/Optuna\ Run\ Stats; do
        [[ -e "$p" ]] && mv "$p" "$dest/"
    done
    # These three are all named after the attribute, so keep their parent folder.
    local sub
    for sub in Optuna_Masks Sensitivity_Scores Sensitivity_Results; do
        [[ -e "$base/$sub/$sens" ]] && { mkdir -p "$dest/$sub"; mv "$base/$sub/$sens" "$dest/$sub/"; }
    done
    shopt -u nullglob
    touch "$dest/DONE"
}

# preclean DATASET SENS — park leftovers from manual runs so they cannot leak in.
preclean() {
    local ds="$1" sens="$2" base="${REPO_DIR}/vit_base/${ds}" stale="${RUNS_DIR}/_stale/$(date +%s)"
    [[ $DRY -eq 1 ]] && return
    shopt -s nullglob
    for p in "$base"/RESULTS_*.csv "$base"/test_predictions_*.npz "$base"/Optuna\ Results "$base"/Optuna\ Run\ Stats; do
        [[ -e "$p" ]] && { mkdir -p "$stale"; mv "$p" "$stale/"; }
    done
    local sub
    for sub in Optuna_Masks Sensitivity_Scores Sensitivity_Results; do
        [[ -e "$base/$sub/$sens" ]] && { mkdir -p "$stale/$sub"; mv "$base/$sub/$sens" "$stale/$sub/"; }
    done
    shopt -u nullglob
}

# search_run DATASET LABEL ATTR SEED METHOD RANK TRIALS [stage1_only]
search_run() {
    local ds="$1" label="$2" attr="$3" seed="$4" method="$5" rank="$6" trials="$7" s1only="${8:-}"
    local sens dest; sens="$(sens_of "$attr")"
    dest="${RUNS_DIR}/${ds}/${label}/${attr}/seed${seed}"
    [[ -f "$dest/DONE" ]] && { say "skip (done): $dest"; return; }
    preclean "$ds" "$sens"
    local t0 t1 t2
    t0=$(date +%s)
    SENSITIVITY_SEED="$seed" RANK_METHOD="$rank" STAGE1_NUM_TRIALS="$trials" \
        run "$PIPE" stage1 "$attr" --dataset "$ds" --search-method "$method" "${SMOKE[@]}"
    t1=$(date +%s)
    if [[ "$s1only" == stage1_only ]]; then
        t2=$t1
    else
        SENSITIVITY_SEED="$seed" RANK_METHOD="$rank" \
            run "$PIPE" stage2 "$attr" --dataset "$ds" --search-method "$method" "${SMOKE[@]}"
        t2=$(date +%s)
    fi
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
    "$PY_BIN" -c "import numpy as np,sys; np.save(sys.argv[1], np.full(36, int(sys.argv[2]), dtype=np.int8))" "$1" "$2"
}

# Union of all datasets' seeds, ascending, for seed-major ordering.
all_seeds() { for ds in $DATASETS; do seeds_for "$ds"; echo; done | tr ' ' '\n' | grep -v '^$' | sort -n -u; }
has_seed() { [[ " $(seeds_for "$1") " == *" $2 "* ]]; }

exp1() {
    for ds in $DATASETS; do for attr in $(attrs_for "$ds") intersectional; do
        search_run "$ds" tpe_sample "$attr" 0 optuna worst_group_sensitivity "$TPE_SAMPLE_TRIALS" stage1_only
    done; done
}

# sens_grid WHICH — WHICH: single | intersectional | both. Seed-major: seed 0 of every
# dataset/attr first, then seed 1, ...
sens_grid() {
    local which="$1" seed ds attrs
    for seed in $(all_seeds); do for ds in $DATASETS; do
        has_seed "$ds" "$seed" || continue
        case "$which" in
            single) attrs="$(attrs_for "$ds")" ;;
            intersectional) attrs="intersectional" ;;
            both) attrs="$(attrs_for "$ds") intersectional" ;;
        esac
        for attr in $attrs; do
            search_run "$ds" sens_wg "$attr" "$seed" sensitivity worst_group_sensitivity k10
        done
    done; done
}

exp3_ablation() {
    for ds in $DATASETS; do for seed in $(seeds_for "$ds"); do
        # Core ablation: single-attribute-searched masks evaluated on the intersectional groups.
        for attr in $(attrs_for "$ds"); do
            local sens maskdir mask
            sens="$(sens_of "$attr")"
            maskdir="${RUNS_DIR}/${ds}/sens_wg/${attr}/seed${seed}/Optuna_Masks/${sens}"
            mask=$(ls -t "$maskdir"/*.npy 2>/dev/null | head -1 || true)
            if [[ -z "$mask" && $DRY -eq 0 ]]; then
                echo "WARNING: no exp2 mask at $maskdir (run exp2 first); skipping xmask_${attr}" >&2; continue
            fi
            stage2_with_mask "$ds" "xmask_${attr}" intersectional "$seed" "${mask:-$maskdir/MASK.npy}"
        done
        make_const_mask "${RUNS_DIR}/_const_masks/ones.npy" 1
        make_const_mask "${RUNS_DIR}/_const_masks/zeros.npy" 0
        stage2_with_mask "$ds" base_full_ft intersectional "$seed" "${RUNS_DIR}/_const_masks/ones.npy"
        stage2_with_mask "$ds" base_head_only intersectional "$seed" "${RUNS_DIR}/_const_masks/zeros.npy"
    done; done
}

ablation() {
    for ds in $DATASETS; do for attr in $(attrs_for "$ds"); do
        search_run "$ds" sens_random "$attr" 0 sensitivity random k10
        search_run "$ds" sens_erm "$attr" 0 sensitivity erm_sensitivity k10
    done; done
}

case "$PHASE" in
    exp1) exp1 ;;
    exp2) sens_grid single ;;
    exp3) sens_grid intersectional ;;
    all) exp1; sens_grid both ;;
    exp3_ablation) exp3_ablation ;;
    ablation) ablation ;;
    *) echo "Unknown phase: $PHASE (expected exp1|exp2|exp3|all|exp3_ablation|ablation)" >&2; exit 1 ;;
esac
say "done: $PHASE"
