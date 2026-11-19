#!/usr/bin/env bash
# Recreate the paper's tables and figures (paper_results/).
#
# Usage:
#   ./orchestration/reproduce_paper.sh                 # DEFAULT: rebuild tables/figures from the committed runs/ (minutes, no GPU)
#   ./orchestration/reproduce_paper.sh --full          # re-run EVERY experiment on your GPU, then rebuild (days; see below)
#   ./orchestration/reproduce_paper.sh --full --dry-run   # print the experiment commands only
#
# Options:
#   --runs-dir DIR   where runs live / are written (default: <repo>/runs). With --full, use a fresh
#                    directory (e.g. runs_repro) to avoid mixing with the committed results.
#   --out DIR        where tables/figures go (default: <repo>/paper_results; use another dir to keep the originals)
#   --n-boot N       bootstrap resamples (default 2000, as in the paper)
#   --dry-run        with --full: only print commands
#
# --full runs, in order (each step skips cells already marked DONE, so it is resumable):
#   all           = exp1 (original FairTune, 5-trial TPE sample per attr -> Table I extrapolation)
#                   + exp2/exp3 (sensitivity search, single + intersectional; PAPILA seeds 0-9, Harvard-GF seeds 0-2)
#   exp3_ablation Stage 2 with single-attribute masks, full fine-tune and head-only masks
#   ablation_x    random / ERM ranking ablations (PAPILA, seed 0)
#   tpe_full      measured 200-trial original FairTune on PAPILA intersectional (~10 h, not resumable mid-search)
#   paired        fixed-order Stage 2 re-runs used for every paired bootstrap comparison (Tables III-V, Fig. 2)
# Expect roughly 2-3 days on an RX 9060 XT-class GPU (PAPILA ~18 min/cell, Harvard-GF ~1.6 h/cell).
# Keep the machine awake and run nothing else on the GPU (timings are a result).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
RUNS_DIR="${REPO_DIR}/runs"; OUT_DIR="${REPO_DIR}/paper_results"; NBOOT=2000
FULL=0; DRY=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --full) FULL=1; shift ;;
        --runs-dir) RUNS_DIR="$2"; shift 2 ;;
        --out) OUT_DIR="$2"; shift 2 ;;
        --n-boot) NBOOT="$2"; shift 2 ;;
        --dry-run) DRY=(--dry-run); shift ;;
        -h|--help) sed -n 2,26p "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done
[[ -n "${VIRTUAL_ENV:-}" ]] || { echo "ERROR: activate the project venv first (see README)." >&2; exit 1; }
source "${SCRIPT_DIR}/common.sh"   # PY_BIN

if [[ $FULL -eq 1 ]]; then
    EXP="${SCRIPT_DIR}/run_experiments.sh"
    SEEDS_P="0 1 2 3 4 5 6 7 8 9"; SEEDS_G="0 1 2"
    S=(--runs-dir "$RUNS_DIR" --seeds-papila "$SEEDS_P" --seeds-glaucoma "$SEEDS_G" "${DRY[@]}")
    "$EXP" all "${S[@]}"          # exp1 + exp2 + exp3
    "$EXP" exp3_ablation "${S[@]}"
    "$EXP" ablation_x --dataset papila "${S[@]}"
    "$EXP" tpe_full --dataset papila "${S[@]}"
    "$EXP" paired "${S[@]}"
    [[ ${#DRY[@]} -eq 0 ]] || { echo "dry run: skipping table/figure build"; exit 0; }
    if [[ ! -f "$RUNS_DIR/tpe_full_pauses.txt" && -f "${REPO_DIR}/runs/tpe_full_pauses.txt" ]]; then
        echo "note: no pause log for tpe_full in $RUNS_DIR; its hours are not pause-corrected"
    fi
fi

[[ -d "$RUNS_DIR" ]] || { echo "ERROR: runs dir not found: $RUNS_DIR" >&2; exit 1; }
"$PY_BIN" "${SCRIPT_DIR}/make_paper_results.py" --runs-dir "$RUNS_DIR" --out "$OUT_DIR" --n-boot "$NBOOT"
echo "Done. Tables: $OUT_DIR/tables  Figures: $OUT_DIR/figures  Data: $OUT_DIR/data"
