# How to run the experiments (order matters)

Two experiments, run on the AMD RX 9060 XT (WSL2/ROCm, see `WSL_ROCM_SETUP.md`):

1. **Speed-up**: sensitivity search (`search_mask_sensitivity.py`) vs FairTune's TPE search (`search_mask.py`), single attributes.
2. **Intersectional**: the same fast method with joint groups (PAPILA age x sex = 4 groups; Harvard-GF age x sex x race = 12 groups).

Everything below runs from the repo root with the project venv active (`source .venv/bin/activate`).
`run_pipeline.sh` refuses to start without `$VIRTUAL_ENV`.

## Step 0 - Environment and data (once)

1. Follow `WSL_ROCM_SETUP.md` (ROCm + ROCm PyTorch wheels), then `pip install -r orchestration/requirements.txt`.
2. Put raw data where the scripts expect it (or pass `--papila-dir` / `--harvard-dir`):
   - `~/fairtune_papila/data/Papila/` (`ClinicalData/`, `FundusImages/`)
   - `~/fairtune_papila/data/harvard-dataset/` (`Training/ Validation/ Test/`, plus `RNFLT_png/` from `preprocess_glaucoma.py`)
3. `python3 orchestration/verify_env.py` - GPU visible, DataLoader does not hang (if it does, set `WORKERS=0`).
4. Fill `config.yaml` for each dataset: `./orchestration/run_pipeline.sh config --dataset papila` and `... --dataset glaucoma`. Do not hand-edit it.
5. Splits (`data_splits_papila/`, `data_splits_glaucoma/`) are already committed; do not regenerate unless needed.

## Step 1 - Smoke tests (minutes; use a disposable clone for the smoke run so a 2-epoch mask never lands in the real tree)

```
git clone <this repo> ~/fairtune_papila/FairTune_smoketest
./orchestration/run_pipeline.sh all gender --smoke-test --search-method sensitivity --repo-dir ~/fairtune_papila/FairTune_smoketest
./orchestration/run_pipeline.sh all intersectional --smoke-test --search-method sensitivity --repo-dir ~/fairtune_papila/FairTune_smoketest
./orchestration/run_pipeline.sh all intersectional --smoke-test --search-method sensitivity --dataset glaucoma --repo-dir ~/fairtune_papila/FairTune_smoketest
./orchestration/run_pipeline.sh all race --smoke-test --search-method sensitivity --dataset glaucoma --repo-dir ~/fairtune_papila/FairTune_smoketest
```
Check: Stage 2 test AUC is not 0.5, `RESULTS_intersectional_groups_min_auc.csv` has 4 / 12 rows, NaN-group warnings are printed (not silent).

## Step 2 - Calibrate and validate the speed-ups (about 1 hour)

1. **Time one real solo run per dataset** (no smoke flag, nothing else on the GPU) and compare to the estimates in `EXPERIMENT_CHECKLIST.md`:
   `./orchestration/run_experiments.sh phase1 --dataset papila --seeds-papila 0` (stop it after the first cell if you only want the timing).
2. **Image-cache check (PAPILA):** run one search with `FAIRTUNE_IMG_CACHE_SHORT_SIDE=0` (cache off) and one with it on. AUC should agree within seed noise; epoch time should drop.
3. **bf16 check (optional):** repeat one run with `AMP=1`. Use it for the whole grid only if AUC agrees within seed noise with fp32.
Decide the settings now and keep them identical for every run that follows (TPE and sensitivity alike).

## Step 3 - Experiment 1: speed-up (single attributes)

```
./orchestration/run_experiments.sh phase1        # TPE (10 trials) vs sensitivity; papila: gender age x3 seeds, glaucoma: gender age race x2 seeds
./orchestration/run_experiments.sh calibration   # ONE 50-trial TPE per dataset (gender) for the extrapolated full-budget cost; run overnight
./orchestration/run_experiments.sh ablation      # rank_method random / erm_sensitivity, seed 0
python3 orchestration/summarize_runs.py          # tables in runs/summary_*.csv
```
Rules: keep the seed-0 runs solo on the GPU (timings.csv is the headline metric). The script is resumable (finished runs have a `DONE` file); `--dry-run` lists commands; `--seeds-papila "0 1 2"` / `--seeds-glaucoma "0 1"` change seed counts.

**Gate:** decide the sensitivity method is a "keep" (worst-group sensitivity beats random and ERM-sensitivity at matched budget, and matches budget-matched TPE) before Experiment 2.

## Step 4 - Experiment 2: intersectional fairness (needs Step 3's masks)

```
./orchestration/run_experiments.sh phase2
python3 orchestration/summarize_runs.py
```
Phase 2 runs: the intersectional search; the **core ablation** (Stage 2 on the intersectional groups using each single-attribute mask from Step 3); full-fine-tune (all-ones mask) and head-only (all-zeros mask) baselines.
Marginal per-attribute gaps of the joint model:
`python3 orchestration/analyze_intersectional_results.py --groups-csv runs/<dataset>/sens_wg/intersectional/seed0/RESULTS_intersectional_groups_min_auc.csv --mapping-csv data_splits_<dataset>/intersectional_mapping.csv`

A gain counts only if |paired seed difference| > seed SD **and** the bootstrap CI excludes 0 (decide this threshold before looking at results). Report overall, best and worst-group AUC, not gaps alone.

## Estimated time (unmeasured; replace after Step 2)

| Stage | Estimate |
|---|---|
| PAPILA sensitivity search / TPE-10 | ~6-12 min / ~5-10 min |
| Harvard-GF sensitivity search / TPE-10 | ~20-35 min / ~15-30 min |
| Whole plan | ~9-20 h (2-3 overnight sessions) |

## Caveats to state in the paper

- PAPILA test has a group with 0 positives (worst-group AUC undefined there): feasibility result only; Harvard-GF is the main evidence.
- A 10-point k-sweep costs about the same as 10-trial TPE; the speed-up is vs full-budget TPE and quality at equal cost. The paper's 48 GPU-h (Fitzpatrick17k) is not comparable; report a measured ratio.
- TPE has no seed flag (its "seeds" are independent repeats). `race` runs write to `intersectional` folders (the driver moves them out per run).
- Novelty: selection of which PEFT components to tune under an intersectional worst-group objective, "to our knowledge, within a limited literature search" (see `literature-review-conversation.md`; related: Gu et al. 2025 arXiv 2512.03477, Lin et al. 2023, FairMedFM).

Output locations: `runs/<dataset>/<label>/<attr>/seed<N>/`, `runs/timings.csv`, `runs/summary_*.csv`.
