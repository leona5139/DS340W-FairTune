# Experiment checklist: speed-up validation → intersectionality

Two claims to validate, in order. The second is built **on top of** the
first and only uses the new method — the original TPE search is never run
on intersectional attributes (too expensive for the payoff: TPE cost scales
with a noisier objective and Harvard-GF's 12-group case would be the most
expensive run in this whole plan).

1. **Speed-up**: the new sensitivity method (`search_mask_sensitivity.py`)
   matches/beats the original TPE method (`search_mask.py`) on the existing
   single sensitive attributes, on both PAPILA and Harvard-GF3300.
2. **Intersectionality**: the new method, run with intersectional attributes
   on both datasets, compared against that *same method's* single-attribute
   results from step 1 — does joint optimization help, hurt, or leave each
   individual attribute's fairness gap unchanged ("fairness gerrymandering")?

**Design decisions locked in for this pass:**
- TPE baseline = **budget-matched only** (`--num_trials 10`, matching the
  new method's ~10 trainings). No full 50-trial TPE reruns — that's the
  comparison the speed-up plan itself calls out as "the one that matters."
  Consequence: the rank-correlation "thesis plot" (needs a full-budget TPE
  `Run_Stats_*.csv`) is out of scope for this pass; revisit later if the
  core comparison lands well.
- Seeds: **3** on PAPILA (small, noisy — per the speed-up plan's own
  warning), **1** on Harvard-GF (larger, more stable, and already the
  costlier dataset per run).
- The FairTune paper's published numbers are a sanity-check reference, not
  a rerun target.
- Hold `objective_metric` (`min_auc`), `tuning_method`, and epochs constant
  across every run in a given dataset, to avoid confounding the comparison
  with search-protocol drift. Use `--objective_metric min_auc` throughout.

---

## Phase 0 — Prerequisites

- [ ] `python3 orchestration/verify_env.py` — confirm GPU visible, no
      `DataLoader` deadlock (fall back to `--workers 0/1/2` if it hangs).
- [ ] Smoke-test the sensitivity method end-to-end for the first time, in a
      disposable clone (never the real tree):
      `./orchestration/run_pipeline.sh all gender --smoke-test --search-method sensitivity --repo-dir ~/fairtune_papila/FairTune_smoketest`
- [ ] Smoke-test intersectional on PAPILA:
      `./orchestration/run_pipeline.sh all intersectional --smoke-test --search-method sensitivity --repo-dir ~/fairtune_papila/FairTune_smoketest`
- [ ] Smoke-test intersectional on Harvard-GF (same command + `--dataset glaucoma`)
- [ ] Time one real (non-smoke) run per dataset to rebuild the wall-clock
      budget estimate from measurement.

## Phase 1 — Speed-up validation (single attributes, both datasets)

Attributes: PAPILA = `{gender, age}`; Harvard-GF = `{gender, age, race}`.

For each (dataset, attribute) pair, run and log wall-clock time + test
Best/Worst/Diff/minAUC for:

- [ ] Budget-matched TPE: `--search-method optuna --num_trials 10`
- [ ] New sensitivity method: `--search-method sensitivity --rank_method worst_group_sensitivity`

Seeds: 3x on PAPILA, 1x on Harvard-GF → 18 PAPILA + 6 Harvard-GF = **24 core runs**.

Optional ablations (after the core 24 land; 1 seed is enough):
- [ ] Random-10-masks: `--rank_method random` — does ranking beat chance at matched budget?
- [ ] ERM-sensitivity: `--rank_method erm_sensitivity` — does scoring on
      worst-group loss (vs. plain ERM loss) actually matter?

Analysis:
- [ ] Table: wall-clock time + test Best/Worst/Diff (mean ± std where
      seeds > 1) per (dataset, attribute, method).
- [ ] Sanity-check against the FairTune paper's published numbers (rough
      agreement expected, not exact reproduction).
- [ ] If ablations run: confirm `worst_group_sensitivity` beats both
      `random` and `erm_sensitivity`.
- [ ] **Gate**: decide the new method is a "keep" before starting Phase 2.

## Phase 2 — Intersectionality (new method only, both datasets)

- [ ] PAPILA intersectional (age × gender, 4 groups), sensitivity method, 3 seeds:
      `./orchestration/run_pipeline.sh all intersectional --search-method sensitivity --rank_method worst_group_sensitivity`
- [ ] Harvard-GF intersectional (age × gender × race, 12 groups), sensitivity
      method, 1 seed: same command + `--dataset glaucoma`
- [ ] Confirm `RESULTS_intersectional_groups_min_auc.csv` has all groups
      (4 / 12) populated with nonzero counts in every split — a 0.0 on a
      tiny subgroup is a real signal, not noise to suppress.
- [ ] Run `orchestration/analyze_intersectional_results.py` on each
      dataset's real results (first real, non-synthetic test of this
      script) to recombine per-group counts into marginal per-attribute gaps.

Analysis:
- [ ] For each attribute, compare its marginal gap from the
      jointly-optimized model against that same attribute's Phase-1
      sensitivity-method single-attribute run. Helped, hurt, or unchanged?
- [ ] Cross-dataset: does any individual-fairness cost of joint optimization
      get worse going from PAPILA's 2-way intersection to Harvard-GF's 3-way?

## Verification

- All Phase 0 smoke tests pass before any real-tree run.
- Each real run produces `RESULTS_<sens_attribute>_min_auc.csv` (Phase 1)
  and `RESULTS_intersectional_groups_min_auc.csv` (Phase 2).
- Wall-clock numbers roughly match expectation (PAPILA ~1-3 min/run,
  Harvard-GF ~10-20 min/run) — large deviations suggest a run-config issue,
  not a real finding.
- Total run count for this pass: 24 core Phase 1 + up to 12 ablation + 4
  Phase 2 = ~40 runs.
