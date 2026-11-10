# Experiment results (v2: jointly stratified PAPILA split, whole-split group AUC)

Everything written by `orchestration/run_experiments.sh` for the final runs, plus
files extracted from the run logs so every number in the paper can be re-plotted.
Machine: AMD RX 9060 XT (ROCm), ViT-Base, 30 epochs, AMP bf16, batch 42 (PAPILA) / 30 (Harvard-GF).

## Layout

```
runs/
  timings.csv                      one row per finished cell: dataset,label,attr,seed,stage1/stage2/total seconds
  summary_cells.csv                mean/std over seeds of test AUC overall/best/worst/diff + stage times
  summary_speedup.csv              TPE extrapolated to 200 trials vs measured sensitivity runtime
  glaucoma_eval_metrics.csv        every whole-split evaluation in runs/glaucoma.log (see below)
  papila_v2_eval_metrics.csv       same for runs/papila_v2.log
  glaucoma.filtered.log.gz         full run logs minus sklearn warning spam (raw logs are >100 MB,
  papila_v2.filtered.log.gz        not committed)
  glaucoma_config.log              Harvard-GF preprocessing / config log
  <dataset>/tpe_sample/<attr>/seed0/            Experiment 1: original FairTune TPE, 5-trial sample
  <dataset>/sens_wg/<attr>/seed<N>/             Experiments 2 & 3: sensitivity-ranking search + fine-tune
                                                (PAPILA seeds 0-9, Harvard-GF 0-2)
  <dataset>/xmask_<attr>/intersectional/seed<N>/  single-attribute mask, fine-tuned + evaluated on intersectional groups
  <dataset>/base_full_ft|base_head_only/intersectional/seed<N>/  all-ones / all-zeros mask (runs/_const_masks/)
  <dataset>/sens_random|sens_erm/intersectional/seed0/  ranking ablation: random order / ERM-loss sensitivity
  papila/tpe_full/intersectional/seed0/         original FairTune TPE, full 200 trials (+ Stage 2)
  <dataset>/paired_<label>/intersectional/seed<N>/  Stage-2 re-run of <label> with the SAME mask + LR, test
                                                predictions in a fixed order -> paired comparisons (see below)
  tpe_full_pauses.txt                          PAUSED/RESUMED times of the tpe_full run (subtract from its time)
  {night1,queue2,queue3}.filtered.log.gz      driver logs of these runs, sklearn warnings removed (raw logs not committed)
```

**Paired re-runs.** The original Stage 2 iterates the test set with a shuffled loader, so `test_predictions_*.npz`
of different runs are in different orders. `finetune_with_mask.py` now dumps predictions with a sequential
loader after training; `run_experiments.sh paired` re-ran every intersectional-evaluated model that way. Use
`paired_*` for any cross-model comparison; identical-mask re-runs also measure Stage-2 noise
(`paper_results/data/rerun_variability.csv`).

`<dataset>` is `papila` (attrs: gender, age, intersectional = age x sex, 4 groups) or
`glaucoma` = Harvard-GF (attrs: gender, age, race, intersectional = age x sex x race, 12 groups).
The Harvard-GF **race** cell runs as `--sens_attribute intersectional --group_cols Race`, so its
files carry the name `intersectional` inside `runs/glaucoma/*/race/`.

## Per-cell files

**TPE sample (`tpe_sample/<attr>/seed0/`)**
- `Optuna Run Stats/Trial_Timing_*.csv` – per trial: state, epochs run, duration, seconds/epoch (basis of the 200-trial extrapolation).
- `Optuna Run Stats/Run_Stats_*.csv` – per trial: objective (val worst-group AUC), start/end time, the 36 mask bits and LR.
- `Optuna Results/Fairness_Optuna_*.csv` – per trial test metrics.
- `Optuna_Masks/<attr>/*.npy` – best mask found (36 bits).

**Sensitivity (`sens_wg/<attr>/seed<N>/`)**
- `Sensitivity_Scores/<attr>/scores_worst_group_sensitivity.csv` – the 36 mask components: block, type, raw and parameter-normalized sensitivity, rank. (Bar chart of which blocks get unfrozen.)
- `Sensitivity_Results/<attr>/k_sweep_worst_group_sensitivity_min_auc.csv` – for each k in 1,2,3,4,6,8,12,18,24,36: val acc/AUC overall, best, worst, gap, trainable params, wall-clock. (k-sweep / params-vs-fairness curves.)
- `Sensitivity_Results/<attr>/timing_worst_group_sensitivity.csv` – scoring vs sweep seconds.
- `Optuna_Masks/<attr>/*.npy` – chosen mask (file name ends in its val worst-group AUC).
- `RESULTS_<attr>_min_auc.csv` – final **test** metrics of the stage-2 fine-tune: overall AUC, per-group AUC (gender: Male, Female columns; age / race / intersectional: best and worst group), gap, equalized-odds diff/ratio, demographic-parity diff/ratio.
- `RESULTS_intersectional_groups_min_auc.csv` (intersectional + race cells) – test correct/count/acc/AUC **per group**.
- `test_predictions_intersectional.npz` (intersectional + race cells) – per test image: `prob` (P(glaucoma)), `label`, `group`. Use for ROC curves or bootstrap CIs. Not saved for gender/age cells.

Group ids for intersectional cells are listed in `data_splits_<dataset>/intersectional_mapping.csv`
(with per-split counts and positives). Race-only cells: 0 Asian, 1 Black, 2 White.

## `*_eval_metrics.csv` (training / search curves)

Built by `python orchestration/extract_log_metrics.py runs/<log> --dataset <dataset>`.
Long format: one row per (evaluation, group).

| column | meaning |
|---|---|
| script | `search_mask` (TPE), `search_mask_sensitivity`, `finetune_with_mask` (stage 2) |
| sens_attribute, group_cols, seed | which cell |
| cfg_epochs | `--epochs` of that process. **Keep 30**; 2 = the smoke test that shares `papila_v2.log` |
| complete | **Keep True**; False = a process stopped mid-cell and later re-run from scratch |
| phase | `tpe_trial`, `k_sweep`, `finetune` |
| trial / k / epoch | TPE trial index / k-sweep size / last finished epoch before this evaluation |
| split | `val` or `test` (inferred from set size) |
| auc_overall, group, group_auc, group_n | whole-split AUC and each group's AUC (NaN = single-class group) |

Examples: per-epoch val worst-group AUC of stage 2 → `phase=="finetune" & split=="val"`, min of
`group_auc` per (seed, epoch). TPE trial learning curves → `phase=="tpe_trial"`, by `trial`.

## Notes
- Sensitivity timings in `summary_speedup.csv` are from seed 0 (machine otherwise idle). TPE hours are
  **extrapolated** (mean trial time x 200), not measured.
- `summary_cells.csv` reads RESULTS columns positionally, so for PAPILA **gender** "best"/"worst" are
  actually Male/Female and appear swapped; use `RESULTS_gender_min_auc.csv` directly.
- `_stale*/` (git-ignored) hold v1 results from the old split / per-batch AUC and partial cells
  from interrupted runs. Do not mix them with these results.
