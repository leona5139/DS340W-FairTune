# How to run the experiments

Three experiments, each on **PAPILA** and **Harvard-GF** (glaucoma):

| # | What | Script | Budget | Output label |
|---|---|---|---|---|
| 1 | **Original FairTune** (Optuna/TPE mask search, the paper's code) — a *sample* of trials, extrapolated to a full run | `search_mask.py` | 5 trials × 30 epochs per (dataset, attr), seed 0, Stage 1 only | `tpe_sample` |
| 2 | **Sped-up search** (SPT-style one-shot gradient-sensitivity ranking + 10-point nested k-sweep), single attributes | `search_mask_sensitivity.py` + `finetune_with_mask.py` | full: 10 trainings × 30 epochs + 30-epoch Stage 2 | `sens_wg` |
| 3 | **Sped-up search, intersectional groups** (PAPILA age × sex = 4 groups; Harvard-GF age × sex × race = 12) | same | same as 2 | `sens_wg`, attr `intersectional` |

Attributes: PAPILA `gender age` (+ `intersectional`); Harvard-GF `gender age race` (+ `intersectional`).
Seeds for 2 and 3: PAPILA `0 1 2`, Harvard-GF `0` (it is the larger, less noisy dataset and ~7× costlier per epoch).
Mixed precision (bf16 autocast, `AMP=1`) is **on by default** for every training stage, TPE and sensitivity alike; `AMP=0` gives full fp32.
30 epochs is the FairTune paper's fine-tuning budget (Sec. 4.1); 200 trials is where the paper reports the search saturating (App. A.1).

> **Graders / Windows + NVIDIA:** jump to [Quick start for graders](#quick-start-for-graders-windows--nvidia).

---

## Step 0 — Environment and data (once per machine)

All commands run from the repo root, in **bash** (Linux) or **Git Bash** (Windows), with the project venv active.
`run_pipeline.sh` refuses to start if `$VIRTUAL_ENV` is unset.

1. **Python venv + PyTorch with GPU support**, installed *before* the other requirements:
   - **AMD GPU, Linux (ROCm):** `python3 -m venv .venv && source .venv/bin/activate`, then install the ROCm build of `torch`/`torchvision` from pytorch.org (select *ROCm*). RX 9000-series (RDNA4) cards need ROCm 6.4 or newer. On NixOS the pip wheels need an FHS-style environment (e.g. `programs.nix-ld` or a `buildFHSEnv` shell) to find system libraries.
   - **NVIDIA GPU (Windows or Linux):** follow [`CUDA_SETUP.md`](CUDA_SETUP.md) (Windows venv lives in `.venv/Scripts/`, use `source .venv/Scripts/activate`).
2. `pip install -r orchestration/requirements.txt`
3. **Raw data** (not in git). Defaults below; pass `--papila-dir` / `--harvard-dir` to the `config` commands if yours live elsewhere.
   - PAPILA → `~/fairtune_papila/data/Papila/` containing `ClinicalData/` and `FundusImages/`.
   - Harvard-GF → download `https://huggingface.co/datasets/harvardairobotics/Harvard-GF/resolve/main/Dataset/dataset.zip` and extract so that `~/fairtune_papila/data/harvard-dataset/` contains `Training/ Validation/ Test/` (one `data_NNNN.npz` per subject).
4. `python3 orchestration/verify_env.py` (Windows: `python`) — GPU visible, DataLoader workers don't hang. If it hangs, prefix every later command with `WORKERS=0`.
5. Fill `config.yaml` for both datasets (never hand-edit it):
   ```
   ./orchestration/run_pipeline.sh config --dataset papila
   ./orchestration/run_pipeline.sh config --dataset glaucoma
   ```
   The split CSVs (`data_splits_papila/`, `data_splits_glaucoma/`) are committed (see *Splits and metrics* below). The glaucoma `config` step additionally converts the raw `.npz` files to `harvard-dataset/RNFLT_png/` the first time (a few minutes; the CSVs it rewrites are identical).

### Splits and metrics

- **PAPILA** (`orchestration/preprocess_papila.py`): patient-level ~70/20/10 (the MEDFAIR split proportions FairTune uses; both eyes of a patient stay in one split), stratified **jointly on age × sex group and glaucoma label**. Each (group, label) stratum is split by largest-remainder rounding with a coverage rule: strata of ≥ 3 patients get at least one patient in every split; a 2-patient stratum goes 1 train / 1 test. Result: 292 / 82 / 46 eyes, and every group has glaucoma cases in train and test (young-male val has none: only 2 such patients exist). `intersectional_mapping.csv` lists per-split counts and positives (`pos_*`). Regenerate with `./orchestration/run_pipeline.sh preprocess --dataset papila --papila-dir <dir> --force-preprocess` (deterministic).
- **Harvard-GF**: the official Training/Validation/Test split, unchanged, so results stay comparable with published Harvard-GF numbers.
- **Group metrics are computed once over the whole split** (`_pooled_fairness_metrics` in `utilities/training_utils.py`), for gender, age and intersectional alike, and for both the search objective (val) and the reported results (test). The original code averaged per-batch AUCs, which is meaningless when a batch holds only 2–3 images of a group (Harvard-GF: 12 groups at batch 30). A group that is empty or single-class in a split gets AUC NaN and is excluded from best/worst.
- Results produced before these two changes (old split + per-batch AUC) are parked in `runs/_stale_v1/`. Don't mix them with new runs.

## Step 1 — Smoke test (tiny epochs/trials; ~30–60 min)

```
./orchestration/run_experiments.sh all --smoke-test --runs-dir runs_smoke --seeds-papila 0 --seeds-glaucoma 0
python3 orchestration/summarize_runs.py --runs-dir runs_smoke
```
`--smoke-test` uses 2 epochs, 2 TPE trials and k ∈ {1, 4, 36}. Outputs go to `runs_smoke/` (git-ignored), so nothing leaks into the real grid.
Check:
- every run finished (`runs_smoke/<dataset>/<label>/<attr>/seed0/DONE` exists for all 14 runs);
- each `tpe_sample` dir has `Optuna Run Stats/Trial_Timing_*.csv`;
- summary section 2 prints one row per (dataset, attr) with an extrapolated TPE cost;
- Stage 2 test AUC is not exactly 0.5; `RESULTS_intersectional_groups_min_auc.csv` has 4 (PAPILA) / 12 (Harvard-GF) groups; NaN-group warnings are printed, not silent.

## Step 2 — Run PAPILA first, then size Harvard-GF from it

Total cost is roughly **training epochs × seconds per epoch**:

| Block | Runs | Training epochs |
|---|---|---|
| Exp 1, PAPILA | 3 cells × 5 trials × 30 | 450 |
| Exp 1, Harvard-GF | 4 cells × 5 trials × 30 | 600 |
| Exp 2+3, PAPILA | 3 attrs × 3 seeds × (10 k × 30 + 30 Stage 2) | ≈ 2,970 |
| Exp 2+3, Harvard-GF | 4 attrs × 1 seed × (10 × 30 + 30) | ≈ 1,320 |

Harvard-GF has 2,100 training images per epoch vs PAPILA's ~292, so expect one Harvard-GF epoch to cost **~7× a PAPILA epoch** (an estimate: compute scales with image count, but per-epoch overheads don't).

Recommended order:
1. **PAPILA, all three experiments:** `./orchestration/run_experiments.sh all --dataset papila`
2. `python3 orchestration/summarize_runs.py` — section 2 gives measured seconds per TPE trial and per epoch (`tpe_s_per_epoch`) and the measured sensitivity run times.
3. **Estimate Harvard-GF:** hours ≈ PAPILA seconds/epoch × 7 × 1,920 epochs / 3600 (600 Exp 1 + 1,320 Exp 2+3). To replace the ×7 guess with a measurement, run one Harvard-GF cell first (≈ 150 epochs, the cheapest real measurement):
   `./orchestration/run_experiments.sh exp1 --dataset glaucoma` — then stop it after the first cell, or let it finish (it's part of the plan anyway). Read `tpe_s_per_epoch` for glaucoma from the summary.
4. Then decide the Harvard-GF budget. If it's too long, the next lever is fewer epochs (`STAGE1_EPOCHS=15 STAGE2_EPOCHS=15`) for **every** Harvard-GF run, TPE and sensitivity alike, written to their own `--runs-dir` (`timings.csv` doesn't record epochs, so don't mix budgets in one folder). The TPE-vs-sensitivity speed-up ratio is roughly unchanged by the epoch count, because both scale linearly with it.

Optional paired checks (keep the chosen settings fixed for every run that is compared):
- **AMP vs fp32:** AMP is on by default. To confirm it doesn't change results, rerun one PAPILA cell in fp32 into a separate folder and compare AUC (should agree within seed noise): `AMP=0 ./orchestration/run_experiments.sh exp2 --dataset papila --seeds-papila 0 --runs-dir runs_fp32`.
- `FAIRTUNE_IMG_CACHE_SHORT_SIDE=0` disables the PAPILA downscaled-image cache (results should agree, epochs get slower). The cache is on for both methods by default, so the speed-up you measure is the search algorithm, not I/O.

## Step 3 — Run the experiments

```
./orchestration/run_experiments.sh exp1     # Experiment 1: TPE sample, 7 cells, seed 0
./orchestration/run_experiments.sh exp2     # Experiment 2: sensitivity, single attrs, all seeds
./orchestration/run_experiments.sh exp3     # Experiment 3: sensitivity, intersectional, all seeds
```
or all three at once, ordered so that every seed-0 run (the timing runs) finishes before the extra seeds start:
```
./orchestration/run_experiments.sh all
```
Useful options: `--dataset papila|glaucoma`, `--seeds-papila "0"` (add seeds later by rerunning with `"0 1 2"` — finished runs are skipped), `--dry-run` (print the command list), `--seeds-glaucoma "0 1"` (a second Harvard-GF seed later), `TPE_SAMPLE_TRIALS=10 ./orchestration/run_experiments.sh exp1` (bigger TPE sample than the default 5; delete that cell's dir first if it already ran).

Rules:
- **Don't let the machine sleep.** On Linux with systemd, `run_experiments.sh` holds a `systemd-inhibit` sleep/idle lock for its whole run (`FAIRTUNE_NO_INHIBIT=1` disables it). On other systems, turn off auto-suspend yourself. A suspend mid-run kills the job, and on AMD/ROCm the GPU reset afterwards can leave compute unusable until a reboot. The driver is resumable, so after a crash just rerun the same command: finished runs are skipped, and the interrupted one restarts from scratch.
- To keep a long run going after the terminal closes (Linux): `setsid nohup ./orchestration/run_experiments.sh all --dataset papila >> runs/papila_run.log 2>&1 < /dev/null &` (NixOS: wrap in `./.fhs-env/bin/fairtune-fhs -c 'source .venv/bin/activate && ...'`).
- **One run on the GPU at a time.** Runtime is a headline result; don't run two copies of the driver or other GPU jobs alongside it.
- Don't run `run_pipeline.sh` by hand in the same tree mid-grid. The driver moves each run's outputs from `vit_base/<dataset>/` into `runs/` and parks any leftovers in `runs/_stale/`. This includes the old class-run files committed under `vit_base/papila/`, so git will show them as deleted. Restore them with `git checkout -- vit_base` once the grid is done.

Optional extras (not part of the three experiments):
- `exp3_ablation` (after exp2 + exp3): Stage 2 on the intersectional groups using each single-attribute mask from Experiment 2, plus full-fine-tune (all-ones mask) and head-only (all-zeros mask) baselines — answers "does searching on the intersectional objective beat reusing a single-attribute mask?"
- `ablation`: sensitivity ranking with `rank_method random` / `erm_sensitivity`, seed 0 — does worst-group ranking beat chance / plain ERM ranking at the same budget?

## Step 4 — Analyse

```
python3 orchestration/summarize_runs.py        # writes runs/summary_*.csv
```
- **Section 1** (`summary_cells.csv`): per (dataset, method, attr) mean ± std over seeds of Stage 1/Stage 2 seconds and test AUC overall / best group / worst group / gap.
- **Section 2** (`summary_speedup.csv`), Experiment 1 vs 2/3: `tpe_stage1_extrapolated_h` = mean measured TPE trial time × 200 (`--tpe-full-trials N` to change). `tpe_total_extrapolated_h` adds the measured Stage 2 time, since Stage 2 is the same 30-epoch fine-tune for both methods. These are compared with the **measured** seed-0 sensitivity runtimes (`speedup_stage1`, `speedup_total`).
  The extrapolation is close to linear because `search_mask.py` reports to its pruner only *after* a trial's full epoch loop, so pruned trials still train all 30 epochs and every trial costs about the same. `tpe_s_per_trial_std` shows how true that held. In the paper, still label these numbers "extrapolated".
- **Section 3** (`summary_phase2.csv`): only if `exp3_ablation` ran.
- Marginal per-attribute gaps of an intersectional model (does joint optimization help or hurt each single attribute?):
  ```
  python3 orchestration/analyze_intersectional_results.py \
    --groups-csv runs/<dataset>/sens_wg/intersectional/seed0/RESULTS_intersectional_groups_min_auc.csv \
    --mapping-csv data_splits_<dataset>/intersectional_mapping.csv
  ```
  Compare with the same dataset's Experiment 2 single-attribute rows.

A difference counts only if |paired seed difference| > seed SD (and, for `exp3_ablation`, the bootstrap CI excludes 0). Fix this threshold before looking at results. Report overall, best- and worst-group AUC, not gaps alone.

## Output layout

```
runs/timings.csv                                   one row per run: stage1/stage2/total seconds
runs/<dataset>/<label>/<attr>/seed<N>/
    DONE
    RESULTS_*_min_auc.csv                          test metrics (exp2/exp3)
    RESULTS_intersectional_groups_min_auc.csv      per-group counts/AUC (intersectional, race)
    test_predictions_*.npz                         per-sample test predictions
    Optuna_Masks/<attr>/*.npy                      the selected 36-bit mask
    Optuna Run Stats/Trial_Timing_*.csv            per-trial seconds (exp1)
    Sensitivity_Results/<attr>/, Sensitivity_Scores/<attr>/   (exp2/exp3: k-sweep table, timing, component scores)
runs/summary_*.csv                                 from summarize_runs.py
```
`race` runs go through the N-group evaluator (`--sens_attribute intersectional --group_cols Race`), so their raw outputs are named `intersectional`. They are still stored under `attr = race`.

## Quick start for graders (Windows + NVIDIA)

The full grid takes days and isn't meant to be reproduced. This section checks that the code runs end to end and shows where the numbers come from.

1. Setup: [`CUDA_SETUP.md`](CUDA_SETUP.md) — Git for Windows (**run everything from Git Bash**, not PowerShell/cmd), Python, `.venv`, CUDA PyTorch, `pip install -r orchestration/requirements.txt`. Use `python` where this guide says `python3`.
2. Data: Step 0.3 above (PAPILA alone is enough). Then `./orchestration/run_pipeline.sh config --dataset papila --papila-dir /c/path/to/Papila` and `python orchestration/verify_env.py`.
3. GPU memory: ViT-B at the fixed batch size (42 PAPILA / 30 Harvard-GF) in fp32 needs roughly 12 GB+ of GPU memory. AMP (bf16) is on by default, which lowers that noticeably (activations are stored in 16-bit). On NVIDIA cards older than RTX 30xx, prefix commands with `AMP_DTYPE=fp16` (they lack fast bf16). Keep the batch size at its default so timings are comparable; it no longer has to divide the split sizes, so lowering it (`BATCH_SIZE=16`) is a safe last resort for a small GPU.
4. Smoke test, PAPILA only (minutes):
   ```
   ./orchestration/run_experiments.sh all --smoke-test --dataset papila --seeds-papila 0 --runs-dir runs_smoke
   python orchestration/summarize_runs.py --runs-dir runs_smoke
   ```
5. Optional, one real cell (30 epochs, roughly an hour or more depending on GPU): `./orchestration/run_experiments.sh exp2 --dataset papila --seeds-papila 0`.

The Windows/CUDA path has run the original PAPILA pipeline end to end before. The sensitivity, Harvard-GF and intersectional paths have only been exercised on Linux, so if anything breaks on Windows, step 4 is where it will show up.

## Caveats to state in the paper

- PAPILA is tiny: the whole dataset has only 2 young-male and 4 young-female glaucoma patients. The jointly stratified split gives every age × sex group positives in train and test, but young-male test AUC rests on one patient's eyes and young-male val positives are 0 (that group is excluded from the search objective). Treat PAPILA intersectional as a feasibility result, report seed SD and bootstrap CIs; Harvard-GF is the main evidence.
- The TPE cost is extrapolated from a 5-trial sample, not measured at 200 trials. The paper's 48 GPU-h (Fitzpatrick17k, batch 512) is not comparable: report the ratio measured on one machine.
- TPE (`search_mask.py`) has no seed flag. Batch sizes (42 / 30) differ from the paper's 512 (GPU memory); they are kept fixed across all runs so timings are comparable.
- The class-weight computation (`--compute_cw`) is PAPILA-only. In the original Stage 1 search code the weights are computed but never passed to the loss; `search_mask_sensitivity.py` deliberately mirrors this, so both searches see the same loss.
- Novelty claim: selecting which PEFT components to tune under an intersectional worst-group objective, "to our knowledge, within a limited literature search" (related: Gu et al. 2025 arXiv 2512.03477, Lin et al. 2023, FairMedFM).
