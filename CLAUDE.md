# Project context for Claude Code

This repo started as a straight, unmodified import of `Raman1121/FairTune`
(ICLR 2024 paper code) plus an `orchestration/` layer added around it — it is
also the actual assignment submission for a class, and while the assignment
was live, the rule here was **never modify `search_mask.py`,
`finetune_with_mask.py`, `parse_args.py`, `data/`, or `utilities/`** (the
assignment required running the authors' code unmodified), with everything
new confined to `orchestration/`, `WSL_ROCM_SETUP.md`, `data_splits_papila/`,
and `config.yaml`.

**That rule is now waived.** The assignment has been submitted; core files
are being edited in place for a post-submission research extension —
generalizing the paper's hardcoded 4-group `age_sex` intersectional-fairness
pattern into an N-way mechanism, across PAPILA and a second dataset
(Harvard-GF glaucoma). See `INTERSECTIONAL_FAIRNESS_PLAN.md` for the design
and status. Data now lives in per-dataset split directories:
`data_splits_papila/` (PAPILA) and `data_splits_glaucoma/` (Harvard-GF).

## Why this exists

The original plan ran this on Google Colab (free T4 GPU). Colab caused two
real problems: its ephemeral `/content` silently reverted `config.yaml`
between runtime resets (crashed `finetune_with_mask.py` deep into a run with
a confusing `TypeError`), and notebooks are painful to screen-record for the
assignment's "run it on camera" requirement (no press-play-and-walk-away).
This runs locally, on the user's own machine, rather than on Google Colab's
free T4 GPU — Colab's ephemeral `/content` silently reverted `config.yaml`
between runtime resets (crashed `finetune_with_mask.py` deep into a run with
a confusing `TypeError`), and notebooks are painful to screen-record for the
assignment's "run it on camera" requirement (no press-play-and-walk-away).

The **primary path is an AMD GPU under WSL2 + ROCm** — see
`WSL_ROCM_SETUP.md`. A native-Windows-with-NVIDIA-GPU (CUDA) path also
exists as a **fallback** (`CUDA_SETUP.md`, from when the hardware situation
briefly looked like it might change) but should not be treated as primary
unless ROCm/WSL2 genuinely can't be made to work on the current machine. The
core files (`search_mask.py`, `finetune_with_mask.py`, `parse_args.py`,
`utilities/`) don't care either way — they already use plain,
vendor-agnostic PyTorch (`--device cuda` default, `torch.cuda.*`), which
works unmodified on both ROCm-as-CUDA and real NVIDIA CUDA.

## Start here

1. Read `WSL_ROCM_SETUP.md` top to bottom — one-time WSL2 + ROCm driver +
   Python venv + ROCm PyTorch wheel install (`CUDA_SETUP.md` is the
   equivalent runbook for the NVIDIA/CUDA fallback, if you ever need it
   instead).
2. Every terminal session (including the one you record in) needs the
   project venv active first. `run_pipeline.sh` hard-fails immediately with
   a clear error if `$VIRTUAL_ENV` is unset, rather than silently running
   against the wrong Python — if you see that error, you forgot to activate.
3. `python3 orchestration/verify_env.py` — fast sanity check. If
   `torch.cuda.is_available()` is `False`, re-check the setup runbook's
   PyTorch-wheel step (a CPU-only torch wheel got installed instead of a
   ROCm/CUDA one). If the `DataLoader(num_workers=4)` check hangs with no
   traceback, rerun the real pipeline with `--workers 0`/`1`/`2` instead of
   chasing it further.
4. Smoke test against a **disposable second clone**, not this one — mask
   discovery is "most-recently-modified `.npy` under `Optuna_Masks/<attr>/`",
   so a 2-epoch smoke-test mask must never land in the same tree as the real
   result:
   ```
   git clone <this repo's remote> ~/fairtune_papila/FairTune_smoketest
   ./orchestration/run_pipeline.sh all gender --smoke-test --repo-dir ~/fairtune_papila/FairTune_smoketest
   ```
5. Only once that's clean: the real, screen-recorded run —
   `./orchestration/run_pipeline.sh all` (both `gender` and `age`, full
   epochs/trials). One command, one terminal — this is what gets recorded.
6. `data_splits_papila/*.csv` are already generated and committed
   (deterministic given fixed seeds — verified train=294/val=84/test=42
   rows). Only run `preprocess_papila.py` again if you need to regenerate
   them; it's idempotent and skips if the CSVs already exist.
   `data_splits_glaucoma/*.csv` (Harvard-GF) are built the same way by
   `preprocess_glaucoma.py` — see `INTERSECTIONAL_FAIRNESS_PLAN.md`.
7. `config.yaml` is deliberately left unfilled in this repo — `run_pipeline.sh
   all`/`config` fills it automatically with paths that are correct on
   *this* machine. Don't hand-edit it.

## Known risk areas

- **VRAM**: `BATCH_SIZE=42` (fixed — see `common.sh`'s comment: it's chosen
  to evenly divide PAPILA's split sizes and avoid a zero-guard-free
  per-subgroup division crash in the repo's own `utilities/utils.py`, not for
  performance). If you hit an OOM, that's a real constraint to raise with the
  user, not something to silently "fix" by changing the batch size — it
  would reintroduce the crash risk it was chosen to avoid unless you check
  the new batch size still evenly divides 294/84/42.
- **`discover_mask()` in `common.sh`** uses GNU `find -printf`, which on the
  primary WSL2/ROCm path is just Ubuntu's normal `find` (fine). On the
  Windows/CUDA fallback it needs Git Bash's bundled `find` (the real MSYS2
  GNU findutils build, not a BusyBox stand-in). If `stage1`/`stage2` can't
  find a mask that Optuna clearly wrote, this is where to look first.
- **`PY_BIN` resolution in `common.sh`** resolves straight from
  `$VIRTUAL_ENV/Scripts/python.exe` (Windows) or `$VIRTUAL_ENV/bin/python`
  (POSIX/WSL2), falling back to a `python3`-then-`python` `PATH` search only
  if sourced without an active venv — this is what makes both the
  WSL2/ROCm primary path and the Windows/CUDA fallback work with the same
  script. On Windows specifically, `python -m venv` never creates a
  `python3.exe` in `Scripts/`, so a plain `PATH` search skips straight past
  the activated venv and hits Windows' `python3` App Execution Alias stub
  (which just errors telling you to install from the Microsoft Store) —
  confirmed failure mode when that fallback was in use, not hypothetical. If
  `PY_BIN` ever resolves wrong, check `$VIRTUAL_ENV` is actually set first.
- **DataLoader worker hangs**: `WORKERS` defaults to 4 in
  `orchestration/common.sh`; `verify_env.py` smoke-tests this exact value
  first, using a module-level worker function (deliberately not a closure —
  needed because Windows' default `spawn` multiprocessing start method can't
  pickle a nested function, which would crash before the hang-detection
  logic even ran; harmless on WSL2/Linux's `fork` too). Drop via
  `WORKERS=0 ./orchestration/run_pipeline.sh ...` if a real hang shows up.

## After a successful real run

Push the results back (second commit, separate from the orchestration
scaffolding already pushed): the now-filled `config.yaml`, `RESULTS_*.csv`,
and optionally the small `Optuna_Masks/**/*.npy` files. Never commit raw
PAPILA images/`.xlsx`, the `TORCH_HOME` weight cache, per-epoch checkpoints,
or the `.venv/` directory (`.gitignore` already covers all of this). Confirm
with the user before pushing — this is their public assignment submission
repo.

Full design rationale (why bash over a Python wrapper, why Stage 1's
epochs/trials were raised from the Colab-era 5/10 to 15/20, why the pruner
was left as `SuccessiveHalving`, exact file responsibilities) is in each
script's own header comment — `orchestration/run_pipeline.sh`,
`common.sh`, `preprocess_papila.py`, and `configure_fairtune.py` are all
short and meant to be read directly rather than summarized twice.
