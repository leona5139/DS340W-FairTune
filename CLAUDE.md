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

The original plan ran this on Google Colab (free T4 GPU) because the user's
own GPU is AMD (RX 9060/9070, RDNA4) and ROCm/Windows support was assumed too
immature. Colab caused two real problems: its ephemeral `/content` silently
reverted `config.yaml` between runtime resets (crashed
`finetune_with_mask.py` deep into a run with a confusing `TypeError`), and
notebooks are painful to screen-record for the assignment's "run it on
camera" requirement (no press-play-and-walk-away). ROCm 7.2 now officially
supports RDNA4 under WSL2, so this pivots to running locally instead.

This `orchestration/` layer and `WSL_ROCM_SETUP.md` were built and tested on
a **different machine with no AMD GPU** — everything that doesn't need a GPU
(bash argument parsing, `preprocess_papila.py`, `configure_fairtune.py`
write/verify, `common.sh`'s `banner`/`discover_mask`) was actually run and
verified there. The GPU-dependent path — `verify_env.py`'s device checks, and
`search_mask.py`/`finetune_with_mask.py` actually running — has **never been
run for real**. That's the part this session, on the real hardware, needs to
prove out.

## Start here

1. Read `WSL_ROCM_SETUP.md` top to bottom — one-time WSL2 + ROCm install
   (Adrenalin driver, `amdgpu-install --usecase=wsl,rocm --no-dkms`,
   `librocdxg` for GPU passthrough, ROCm PyTorch wheels from
   `repo.radeon.com`). AMD renames point-release filenames often — confirm
   exact filenames at the linked URLs rather than trusting what's written.
2. `python3 orchestration/verify_env.py` — fast sanity check. If the
   `DataLoader(num_workers=4)` check hangs with no traceback, that's the
   documented ROCm DataLoader deadlock, not a bug in this script — rerun the
   real pipeline with `--workers 0`/`1`/`2` instead of chasing it further. If
   `torch.cuda.is_available()` is `False`, that's almost always a librocdxg
   passthrough problem (step 4 of the setup doc), not a PyTorch problem.
3. Smoke test against a **disposable second clone**, not this one — mask
   discovery is "most-recently-modified `.npy` under `Optuna_Masks/<attr>/`",
   so a 2-epoch smoke-test mask must never land in the same tree as the real
   result:
   ```
   git clone <this repo's remote> ~/fairtune_papila/FairTune_smoketest
   ./orchestration/run_pipeline.sh all gender --smoke-test --repo-dir ~/fairtune_papila/FairTune_smoketest
   ```
4. Only once that's clean: the real, screen-recorded run —
   `./orchestration/run_pipeline.sh all` (both `gender` and `age`, full
   epochs/trials). One command, one terminal — this is what gets recorded.
5. `data_splits_papila/*.csv` are already generated and committed
   (deterministic given fixed seeds — verified train=294/val=84/test=42
   rows). Only run `preprocess_papila.py` again if you need to regenerate
   them; it's idempotent and skips if the CSVs already exist.
   `data_splits_glaucoma/*.csv` (Harvard-GF) are built the same way by
   `preprocess_glaucoma.py` — see `INTERSECTIONAL_FAIRNESS_PLAN.md`.
6. `config.yaml` is deliberately left unfilled in this repo — `run_pipeline.sh
   all`/`config` fills it automatically with paths that are correct on
   *this* machine. Don't hand-edit it.

## Known risk areas (things to watch for, not yet confirmed on real hardware)

- **ROCm DataLoader `num_workers > 0` deadlock** — a documented history on
  ROCm generally. `WORKERS` defaults to 4 in `orchestration/common.sh`;
  `verify_env.py` smoke-tests this exact value first. Drop via
  `WORKERS=0 ./orchestration/run_pipeline.sh ...` if needed.
- **`librocdxg` / GPU passthrough under WSL2** — the newest, least-trodden
  part of the ROCm-on-Windows story as of early 2026. If `rocminfo` doesn't
  show the GPU as an agent, this is where to look first.
- **VRAM**: `BATCH_SIZE=42` (fixed — see `common.sh`'s comment: it's chosen
  to evenly divide PAPILA's split sizes and avoid a zero-guard-free
  per-subgroup division crash in the repo's own `utilities/utils.py`, not for
  performance). If you hit an OOM on an 8GB card, that's a real constraint to
  raise with the user, not something to silently "fix" by changing the batch
  size — it would reintroduce the crash risk it was chosen to avoid unless
  you check the new batch size still evenly divides 294/84/42.

## After a successful real run

Push the results back (second commit, separate from the orchestration
scaffolding already pushed): the now-filled `config.yaml`, `RESULTS_*.csv`,
and optionally the small `Optuna_Masks/**/*.npy` files. Never commit raw
PAPILA images/`.xlsx`, the `TORCH_HOME` weight cache, or per-epoch
checkpoints (`.gitignore` already covers this). Confirm with the user before
pushing — this is their public assignment submission repo.

Full design rationale (why bash over a Python wrapper, why Stage 1's
epochs/trials were raised from the Colab-era 5/10 to 15/20, why the pruner
was left as `SuccessiveHalving`, exact file responsibilities) is in each
script's own header comment — `orchestration/run_pipeline.sh`,
`common.sh`, `preprocess_papila.py`, and `configure_fairtune.py` are all
short and meant to be read directly rather than summarized twice.
