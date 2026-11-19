# One-time NVIDIA/CUDA setup (native Windows)

This is a runbook, not a script — the driver install needs a reboot and
Windows-side interaction, so a single unattended script would be misleading.
Do these once; everything after that is just `orchestration/run_pipeline.sh`.

`run_pipeline.sh` and the rest of `orchestration/` are bash scripts. Run them
from **Git Bash** (installed with Git for Windows), not PowerShell or
cmd.exe.

## 1. NVIDIA driver

Install/update the NVIDIA driver (Game Ready or Studio, whichever you
normally use) from nvidia.com. Reboot if prompted, then confirm the GPU is
visible:

```bash
nvidia-smi
```

This should print your GPU name and the maximum CUDA version the driver
supports — note that CUDA version, you'll need it in step 5.

## 2. Python

Install Python from python.org if it isn't already on `PATH`. Confirm:

```bash
python --version
```

## 3. Clone the repo

The repo already contains the `orchestration/` scripts and the PAPILA split
CSVs (`data_splits_papila/`):

```bash
git clone https://github.com/leona5139/DS340W-FairTune.git
cd DS340W-FairTune
```

(Any path works — PAPILA and the repo no longer need to live inside a Linux
filesystem the way they did under WSL2/ROCm; there's no `/mnt/c` bridge to
route around on native Windows.)

## 4. Create and activate the project virtualenv

All Python packages for this project — torch/torchvision included — go into
a project-local `.venv`, never the system/user Python. `Scripts/activate` is
the Windows venv layout (not the POSIX `bin/activate`):

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Everything below runs inside this activated venv. **Every new terminal**
(including the one you screen-record in) needs `source .venv/Scripts/activate`
run again first — `run_pipeline.sh` refuses to run at all if `$VIRTUAL_ENV`
isn't set, rather than silently falling back to whatever Python happens to be
on `PATH`.

## 5. Install CUDA-enabled PyTorch

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

`cu128` needs a driver whose `nvidia-smi` shows "CUDA Version: 12.8" or
higher (any driver from 2025 on), and it is the minimum for RTX 50-series
cards. If your driver shows less, update the driver (easiest), or use the
selector at pytorch.org/get-started/locally (Windows / Pip / Python / CUDA)
and take the command it gives. Don't use older indexes like `cu121`/`cu124`:
they stop at old PyTorch versions with no wheels for current Python, so pip
fails with "No matching distribution found for torch".

## 6. Install the remaining Python dependencies

```bash
pip install -r orchestration/requirements.txt
```

`openpyxl` is required for `pandas.read_excel` on PAPILA's raw `.xlsx` files.

## 7. Sanity check

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Both should succeed (`True` plus your GPU's name) before you go any further.
If `is_available()` is `False`, re-check step 5 — a CPU-only wheel got
installed instead of a CUDA one, usually from omitting `--index-url` or
mismatching the CUDA version against the driver.

## 8. Get PAPILA and turn off sleep

Download PAPILA (link in the README) and extract it so that
`~/fairtune_papila/data/Papila/` contains `ClinicalData/` and `FundusImages/`
(in Git Bash, `~` is `C:\Users\<you>`). Any other location works if you add
`--papila-dir <path>` to the `config` command below.

Set Settings > System > Power > "Put my device to sleep" to **Never** (on
battery too, for a laptop) for the duration of the run. Windows sleeping
mid-run kills the job.

## 9. Run PAPILA, sensitivity search, one seed

Still inside the activated `.venv`, from the repo root:

```bash
python orchestration/verify_env.py                              # seconds; must end with "Environment looks good."
./orchestration/run_pipeline.sh config --dataset papila          # writes config.yaml for this machine

# optional smoke test (a few minutes; outputs go to runs_smoke/, not the real results)
./orchestration/run_experiments.sh exp2 --smoke-test --dataset papila --seeds-papila 0 --runs-dir runs_smoke

# the real run: sensitivity mask search + fine-tune, seed 0 (~20-30 min on a modern GPU)
./orchestration/run_pipeline.sh all gender --dataset papila --search-method sensitivity
```

Swap `gender` for `age` or `intersectional` (age x sex) for the other
fairness attributes. To do all three for seed 0 in one go, with each run's
outputs kept separate under `runs_windows/`:

```bash
./orchestration/run_experiments.sh exp2 --dataset papila --seeds-papila 0 --runs-dir runs_windows   # gender, age
./orchestration/run_experiments.sh exp3 --dataset papila --seeds-papila 0 --runs-dir runs_windows   # intersectional
```

Results: `vit_base/papila/RESULTS_<attr>_min_auc.csv` (or under
`runs_windows/papila/sens_wg/<attr>/seed0/` for the second form).

If something goes wrong:

- **`verify_env.py` reports a DataLoader hang**: put `WORKERS=0 ` in front of
  the run commands (e.g. `WORKERS=0 ./orchestration/run_pipeline.sh ...`).
- **CUDA out of memory**: `BATCH_SIZE=16 ./orchestration/run_pipeline.sh ...`.
- **RTX 20-series or older (no bf16)**: `AMP_DTYPE=fp16 ./orchestration/run_pipeline.sh ...`.
