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
supports — note that CUDA version, you'll need it in step 4.

## 2. Python

Install Python from python.org if it isn't already on `PATH`. Confirm:

```bash
python --version
```

## 3. Clone the repo

This clone is the actual assignment submission
(`github.com/leona5139/DS340W-FairTune`), already containing the
`orchestration/` scripts and pre-generated `data_splits/*.csv` prepared ahead
of time:

```bash
git clone https://github.com/leona5139/DS340W-FairTune.git ~/fairtune_papila/FairTune
cd ~/fairtune_papila/FairTune
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

Get the current recommended install command for your CUDA version (from
step 1) at pytorch.org/get-started/locally — it looks like:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
```

(`cuXXX` e.g. `cu121`/`cu124` — pick the one matching your driver's supported
CUDA version, rounding down if there's no exact match.)

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

## Now you're ready

Still inside the activated `.venv`:

```bash
python orchestration/verify_env.py                               # seconds
bash orchestration/run_pipeline.sh all gender --smoke-test \
    --repo-dir ~/fairtune_papila/FairTune_smoketest               # minutes, disposable clone
bash orchestration/run_pipeline.sh all                             # the real, recorded run
```

See `orchestration/run_pipeline.sh`'s header comment for full usage, and
`CLAUDE.md` for why the smoke test uses a separate throwaway clone (mask
discovery is "most-recently-modified `.npy`," so a smoke-test mask must never
land in the same `Optuna_Masks/` tree as the real result).
