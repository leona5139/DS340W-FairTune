# One-time WSL2 + ROCm setup (AMD RX 9060/9070, RDNA4)

This is a runbook, not a script — the driver install and `wsl --install` steps
need reboots and Windows-side interaction, so a single unattended script would
be misleading. Do these once; everything after that is just
`./orchestration/run_pipeline.sh`.

AMD revises exact package/wheel filenames with almost every ROCm point
release, so **confirm the exact filenames at the linked URLs when you actually
run this** rather than trusting anything baked in below.

## 1. Windows host

Install/update **AMD Software: Adrenalin Edition 26.2.2 or later** (this is
the minimum version with WSL2 ROCm support for RDNA4 — check AMD's current
WSL support page for the current minimum). Reboot.

## 2. Install WSL2 + Ubuntu 24.04

In an elevated PowerShell:

```powershell
wsl --install -d Ubuntu-24.04
```

(If WSL is already installed, `wsl --update` instead.) Reboot if prompted,
then complete Ubuntu's first-run user setup.

## 3. Install ROCm inside WSL Ubuntu

```bash
sudo apt update
# Confirm the exact .deb filename at:
#   https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_<exact-version>_all.deb
sudo apt install ./amdgpu-install_<exact-version>_all.deb
sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms
```

`--no-dkms` is mandatory under WSL2 — it can't build kernel modules. On ROCm
>=7.2.x you do **not** need the older `hsa-runtime-rocr4wsl-amdgpu` package.

## 4. Build librocdxg (WSL GPU-passthrough shim)

Requires the Windows SDK installed on the Windows host first (Visual Studio
Installer -> Individual Components -> Windows SDK, or the standalone
installer) — check which version you have under
`C:\Program Files (x86)\Windows Kits\10\Include\`.

```bash
sudo apt install -y cmake build-essential   # need CMake >=3.15, GCC >=11.4
git clone https://github.com/ROCm/librocdxg.git
cd librocdxg
export win_sdk='/mnt/c/Program Files (x86)/Windows Kits/10/Include/<your-installed-sdk-version>'
mkdir build && cd build
cmake .. -DWIN_SDK="${win_sdk}/shared"
make
sudo make install
rocminfo   # your RDNA4 GPU should appear as an agent
```

## 5. Install ROCm PyTorch wheels

AMD explicitly recommends `repo.radeon.com` wheels over pytorch.org's for
WSL. Get the current filenames for your Python version (3.12 on Ubuntu
24.04) from `https://repo.radeon.com/rocm/manylinux/rocm-rel-<current>/`:

```bash
pip3 uninstall -y torch torchvision pytorch-triton-rocm torchaudio
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-<X.Y.Z>/torch-...-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-<X.Y.Z>/torchvision-...-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-<X.Y.Z>/pytorch_triton_rocm-...-cp312-cp312-linux_x86_64.whl
pip3 install torch-*.whl torchvision-*.whl pytorch_triton_rocm-*.whl
```

## 6. Sanity check

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
rocm-smi   # the ROCm equivalent of nvidia-smi
```

Both should succeed before you go any further. If `is_available()` is
`False`, revisit step 4 (librocdxg) — this is almost always a passthrough
problem, not a PyTorch problem.

## 7. Copy PAPILA into the Linux filesystem — not `/mnt/c/...`

```bash
mkdir -p ~/fairtune_papila/data
cp -r "/mnt/c/path/to/your/Papila" ~/fairtune_papila/data/Papila
```

The `/mnt/c/...` 9p bridge is known to be slow for datasets with many small
files (PAPILA is ~488 images) — copying into WSL2's own ext4 filesystem
avoids that entirely and matches `run_pipeline.sh`'s default `--papila-dir`.

## 8. Clone the repo

This clone is the actual assignment submission
(`github.com/leona5139/DS340W-FairTune`), already containing the
`orchestration/` scripts and pre-generated `data_splits_papila/*.csv`
prepared ahead of time:

```bash
git clone https://github.com/leona5139/DS340W-FairTune.git ~/fairtune_papila/FairTune
```

## 9. Install the remaining Python dependencies

```bash
cd ~/fairtune_papila/FairTune
pip3 install -r orchestration/requirements.txt
```

`openpyxl` is required for `pandas.read_excel` on PAPILA's raw `.xlsx`
files — Colab shipped it preinstalled, WSL2 needs it explicit.

## Now you're ready

```bash
python3 orchestration/verify_env.py                              # seconds
./orchestration/run_pipeline.sh all gender --smoke-test \
    --repo-dir ~/fairtune_papila/FairTune_smoketest               # minutes, disposable clone
./orchestration/run_pipeline.sh all                               # the real, recorded run
```

See `orchestration/run_pipeline.sh`'s header comment for full usage, and the
implementation plan for why the smoke test uses a separate throwaway clone
(mask discovery is "most-recently-modified `.npy`," so a smoke-test mask must
never land in the same `Optuna_Masks/` tree as the real result).
