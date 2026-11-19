# *FairTune*: Optimizing Parameter-Efficient Fine-Tuning for Fairness in Medical Image Analysis

:star2: Accepted to ICLR 2024! | [Paper Link](https://arxiv.org/abs/2310.05055)

![Fairtune](assets/fairtune_framework.png)  

Abstract: Training models with robust group fairness properties is crucial in ethically sensitive application areas such as medical diagnosis. Despite the growing body of work aiming to minimise demographic bias in AI, this problem remains challenging. A key reason for this challenge is the fairness generalisation gap: High-capacity deep learning models can fit all training data nearly perfectly, and thus also exhibit perfect fairness during training. In this case, bias emerges only during testing when generalisation performance differs across subgroups. This motivates us to take a bi-level optimisation perspective on fair learning: Optimising the learning strategy based on validation fairness. Specifically, we consider the highly effective workflow of adapting pre-trained models to downstream medical imaging tasks using parameter-efficient fine-tuning (PEFT) techniques. There is a trade-off between updating more parameters, enabling a better fit to the task of interest vs. fewer parameters, potentially reducing the generalisation gap. To manage this tradeoff, we propose *FairTune*, a framework to optimise the choice of PEFT parameters with respect to fairness. We demonstrate empirically that *FairTune* leads to improved fairness on a range of medical imaging datasets. 

---

## About this repository

This is a fork of the authors' code ([Raman1121/FairTune](https://github.com/Raman1121/FairTune), ICLR 2024) with a
post-submission research extension:

- the paper's hard-coded 4-group `age_sex` fairness pattern is generalized to an **N-way intersectional** mechanism;
- a second dataset, **Harvard-GF** (glaucoma), next to **PAPILA**;
- a faster mask search, `search_mask_sensitivity.py` (one-shot gradient-sensitivity ranking plus a nested k-sweep),
  compared against the original Optuna/TPE search (`search_mask.py`);
- an `orchestration/` layer (bash + Python) that runs the whole pipeline with one command.

| File | What it is |
|---|---|
| [RUN_GUIDE.md](RUN_GUIDE.md) | Full runbook for the three experiments (days of GPU time) |
| [CUDA_SETUP.md](CUDA_SETUP.md) | Detailed Windows + NVIDIA setup |
| [paper_results/](paper_results/README.md) | Tables, figures and data behind the write-up |
| `orchestration/run_pipeline.sh` | One dataset / one attribute: preprocess, config, Stage 1 (mask search), Stage 2 (fine-tune) |
| `orchestration/run_experiments.sh` | The experiment grid, with timings |

## Requirements

- **GPU**: NVIDIA (CUDA) or AMD (ROCm, Linux only). About 12 GB of GPU memory for fp32; the default bf16 mixed
  precision needs less. There is no CPU fallback worth waiting for.
- **Python 3.10+** and **git**. On Windows you also need **Git for Windows**, because the scripts are bash
  and must be run from *Git Bash* (not PowerShell or cmd).
- **PyTorch + torchvision** with GPU support (install first, see below).
- Everything else is in [orchestration/requirements.txt](orchestration/requirements.txt):
  `timm`, `optuna`, `loralib`, `fairlearn`, `wandb`, `pyyaml`, `pandas`, `numpy`, `pillow`, `scikit-learn`,
  `openpyxl`, `matplotlib`, `seaborn`, `scipy`.

## Datasets

Neither dataset can be redistributed, so download them yourself.

| Dataset | Link | Used for |
|---|---|---|
| **PAPILA** | https://www.nature.com/articles/s41597-022-01388-1#Sec6 | main (small, fast) experiments |
| **Harvard-GF** | https://ophai.hms.harvard.edu/datasets/harvard-glaucoma-fairness-3300-samples/ (direct zip: https://huggingface.co/datasets/harvardairobotics/Harvard-GF/resolve/main/Dataset/dataset.zip) | larger second dataset (optional) |

Put them here (or pass `--papila-dir` / `--harvard-dir` to the `config` command):

```
~/fairtune_papila/data/Papila/             # contains ClinicalData/ and FundusImages/
~/fairtune_papila/data/harvard-dataset/    # contains Training/ Validation/ Test/ (data_NNNN.npz files)
```

The train/val/test split CSVs (`data_splits_papila/`, `data_splits_glaucoma/`) are already committed, and
`config.yaml` is filled in for you by the `config` step below. Do not hand-edit it. PAPILA alone is enough to
reproduce the main results.

## Setup and run

Pick your platform. All commands are run from the repo root.

### 1. Get the code

```bash
git clone https://github.com/leona5139/DS340W-FairTune.git
cd DS340W-FairTune
```

### 2. Create the virtual environment and install PyTorch

Install the GPU build of PyTorch **before** the other requirements, otherwise pip may pull a CPU-only build.

<details open>
<summary><b>Linux + NVIDIA</b> (Ubuntu)</summary>

```bash
sudo apt install python3-venv python3-pip git
nvidia-smi                                   # should list your GPU; install the NVIDIA driver first if not
python3 -m venv .venv && source .venv/bin/activate
# pick the cuXXX that matches your driver (pytorch.org/get-started/locally), e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r orchestration/requirements.txt
```
</details>

<details>
<summary><b>Linux + AMD</b> (Ubuntu, ROCm)</summary>

```bash
sudo apt install python3-venv python3-pip git
# install the ROCm stack for your distro first (https://rocm.docs.amd.com); RDNA4 (RX 9000-series) needs ROCm 6.4+
rocminfo | grep -i gfx                       # your GPU should be listed
python3 -m venv .venv && source .venv/bin/activate
# select "ROCm" at pytorch.org/get-started/locally and use the command it gives, e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
pip install -r orchestration/requirements.txt
```
ROCm builds of PyTorch expose the GPU through the normal `torch.cuda` API, so the code needs no changes.
</details>

<details>
<summary><b>Windows + NVIDIA</b> (Git Bash)</summary>

1. Install the NVIDIA driver, [Python](https://www.python.org/downloads/) (tick "Add to PATH") and
   [Git for Windows](https://git-scm.com/download/win). Check `nvidia-smi` works.
2. Open **Git Bash** and run:

```bash
python -m venv .venv
source .venv/Scripts/activate                # note: Scripts, not bin
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # match your driver's CUDA version
pip install -r orchestration/requirements.txt
```
Use `python` wherever this README says `python3`. More detail: [CUDA_SETUP.md](CUDA_SETUP.md).
</details>

<details>
<summary><b>Windows + AMD</b></summary>

Native Windows PyTorch with ROCm has **not been tested** with this repo, so no instructions are given for it.
Use WSL2 (Ubuntu) with AMD's ROCm-for-WSL driver and follow the *Linux + AMD* steps inside WSL, or boot Linux.
</details>

Every new terminal needs the venv activated again (`source .venv/bin/activate`, or `.venv/Scripts/activate` on
Windows). The scripts refuse to run without it.

### 3. Check the environment

```bash
python3 orchestration/verify_env.py          # Windows: python
```

`torch.cuda.is_available()` must be `True`. If it is `False`, a CPU-only PyTorch got installed; redo step 2.
If the DataLoader check hangs, prefix later commands with `WORKERS=0`.

### 4. Point the config at your data

```bash
./orchestration/run_pipeline.sh config --dataset papila                              # add --papila-dir <path> if not in the default location
./orchestration/run_pipeline.sh config --dataset glaucoma                            # optional; also converts the raw .npz files once
```

### 5. Smoke test (a few minutes), then a real run

```bash
# tiny epochs/trials, writes to runs_smoke/ so nothing pollutes real results
./orchestration/run_experiments.sh all --smoke-test --dataset papila --seeds-papila 0 --runs-dir runs_smoke

# one real PAPILA run, fast mask search, gender fairness (~20 min on an RX 9060 XT)
./orchestration/run_pipeline.sh all gender --dataset papila --search-method sensitivity

# the original paper's TPE search instead (8 Optuna trials, ~30 min)
./orchestration/run_pipeline.sh all gender --dataset papila
```

Use `age` or `intersectional` in place of `gender` for the other fairness attributes (`race` also exists for Harvard-GF).
The whole experiment grid (days of GPU time) is `./orchestration/run_experiments.sh all`; see [RUN_GUIDE.md](RUN_GUIDE.md).

### Reproduce the paper's tables and figures

```bash
./orchestration/reproduce_paper.sh                # rebuild every table/figure from the committed runs/ (minutes, no GPU)
./orchestration/reproduce_paper.sh --full         # re-run every experiment first (2-3 days of GPU), then rebuild
./orchestration/reproduce_paper.sh --full --dry-run   # just list the experiment commands
```

Output goes to `paper_results/` (LaTeX tables `T1-T6`, figures `F1-F5` as PDF/PNG, and the CSVs behind them);
see [paper_results/README.md](paper_results/README.md) for what each item is. Use `--out DIR` / `--runs-dir DIR` to
keep the originals untouched. With `--full` the paper's seeds are used (PAPILA 10, Harvard-GF 3).

### What a PAPILA run produces

Stage 1 searches for a 36-bit mask choosing which ViT components to fine-tune; Stage 2 fine-tunes with that mask
for 30 epochs and evaluates on the test split. Outputs land under `vit_base/papila/`
(or, when launched through `run_experiments.sh`, under `runs/papila/<label>/<attr>/seed<N>/`):

- `RESULTS_<attr>_min_auc.csv`: test AUC overall and per group, AUC gap, equalized-odds and demographic-parity
  metrics, % of parameters trained, learning rate, mask used
- `Optuna_Masks/<attr>/*.npy`: the selected mask
- `Sensitivity_Results/`: the k-sweep table (validation worst-group AUC vs trainable parameters), sensitivity method only
- `Optuna Run Stats/` / `Optuna Results/`: per-trial timing and results, TPE method only
- `runs/timings.csv`: one row of wall-clock seconds per run (experiment driver only)

Summarize finished runs with `python3 orchestration/summarize_runs.py`.

### Troubleshooting

- **`is_available()` is False**: CPU-only wheel installed; reinstall PyTorch with the right `--index-url`.
- **Out of GPU memory**: keep the default batch size when you can (timings stay comparable); `BATCH_SIZE=16 ./orchestration/run_pipeline.sh ...` is the last resort. Older NVIDIA cards (pre RTX 30xx): add `AMP_DTYPE=fp16`.
- **DataLoader hang**: `WORKERS=0 ./orchestration/run_pipeline.sh ...`.
- **Do not let the machine sleep** during a run; a suspend kills the job (and can wedge an AMD GPU until reboot).
- Reruns are safe: finished runs are skipped.

## Original FairTune usage

The upstream scripts can still be called directly:

### Run HPO search for finding the best mask (Stage 1)
```python
python search_mask.py --model [model] --epochs [epochs] --batch-size [batch-size] \
     --opt [opt] --lr [lr] --lr-scheduler [lr-scheduler] --lr-warmup-method [lr-warmup-method] --lr-warmup-epochs [lr-warmup-epochs] \
     --tuning_method [tuning_method] --dataset [dataset] --sens_attribute [sens_attribute] \
     --objective_metric [objective_metric] --num_trials [num_trials] --disable_storage --disable_checkpointing
```
You can use different metrics as objectives for the HPO search. Please check `parse_args.py` for more options.

### Fine-tune on the downstream task using the searched mask (Stage 2)
```python
python finetune_with_mask.py --model [model] --epochs [epochs] --batch-size [batch-size] \
     --opt [opt] --lr [lr] --lr-scheduler [lr-scheduler] --lr-warmup-method [lr-warmup-method] --lr-warmup-epochs [lr-warmup-epochs] \
     --tuning_method [tuning_method] --dataset [dataset] --sens_attribute [sens_attribute] \
     --cal_equiodds --mask_path [mask_path] --use_metric auc
```

`Note: It is advisable to use a weighted loss when working with Papila and OL3I datasets because of high imbalance, hence, use the --compute_cw argument.`

### Other datasets from the paper

The paper also used the datasets below (we did not run them); see [MEDFAIR](https://github.com/ys-zong/MEDFAIR/tree/main#data-preprocessing) for preprocessing.

| **Dataset**  | **Access**                                                                                    |
|--------------|-----------------------------------------------------------------------------------------------|
| CheXpert     | https://stanfordmlgroup.github.io/competitions/chexpert/                                      |
| OL3I         | https://stanfordaimi.azurewebsites.net/datasets/3263e34a-252e-460f-8f63-d585a9bfecfc          |
| HAM10000     | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T               |
| Oasis-1      | https://www.oasis-brains.org/#data                                                            |
| Fitzpatrick17k | https://github.com/mattgroh/fitzpatrick17k                                                  |

### Citing FairTune

```
@inproceedings{dutt2023fairtune,
  title={Fairtune: Optimizing parameter efficient fine tuning for fairness in medical image analysis},
  author={Dutt, Raman and Bohdal, Ondrej and Tsaftaris, Sotirios A and Hospedales, Timothy},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```



