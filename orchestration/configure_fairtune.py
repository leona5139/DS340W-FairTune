#!/usr/bin/env python3
"""Fill in FairTune's config.yaml for the papila dataset, or verify it's
already filled in correctly.

Port of the Colab notebook's Section 4, plus the hardening added after a
runtime reset silently reverted config.yaml between sessions and crashed
finetune_with_mask.py deep into an otherwise-successful run
(`yaml_data["data"]["papila"]["train_csv"]` -> TypeError, because
`yaml_data["data"]["papila"]` had reverted to a stale/empty state).

search_mask.py and finetune_with_mask.py both do
`with open("config.yaml") as file: yaml_data = yaml.safe_load(file)` — a path
relative to the process's cwd, with NO shared state with any other Python
session. This script is the single source of truth for what's actually on
disk; it never trusts an in-memory value it didn't just read back.
"""
import argparse
import os
import re

import yaml

# The repo's own config.yaml ships with invalid YAML syntax: the `fitzpatrick:`
# entry is written as `fitzpatrick: ''` (a scalar) immediately followed by an
# indented block of keys, which no YAML parser can load. Every other dataset
# entry (HAM10000, papila, ...) correctly has no value after the colon. This
# is a syntax fix to make the file loadable at all, not a change to its content.
_FITZPATRICK_BUG = re.compile(r"^(\s*fitzpatrick):\s*''\s*$", flags=re.MULTILINE)

REQUIRED_FIELDS = ("root_path", "img_path", "train_csv", "val_csv", "test_csv")


def _config_path(repo_dir):
    return os.path.join(repo_dir, "config.yaml")


def _load_fixed(repo_dir):
    path = _config_path(repo_dir)
    with open(path) as f:
        raw = f.read()
    raw = _FITZPATRICK_BUG.sub(r"\1:", raw)
    return path, yaml.safe_load(raw)


def write_config(repo_dir, dataset, img_dir, train_csv, val_csv, test_csv):
    path, cfg = _load_fixed(repo_dir)

    cfg["data"][dataset]["root_path"] = img_dir
    cfg["data"][dataset]["img_path"] = img_dir
    cfg["data"][dataset]["train_csv"] = train_csv
    cfg["data"][dataset]["val_csv"] = val_csv
    cfg["data"][dataset]["test_csv"] = test_csv

    for field in REQUIRED_FIELDS:
        val = cfg["data"][dataset][field]
        assert val, (
            f"config.yaml['data']['{dataset}']['{field}'] is empty/None after writing it. "
            "One of --img-dir/--train-csv/--val-csv/--test-csv was falsy."
        )

    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)

    verify(repo_dir, dataset)
    print(f"Wrote and verified config.yaml['{dataset}']:", cfg["data"][dataset])


def verify(repo_dir, dataset):
    """Re-reads config.yaml FRESH OFF DISK (not any in-memory dict) and
    confirms search_mask.py/finetune_with_mask.py will see a valid entry for
    `dataset` when they run. Raises with a clear message otherwise.

    Applies the same fitzpatrick-syntax-bug fix as write_config before
    parsing — a never-yet-configured (pristine) config.yaml is NOT valid YAML
    on its own, so without this fix every call here would fail with a raw
    yaml.parser.ParserError instead of the intended clear, actionable message.
    """
    path, cfg = _load_fixed(repo_dir)

    dataset_cfg = (cfg or {}).get("data", {}).get(dataset)
    if not dataset_cfg or not all(dataset_cfg.get(k) for k in REQUIRED_FIELDS):
        raise RuntimeError(
            f"{path}'s {dataset} entry is missing/empty: {dataset_cfg}. "
            "config.yaml has reverted to (or never left) its unfilled state — "
            "run `configure_fairtune.py` in write mode (not --verify-only) first."
        )

    for csv_key in ("train_csv", "val_csv", "test_csv"):
        csv_path = dataset_cfg[csv_key]
        assert os.path.exists(csv_path), f"{csv_key} path in config.yaml does not exist on disk: {csv_path}"

    print(f"config.yaml['{dataset}'] verified valid:", dataset_cfg)
    return dataset_cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-dir", required=True, help="Path to the FairTune repo clone (contains config.yaml)")
    ap.add_argument("--dataset", default="papila", choices=["papila", "glaucoma"], help="Which config.yaml['data'][...] section to fill/verify")
    ap.add_argument("--verify-only", action="store_true", help="Only verify config.yaml on disk is valid; do not write")
    ap.add_argument("--img-dir", help="Absolute path to the dataset's image directory (write mode only)")
    ap.add_argument("--train-csv", help="Absolute path to train.csv (write mode only)")
    ap.add_argument("--val-csv", help="Absolute path to val.csv (write mode only)")
    ap.add_argument("--test-csv", help="Absolute path to test.csv (write mode only)")
    args = ap.parse_args()

    if args.verify_only:
        verify(args.repo_dir, args.dataset)
        return

    missing = [name for name, val in [
        ("--img-dir", args.img_dir), ("--train-csv", args.train_csv),
        ("--val-csv", args.val_csv), ("--test-csv", args.test_csv),
    ] if not val]
    if missing:
        ap.error(f"write mode requires {', '.join(missing)} (or pass --verify-only)")

    write_config(args.repo_dir, args.dataset, args.img_dir, args.train_csv, args.val_csv, args.test_csv)


if __name__ == "__main__":
    main()
