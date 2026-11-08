#!/usr/bin/env python3
"""Build train.csv/val.csv/test.csv for FairTune's PapilaDataset from the raw
PAPILA release (ClinicalData/*.xlsx + FundusImages/*.jpg).

Port of the Colab notebook's Section 3, unchanged in logic. Pure pandas/openpyxl/
sklearn — no GPU required — safe to run on any machine that has the raw Papila/
folder, independent of wherever the actual training later happens.

Split naming note: the assignment defines Test as the 20% touched iteratively
during development, and Validation as the 10% touched only once, at the very
end. That's the OPPOSITE of FairTune's own file-naming convention, where
val_csv is what Optuna evaluates every trial / every epoch, and test_csv is
the final holdout evaluated once. To satisfy the assignment's actual
behavioral rule rather than just matching names:

  Assignment role   Size  Behavior          -> FairTune file   Touched
  Training          70%   fit weights       -> train.csv       every step
  Test (assignment) 20%   iterative dev     -> val.csv         every trial + epoch
  Validation (asg.) 10%   unseen until done -> test.csv         once, final results

Split sizes follow MEDFAIR's PAPILA.ipynb (patient-level ~70/20/10, the split
FairTune itself uses), but patients are stratified JOINTLY on
Intersectional_Group (Age_binary x Sex) x patient-level Diagnosis, so every
age x sex group keeps its glaucoma/healthy ratio in every split. The earlier
Diagnosis-only stratification left young men with zero glaucoma eyes in
train.csv and young women with zero in test.csv. sklearn's train_test_split
can't stratify strata this small (young-male glaucoma = 2 patients), so a
small deterministic allocator is used instead -- see allocate_stratum().
"""
import argparse
import os

import numpy as np
import pandas as pd

from intersectional_utils import build_intersectional_group


def _fix_df(df):
    """Reproduces HelpCode/utils.py::_fix_df — PAPILA's xlsx files ship with a
    messy 3-row header; this promotes the real header row and drops the spacer."""
    df_new = df.drop(["ID"], axis=0)
    df_new.columns = df_new.iloc[0, :]
    df_new = df_new.drop([np.nan], axis=0)
    df_new.columns.name = "ID"
    return df_new


SPLIT_FRACS = (0.7, 0.2, 0.1)  # train, val (assignment "Test"), test (assignment "Validation")
SPLIT_SEED = 5


def allocate_stratum(n, fracs=SPLIT_FRACS):
    """Patients per split (train, val, test) for one (group, diagnosis) stratum.

    Largest-remainder rounding of n * fracs, then a coverage rule so thin
    strata still reach the splits that matter:
      n >= 3: at least 1 patient in every split (taken from train)
      n == 2: 1 train, 1 test (test is what gets reported; val is skipped and
              that group's val AUC is NaN, which the search already excludes)
      n == 1: train only
    """
    if n == 1:
        return [1, 0, 0]
    if n == 2:
        return [1, 0, 1]
    raw = [n * f for f in fracs]
    counts = [int(np.floor(r)) for r in raw]
    order = sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[: n - sum(counts)]:
        counts[i] += 1
    for i in (1, 2):
        if counts[i] == 0:
            counts[i] = 1
            counts[0] -= 1
    assert sum(counts) == n and min(counts) >= 1, (n, counts)
    return counts


def load_clinical(path):
    raw = pd.read_excel(path, index_col=[0])
    return _fix_df(raw)


def build_eye_df(df, eye_suffix):
    out = df.copy()
    out["patient_id"] = [int(str(i).replace("#", "")) for i in out.index]
    out["Path"] = out["patient_id"].apply(lambda pid: f"RET{pid:03d}{eye_suffix}.jpg")
    out["Age"] = out["Age"].astype(float)
    out["Gender"] = out["Gender"].astype(float)
    out["Diagnosis"] = out["Diagnosis"].astype(float)
    return out[["patient_id", "Age", "Gender", "Diagnosis", "Path"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--papila-dir", required=True, help="Path to the extracted Papila/ folder (containing ClinicalData/ and FundusImages/)")
    ap.add_argument("--out-dir", required=True, help="Directory to write train.csv/val.csv/test.csv into")
    args = ap.parse_args()

    papila_dir = args.papila_dir
    img_dir = os.path.join(papila_dir, "FundusImages")
    os.makedirs(args.out_dir, exist_ok=True)

    df_od = load_clinical(os.path.join(papila_dir, "ClinicalData", "patient_data_od.xlsx"))
    df_os = load_clinical(os.path.join(papila_dir, "ClinicalData", "patient_data_os.xlsx"))
    print(f"OD rows: {len(df_od)} | OS rows: {len(df_os)}")

    od_rows = build_eye_df(df_od, "OD")
    os_rows = build_eye_df(df_os, "OS")
    meta = pd.concat([od_rows, os_rows], ignore_index=True)

    # Gender 0.0/1.0 -> Sex M/F (PAPILA/MEDFAIR convention)
    meta["Sex"] = meta["Gender"].map({0.0: "M", 1.0: "F"})
    # Age >= 60 binary threshold (MEDFAIR convention)
    meta["Age_binary"] = (meta["Age"] >= 60).astype(int)

    print(f"Total eye-level rows: {len(meta)}")
    print(f"Diagnosis value counts: {meta['Diagnosis'].value_counts().to_dict()}")

    # Drop Diagnosis == 2 (suspect) -- FairTune's PapilaDataset only understands
    # binary healthy(0)/glaucoma(1)
    meta_binary = meta[meta["Diagnosis"].isin([0.0, 1.0])].copy()
    meta_binary["Diagnosis"] = meta_binary["Diagnosis"].astype(int)
    print(f"Rows after dropping suspect: {len(meta_binary)} / {len(meta)}")

    # Assign Intersectional_Group on the FULL dataframe, before splitting --
    # guarantees train/val/test share one consistent 0..N-1 numbering
    # (Age_binary x Sex here -> 4 groups) regardless of which groups end up
    # in which split.
    meta_binary["Intersectional_Group"], intersectional_mapping = build_intersectional_group(
        meta_binary, ["Age_binary", "Sex"]
    )
    print("Intersectional group mapping (Age_binary x Sex):")
    print(intersectional_mapping.to_string(index=False))

    # Patient-level split (avoids leaking a patient's OD/OS pair across
    # splits), stratified jointly on Intersectional_Group x patient-level
    # Diagnosis (positive if either eye is positive). Age and sex are
    # per-patient, so a patient's group is the same for both eyes.
    patients = meta_binary.groupby("patient_id").agg(
        group=("Intersectional_Group", "first"),
        n_groups=("Intersectional_Group", "nunique"),
        diag=("Diagnosis", "max"),
    )
    assert (patients["n_groups"] == 1).all(), "a patient's two eyes landed in different age x sex groups"

    rng = np.random.default_rng(SPLIT_SEED)
    train_ids, iterative_dev_ids, final_holdout_ids = [], [], []
    print("Patients per stratum (group, diagnosis) -> train/val/test:")
    for (g, y), stratum in patients.groupby(["group", "diag"]):
        ids = np.sort(stratum.index.to_numpy())
        rng.shuffle(ids)
        n_tr, n_va, n_te = allocate_stratum(len(ids))
        train_ids += ids[:n_tr].tolist()
        iterative_dev_ids += ids[n_tr:n_tr + n_va].tolist()
        final_holdout_ids += ids[n_tr + n_va:].tolist()
        print(f"  group {g} diag {y}: {len(ids)} patients -> {n_tr}/{n_va}/{n_te}")

    train_df = meta_binary[meta_binary["patient_id"].isin(train_ids)].reset_index(drop=True)
    # iterative_dev_ids (~20% of patients) = assignment's "Test" -> val.csv
    val_df = meta_binary[meta_binary["patient_id"].isin(iterative_dev_ids)].reset_index(drop=True)
    # final_holdout_ids (~10% of patients) = assignment's "Validation" -> test.csv
    test_df = meta_binary[meta_binary["patient_id"].isin(final_holdout_ids)].reset_index(drop=True)

    print(f"train.csv: {len(train_df)} rows, {len(train_ids)} patients (assignment: Training, 70%)")
    print(f"val.csv:   {len(val_df)} rows, {len(iterative_dev_ids)} patients (assignment: Test, 20%, touched iteratively)")
    print(f"test.csv:  {len(test_df)} rows, {len(final_holdout_ids)} patients (assignment: Validation, 10%, touched once)")

    # Sanity checks before trusting these splits
    assert set(train_ids) & set(iterative_dev_ids) == set(), "train/val patient overlap!"
    assert set(train_ids) & set(final_holdout_ids) == set(), "train/test patient overlap!"
    assert set(iterative_dev_ids) & set(final_holdout_ids) == set(), "val/test patient overlap!"

    for name, d in [("train", train_df), ("val (assignment Test)", val_df), ("test (assignment Validation)", test_df)]:
        print(name, "-> rows:", len(d), "| class balance:", d["Diagnosis"].value_counts().to_dict())

    missing = [p for p in meta_binary["Path"] if not os.path.exists(os.path.join(img_dir, p))]
    print("Missing image files:", len(missing))
    assert len(missing) == 0, f"Missing images: {missing[:5]}..."

    cols = ["Path", "Diagnosis", "Sex", "Age_binary", "Intersectional_Group"]
    train_csv_path = os.path.join(args.out_dir, "train.csv")
    val_csv_path = os.path.join(args.out_dir, "val.csv")
    test_csv_path = os.path.join(args.out_dir, "test.csv")
    mapping_csv_path = os.path.join(args.out_dir, "intersectional_mapping.csv")

    train_df[cols].to_csv(train_csv_path, index=False)
    val_df[cols].to_csv(val_csv_path, index=False)
    test_df[cols].to_csv(test_csv_path, index=False)

    # Per-split group counts, appended onto the overall mapping -- a group
    # with 0 rows in a given split is a real signal (too-small subgroup),
    # not something to hide.
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = d["Intersectional_Group"].value_counts().reindex(
            intersectional_mapping["Intersectional_Group"], fill_value=0
        )
        intersectional_mapping[f"count_{name}"] = counts.values
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        pos = d.groupby("Intersectional_Group")["Diagnosis"].sum().reindex(
            intersectional_mapping["Intersectional_Group"], fill_value=0
        )
        intersectional_mapping[f"pos_{name}"] = pos.values.astype(int)
    intersectional_mapping.to_csv(mapping_csv_path, index=False)
    print("Per-split group counts:")
    print(intersectional_mapping.to_string(index=False))

    print(
        "All sanity checks passed. Wrote:",
        train_csv_path, val_csv_path, test_csv_path, mapping_csv_path,
    )


if __name__ == "__main__":
    main()
