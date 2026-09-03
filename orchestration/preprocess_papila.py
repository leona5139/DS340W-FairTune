#!/usr/bin/env python3
"""Build train.csv/val.csv/test.csv for FairTune's PapilaDataset from the raw
PAPILA release (ClinicalData/*.xlsx + FundusImages/*.jpg).

Port of the Colab notebook's Section 3, unchanged in logic. Pure pandas/openpyxl/
sklearn — no GPU, no ROCm — safe to run on any machine that has the raw Papila/
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

This mirrors MEDFAIR's own PAPILA.ipynb split parameters exactly
(test_size=0.3, random_state=5, then test_size=0.66, random_state=15 on the
remainder) so the resulting splits match what the notebook already produced.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _fix_df(df):
    """Reproduces HelpCode/utils.py::_fix_df — PAPILA's xlsx files ship with a
    messy 3-row header; this promotes the real header row and drops the spacer."""
    df_new = df.drop(["ID"], axis=0)
    df_new.columns = df_new.iloc[0, :]
    df_new = df_new.drop([np.nan], axis=0)
    df_new.columns.name = "ID"
    return df_new


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

    # Patient-level split (avoids leaking a patient's OD/OS pair across splits,
    # unlike the repo's own eye-level HelpCode/kfold/ files). Matches MEDFAIR's
    # exact parameters.
    patient_ids = meta_binary["patient_id"].unique()
    train_ids, remainder_ids = train_test_split(patient_ids, test_size=0.3, random_state=5)
    # train_test_split(X, test_size=0.66) returns (X_train=34%, X_test=66%) of `remainder_ids`
    final_holdout_ids, iterative_dev_ids = train_test_split(remainder_ids, test_size=0.66, random_state=15)

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

    cols = ["Path", "Diagnosis", "Sex", "Age_binary"]
    train_csv_path = os.path.join(args.out_dir, "train.csv")
    val_csv_path = os.path.join(args.out_dir, "val.csv")
    test_csv_path = os.path.join(args.out_dir, "test.csv")

    train_df[cols].to_csv(train_csv_path, index=False)
    val_df[cols].to_csv(val_csv_path, index=False)
    test_df[cols].to_csv(test_csv_path, index=False)

    print("All sanity checks passed. Wrote:", train_csv_path, val_csv_path, test_csv_path)


if __name__ == "__main__":
    main()
