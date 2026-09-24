#!/usr/bin/env python3
"""Build train.csv/val.csv/test.csv for FairTune's HarvardGlaucoma dataset
(data/glaucoma.py) from the raw Harvard-GF release
(https://huggingface.co/datasets/harvardairobotics/Harvard-GF).

Layout note: the full 3300-subject release ships pre-split into
Training/ (2100), Test/ (900), Validation/ (300) folders, each containing
one combined data_NNNN.npz per subject (both `rnflt` and `oct_bscans` in the
same file -- unlike a smaller partial copy of this release tried first,
whose schema instead had separate Bscan/ and RNFLT/ folders with one field
per npz; that partial copy also had no split at all, which is what this
script originally assumed before the full download's actual structure was
inspected). Subject ids are globally unique across all three folders
(verified: 3300 unique ids, zero collisions), so this script uses the
shipped split directly rather than re-deriving one -- FairTune's own
train_csv/val_csv/test_csv naming convention (val is what a trial/epoch
loop monitors, test is the untouched final holdout) already lines up
naturally with Training/Validation/Test, so no role-remapping is needed
(unlike preprocess_papila.py's inverted PAPILA-assignment naming, which was
a quirk of that specific class assignment, not applicable here).

Only the RNFLT/ 2D thickness maps are used as image input (per
INTERSECTIONAL_FAIRNESS_PLAN.md's decision) -- oct_bscans' 3D OCT volumes and
the visual-field/progression/language/maritalstatus fields are out of scope.
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

from intersectional_utils import build_intersectional_group

# Clip range for normalizing RNFLT (retinal nerve fiber layer thickness, in
# microns) to uint8. Derived from scanning a same-schema partial copy of
# this release: observed range was approximately -2..340, with ~8% exact
# zeros (background outside the optic disc). 350 gives headroom above the
# observed max.
RNFLT_CLIP_MIN = 0.0
RNFLT_CLIP_MAX = 350.0

SPLIT_FOLDERS = {"train": "Training", "val": "Validation", "test": "Test"}


def load_subject(npz_path):
    with np.load(npz_path, allow_pickle=True) as d:
        return {
            "rnflt": d["rnflt"],
            "glaucoma": int(d["glaucoma"]),
            "age": float(d["age"]),
            "male": int(d["male"]),
            "race": str(d["race"]),
        }


def rnflt_to_png(rnflt, out_path):
    clipped = np.clip(rnflt, RNFLT_CLIP_MIN, RNFLT_CLIP_MAX)
    scaled = ((clipped - RNFLT_CLIP_MIN) / (RNFLT_CLIP_MAX - RNFLT_CLIP_MIN) * 255.0).astype(np.uint8)
    rgb = np.stack([scaled] * 3, axis=-1)
    Image.fromarray(rgb, mode="RGB").save(out_path)


def load_split(harvard_dir, png_dir, split_folder):
    src_dir = os.path.join(harvard_dir, split_folder)
    npz_files = sorted(f for f in os.listdir(src_dir) if f.endswith(".npz"))
    assert len(npz_files) > 0, f"No .npz files found under {src_dir}"

    rows = []
    for fname in npz_files:
        subject_id = os.path.splitext(fname)[0]  # e.g. "data_0001"
        subj = load_subject(os.path.join(src_dir, fname))

        png_name = f"{subject_id}.png"
        rnflt_to_png(subj["rnflt"], os.path.join(png_dir, png_name))

        rows.append({
            "subject_id": subject_id,
            "Path": png_name,
            "Glaucoma": subj["glaucoma"],
            # utils.py::accuracy_by_gender checks literal "M"/"F".
            "Sex": "M" if subj["male"] == 1 else "F",
            # PAPILA/MEDFAIR convention (age is continuous here, unlike
            # PAPILA's already-binarized source).
            "Age_binary": int(subj["age"] >= 60),
            # Kept as the raw string category for intersectional grouping --
            # deliberately NOT collapsed to accuracy_by_race_binary's binary
            # 0/1 encoding, which this work doesn't use.
            "Race": subj["race"],
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harvard-dir", required=True, help="Path to the extracted harvard-dataset/ folder (containing Training/, Validation/, Test/)")
    ap.add_argument("--out-dir", required=True, help="Directory to write train.csv/val.csv/test.csv into")
    args = ap.parse_args()

    png_dir = os.path.join(args.harvard_dir, "RNFLT_png")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    split_dfs = {}
    for split_name, folder in SPLIT_FOLDERS.items():
        print(f"Loading {folder}/ -> {split_name}")
        split_dfs[split_name] = load_split(args.harvard_dir, png_dir, folder)
        print(f"  {len(split_dfs[split_name])} subjects")

    train_df, val_df, test_df = split_dfs["train"], split_dfs["val"], split_dfs["test"]

    all_ids = pd.concat([train_df["subject_id"], val_df["subject_id"], test_df["subject_id"]])
    assert all_ids.is_unique, "subject_id collides across Training/Validation/Test -- unexpected, investigate before trusting these splits"

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"\nTotal subjects: {len(combined)}")
    print(f"Glaucoma value counts: {combined['Glaucoma'].value_counts().to_dict()}")
    print(f"Sex value counts: {combined['Sex'].value_counts().to_dict()}")
    print(f"Race value counts: {combined['Race'].value_counts().to_dict()}")
    print(f"Age_binary value counts: {combined['Age_binary'].value_counts().to_dict()}")

    # Assign Intersectional_Group on the FULL combined dataframe, before
    # splitting back out -- guarantees train/val/test share one consistent
    # 0..N-1 numbering (Age_binary x Sex x Race here -> 12 groups) regardless
    # of which groups end up in which split.
    combined["Intersectional_Group"], intersectional_mapping = build_intersectional_group(
        combined, ["Age_binary", "Sex", "Race"]
    )
    print("\nIntersectional group mapping (Age_binary x Sex x Race):")
    print(intersectional_mapping.to_string(index=False))

    group_by_id = combined.set_index("subject_id")["Intersectional_Group"]
    for split_name, d in split_dfs.items():
        d["Intersectional_Group"] = d["subject_id"].map(group_by_id)

    print(f"\ntrain.csv: {len(train_df)} rows (from Training/)")
    print(f"val.csv:   {len(val_df)} rows (from Validation/)")
    print(f"test.csv:  {len(test_df)} rows (from Test/)")
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(name, "-> rows:", len(d), "| class balance:", d["Glaucoma"].value_counts().to_dict())

    cols = ["Path", "Glaucoma", "Sex", "Age_binary", "Race", "Intersectional_Group"]
    train_csv_path = os.path.join(args.out_dir, "train.csv")
    val_csv_path = os.path.join(args.out_dir, "val.csv")
    test_csv_path = os.path.join(args.out_dir, "test.csv")
    mapping_csv_path = os.path.join(args.out_dir, "intersectional_mapping.csv")

    train_df[cols].to_csv(train_csv_path, index=False)
    val_df[cols].to_csv(val_csv_path, index=False)
    test_df[cols].to_csv(test_csv_path, index=False)

    # Per-split group counts. At 12 groups this dataset is far thinner per
    # cell than PAPILA's 4 -- a 0 or single-digit count here is a real
    # signal (too-small subgroup), not something to hide or re-engineer
    # around (see INTERSECTIONAL_FAIRNESS_PLAN.md's "Practical constraints").
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = d["Intersectional_Group"].value_counts().reindex(
            intersectional_mapping["Intersectional_Group"], fill_value=0
        )
        intersectional_mapping[f"count_{name}"] = counts.values
    intersectional_mapping.to_csv(mapping_csv_path, index=False)
    print("\nPer-split group counts:")
    print(intersectional_mapping.to_string(index=False))

    thin_groups = intersectional_mapping[
        (intersectional_mapping["count_val"] < 5) | (intersectional_mapping["count_test"] < 5)
    ]
    if len(thin_groups) > 0:
        print("\nWARNING: the following intersectional groups have <5 rows in val and/or test:")
        print(thin_groups.to_string(index=False))

    print(
        "\nAll sanity checks passed. Wrote:",
        train_csv_path, val_csv_path, test_csv_path, mapping_csv_path,
    )


if __name__ == "__main__":
    main()
