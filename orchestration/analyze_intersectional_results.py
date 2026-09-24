#!/usr/bin/env python3
"""Recombine a jointly-optimized intersectional model's per-group results
into marginal (single-attribute) fairness gaps, and compare those against
standalone single-attribute runs.

Inputs:
  --groups-csv   RESULTS_intersectional_groups_<objective_metric>.csv,
                 written by finetune_with_mask.py's "intersectional" branch.
                 One row per (Tuning Method, Mask Path, Intersectional_Group)
                 with raw `correct`/`count`, plus `acc`/`auc`.
  --mapping-csv  intersectional_mapping.csv, written by preprocess_papila.py
                 or preprocess_glaucoma.py. Decodes each Intersectional_Group
                 id back into its original attribute values (e.g.
                 Age_binary=1, Sex=F[, Race=White]).

Marginal gaps are computed by summing raw correct/count across every
Intersectional_Group that shares a given attribute value, THEN dividing --
weighted by how many samples actually fall in each joint group, not a naive
mean of per-group accuracies (which would silently over-weight small
groups). This is the whole reason evaluate_fairness_intersectional
(utilities/training_utils.py) logs raw counts instead of only accuracy%.
"""
import argparse
import os

import pandas as pd


def marginal_gap(groups_df, mapping_df, attr_col):
    """Recombine per-Intersectional_Group raw counts into a marginal
    accuracy per value of `attr_col` (e.g. Age_binary, Sex, Race), weighted
    by raw sample count. Returns (per_value_df, gap) where gap is the
    max-minus-min marginal accuracy across that attribute's values.
    """
    merged = groups_df.merge(
        mapping_df[["Intersectional_Group", attr_col]], on="Intersectional_Group"
    )
    agg = merged.groupby(attr_col)[["correct", "count"]].sum()
    agg["accuracy"] = (agg["correct"] / agg["count"] * 100).round(3)
    gap = round(agg["accuracy"].max() - agg["accuracy"].min(), 3)
    return agg, gap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groups-csv", required=True, help="RESULTS_intersectional_groups_<objective_metric>.csv")
    ap.add_argument("--mapping-csv", required=True, help="intersectional_mapping.csv from the matching preprocess_*.py run")
    ap.add_argument(
        "--attrs", nargs="+", default=None,
        help="Attribute columns to compute marginals for (default: every mapping_csv column except Intersectional_Group and count_*)",
    )
    ap.add_argument(
        "--compare-csv", nargs="*", default=[],
        help="Standalone single-attribute RESULTS_<attr>_<objective_metric>.csv files to print alongside the recombined marginals (e.g. RESULTS_age_min_auc.csv RESULTS_gender_min_auc.csv)",
    )
    ap.add_argument(
        "--tuning-method", default=None,
        help="If groups-csv has rows from multiple runs, filter to just this Tuning Method before recombining",
    )
    args = ap.parse_args()

    groups_df = pd.read_csv(args.groups_csv)
    mapping_df = pd.read_csv(args.mapping_csv)

    if args.tuning_method is not None:
        groups_df = groups_df[groups_df["Tuning Method"] == args.tuning_method]
        assert len(groups_df) > 0, f"No rows for Tuning Method={args.tuning_method!r} in {args.groups_csv}"

    if len(groups_df["Mask Path"].unique()) > 1 and args.tuning_method is None:
        print(
            f"NOTE: {args.groups_csv} has {len(groups_df['Mask Path'].unique())} distinct runs "
            "(Mask Path varies) -- recombining across ALL of them together. "
            "Pass --tuning-method to isolate one run."
        )

    attr_cols = args.attrs or [
        c for c in mapping_df.columns
        if c != "Intersectional_Group" and not c.startswith("count")
    ]
    assert len(attr_cols) > 0, "No attribute columns found in mapping_csv to compute marginals for"

    print(f"Recombining {len(groups_df)} group-rows from {args.groups_csv}")
    print(f"Attribute columns: {attr_cols}\n")

    for attr in attr_cols:
        agg, gap = marginal_gap(groups_df, mapping_df, attr)
        print(f"Marginal {attr} accuracy (weighted by raw count):")
        print(agg.to_string())
        print(f"{attr} accuracy gap (max - min): {gap}\n")

    for path in args.compare_csv:
        if not os.path.exists(path):
            print(f"WARNING: --compare-csv path does not exist, skipping: {path}")
            continue
        standalone = pd.read_csv(path)
        print(f"Standalone comparison file: {path}")
        print(standalone.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
