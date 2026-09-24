#!/usr/bin/env python3
"""Compare the one-shot sensitivity scores from search_mask_sensitivity.py
against the empirical component-selection frequency of a real Optuna/TPE
search_mask.py run, via Spearman rank correlation. This is the analysis
figure fairtune-speedup-plan.md calls "the thesis of the report in one
plot" -- if a single gradient pass predicts what 48 GPU-hours of blind
search discovered, that's the headline result.

Inputs:
  --optuna-stats-csv      An EXISTING search_mask.py output:
                           "<model>/<dataset>/Optuna Run Stats/
                           Run_Stats_<sens_attribute>_auto_peft1_<model>_
                           <objective_metric>.csv", written unconditionally
                           by search_mask.py's __main__ via
                           study.trials_dataframe() (no --disable_storage
                           guard). One row per Optuna trial, with a
                           "params_Mask Idx {i}" column for each of the 36
                           auto_peft1 components (0/1, the bit that trial
                           sampled) and a `state` column
                           (COMPLETE/PRUNED/FAIL). No new export code is
                           needed anywhere in search_mask.py for this --
                           it already saves everything this script needs.
  --sensitivity-scores-csv  "<model>/<dataset>/Sensitivity_Scores/
                             <sens_attribute>/scores_<rank_method>.csv",
                             written by search_mask_sensitivity.py's
                             compute_sensitivity_scores(). One row per
                             component with `component_idx`/
                             `normalized_score`.

Both CSVs must be for the SAME dataset/sens_attribute (this script doesn't
cross-check that beyond requiring exactly 36 components to merge cleanly --
pass matching files yourself).
"""
import argparse

import pandas as pd


def selection_frequency(optuna_stats_df, num_components=36):
    """Mean of each 'params_Mask Idx {i}' column across the (already
    state-filtered) trials -- the empirical proportion of trials in which
    Optuna's TPE sampler turned component i on."""
    rows = []
    for i in range(num_components):
        col = f"params_Mask Idx {i}"
        assert col in optuna_stats_df.columns, (
            f"{col!r} not found in --optuna-stats-csv. Was it produced by "
            "search_mask.py with --tuning_method auto_peft1 (36 components)?"
        )
        rows.append({"component_idx": i, "selection_freq": optuna_stats_df[col].mean()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--optuna-stats-csv", required=True,
        help="Optuna Run Stats/Run_Stats_<attr>_auto_peft1_<model>_<objective_metric>.csv from a real search_mask.py run",
    )
    ap.add_argument(
        "--sensitivity-scores-csv", required=True,
        help="Sensitivity_Scores/<attr>/scores_<rank_method>.csv from search_mask_sensitivity.py",
    )
    ap.add_argument(
        "--include-pruned", action="store_true",
        help="Include PRUNED trials, not just COMPLETE ones (default: COMPLETE only -- a pruned "
        "trial's objective value reflects an early-stopped training run, a noisier selection signal)",
    )
    args = ap.parse_args()

    optuna_stats_df = pd.read_csv(args.optuna_stats_csv)
    scores_df = pd.read_csv(args.sensitivity_scores_csv)

    if not args.include_pruned:
        before = len(optuna_stats_df)
        optuna_stats_df = optuna_stats_df[optuna_stats_df["state"] == "COMPLETE"]
        print(
            f"Filtered to COMPLETE trials only: {len(optuna_stats_df)}/{before} "
            "(pass --include-pruned to keep pruned trials too)"
        )
    assert len(optuna_stats_df) > 0, f"No trials left in {args.optuna_stats_csv} after filtering"

    freq_df = selection_frequency(optuna_stats_df, num_components=len(scores_df))

    merged = scores_df.merge(freq_df, on="component_idx", how="inner")
    assert len(merged) == len(scores_df) == len(freq_df), (
        f"Merge produced {len(merged)} rows; expected {len(scores_df)} -- "
        "component_idx doesn't line up 1:1 between the two CSVs."
    )

    correlation = merged["normalized_score"].corr(merged["selection_freq"], method="spearman")

    print(f"\n{len(optuna_stats_df)} trials from {args.optuna_stats_csv}")
    print(f"{len(scores_df)} components from {args.sensitivity_scores_csv}")
    print(
        f"\nSpearman rank correlation (one-shot sensitivity score vs. "
        f"empirical TPE selection frequency): {correlation:.4f}"
    )

    display_cols = [
        "component_idx", "block_idx", "bit_idx", "component_type",
        "normalized_score", "selection_freq",
    ]
    display = merged[display_cols].copy()
    display["score_rank"] = display["normalized_score"].rank(ascending=False, method="first").astype(int)
    display["selection_rank"] = display["selection_freq"].rank(ascending=False, method="first").astype(int)
    display = display.sort_values("score_rank")
    print("\nPer-component comparison (sorted by one-shot sensitivity score, best first):")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
