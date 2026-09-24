#!/usr/bin/env python3
"""Shared helper for building a single N-way intersectional group id out of
several demographic attribute columns (e.g. Age_binary x Sex x Race).

Used by preprocess_papila.py (["Age_binary", "Sex"], 4 groups) and
preprocess_glaucoma.py (["Age_binary", "Sex", "Race"], 12 groups).

Group ids come from the cartesian product of each attribute column's SORTED
UNIQUE VALUES -- fixed and deterministic -- not from just the combinations
that happen to be observed in whatever dataframe is passed in. This matters
because build_intersectional_group must be called once on the full dataframe
BEFORE it gets split into train/val/test: that way every split's
Intersectional_Group values share one consistent 0..N-1 numbering, and
`args.num_sens_groups` (computed downstream as `df["Intersectional_Group"].nunique()`
on whichever split happens to be loaded) is the same constant no matter which
groups a particular split's rows actually land in.
"""
import itertools

import pandas as pd


def build_intersectional_group(df: pd.DataFrame, attr_cols: list[str]) -> tuple[pd.Series, pd.DataFrame]:
    """Factorizes the tuple of attr_cols into a single 0..N-1 int group id.

    Returns:
        group_id: pd.Series aligned to df.index, one int per row.
        mapping_df: one row per group id (all N = product of per-column
            cardinalities, even ones with zero rows in df), columns
            ["Intersectional_Group", *attr_cols, "count"].
    """
    assert len(attr_cols) >= 2, "intersectional grouping needs 2+ attribute columns"
    for col in attr_cols:
        assert col in df.columns, f"missing attribute column: {col}"

    value_domains = [sorted(df[col].unique().tolist()) for col in attr_cols]
    combos = list(itertools.product(*value_domains))
    combo_to_id = {combo: i for i, combo in enumerate(combos)}

    row_tuples = list(df[attr_cols].itertuples(index=False, name=None))
    group_id = pd.Series([combo_to_id[t] for t in row_tuples], index=df.index, name="Intersectional_Group")

    counts = group_id.value_counts()
    mapping_rows = []
    for i, combo in enumerate(combos):
        row = {"Intersectional_Group": i}
        row.update(dict(zip(attr_cols, combo)))
        row["count"] = int(counts.get(i, 0))
        mapping_rows.append(row)
    mapping_df = pd.DataFrame(mapping_rows)

    return group_id, mapping_df
