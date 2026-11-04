#!/usr/bin/env python3
"""Summarize everything under runs/ (written by run_experiments.sh).

Prints and saves to <runs>/summary_*.csv:
  1. Per (dataset, label, attr): Stage-1 seconds and test overall/best/worst AUC/diff, mean +- std over seeds.
  2. Speed-up: TPE(10 trials) vs sensitivity Stage-1 wall-clock, plus an EXTRAPOLATED
     full-budget TPE cost (scaled by the measured tpe50/tpe10 ratio of the calibration run).
  3. Phase 2 core ablation (intersectional test groups): worst-group AUC of the
     intersectional-searched mask vs single-attribute masks vs full-FT / head-only baselines,
     with paired seed differences and a paired bootstrap CI over test samples
     (needs test_predictions_intersectional.npz, saved by finetune_with_mask.py).

Result CSV columns are read positionally (row layout shared by every attribute branch of
finetune_with_mask.py): [Tuning Method, Train Percent, LR, Test AUC Overall, best, worst, diff, ...].
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load_run_result(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "RESULTS_*_min_auc.csv")))
    files = [f for f in files if "intersectional_groups" not in os.path.basename(f)]
    if not files:
        return None
    df = pd.read_csv(files[0])
    if df.empty:
        return None
    r = df.iloc[-1]
    return {
        "train_pct": r.iloc[1], "lr": r.iloc[2], "auc_overall": r.iloc[3],
        "auc_best": r.iloc[4], "auc_worst": r.iloc[5], "auc_diff": r.iloc[6],
    }


def worst_group_auc(prob, label, group, idx=None):
    if idx is not None:
        prob, label, group = prob[idx], label[idx], group[idx]
    aucs = []
    for g in np.unique(group):
        m = group == g
        if len(np.unique(label[m])) == 2:
            aucs.append(roc_auc_score(label[m], prob[m]))
    return np.min(aucs) if aucs else np.nan


def paired_bootstrap(pred_a, pred_b, n_boot=1000, seed=0):
    """Paired bootstrap of worst-group AUC difference (a - b), resampling test samples
    within each group (stratified) so group sizes stay fixed."""
    rng = np.random.default_rng(seed)
    label, group = pred_a["label"], pred_a["group"]
    if not (np.array_equal(label, pred_b["label"]) and np.array_equal(group, pred_b["group"])):
        return None
    by_group = [np.where(group == g)[0] for g in np.unique(group)]
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in by_group])
        a = worst_group_auc(pred_a["prob"], label, group, idx)
        b = worst_group_auc(pred_b["prob"], label, group, idx)
        if not (np.isnan(a) or np.isnan(b)):
            diffs.append(a - b)
    if not diffs:
        return None
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"))
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    timings = pd.read_csv(os.path.join(args.runs_dir, "timings.csv"))
    rows = []
    for d in glob.glob(os.path.join(args.runs_dir, "*", "*", "*", "seed*")):
        if not os.path.exists(os.path.join(d, "DONE")):
            continue
        ds, label, attr, seedname = d.split(os.sep)[-4:]
        seed = int(seedname.replace("seed", ""))
        res = load_run_result(d)
        if res is None:
            continue
        t = timings[(timings.dataset == ds) & (timings.label == label) & (timings.attr == attr) & (timings.seed == seed)]
        res.update(dataset=ds, label=label, attr=attr, seed=seed, run_dir=d,
                   stage1_s=t.stage1_seconds.iloc[-1] if len(t) else np.nan,
                   stage2_s=t.stage2_seconds.iloc[-1] if len(t) else np.nan)
        rows.append(res)
    runs = pd.DataFrame(rows)
    if runs.empty:
        print("No finished runs found under", args.runs_dir)
        return

    metrics = ["stage1_s", "stage2_s", "auc_overall", "auc_best", "auc_worst", "auc_diff"]
    summ = runs.groupby(["dataset", "label", "attr"])[metrics].agg(["mean", "std", "count"]).round(3)
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print("\n=== 1. Per-cell summary (mean/std over seeds) ===\n", summ.to_string())
    summ.to_csv(os.path.join(args.runs_dir, "summary_cells.csv"))

    # 2. speed-up
    print("\n=== 2. Stage-1 speed-up (sensitivity vs TPE) ===")
    sp = []
    for (ds, attr), g in runs.groupby(["dataset", "attr"]):
        t10 = g[g.label == "tpe10"].stage1_s.mean()
        ts = g[g.label == "sens_wg"].stage1_s.mean()
        if np.isnan(t10) or np.isnan(ts):
            continue
        sp.append({"dataset": ds, "attr": attr, "tpe10_s": t10, "sens_s": ts, "ratio_vs_tpe10": t10 / ts})
    sp = pd.DataFrame(sp)
    for ds in sp.dataset.unique() if len(sp) else []:
        cal = runs[(runs.dataset == ds) & (runs.label == "tpe50")].stage1_s.mean()
        base = runs[(runs.dataset == ds) & (runs.label == "tpe10") & (runs.attr == "gender")].stage1_s.mean()
        if not (np.isnan(cal) or np.isnan(base)):
            f = cal / base
            m = sp.dataset == ds
            sp.loc[m, "tpe50_extrapolated_s"] = sp.loc[m, "tpe10_s"] * f
            sp.loc[m, "ratio_vs_tpe50_extrapolated"] = sp.loc[m, "tpe50_extrapolated_s"] / sp.loc[m, "sens_s"]
    print(sp.round(2).to_string(index=False) if len(sp) else "(need both tpe10 and sens_wg runs)")
    print("NOTE: *_extrapolated columns scale tpe10 by the measured tpe50/tpe10 ratio (gender); label them 'extrapolated' in the paper.")
    sp.to_csv(os.path.join(args.runs_dir, "summary_speedup.csv"), index=False)

    # 3. Phase 2 core ablation
    print("\n=== 3. Intersectional core ablation (test, intersectional groups) ===")
    inter = runs[(runs.attr == "intersectional") & (runs.label != "tpe10")]
    out = []
    for ds, g in inter.groupby("dataset"):
        ref = g[g.label == "sens_wg"].set_index("seed")
        for label, gl in g.groupby("label"):
            if label == "sens_wg":
                continue
            gl = gl.set_index("seed")
            common = sorted(set(ref.index) & set(gl.index))
            if not common:
                continue
            dw = np.array([ref.loc[s, "auc_worst"] - gl.loc[s, "auc_worst"] for s in common], dtype=float)
            rec = {"dataset": ds, "compared_to": label, "n_seeds": len(common),
                   "worst_auc_intersectional_mask": ref.loc[common, "auc_worst"].mean(),
                   "worst_auc_other": gl.loc[common, "auc_worst"].mean(),
                   "paired_diff_mean": dw.mean(), "paired_diff_std": dw.std(ddof=1) if len(dw) > 1 else np.nan}
            rec["exceeds_seed_sd"] = bool(len(dw) > 1 and abs(dw.mean()) > dw.std(ddof=1))
            boots = []
            for s in common:
                pa = os.path.join(ref.loc[s, "run_dir"], "test_predictions_intersectional.npz")
                pb = os.path.join(gl.loc[s, "run_dir"], "test_predictions_intersectional.npz")
                if os.path.exists(pa) and os.path.exists(pb):
                    r = paired_bootstrap(dict(np.load(pa)), dict(np.load(pb)), n_boot=args.n_boot, seed=s)
                    if r:
                        boots.append(r)
            if boots:
                b = np.array(boots)
                rec.update(boot_mean=b[:, 0].mean(), boot_ci_lo_mean=b[:, 1].mean(), boot_ci_hi_mean=b[:, 2].mean())
            out.append(rec)
    out = pd.DataFrame(out)
    print(out.round(4).to_string(index=False) if len(out) else "(no Phase 2 runs yet)")
    print("A gain counts only if |paired diff| > seed SD AND the (per-seed) bootstrap CI excludes 0.")
    print("NOTE: worst-group AUC ignores groups with a single class (NaN); check group counts in RESULTS_intersectional_groups_*.csv.")
    out.to_csv(os.path.join(args.runs_dir, "summary_phase2.csv"), index=False)


if __name__ == "__main__":
    main()
