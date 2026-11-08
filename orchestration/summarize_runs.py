#!/usr/bin/env python3
"""Summarize everything under runs/ (written by run_experiments.sh).

Prints and saves to <runs>/summary_*.csv:
  1. Per (dataset, label, attr): Stage-1 seconds and test overall/best/worst AUC/diff, mean +- std over seeds.
  2. Experiment 1 vs 2/3: EXTRAPOLATED full-budget TPE cost (mean measured trial time of the
     exp1 tpe_sample run x --tpe-full-trials, default 200) vs the measured sensitivity runtime.
  3. (optional exp3_ablation) core ablation (intersectional test groups): worst-group AUC of the
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
    ap.add_argument("--tpe-full-trials", type=int, default=int(os.environ.get("TPE_FULL_TRIALS", 200)),
                    help="full TPE budget the exp1 sample is extrapolated to (paper: ~200)")
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
    cols = ["dataset", "label", "attr", "seed", "run_dir", "stage1_s", "stage2_s",
            "train_pct", "lr", "auc_overall", "auc_best", "auc_worst", "auc_diff"]
    runs = pd.DataFrame(rows, columns=cols)
    pd.set_option("display.width", 250, "display.max_columns", 50)

    metrics = ["stage1_s", "stage2_s", "auc_overall", "auc_best", "auc_worst", "auc_diff"]
    if runs.empty:
        print("\n=== 1. Per-cell summary ===\n(no finished runs with results yet)")
    else:
        summ = runs.groupby(["dataset", "label", "attr"])[metrics].agg(["mean", "std", "count"]).round(3)
        print("\n=== 1. Per-cell summary (mean/std over seeds; times in seconds) ===\n", summ.to_string())
        summ.to_csv(os.path.join(args.runs_dir, "summary_cells.csv"))

    # 2. Experiment 1 extrapolation vs measured sensitivity runtime
    print(f"\n=== 2. Original FairTune (TPE) cost, EXTRAPOLATED to {args.tpe_full_trials} trials, vs measured sensitivity ===")
    sp = []
    for d in sorted(glob.glob(os.path.join(args.runs_dir, "*", "tpe_sample", "*", "seed*"))):
        tfiles = glob.glob(os.path.join(d, "Optuna Run Stats", "Trial_Timing_*.csv"))
        if not os.path.exists(os.path.join(d, "DONE")) or not tfiles:
            continue
        ds, _, attr, _ = d.split(os.sep)[-4:]
        tt = pd.read_csv(tfiles[0])
        if tt.empty:
            continue
        # search_mask.py reports to the pruner only after the full epoch loop, so every
        # trial (pruned or not) trains all epochs: cost per trial is ~constant.
        per_trial = tt.duration_seconds.mean()
        rec = {"dataset": ds, "attr": attr, "tpe_trials_run": len(tt),
               "tpe_epochs_per_trial": int(tt.epochs.iloc[0]),
               "tpe_s_per_trial_mean": per_trial, "tpe_s_per_trial_std": tt.duration_seconds.std(ddof=1),
               "tpe_s_per_epoch": tt.seconds_per_epoch.mean(),
               "tpe_stage1_extrapolated_h": per_trial * args.tpe_full_trials / 3600}
        sens = runs[(runs.dataset == ds) & (runs.label == "sens_wg") & (runs.attr == attr) & (runs.seed == 0)]
        if len(sens):
            s1, s2 = float(sens.stage1_s.iloc[0]), float(sens.stage2_s.iloc[0])
            # TPE's Stage 2 is the same fine-tune (same epochs/batch) as sensitivity's.
            rec.update(tpe_total_extrapolated_h=rec["tpe_stage1_extrapolated_h"] + s2 / 3600,
                       sens_stage1_h=s1 / 3600, sens_total_h=(s1 + s2) / 3600)
            rec["speedup_stage1"] = rec["tpe_stage1_extrapolated_h"] / rec["sens_stage1_h"]
            rec["speedup_total"] = rec["tpe_total_extrapolated_h"] / rec["sens_total_h"]
        sp.append(rec)
    sp = pd.DataFrame(sp)
    print(sp.round(3).to_string(index=False) if len(sp) else "(need exp1 tpe_sample runs; sens_wg seed-0 runs for the ratio)")
    print(f"NOTE: tpe_*_extrapolated_h = mean measured trial time x {args.tpe_full_trials} trials; label it 'extrapolated' "
          "in the paper. Sensitivity numbers are measured (seed 0). Timings are only valid for solo-GPU runs.")
    sp.to_csv(os.path.join(args.runs_dir, "summary_speedup.csv"), index=False)

    # 3. exp3_ablation core ablation
    print("\n=== 3. (optional exp3_ablation) Intersectional core ablation (test, intersectional groups) ===")
    inter = runs[(runs.attr == "intersectional") & (runs.label != "tpe_sample")]
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
    print(out.round(4).to_string(index=False) if len(out) else "(no exp3_ablation runs yet)")
    print("A gain counts only if |paired diff| > seed SD AND the (per-seed) bootstrap CI excludes 0.")
    print("NOTE: worst-group AUC ignores groups with a single class (NaN); check group counts in RESULTS_intersectional_groups_*.csv.")
    out.to_csv(os.path.join(args.runs_dir, "summary_phase2.csv"), index=False)


if __name__ == "__main__":
    main()
