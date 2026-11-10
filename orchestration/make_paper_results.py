#!/usr/bin/env python3
"""Build every table (LaTeX) and figure (PDF + PNG) for the paper's Results section from runs/.

Usage:  python orchestration/make_paper_results.py [--runs-dir runs] [--out paper_results] [--n-boot 2000]

Writes  <out>/tables/T*.tex   (booktabs; \\usepackage{booktabs} in the preamble)
        <out>/figures/F*.pdf/.png  (sized for IEEE: 3.5in column / 7.16in page width)
        <out>/data/*.csv      (the numbers behind each table/figure)
        (paper_results/README.md is maintained by hand: what each item is + caveats)

Conventions
- "Worst"/"best" = min/max group test AUC over groups with both classes present (NaN groups skipped).
- Bootstrap CIs: groups x label stratified resampling of the test set. Every model that is compared on
  the intersectional groups (T3, T4, F2, F3, T5) comes from the `paired_*` re-runs: the same masks/LRs as
  the original cells, fine-tuned again with test predictions dumped in a fixed order, so all runs of a
  dataset share one (label, group) array (asserted). Paired-difference CIs therefore apply the SAME
  resample indices to both models. Seed-level variation is reported separately (SD).
- Stage-1 artefacts (masks, k-sweeps, scores, timings) come from the original cells (sens_wg, tpe_*).
- Sensitivity runtimes are seed 0 (machine idle). TPE: 5-trial sample extrapolated to 200; PAPILA
  intersectional also has a measured full 200-trial run (pause-corrected via runs/tpe_full_pauses.txt).
"""
import argparse
import glob
import os
import re
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

# Reference categorical palette (slots 1-4, fixed order), text inks, sequential blue ramp.
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100"}
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#d9d8d2"
SEQ = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
COL_W, PAGE_W = 3.5, 7.16

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "lines.linewidth": 1.5, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300,
})

DS_NAME = {"papila": "PAPILA", "glaucoma": "Harvard-GF"}
ATTR_NAME = {"gender": "Sex", "age": "Age", "race": "Race", "intersectional": "Intersectional"}
RACE_SHORT = {"Asian": "Asian", "Black or African American": "Black", "White or Caucasian": "White"}
METHOD_ROWS = [  # (label dir, attr dir, display name)
    ("base_full_ft", "intersectional", "Full fine-tuning"),
    ("base_head_only", "intersectional", "Head only"),
    ("xmask_gender", "intersectional", "Mask searched on sex"),
    ("xmask_age", "intersectional", "Mask searched on age"),
    ("xmask_race", "intersectional", "Mask searched on race"),
    ("sens_wg", "intersectional", "Mask searched on intersection (ours)"),
]
OURS = "paired_sens_wg"  # the intersectional model used everywhere models are compared
EXTRA_ROWS = [  # seed-0-only rows, shown under a rule
    ("sens_random", "intersectional", "  Ranking ablation: random"),
    ("sens_erm", "intersectional", "  Ranking ablation: ERM sensitivity"),
    ("tpe_full", "intersectional", "  Original FairTune, TPE 200 trials"),
]


# ----------------------------------------------------------------------------- helpers
def auc_fast(y, p):
    y = np.asarray(y)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def group_aucs(d, groups):
    return np.array([auc_fast(d["label"][d["group"] == g], d["prob"][d["group"] == g]) for g in groups])


def worst_best(a):
    return np.nanmin(a), np.nanmax(a)


def strata_index(d):
    keys = d["group"] * 2 + d["label"]
    return [np.flatnonzero(keys == k) for k in np.unique(keys)]


def boot_indices(d, n_boot, rng):
    strata = strata_index(d)
    return [np.concatenate([rng.choice(s, len(s)) for s in strata]) for _ in range(n_boot)]


def boot_worst_idx(d, groups, idxs):
    """Worst-group AUC under fixed resample indices (paired bootstrap)."""
    return np.array([np.nanmin(group_aucs({k: d[k][i] for k in ("label", "prob", "group")}, groups)) for i in idxs])


def boot_groups(d, groups, n_boot, rng):
    strata = strata_index(d)
    out = np.empty((n_boot, len(groups)))
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(s, len(s)) for s in strata])
        out[b] = group_aucs({k: d[k][idx] for k in ("label", "prob", "group")}, groups)
    return out


def pm(m, s, nd=3):
    return f"{m:.{nd}f}" if (s is None or np.isnan(s)) else f"{m:.{nd}f} $\\pm$ {s:.{nd}f}"


def ms(x, nd=3):
    x = np.asarray(x, float)
    return pm(np.nanmean(x), np.nanstd(x, ddof=1) if len(x) > 1 else np.nan, nd)


def seeds_of(runs, ds, label, attr):
    out = []
    for p in sorted(glob.glob(f"{runs}/{ds}/{label}/{attr}/seed*/DONE")):
        out.append(int(re.search(r"seed(\d+)", p).group(1)))
    return sorted(out)


def cell(runs, ds, label, attr, seed):
    return f"{runs}/{ds}/{label}/{attr}/seed{seed}"


def load_npz(path):
    z = np.load(path)
    return {k: z[k] for k in ("prob", "label", "group")}


def results_row(c):
    f = glob.glob(f"{c}/RESULTS_*_min_auc.csv")
    f = [x for x in f if "groups" not in x]
    return pd.read_csv(f[0]).iloc[0] if f else None


def mask_of(c):
    m = glob.glob(f"{c}/Optuna_Masks/*/*.npy")
    return np.load(m[0]).astype(int) if m else None


def save_table(out, name, latex, df=None):
    with open(f"{out}/tables/{name}.tex", "w") as f:
        f.write(latex)
    if df is not None:
        df.to_csv(f"{out}/data/{name}.csv", index=False)


def save_fig(fig, out, name):
    fig.savefig(f"{out}/figures/{name}.pdf")
    fig.savefig(f"{out}/figures/{name}.png")
    plt.close(fig)


def group_labels(repo, ds):
    m = pd.read_csv(f"{repo}/data_splits_{ds}/intersectional_mapping.csv")
    labs = []
    for _, r in m.iterrows():
        age = "Young" if int(r["Age_binary"]) == 0 else "Older"
        sex = "F" if r["Sex"] == "F" else "M"
        lab = f"{age} {sex}"
        if "Race" in m.columns:
            lab = f"{age} {RACE_SHORT.get(r['Race'], r['Race'])} {sex}"
        labs.append(lab)
    return m, labs


def pause_intervals(runs):
    p = f"{runs}/tpe_full_pauses.txt"
    if not os.path.exists(p):
        return []
    ev = [(ln.split()[0], int(ln.split()[-1])) for ln in open(p) if ln.strip()]
    iv, start = [], None
    for kind, t in ev:
        if kind == "PAUSED":
            start = t
        elif kind == "RESUMED" and start is not None:
            iv.append((datetime.fromtimestamp(start), datetime.fromtimestamp(t)))
            start = None
    return iv


def overlap_s(a0, a1, iv):
    return sum(max(0.0, (min(a1, b1) - max(a0, b0)).total_seconds()) for b0, b1 in iv)


# ----------------------------------------------------------------------------- T1 + F1: compute
def t1_compute(runs, out):
    tim = pd.read_csv(f"{runs}/timings.csv")
    rows = []
    for ds, attrs in (("papila", ["gender", "age", "intersectional"]),
                      ("glaucoma", ["gender", "age", "race", "intersectional"])):
        for attr in attrs:
            tt = glob.glob(f"{runs}/{ds}/tpe_sample/{attr}/seed0/Optuna Run Stats/Trial_Timing_*.csv")
            if not tt:
                continue
            tt = pd.read_csv(tt[0])
            sens = cell(runs, ds, "sens_wg", attr, 0)
            st = glob.glob(f"{sens}/Sensitivity_Results/*/timing_*.csv")
            st = pd.read_csv(st[0]).iloc[0]
            tr = tim[(tim.dataset == ds) & (tim.label == "sens_wg") & (tim.attr == attr) & (tim.seed == 0)].iloc[-1]
            tpe_s = tt.duration_seconds.mean()
            tpe_ext = tpe_s * 200
            r = dict(dataset=DS_NAME[ds], grouping=ATTR_NAME[attr], tpe_s_per_trial=tpe_s,
                     tpe_s_per_trial_sd=tt.duration_seconds.std(ddof=1), tpe_200_h=tpe_ext / 3600,
                     scoring_min=st.scoring_seconds / 60, sweep_h=st.sweep_train_seconds / 3600,
                     stage2_min=tr.stage2_seconds / 60, sens_search_h=tr.stage1_seconds / 3600,
                     sens_total_h=tr.total_seconds / 3600,
                     speedup_search=tpe_ext / tr.stage1_seconds,
                     speedup_total=(tpe_ext + tr.stage2_seconds) / tr.total_seconds, tpe_measured_h=np.nan)
            # measured full TPE (pause corrected)
            full = cell(runs, ds, "tpe_full", attr, 0)
            if os.path.exists(f"{full}/DONE"):
                ftr = tim[(tim.dataset == ds) & (tim.label == "tpe_full") & (tim.attr == attr)].iloc[-1]
                iv = pause_intervals(runs)
                paused = sum((b - a).total_seconds() for a, b in iv)
                r["tpe_measured_h"] = (ftr.stage1_seconds - paused) / 3600
            rows.append(r)
    return pd.DataFrame(rows)


def t1_table(df, out):
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Search cost. TPE: original FairTune, mean seconds per trial over a 5-trial sample, "
             r"extrapolated to 200 trials (PAPILA intersectional also measured over a full 200-trial run, "
             r"in parentheses). Sensitivity: seed 0, scoring pass + 10-point $k$-sweep + Stage~2 fine-tune. "
             r"Speed-up = TPE (200) / sensitivity search; total includes one Stage~2 for both.}",
             r"\label{tab:compute}", r"\small",
             r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{TPE} & \multicolumn{4}{c}{Sensitivity (ours)} & \multicolumn{2}{c}{Speed-up} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-8}\cmidrule(lr){9-10}",
             r"Dataset & Grouping & s/trial & 200 trials (h) & Scoring (min) & Sweep (h) & Stage 2 (min) & Total (h) & Search & Total \\",
             r"\midrule"]
    prev = None
    for _, r in df.iterrows():
        ds = r.dataset if r.dataset != prev else ""
        if prev is not None and r.dataset != prev:
            lines.append(r"\midrule")
        prev = r.dataset
        tpe200 = f"{r.tpe_200_h:.1f}"
        if not np.isnan(r.tpe_measured_h):
            tpe200 += f" ({r.tpe_measured_h:.1f})"
        lines.append(f"{ds} & {r.grouping} & {r.tpe_s_per_trial:.0f} $\\pm$ {r.tpe_s_per_trial_sd:.0f} & {tpe200} & "
                     f"{r.scoring_min:.1f} & {r.sweep_h:.2f} & {r.stage2_min:.1f} & {r.sens_total_h:.2f} & "
                     f"{r.speedup_search:.0f}$\\times$ & {r.speedup_total:.0f}$\\times$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    save_table(out, "T1_compute", "\n".join(lines), df)


def f1_cost_quality(runs, out):
    """PAPILA intersectional: best-so-far validation worst-group AUC vs cumulative GPU hours."""
    ds, attr = "papila", "intersectional"
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    data = []
    full = cell(runs, ds, "tpe_full", attr, 0)
    rs = glob.glob(f"{full}/Optuna Run Stats/Run_Stats_*.csv")
    have_tpe = bool(rs) and os.path.exists(f"{full}/DONE")
    if have_tpe:
        r = pd.read_csv(rs[0]).sort_values("number").reset_index(drop=True)
        r["s"] = pd.to_datetime(r.datetime_start)
        r["e"] = pd.to_datetime(r.datetime_complete)
        iv = pause_intervals(runs)
        r["dur"] = [(e - s).total_seconds() - overlap_s(s, e, iv) for s, e in zip(r.s, r.e)]
        r["cum_h"] = r.dur.cumsum() / 3600
        comp = r.state == "COMPLETE"
        r["best"] = r.min_auc.where(comp).ffill().cummax()  # ffill: pruned trials keep the line going
        ax.scatter(r.cum_h[~comp], r.min_auc[~comp], s=4, color=C["blue"], alpha=0.25, lw=0,
                   label="TPE trial (pruned)")
        ax.scatter(r.cum_h[comp], r.min_auc[comp], s=9, color=C["blue"], lw=0, label="TPE trial (completed)")
        ax.step(r.cum_h, r.best, where="post", color=C["blue"], label="TPE best so far")
        for _, x in r.iterrows():
            data.append(dict(method="tpe", seed=0, step=int(x.number), state=x.state, cum_h=x.cum_h,
                             value=x.min_auc, best_so_far=x.best))
    sens_curves = []
    for i, seed in enumerate(seeds_of(runs, ds, "sens_wg", attr)):
        c = cell(runs, ds, "sens_wg", attr, seed)
        ks = pd.read_csv(glob.glob(f"{c}/Sensitivity_Results/*/k_sweep_*.csv")[0])
        st = pd.read_csv(glob.glob(f"{c}/Sensitivity_Results/*/timing_*.csv")[0]).iloc[0]
        cum = (st.scoring_seconds + ks.wall_clock_seconds.cumsum()) / 3600
        best = ks.objective_metric_value.cummax()
        ax.step(np.r_[st.scoring_seconds / 3600, cum], np.r_[np.nan, best], where="post", color=C["orange"],
                lw=0.6, alpha=0.35, label="Sensitivity, individual seeds" if i == 0 else None)
        sens_curves.append((cum.to_numpy(), best.to_numpy()))
        for j, x in ks.iterrows():
            data.append(dict(method="sensitivity", seed=seed, step=int(x.k), state="k", cum_h=cum[j],
                             value=x.objective_metric_value, best_so_far=best[j]))
    if sens_curves:
        xs = np.mean([c[0] for c in sens_curves], axis=0)
        ys = np.median([c[1] for c in sens_curves], axis=0)
        ax.step(xs, ys, where="post", color=C["orange"], lw=1.8, label="Sensitivity best so far (median of seeds)")
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
    ax.xaxis.set_major_locator(FixedLocator([0.03, 0.1, 0.3, 1, 3, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Cumulative search time (GPU hours, log scale)")
    ax.set_ylabel("Validation worst-group AUC")
    ax.set_title("PAPILA, age $\\times$ sex groups" + ("" if have_tpe else "  [TPE run pending]"), loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=6)
    save_fig(fig, out, "F1_cost_vs_quality")
    pd.DataFrame(data).to_csv(f"{out}/data/F1_cost_vs_quality.csv", index=False)


# ----------------------------------------------------------------------------- T2 single attribute
def t2_single(runs, out):
    rows = []
    for ds, attrs in (("papila", ["gender", "age"]), ("glaucoma", ["gender", "age", "race"])):
        for attr in attrs:
            vals = []
            for seed in seeds_of(runs, ds, "sens_wg", attr):
                r = results_row(cell(runs, ds, "sens_wg", attr, seed))
                if attr == "gender":
                    g = [r["Test AUC Male"], r["Test AUC Female"]]
                    best, worst = max(g), min(g)
                else:
                    best, worst = r["Test AUC (Best)"], r["Test AUC (Worst)"]
                vals.append(dict(seed=seed, overall=r["Test AUC Overall"], best=best, worst=worst,
                                 gap=best - worst, eod=r["EquiOdd_diff"], dpd=r["DPD"], train_pct=r["Train Percent"]))
            v = pd.DataFrame(vals)
            rows.append(dict(dataset=DS_NAME[ds], grouping=ATTR_NAME[attr], n_seeds=len(v),
                             **{c: v[c].mean() for c in ["overall", "best", "worst", "gap", "eod", "dpd", "train_pct"]},
                             **{c + "_sd": v[c].std(ddof=1) for c in ["overall", "best", "worst", "gap", "eod", "dpd"]}))
            v.assign(dataset=ds, attr=attr).to_csv(f"{out}/data/T2_{ds}_{attr}_per_seed.csv", index=False)
    df = pd.DataFrame(rows)
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Single-attribute fairness: test results of the sensitivity-searched model for each "
             r"single sensitive attribute (mean $\pm$ SD over seeds; PAPILA 10 seeds, Harvard-GF 3). "
             r"EOD = equalized-odds difference, DPD = demographic-parity difference (lower is fairer).}",
             r"\label{tab:single}", r"\small", r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Dataset & Attribute & Overall AUC & Best group & Worst group & Gap & EOD & DPD \\", r"\midrule"]
    prev = None
    for _, r in df.iterrows():
        if prev is not None and r.dataset != prev:
            lines.append(r"\midrule")
        ds = r.dataset if r.dataset != prev else ""
        prev = r.dataset
        lines.append(f"{ds} & {r.grouping} & " + " & ".join(pm(r[c], r[c + '_sd']) for c in
                     ["overall", "best", "worst", "gap", "eod", "dpd"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    save_table(out, "T2_single_attribute", "\n".join(lines), df)


# ----------------------------------------------------------------------------- T3 audits
def t3_audits(runs, repo, out):
    rows = []
    for ds in ("papila", "glaucoma"):
        m, _ = group_labels(repo, ds)
        cols = [("Sex", "Sex"), ("Age", "Age_binary")] + ([("Race", "Race")] if "Race" in m.columns else [])
        per = []
        for seed in seeds_of(runs, ds, OURS, "intersectional"):
            d = load_npz(f"{cell(runs, ds, OURS, 'intersectional', seed)}/test_predictions_intersectional.npz")
            gi = d["group"]
            for name, col in cols:
                lab = m.set_index("Intersectional_Group")[col].reindex(gi).to_numpy()
                a = np.array([auc_fast(d["label"][lab == v], d["prob"][lab == v]) for v in pd.unique(lab)])
                per.append(dict(seed=seed, audit=name, n_groups=len(a), worst=np.nanmin(a), gap=np.nanmax(a) - np.nanmin(a)))
            a = group_aucs(d, sorted(m.Intersectional_Group))
            per.append(dict(seed=seed, audit=f"Intersectional ({len(a)} groups)", n_groups=len(a),
                            worst=np.nanmin(a), gap=np.nanmax(a) - np.nanmin(a)))
        p = pd.DataFrame(per)
        p.assign(dataset=ds).to_csv(f"{out}/data/T3_{ds}_per_seed.csv", index=False)
        for audit, g in p.groupby("audit", sort=False):
            rows.append(dict(dataset=DS_NAME[ds], audit=audit, n_seeds=len(g), worst=g.worst.mean(),
                             worst_sd=g.worst.std(ddof=1), gap=g.gap.mean(), gap_sd=g.gap.std(ddof=1)))
    df = pd.DataFrame(rows)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{One model, four audits: the intersectionally-searched model's test predictions grouped "
             r"by a single attribute vs.\ by the full intersection (mean $\pm$ SD over seeds).}",
             r"\label{tab:audit}", r"\small", r"\begin{tabular}{llcc}", r"\toprule",
             r"Dataset & Audit grouping & Worst-group AUC & Gap \\", r"\midrule"]
    prev = None
    for _, r in df.iterrows():
        if prev is not None and r.dataset != prev:
            lines.append(r"\midrule")
        ds = r.dataset if r.dataset != prev else ""
        prev = r.dataset
        lines.append(f"{ds} & {r.audit} & {pm(r.worst, r.worst_sd)} & {pm(r.gap, r.gap_sd)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    save_table(out, "T3_single_vs_intersectional_audit", "\n".join(lines), df)


# ----------------------------------------------------------------------------- T4 + F2 ablation
def t4_ablation(runs, repo, out, n_boot, rng):
    allrows, dots = [], []
    for ds in ("papila", "glaucoma"):
        m, _ = group_labels(repo, ds)
        groups = sorted(m.Intersectional_Group)
        ref_seeds = seeds_of(runs, ds, OURS, "intersectional")
        ref = {s: load_npz(f"{cell(runs, ds, OURS, 'intersectional', s)}/test_predictions_intersectional.npz")
               for s in ref_seeds}
        d0 = ref[ref_seeds[0]]
        idxs = boot_indices(d0, n_boot, rng)  # one set of resamples, shared by every model of this dataset
        ref_boot = {s: boot_worst_idx(ref[s], groups, idxs) for s in ref_seeds}
        for src_label, attr, name in METHOD_ROWS + EXTRA_ROWS:
            label = "paired_" + src_label
            seeds = seeds_of(runs, ds, label, attr)
            if not seeds:
                continue
            per, boots = [], []
            for s in seeds:
                c = cell(runs, ds, label, attr, s)
                d = load_npz(f"{c}/test_predictions_intersectional.npz")
                assert np.array_equal(d["label"], d0["label"]) and np.array_equal(d["group"], d0["group"]), \
                    f"{c}: test predictions not in the shared order; paired bootstrap invalid"
                a = group_aucs(d, groups)
                w, b = worst_best(a)
                r = results_row(c)
                # the mask lives in the original (stage-1) cell; xmask_* masks came from single-attribute cells
                if src_label.startswith("xmask_"):
                    mk = mask_of(cell(runs, ds, "sens_wg", src_label[6:], s))
                else:
                    mk = mask_of(cell(runs, ds, src_label, attr, s))
                k = int(mk.sum()) if mk is not None else (36 if src_label == "base_full_ft" else 0 if src_label == "base_head_only" else np.nan)
                ov = auc_fast(d["label"], d["prob"])
                per.append(dict(seed=s, overall=ov, best=b, worst=w, gap=b - w, train_pct=r["Train Percent"], k=k))
                dots.append(dict(dataset=ds, method=name.strip(), seed=s, worst=w, overall=ov))
                if s in ref_boot:
                    boots.append(boot_worst_idx(d, groups, idxs) - ref_boot[s])
            p = pd.DataFrame(per)
            is_ref = label == OURS
            if boots and not is_ref:
                dist = np.mean(boots, axis=0)
                ref_w = np.array([np.nanmin(group_aucs(ref[s], groups)) for s in seeds if s in ref])
                diff = p.worst[p.seed.isin(ref_seeds)].to_numpy() - ref_w
                d_mean, lo, hi = diff.mean(), *np.nanpercentile(dist, [2.5, 97.5])
                n_better = int((diff > 0).sum())  # seeds where this model beats ours
            else:
                d_mean = lo = hi = np.nan
                n_better = -1
            allrows.append(dict(dataset=DS_NAME[ds], method=name.strip(), extra=(label, attr, name) in EXTRA_ROWS,
                                n_seeds=len(p), overall=p.overall.mean(), overall_sd=p.overall.std(ddof=1),
                                best=p.best.mean(), best_sd=p.best.std(ddof=1), worst=p.worst.mean(),
                                worst_sd=p.worst.std(ddof=1), gap=p.gap.mean(), gap_sd=p.gap.std(ddof=1),
                                train_pct=p.train_pct.mean(), k_values=",".join(str(int(x)) if not np.isnan(x) else "?" for x in p.k),
                                n_full_mask=int((p.k == 36).sum()), d_worst=d_mean, ci_lo=lo, ci_hi=hi,
                                seeds_beating_ours=n_better))
            p.assign(dataset=ds, method=name.strip()).to_csv(f"{out}/data/T4_{ds}_{label}_per_seed.csv", index=False)
    df = pd.DataFrame(allrows)
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Every model evaluated on the intersectional groups (test). Mean $\pm$ SD over seeds "
             r"(PAPILA 10, Harvard-GF 3; rows marked $^\dagger$ are seed 0 only). Trainable \% = share of "
             r"parameters fine-tuned (mean); \#full = seeds whose searched mask selected all 36 components, "
             r"i.e.\ full fine-tuning. $\Delta$worst = worst-group AUC minus that of our intersectional mask "
             r"on the same seed, averaged over seeds, with a 95\% paired stratified-bootstrap CI over test images "
             r"(the same resamples applied to both models); bold if the CI excludes 0. All rows are re-runs "
             r"evaluated on one fixed test ordering.}",
             r"\label{tab:ablation}", r"\small", r"\begin{tabular}{llccccrcc}", r"\toprule",
             r"Dataset & Model & Overall AUC & Best group & Worst group & Gap & Train.\ \% & \#full & $\Delta$worst [95\% CI] \\",
             r"\midrule"]
    prev = None
    for _, r in df.iterrows():
        if prev is not None and r.dataset != prev:
            lines.append(r"\midrule")
        ds = r.dataset if r.dataset != prev else ""
        prev = r.dataset
        name = r.method + (r"$^\dagger$" if r.extra else "")
        if np.isnan(r.d_worst):
            dcol = "---"
        else:
            dcol = f"{r.d_worst:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
            if r.ci_lo > 0 or r.ci_hi < 0:
                dcol = r"\textbf{" + dcol + "}"
        nfull = f"{r.n_full_mask}/{r.n_seeds}" if r.method not in ("Full fine-tuning", "Head only") else "---"
        sd = (lambda c: r[c + "_sd"]) if r.n_seeds > 1 else (lambda c: np.nan)
        lines.append(f"{ds} & {name} & {pm(r.overall, sd('overall'))} & {pm(r.best, sd('best'))} & "
                     f"{pm(r.worst, sd('worst'))} & {pm(r.gap, sd('gap'))} & {r.train_pct:.1f} & {nfull} & {dcol} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    save_table(out, "T4_intersectional_ablation", "\n".join(lines), df)

    # F2 dot plot
    dd = pd.DataFrame(dots)
    dd.to_csv(f"{out}/data/F2_worst_group_dots.csv", index=False)
    order = [n.strip() for _, _, n in METHOD_ROWS + EXTRA_ROWS]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.6), sharey=False)
    for ax, ds in zip(axes, ("papila", "glaucoma")):
        sub = dd[dd.dataset == ds]
        meths = [mm for mm in order if mm in set(sub.method)]
        for i, mm in enumerate(meths):
            v = sub[sub.method == mm].worst.to_numpy()
            ours = "ours" in mm
            col = C["orange"] if ours else C["blue"]
            jitter = (rng.random(len(v)) - 0.5) * 0.25
            ax.scatter(np.full(len(v), i) + jitter, v, s=11, color=col, alpha=0.8, lw=0.4, edgecolor="white", zorder=3)
            if len(v) > 1:
                ax.hlines(v.mean(), i - 0.3, i + 0.3, color=INK, lw=1.4, zorder=4)
        ax.set_xticks(range(len(meths)))
        ax.set_xticklabels([mm.replace("Mask searched on ", "Mask: ").replace(" (ours)", "\n(ours)")
                            .replace("Ranking ablation: ", "Rank: ").replace("Original FairTune, ", "")
                            .replace("Full fine-tuning", "Full FT") for mm in meths], rotation=35, ha="right")
        ax.set_title(f"{DS_NAME[ds]}", loc="left")
        ax.set_ylabel("Test worst-group AUC")
        ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.02, "Dots: seeds. Bar: mean. Orange: our intersectional mask.", color=INK2, fontsize=6.5)
    fig.tight_layout()
    save_fig(fig, out, "F2_worst_group_by_method")
    return df


# ----------------------------------------------------------------------------- F3 Harvard per group
def f3_harvard_groups(runs, repo, out, t4):
    ds = "glaucoma"
    m, labs = group_labels(repo, ds)
    groups = sorted(m.Intersectional_Group)
    sub = t4[(t4.dataset == DS_NAME[ds]) & t4.method.str.startswith("Mask searched on ") & ~t4.method.str.contains("ours")]
    best_single = sub.sort_values("worst", ascending=False).iloc[0].method
    best_label = {"Mask searched on sex": "paired_xmask_gender", "Mask searched on age": "paired_xmask_age",
                  "Mask searched on race": "paired_xmask_race"}[best_single]
    series = [("paired_base_full_ft", "Full fine-tuning", C["blue"]),
              (best_label, f"Best single-attribute mask ({best_single.split()[-1]})", C["aqua"]),
              (OURS, "Intersectional mask (ours)", C["orange"])]
    rows = []
    for label, name, _ in series:
        for s in seeds_of(runs, ds, label, "intersectional"):
            d = load_npz(f"{cell(runs, ds, label, 'intersectional', s)}/test_predictions_intersectional.npz")
            for g, a in zip(groups, group_aucs(d, groups)):
                rows.append(dict(model=name, seed=s, group=g, group_label=labs[g], auc=a))
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/data/F3_harvard_group_auc_per_seed.csv", index=False)
    agg = df.groupby(["model", "group"]).auc.agg(["mean", "std"]).reset_index()
    ours = agg[agg.model == series[2][1]].set_index("group")["mean"]
    # worst-3 consistency under our model
    per_seed_worst = df[df.model == series[2][1]].sort_values("auc").groupby("seed").head(3)
    cnt = per_seed_worst.group.value_counts()
    fig, ax = plt.subplots(figsize=(PAGE_W, 2.5))
    w = 0.26
    x = np.arange(len(groups))
    for i, (_, name, col) in enumerate(series):
        a = agg[agg.model == name].set_index("group").reindex(groups)
        ax.bar(x + (i - 1) * w, a["mean"], w * 0.92, yerr=a["std"], color=col, label=name,
               error_kw=dict(lw=0.6, capsize=1.2, ecolor=INK2), edgecolor="white", linewidth=0.3)
    n = m.set_index("Intersectional_Group")["count_test"]
    ticks = []
    for g in groups:
        star = r"$\bf{*}$ " if cnt.get(g, 0) == cnt.max() and cnt.max() >= 2 else ""
        age, *mid = labs[g].split(" ")
        ticks.append(f"{star}{age}\n{' '.join(mid)}\nn={n[g]}")
    ax.set_xticks(x)
    ax.set_xticklabels(ticks, fontsize=6, linespacing=1.1)
    lo = np.floor((agg["mean"] - agg["std"].fillna(0)).min() * 20) / 20
    ax.set_ylim(max(0.5, lo - 0.02), 1.0)
    ax.set_ylabel("Test AUC (mean $\\pm$ SD, 3 seeds)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.12, "* among the three worst groups for our model in every seed.", color=INK2, fontsize=6.5)
    save_fig(fig, out, "F3_harvard_per_group_auc")
    pd.DataFrame({"group": cnt.index, "group_label": [labs[g] for g in cnt.index], "seeds_in_worst3": cnt.values}) \
        .to_csv(f"{out}/data/F3_worst3_counts.csv", index=False)
    return best_single, cnt, labs


# ----------------------------------------------------------------------------- T5 PAPILA deep dive
def t5_papila(runs, repo, out, n_boot, rng):
    ds = "papila"
    m, labs = group_labels(repo, ds)
    groups = sorted(m.Intersectional_Group)
    rows = []
    for s in seeds_of(runs, ds, OURS, "intersectional"):
        d = load_npz(f"{cell(runs, ds, OURS, 'intersectional', s)}/test_predictions_intersectional.npz")
        a = group_aucs(d, groups)
        bd = boot_groups(d, groups, n_boot, rng)
        lo, hi = np.nanpercentile(bd, 2.5, axis=0), np.nanpercentile(bd, 97.5, axis=0)
        for g in groups:
            rows.append(dict(seed=s, group=g, group_label=labs[g], auc=a[g], ci_lo=lo[g], ci_hi=hi[g],
                             is_worst=bool(a[g] == np.nanmin(a))))
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/data/T5_papila_group_auc_ci.csv", index=False)
    mi = m.set_index("Intersectional_Group")
    head = " & ".join(f"{labs[g]}" for g in groups)
    npos = " & ".join(f"{mi.count_test[g]} / {mi.pos_test[g]}" for g in groups)
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{PAPILA intersectional model: test AUC per age $\times$ sex group for every seed, with "
             r"95\% stratified-bootstrap CIs. Bold: that seed's worst group. Second header row: test eyes / "
             r"glaucomatous eyes per group (two eyes per patient).}",
             r"\label{tab:papila}", r"\small", r"\begin{tabular}{c" + "c" * len(groups) + "}", r"\toprule",
             f"Seed & {head} \\\\", f"Test eyes / positive & {npos} \\\\", r"\midrule"]
    for s, g in df.groupby("seed"):
        cells = []
        for _, r in g.iterrows():
            t = f"{r.auc:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]"
            cells.append(r"\textbf{" + t + "}" if r.is_worst else t)
        lines.append(f"{s} & " + " & ".join(cells) + r" \\")
    worst_counts = df[df.is_worst].group_label.value_counts()
    meanrow = df.groupby("group").auc.agg(["mean", "std"])
    lines.append(r"\midrule")
    lines.append("Mean $\\pm$ SD & " + " & ".join(pm(meanrow.loc[g, 'mean'], meanrow.loc[g, 'std'], 2) for g in groups) + r" \\")
    lines.append("\\# seeds worst & " + " & ".join(str(worst_counts.get(labs[g], 0)) for g in groups) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    save_table(out, "T5_papila_per_group", "\n".join(lines))
    return worst_counts


# ----------------------------------------------------------------------------- T6 + F4/F5 masks & k-sweep
def masks_and_sweep(runs, out):
    rows, heat, hlabels, sweep = [], [], [], []
    meta = None
    for ds, attrs in (("papila", ["gender", "age", "intersectional"]),
                      ("glaucoma", ["gender", "age", "race", "intersectional"])):
        for attr in attrs:
            masks, ks = [], []
            for s in seeds_of(runs, ds, "sens_wg", attr):
                c = cell(runs, ds, "sens_wg", attr, s)
                mk = mask_of(c)
                if meta is None:
                    meta = pd.read_csv(glob.glob(f"{c}/Sensitivity_Scores/*/scores_*.csv")[0]).sort_values("component_idx")
                masks.append(mk)
                ks.append(int(mk.sum()))
                kc = pd.read_csv(glob.glob(f"{c}/Sensitivity_Results/*/k_sweep_*.csv")[0])
                for _, x in kc.iterrows():
                    sweep.append(dict(dataset=DS_NAME[ds], grouping=ATTR_NAME[attr], seed=s, k=int(x.k),
                                      trainable_pct=x.trainable_percentage, val_worst=x.min_auc, val_overall=x.val_auc))
            M = np.array(masks)
            heat.append(M.mean(0))
            hlabels.append(f"{DS_NAME[ds]} {ATTR_NAME[attr].lower()}")
            types = meta.component_type.to_numpy()
            share = {t: M[:, types == t].sum() / max(M.sum(), 1) for t in ("layernorm", "attention", "mlp")}
            rows.append(dict(dataset=DS_NAME[ds], grouping=ATTR_NAME[attr], n_seeds=len(ks),
                             k_median=float(np.median(ks)), k_min=min(ks), k_max=max(ks), n_full=sum(k == 36 for k in ks),
                             ln_share=share["layernorm"], attn_share=share["attention"], mlp_share=share["mlp"],
                             mean_block_selected=float(np.mean([meta.block_idx.to_numpy()[mm == 1].mean() for mm in M if mm.sum()]))))
    df = pd.DataFrame(rows)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Masks chosen by the sensitivity search (36 components = 12 blocks $\times$ \{LayerNorm, attention, MLP\}). "
             r"$k$ = components selected (median [range] over seeds); \#full = seeds with $k{=}36$; "
             r"type shares are over all selected components.}",
             r"\label{tab:masks}", r"\small", r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Dataset & Grouping & $k$ & \#full & LN & Attn & MLP & Mean block \\", r"\midrule"]
    prev = None
    for _, r in df.iterrows():
        if prev is not None and r.dataset != prev:
            lines.append(r"\midrule")
        ds = r.dataset if r.dataset != prev else ""
        prev = r.dataset
        lines.append(f"{ds} & {r.grouping} & {r.k_median:.0f} [{r.k_min}--{r.k_max}] & {r.n_full}/{r.n_seeds} & "
                     f"{r.ln_share:.2f} & {r.attn_share:.2f} & {r.mlp_share:.2f} & {r.mean_block_selected:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    save_table(out, "T6_mask_composition", "\n".join(lines), df)

    # F4 heatmap: rows = grouping, cols = 36 components ordered block-major
    H = np.array(heat)
    fig, ax = plt.subplots(figsize=(PAGE_W, 2.2))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ)
    im = ax.imshow(H, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(hlabels)))
    ax.set_yticklabels(hlabels)
    ab = {"layernorm": "LN", "attention": "A", "mlp": "M"}
    ax.set_xticks(range(36))
    ax.set_xticklabels([ab[t] for t in meta.component_type], fontsize=5)
    for b in range(1, 12):
        ax.axvline(b * 3 - 0.5, color="white", lw=1.5)
    for b in range(12):
        ax.text(b * 3 + 1, -0.9, f"B{b}", ha="center", va="bottom", fontsize=6, color=INK2)
    ax.axhline(2.5, color="white", lw=2.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("Share of seeds selecting", fontsize=6.5)
    cb.outline.set_visible(False)
    save_fig(fig, out, "F4_mask_heatmap")
    pd.DataFrame(H, index=hlabels, columns=[f"B{b}_{ab[t]}" for b, t in zip(meta.block_idx, meta.component_type)]) \
        .to_csv(f"{out}/data/F4_mask_selection_frequency.csv")

    # F5 k-sweep
    sw = pd.DataFrame(sweep)
    sw.to_csv(f"{out}/data/F5_k_sweep.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.3), sharey=False)
    cols = {"Sex": C["blue"], "Age": C["aqua"], "Race": C["yellow"], "Intersectional": C["orange"]}
    marks = {"Sex": "o", "Age": "s", "Race": "^", "Intersectional": "D"}
    for ax, ds in zip(axes, ("PAPILA", "Harvard-GF")):
        s = sw[sw.dataset == ds]
        for g, gg in s.groupby("grouping", sort=False):
            a = gg.groupby("k").agg(x=("trainable_pct", "mean"), y=("val_worst", "mean"), sd=("val_worst", "std")).reset_index()
            ax.plot(a.x, a.y, color=cols[g], marker=marks[g], ms=3.2, lw=1.3, label=g)
            ax.fill_between(a.x, a.y - a.sd, a.y + a.sd, color=cols[g], alpha=0.12, lw=0)
        ax.set_xscale("log")
        ax.set_xlabel("Trainable parameters (%, log scale)")
        ax.set_ylabel("Validation worst-group AUC")
        ax.set_title(ds, loc="left")
    h = {}
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            h.setdefault(ll, hh)
    fig.legend(h.values(), h.keys(), frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save_fig(fig, out, "F5_k_sweep")
    return df


# ----------------------------------------------------------------------------- Stage-2 re-run variability
def rerun_variability(runs, repo, out):
    """Same mask + LR fine-tuned twice (original cell vs its paired_* re-run): how much does test move?"""
    rows = []
    for ds in ("papila", "glaucoma"):
        m, _ = group_labels(repo, ds)
        groups = sorted(m.Intersectional_Group)
        for src_label, attr, name in METHOD_ROWS + EXTRA_ROWS:
            for s in seeds_of(runs, ds, "paired_" + src_label, attr):
                a0 = f"{cell(runs, ds, src_label, attr, s)}/test_predictions_intersectional.npz"
                if not os.path.exists(a0):
                    continue
                d0, d1 = load_npz(a0), load_npz(f"{cell(runs, ds, 'paired_' + src_label, attr, s)}/test_predictions_intersectional.npz")
                g0, g1 = group_aucs(d0, groups), group_aucs(d1, groups)
                rows.append(dict(dataset=DS_NAME[ds], method=name.strip(), seed=s,
                                 overall_orig=auc_fast(d0["label"], d0["prob"]), overall_rerun=auc_fast(d1["label"], d1["prob"]),
                                 worst_orig=np.nanmin(g0), worst_rerun=np.nanmin(g1),
                                 worst_group_orig=int(np.nanargmin(g0)), worst_group_rerun=int(np.nanargmin(g1))))
    df = pd.DataFrame(rows)
    df["abs_d_overall"] = (df.overall_rerun - df.overall_orig).abs()
    df["abs_d_worst"] = (df.worst_rerun - df.worst_orig).abs()
    df.to_csv(f"{out}/data/rerun_variability.csv", index=False)
    return df.groupby("dataset")[["abs_d_overall", "abs_d_worst"]].agg(["mean", "median", "max"])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="paper_results")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs = os.path.join(repo, a.runs_dir) if not os.path.isabs(a.runs_dir) else a.runs_dir
    out = os.path.join(repo, a.out) if not os.path.isabs(a.out) else a.out
    for sub in ("tables", "figures", "data"):
        os.makedirs(f"{out}/{sub}", exist_ok=True)
    rng = np.random.default_rng(0)

    t1 = t1_compute(runs, out)
    t1_table(t1, out)
    f1_cost_quality(runs, out)
    t2_single(runs, out)
    t3_audits(runs, repo, out)
    t4 = t4_ablation(runs, repo, out, a.n_boot, rng)
    best_single, cnt, labs = f3_harvard_groups(runs, repo, out, t4)
    wc = t5_papila(runs, repo, out, a.n_boot, rng)
    t6 = masks_and_sweep(runs, out)
    rv = rerun_variability(runs, repo, out)

    pd.set_option("display.width", 200)
    print("== T1 compute\n", t1.round(2).to_string(index=False))
    print("== T4 ablation\n", t4[["dataset", "method", "n_seeds", "overall", "worst", "worst_sd", "gap", "train_pct",
                                  "n_full_mask", "d_worst", "ci_lo", "ci_hi", "seeds_beating_ours"]].round(3).to_string(index=False))
    print("== Harvard-GF best single-attribute mask:", best_single)
    print("== Harvard-GF seeds-in-worst-3:", {labs[g]: int(c) for g, c in cnt.items()})
    print("== PAPILA worst-group counts:", wc.to_dict())
    print("== T6 masks\n", t6.round(2).to_string(index=False))
    print("== Stage-2 re-run variability (|rerun - original|, same mask + LR)\n", rv.round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
