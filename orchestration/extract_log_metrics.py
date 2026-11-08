#!/usr/bin/env python3
"""Extract every whole-split evaluation from a run_experiments.sh log into a tidy CSV.

The raw logs (runs/<dataset>*.log) are too large to commit (mostly repeated sklearn
UndefinedMetricWarning lines from per-batch AUCs), but they hold the only record of
per-epoch / per-k validation curves. This turns each "Pooled (whole-split) metrics:" line
into rows of a long-format CSV, one row per (evaluation, group), tagged with where it came
from:

  dataset, log_line, script (search_mask | search_mask_sensitivity | finetune_with_mask),
  sens_attribute, group_cols, seed, cfg_epochs (--epochs of that process; 2 = a smoke-test
  run that shares the log -- filter to 30 for the real experiments), complete (False if the process was interrupted, e.g.
  the cell was stopped and later re-run from scratch -- filter these out for plots),
  phase (tpe_trial | sens_scoring | k_sweep | finetune), trial (TPE trial index),
  k (k-sweep size), epoch (last finished training epoch before this eval),
  split (val | test, inferred from the evaluated set size), n_total, auc_overall,
  group, group_auc (NaN if the group is single-class on that split), group_n.

Group ids: gender -> M/F, age -> 0 (young) / 1 (old). For --sens_attribute intersectional the
id is the intersectional group index for that cell's --group_cols: see
data_splits_<dataset>/intersectional_mapping.csv for the full (default) group set; the
race-only cells (group_cols Race) number races in sorted order (0 Asian, 1 Black, 2 White).

Usage: python orchestration/extract_log_metrics.py runs/glaucoma.log --dataset glaucoma
"""
import argparse
import csv
import math
import re

SPLIT_SIZES = {
    # n evaluated -> split name
    "glaucoma": {300: "val", 900: "test"},
    "papila": {82: "val", 46: "test"},
}

RE_CMD = re.compile(r"^\+ \S.*?python (?:\S*/)?(search_mask_sensitivity|search_mask|finetune_with_mask)\.py (.*)$")
RE_SESSION = re.compile(r"^\[[0-9:]+\] holding a sleep/idle inhibitor lock")
RE_DONE_ALL = re.compile(r"^\[[0-9:]+\] done: all")
RE_POOLED = re.compile(r"^Pooled \(whole-split\) metrics: AUC (\S+) \| per-group AUC (.*)$")
RE_GROUP = re.compile(r"(\S+): (\S+) \(n=(\d+)\)")
RE_EPOCH_DONE = re.compile(r"^Epoch: \[(\d+)\] Total time")
RE_TRIAL_END = re.compile(r"Trial (\d+) (?:finished|pruned)")
RE_KSWEEP = re.compile(r"^=== k-sweep: k=(\d+)")
RE_SCORING = re.compile(r"^=== Scoring the")


def arg_value(args, name):
    m = re.search(r"--" + name + r" (\S+)", args)
    return m.group(1) if m else ""


def arg_list(args, name):
    m = re.search(r"--" + name + r" ((?:[^-\s]\S*\s?)+)", args)
    return " ".join(m.group(1).split()) if m else ""


def to_float(s):
    try:
        return float(s.rstrip(","))
    except ValueError:
        return math.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--dataset", required=True, choices=sorted(SPLIT_SIZES))
    ap.add_argument("--out", help="default: <log minus .log>_eval_metrics.csv")
    a = ap.parse_args()
    out = a.out or re.sub(r"\.log$", "", a.log) + "_eval_metrics.csv"

    runs = []  # one dict per python process
    cur = None
    last_seed = ""

    def close(complete):
        nonlocal cur
        if cur is not None:
            cur["complete"] = complete
            runs.append(cur)
        cur = None

    with open(a.log, errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if line.startswith(("/", "  warnings.warn")):
                continue
            m = RE_CMD.match(line)
            if m:
                close(True)
                script, args = m.groups()
                seed = arg_value(args, "sensitivity_seed")
                if script == "search_mask":
                    seed = "0"
                elif script == "finetune_with_mask":
                    seed = last_seed  # stage 2 follows its own cell's stage 1
                last_seed = seed
                cur = dict(script=script, sens_attribute=arg_value(args, "sens_attribute"),
                           group_cols=arg_list(args, "group_cols"), seed=seed,
                           cfg_epochs=arg_value(args, "epochs"),
                           phase={"search_mask": "tpe_trial", "finetune_with_mask": "finetune"}.get(script, "sens_scoring"),
                           trial=0 if script == "search_mask" else "", k="", epoch="", rows=[])
                continue
            if RE_SESSION.match(line):
                close(False)  # a new session started while a process was still open -> it was killed
                continue
            if RE_DONE_ALL.match(line):
                close(True)
                continue
            if cur is None:
                continue
            if (m := RE_EPOCH_DONE.match(line)):
                cur["epoch"] = int(m.group(1))
            elif (m := RE_TRIAL_END.search(line)) and cur["script"] == "search_mask":
                cur["trial"] = int(m.group(1)) + 1
                cur["epoch"] = ""
            elif (m := RE_KSWEEP.match(line)):
                cur["phase"], cur["k"], cur["epoch"] = "k_sweep", int(m.group(1)), ""
            elif RE_SCORING.match(line):
                cur["phase"] = "sens_scoring"
            elif (m := RE_POOLED.match(line)):
                groups = RE_GROUP.findall(m.group(2))
                n_total = sum(int(n) for _, _, n in groups)
                split = SPLIT_SIZES[a.dataset].get(n_total, "unknown")
                for g, auc, n in groups:
                    cur["rows"].append(dict(log_line=lineno, phase=cur["phase"], trial=cur["trial"], k=cur["k"],
                                            epoch=cur["epoch"], split=split, n_total=n_total,
                                            auc_overall=to_float(m.group(1)), group=g,
                                            group_auc=to_float(auc), group_n=int(n)))
    close(False)  # EOF with a process still open

    cols = ["dataset", "log_line", "script", "sens_attribute", "group_cols", "seed", "cfg_epochs", "complete", "phase",
            "trial", "k", "epoch", "split", "n_total", "auc_overall", "group", "group_auc", "group_n"]
    n = 0
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in runs:
            for row in r["rows"]:
                w.writerow(dict(dataset=a.dataset, script=r["script"], sens_attribute=r["sens_attribute"],
                                group_cols=r["group_cols"], seed=r["seed"],
                                cfg_epochs=r["cfg_epochs"], complete=r["complete"], **row))
                n += 1
    print(f"{len(runs)} processes, {n} rows -> {out}")


if __name__ == "__main__":
    main()
