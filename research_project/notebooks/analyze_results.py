"""Week-2 analysis of the saved taxi results.

Reads the CSVs in ``results/``. Writes the paired-test tables and the
figure files into this folder (``notebooks/``).

Run from ``research_project/``::

    python -m notebooks.analyze_results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
HERE = Path(__file__).resolve().parent

N_BOOT = 10_000
N_PERM = 10_000
ALPHA = 0.05
RNG_SEED = 0

METRICS = ["ret", "disc_ret", "pct_ep_with_query", "queries_per_ep"]
GROUP = ["env", "dataset", "rho", "n_traj", "agent"]
PAIR_KEYS = ["env", "dataset", "rho", "n_traj", "seed"]

# Methods that stay on a comparable return scale. The execution-rule ablation
# is drawn separately: at rho = 0.5 its return collapses and flattens the rest.
LINE_AGENTS = [
    "Ours",
    "RECO-greedy",
    "RECO",
    "Ours w/o stitching",
    "In-data planning (POR-style)",
    "Factored Batch RL",
]
# Okabe–Ito, the colour-blind-safe palette used for Nature-style figures.
COLORS = {
    "Ours": "#0072B2",
    "RECO-greedy": "#E69F00",
    "RECO": "#009E73",
    "Ours w/o stitching": "#D55E00",
    "Ours w/o execution rule": "#000000",
    "In-data planning (POR-style)": "#CC79A7",
    "Factored Batch RL": "#56B4E9",
    "Imitation": "#882255",
    "BC": "#999999",
}
SHORT = {
    "In-data planning (POR-style)": "In-data",
    "Factored Batch RL": "Factored",
    "Ours w/o stitching": "No stitching",
    "Ours w/o execution rule": "No exec. rule",
}
STEP_ORDER = [
    "Ours", "RECO-greedy", "RECO", "Ours w/o stitching", "Ours w/o execution rule",
    "Factored Batch RL", "In-data planning (POR-style)", "Imitation", "BC",
]
# Nature double column is 183 mm.
DOUBLE_COL = 183 / 25.4


def bootstrap_mean_ci(values, rng, n_boot=N_BOOT, alpha=ALPHA):
    """Percentile bootstrap CI of the mean. Returns mean, low, high."""
    x = np.asarray(values, dtype=float)
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan
    if n == 1:
        return float(x[0]), float(x[0]), float(x[0])
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def sign_flip_pvalue(diff, rng, n_perm=N_PERM):
    """Two-sided paired sign-flip test on the mean of paired differences.

    Under the null, each seed's difference is as likely to flip sign as not.
    The p-value is (count + 1) / (n_perm + 1), so it cannot be zero.
    """
    d = np.asarray(diff, dtype=float)
    if d.size == 0:
        return np.nan
    obs = abs(float(d.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, d.size))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= obs - 1e-12) + 1) / (n_perm + 1))


def with_query_cost(ret, queries_per_ep, cost):
    """Undiscounted return after charging ``cost`` once per supervisor query.

    The CSV stores episode returns and the mean number of queries, not the
    timestep of each query, so the charge is applied to the undiscounted return.
    """
    return np.asarray(ret, dtype=float) - float(cost) * np.asarray(queries_per_ep, dtype=float)


def load_results(results_dir=RESULTS):
    files = {
        "reproduction": results_dir / "reproduction_30seeds.csv",
        "taxi_M": results_dir / "main_M.csv",
        "taxi_L": results_dir / "main_L.csv",
    }
    frames = {}
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df.insert(0, "experiment", name)
        frames[name] = df
    return frames



def summarise(df, rng):
    records = []
    grouped = df.groupby(GROUP, sort=True)
    for key, part in grouped:
        rec = dict(zip(GROUP, key))
        rec["n_seeds"] = int(part["seed"].nunique())
        for metric in METRICS:
            mean, lo, hi = bootstrap_mean_ci(part[metric].to_numpy(), rng)
            rec[f"{metric}_mean"] = mean
            rec[f"{metric}_lo"] = lo
            rec[f"{metric}_hi"] = hi
        records.append(rec)
    return pd.DataFrame(records)


def paired_against(df, baseline, rng, metrics=("ret", "disc_ret", "pct_ep_with_query")):
    """Seed-paired Ours minus ``baseline``. Positive return diff favours Ours."""
    if "Ours" not in set(df["agent"]) or baseline not in set(df["agent"]):
        return pd.DataFrame()
    left = df[df["agent"] == "Ours"]
    right = df[df["agent"] == baseline]
    merged = left.merge(right, on=PAIR_KEYS, suffixes=("_ours", "_base"))
    records = []
    group_cols = ["env", "dataset", "rho", "n_traj"]
    for key, part in merged.groupby(group_cols, sort=True):
        rec = dict(zip(group_cols, key))
        rec["baseline"] = baseline
        rec["n_pairs"] = int(len(part))
        for metric in metrics:
            diff = (part[f"{metric}_ours"] - part[f"{metric}_base"]).to_numpy()
            mean, lo, hi = bootstrap_mean_ci(diff, rng)
            rec[f"{metric}_diff"] = mean
            rec[f"{metric}_lo"] = lo
            rec[f"{metric}_hi"] = hi
            rec[f"{metric}_p"] = sign_flip_pvalue(diff, rng)
            if metric == "pct_ep_with_query":
                rec["n_fewer_query_seeds"] = int(np.sum(diff < -1e-9))
            if metric == "ret":
                rec["n_higher_return_seeds"] = int(np.sum(diff > 1e-9))
                rec["n_lower_return_seeds"] = int(np.sum(diff < -1e-9))
        records.append(rec)
    return pd.DataFrame(records)


def _style():
    """Nature figure defaults: Arial, 183 mm double column, outward ticks, no grid."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "black",
        "axes.grid": False,
        "axes.axisbelow": True,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "legend.fontsize": 6,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig, name):
    path = HERE / "figures" / name
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _panel(ax, letter):
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", ha="left", va="bottom", clip_on=False)


def _legend(fig, ncol=4):
    handles, labels, seen = [], [], set()
    for ax in fig.axes:
        h, lab = ax.get_legend_handles_labels()
        for handle, label in zip(h, lab):
            if label not in seen:
                seen.add(label)
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False,
                   handlelength=1.6, columnspacing=0.8)
    return handles


def _errorbar(ax, x, mean, lo, hi, label, color):
    yerr = np.vstack([np.asarray(mean) - np.asarray(lo), np.asarray(hi) - np.asarray(mean)])
    ax.errorbar(
        x, mean, yerr=yerr, label=SHORT.get(label, label), color=color,
        marker="o", linewidth=1.0, markersize=2.6, capsize=1.4,
        elinewidth=0.5, capthick=0.5, markeredgewidth=0,
    )


def plot_reproduction(summary):
    sub = summary[summary["env"] == "taxi-v3"].copy()
    agents = [a for a in ["Imitation", "RECO", "Ours", "Factored Batch RL"] if a in set(sub["agent"])]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.35), layout="constrained")
    specs = [
        (axes[0], "a", "ret", "Undiscounted return"),
        (axes[1], "b", "pct_ep_with_query", "Episodes with a query (%)"),
    ]
    for ax, letter, metric, ylabel in specs:
        for agent in agents:
            part = sub[sub["agent"] == agent].sort_values("n_traj")
            _errorbar(ax, part["n_traj"], part[f"{metric}_mean"], part[f"{metric}_lo"], part[f"{metric}_hi"],
                      agent, COLORS[agent])
        ax.set_xlabel("Expert trajectories, |D|")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(sub["n_traj"].unique()))
        _panel(ax, letter)
    _legend(fig, ncol=4)
    return _save(fig, "fig1_reproduction_return_and_queries.png")


def plot_vs_rho(summary):
    sub = summary[(summary["env"] == "taxi-M") & (summary["n_traj"] == 150)].copy()
    datasets = ["drivers", "noisy expert", "random mix"]
    keys = ["drivers", "noisy_expert", "random_mix"]
    letters = [["a", "b", "c"], ["d", "e", "f"]]
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, 4.15), sharex=True, layout="constrained")
    metrics = [("ret", "Undiscounted return"), ("pct_ep_with_query", "Episodes with a query (%)")]
    for col, (dataset, key) in enumerate(zip(datasets, keys)):
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            part = sub[sub["dataset"] == key]
            for agent in LINE_AGENTS:
                one = part[part["agent"] == agent].sort_values("rho")
                if one.empty:
                    continue
                if metric == "ret" and float(one[f"{metric}_mean"].min()) < -200:
                    continue
                _errorbar(ax, one["rho"], one[f"{metric}_mean"], one[f"{metric}_lo"], one[f"{metric}_hi"],
                          agent, COLORS[agent])
            ax.set_xticks([0.0, 0.25, 0.5])
            if row == 0:
                ax.set_title(dataset, fontsize=7, pad=8)
            if row == 1:
                ax.set_xlabel("Abstraction error, ρ")
            if col == 0:
                ax.set_ylabel(ylabel)
            _panel(ax, letters[row][col])
    axes[0, 2].text(0.98, 0.05, "RECO ≈ −560\n(off scale)", transform=axes[0, 2].transAxes,
                    ha="right", va="bottom", fontsize=5.5, color="#444444")
    _legend(fig, ncol=3)
    return _save(fig, "fig2_return_and_queries_vs_rho.png")


def plot_vs_size(summary):
    sub = summary[(summary["env"] == "taxi-M") & (summary["dataset"] == "drivers")].copy()
    rhos = [0.0, 0.25, 0.5]
    letters = [["a", "b", "c"], ["d", "e", "f"]]
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, 4.05), sharex=True, layout="constrained")
    metrics = [("ret", "Undiscounted return"), ("pct_ep_with_query", "Episodes with a query (%)")]
    for col, rho in enumerate(rhos):
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            part = sub[np.isclose(sub["rho"], rho)]
            for agent in LINE_AGENTS:
                one = part[part["agent"] == agent].sort_values("n_traj")
                if one.empty:
                    continue
                _errorbar(ax, one["n_traj"], one[f"{metric}_mean"], one[f"{metric}_lo"], one[f"{metric}_hi"],
                          agent, COLORS[agent])
            ax.set_xticks([50, 150])
            if row == 0:
                ax.set_title(f"ρ = {rho:g}", fontsize=7, pad=8)
            if row == 1:
                ax.set_xlabel("Trajectories, |D|")
            if col == 0:
                ax.set_ylabel(ylabel)
            _panel(ax, letters[row][col])
    _legend(fig, ncol=3)
    return _save(fig, "fig3_return_and_queries_vs_dataset_size.png")



def plot_frontier(summary):
    sub = summary[(summary["env"] == "taxi-M") & (summary["dataset"] == "drivers") & (summary["n_traj"] == 150)]
    agents = [a for a in list(COLORS) if a in set(sub["agent"])]
    letters = ["a", "b", "c"]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.45), sharex=True, layout="constrained")
    for ax, rho, letter in zip(axes, [0.0, 0.25, 0.5], letters):
        part = sub[np.isclose(sub["rho"], rho)]
        for agent in agents:
            one = part[part["agent"] == agent]
            if one.empty:
                continue
            ax.scatter(one["pct_ep_with_query_mean"], one["ret_mean"], s=16, color=COLORS[agent],
                       label=SHORT.get(agent, agent), zorder=3, linewidths=0)
        ax.set_title(f"ρ = {rho:g}", fontsize=7, pad=8)
        ax.set_xlabel("Episodes with a query (%)")
        _panel(ax, letter)
    axes[0].set_ylabel("Undiscounted return")
    _legend(fig, ncol=3)
    return _save(fig, "fig5_return_query_frontier.png")


def plot_steps(df):
    """Horizontal composition bars. Labels sit on the y-axis, so nothing is rotated."""
    sub = df[(df["env"] == "taxi-M") & (df["dataset"] == "drivers") & (df["n_traj"] == 150)]
    agents = [a for a in STEP_ORDER if a in set(sub["agent"])]
    labels = [SHORT.get(a, a) for a in agents]
    y = np.arange(len(agents))
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.85), sharey=True, layout="constrained")
    stack_colors = {"Data": "#0072B2", "Model": "#E69F00", "Query": "#CC79A7"}
    for ax, rho, letter in zip(axes, [0.0, 0.25, 0.5], ["a", "b", "c"]):
        part = sub[np.isclose(sub["rho"], rho)]
        data, model, query = [], [], []
        for agent in agents:
            one = part[part["agent"] == agent]
            data.append(float(one["frac_data_steps"].mean()))
            model.append(float(one["frac_model_steps"].mean()))
            query.append(float(one["frac_query_steps"].mean()))
        data, model, query = map(np.asarray, (data, model, query))
        ax.barh(y, data, color=stack_colors["Data"], label="Data", height=0.72, linewidth=0)
        ax.barh(y, model, left=data, color=stack_colors["Model"], label="Model", height=0.72, linewidth=0)
        ax.barh(y, query, left=data + model, color=stack_colors["Query"], label="Query", height=0.72, linewidth=0)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction of steps")
        ax.set_title(f"ρ = {rho:g}", fontsize=7, pad=8)
        ax.tick_params(axis="y", length=0)
        _panel(ax, letter)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    _legend(fig, ncol=3)
    return _save(fig, "fig6_step_breakdown.png")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    (HERE / "figures").mkdir(parents=True, exist_ok=True)
    (HERE / "tables").mkdir(parents=True, exist_ok=True)
    frames = load_results(args.results)
    combined = pd.concat(frames.values(), ignore_index=True)
    rng = np.random.default_rng(RNG_SEED)
    summary = summarise(combined, rng)
    paired_greedy = paired_against(combined, "RECO-greedy", rng)
    paired_reco = paired_against(combined, "RECO", rng)
    paired_greedy.to_csv(HERE / "tables" / "paired_ours_vs_reco_greedy.csv", index=False)
    paired_reco.to_csv(HERE / "tables" / "paired_ours_vs_reco.csv", index=False)
    _style()
    plot_reproduction(summary)
    plot_vs_rho(summary)
    plot_vs_size(summary)
    plot_frontier(summary)
    plot_steps(frames["taxi_M"])
    print(f"wrote {HERE / 'figures'}")
    print(f"wrote {HERE / 'tables'}")


if __name__ == "__main__":
    main()
