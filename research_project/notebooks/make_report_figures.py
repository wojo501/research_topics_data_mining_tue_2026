"""Figures used in the report (OVERLEAF/main.tex).

Reads the saved results and writes two PDF files:

* ``return_diff_vs_rho.pdf``: paired return difference VARS minus
  RECO-greedy against rho, with 95% bootstrap intervals over seeds
  (from ``notebooks/tables/paired_ours_vs_reco_greedy.csv``, the data of
  Table 2).
* ``query_delivery_M.pdf``: share of episodes with a supervisor query and
  share of delivered episodes on Taxi M, driver data, |D| = 150, rho = 0
  and 0.5 (from ``results/main_M_solvable.csv``, the setting of Table 1).

In the CSVs the method is called "Ours"; the figures use the report name
VARS. Run from ``research_project/``::

    python -m notebooks.make_report_figures
    python -m notebooks.make_report_figures --out-dir ../../OVERLEAF
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = Path(__file__).resolve().parent / "tables"
DEFAULT_OUT = Path(__file__).resolve().parent / "figures"

METHOD = "VARS"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Series of panel (a): env, dataset, |D|, label, colour, marker, x offset.
RHO_SERIES = [
    ("taxi-M", "drivers", 50, "Taxi M, drivers, $|D|=50$", "black", "o", -0.036),
    ("taxi-M", "drivers", 150, "Taxi M, drivers, $|D|=150$", "#0072B2", "s", -0.018),
    ("taxi-M", "noisy_expert", 150, "Taxi M, noisy expert, $|D|=150$", "#E69F00", "^", 0.0),
    ("taxi-L", "drivers", 150, "Taxi L, drivers, $|D|=150$", "#009E73", "D", 0.018),
    ("taxi-L", "drivers", 400, "Taxi L, drivers, $|D|=400$", "#D55E00", "v", 0.036),
]

# Agents of Table 1, top to bottom, with their report names.
TABLE1_AGENTS = [
    ("RECO", "RECO"),
    ("RECO-greedy", "RECO-greedy"),
    ("In-data planning (POR-style)", "In-data planning"),
    ("Ours", METHOD),
    ("Ours w/o stitching", f"{METHOD} w/o stitching"),
    ("Ours w/o execution rule", f"{METHOD} w/o execution rule"),
]


def _errorbar(ax, rows, x, **kw):
    yerr = [rows.ret_diff - rows.ret_lo, rows.ret_hi - rows.ret_diff]
    ax.errorbar(x, rows.ret_diff, yerr=yerr, ms=4.5, capsize=2, lw=1, **kw)


def plot_return_diff_vs_rho(out: Path) -> Path:
    d = pd.read_csv(TABLES / "paired_ours_vs_reco_greedy.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.81),
                               gridspec_kw={"width_ratios": [1.6, 1]})
    for env, ds, n, label, colour, marker, dx in RHO_SERIES:
        rows = d[(d.env == env) & (d.dataset == ds) & (d.n_traj == n)].sort_values("rho")
        _errorbar(a, rows, rows.rho + dx, fmt=marker, color=colour, label=label)
    rows = d[(d.env == "taxi-M") & (d.dataset == "random_mix") & (d.n_traj == 150)].sort_values("rho")
    _errorbar(b, rows, rows.rho, fmt="s", color="#0072B2")

    for ax in (a, b):
        ax.axhline(0, color="0.6", lw=1, zorder=0)
        ax.set_xticks([0, 0.25, 0.5])
        ax.set_xticklabels(["0.00", "0.25", "0.50"])
        ax.set_xlim(-0.07, 0.57)
        ax.set_xlabel(r"Conditional-door fraction $\rho$")
    a.yaxis.set_major_locator(MultipleLocator(2))
    a.set_ylabel(f"Return difference\n({METHOD} $-$ RECO-greedy)")
    b.set_ylabel("Return difference")
    a.set_title("(a)  Driver data and noisy expert", loc="left", fontsize=9)
    b.set_title(r"(b)  Taxi M, random mix, $|D|=150$", loc="left", fontsize=9)
    a.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    path = out / "return_diff_vs_rho.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_query_delivery_M(out: Path) -> Path:
    d = pd.read_csv(RESULTS / "main_M_solvable.csv")
    d = d[(d.dataset == "drivers") & (d.n_traj == 150) & d.rho.isin([0.0, 0.5])]
    means = d.groupby(["agent", "rho"])[["pct_ep_with_query", "pct_delivered"]].mean()

    agents = [a for a, _ in TABLE1_AGENTS]
    labels = [name for _, name in TABLE1_AGENTS]
    y = list(range(len(agents)))[::-1]
    h = 0.38
    styles = [(0.0, r"$\rho=0$", "#0072B2"), (0.5, r"$\rho=0.5$", "#E69F00")]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    panels = [("pct_ep_with_query", "(a)  Episodes with a query (%)"),
              ("pct_delivered", "(b)  Episodes delivered (%)")]
    for ax, (metric, title) in zip(axes, panels):
        for k, (rho, rho_label, colour) in enumerate(styles):
            vals = [means.loc[(a, rho), metric] for a in agents]
            ys = [yy + (h / 2 if k == 0 else -h / 2) for yy in y]
            ax.barh(ys, vals, height=h, color=colour, label=rho_label)
            for yy, v in zip(ys, vals):
                ax.text(v + 1, yy, f"{v:.1f}", va="center", fontsize=6.5)
        ax.set_xlim(0, 115)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(title, loc="left", fontsize=9)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].legend(frameon=False, fontsize=7.5, loc="center right")
    fig.tight_layout()
    path = out / "query_delivery_M.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                        help="folder for the PDF files (default: notebooks/figures)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in (plot_return_diff_vs_rho(args.out_dir), plot_query_delivery_M(args.out_dir)):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
