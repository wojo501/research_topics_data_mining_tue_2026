"""DoorKey experiment backend; launch tests and experiment runs from the notebook.

Keeps the existing one-row-per-agent/configuration CSV format. DoorKey success
also has explicit names, while delivery aliases preserve analysis compatibility.
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src import data
from src.agents import BC, Imitation, InData, Ours, RECO, RECOGreedy
from src.doorkey import DoorKey, DoorKeyModel
from src.evaluate import evaluate, optimal_return


AGENTS = ("BC", "RECO", "RECO-greedy", "In-data planning (POR-style)",
          "Ours", "Ours w/o stitching", "Ours w/o execution rule")
DATASETS = ("expert", "noisy_expert", "random_mix")


def make_agents(model, seed, names):
    makers = {
        "Imitation": lambda: Imitation(model, seed),
        "BC": lambda: BC(model, seed),
        "RECO": lambda: RECO(model, seed),
        "RECO-greedy": lambda: RECOGreedy(model, seed),
        "In-data planning (POR-style)": lambda: InData(model, seed),
        "Ours": lambda: Ours(model, seed),
        "Ours w/o stitching": lambda: Ours(model, seed, stitching=False),
        "Ours w/o execution rule": lambda: Ours(model, seed, exec_rule=False),
    }
    if any(name not in makers for name in names):
        raise ValueError("Unknown DoorKey agent")
    return [(name, makers[name]()) for name in names]


def run_config(n=7, dataset="noisy_expert", n_traj=20, seed=0,
               abstraction="correct", agent_names=AGENTS, reps_stochastic=3):
    if dataset not in DATASETS:
        raise ValueError("DoorKey supports expert, noisy_expert, and random_mix")
    if n_traj < 1 or reps_stochastic < 1:
        raise ValueError("Trajectory and repetition counts must be positive")
    start = time.perf_counter()
    env = DoorKey(n, seed)
    batch = data.make(dataset, env, n_traj, seed)
    transitions = data.transitions(batch)
    model = DoorKeyModel(env, transitions, abstraction)
    supervisor, _ = env.optimal_policy()
    optimal = optimal_return(env, supervisor)
    rows = []
    for name, agent in make_agents(model, seed, agent_names):
        reps = reps_stochastic if name in ("Imitation", "RECO") else 1
        metrics = evaluate(env, agent, supervisor, reps=reps)
        rows.append(dict(
            env=f"doorkey-{n}", abstraction=abstraction,
            dataset=dataset, n_traj=n_traj, seed=seed, agent=name,
            n_states=env.S, n_transitions=len(transitions), n_SD=len(model.SD),
            n_inconsistent=len(model.inconsistent), n_projected_keys=len(model.proj),
            n_eval_starts=len(env.starts), reps_stochastic=reps_stochastic,
            door_row=env.door[0], key_row=env.key[0], key_col=env.key[1],
            goal_row=env.goal[0], goal_col=env.goal[1],
            optimal_ret=optimal, data_pct_delivered=data.pct_delivered(batch),
            data_pct_success=data.pct_delivered(batch),
            pct_success=metrics["pct_delivered"], **metrics))
    for row in rows:
        row["config_seconds"] = time.perf_counter() - start
    return rows


def run_sweep(out, sizes=(7,), datasets=("noisy_expert",), n_trajs=(20,),
              seeds=(0,), abstractions=("correct", "ignore_door"), agent_names=AGENTS):
    """Run a notebook-configured sweep. Refuse existing output before computing."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} already exists; choose a new output path")
    rows = []
    for n in sizes:
        for dataset in datasets:
            for count in n_trajs:
                for seed in seeds:
                    for abstraction in abstractions:
                        rows.extend(run_config(n, dataset, count, seed, abstraction, agent_names))
    if not rows:
        raise ValueError("The experiment grid must not be empty")
    frame = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against another process creating the file.
    with out.open("x") as stream:
        frame.to_csv(stream, index=False)
    return frame


def paired_summary(frame, n_boot=2000, seed=0):
    """Paired seed bootstrap within each configuration; never pool start states."""
    keys = ["env", "abstraction", "dataset", "n_traj", "seed"]
    if frame.duplicated(keys + ["agent"]).any():
        raise ValueError("Duplicate agent/configuration rows")
    metrics = ["ret", "queries_per_ep", "pct_success"]
    baseline = frame[frame.agent == "RECO-greedy"][keys + metrics]
    ours = frame[frame.agent == "Ours"][keys + metrics]
    pairs = ours.merge(baseline, on=keys, suffixes=("_ours", "_reco"), validate="one_to_one")
    rng = np.random.default_rng(seed)
    rows = []
    for config, group in pairs.groupby(keys[:-1]):
        row = dict(zip(keys[:-1], config))
        row["n_pairs"] = len(group)
        for metric in metrics:
            delta = (group[metric + "_ours"] - group[metric + "_reco"]).to_numpy()
            row[metric + "_difference"] = delta.mean()
            # One seed gives no estimate of between-seed uncertainty.
            lo, hi = (np.nan, np.nan)
            if len(delta) > 1:
                means = rng.choice(delta, (n_boot, len(delta)), replace=True).mean(axis=1)
                lo, hi = np.quantile(means, [0.025, 0.975])
            row[metric + "_ci_low"], row[metric + "_ci_high"] = lo, hi
        rows.append(row)
    return pd.DataFrame(rows)
