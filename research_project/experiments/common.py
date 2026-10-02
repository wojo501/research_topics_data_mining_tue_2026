"""Shared helpers for the experiment scripts."""
import time
import numpy as np
import pandas as pd

from src import data as data_mod
from src.agents import BC, RECO, Imitation, InData, Ours, RECOGreedy
from src.evaluate import evaluate, optimal_return
from src.model import AbstractModel

ALL_AGENTS = ["Imitation", "BC", "RECO", "RECO-greedy", "In-data planning (POR-style)",
              "Ours", "Ours w/o stitching", "Ours w/o execution rule", "Factored Batch RL"]


def make_agents(model, fmodel, seed, names=ALL_AGENTS):
    makers = {
        "Imitation": lambda: Imitation(model, seed),
        "BC": lambda: BC(model, seed),
        "RECO": lambda: RECO(model, seed),
        "RECO-greedy": lambda: RECOGreedy(model, seed),
        "In-data planning (POR-style)": lambda: InData(model, seed),
        "Ours": lambda: Ours(model, seed),
        "Ours w/o stitching": lambda: Ours(model, seed, stitching=False, name="Ours w/o stitching"),
        "Ours w/o execution rule": lambda: Ours(model, seed, exec_rule=False, name="Ours w/o execution rule"),
        "Factored Batch RL": lambda: Ours(fmodel, seed, name="Factored Batch RL"),
    }
    return [makers[n]() for n in names]


def run_config(env, env_name, rho, dataset, n_traj, seed, agent_names=ALL_AGENTS, reps_stochastic=3, eval_starts=None):
    """One (environment, dataset, |D|, seed) configuration -> list of result rows.
    eval_starts: evaluate on a fixed random subset of that many start states (None = all)."""
    t0 = time.time()
    starts = env.starts
    if eval_starts and eval_starts < len(env.starts):
        rng = np.random.default_rng(10_000 + seed)
        starts = [env.starts[i] for i in rng.choice(len(env.starts), eval_starts, replace=False)]
    ds = data_mod.make(dataset, env, n_traj, seed)
    trans = data_mod.transitions(ds)
    model = AbstractModel(env, trans)
    fmodel = AbstractModel(env, trans, factored=True)
    supervisor, _ = env.optimal_policy()
    opt = optimal_return(env, supervisor, starts)
    rows = []
    for ag in make_agents(model, fmodel, seed, agent_names):
        reps = reps_stochastic if ag.name in ("Imitation", "RECO") else 1
        res = evaluate(env, ag, supervisor, reps=reps, starts=starts)
        rows.append(dict(env=env_name, rho=rho, dataset=dataset, n_traj=n_traj, seed=seed, agent=ag.name,
                         optimal_ret=opt, data_pct_delivered=data_mod.pct_delivered(ds), n_SD=len(model.SD), n_inconsistent=len(model.inconsistent), n_eval_starts=len(starts), **res))
    for r in rows:
        r["config_seconds"] = time.time() - t0
    return rows


def save(rows, path):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df
