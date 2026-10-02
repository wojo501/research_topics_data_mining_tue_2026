"""RECO reproduction: exact Taxi-v3, expert-only data, |D| = 10..150 (RECO, Fig. 4).

    python -m experiments.run_reproduction --seeds 30
"""
import argparse
import os

from joblib import Parallel, delayed

from experiments.common import run_config, save
from src.envs import Taxi

AGENTS = ["Imitation", "RECO", "Ours", "Factored Batch RL"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--n-traj", type=int, nargs="+", default=[10, 30, 60, 100, 150])
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--out", default="results/reproduction.csv")
    args = ap.parse_args()
    env = Taxi.v3()
    jobs = [(n, s) for n in args.n_traj for s in range(args.seeds)]
    out = Parallel(n_jobs=args.jobs, verbose=10)(delayed(run_config)(env, "taxi-v3", 0.0, "expert", n, s, AGENTS) for n, s in jobs)
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} already exists: choose another --out (results are never overwritten)")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = save([r for rows in out for r in rows], args.out)
    print(df.groupby(["agent", "n_traj"])[["ret", "pct_ep_with_query"]].mean().unstack("n_traj").round(2))
