"""Main experiments: generated taxis, abstraction error rho x dataset x |D| x seeds.

    python -m experiments.run_main --size M --rhos 0 0.25 0.5 --datasets drivers noisy_expert --n-traj 50 150 --seeds 30
"""
import argparse
import os

from joblib import Parallel, delayed

from experiments.common import ALL_AGENTS, run_config, save
from src.envs import Taxi

SIZES = {"M": dict(n=10, rooms=2, n_stops=4), "L": dict(n=20, rooms=3, n_stops=8)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=SIZES, default="M")
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    ap.add_argument("--datasets", nargs="+", default=["drivers", "noisy_expert", "random_mix"])
    ap.add_argument("--n-traj", type=int, nargs="+", default=[50, 150])
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--agents", nargs="+", default=ALL_AGENTS)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--eval-starts", type=int, default=None, help="random subset of start states (None = all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    def one(rho, ds, n, seed):
        env = Taxi.generated(rho=rho, seed=seed, **SIZES[args.size])  # conditional doors drawn per seed
        return run_config(env, f"taxi-{args.size}", rho, ds, n, seed, args.agents, eval_starts=args.eval_starts)

    jobs = [(r, d, n, s) for r in args.rhos for d in args.datasets for n in args.n_traj for s in range(args.seeds)]
    out = Parallel(n_jobs=args.jobs, verbose=10)(delayed(one)(*j) for j in jobs)
    path = args.out or f"results/main_{args.size}.csv"
    if os.path.exists(path):
        raise SystemExit(f"{path} already exists: choose another --out (results are never overwritten)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = save([r for rows in out for r in rows], path)
    print(df.groupby(["dataset", "rho", "agent"])[["ret", "pct_ep_with_query"]].mean().round(2).to_string())
