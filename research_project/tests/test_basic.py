"""Sanity tests:  python -m pytest -q"""
import numpy as np
import pytest

from src import data
from src.agents import RECO, Ours, RECOGreedy
from src.envs import Taxi
from src.evaluate import evaluate, run_episode
from src.model import AbstractModel


def test_state_counts():
    assert Taxi.v3().S == 500 and len(Taxi.v3().starts) == 300
    assert Taxi.generated(10, 2, 4).S == 2_000
    assert Taxi.generated(20, 3, 8).S == 28_800


def test_v3_matches_gym():
    gym = pytest.importorskip("gymnasium")
    name = "Taxi-v3" if "Taxi-v3" in gym.registry else "Taxi-v4"
    P = gym.make(name).unwrapped.P
    env = Taxi.v3()
    for s in range(env.S):
        for a in range(6):
            (prob, s2, r, done), = P[s][a]
            assert prob == 1.0
            assert (env.T[s, a], env.R[s, a], env.D[s, a]) == (s2, r, done), (s, a)


def test_generated_map_connected():
    env = Taxi.generated(20, 3, 8)
    seen, todo = {(0, 0)}, [(0, 0)]
    while todo:
        r, c = todo.pop()
        s = env.enc(r, c, env.K, 0)
        for a in range(4):
            r2, c2, _, _ = env.dec(int(env.T[s, a]))
            if (r2, c2) not in seen:
                seen.add((r2, c2)); todo.append((r2, c2))
    assert len(seen) == env.n ** 2


def test_csv_roundtrip(tmp_path):
    env = Taxi.v3()
    d = data.expert(env, 5, seed=0)
    data.save_csv(d, str(tmp_path / "d.csv"))
    assert data.load_csv(str(tmp_path / "d.csv")) == d


def test_proposition_never_worse_than_reco():
    """rho = 0: on every start state, Ours >= RECO-greedy (discounted) and never more queries;
    on average Ours >= RECO."""
    env = Taxi.generated(10, 2, 4, rho=0.0, seed=0)
    model = AbstractModel(env, data.transitions(data.drivers(env, 100, seed=0)))
    sup, _ = env.optimal_policy()
    ours, rg = Ours(model), RECOGreedy(model)
    for s0 in env.starts:
        a, b = run_episode(env, ours, s0, sup), run_episode(env, rg, s0, sup)
        assert a["disc_ret"] >= b["disc_ret"] - 1e-9 and a["queries"] <= b["queries"], s0
    assert evaluate(env, ours, sup)["disc_ret"] >= evaluate(env, RECO(model), sup, reps=3)["disc_ret"]


def test_execution_rule_matters_with_wrong_abstraction():
    env = Taxi.generated(10, 2, 4, rho=0.5, seed=1)
    model = AbstractModel(env, data.transitions(data.drivers(env, 150, seed=1)))
    sup, _ = env.optimal_policy()
    with_rule = evaluate(env, Ours(model), sup)
    without = evaluate(env, Ours(model, exec_rule=False), sup)
    assert with_rule["ret"] > without["ret"]
    assert with_rule["pct_delivered"] >= without["pct_delivered"]
