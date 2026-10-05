"""Checks for the week-2 statistics, independent of the result CSVs."""
import numpy as np
import pandas as pd

from notebooks.analyze_results import bootstrap_mean_ci, sign_flip_pvalue, with_query_cost


def test_bootstrap_ci_covers_a_constant():
    rng = np.random.default_rng(0)
    mean, lo, hi = bootstrap_mean_ci(np.full(20, 3.0), rng)
    assert mean == 3.0
    assert lo == hi == 3.0


def test_bootstrap_ci_shrinks_around_the_mean():
    rng = np.random.default_rng(1)
    x = np.array([0.0, 0.0, 0.0, 10.0])
    mean, lo, hi = bootstrap_mean_ci(x, rng, n_boot=2000)
    assert mean == 2.5
    assert lo < mean < hi


def test_sign_flip_is_small_when_every_difference_is_positive():
    rng = np.random.default_rng(2)
    p = sign_flip_pvalue(np.ones(12), rng, n_perm=4000)
    assert p < 0.01


def test_sign_flip_is_large_when_differences_cancel():
    rng = np.random.default_rng(3)
    p = sign_flip_pvalue(np.array([1.0, -1.0, 1.0, -1.0]), rng, n_perm=2000)
    assert p > 0.2


def test_query_cost_charges_each_query_once():
    out = with_query_cost(np.array([10.0, 8.0]), np.array([2.0, 0.0]), cost=5)
    assert list(out) == [0.0, 8.0]


def test_paired_difference_uses_the_same_seed():
    from notebooks.analyze_results import paired_against

    df = pd.DataFrame([
        dict(env="taxi-M", dataset="drivers", rho=0.0, n_traj=50, seed=0, agent="Ours",
             ret=1.0, disc_ret=1.0, pct_ep_with_query=10.0, queries_per_ep=1.0),
        dict(env="taxi-M", dataset="drivers", rho=0.0, n_traj=50, seed=1, agent="Ours",
             ret=3.0, disc_ret=3.0, pct_ep_with_query=0.0, queries_per_ep=0.0),
        dict(env="taxi-M", dataset="drivers", rho=0.0, n_traj=50, seed=0, agent="RECO-greedy",
             ret=0.0, disc_ret=0.0, pct_ep_with_query=10.0, queries_per_ep=1.0),
        dict(env="taxi-M", dataset="drivers", rho=0.0, n_traj=50, seed=1, agent="RECO-greedy",
             ret=1.0, disc_ret=1.0, pct_ep_with_query=0.0, queries_per_ep=0.0),
        dict(env="taxi-M", dataset="drivers", rho=0.0, n_traj=50, seed=2, agent="RECO-greedy",
             ret=9.0, disc_ret=9.0, pct_ep_with_query=0.0, queries_per_ep=0.0),
    ])
    out = paired_against(df, "RECO-greedy", np.random.default_rng(0))
    assert len(out) == 1
    assert out.iloc[0]["n_pairs"] == 2
    assert out.iloc[0]["ret_diff"] == 1.5
