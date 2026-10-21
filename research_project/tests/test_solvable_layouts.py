"""Regression checks for rejection-sampled Taxi layouts and the episode horizon."""
import numpy as np
import pytest

from src.envs import MAX_STEPS, Taxi, _layout_max_steps


@pytest.mark.parametrize('size,seeds', [((10, 2, 4), 30), ((20, 3, 8), 10)])
def test_all_experiment_seeds_are_solvable(size, seeds):
    for seed in range(seeds):
        env = Taxi.generated(*size, rho=0.5, seed=seed)
        assert len(env.cond) == round(0.5 * len(env.doors))
        assert 0 < env.max_optimal_steps <= MAX_STEPS
        # Independent check using actual MDP transitions, including pickup/dropoff.
        distance = np.full(env.S, np.inf)
        for _ in range(MAX_STEPS):
            updated = np.min(np.where(env.D, 1, 1 + distance[env.T]), axis=1)
            if np.array_equal(updated, distance):
                break
            distance = updated
        assert np.isfinite(distance[env.starts]).all()
        assert distance[env.starts].max() == env.max_optimal_steps


def test_rejection_is_reproducible_and_preserves_valid_first_draws():
    retried = []
    for seed in range(30):
        old = Taxi.generated(rho=0.5, seed=seed, require_solvable=False)
        new = Taxi.generated(rho=0.5, seed=seed)
        if old.max_optimal_steps is not None and old.max_optimal_steps <= MAX_STEPS:
            assert new.cond == old.cond
            assert new.layout_attempts == 1
        else:
            retried.append(seed)
            assert new.layout_attempts > 1
            again = Taxi.generated(rho=0.5, seed=seed)
            assert new.cond == again.cond
            assert new.layout_attempts == again.layout_attempts
    assert retried  # Exercise real rejected draws, not just easy connected maps.


def test_zero_rho_and_impossible_requests():
    new = Taxi.generated(rho=0)
    old = Taxi.generated(rho=0, require_solvable=False)
    np.testing.assert_array_equal(new.T, old.T)
    assert new.layout_attempts == 1
    with pytest.raises(ValueError, match='No solvable layout'):
        Taxi.generated(rho=1)
    with pytest.raises(ValueError, match='rho'):
        Taxi.generated(rho=-0.1)
    with pytest.raises(ValueError, match='max_layout_attempts'):
        Taxi.generated(max_layout_attempts=0)
    # A seed with a disconnected historical first draw must fail rather than
    # return it when the retry budget is exhausted.
    seed = next(s for s in range(30)
                if Taxi.generated(rho=0.5, seed=s, require_solvable=False).max_optimal_steps is None)
    with pytest.raises(ValueError, match='no seed was skipped'):
        Taxi.generated(rho=0.5, seed=seed, max_layout_attempts=1)


def test_distance_check_counts_pickup_and_dropoff():
    # Empty and loaded grids both connected; worst start needs 4 moves to pickup,
    # 4 to the other stop, and 2 service actions.
    assert _layout_max_steps(3, [(0, 0), (2, 2)], set(), set()) == 10
