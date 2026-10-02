"""Offline datasets, generated from (seed, number of trajectories).

A dataset is a list of trajectories; a trajectory is a list of (s, a, r, s_next, done).

* ``expert``       optimal policy from the Gym start distribution (RECO reproduction)
* ``noisy_expert`` expert with epsilon-random actions (control)
* ``drivers``      structured sub-optimal drivers plus a fraction of expert trajectories (main):
                   each driver ignores part of the doors (plans as if they were walls), starts
                   only in the top half of the map and takes epsilon-random actions
* ``random_mix``   expert plus uniformly random trajectories (generic extra dataset)
"""
import os
import numpy as np
import pandas as pd

from .envs import MAX_STEPS, N_ACTIONS


def _rollout(env, s, policy, rng, eps):
    traj = []
    for _ in range(MAX_STEPS):
        a = int(rng.integers(N_ACTIONS)) if (eps > 0 and rng.random() < eps) else int(policy[s])
        s2, r, done = int(env.T[s, a]), float(env.R[s, a]), bool(env.D[s, a])
        traj.append((s, a, r, s2, done))
        s = s2
        if done:
            break
    return traj


def expert(env, n_traj, seed, eps=0.0):
    rng = np.random.default_rng(seed)
    pi, _ = env.optimal_policy()
    return [_rollout(env, env.starts[rng.integers(len(env.starts))], pi, rng, eps) for _ in range(n_traj)]


def noisy_expert(env, n_traj, seed, eps=0.1):
    return expert(env, n_traj, seed, eps)


def _connected(env, extra_blocked):
    """Every cell reachable from (0, 0) by an empty taxi (conditional doors closed)."""
    blocked = set(env.blocked) | set(env.cond) | set(extra_blocked)
    seen, todo = {(0, 0)}, [(0, 0)]
    while todo:
        r, c = todo.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if 0 <= nb[0] < env.n and 0 <= nb[1] < env.n and nb not in seen                     and frozenset(((r, c), nb)) not in blocked:
                seen.add(nb)
                todo.append(nb)
    return len(seen) == env.n ** 2


def _ignored_doors(env, rng, ignore_p):
    """A driver does not know some connections between rooms: with probability ignore_p it
    ignores BOTH doors of a wall segment, as long as the map stays connected. It then makes real
    detours through other rooms but never gets stuck (every trajectory can be completed)."""
    ignored = []
    for i in rng.permutation(len(env.segments)):
        seg = env.segments[i]
        if rng.random() < ignore_p and _connected(env, ignored + list(seg)):
            ignored += list(seg)
    return ignored


def drivers(env, n_traj, seed, frac_expert=0.1, eps=0.1, n_drivers=3, ignore_p=0.5):
    if not env.segments:
        raise ValueError("drivers need a generated map with doors (Taxi.generated)")
    rng = np.random.default_rng(seed)
    pi_star, _ = env.optimal_policy()
    pols = [env.optimal_policy(extra_blocked=_ignored_doors(env, rng, ignore_p))[0] for _ in range(n_drivers)]
    top = [s for s in env.starts if env.dec(s)[0] < env.n // 2]
    data = []
    for _ in range(n_traj):
        if rng.random() < frac_expert:
            data.append(_rollout(env, env.starts[rng.integers(len(env.starts))], pi_star, rng, 0.0))
        else:
            data.append(_rollout(env, top[rng.integers(len(top))], pols[rng.integers(n_drivers)], rng, eps))
    return data


def random_mix(env, n_traj, seed, frac_expert=0.4):
    rng = np.random.default_rng(seed)
    pi_star, _ = env.optimal_policy()
    data = []
    for _ in range(n_traj):
        s0 = env.starts[rng.integers(len(env.starts))]
        data.append(_rollout(env, s0, pi_star, rng, 0.0 if rng.random() < frac_expert else 1.0))
    return data


GENERATORS = {"expert": expert, "noisy_expert": noisy_expert, "drivers": drivers, "random_mix": random_mix}


def make(kind, env, n_traj, seed, **kw):
    return GENERATORS[kind](env, n_traj, seed, **kw)


def pct_delivered(dataset):
    return 100.0 * sum(bool(traj[-1][4]) for traj in dataset) / len(dataset)


def transitions(dataset):
    """Flatten trajectories into a list of (s, a, r, s_next, done)."""
    return [t for traj in dataset for t in traj]


def save_csv(dataset, path):
    rows = [dict(episode_id=i, step=j, state=s, action=a, reward=r, next_state=s2, done=d)
            for i, traj in enumerate(dataset) for j, (s, a, r, s2, d) in enumerate(traj)]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def load_csv(path):
    df = pd.read_csv(path)
    return [[(int(r.state), int(r.action), float(r.reward), int(r.next_state), bool(r.done))
             for r in g.itertuples()] for _, g in df.sort_values(["episode_id", "step"]).groupby("episode_id")]
