"""Evaluation over every start state of the Gym start distribution.

When the agent returns QUERY the supervisor (the optimal policy) chooses the action for that step.
Deterministic agents need one rollout per start state (exact); stochastic agents (Imitation,
RECO) are averaged over ``reps`` rollouts.
"""
import numpy as np

from .agents import QUERY
from .envs import GAMMA, MAX_STEPS


def run_episode(env, agent, s, supervisor):
    agent.reset()
    G = G_disc = 0.0
    n_query = n_data = n_model = 0
    for t in range(MAX_STEPS):
        a = agent.act(s)
        if a == QUERY:
            a = int(supervisor[s])
            n_query += 1
        elif (s, a) in agent.model.obs:
            n_data += 1
        else:
            n_model += 1
        s2, r, done = int(env.T[s, a]), float(env.R[s, a]), bool(env.D[s, a])
        agent.observe(s, a, s2)
        G += r
        G_disc += GAMMA ** t * r
        s = s2
        if done:
            break
    return dict(ret=G, disc_ret=G_disc, queries=n_query, steps=t + 1,
                data_steps=n_data, model_steps=n_model, delivered=done)


def evaluate(env, agent, supervisor, reps=1, starts=None):
    starts = env.starts if starts is None else starts
    eps = [run_episode(env, agent, s0, supervisor) for s0 in starts for _ in range(reps)]
    q = np.array([e["queries"] for e in eps])
    steps = sum(e["steps"] for e in eps)
    return dict(
        ret=float(np.mean([e["ret"] for e in eps])),
        disc_ret=float(np.mean([e["disc_ret"] for e in eps])),
        queries_per_ep=float(q.mean()),
        pct_ep_with_query=float(100 * (q > 0).mean()),
        frac_data_steps=sum(e["data_steps"] for e in eps) / steps,
        frac_model_steps=sum(e["model_steps"] for e in eps) / steps,
        frac_query_steps=float(q.sum()) / steps,
        pct_delivered=float(100 * np.mean([e["delivered"] for e in eps])),
    )


def optimal_return(env, supervisor, starts=None):
    rets = []
    for s in (env.starts if starts is None else starts):
        G = 0.0
        for _ in range(MAX_STEPS):
            a = int(supervisor[s])
            G += float(env.R[s, a])
            done = bool(env.D[s, a])
            s = int(env.T[s, a])
            if done:
                break
        rets.append(G)
    return float(np.mean(rets))
