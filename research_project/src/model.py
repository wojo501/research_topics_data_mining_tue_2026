"""Model built from the offline data, and one planner for every variant.

Known transitions ("edges") are
* observed: (s, a) -> s' exactly as in the data (with the observed reward);
* projected: RECO's abstraction {row, col}. An abstract move (row, col, a) -> (row', col') seen in
  the data is copied to every ground state with the same position (passenger and destination
  unchanged), with the observed reward. Projected only if the data never shows two different
  outcomes or rewards for it (consistency check).
* factored (optional, "Factored Batch RL" reference): the true structure of the dynamics is known,
  so pick-up also generalises over the destination, and with conditional doors moves are keyed
  on (row, col, passenger aboard).
"""
from collections import defaultdict
import numpy as np

from .envs import GAMMA, N_ACTIONS


class AbstractModel:
    def __init__(self, env, transitions, factored=False):
        self.env, self.factored = env, factored
        self.SD = set()
        self.obs = {}
        self.counts = defaultdict(lambda: defaultdict(int))
        for s, a, r, s2, done in transitions:
            self.SD.add(s)
            self.obs[(s, a)] = (s2, r, done)
            self.counts[s][a] += 1
        move_on_passenger = factored and bool(env.cond)
        outs = defaultdict(set)
        pick = defaultdict(set)
        for s, a, r, s2, done in transitions:
            row, col, p, d = env.dec(s)
            if a < 4:
                row2, col2, _, _ = env.dec(s2)
                outs[self._move_key(row, col, p, a, move_on_passenger)].add((row2, col2, r))
            elif a == 4 and factored:
                pick[(row, col, p)].add((env.dec(s2)[2], r))
        self.move_on_passenger = move_on_passenger
        self.proj = {k: next(iter(v)) for k, v in outs.items() if len(v) == 1}
        self.inconsistent = {k for k, v in outs.items() if len(v) > 1}
        self.pick = {k: next(iter(v)) for k, v in pick.items() if len(v) == 1}

    def _move_key(self, row, col, p, a, on_passenger):
        return (row, col, a, p == self.env.K) if on_passenger else (row, col, a)

    def projected(self, s, a):
        """Predicted (s', r) for an unobserved action, or None."""
        row, col, p, d = self.env.dec(s)
        if a < 4:
            t = self.proj.get(self._move_key(row, col, p, a, self.move_on_passenger))
            if t is not None:
                return self.env.enc(t[0], t[1], p, d), t[2]
        elif a == 4 and self.factored:
            t = self.pick.get((row, col, p))
            if t is not None:
                return self.env.enc(row, col, t[0], d), t[1]
        return None

    def edges(self, s):
        """All known edges from s: (a, s_next, reward, done, projected)."""
        out = []
        for a in range(N_ACTIONS):
            if (s, a) in self.obs:
                s2, r, done = self.obs[(s, a)]
                out.append((a, s2, r, done, False))
            else:
                t = self.projected(s, a)
                if t is not None:
                    out.append((a, t[0], t[1], False, True))
        return out


class Planner:
    """Value iteration over known edges.

    variant:
      'ours'      value-aware recovery + stitching: observed and projected edges everywhere
      'no_stitch' observed edges in the data, projected edges only off the data (ablation)
      'in_data'   observed edges only; nothing off the data (POR-style in-data planning)
      'reco'      RECO's recovery MDP: projected edges off the data, reward 1 for entering S_D
    A state is valid if a terminal edge (delivery, or entering S_D for 'reco') is reachable.
    """
    def __init__(self, model, variant="ours", gamma=None):
        self.model, self.variant = model, variant
        self.gamma = gamma if gamma is not None else (0.95 if variant == "reco" else GAMMA)
        SD, env = model.SD, model.env
        rows = []
        for s in range(env.S):
            ins = s in SD
            if variant == "reco" and ins:
                continue
            if variant == "in_data" and not ins:
                continue
            for a, s2, r, done, proj in model.edges(s):
                if ins and proj and variant in ("no_stitch", "in_data"):
                    continue
                if variant == "reco":
                    enter = s2 in SD
                    rows.append((s, s2, 1.0 if enter else 0.0, enter, a))
                else:
                    rows.append((s, s2, r, done, a))
        arr = np.array(rows, dtype=float).reshape(-1, 5)
        arr = arr[np.argsort(arr[:, 0], kind="stable")]
        self.src = arr[:, 0].astype(np.int64)
        self.dst = arr[:, 1].astype(np.int64)
        self.rew = arr[:, 2]
        self.done = arr[:, 3].astype(bool)
        self.act = arr[:, 4].astype(np.int64)
        self.S = env.S
        self._cache = {}

    def solve(self, black=frozenset()):
        """Returns (best_action, valid, V); best_action[s] = -1 where no valid plan exists.
        ``black`` = set of (s, a) dropped by the execution rule."""
        if black in self._cache:
            return self._cache[black]
        keep = np.ones(len(self.src), dtype=bool)
        if black:
            codes = np.array([s * N_ACTIONS + a for s, a in black])
            keep = ~np.isin(self.src * N_ACTIONS + self.act, codes)
        src, dst, rew, done, act = self.src[keep], self.dst[keep], self.rew[keep], self.done[keep], self.act[keep]
        valid = np.zeros(self.S, dtype=bool)
        valid[src[done]] = True
        while True:
            nv = valid.copy()
            nv[src[valid[dst] & ~done]] = True
            if (nv == valid).all():
                break
            valid = nv
        k = done | valid[dst]
        src, dst, rew, done, act = src[k], dst[k], rew[k], done[k], act[k]
        best = np.full(self.S, -1, dtype=np.int64)
        V = np.zeros(self.S)
        if len(src):
            starts = np.flatnonzero(np.r_[True, src[1:] != src[:-1]])
            base = self._cache.get(frozenset())
            if base is not None:
                V = base[2].copy()
                V[~valid] = 0.0
            for _ in range(20_000):
                q = rew + self.gamma * np.where(done, 0.0, V[dst])
                V_new = np.zeros(self.S)
                V_new[src[starts]] = np.maximum.reduceat(q, starts)
                diff = np.abs(V_new - V).max()
                V = V_new
                if diff < 1e-6:
                    break
            q = rew + self.gamma * np.where(done, 0.0, V[dst])
            order = np.lexsort((-q, src))
            first = order[np.r_[True, src[order][1:] != src[order][:-1]]]
            best[src[first]] = act[first]
        out = (best, valid, V)
        self._cache[black] = out
        return out
