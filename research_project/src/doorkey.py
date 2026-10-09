"""Tabular DoorKey-inspired benchmark (not an exact MiniGrid implementation).

Two rooms are separated by a wall with one locked door. Six actions use the
Taxi move convention: south, north, east, west, pick up, unlock. The key is
retained, unlocking is permanent, and walking over the key does not collect it.
"""
from collections import defaultdict

import numpy as np

from .envs import GAMMA, MOVES, N_ACTIONS


class DoorKey:
    """State = (row, column, phase); phases: 0=no key, 1=key, 2=open door.

    Only traversable cells are encoded. Phase 0/1 states in the right room
    remain representable for imperfect model predictions, but are not starts.
    """

    def __init__(self, n=7, seed=0):
        if n < 5 or n % 2 == 0:
            raise ValueError("DoorKey size must be odd and at least 5")
        self.n, self.seed = n, seed
        rng = np.random.default_rng(seed)
        self.wall_col = n // 2
        self.door = (int(rng.integers(n)), self.wall_col)
        self.key = (int(rng.integers(n)), int(rng.integers(self.wall_col)))
        self.goal = (int(rng.integers(n)), int(rng.integers(self.wall_col + 1, n)))
        self.cells = [(r, c) for r in range(n) for c in range(n)
                      if c != self.wall_col or (r, c) == self.door]
        self.cell_ids = {cell: i for i, cell in enumerate(self.cells)}
        self.S = 3 * len(self.cells)
        self.starts = [self.enc(r, c, 0) for r, c in self.cells if c < self.wall_col]
        self.T = np.empty((self.S, N_ACTIONS), dtype=np.int64)
        self.R = np.empty((self.S, N_ACTIONS))
        self.D = np.empty((self.S, N_ACTIONS), dtype=bool)
        for s in range(self.S):
            for a in range(N_ACTIONS):
                self.T[s, a], self.R[s, a], self.D[s, a] = self.step_raw(s, a)

    def enc(self, row, col, phase):
        if phase not in (0, 1, 2):
            raise ValueError("Invalid DoorKey phase")
        return self.cell_ids[(row, col)] * 3 + phase

    def dec(self, s):
        row, col = self.cells[s // 3]
        return row, col, s % 3

    def step_raw(self, s, a):
        if a not in range(N_ACTIONS):
            raise ValueError("Invalid DoorKey action")
        row, col, phase = self.dec(s)
        if (row, col) == self.goal:
            return s, 0.0, True
        reward = -1.0
        if a < 4:
            dr, dc = MOVES[a]
            target = (row + dr, col + dc)
            if target in self.cell_ids and (target != self.door or phase == 2):
                row, col = target
        elif a == 4:
            if (row, col) == self.key and phase == 0:
                phase = 1
            else:
                reward = -10.0
        elif phase == 1 and abs(row - self.door[0]) + abs(col - self.door[1]) == 1:
            phase = 2
        else:
            reward = -10.0
        done = (row, col) == self.goal
        return self.enc(row, col, phase), 20.0 if done else reward, done

    def optimal_policy(self, gamma=GAMMA):
        """Oracle used only for data generation and evaluation supervision."""
        value = np.zeros(self.S)
        for _ in range(10_000):
            q = self.R + gamma * np.where(self.D, 0.0, value[self.T])
            updated = q.max(axis=1)
            if np.max(np.abs(updated - value)) < 1e-9:
                return q.argmax(axis=1), updated
            value = updated
        raise RuntimeError("DoorKey oracle value iteration did not converge")


class DoorKeyModel:
    """Offline model exposing the interface consumed by the existing agents.

    Only navigation is projected. The correct mask retains position and door
    openness, ignoring key possession when the door is closed. The deliberately
    incorrect mask also drops door openness. Pickup/unlock remain observed-only.
    Projected outcomes, rewards, and terminal flags are learned from the batch;
    neither transition tables nor the oracle are read during model fitting.
    """

    def __init__(self, env, transitions, abstraction="correct"):
        if abstraction not in ("correct", "ignore_door"):
            raise ValueError("Unknown DoorKey abstraction")
        self.env, self.abstraction = env, abstraction
        self.SD, self.obs = set(), {}
        self.counts = defaultdict(lambda: defaultdict(int))
        outcomes = defaultdict(set)
        for s, a, r, s2, done in transitions:
            self.SD.add(s)
            self.obs[s, a] = (s2, r, done)
            self.counts[s][a] += 1
            if a < 4:
                row2, col2, phase2 = env.dec(s2)
                if phase2 != env.dec(s)[2]:
                    raise ValueError("Navigation must preserve the DoorKey phase")
                outcomes[self._key(s, a)].add((row2, col2, r, done))
        self.proj = {k: next(iter(v)) for k, v in outcomes.items() if len(v) == 1}
        self.inconsistent = {k for k, v in outcomes.items() if len(v) > 1}

    def _key(self, s, a):
        row, col, phase = self.env.dec(s)
        key = (row, col, a)
        return key + (phase == 2,) if self.abstraction == "correct" else key

    def _prediction(self, s, a):
        outcome = self.proj.get(self._key(s, a)) if a < 4 else None
        if outcome is None:
            return None
        row, col, reward, done = outcome
        return self.env.enc(row, col, self.env.dec(s)[2]), reward, done

    def projected(self, s, a):
        prediction = self._prediction(s, a)
        return prediction[:2] if prediction is not None else None

    def edges(self, s):
        out = []
        for a in range(N_ACTIONS):
            if (s, a) in self.obs:
                out.append((a, *self.obs[s, a], False))
            else:
                prediction = self._prediction(s, a)
                if prediction is not None:
                    out.append((a, *prediction, True))
        return out
