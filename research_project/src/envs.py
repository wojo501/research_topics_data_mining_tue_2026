"""Taxi environments.

* ``Taxi.v3()``: the exact Gym Taxi-v3 map (500 states), used for the RECO reproduction.
* ``Taxi.generated(n, rooms, n_stops, rho, seed)``: larger maps split into rooms connected by
  doors. A fraction ``rho`` of the doors is *conditional*: passable only with the passenger
  aboard. The abstraction {x, y} ignores the passenger, so conditional doors make it wrong.

State encoding and actions follow Gym Taxi-v3, so state ids are interchangeable with Gym:
``s = ((row * n + col) * (K + 1) + passenger) * K + destination``, passenger == K means
"in the taxi". Actions: 0 south, 1 north, 2 east, 3 west, 4 pick-up, 5 drop-off.
All dynamics are deterministic.
"""
import numpy as np

GAMMA = 0.99
MAX_STEPS = 200
MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # (d_row, d_col) for south, north, east, west
N_ACTIONS = 6


def _edge(a, b):
    return frozenset((a, b))


class Taxi:
    def __init__(self, n, stops, blocked, doors=(), cond=(), segments=()):
        self.n, self.stops, self.K = n, list(stops), len(stops)
        self.blocked = set(blocked)            # walls between adjacent cells
        self.doors = list(doors)               # openings in room walls (for drivers and rho)
        self.segments = list(segments)         # the two doors of each wall segment
        self.cond = set(cond)                  # conditional doors: passable only with passenger
        self.S = n * n * (self.K + 1) * self.K
        self.T, self.R, self.D = self._tables()
        # Gym initial distribution: passenger waiting at a stop different from the destination
        self.starts = [self.enc(r, c, p, d) for r in range(n) for c in range(n)
                       for p in range(self.K) for d in range(self.K) if p != d]

    # ----- constructors -------------------------------------------------------------------
    @classmethod
    def v3(cls):
        """Exact Gym Taxi-v3 map: R(0,0), G(0,4), Y(4,0), B(4,3)."""
        walls = [(0, 1), (1, 1), (3, 0), (4, 0), (3, 2), (4, 2)]  # wall between (r,c) and (r,c+1)
        blocked = {_edge((r, c), (r, c + 1)) for r, c in walls}
        return cls(5, [(0, 0), (0, 4), (4, 0), (4, 3)], blocked)

    @classmethod
    def generated(cls, n=10, rooms=2, n_stops=4, rho=0.0, seed=0):
        """n x n grid split into rooms x rooms rooms; two doors per wall segment.
        M = generated(10, 2, 4) -> 2,000 states; L = generated(20, 3, 8) -> 28,800 states."""
        cuts = [round(k * n / rooms) for k in range(1, rooms)]
        bounds = [0] + cuts + [n]
        blocked, doors, segments = set(), [], []
        for c in cuts:                      # wall between index c-1 and c, horizontal and vertical
            for i in range(rooms):
                lo, hi = bounds[i], bounds[i + 1]
                ln = hi - lo
                door_pos = sorted({lo + ln // 4, lo + (3 * ln) // 4})
                for wall in (lambda j: _edge((c - 1, j), (c, j)), lambda j: _edge((j, c - 1), (j, c))):
                    for j in range(lo, hi):
                        (doors.append(wall(j)) if j in door_pos else blocked.add(wall(j)))
                    segments.append(tuple(wall(j) for j in door_pos))
        stops = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        if n_stops == 8:
            h = n // 2
            stops += [(0, h), (n - 1, h), (h, 0), (h, n - 1)]
        elif n_stops != 4:
            raise ValueError("n_stops must be 4 or 8")
        rng = np.random.default_rng(seed)
        k = int(round(rho * len(doors)))
        cond = [doors[i] for i in rng.choice(len(doors), k, replace=False)] if k else []
        return cls(n, stops, blocked, doors, cond, segments)

    # ----- encoding -------------------------------------------------------------------------
    def enc(self, r, c, p, d):
        return ((r * self.n + c) * (self.K + 1) + p) * self.K + d

    def dec(self, s):
        d = s % self.K
        s //= self.K
        p = s % (self.K + 1)
        s //= self.K + 1
        return s // self.n, s % self.n, p, d

    # ----- dynamics ---------------------------------------------------------------------------
    def step_raw(self, s, a, extra_blocked=frozenset()):
        """Deterministic step: returns (next_state, reward, done). Same rules as Gym Taxi-v3."""
        r, c, p, d = self.dec(s)
        if a < 4:
            nr, nc = r + MOVES[a][0], c + MOVES[a][1]
            e = _edge((r, c), (nr, nc))
            ok = (0 <= nr < self.n and 0 <= nc < self.n
                  and e not in self.blocked and e not in extra_blocked)
            if ok and e in self.cond and p != self.K:
                ok = False
            if ok:
                r, c = nr, nc
            return self.enc(r, c, p, d), -1.0, False
        if a == 4:  # pick-up
            if p < self.K and self.stops[p] == (r, c):
                return self.enc(r, c, self.K, d), -1.0, False
            return s, -10.0, False
        # drop-off
        if p == self.K and self.stops[d] == (r, c):
            return self.enc(r, c, d, d), 20.0, True
        if p == self.K and (r, c) in self.stops:      # Gym: passenger left at another stop
            return self.enc(r, c, self.stops.index((r, c)), d), -1.0, False
        return s, -10.0, False

    def _tables(self, extra_blocked=frozenset()):
        T = np.zeros((self.S, N_ACTIONS), dtype=np.int64)
        R = np.zeros((self.S, N_ACTIONS))
        D = np.zeros((self.S, N_ACTIONS), dtype=bool)
        for s in range(self.S):
            for a in range(N_ACTIONS):
                T[s, a], R[s, a], D[s, a] = self.step_raw(s, a, extra_blocked)
        return T, R, D

    def optimal_policy(self, extra_blocked=frozenset(), gamma=GAMMA):
        """Value iteration on the true model (optionally with extra walls, for drivers)."""
        T, R, D = (self.T, self.R, self.D) if not extra_blocked else self._tables(frozenset(extra_blocked))
        V = np.zeros(self.S)
        for _ in range(10_000):
            Q = R + gamma * np.where(D, 0.0, V[T])
            V_new = Q.max(1)
            if np.abs(V_new - V).max() < 1e-9:
                break
            V = V_new
        return Q.argmax(1), V
