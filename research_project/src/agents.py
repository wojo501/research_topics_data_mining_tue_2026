"""Agents. Common interface:

    agent.reset()               start of an episode (clears the execution-rule memory)
    agent.act(s) -> action      an action in 0..5, or QUERY to ask the supervisor
    agent.observe(s, a, s_next) after every step (execution rule)

Methods (see the research plan):
    Imitation      RECO's imitation agent: samples actions by counts in S_D, queries elsewhere
    BC             most frequent action in S_D, queries elsewhere
    RECO           imitation in S_D, RECO recovery off S_D, query if not recoverable
    RECOGreedy     RECO with the best observed action in S_D (in-data planning)
    InData         POR-style in-data planning, queries off S_D (no recovery)
    Ours           value-aware recovery + stitching; falls back to RECO when no valid plan
                   (stitching=False / exec_rule=False give the ablations; a factored model
                   gives the Factored Batch RL reference)
"""
import numpy as np

from .model import Planner

QUERY = -1


class Agent:
    name = "agent"
    uses_model = False

    def __init__(self, model, seed=0):
        self.model = model
        self.rng = np.random.default_rng(seed)
        self.black = frozenset()

    def reset(self):
        self.black = frozenset()

    def act(self, s):
        raise NotImplementedError

    def observe(self, s, a, s_next):
        pass

    # helpers ---------------------------------------------------------------------------
    def _sample_data_action(self, s):
        c = self.model.counts[s]
        acts = list(c)
        p = np.array([c[a] for a in acts], dtype=float)
        return int(acts[self.rng.choice(len(acts), p=p / p.sum())])

    def _mode_action(self, s):
        c = self.model.counts[s]
        return int(max(c, key=c.get))


class Imitation(Agent):
    name = "Imitation"

    def act(self, s):
        return self._sample_data_action(s) if s in self.model.SD else QUERY


class BC(Agent):
    name = "BC"

    def act(self, s):
        return self._mode_action(s) if s in self.model.SD else QUERY


class ModelAgent(Agent):
    """Agents that use projected transitions; they all apply the execution rule
    (unless exec_rule=False): a projected move whose outcome differs from the prediction is
    dropped for the rest of the episode and the agent replans."""
    uses_model = True

    def __init__(self, model, seed=0, exec_rule=True):
        super().__init__(model, seed)
        self.exec_rule = exec_rule
        self._reco = Planner(model, "reco")

    def observe(self, s, a, s_next):
        if not self.exec_rule or a < 0 or (s, a) in self.model.obs:
            return
        pred = self.model.projected(s, a)
        if pred is not None and pred[0] != s_next:
            self.black = self.black | {(s, a)}

    def _reco_action(self, s, greedy_planner=None):
        """RECO's behaviour: imitation (or greedy in-data planning) in S_D, recovery outside."""
        if s in self.model.SD:
            if greedy_planner is not None:
                best, _, _ = greedy_planner.solve()
                if best[s] >= 0:
                    return int(best[s])
                return self._mode_action(s)
            return self._sample_data_action(s)
        best, valid, _ = self._reco.solve(self.black)
        return int(best[s]) if valid[s] else QUERY


class RECO(ModelAgent):
    name = "RECO"

    def act(self, s):
        return self._reco_action(s)


class RECOGreedy(ModelAgent):
    name = "RECO-greedy"

    def __init__(self, model, seed=0, exec_rule=True):
        super().__init__(model, seed, exec_rule)
        self._data = Planner(model, "in_data")

    def act(self, s):
        return self._reco_action(s, self._data)


class InData(Agent):
    name = "In-data planning (POR-style)"

    def __init__(self, model, seed=0):
        super().__init__(model, seed)
        self._data = Planner(model, "in_data")

    def act(self, s):
        if s not in self.model.SD:
            return QUERY
        best, _, _ = self._data.solve()
        return int(best[s]) if best[s] >= 0 else self._mode_action(s)


class Ours(ModelAgent):
    name = "Ours"

    def __init__(self, model, seed=0, stitching=True, exec_rule=True, name=None):
        super().__init__(model, seed, exec_rule)
        self._plan = Planner(model, "ours" if stitching else "no_stitch")
        if name:
            self.name = name

    def act(self, s):
        best, valid, _ = self._plan.solve(self.black)
        if valid[s]:
            return int(best[s])
        return self._reco_action(s)   # fallback: do what RECO would do (query only if RECO would)
