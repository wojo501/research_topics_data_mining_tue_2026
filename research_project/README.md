# Value-aware recovery (RECO + POR-style stitching)

Code for the 2AMM20 project. Tabular, deterministic, numpy only (no neural networks).

## Layout

| Path | Content |
| --- | --- |
| `src/envs.py` | `Taxi.v3()` exact Gym Taxi-v3 map (identical to Gymnasium 1.3 `Taxi-v4`, checked by a test); `Taxi.generated(n, rooms, n_stops, rho, seed)` larger maps with rooms and a fraction `rho` of conditional doors (passable only with the passenger aboard) |
| `src/data.py` | Datasets generated from `(seed, n_traj)`: `expert`, `noisy_expert`, `drivers` (structured sub-optimal drivers + 10% expert), `random_mix`; real CSV save/load |
| `src/model.py` | RECO abstraction {row, col} with consistency check, optional factored model, one vectorised planner for all variants |
| `src/agents.py` | `Imitation`, `BC`, `RECO`, `RECOGreedy`, `InData` (POR-style), `Ours` (+ ablations, + factored reference). Interface: `reset()`, `act(s)` -> action or `QUERY`, `observe(s, a, s_next)` |
| `src/evaluate.py` | Evaluation over every start state; supervisor = optimal policy; return, discounted return, queries, step breakdown |
| `experiments/` | `run_reproduction.py` (RECO Fig. 4 setting), `run_main.py` (rho x dataset x \|D\| x seeds, parallel with joblib) |
| `tests/` | Gym equivalence, map connectivity, CSV round trip, the "never worse than RECO" proposition, the execution rule |
| `results/` | CSV outputs (one row per agent x configuration) |


## Quick start (from this folder)

```
pip install -r requirements.txt
python -m pytest -q                                   # ~10 s
python -m experiments.run_reproduction --seeds 30     # ~30 s
python -m experiments.run_main --size M --seeds 30    # minutes; --size L for the 20x20 map
```

## Notes

* State ids and actions are the same as Gym Taxi (0 south, 1 north, 2 east, 3 west, 4 pick-up, 5 drop-off).
* The execution rule (drop a projected move whose outcome differs from the prediction, replan) is applied to every model-based agent, RECO included. Its memory is reset every episode.
* Drivers ignore whole connections between rooms but the map always stays connected, so every driver trajectory ends with a delivery. Earlier versions could disconnect the map and produced stuck trajectories that inflated the gains.
* `In-data planning` queries the supervisor off the data, so compare it with the query rate or with the query cost, not with return alone.
