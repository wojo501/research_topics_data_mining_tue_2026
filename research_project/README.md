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

## DoorKey benchmark

Open [`notebooks/doorkey_benchmark.ipynb`](notebooks/doorkey_benchmark.ipynb) to run
the tests, configure experiments, and inspect results. Use a Python kernel with
the project requirements installed. The notebook initially loads the included
three-seed smoke results. To run a new experiment, set `RUN_EXPERIMENTS = True`
and choose a new `NEW_OUTPUT` filename; existing CSVs are never overwritten.

This is a **tabular DoorKey-inspired variant**, not an exact MiniGrid reproduction.
The agent must collect a key, permanently unlock a door, and reach a goal.
The correct movement abstraction retains door openness; `ignore_door` deliberately
drops it. Existing agents, planner, dataset generators, and evaluator are reused.

| Path | Content |
| --- | --- |
| `src/doorkey.py` | Deterministic environment and offline abstract model |
| `experiments/run_doorkey.py` | Configuration/sweep functions and paired seed analysis, invoked by the notebook |
| `tests/test_doorkey.py` | Prerequisites, projection consistency, fallback, reproducibility, and runner checks |
| `notebooks/doorkey_benchmark.ipynb` | Executable test, experiment, and analysis workflow |
| `results/doorkey_smoke.csv` | Small implementation-validation run; not a research-scale experiment |
| `doorkey_implementation_report.md` | Rationale, implementation, validation, and limitations |

Current DoorKey finding: the saved smoke run shows **no improvement over
RECO-greedy with the correct abstraction**. Ignoring door state gives lower return
and **60% more supervisor requests per episode** (0.80 versus 0.50). The notebook
shows this comparison first; other diagnostics are optional.

With door state ignored, episodes needing help also increase from **36.67% to
63.33%**. Both methods reach 100% success, but this includes supervisor assistance.
Consequently, this small experiment provides **no evidence of an advantage for
our method**. See the [implementation report](doorkey_implementation_report.md)
for the comparison and the distinction between request counts and percentages.
