# DoorKey benchmark: team handoff

## What we added and why

We added a small, deterministic DoorKey-inspired environment alongside Taxi to
check whether value-aware recovery works with an explicit prerequisite chain:
collect a key, unlock a door, cross it, and reach the goal. Unlocking permanently
changes the environment. The purpose was to test transfer to a different task
structure, not simply increase the number of states.

This is a **custom tabular variant**, not an exact reproduction of MiniGrid
DoorKey. It uses the existing six-action architecture: four cardinal moves,
pickup, and unlock. The state stores position and one of three phases: no key,
key held, or door open. Ordinary steps cost 1, invalid interactions cost 10, and
reaching the goal gives 20. Episodes are capped at 200 steps.

## How it fits the project

The existing Taxi code and results remain in place. DoorKey reuses the dataset
generators, agents, planner, and evaluator. The new model learns transitions from
an offline batch; the true dynamics are used for data generation and evaluation,
not for fitting projected transitions.

We compare two movement abstractions:

- **Correct:** retains position and whether the door is open.
- **Ignore door:** drops door openness and may incorrectly predict that a locked
  doorway can be crossed.

Pickup and unlock must be observed at the concrete state; only movement is
projected. Conflicting observed outcomes disable a projection. The existing
execution rule removes a failed concrete state-action pair for the remainder of
an episode. The same supervisor fallback and agent implementations are reused.

## What the results actually show

The saved smoke experiment uses a 5×5 map, three seeds, five noisy-expert
trajectories per seed, two abstractions, and seven agents (42 CSV rows).
It validates the implementation; it is not a large statistical study.

| Condition | Method | Mean return | Help requests/episode | Episodes needing help | Success |
| --- | --- | ---: | ---: | ---: | ---: |
| Correct | RECO-greedy | 10.833 | 0.533 | 40.00% | 100% |
| Correct | Ours | 10.833 | 0.533 | 40.00% | 100% |
| Ignore door | RECO-greedy | 10.767 | 0.500 | 36.67% | 100% |
| Ignore door | Ours | 5.633 | 0.800 | 63.33% | 100% |

**There is no demonstrated advantage over RECO-greedy in this experiment.**
The methods tie with the correct abstraction. With the incorrect abstraction,
our method has lower return and makes **60% more requests for help per episode**
(0.80 versus 0.50). Episodes requiring help increase by **26.67 percentage points**,
computed before rounding. These are different measures: one counts requests,
the other counts episodes with at least one request.

The 100% success rate includes optimal-supervisor assistance, so it is not
independent task completion. The wrong-abstraction ablation without the execution
rule has 0% success in this sample. This demonstrates a failure mode and the
importance of the execution rule, but does not establish a benefit over
RECO-greedy. We must not present this experiment as positive evidence for the
research hypothesis or extrapolate its negative result to all DoorKey settings.

## Where to look

| File | Purpose |
| --- | --- |
| [Notebook](notebooks/doorkey_benchmark.ipynb) | Main entry point: compact comparison, tests, optional experiments and diagnostics |
| [Environment and model](src/doorkey.py) | DoorKey dynamics and data-based projections |
| [Experiment backend](experiments/run_doorkey.py) | Sweeps, output protection, paired summaries |
| [Tests](tests/test_doorkey.py) | Nine new correctness/integration checks |
| [Saved results](results/doorkey_smoke.csv) | Full unfiltered smoke results, including all baselines |
| [Detailed report](doorkey_implementation_report.md) | Full design, assumptions, validation, and limitations |

## How to use it

1. Install `research_project/requirements.txt` in your notebook's Python environment.
2. Open `notebooks/doorkey_benchmark.ipynb` from within the repository.
3. Run the loading and comparison cells. They read the saved CSV by default.
4. Run the test cell to execute both the new checks and existing regressions.
5. Only when intentionally starting a new experiment, set `RUN_EXPERIMENTS=True`
   and choose an unused `NEW_OUTPUT` path. Existing CSVs cannot be overwritten.
6. Set `RESULT_FILE` to the new CSV to inspect it. Reassess the written conclusion,
   which explicitly refers to the original smoke experiment.

The notebook was simplified after the initial implementation: the default view
now compares only Ours and RECO-greedy with clear metric labels. Raw previews,
repeated plots, and wide diagnostic tables no longer clutter the main view.
Set `SHOW_DETAILS=True` to inspect the other baselines and bootstrap summaries.

## Validation and scope

The notebook was executed successfully with **21 tests passing**: nine DoorKey
checks plus twelve existing Taxi/analysis tests. Tests cover prerequisites,
persistent unlocking, projection correctness, deliberate abstraction errors,
execution-memory reset, data-only fitting, fallback, reproducibility, and output
protection. Passing these tests does not mean the algorithm outperforms RECO.

No large experiment sweep, new recovery algorithm, abstract-wide blacklist,
UnlockPickup implementation, or exact MiniGrid reproduction was added. All
new code, notebook explanations, and documentation are in English.
