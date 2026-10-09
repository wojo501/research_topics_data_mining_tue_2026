# DoorKey benchmark implementation report

## Current finding: no demonstrated benefit

**The saved smoke experiment does not show an improvement over RECO-greedy.**
With the correct abstraction, return, success, and supervisor use are identical.
When door state is ignored, our approach has lower mean return (5.633 versus
10.767) and asks for help more often (0.80 versus 0.50 requests per episode):
**0.30 additional requests per episode, a 60% relative increase**. Both methods
reach 100% success, but that includes supervisor assistance.

This benchmark currently supplies a negative result for the proposed advantage,
not supporting evidence for it. This conclusion is limited to the implemented
variant and the three-seed smoke experiment; it is not a claim about every
DoorKey configuration. The notebook now leads with this result and a compact
four-row comparison. Additional baselines and bootstrap diagnostics are optional.

### What the numbers mean

| Condition | RECO-greedy return | Our return | RECO-greedy help requests/episode | Our help requests/episode |
| --- | ---: | ---: | ---: | ---: |
| Correct abstraction | 10.833 | 10.833 | 0.533 | 0.533 |
| Door state ignored | 10.767 | 5.633 | 0.500 | 0.800 |

With the incorrect abstraction, episodes needing at least one supervisor request
increase from **36.67% to 63.33%**, a rise of **26.67 percentage points** (computed
before rounding). This is a different measure from the **60% relative increase in
requests per episode**, which counts repeated requests within the same episode.

The practical interpretation is straightforward: **this experiment does not
provide evidence that our method is a better alternative to RECO-greedy**.
It ties when the abstraction is correct and performs worse while relying more on
help when the abstraction is incorrect. Reaching the goal in every episode does
not cancel that disadvantage, because the supervisor contributes to that success.
Passing the implementation tests establishes that the tested behavior is correct;
it does not establish a performance improvement.


## Purpose and scope

The project now includes a deterministic, tabular DoorKey-inspired benchmark in
addition to the existing Taxi environments. The purpose is to evaluate value-aware
recovery and state stitching on an explicit prerequisite chain:

**collect key → unlock door → cross doorway → reach goal.**

Taxi already has prerequisites: pickup precedes delivery, and the generated Taxi
maps can condition door traversal on passenger possession. DoorKey adds an explicit
action that permanently changes a passage from locked to open. It therefore offers
a second task structure, rather than simply a larger state space.

The benchmark tests whether selecting high-value continuations and composing
observed/projected transitions helps while respecting those prerequisites. It also
tests the consequences of omitting door state from the abstraction. Improvement
is an experimental hypothesis, not a property imposed on the benchmark or a claim
established by the smoke run.

The implementation is a **DoorKey-inspired tabular variant, not an exact MiniGrid
benchmark reproduction**. MiniGrid DoorKey motivates the key/door/goal sequence:
https://minigrid.farama.org/environments/minigrid/DoorKeyEnv/

To preserve the existing project architecture, this version uses cardinal moves,
symbolic states, stationary Taxi-style rewards, and the existing six-action planner.
It does not introduce MiniGrid's orientation, partial observations, object-dropping
mechanics, time-dependent success reward, or an external MiniGrid dependency.
UnlockPickup and additional benchmarks are outside this implementation.

## Files and architecture

| File | Responsibility |
| --- | --- |
| `src/doorkey.py` | `DoorKey` environment and `DoorKeyModel` fitted from offline transitions |
| `experiments/run_doorkey.py` | Reusable configuration/sweep functions and paired bootstrap summaries |
| `tests/test_doorkey.py` | Nine benchmark correctness/integration tests |
| `notebooks/doorkey_benchmark.ipynb` | Focused result comparison, test execution, optional experiments and diagnostics |
| `results/doorkey_smoke.csv` | Executed three-seed validation experiment |
| `README.md` | Entry point and file descriptions |

The existing Taxi environment, datasets, planner, agents, evaluator, and result
files were not modified. DoorKey implements the interfaces they already consume:
integer states, `S`, `starts`, `T/R/D`, `enc/dec`, `optimal_policy`, and an offline
model exposing `SD`, `obs`, `counts`, `projected`, and `edges`.

The experiment backend is invoked from the notebook. Test definitions remain in
`tests/`, matching the repository's separation of implementation and tests; the
notebook runs the complete test suite, including existing Taxi and analysis tests.

## Environment specification

An odd-sized `n × n` grid, with `n >= 5`, is divided by a vertical wall. One cell
in that wall is a locked door. A key is placed in the left room and a goal in the
right room. A seeded generator selects the door row, key position, and goal position.
There is no route around the wall. The outer grid boundary is impassable.

State is `(row, column, phase)`:

| Phase | Key held | Door open |
| --- | --- | --- |
| 0 | No | No |
| 1 | Yes | No |
| 2 | Yes | Yes |

The key is retained after unlocking and the door cannot be closed again. Thus the
three phases encode all reachable combinations of these two logical variables.
Wall cells other than the doorway are excluded from the encoding. There are
`3 × (n × (n - 1) + 1)` encoded states: 63 for size 5, 129 for size 7, and 219 for
size 9. Some encoded combinations, such as a locked-door phase in the right room,
are unreachable from the initial distribution. They remain representable so a
wrong abstract projection can propose an impossible crossing without causing an
encoding error. Evaluation never starts from those combinations.

Every left-room cell is an evaluation start with phase 0. The map is fixed within
a configuration. Changing the seed changes map placement and the offline batch;
both abstractions and all agents share the same seeded configuration.

Actions retain the Taxi numbering:

| Action | Meaning |
| --- | --- |
| 0–3 | South, north, east, west |
| 4 | Pick up the key, only at its cell in phase 0 |
| 5 | Permanently unlock the door, only when adjacent and holding the key |

Walking onto the key does not pick it up automatically. Entering the doorway
requires phase 2. Unlocking does not move the agent. Repeating pickup after
collection or unlock after opening is invalid.

Ordinary steps, including blocked movement, give -1. Invalid pickup/unlock gives
-10. Entering the goal gives +20 and terminates the episode. Goal states are
absorbing with zero subsequent reward. Evaluation uses the existing discount
factor 0.99 and 200-step cap. The cap is an evaluation truncation; it is not added
to the planner state, consistent with the existing Taxi implementation.

## Offline data and information available to the learner

The existing `expert`, `noisy_expert`, and `random_mix` generators are reused.
They produce the unchanged `(state, action, reward, next_state, done)` format.
The Taxi-specific `drivers` generator is rejected explicitly.

The oracle policy is computed by value iteration on the real environment. As in
Taxi, it is used to generate demonstrations and provide the supervisor's action
when an agent requests help. The learner is fitted from the fixed offline batch.
It does not receive the oracle policy/value or read `T/R/D` during fitting or
projection. A test deletes those tables before constructing and using the model.

The state encoding, action meanings, and hand-designed abstraction are supplied
structural knowledge. This is not a claim to learn the representation from pixels
or infer the abstraction from data. The oracle remains available during evaluation,
and execution-rule memory adapts within the episode; the protocol is offline
training with supervised fallback, not fully autonomous deployment.

## Abstract model and the two experimental conditions

Observed transitions always take precedence. Navigation is the only projected
action family; pickup and unlock are available in the model only where observed.

The **correct abstraction** keys a move by `(row, column, action, door_is_open)`.
It ignores key possession when the door is closed, because possession alone does
not affect movement. A move seen before key collection can therefore be reused
after collection, provided the door has the same openness condition.

The **ignore_door abstraction** keys moves by `(row, column, action)` only.
A successful crossing observed after unlocking may then be proposed before
unlocking. This is an intentionally incorrect abstraction, not a change to the
real environment or to the data-generating dynamics.

For each abstract key, the model collects observed successor positions, rewards,
and terminal flags. Projection is allowed only if these outcomes agree. If a
batch contains both blocked and successful versions of a crossing, the conflicting
projection is removed. A sparse batch can miss that conflict, which is exactly
the setting in which the execution rule matters.

Projection preserves the concrete phase and transfers the observed movement
outcome. Reward and termination can be transferred in this fixed-goal environment
because they do not depend on key possession. Unlike the Taxi navigation model,
DoorKey movement can itself terminate at a goal, so the new model includes the
learned terminal flag in its planner edges.

No new recovery rule is introduced. The existing planner uses task rewards for
Ours, recovery rewards for RECO, and the existing restrictions for the ablations.
The execution rule still blacklists a failed concrete `(state, action)` for the
rest of that episode and resets at the next episode. Abstract-wide blacklisting
was not added.

## Notebook workflow and experimental outputs

Open `notebooks/doorkey_benchmark.ipynb` in a Python kernel with the project
requirements installed. Its cells:

1. Load the saved CSV and state the current negative result.
2. Compare only RECO-greedy and Ours, separately for each abstraction.
3. Explain the return loss and increased supervisor use in plain language.
4. Run the full correctness/regression test suite.
5. Optionally launch a new sweep using `RUN_EXPERIMENTS=True` and a new filename.
6. Optionally show all baselines and bootstrap diagnostics using `SHOW_DETAILS=True`.

The main view has no raw CSV preview, wide statistics table, or repeated charts.
Metric labels explicitly distinguish percentages from requests per episode.
To inspect a new run, change `RESULT_FILE` and rerun the comparison. The written
smoke-run conclusion must be reassessed for any new data. The backend checks for
an existing file before computing and uses exclusive file creation when saving.

Available agents are BC, RECO, RECO-greedy, in-data planning, Ours, Ours without
stitching, and Ours without the execution rule. Imitation is also available as an
optional agent. The Taxi-specific factored reference is not relabelled as a new
DoorKey baseline. RECO uses three stochastic rollouts per start; deterministic
agents use one. Every left-room start is evaluated.

CSV rows retain existing return, discounted-return, query, and step-fraction
columns. `pct_success` and `data_pct_success` describe goal completion; legacy
delivery columns remain equivalent compatibility aliases. Extra columns record
abstraction, map placement, number of states, transition count, projected keys,
and evaluation starts. DoorKey uses its own notebook grouping by `abstraction`;
it does not pretend this binary mask comparison is Taxi's `rho` sweep.

Paired summaries match Ours and RECO-greedy on environment, abstraction, dataset,
trajectory count, and seed. Percentile 95% bootstrap intervals resample paired
seeds within each configuration. One seed produces undefined intervals rather
than a misleading zero-width interval. Seeds vary maps and batches jointly;
these intervals do not separately estimate map and dataset effects.

## Validation and observed smoke results

The delivered notebook was executed end to end. Its test cell reports **21 tests
passed**, covering the nine new DoorKey tests and all twelve existing Taxi/analysis
tests, with no skips. The simplified comparison table was generated successfully.

Nine new tests verify encoding and seeded determinism; key/door prerequisites and
persistent unlocking; successful oracle demonstrations from every start on three
maps; correctness of all navigation projections under the correct mask; incorrect
crossing predictions, consistency filtering, and execution-memory reset; fitting
without transition-table access; optimal performance with a complete model;
data reproducibility and empty-batch supervisor fallback; and experiment output
protection, schema, and supported datasets.

The smoke run uses size 5, seeds 0–2, five noisy-expert trajectories per seed,
both abstractions, and seven agents: **42 CSV rows**. These are implementation
validation results, not a research-scale performance claim.

| Abstraction | Agent | Mean return | Queries/episode | Success (%) |
| --- | --- | ---: | ---: | ---: |
| Correct | RECO-greedy | 10.833 | 0.533 | 100 |
| Correct | Ours | 10.833 | 0.533 | 100 |
| Correct | Ours without execution rule | 10.833 | 0.533 | 100 |
| Ignore door | RECO-greedy | 10.767 | 0.500 | 100 |
| Ignore door | Ours | 5.633 | 0.800 | 100 |
| Ignore door | Ours without execution rule | -200.000 | 0.467 | 0 |

The correct-mask smoke configuration gives a tie with RECO-greedy. The wrong mask
reduces Ours' return and increases supervisor use. Without the execution rule,
the wrong-mask agent fails throughout this sample. This validates that the new
benchmark exposes the intended failure mode, but does not establish that the
proposed algorithm improves on RECO-greedy in DoorKey.

## Interpretation limits

The task has one key, one permanently opening door, and one goal. It is a compact
prerequisite benchmark, not a broad test of hierarchical planning. There is no
visual learning, continuous control, stochastic transition kernel, or learned
abstraction. Map size increases navigational cost, while the prerequisite chain
length stays fixed. Optimal or well-covered batches may leave little room for
improvement. Projecting only navigation also limits which missing subtask actions
can be composed.

Return includes the supervisor's optimal actions, so it must be interpreted
alongside query use and success. Model-step fractions combine projected recovery
and stitching; they do not independently establish how much improvement came
from stitching. The existing no-stitch ablation supplies the controlled comparison.

Neither fallback nor a correct abstraction alone is presented as a new proof of
per-episode return or query dominance. Larger experiments can be configured in
the notebook, but were not run as part of this implementation.
