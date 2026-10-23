# 📊 Research Topics in Data Mining (2AMM20 - TU/e)

Welcome to the repository for **2AMM20 Research Topics in Data Mining** at Eindhoven University of Technology (TU/e).

[![Canvas Course Page](https://img.shields.io/badge/Canvas-Course%20Page-%23E05A47?style=for-the-badge&logo=instructure&logoColor=white)](https://canvas.tue.nl/courses/34514)

---

## 📝 Course Assignments & Evaluation

There will be three assignments for this course.
* **Grading Policy:** Only the final assignment will be graded. However, participation in the first two assignments is **mandatory**, and exceptional performance on them can positively skew the conversion of your group grade into your individual grade.

### 📋 Assignment Structure & Links

| Assignment | Phase | Description / Links |
| :--- | :--- | :--- |
| **Assignment 1** | Paper Review Phase | [Download Paper Review Phase](#[link-to-assignment-1]) |
| **Assignment 2** | Paper Set Review Phase | [Download Paper Set Review Phase](#[link-to-assignment-2]) |
| **Assignment 3** | Research Project Phase | [Download Research Project Phase](#[link-to-assignment-3]), code in [`research_project/`](research_project/) |

### 📊 Evaluation
* **Research Project Phase Evaluation:** You can download the official evaluation form for the final phase here: [Download Evaluation Form](#[link-to-evaluation-form])

---

## 🔬 Research Project: Value-Aware Recovery

We combine **RECO** (abstraction-guided policy recovery, Ponnambalam et al., ICAPS 2021) with the idea of **POR** (policy-guided state stitching, Xu et al., NeurIPS 2022). When an offline agent is off its data, it recovers towards the **best** known state (not just the nearest one), takes shortcuts between trajectories on the way using RECO's abstraction as a model, and asks a supervisor only when RECO would also ask. We also study what happens when the hand-built abstraction is **wrong**.

**Research question:** *Does value-aware recovery improve on RECO without more supervisor queries, and under which conditions (data quality, data quantity, environment size, abstraction error)?*

### Method in short

We plan on one graph built from the batch:
* **Observed transitions**: the logged pairs (s, a) with their successor and reward.
* **Projected transitions**: RECO's abstraction keeps only the taxi coordinates. A move logged at a cell is copied to every state with the same coordinates, whatever the passenger location or destination. If the same move at the same cell was logged with different outcomes, it is inconsistent and is not projected. Pick-up and drop-off are never projected.

At each step the agent (`Ours` in `src/agents.py`):
1. runs value iteration with the true task rewards over observed and projected transitions. A state is *valid* if a delivery is reachable from it. Inside the data this lets the agent jump between trajectories (state stitching); off the data it heads for the goal instead of only returning to the data (value-aware recovery);
2. if the state is not valid, acts exactly like RECO: it imitates the data on logged states, follows RECO's recovery off them, and queries the supervisor only when RECO would;
3. applies the **execution rule**: if a projected move gives a different outcome than predicted, that pair (s, a) is dropped for the rest of the episode and the agent replans (free-space assumption, as in D* Lite).

POR in a tabular, deterministic setting: with expectile τ → 1 its in-sample value approaches value iteration over the logged transitions. Our `In-data planning` baseline is inspired by POR and plans on those transitions directly; it has no guide policy, executor or expectile regression, and we do not claim equivalence with POR.

Sharing RECO's recovery rule as a fallback does not guarantee that Ours asks for fewer or equal queries, or that its return is never lower: the two agents can visit different states and drop different projected moves. In our experiments with a correct abstraction (ρ = 0) Ours and RECO-greedy have identical query rates on every seed and Ours has the higher return; with conditional doors Ours can query more and lose return. `tests/test_basic.py::test_proposition_never_worse_than_reco` checks one configuration (Taxi M, ρ = 0, one seed), not a general theorem.

Planner variants (`src/model.py`, class `Planner`):

| Variant | Inside the data | Off the data | Objective | Used by |
| --- | --- | --- | --- | --- |
| `in_data` | observed | none (query) | task reward | In-data planning; RECO-greedy inside the data |
| `reco` | not planned (imitation) | projected | reach the data | RECO; RECO-greedy off the data |
| `no_stitch` | observed | projected | task reward | Ours w/o stitching |
| `ours` | observed + projected | projected | task reward | Ours |

### Repository layout

| Path | Content |
| --- | --- |
| [`research_project/src/`](research_project/src/) | environments (exact Taxi-v3, generated taxis with conditional doors ρ), datasets, abstract model and planner, agents, evaluation |
| [`research_project/src/doorkey.py`](research_project/src/doorkey.py), [`experiments/run_doorkey.py`](research_project/experiments/run_doorkey.py) | DoorKey-style benchmark: collect a key, unlock a door, reach the goal |
| [`research_project/experiments/`](research_project/experiments/) | `run_reproduction.py` (RECO reproduction), `run_main.py` (ρ × dataset × \|D\| × seeds) |
| [`research_project/tests/`](research_project/tests/) | Gym equivalence, map connectivity, the proposition, the execution rule, analysis, DoorKey, solvable layouts (26 tests) |
| [`research_project/notebooks/`](research_project/notebooks/) | `analyze_results.py` (figures, paired bootstrap tables, `week2_notes.md` for the results section), `make_report_figures.py` (the report figures `return_diff_vs_rho.pdf` and `query_delivery_M.pdf`), `doorkey_benchmark.ipynb` |
| [`research_project/results/`](research_project/results/) | CSV results of all runs (reproduction, taxi M, taxi L, DoorKey smoke run) |

Quick start (Python 3.10+, from `research_project/`):

```
pip install -r requirements.txt
python -m pytest -q
python -m experiments.run_reproduction --seeds 30 --out results/my_reproduction.csv
python -m experiments.run_main --size M --seeds 10 --out results/my_main_M.csv
python -m notebooks.make_report_figures
```

`make_report_figures` redraws the two result figures of the report from the saved CSVs (`notebooks/tables/paired_ours_vs_reco_greedy.csv` and `results/main_M_solvable.csv`) into `notebooks/figures/`; pass `--out-dir` to write them elsewhere. The figures call the method VARS, which is `Ours` in the CSVs.

Scripts never overwrite an existing results file: always pass a new `--out`.

### Results so far (October 2026)

| Experiment | File | Main finding |
| --- | --- | --- |
| RECO reproduction: Taxi-v3, expert data, \|D\| = 10 to 150, 30 seeds | `results/reproduction_30seeds.csv` | Same query curve as RECO's paper (down to 0%); our method has the same queries and never a lower return (150/150 seeds) |
| Taxi M (10×10, 2,000 states), 3 datasets × ρ ∈ {0, 0.25, 0.5} × \|D\| ∈ {50, 150}, 30 seeds | `results/main_M_solvable.csv` | All nine methods have all 30 seeds in every condition. Driver return difference, Ours minus RECO-greedy: ρ = 0: +1.20 / +0.65; ρ = 0.5: -1.45 / -1.22 (50 / 150 trajectories). |
| Taxi L (20×20, 28,800 states), drivers/random_mix × ρ ∈ {0, 0.5} × \|D\| ∈ {150, 400}, 10 seeds, 1,000 start states | `results/main_L_solvable.csv` | All nine methods have all 10 seeds in every condition. Driver return difference: ρ = 0: +3.71 / +2.61; ρ = 0.5: -4.70 / -6.65 (150 / 400 trajectories). |
| DoorKey smoke run: 5×5, 3 seeds, 5 noisy-expert trajectories | `results/doorkey_smoke.csv` | Validation only. Correct abstraction: tie with RECO-greedy (the model is never used). Abstraction that ignores the door: Ours is worse (return 5.6 vs 10.8, 0.80 vs 0.50 queries per episode) |

**Layout protocol.** Conditional-door subsets are rejection-sampled with a fixed seed and fixed door count until every initial state can complete pickup and delivery within 200 steps. All methods use the same accepted layouts: no seeds are excluded. This conditions the map distribution on solvability. The original `main_M.csv` and `main_L.csv` are retained as historical results; current analyses use the `_solvable.csv` reruns. See the [rerun notebook](research_project/notebooks/taxi_solvable_rerun.ipynb) and [updated results notes](research_project/notebooks/week2_notes.md).

**Takeaways.** Compare the updated paired estimates and confidence intervals in `research_project/notebooks/week2_notes.md`. A solvable environment does not imply that every learned policy succeeds or asks less often. Return must be assessed together with supervisor use; all current comparisons retain every seed. The prototype numbers in the first email were inflated by a driver-data bug and should not be reused.

### Team

| Member | Role |
| --- | --- |
| Davide | Code: infrastructure (environments, data, experiment runner) |
| Simone | Code: algorithms (model, planner, agents) |
| Francesco | Writing: framing and literature |
| Wojciech | Writing: method and theory |
| Randy (Hongda) | Writing: experiments and analysis |

### Roadmap

Done:
- [x] Clean, tested codebase (Taxi-v3 verified identical to Gymnasium's Taxi)
- [x] RECO reproduction with 30 seeds
- [x] Main experiments on taxi M (30 seeds) and taxi L (10 seeds)
- [x] Fix of the driver data bug (drivers can no longer get stuck)
- [x] Analysis notebook: return vs queries frontier, bootstrap confidence intervals, step breakdown (data / model / query)
- [x] Research question reframed around the method (hypothesis in the paper)
- [x] DoorKey benchmark: environment, tests, smoke run
- [x] Report draft: method section aligned with the code (12 October)
- [x] Report: result figures (`make_report_figures.py`), DoorKey section, limitations (23 October)

Next steps and possible future developments:
- [ ] Correct the prototype numbers sent to the supervisor
- [ ] Abstract execution rule: after a failed projected move, distrust that abstract move for the whole episode in every state (aims at the loss at ρ = 0.5 and at DoorKey with the wrong abstraction)
- [x] Rerun Taxi M and L with the current 1e-6 tolerance and solvable layouts (30 / 10 seeds)
- [x] Rejection-sample conditional doors and report all seeds without post-hoc layout exclusions
- [ ] Link Overleaf to this repository
- [ ] Larger DoorKey run (bigger maps, more trajectories and seeds)
- [ ] Taxi L with 30 seeds, noisy-expert data and more values of \|D\|
- [ ] Close the gap with Factored Batch RL on poor data (generalise pick-up / drop-off)
- [ ] Optional: SPIBB baseline, dynamic RECO with several abstractions, stochastic dynamics (the proposition assumes deterministic dynamics), POR's toy grid example
