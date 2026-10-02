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
| **Assignment 3** | Research Project Phase | [Download Research Project Phase](#[link-to-assignment-3]) — code in [`research_project/`](research_project/) |

### 📊 Evaluation
* **Research Project Phase Evaluation:** You can download the official evaluation form for the final phase here: [Download Evaluation Form](#[link-to-evaluation-form])

---

## 🔬 Research Project: Value-Aware Recovery

We combine **RECO** (abstraction-guided policy recovery, Ponnambalam et al., ICAPS 2021) with the idea of **POR** (policy-guided state stitching, Xu et al., NeurIPS 2022). When an offline agent is off its data, it recovers towards the **best** known state (not just the nearest one), takes shortcuts between trajectories on the way using RECO's abstraction as a model, and asks a supervisor only when RECO would also ask. We also study what happens when the hand-built abstraction is **wrong**.

**Research question** (to be reframed around the method, as suggested by the supervisor): *Does value-aware recovery improve on RECO without more supervisor queries, and under which conditions (data quality, data quantity, environment size, abstraction error)?*

### Method in short
* Value iteration in RECO's abstract model with the true task rewards: recovery towards the highest-value known state + state stitching through projected moves.
* Consistency check on abstract transitions; **execution rule**: a projected move whose outcome differs from the prediction is dropped for the rest of the episode and the agent replans (free-space assumption, as in D* Lite).
* Fallback: with no valid plan the agent acts like RECO, so with a correct abstraction it is never worse than RECO and never queries more (proposition, checked on every start state).

### Repository layout

| Path | Content |
| --- | --- |
| [`research_project/src/`](research_project/src/) | environments (exact Taxi-v3, generated taxis with conditional doors ρ), datasets, abstract model and planner, agents, evaluation |
| [`research_project/experiments/`](research_project/experiments/) | `run_reproduction.py` (RECO reproduction), `run_main.py` (ρ × dataset × \|D\| × seeds) |
| [`research_project/tests/`](research_project/tests/) | Gym equivalence, map connectivity, the proposition, the execution rule |
| [`research_project/results/`](research_project/results/) | CSV results used for the first meeting |

Quick start (Python 3.10+, from `research_project/`):

```
pip install -r requirements.txt
python -m pytest -q
python -m experiments.run_reproduction --seeds 30 --out results/my_reproduction.csv
python -m experiments.run_main --size M --seeds 10 --out results/my_main_M.csv
```

Scripts never overwrite an existing results file: always pass a new `--out`.

### Results so far (October 2026)

| Experiment | File | Main finding |
| --- | --- | --- |
| RECO reproduction: Taxi-v3, expert data, \|D\| = 10–150, 30 seeds | `results/reproduction_30seeds.csv` | Same pattern as RECO's paper (queries drop to 0%); our method has the same queries and never lower return (150/150 seeds) |
| Taxi M (10×10, 2,000 states), 3 datasets × ρ ∈ {0, 0.25, 0.5} × \|D\| ∈ {50, 150}, 30 seeds | `results/main_M.csv` | Correct abstraction: beats RECO-greedy on 30/30 seeds (+0.5 to +1.2 on good data, very large on poor data). ρ = 0.5 on good data: about −1 |
| Taxi L (20×20, 28,800 states), 2 datasets × ρ ∈ {0, 0.5} × \|D\| ∈ {150, 400}, 10 seeds, 1,000 start states | `results/main_L.csv` | Correct abstraction: +2.6 / +3.7, near optimal, 10/10 seeds; ρ = 0.5: statistical tie, stitching costs 3–4 and slightly more queries |

Takeaways: the gain grows with poorer data and larger maps; with a wrong abstraction stitching loses its advantage; the execution rule prevents a collapse (−35 to −120 without it). Baselines with a high return but 55–90% of episodes with a query (in-data planning) must be compared with a query cost. Note: the prototype numbers in our first email were inflated by a data bug (drivers could get stuck), fixed in this code.

### Team

| Member | Role |
| --- | --- |
| Davide | Code: infrastructure (environments, data, experiment runner) |
| Simone | Code: algorithms (model, planner, agents) |
| Francesco | Writing: framing and literature |
| Wojciech | Writing: method and theory |
| Randy | Writing: experiments and analysis |

### Roadmap

Done:
- [x] Clean, tested codebase (6 tests; Taxi-v3 verified identical to Gymnasium's Taxi)
- [x] RECO reproduction with 30 seeds
- [x] Main experiments on taxi M (30 seeds) and taxi L (10 seeds)
- [x] Fix of the driver data bug (drivers can no longer get stuck)

Next steps and possible future developments:
- [ ] Correct the prototype numbers sent to the supervisor (first meeting)
- [ ] Reframe the research question around the method ("does it improve, under which conditions")
- [ ] Rewrite `main.tex` on Overleaf following the plan (introduction, related work, methodology) and link Overleaf to this repository
- [ ] Abstract execution rule: after a failed projected move, distrust that abstract move for the whole episode in every state (aims at the loss at ρ = 0.5)
- [ ] Rerun taxi M with the final code (value-iteration tolerance changed from 1e-9 to 1e-6)
- [ ] Taxi L with 30 seeds, noisy-expert data and more values of \|D\|
- [ ] Analysis notebook (new `research_project/notebooks/` folder): return vs queries frontier, bootstrap confidence intervals, step breakdown (data / model / query)
- [ ] Close the gap with Factored Batch RL on poor data (generalise pick-up / drop-off)
- [ ] SPIBB baseline (optional)
- [ ] Dynamic RECO with several candidate abstractions (optional)
- [ ] Stochastic dynamics / slippery-taxi robustness test (optional; the proposition assumes deterministic dynamics)
- [ ] Reproduce POR's toy grid example as a qualitative check (optional)
