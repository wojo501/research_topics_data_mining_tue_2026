# Notes for the results section

From `results/reproduction_30seeds.csv` and `results/main_M.csv`, read by `python -m notebooks.analyze_results`. Intervals are 95% percentile bootstrap CIs over seeds (10,000 resamples). A return difference is Ours minus the named baseline. Query differences are in percentage points of episodes with at least one supervisor call.

`results/main_L.csv` is in the paired tables and not in figures 2–6. It has 10 seeds and 1,000 start states.

The value-iteration tolerance in the code is 1e-6. The README still lists a rerun of taxi M, because the stored CSVs may have been produced at 1e-9. Treat the numbers below as the current files, and replace them if that rerun happens.

Do not reuse the figures from the first email to the supervisor. Drivers in that version could get stuck; the gains were too large. The present drivers keep the map connected.

## Reproduction

`figures/fig1_reproduction_return_and_queries`. Taxi-v3, expert data, |D| = 10, 30, 60, 100, 150, 30 seeds.

Ours and RECO have the same query rate at every |D|: 50% at 10 trajectories, 8% at 30, 2% at 60, and 0% at 150. Imitation is still at 32% when |D| = 150. Its undiscounted return stays at 7.93, because a query is answered by the optimal policy. The comparison with RECO's figure is the query curve.

`tables/paired_ours_vs_reco.csv`. The query difference is 0 at every |D|. Undiscounted return is higher for Ours on 24/30 seeds at |D| = 10 (+0.07, CI [+0.05, +0.10]) and on 30/30 seeds from |D| = 30 on (+0.13, +0.10, +0.07, +0.05). RECO-greedy was not part of this run.

## Taxi M against RECO-greedy

`figures/fig2_return_and_queries_vs_rho` is |D| = 150, three datasets, against ρ. `figures/fig3_return_and_queries_vs_dataset_size` is driver data, |D| = 50 and 150. `tables/paired_ours_vs_reco_greedy.csv` is the paired test.

Driver data, 30 seeds:

| ρ | \|D\| | return diff | 95% CI | seeds with higher return | query diff |
| --- | --- | --- | --- | --- | --- |
| 0 | 50 | +1.20 | [+1.06, +1.35] | 30/30 | 0 |
| 0 | 150 | +0.65 | [+0.60, +0.70] | 30/30 | 0 |
| 0.25 | 50 | +0.34 | [−0.01, +0.67] | 21/30 | −0.06 |
| 0.25 | 150 | +0.09 | [−0.20, +0.34] | 18/30 | 0 |
| 0.5 | 50 | −1.59 | [−2.42, −0.74] | 6/30 | +0.42 |
| 0.5 | 150 | −1.02 | [−2.00, +0.26] | 7/30 | +0.33 |

At ρ = 0 the gain is on every seed and the query rate does not move (0.5% of episodes). At ρ = 0.25 the interval includes zero. At ρ = 0.5 the sign flips. The loss at |D| = 50 is stable across seeds; at |D| = 150 the interval still includes zero. Mean returns on this slice, |D| = 150: Ours −3.7 / −4.9 / −16.3 and RECO-greedy −4.4 / −5.0 / −15.3 at ρ = 0 / 0.25 / 0.5. RECO itself is about −16 even at ρ = 0, because it imitates the drivers on the data.

Noisy expert at ρ = 0 is the same pattern, smaller: +0.70 at |D| = 50 and +0.47 at |D| = 150, 30/30 seeds, query difference 0. At ρ = 0.5 the interval covers zero.

Random mix is the "poor data" case in the README. RECO's mean return is about −560, so fig2 leaves it off the return panel. Ours minus RECO-greedy is about +227 at ρ = 0, |D| = 50, and about +54 at ρ = 0.5, |D| = 150, on 30/30 seeds.

## Return against queries

`figures/fig5_return_query_frontier`. Drivers, |D| = 150.

At ρ = 0, in-data planning has return −3.76, next to Ours at −3.74, and a query in 56% of episodes (1.80 queries per episode). Ours queries in 0.5% of episodes. The same gap is there at ρ = 0.25 (57% vs 0.3%) and ρ = 0.5 (59% vs 3.5%). Compare in-data planning on the query rate, or on return after a per-query cost. The plan does not fix the cost. Charging 1 per query on the undiscounted return moves in-data planning from −3.76 to −5.56 at ρ = 0; Ours stays at −3.74.

## Where the steps come from

`figures/fig6_step_breakdown`. Same slice: drivers, |D| = 150. Bars are the share of steps taken from the data, from a projected move, or from a query.

With the execution rule, Ours is about 90% / 10% / 0% data / model / query at ρ = 0, and 84% / 14% / 2% at ρ = 0.5. In-data planning has no model steps; its third component is the query (about 7% of steps at ρ = 0, 10% at ρ = 0.5).

Without the execution rule the return is unchanged at ρ = 0 (−3.7) and falls to −40.9 at ρ = 0.25 and −69.7 at ρ = 0.5. The step bar moves with it: model steps go from 10% to 65% and then 78%. That is the collapse the README summarises as −35 to −120; those two wider numbers are not this slice.

## Stitching

On the same driver slice, turning stitching off does not change the query rate (0.5%, 0.3%, 3.3% against 0.5%, 0.3%, 3.5%). Return at ρ = 0 goes from −3.74 to −3.89. At ρ = 0.5, no stitching is −15.4 and Ours is −16.3, so the projected shortcuts are not helping once half the doors are conditional. The query rates of Ours and RECO-greedy stay matched. This run does not support a claim that stitching reduces supervisor calls.

## Taxi L, if it is mentioned

Drivers, 10 seeds. At ρ = 0, Ours minus RECO-greedy is +3.7 at |D| = 150 and +2.6 at |D| = 400, 10/10 seeds, query difference 0. At ρ = 0.5 both intervals cover zero (about −0.5 and −1.9). Ten seeds is thin next to the taxi M tables.
