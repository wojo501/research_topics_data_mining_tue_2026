# Results notes: solvable-layout rerun

The current Taxi analyses use `results/main_M_solvable.csv` and
`results/main_L_solvable.csv`. The original CSVs are historical and are not mixed
with the reruns. The Taxi-v3 reproduction file is unchanged.

## Protocol

Each layout draw has exactly `round(rho * number_of_doors)` conditional doors.
The same seeded generator resamples only the conditional-door subset until the
empty taxi can reach every pickup and every start can finish within 200 steps.
The first valid draw is retained; the data-generation seed is unchanged. This
conditions the layout distribution on solvability, rather than removing seeds
from the final comparison. All nine agents receive identical maps and batches.
The current value-iteration tolerance is 1e-6.

M uses all 1,200 initial states and 30 seeds per configuration. L uses the same
seeded sample of 1,000 initial states per configuration as the original protocol,
with 10 seeds. Solvability is checked over **all** starts, including starts not
selected for L evaluation. Imitation and RECO use three rollouts per evaluated
start; the other methods use one. No learned-policy failures are filtered out.

## Coverage and resampling

- **Taxi M:** 4,860 rows; all nine methods have all 30 seeds in every condition. At rho = 0.5, resampled seeds: 6, 13, 15. Maximum number of draws: 2; worst optimal episode length: 47 steps.
- **Taxi L:** 720 rows; all nine methods have all 10 seeds in every condition. At rho = 0.5, resampled seeds: 3, 4, 7. Maximum number of draws: 4; worst optimal episode length: 99 steps.

## Paired return and query differences

All differences are **Ours minus RECO-greedy**. Return is undiscounted. Query
changes are percentage points of episodes with at least one supervisor call.
Intervals are 95% percentile bootstrap intervals over paired seeds (10,000
resamples). The paired tables also contain discounted return and sign-flip tests.
Positive return favors Ours; positive query change means more episodes need help.

| Environment | Dataset | rho | Trajectories | Seeds | Return difference | 95% CI | Query change (pp) |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| taxi-L | drivers | 0 | 150 | 10 | +3.71 | [+3.45, +3.98] | +0.00 |
| taxi-L | drivers | 0 | 400 | 10 | +2.61 | [+2.36, +2.85] | +0.00 |
| taxi-L | drivers | 0.5 | 150 | 10 | -4.70 | [-6.76, -2.57] | +4.07 |
| taxi-L | drivers | 0.5 | 400 | 10 | -6.65 | [-8.95, -4.25] | -0.10 |
| taxi-L | random_mix | 0 | 150 | 10 | +422.24 | [+389.09, +468.33] | +0.00 |
| taxi-L | random_mix | 0 | 400 | 10 | +341.32 | [+314.43, +367.15] | +0.00 |
| taxi-L | random_mix | 0.5 | 150 | 10 | +392.73 | [+350.68, +438.59] | +0.00 |
| taxi-L | random_mix | 0.5 | 400 | 10 | +360.60 | [+339.22, +382.30] | +0.00 |
| taxi-M | drivers | 0 | 50 | 30 | +1.20 | [+1.06, +1.34] | +0.00 |
| taxi-M | drivers | 0 | 150 | 30 | +0.65 | [+0.60, +0.70] | +0.00 |
| taxi-M | drivers | 0.25 | 50 | 30 | +0.34 | [-0.01, +0.67] | -0.06 |
| taxi-M | drivers | 0.25 | 150 | 30 | +0.09 | [-0.20, +0.34] | +0.00 |
| taxi-M | drivers | 0.5 | 50 | 30 | -1.45 | [-1.93, -0.95] | +0.45 |
| taxi-M | drivers | 0.5 | 150 | 30 | -1.22 | [-1.66, -0.81] | +0.33 |
| taxi-M | noisy_expert | 0 | 50 | 30 | +0.70 | [+0.65, +0.75] | +0.00 |
| taxi-M | noisy_expert | 0 | 150 | 30 | +0.47 | [+0.45, +0.50] | +0.00 |
| taxi-M | noisy_expert | 0.25 | 50 | 30 | +0.06 | [-0.19, +0.29] | +0.00 |
| taxi-M | noisy_expert | 0.25 | 150 | 30 | -0.06 | [-0.24, +0.12] | +0.00 |
| taxi-M | noisy_expert | 0.5 | 50 | 30 | -1.27 | [-1.60, -0.92] | +0.00 |
| taxi-M | noisy_expert | 0.5 | 150 | 30 | -1.23 | [-1.51, -0.93] | +0.00 |
| taxi-M | random_mix | 0 | 50 | 30 | +226.74 | [+190.37, +262.99] | +0.00 |
| taxi-M | random_mix | 0 | 150 | 30 | +4.83 | [+2.66, +7.29] | +0.00 |
| taxi-M | random_mix | 0.25 | 50 | 30 | +225.90 | [+188.17, +262.42] | +0.00 |
| taxi-M | random_mix | 0.25 | 150 | 30 | +10.96 | [+5.75, +17.14] | +0.00 |
| taxi-M | random_mix | 0.5 | 50 | 30 | +245.25 | [+207.32, +283.95] | +0.00 |
| taxi-M | random_mix | 0.5 | 150 | 30 | +31.45 | [+20.77, +43.58] | +0.00 |

## Files and interpretation

`notebooks/taxi_solvable_rerun.ipynb` contains the tests, exact rerun commands,
per-seed layout audit, and complete-method/seed assertions. The CSV metadata
records `layout_attempts`, `max_optimal_steps`, and `require_solvable`.

`python -m notebooks.analyze_results` regenerates the paired tables and existing
figures from the new CSVs. Figures 2, 3, 5, and 6 show Taxi M; the paired tables
also include L. Figure 1 still uses the original Taxi-v3 reproduction. The
results-analysis notebook also loads the new M/L files.

These results replace the previous feasibility-filtered summaries. The map
selection rule uses only true environment reachability and the episode horizon,
never any agent's return or query rate. Solvability does not guarantee a learned
agent's success, nor does it establish a return/query dominance theorem.
