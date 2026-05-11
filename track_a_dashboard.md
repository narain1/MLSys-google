# Track A Dashboard

MLSys 2026 contest · Top 10 of 56 submissions · 78 total raw entries · 24 benchmarks

> When a participant submitted multiple times, only the most recent counts.

**Top score:** 19.22 points (22/24 benchmarks accepted).

## Leaderboard

Contest-rule normalized scoring: per-benchmark points = min(latency)/your(latency) when accepted, else 0. Total = sum across 24 benchmarks. Submission identifiers (A1..A10) are rank-ordered. Cells in **bold** are per-benchmark winners (points = 1.00). An em-dash means the submission did not produce a valid output for that benchmark.

| Rank | Submission | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | Total |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | A1: Vinci | 0.88 | **1.00** | **1.00** | 0.67 | 0.81 | 0.93 | 0.75 | **1.00** | 0.93 | 0.81 | 0.54 | 0.86 | 0.75 | 0.84 | 0.77 | 0.89 | 0.99 | 0.93 | — | 0.93 | 0.98 | — | 0.99 | 0.98 | 19.22 |
| 2 | A2: ZRC | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | 0.98 | **1.00** | — | **1.00** | **1.00** | 0.70 | 0.99 | — | — | 0.93 | 0.96 | **1.00** | 0.97 | **1.00** | 1.00 | **1.00** | — | — | **1.00** | 18.52 |
| 3 | A3: curling-grad | 0.64 | **1.00** | 0.78 | 0.45 | 0.67 | 0.71 | 0.65 | — | 0.51 | 0.82 | 0.71 | 0.88 | 0.75 | **1.00** | 0.80 | **1.00** | 1.00 | 0.98 | 0.99 | **1.00** | — | 0.99 | **1.00** | 1.00 | 18.32 |
| 4 | A4 | — | **1.00** | — | — | 0.96 | — | — | — | **1.00** | 0.88 | — | 1.00 | **1.00** | — | **1.00** | **1.00** | **1.00** | 0.97 | **1.00** | **1.00** | **1.00** | — | — | **1.00** | 13.80 |
| 5 | A5 | 0.88 | **1.00** | **1.00** | — | 0.79 | **1.00** | 0.81 | — | 0.93 | 0.82 | 0.66 | — | 0.75 | 0.95 | — | 0.94 | 1.00 | — | 1.00 | 0.96 | — | — | — | — | 13.49 |
| 6 | A6 | — | 0.84 | — | — | — | — | — | — | **1.00** | — | — | 1.00 | 0.99 | 0.99 | 0.88 | **1.00** | — | 0.96 | — | 0.98 | 1.00 | — | 0.98 | — | 10.63 |
| 7 | A7 | — | 0.71 | — | — | — | — | — | — | **1.00** | 0.89 | — | **1.00** | — | 1.00 | — | **1.00** | **1.00** | 0.96 | **1.00** | **1.00** | — | — | — | **1.00** | 10.56 |
| 8 | A8 | 0.70 | 0.43 | 0.76 | 0.45 | — | 0.76 | 0.59 | 0.51 | — | 0.81 | — | — | — | 0.88 | — | 0.90 | 0.98 | 0.92 | — | 0.93 | — | — | — | — | 9.63 |
| 9 | A9 | — | **1.00** | — | — | — | — | 0.47 | — | — | 0.79 | 0.70 | 0.98 | — | 1.00 | — | 0.89 | — | 0.92 | — | **1.00** | — | — | **1.00** | — | 8.76 |
| 10 | A10 | 0.88 | **1.00** | 0.78 | 0.67 | — | 0.81 | 0.81 | — | 0.80 | 0.94 | — | 0.89 | — | — | 0.84 | — | — | — | — | — | — | — | — | — | 8.40 |

## Special Innovation Award

A48: Jag

The Innovation Award rewards **pure novelty of mechanism**, independent of leaderboard score, validation, or cost-model accuracy. Each of the 56 deduped submissions is tagged with one or more techniques from a fixed taxonomy. The frequency table below shows how many submissions used each technique.

### Technique frequency across the field

Sorted within each category by descending frequency. _(unique)_ = used by exactly 1 sub. _(rare)_ = used by 2&ndash;3 subs.

#### Search paradigm

| Technique | # subs | Examples |
|:---|---:|:---|
| Topological greedy walk + adjacent-merge | 32 | A4, A6, A7, A8, A11, A15, … (26 more) |
| Simulated annealing | 8 | A5, A19, A20, A28, A29, A43, … (2 more) |
| Multi-strategy parallel portfolio | 8 | A9, A12, A17, A28, A31, A34, … (2 more) |
| Dynamic-programming partition | 7 | A4, A5, A13, A14, A33, A38, … (1 more) |
| Iterated local search / VND | 6 | A2, A28, A34, A36, A46, A47 |
| Beam search _(rare)_ | 3 | A1, A29, A53 |
| ILP / MIP / BLP / set-cover _(rare)_ | 2 | A3, A52 |
| Exact bitmask set-partition DP _(unique)_ | 1 | A34 |

#### Distinctive mechanisms

| Technique | # subs | Examples |
|:---|---:|:---|
| Anytime architecture (atomic-write valid baseline) | 6 | A8, A21, A22, A23, A24, A29 |
| Recomputation as first-class DP transition | 4 | A38, A39, A40, A48 |
| Online learned cost model (RF / ANN / surrogate) _(rare)_ | 3 | A8, A46, A47 |
| Bandit-driven move selection (UCB1 etc.) _(rare)_ | 3 | A28, A46, A47 |
| Fiduccia-Mattheyses local search _(rare)_ | 2 | A2, A28 |
| Pattern-catalog enumeration _(rare)_ | 2 | A3, A52 |
| Structural-fingerprint memoization _(rare)_ | 2 | A35, A48 |
| Forward + reverse Kahn topological order axis _(unique)_ | 1 | A17 |
| Structural region detector + per-region solvers _(unique)_ | 1 | A30 |
| Cycle-break preprocessing on undirected adj graph _(unique)_ | 1 | A11 |
| Edge-flip / is_cut[t] SA representation _(unique)_ | 1 | A5 |
| Treewidth-guided contraction or DP _(unique)_ | 1 | A34 |
| Brute-force oracle (validation only) _(unique)_ | 1 | A31 |
| Pareto multi-objective DP pruning _(unique)_ | 1 | A48 |
| Closed-form analytical cost from spec geometry _(unique)_ | 1 | A48 |
