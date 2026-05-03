# Track B Dashboard

MLSys 2026 contest · Top 10 of 32 submissions · 41 total raw entries · 24 benchmarks

> When a participant submitted multiple times, only the most recent counts.

**Top score:** 20.97 points (23/24 benchmarks accepted).

## Leaderboard

Contest-rule normalized scoring: per-benchmark points = min(latency)/your(latency) when accepted (median over 3 attempts), else 0. Total = sum across 24 benchmarks. Submission identifiers (B1..B10) are rank-ordered. Cells in **bold** are per-benchmark winners (points = 1.00). An em-dash means no valid output.

| Rank | Submission | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | Total |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | B1 | **1.00** | 0.50 | **1.00** | 0.88 | 0.92 | 0.99 | 0.83 | **1.00** | **1.00** | **1.00** | 0.72 | 0.81 | — | 0.87 | **1.00** | 0.79 | 0.98 | 0.96 | 0.85 | 0.92 | 0.98 | **1.00** | **1.00** | 0.98 | 20.97 |
| 2 | B2 | **1.00** | 0.50 | **1.00** | 0.88 | 0.85 | 0.99 | 0.83 | **1.00** | 0.79 | 0.78 | 0.56 | 0.81 | **1.00** | 0.83 | 0.81 | 0.54 | 0.98 | 0.96 | 0.84 | 0.91 | 0.98 | **1.00** | **1.00** | 0.98 | 20.83 |
| 3 | B3 | **1.00** | **1.00** | **1.00** | **1.00** | 0.74 | 0.99 | 0.96 | **1.00** | 0.56 | 0.81 | **1.00** | 0.89 | — | **1.00** | 0.95 | **1.00** | 0.98 | — | **1.00** | **1.00** | — | **1.00** | — | 0.98 | 18.86 |
| 4 | B4 | 0.40 | 0.50 | 0.40 | 0.36 | 0.49 | 0.58 | 0.44 | 0.45 | 0.34 | 0.78 | 0.68 | 0.81 | **1.00** | 0.87 | 0.95 | 0.76 | 0.98 | 0.96 | 0.85 | 0.92 | 0.98 | **1.00** | **1.00** | 0.98 | 17.48 |
| 5 | B5 | **1.00** | — | — | 0.88 | — | — | — | — | — | 0.78 | 0.54 | 0.81 | — | 0.80 | 0.81 | 0.54 | 0.97 | 0.92 | 0.80 | 0.88 | 0.95 | 0.96 | 0.95 | 0.94 | 13.53 |
| 6 | B6 | — | **1.00** | **1.00** | 0.94 | **1.00** | — | — | — | 0.85 | — | — | **1.00** | — | 0.96 | — | 0.58 | **1.00** | — | — | 0.96 | — | — | — | — | 9.30 |
| 7 | B7 | — | **1.00** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.99 | **1.00** | 0.85 | — | **1.00** | — | — | **1.00** | 5.84 |
| 8 | B8 | — | **1.00** | — | 0.79 | — | — | — | — | — | — | — | — | **1.00** | — | — | — | 0.98 | — | — | — | — | — | — | — | 3.77 |
| 9 | B9 | — | **1.00** | — | — | — | **1.00** | — | — | 0.45 | — | — | — | — | — | — | — | — | — | — | — | 0.98 | — | — | — | 3.43 |
| 10 | B10 | — | **1.00** | — | **1.00** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.00 |

## Special Innovation Award

The Innovation Award rewards **pure novelty of mechanism**, independent of leaderboard score, validation, or spec-compliance (those are separate gates). Each of the 32 deduped submissions is tagged with one or more techniques from a fixed taxonomy. The frequency table below shows how many submissions used each technique.

### Technique frequency across the field

Sorted within each category by descending frequency. _(unique)_ = used by exactly 1 sub. _(rare)_ = used by 2&ndash;3 subs.

#### LLM role / agent paradigm

| Technique | # subs | Examples |
|:---|---:|:---|
| Best-of-N + iterative-feedback (BASELINE) | 14 | B1, B3, B4, B5, B8, B10, … (8 more) |
| Multiple LLM agents (debate / personas / role specialization) | 5 | B2, B7, B9, B17, B27 |
| LLM picks strategy / persona for deterministic search | 4 | B7, B12, B18, B25 |
| LLM emits Python that runs (FunSearch family) _(rare)_ | 2 | B21, B32 |
| LLM picks search-control plan only (never a schedule) _(rare)_ | 2 | B6, B9 |
| LLM ranks pre-computed candidate moves _(unique)_ | 1 | B16 |
| LLM picks hyperparameters for deterministic solver _(unique)_ | 1 | B28 |

#### Notable mechanisms

| Technique | # subs | Examples |
|:---|---:|:---|
| Valid baseline computed before any LLM call | 8 | B6, B10, B13, B14, B15, B22, … (2 more) |
| Validator errors injected into next prompt (Reflexion-style) | 5 | B2, B5, B13, B21, B32 |
| JSON-schema-enforced LLM output | 5 | B2, B6, B7, B12, B26 |
| Multi-step action macros / waypoints _(rare)_ | 2 | B17, B27 |
| Multi-restart with strategy / temperature variation _(unique)_ | 1 | B4 |
| Multi-API-key rotation for quota _(unique)_ | 1 | B25 |
| Deterministic strategy ensemble + LLM overlay _(unique)_ | 1 | B11 |
| Local-search confidence gates the LLM call _(unique)_ | 1 | B12 |
| Agent pre-detects structural opportunities and surfaces in prompt _(unique)_ | 1 | B1 |
| LLM emits Python that runs in restricted namespace _(unique)_ | 1 | B9 |
| Direct urllib transport, no Gemini SDK _(unique)_ | 1 | B6 |
| Forbidden-execution-content filter on LLM output _(unique)_ | 1 | B6 |
| Gemini Context Caching (routes through aiplatform — banned per spec) _(unique)_ | 1 | B26 |
