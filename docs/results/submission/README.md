# Leaderboard submission — SimpleToBuildQIStellarator

## What this is
`best_feasible_simple_to_build.json` — our best canonically-feasible boundary,
in the exact `SurfaceRZFourier` format the leaderboard winner files use
(`r_cos` / `z_sin` / `r_sin` / `z_cos` / `n_field_periods` /
`is_stellarator_symmetric`).

## Verified score (reproduced the leaderboard's own metric)
| field | value |
|---|---|
| **score** | **0.5818** |
| objective (min normalized L_grad-B) | 11.636 |
| feasibility violation | 0.0072 (feasible) |
| mode budget | 8×8, NFP=3 |

Verification is end-to-end: the number above comes from **reloading the saved
file from disk** and running `problems.SimpleToBuildQIStellarator().evaluate(...).score`
— the exact call the leaderboard makes. That same pipeline reproduced the
published winner scores to four decimals (davidkh 0.6361, lhhhhappy 0.5371),
so this is the real metric, not a proxy. See `PROVENANCE.json`.

## Where it lands
Score = objective/20 for feasible designs. Against the on-disk top-8 snapshot,
0.5818 **clears the current bottom-of-top-8 bar** (lhhhhappy 0.5371) and sits
below 7th (RisoLiao 0.6071). Gap to 1st (davidkh 0.6361 / objective 12.72):
**+1.08 objective ≈ +0.054 score**. Caveat: the snapshot is on-disk, not the
live board — treat this as "credible, board-clearing entry", not a guaranteed
public rank.

## Provenance
- Source: `results/family-study/climb/c8_climb_bank.jsonl`, `call_index` 30548 —
  the max-objective member of the canonically-confirmed-feasible bank, produced
  by the Phase C climb (own-lineage NAE/descent search lifted from 4×4 to 8×8;
  the higher-mode content is what carried objective 11.1 → 11.64). This is the
  campaign's genuine best; it beats the older 4×4 bank best (11.124 / 0.5562),
  confirmed by a repo-wide scan of all canonically-feasible designs by objective.
- NOT from the recent field-space work, which is infeasible and scores 0.

## Submitting (your action)
Submission to the ConStellaration leaderboard is an outward-facing step and is
yours to perform — this package does not auto-submit. The artifact to upload is
`best_feasible_simple_to_build.json`. Confirm the current submission channel /
format on the benchmark's own instructions before uploading.
