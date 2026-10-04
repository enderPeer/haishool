# Test run t1: out-of-sample replication results

Completed 4 October 2026, about 19:00 UTC. The experiment is `runs/life10/v10-t1-oos-20261004`; the design and
predictions were fixed beforehand in [TESTRUN-T1.md](TESTRUN-T1.md).

## Scope and verification

- **65 of 65 runs completed:** 13 independent repeat lanes x 5 profiles, on all nine discrete GPUs and the four
  host CPUs. There were no errors and no capacity censorship.
- **Inputs:** cosmic seeds 16-31, all of them: 13 systems formed planets (75 planets), 2 have no star, 1 has no
  disc solids.
- **Source:** byte-identical to r1, with source identity `b809aab4…`.
- **Pairs:** all 52 normal/control pairs are valid. The raw manifests were re-read independently: each pair differs
  in config only in `profile`, and the source, chain inputs, seeds, steps and device are identical.
- **Conservation:** the worst relative element error is 1.0e-14 and the worst energy error 1.7e-14.

## Observed structures (totals over all 75 planets per repeat, final sample)

| Profile | Repeats | Oligomer packets, mean (range) | Compartments, mean (range) | Longest | Screen passes |
|---|---:|---:|---:|---:|---:|
| Normal | 13 | 158.2 (142-179) | 336.5 (311-356) | 7 | 13/13 |
| Dark | 13 | 0 | 0 | 0 | 0/13 |
| No template affinity | 13 | 168.9 (155-185) | 338.3 (315-357) | 7 | 13/13 |
| No structural catalysis | 13 | 159.2 (145-179) | 336.5 (311-355) | 7 | 13/13 |
| No compartments | 13 | 167.5 (150-183) | 0 | 7 | 13/13 |

r1 on seeds 0-15 gave 187.6 oligomer packets and 327.5 compartments for the normal profile. The counts differ
because the planets differ; the pattern is the same.

## Paired comparisons (two-sided sign tests over the 13 repeats, as pre-registered)

| Comparison | Normal higher / lower / tie | p | r1 (seeds 0-15) |
|---|---|---:|---|
| normal vs dark | 13 / 0 / 0 | 0.0002 | 14 / 0 / 0, p 0.0001 |
| normal vs no_template | **2 / 11 / 0** | **0.023** | 4 / 10 / 0, p 0.18 |
| normal vs no_catalysis | 0 / 2 / 11 | 0.50 | 3 / 4 / 7, p 1.0 |
| normal vs no_compartments | 2 / 10 / 1 | 0.039 | 4 / 10 / 0, p 0.18 |

## Predictions

| # | Prediction (fixed before running) | Outcome |
|---|---|---|
| 1 | dark gives no oligomers and no compartments in every repeat | **held** (13/13) |
| 2 | normal gives oligomers and compartments in every repeat | **held** (13/13) |
| 3 | no_template and no_catalysis do not differ significantly from normal | **failed for no_template:** switching template affinity off gave *more* oligomer packets in 11 of 13 repeats (p 0.023). Held for no_catalysis (p 0.50) |
| 4 | no_compartments gives no compartments and not fewer oligomers than normal | **held** (0 compartments; more oligomers in 10 of 13, p 0.039) |
| 5 | conservation errors below 1e-12 relative | **held** (1.0e-14 elements, 1.7e-14 energy) |

## What it means

- **The energy dependence replicates out of sample.** Without the model's energy input, nothing forms.
- **Template affinity has an effect, but against copying.** With the affinity on, the model makes fewer separate
  oligomer packets. r1 showed the same direction, without significance. One plausible reading is that scaffold
  adsorption holds residues and chains, so fewer free packets form; this was not tested here. Either way it is
  not evidence of template copying, which would mean more matching chains, not fewer packets.
- **Compartments reduce free oligomer packets in the same way** (p 0.039). The fatty-acid compartments take up
  material.
- **Catalysis has no detectable effect,** as in r1.
- **Multiple comparisons.** The pre-registered criterion was p <= 0.05 per test, without correction. With a Holm
  correction over the three control comparisons (not pre-registered), no_template (0.068) and no_compartments
  (0.077) would no longer be significant. Both readings are reported.
- **Life verdict: not established.** Nothing here shows heritable reproduction, self-maintenance or selection.
  The run covers 100 modelled seconds, and the model's limits are as in [AUDIT.md](AUDIT.md) and
  [RESULTS-r1.md](RESULTS-r1.md).

## Evidence

- `runs/life10/v10-t1-oos-20261004/report.json`: every run and pair, from `haishool.life10.analysis`.
- `runs/life10/v10-t1-oos-20261004/verification.json`: the independent pair and identity checks.
- Per-host raw lane folders, `manifest.json`, `chains.json` (seeds 16-31) and the frozen `source.tar.gz`.
