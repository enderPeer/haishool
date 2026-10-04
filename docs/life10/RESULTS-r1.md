# Version 10.0.1 validation results

Completed 3 October 2026. The revised frozen experiment is
`runs/life10/v10-r1-validation-20261003`.

## Scope and verification

- **70 of 70 runs completed:** 14 independent repeat lanes, five profiles each.
- **Hardware:** nine discrete GPUs (adler40: two; knecht24: three; falke64: two;
  specht32: two) and five CPU hosts including the local Windows machine.
- **Inputs:** all cosmic seeds 0–15; four no-star systems and all 68 final planets
  from the other 12 systems. No successful-seed or habitability selection.
- **Duration:** 100 modeled seconds per chemical history, not generations or
  geological time. Grid 4×4; 1,000,000 atom-equivalent sample quanta per planet.
- **Validation:** 100 local tests passed; two locally skipped GPU checks were
  supplemented by the distributed device runs.
- **Pair checks:** all 56 normal/control pairs completed, were uncensored, and
  matched sources, actual input chains, ordered seeds, repeat and step count.
  Raw manifests and chains were independently reread, not only analyzer flags.
- **Conservation:** worst relative elemental error 1.13×10⁻¹⁴; energy error
  1.06×10⁻¹⁴. No recorded chain/model errors or capacity censorship.
- **One source identity:**
  `b809aab441b6301eae614e8decbcbdbba6f4c4a5c7caa634c1c9a56ec4580a81`.

## Observed structures

Values below are totals across all 68 planetary parcels in one repeat, at the
final sample. Four no-star systems remain in each archive with no chemical planets.

| Profile | Repeats | Oligomer packets, mean (range) | Compartments, mean (range) |
|---|---:|---:|---:|
| Normal | 14 | 187.6 (171–214) | 327.5 (308–353) |
| Dark | 14 | 0 | 0 |
| No template affinity | 14 | 195.4 (183–214) | 327.7 (313–349) |
| No structural catalysis | 14 | 188.1 (171–211) | 328.5 (308–354) |
| No compartments | 14 | 195.5 (178–213) | 0 |

The normal and enabled-structure controls produced organic precursors, short
peptide-like oligomers and, where enabled, fatty-acid compartments. The dark
control produced neither oligomers nor compartments under these fixed routes.
This establishes a dependency on the model's energy input in this pilot. It does
not establish useful template replication, a catalytic advantage or biological
life. Differences between enabled controls have not been shown to be significant.

The repeated population screen passed in 56 runs; all 14 dark runs failed that
screen. The screen requires at least 10 oligomer packets and a length of at least
four on the same planet in the last two distinct samples. It does not track the
same molecules, self-maintenance or heritable reproduction.

## Limits of this result

All reaction energies, rates, scaffold affinities and compartment rules are
phenomenological assumptions. The chemistry has no phase/pressure-aware solvent
model and peptide event rates do not depend on local temperature. Therefore,
structures on very cold or hot modeled planets are not physically plausible-life
evidence. Sparse polymers are not inherited inside daughter compartments.
Whole-cosmos atom identities and energy continuity remain absent in the inherited
upstream models. See `AUDIT.md` for the full scientific boundaries.

**Life verdict: not established.** The experiment validates the numerical
foundation and shows structures within its supplied simplified laws.

## Evidence files

- `runs/life10/v10-r1-validation-20261003/report.json`: every run, profile and pair.
- `runs/life10/v10-r1-validation-20261003/verification.json`: independent checks.
- Per-host lane/profile folders: manifests, all input chains, sampled histories
  and final checkpoints.
- `runs/life10/v10-r1-validation-20261003/source.tar.gz`: immutable source snapshot.

The earlier 300-second `v10-pilot-20261003` is a separate immutable experiment.
It encountered global-cap censorship and device timeouts. Its remote lanes have
ended and were collected separately: 50 completed profiles, four timeouts and
16 controls that never started.
Its observations must not be merged with this revised source's controlled pairs.
