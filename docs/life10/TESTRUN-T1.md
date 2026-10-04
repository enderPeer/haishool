# Test run t1: out-of-sample replication of version 10.0.1 (pre-registration)

Written 4 October 2026, before the run was prepared or launched. Nothing below may be changed after
launch. A change would be a new experiment.

## Question

Do the r1 observations (`RESULTS-r1.md`) hold on cosmic seeds that the implementation has never seen?
- Structures appear only with the model's energy input.
- Switching template affinity, structural catalysis or compartments off makes no detectable difference to
  oligomer counts.

## Fixed design

| Item | Value |
|---|---|
| experiment | `v10-t1-oos-20261004` |
| source | version 10.0.1, byte-identical to r1's frozen snapshot (all 39 source files checked by sha256 before preparation) |
| cosmic seeds | **16-31**, all of them, declared now. r1 used 0-15. No seed is skipped, replaced or ranked |
| chain inputs | `runs/life10/prebiotic-seeds-16-31-t1.json`, built with `build_chain(seed, 0, ChainConfig(sample_atoms=1_000_000))`. The same procedure reproduces r1's seed-0 chain byte for byte. 13 systems formed planets (75 planets), 2 have no star (22, 23), 1 has no disc solids (31), 0 chain errors. All 16 are kept |
| profiles | normal, dark, no_template, no_catalysis, no_compartments |
| resolution | grid 4, 1,000,000 sample quanta per planet, 100 modelled seconds (as r1) |
| lanes | 13 remote lanes, each an independent repeat (0-12): adler40 gpu0, gpu1, cpu; knecht24 gpu0-2, cpu; falke64 gpu0, gpu1, cpu; specht32 gpu0, gpu1, cpu. r1 also had a local Windows CPU lane (repeat 13); t1 does not, because the workstation is busy |
| bounds | 256 MiB per GPU lane, 900 s per lane, 512 MB output cap, 2 GB disk reserve (as r1) |

## Fixed analysis (decided now)

- Every run is reported: completed, failed, timed out and censored. Only uncensored, error-free, metadata-matched
  pairs are used for comparisons (`haishool.life10.analysis`, and the same independent re-reading of
  manifests and chains as r1's `verification.json`).
- Per profile: total oligomer packets and compartments across all planets at the final sample, mean and range
  over repeats.
- Repeated population screen: at least 10 oligomer packets and length at least 4 on the same planet in the last
  two samples. It is the same screen as r1, and passing it is not evidence of life.
- Comparisons:
  - normal against dark: a sign test over the paired repeats
  - normal against each of no_template, no_catalysis and no_compartments: a two-sided sign test over the paired
    repeats on total oligomer packets, at p <= 0.05
- Conservation: report the worst relative element and energy errors.
- Life verdict: stays `not_established` whatever the counts. No result of this design can establish heritable
  reproduction.

## Predictions recorded before running (from r1)

1. Dark gives no oligomers and no compartments in every repeat.
2. Normal gives oligomers and compartments in every repeat.
3. No_template and no_catalysis do not differ significantly from normal in oligomer packets.
4. No_compartments gives zero compartments and an oligomer count not lower than normal.
5. Conservation errors stay below 1e-12 relative.

A failed prediction is a result. It is reported, not explained away.
