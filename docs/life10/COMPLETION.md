# Version 10 completion record

Completed 3 October 2026, after the final original pilot lane ended. No simulation
jobs remain running for these two experiments, and no new search was launched.
Update 9 and unrelated services were preserved.

## Delivered revision

Version **10.0.1** provides unselected prebiotic chain inputs, conservative named
precursor kinetics, energy coupling, spatial transport, peptide-like assemblies,
structural rate effects and fatty-acid compartments. It records no-star/barren
outcomes and every formed planet. Independent per-planet RNGs/capacities, frozen
input/source identities, validated control pairing and sparse CPU/GPU execution
are implemented.

**100 local tests passed; two GPU-only checks skipped locally.** Distributed
validation completed **70/70 runs** over **nine discrete GPUs and five CPU hosts**.
All **56 control pairs** were independently verified, with no errors or censorship.
See [RESULTS-r1.md](RESULTS-r1.md) for the numerical observations and limits.

## Original frozen pilot, fully preserved

| Planned profiles | Completed | GPU timeouts | Controls never started | Completed censored profiles |
|---:|---:|---:|---:|---:|
| 70 | 50 | 4 | 16 | 30 |

The four timeouts were knecht24 GPU0/GPU1/GPU2 and specht32 GPU1, each at its
configured 30-minute limit on the normal profile. Their later controls never ran.
Partial files and raw manifests remain unchanged; terminal worker status identifies
the timeouts even where a raw manifest still says `running`.

All 40 completed normal/control pairs have matching metadata, but every normal
profile reached the original global compartment capacity. Thus **zero original
pairs are uncensored and valid for causal interpretation**. They were not merged
with revised-source pairs. Original completed element/energy ledger residuals were
below 3.11e-14 and 3.08e-14 relative, respectively.

Independent audit verified source archives, every frozen file hash, cached input
hashes, effective configuration, ordered seeds, lane repeat/device and completed
steps. No metadata mismatch was found. Both cohorts retain the four no-star systems
and all 68 planets from predeclared cosmic seeds 0–15.

## Evidence

- Original: `runs/life10/v10-pilot-20261003/report.json`, `verification.json`,
  `independent-audit.json` and all per-host/local raw profile archives.
- Revised: `runs/life10/v10-r1-validation-20261003/report.json`,
  `verification.json` and all per-host/local raw profile archives.
- Source snapshots and manifests remain in both experiment roots.

## Scientific conclusion

**Structures formed under the supplied simplified rules; biological life is not
established.** Rates, energies and structural affinities are phenomenological.
The model still lacks realistic phase/pressure-aware chemistry, complementary RNA
replication, polymer inheritance inside daughter compartments and whole-cosmos
matter/energy continuity. The runs span 300 and 100 modeled seconds, not generations
or geological time. See [README.md](README.md) and [AUDIT.md](AUDIT.md).

The follow-up automation is disabled after this collection/reporting step.
