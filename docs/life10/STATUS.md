# Version 10 status, 3 October 2026

## Complete, 14:26 UTC follow-up

Both experiments ended and every original/revised outcome was collected. Original:
50 completed profiles, four GPU timeouts, 16 unstarted controls, 30 completed
censored profiles, zero uncensored valid pairs. Revised: 70/70 completed runs,
56/56 valid pairs, no errors or censorship. Independent hashes/configuration/seed
checks passed. See `COMPLETION.md`, both experiment reports and the original
`independent-audit.json`. No further jobs are needed; the heartbeat is disabled.
Life remains not established. The following sections are historical progress notes.

## Current revision, 13:26 UTC follow-up

Version **10.0.1** includes all three reviewed patches below. They were applied
after the workspace's original CPU pilot finished; remote originals keep their
immutable snapshots. **100 local tests passed**, with two GPU-only skips.
Per-planet RNGs, IDs and capacities; cached-input hashing/configuration/law checks;
control pairing and a same-planet repeated-population screen are implemented.
Sparse graph work now runs on CPU around batched chemistry on the chosen device.
Physical rates were not changed.

The original knecht24 GPU0/GPU1/GPU2 and specht32 GPU1 lanes hit their declared
30-minute timeout; retain their partial records as incomplete experiments. Other
original lanes may still be draining. Local original CPU completed all profiles.

Fresh experiment **v10-r1-validation-20261003** runs 100 modeled seconds, all
16 seeds, all 68 planets, five profiles on nine GPU and five CPU hosts. It uses
regenerated `runs/life10/prebiotic-seeds-0-15-r1.json`. Do not change its runtime
sources while running. Status and collection use the same cluster commands with
this new name. Read both experiment statuses before concluding work is finished.

Once both cohorts end, collect and independently verify every outcome. Use the
revised analyzer to preserve failed, timed-out and censored runs as invalid for
causal claims. No life is established by oligomer growth or compartment fission.
The heartbeat has been updated with these two experiment names and must disable
itself only after collection and revised validation/reporting finish.

## Original implementation and pilot notes (historical)

**Revised validation completed:** 70/70 runs, all nine GPUs and five CPU hosts,
56/56 independently verified valid control pairs, no errors or censorship.
Worst relative element/energy errors were 1.13e-14 and 1.06e-14. Evidence and
interpretation are in `RESULTS-r1.md` and the revised experiment's `report.json`
and `verification.json`. Life remains not established. At the last check the
original falke64 GPU1 and specht32 GPU0 lanes were still running; the heartbeat
must collect originals before disabling itself.

The requested separate prebiotic package is implemented and the first bounded
distributed experiment is running. This is not a completed abiogenesis claim.

- Local gates: 63 passed, one GPU-only test skipped locally.
- Hardware compatibility: 28/28 normal/dark runs completed on nine discrete GPUs
  and five CPU hosts. No other services were stopped. Full report:
  `runs/life10/v10-smoke-20261003/report.json`.
- Predeclared cosmic seeds: 0 through 15; four no-star systems and 68 retained
  planets. No biological stage, successful-seed filter or fallback was used.
- Frozen pilot: `v10-pilot-20261003`; 300 modeled seconds, five profiles, 14
  independent repeat lanes. Sources are archived in the experiment's source tar.
  Original results must not be edited when implementation fixes are applied.
- The pilot produces organic precursors, peptide-like oligomers and compartments.
  Global compartment capacity is reached; affected batches are censored and cannot
  establish independent planetary population limits or life.
- A reviewed correction is staged at `runs/life10/protocol-integrity.patch`.
  It must not be applied while local pilot processes still read mutable source
  identities. It fixes input hashing/config validation and paired-control checks.
- On the 13:16 UTC follow-up, isolated work was dispatched for
  `runs/life10/per-planet-independence.patch` (separate planetary random streams
  and capacities) and `runs/life10/sparse-phase-performance.patch` (CPU sparse
  graph work with batched GPU chemistry). These artifacts must be tested and
  combined only after live workspace processes end; they never alter the first
  frozen pilot. They may still be in preparation when this record is read.
- Scientific and architectural boundaries are recorded in `AUDIT.md` and README.
  In particular, no inherited polymer genomes in daughter compartments, realistic
  complementary RNA copying, or complete matter/energy continuity across cosmic
  stages has been implemented. Extreme-temperature/solvent claims are unsupported.

The heartbeat `finish-version-10-chemical-life-pilot` checks this chat every ten
minutes, quietly while unchanged. It is authorized to collect every completed
outcome, validate pairing independently, apply staged integrity fixes after the
frozen pilot ends, validate a revised frozen experiment on the available hardware,
report actual results and limitations, and disable itself after finishing.

Remote folders: `~/life10/v10-pilot-20261003` on adler40, knecht24, falke64, specht32.
Local lane: `runs/life10/v10-pilot-20261003/local/cpu` (repeat 13).
Status command: `python scripts/life10_cluster.py status v10-pilot-20261003`.
Collect command: `python scripts/life10_cluster.py collect v10-pilot-20261003`.
Never remove results to make collection succeed; keep existing destinations and
recover missing files into a fresh directory if needed.
