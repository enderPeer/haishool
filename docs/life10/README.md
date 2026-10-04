# Version 10: unselected prebiotic experiments

Version 10 is a separate package. It never changes update 9 or inserts its bodies,
brains, plants or replicators. A cosmic seed is still one inherited stellar-system
model, not an atom-resolved universe. All requested seeds and all final planets are
retained, including no-star, barren and numerical-error outcomes.

Revision **10.0.1** uses independent random streams, local IDs and numerical
capacities for each physical planet. Cached inputs are part of experiment identity
and must match the recorded upstream configuration/laws. Control pairs are checked
for matching sources, actual inputs, seeds, repeat, physical configuration and
completed steps, and must be free of errors and censorship. Sparse graph work runs
on CPU between batched chemistry steps on CPU/CUDA/ROCm. The original frozen
10.0.0 pilot remains archived with its earlier global-cap and protocol limitations.

## Fixed experiment contract

- Seeds are declared before a run. No search for successful seeds, ranking, pruning,
  replacement planet, synthetic fallback or habitability selection is performed.
- The prebiotic chain ends at planet formation. No older biological stage is run.
- Every planet receives an elemental surface parcel by a fixed, reported allocation.
  The parcel is debited from planetary material; the rest remains in the bulk ledger.
- Chemical initialization partitions that elemental parcel into simple gases/water
  and unallocated atoms. No organic precursor, polymer, catalyst or vesicle is seeded.
- The source identity, input chains, config, profile, device and repeat seed are saved.
  Source changes during a run stop it. Existing experiment folders are never replaced.
- Energy and every modeled element are accounted across dissolved species,
  assemblies, compartment membranes/cargo, external photons and the thermal bath.
- Numerical capacities are reported as censorship, not extinction or absence of life.

## What is implemented

| Mechanism | Implementation | Boundary |
|---|---|---|
| Reaction kinetics | Bounded reversible mass action with Arrhenius barriers | Rates and energy potentials are assumed, not measured |
| Energy coupling | Photon-dependent precursor formation; explicit bath exchange and bond costs | Supplied stellar energy is a declared external flux |
| Precursor chemistry | Balanced named species: HCN, aldehydes, sugars, adenine, glycine, alanine and fatty acid | Collapsed pathways omit intermediates; balance does not prove chemical feasibility |
| Spatial transport | Conservative diffusion on local periodic grids | No molecular hydrodynamics or realistic solvent model |
| Bonds and breakdown | Residue-by-residue condensation, water bookkeeping and hydrolysis | Peptide-like phenomenology; no atomistic folding |
| Structural catalysis | Generic contact score affects rates in both directions | Binary contact approximation; no validated enzyme activity |
| Template assembly | Reversible scaffold adsorption and affinity-biased residue addition/release | Not RNA complementary copying; no whole sequence is copied or given a replicator flag |
| Compartments | Fatty-acid uptake, thermal selfassembly, cargo exchange, growth, fission and dissolution | No bilayer curvature/osmotic model; sparse polymers are not yet cargo inherited at fission |

The scientific purpose is to investigate a simplified chemical system, not to claim
that realistic abiogenesis has been solved. In particular, the absence of phosphorus
in the inherited cloud stays an absence; nucleotide precursors are not conjured to
repair it. The polymer alphabet is glycine/alanine-like residues, not fake RNA bases.

## Continuous-state boundary that remains

The inherited cosmos modules do not preserve atom identities or a universal energy
ledger through nucleosynthesis, stellar populations, gravitational collapse and disc
formation. `chain.py` retains their raw states and makes the planetary allocation
explicit; it cannot turn those models into one atomistic trajectory. Cloud material
is allocated homogeneously, without real condensation/fractionation. Full continuity
across those stages remains unfinished. Chemistry-to-assembly-to-compartment material
and energy are continuous within version 10's numerical resolution.

Existing static reaction catalogs, structural affinities, initial simple-molecule
partition, finite alphabet, thresholds, grid sizes, time step and capacity are fixed
model choices. None are learned laws of nature. Full life would additionally require
demonstrated heritable reproduction, sustained energy/resource processing and selection;
growth or membrane fission alone is not that evidence.

## Run and inspect

```
python -m pytest tests/life10 -q
python -m haishool.life10 run --seeds 0 1 2 3 --steps 300 --samples 1000000 --grid 4 --threads 2 --out runs/life10/example
python -m haishool.life10 run --seeds 0 1 2 3 --steps 300 --profile dark --out runs/life10/example-dark
python -m haishool.life10.analysis runs/life10/EXPERIMENT --out runs/life10/EXPERIMENT/report.json
```

Normal, dark, no-template, no-catalysis and no-compartment controls use the same declared
cosmic seeds and repeat. CPU/GPU repeats are independent experiments with recorded
random streams. CPU component checkpoints reproduce exactly; GPU float rounding may
diverge, so GPU agreement is statistical. The final `state.pt` contains chemical,
assembly and compartment state. CLI checkpoint continuation is not exposed yet.

Output files: `manifest.json`, `chains.json`, `summary.jsonl`, `summary.json`, `state.pt`.
No successful-state-only archive exists. The structural screen is fixed before the
pilot: at least 10 oligomer packets and length at least 4 in the last two samples.
The revised screen requires both thresholds on the same planet in two successive
samples; it is a repeated population screen, not proof that the same molecules
persisted. All reported life verdicts remain `not_established`.

## Cluster protocol

`scripts/life10_cluster.py` freezes sources and inputs, inventories CPU/GPU resources,
and runs bounded independent lanes. It preserves existing services, excludes integrated
GPUs, uses physical free-VRAM estimates and limits allocations to 256 MiB per GPU lane.
Each remote experiment has an exclusive folder, disk reserve, output cap and timeout.
All discrete cards that fit those bounds and all host CPUs are tested; local CPU runs
are added separately. A skip or failed lane is preserved in the final report.

## Research context

Prebiotic glycine/alanine chemistry motivates the named precursor routes, while the
implemented rates remain uncalibrated. Peptide template chemistry is possible in
specific laboratory systems, but those results do not validate this binary model:

- Parker et al., amino acids in archived spark-discharge experiments:
  https://hsd.gsfc.nasa.gov/sed/content/uploadFiles/publication_files/Parker2011a.pdf
- Lee et al., a designed self-replicating peptide (1996):
  https://www.nature.com/articles/382525a0
- Ashkenasy et al., peptide network simulations and experiments (2017):
  https://www.nature.com/articles/s41467-017-00463-1

No coefficient in version 10 is presented as measured by these papers.
