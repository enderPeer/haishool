# Independent audit of the first version 10 pilot

This is a read-only source audit of the frozen pilot implementation. No runtime
source was changed during the audit. The coordinator reported 63 local tests
passing, one local GPU test skipped, and 28/28 distributed smoke runs passing;
those are reported results, not an independent rerun by this audit. The 300-step
distributed pilot was still running when this document was written.

## What the implementation supports

The source contains no successful-seed search, habitability filter, synthetic
fallback, inserted organisms, copying motif or organic feed that replenishes
depleted matter. Every requested cosmic seed is archived and every final planet
returned by the inherited planet model is included in the chemical batch.
No-star and empty-system endings are retained. Numeric/model errors are retained
with the completed upstream states and error messages.

Within the chemical experiment, dissolved CHNOP species, peptide-like assemblies,
compartment membrane/cargo, photons and heat have explicit combined ledgers.
Condensation releases water and pays a linkage energy cost; hydrolysis consumes
water and returns stored linkage energy. Compartment uptake/fission/dissolution
debits or returns existing material. External photons and the signed thermal-bath
exchange are recorded. I found no unaccounted mass source or demonstrated
conservation failure in these inspected paths. That finding is bounded by the
implemented catalogue, the recorded numerical tolerances and the tests performed.

The earlier stale-scaffold, affinity-after-release and length-cap mobility issues
have been corrected in the audited source. The added regressions cover them.

## Material issues for interpretation and subsequent experiments

| Issue | Verified source behavior | Required treatment |
|---|---|---|
| Batch-dependent stochastic histories | Chemical initialization uses independent per-planet seeds, but `__main__.py:106` and `:108` create one assembly RNG and one compartment RNG from the first cosmic seed. Their draw order depends on the complete ordered batch. | Treat the whole ordered batch as an initial condition. A standalone planet is not guaranteed to replay its batched assembly/compartment history. A later implementation can use per-planet streams. |
| Cached inputs omitted from the top-level experiment hash | `__main__.py:65` calculates `experiment_sha256` before the cached chain digest is added at `:80`. Two different `--chains` payloads can therefore receive the same top-level experiment hash. | Preserve and verify `chain_inputs_sha256` and `chains.json` separately for this pilot. Include the input digest in experiment identity in a subsequent source version. |
| Manifest chain config can disagree with a supplied cache | The manifest reports a fresh `ChainConfig(sample_atoms=args.samples)`, while a cache can contain different resolution/config values. The actual cache is used without those configuration values being reconciled. | Cached `chains.json` is authoritative for actual planetary allocation/resolution. Validate or explicitly report requested versus effective config in the next protocol. |
| Pair validity is weaker than scientific comparability | `analysis.py:60` checks completed status and censorship. It does not require identical source/input hashes, matching effective configuration, or absence of chain errors. | Independently verify these fields before interpreting a paired difference. The coordinator states that this pilot uses identical cache/config/source and ordered seeds across controls; verify this against the archives rather than relying on `valid` alone. |
| Model-error planets can still be simulated | `chain.py` retains raw/partially constructed planets when an allocation/model error occurs; the CLI still includes all retained planet rows and records `chain_errors`. | These chemical histories are error diagnostics, not valid evidence that a planet's source budget was feasible. Exclude them from causal/physical claims while retaining their records. |
| Numerical capacity changes dynamics | Assembly capacity suppresses new chains and can suppress a hydrolysis split that would add a chain; length capacity suppresses elongation. Compartment capacity suppresses nucleation/fission. Both count capacities are global to the batch, so one planet can consume slots available to another physically independent planet. Events are marked censored. | A censored trajectory is computationally truncated, not a physical extinction or proof that structures cannot emerge. Repeat under larger predeclared capacities before making a conclusion. A later implementation should use per-planet capacities or explicitly describe a global batch resource ceiling. |

The above identity and pairing defects need not invalidate this frozen pilot if
the actual records agree. Fixing them during the running experiment would change
the source identity. Keep the first pilot unchanged and validate the corrections
under a new source hash.

## Scientific mechanisms that remain assumed or missing

- **No continuous whole-cosmos matter/energy trajectory.** Nuclear yields, stellar
  enrichment, gravitational collapse and disc formation remain separate coarse
  legacy models. Their raw states are retained, but atom identities and a universal
  energy ledger do not span them. `chain.py` allocates cloud fractions homogeneously
  across planetary masses; this is not simulated condensation, differentiation,
  atmospheric escape or local geochemistry. Microscopic parcel masses are parts
  of planetary material, not new inventories. Only CHNOP evolves in the chemical
  engine; other elements remain outside that reactive catalogue.
- **No chemistry in terminal nonstellar clouds.** The prefix keeps a non-fusing
  outcome and stops before planet formation. It does not investigate dark-cloud
  chemistry or brown-dwarf planetary systems. This is an inherited scope boundary,
  not a finding that such environments cannot contain structures.
- **Rates and chemistry are phenomenological.** Stoichiometry is balanced, but
  pathways, potential energies, selectivities and barriers are supplied. Species
  names such as ribose or adenine do not establish that realistic prebiotic
  mixtures would produce them selectively.
- **No phase- or pressure-aware solvent model.** Species diffuse on a periodic
  grid regardless of frozen/liquid/vapor conditions. Assembly condensation and
  hydrolysis probabilities have no local temperature dependence. Consequently,
  assemblies appearing on a 60 K or 900 K planet are behavior of these declared
  rules, not evidence of physically plausible survival there.
- **No demonstrated template reproduction.** Binary peptide-like scaffold
  adsorption biases addition of equal residue classes. It does not implement RNA
  complementary pairing, duplex separation or whole-strand copying. A scaffold
  association and sequence correlation are insufficient to claim heredity.
- **No polymer inheritance at compartment fission.** Dissolved cargo partitions,
  but sparse polymer graphs remain outside the cargo system. Membrane fission has
  no inherited sequence/genotype mechanism. Compartments use assumed uptake,
  leakage and fission rules; curvature, osmotic pressure and daughter geometry
  are unresolved. Multiple compartments sharing a grid cell do not have an
  aggregate geometric volume-exclusion model.
- **Catalysis has a limited structural interpretation.** A binary contact score
  increases both directions of the chemical catalogue. The assembly layer also
  boosts photon-paid chain nucleation without a corresponding structural
  hydrolysis-rate modulation. This is a supplied kinetic approximation, not a
  validated fold-to-catalytic-function model.
- **The structural screen does not establish persistence or life.**
  `analysis.py:28` checks global chain count and global maximum length at two
  sampled times. The same molecules need not survive between samples, and the
  count threshold and long chain can occur on different planets. Describe this
  as a repeated structural-population screen. It does not test individual
  persistence, self-maintaining replication, heritable variation or selection.
- **The pilot is short.** The CLI uses the default `dt_s=1`; 300 steps represent
  300 modeled seconds, not biological generations or geological time. More
  hardware increases the number of independent short experiments, not the
  scientific completeness of their mechanisms.

## Additional protocol edge cases

These do not appear necessary to invalidate the coordinator's frozen inputs, but
deserve regression tests before reusing the CLI as a general research service:

1. `__main__.py` clamps cached planet temperatures with `max(1e-6, temperature)`.
   Normal generated temperatures exceed this bound; malformed/nonpositive/NaN
   cache values can nevertheless be replaced silently. Reject invalid supplied
   temperatures rather than repairing them in a subsequent protocol.
2. A no-planet early completion occurs before `require_source`. In a live mutable
   checkout, source changes during chain generation would not be caught on that
   branch. Frozen deployment snapshots mitigate this pilot's exposure.
3. On refusal to reuse an existing folder, the outer CLI exception handler can
   modify an existing failed/incomplete manifest. Completed archives are
   protected. Preserve every preexisting archive, including failures, in the next
   version.
4. The CLI's combined ledger checks totals and finiteness, but does not explicitly
   check every local amount/heat/cargo for nonnegativity. Component tests check
   nonnegativity; add a runtime invalid-state diagnostic for longer experiments.

The next biological gate should require resource-paid reproduction with a
heritable polymer state and variation, followed by intervention controls and
independent repeats. Until that is demonstrated, the supported description is
**a conserved coarse prebiotic chemistry/assembly experiment**, with the life
verdict remaining `not_established`.
