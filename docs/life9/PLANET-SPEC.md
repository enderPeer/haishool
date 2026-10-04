# life9 planet engine: design contract and as-built record (update 9, version 3)

This file has two jobs:
- It is the contract the `haishool/life9/planet/` modules were built against.
- It is the design record: what the code actually does.

The engine simulates **a spherical planet derived from the world7 chain**: star, then elements, then planet,
then chemistry, then life. It has:
- gravity and layers (metal core, rock mantle, crust, ocean, atmosphere)
- climate and a biosphere
- chemistry-based materials and open-ended crafting
- living individuals with evolving brains

It runs on the GPU (PyTorch, CUDA or ROCm) and exactly on the CPU.

**How to read this record.** Each section 2.x has two parts:

- **Contract.** The design as agreed before building. It is kept as written, with one exception: plain factual
  errors are corrected in place and marked *(corrected v3)*.
- **As built.** What the code does today:
  - the formulas and constants
  - deviations from the contract, with their reasons
  - the `new_rule` choices
  - the open issues

  Where the two differ, the code and its As built text hold.

**Status on 2026-10-03:**
- 13 modules: `constants`, `chain`, `formation`, `materials`, `items`, `globe`, `climate`, `biosphere`, `brain`,
  `creatures`, `senses` and `crafting`, plus `__init__`.
- 296 tests are collected in `tests/life9/planet/`. The two CUDA tests skip on a CPU host.
- Everything has run on the CPU only. Nothing has run on a GPU yet.
- `world.py`, `__main__.py` and `view.py` are being written in parallel. This record does not describe them.
  Section 2.12 lists what they must do.

Sources for the as-built text:
- the code itself
- the builder and reviewer reports of the build workflow
- two CPU probes run for this record (G = 16, terrain seed 1; quoted as "probe")

## 0. Rules every module keeps

1. **Provenance.** Every number that sets up a world carries a tag:
   - `chain`: taken from a world7 era or level.
   - `derived`: computed from chain values with a physical formula.
   - `reference`: published physical or chemical data with a citation in the code, such as CODATA, NIST-JANAF,
     the CRC Handbook, or a Mohs hardness table.
   - `new_rule`: a modelling choice that the chain does not decide. It is stated in the code and in
     `PlanetSpec.provenance`.

   No silent invention.
2. **SI units inside.** m, kg, s, K, J, W, Pa, mol. One tick is `dt = 86400 s` (one day; `Config.dt_days`).
   Any non-SI field name carries its unit, for example `oxygen_pal`.
3. **Closed ledgers**, checked by tests:
   - energy of the individuals
   - carbon: plants, litter, atmosphere CO2, bodies, items
   - oxygen: atmosphere O2 against photosynthesis, respiration and combustion
   - water: ocean, air moisture, soil, snow, bodies
   - element and species mass of items: collected from deposits, transformed, dropped, decayed

   Each ledger closes within float32 tolerance per world, summed in float64.
4. **No meaning is supplied.** There is no reward for calling, crafting or cooperating, and no tech tree.
   - **Tool effects come from physics formulas.** A tool's value comes from its hardness, sharpness, mass and
     lever, never from a recipe name.
   - **Transformations follow from conditions.** They depend on temperature, reducing conditions and the inputs
     present, never on a label.
   - **Learning uses only the individual's own outcome:** its energy, water and health change.
   - **Nothing learned is inherited.**
5. **Batched worlds.** Tensors have a leading `W` (worlds) axis. Different worlds may be different planets
   (different seeds). The grid topology is shared, and only the values differ. Everything must run on
   `device="cpu"` (exact, the tests) and on `"cuda"` (CUDA or ROCm).
6. **Pure PyTorch and standard library** in the engine. NumPy is allowed only in `chain.py`, `formation.py`,
   tests and analysis. world7 itself needs NumPy.
7. **Determinism.** One `torch.Generator` per world object, saved in checkpoints. The same seed on the CPU gives
   the same state hash, and `from_state` continues exactly on the CPU.

### As built

**Where provenance lives.** Tags may be joined with `+` when a value combines several, for example
`derived+new_rule`: a physical formula applied under a stated choice.

| Module | Where |
|---|---|
| `chain` | `inputs["provenance"]`: dotted key to era or level and `file:line` |
| `formation` | `PlanetSpec.provenance[field] = (tag, note)`. `check_spec` fails on a missing or unknown tag. |
| `materials` | Every table entry is a `Val` (a float with `.tag` and `.source`). `provenance()` has 562 entries. |
| `globe` | `PROVENANCE` holds every public upper-case constant as `NAME = value`. A test inspects the module. |
| `climate` | `PROVENANCE` (one entry per `ClimateRules` field) and `make_params(...)["provenance"]` (per-world values) |
| `biosphere` | `PROVENANCE`, one entry per `BioRules` field |
| `brain` | `PROVENANCE`, keyed by `BrainParams` fields, gene names, layout and outcome constants |
| `creatures`, `senses` | `PROVENANCE`. A test requires every `reference` note to cite an author and year or a standard. |
| `crafting` | Every constant is a `materials.Val`. `provenance()` lists them. |

Merging all of these into one export is a job for `world.py` (open).

**Ledgers.** Each module closes its own ledgers. The table gives the measured errors.

| Ledger | Function | Measured error |
|---|---|---|
| energy, carbon and water of the individuals | `creatures.ledger_errors` | below 2e-6 relative in a busy 120-day test; about 4e-8 in a 200-day run |
| carbon of plants, litter and air CO2 | `biosphere.carbon_ledger` | about 1e-7 kg C/m² after 2 years |
| oxygen | `biosphere.oxygen_ledger` | about 1e-5 mol/m², against a column of about 10,000 mol/m² |
| water of the climate | `climate.water_ledger` | about 1e-6 to 1e-5 kg/m² over 1,000 days |
| species and element mass of items and fire beds | `crafting.ledger_mass`, `items.element_mass` | below 2e-5 kg per transform; about 1e-5 relative over a whole tick |

One ledger across all modules (for example carbon from plants through bodies to items and back to litter) needs
`world.py` to route the flows. It is open.

**Units and precision.**
- State is float32.
- Ledger sums are float64.
- The climate's gas column `[W, 4]` and ocean store `[W]` are float64. A day's biological exchange is about 1e-6
  of the column, below float32 resolution.

**Determinism.**
- Every module tests that CPU runs are exact and resume exactly.
- On a GPU, `index_add_` and scatter order are not bit-reproducible.
- `torch.multinomial` (ore patch centres) gives other values on a GPU.
- So GPU runs can only agree with the CPU "within noise". Untested so far.

**NumPy.**
- Only `formation` imports NumPy. `chain` uses the standard library, though world7 itself needs NumPy.
- `formation` draws rotation and obliquity with a NumPy generator, `SeedSequence([seed, 0x666F726D])`, not a
  `torch.Generator`. This is allowed by rule 6, because formation is CPU set-up code.

**No supplied meaning.**
- `materials.CLASS` is read only by `senses` (four richness channels) and by `crafting.collect`, to group what
  the brain's four collect outputs reach. No outcome depends on it.
- The only innate wiring is the forage relay (section 2.10).

## 1. Scales (the one compression, stated)

A real planet (radius 5,674 km for planet 781; *(corrected v3: was "about 5,400 km", an Earth-density estimate)*)
cannot be inhabited at the scale of individuals. life9 therefore simulates a **habitat globe**: a sphere of
radius `Config.habitat_radius_m` (default **10 km**).
- Its surface, gravity, air, light, temperature and chemistry are the planet's real derived values.
- Its geography is the planet's pattern in miniature: ocean coverage, latitude and substellar climate, and
  crust composition.
- The climate uses the angular (Budyko-Sellers) energy-balance form. That form does not depend on the absolute
  radius, so the habitat globe has the planet's climate pattern.
- Curvature, the horizon and walking distances are those of the habitat globe.

This compression is labelled `new_rule` and named in every export.

| Quantity | Value |
|---|---|
| tick | 1 day (life-history clock: metabolism, growth, ageing and the climate step once per tick; movement is the daily travel) |
| habitat radius | 10 km (config) |
| cells | cube-sphere `6 x G x G`, G = 48 by default: 13,824 cells of about 300 m *(corrected v3: was 230 m)* |
| individuals per world | capacity 1,024 by default |
| worlds per batch | 4-16 |

### As built

**Grid at G = 48 and R = 10 km:**
- The mean cell size, sqrt(area), is 301.5 m. Cells range from 277 to 327 m.
- Side neighbours have their centres 235 to 327 m apart.
- The largest cell has 1.39 times the area of the smallest.

**The habitat's water** (`globe.habitat_water_volume`, `new_rule`):
- The habitat keeps the real planet's ratio of equivalent ocean depth to relief.
- It is calibrated on Earth. An Earth ocean fills `EARTH_FILL = 0.143` of the habitat's relief shell. On this
  terrain generator that gives Earth's ocean fraction of 0.709 on average.
- Planet 781 gets an ocean fraction of 0.55-0.58 on the sampled terrains.

**What the compression does to individuals.** These follow from the rule. They are not bugs.
- **Short sight.** The real horizon on level ground is 2 sqrt(2 R h): about 126 m between two 1 kg animals and
  223 m between two 30 kg animals. Sight beyond that needs high ground.
- **High relief.** Relief is 600 m x 9.81 / g, which is 780 m for 781. That is 7.8 % of the habitat radius.
- **Long daily travel.** A 30 kg animal may travel up to 24.6 km a day. That is 2 h at its trot-gallop speed,
  0.4 of the habitat's 62.8 km circumference.
  - On 781 the aerobic factor of 0.455 cuts this to about 11 km.
  - The figure is 10 times Garland's (1983) mean daily movement.

**Two quantities keep the real planet's scale.** The weak-temperature-gradient exchange of the climate uses
the real planet's radius and rotation (section 2.6). So the habitat has the real planet's dynamical regime,
not one set by the 10 km radius.

## 2. Modules and their contracts

All modules live in `haishool/life9/planet/`, and the tests in `tests/life9/planet/`.

### 2.1 `constants.py` (reference)
- **Physical constants (CODATA 2018):** `G`, `SIGMA`, `K_B`, `R_GAS`, `N_A`, `H_PLANCK`, `C_LIGHT`.
- **Astronomical values:** `S0 = 1361.0` W/m² (the solar constant, Kopp & Lean 2011), `L_SUN = 3.828e26` W,
  `GM_SUN = 1.3271244e20` and `GM_EARTH = 3.986004e14` m³/s² (IAU 2015 B3), `M_SUN = GM_SUN / G` (1.988410e30),
  `R_SUN = 6.957e8`, `T_SUN = 5772.0`, `AU = 1.495978707e11`, `M_EARTH = GM_EARTH / G` (5.972168e24),
  `R_EARTH = 6.371e6`. The masses come from the GM so that G M is the IAU value exactly.
- **Time units:** `YEAR_S = 31557600.0`, `DAY_S = 86400.0`.
- **Molar masses** in kg/mol, from `data/truth-v5/atomic_weights.json`, through `molar_mass(formula)` (use
  `haishool.truth.formula.parse`).
- **Tests** check the constants against `scipy.constants` where it has them.

#### As built
- The contract is met. `SIGMA` and `R_GAS` are computed exactly from the exact SI constants.
- **Helpers:**
  - `element_mass(symbol)`
  - `atomic_weights()` (g/mol)
  - `solar_number_ratio(a, b)` (Asplund et al. 2009)
- **Earth and material references** used by other modules, each with its citation:

  | Constant | Value |
  |---|---|
  | `G_STANDARD` | 9.80665 m/s² |
  | `P_STANDARD` | 101325 Pa |
  | `EARTH_AIR_MOLE_FRACTION` | N2 0.78084, O2 0.209476, Ar 0.00934 (US Standard Atmosphere 1976); CO2 278 ppm (pre-industrial) |
  | `EARTH_ALBEDO` | 0.30 (world7's) |
  | `Z_SUN` | 0.0134 |
  | `SOLAR_LOG_EPS` | Asplund et al. 2009 |
  | `EARTH_AGE_YR` | 4.5673e9 |
  | `EARTH_CORE_MASS_FRACTION` | 0.3235 (PREM) |
  | `EARTH_OCEAN_MASS_FRACTION` | 2.3e-4 |
  | `RHO_IRON` | 7870 kg/m³ |
  | `RHO_SILICATE` | 3300 kg/m³ |
  | `RHO_WATER` | 1000 kg/m³ |
  | `L_VAPORIZATION` | 2.501e6 J/kg |
  | `EARTH_LAPSE_RATE` | 6.5e-3 K/m |
  | `EARTH_VAPOUR_COLUMN` | 24.6 kg/m² (Trenberth et al. 2005) |
  | `CP_MOLAR_298` | NIST-JANAF |
  | `HALF_LIFE_YR` | NUBASE2016 |
  | `EARTH_RADIOGENIC_SHARE` | per-isotope shares (Turcotte & Schubert 2002) |
  | `EARTH_RADIOGENIC_FLUX` | 0.047 W/m² |

- **Test:** `test_formation.py::test_constants_match_scipy`.
- **Deviations:** none.
- **Open issues.** Two citations were written from memory and are not re-checked: the Turcotte & Schubert
  isotope heat values, and Stacey & Davis 2008. Stacey & Davis is cited here for the 3,300 kg/m³ silicate and in
  formation for K' = 4. The code labels both.

### 2.2 `chain.py` (CPU, NumPy allowed)
`chain_inputs(seed: int) -> dict` runs the world7 chain up to `senses`. It reuses `haishool.life8.bridge`
(`scan_world` or the same prefix) and, where needed, the levels behind it: `stars.cloud` gives the full 10-element
cloud, and `r.levels['planets'].system` gives the raw planet and its sibling planets. It returns JSON-ready values:

```
{"seed", "source": "world7", "reached": rung,
 "star":   {"mass_msun", "luminosity_lsun", "age_yr", "lifetime_yr"},
 "planet": {"mass_earth", "orbit_au", "flux_earth", "era_flux_earth", "t_eq_k", "greenhouse_k", "t_surface_k", "water_state"},
 "cloud":  {"hydrogen","helium","carbon","nitrogen","oxygen","neon","magnesium","silicon","iron","other","metallicity"},  # mass fractions
 "biosphere": {"oxygen_pal", "light", "trophic_levels", ... whatever the bodies/senses eras give},
 "sky": [{"orbit_au","mass_earth","t_k","t_eq_k"}...],
 "provenance": {dotted key: "era/level and file:line"}}
```

`flux_earth` is the unrounded level flux L/a² (it sets the insolation). `era_flux_earth` is the surface era's own
flux (3-digit L over the 3-digit orbit squared, rounded to 3 digits), and `t_eq_k` is the unrounded equilibrium
temperature of that era flux, so `t_eq_k + greenhouse_k` is exactly the value world7 rounds to the whole-K
`t_surface_k` (taking t_eq from the level flux instead missed it by up to 0.03 K and failed the 0.5 K gate for
seed 34). In `sky`, `t_k` is level 3's whole-K temperature including its greenhouse (500 K for a runaway planet;
cloud tops for a gas giant) and `t_eq_k` the bare equilibrium temperature, for world7 and synthetic siblings alike.

`synthetic_inputs(seed)` returns the same structure, drawn from documented ranges and labelled
`"source": "synthetic"`. Results are cached as JSON under `runs/life9/chain-cache/` (git-ignored) so world7 runs
once per seed. A test checks seed 781 against the values in section 3.

#### As built
**API:**

| Function | Does |
|---|---|
| `chain_inputs(seed, cache_dir=None, refresh=False)` | cached world7 inputs (`CHAIN_VERSION = "life9-chain-v2"`) |
| `run_chain(seed)` | the same without the cache, about 1-4 s per seed |
| `synthetic_inputs(seed)` | a drawn planet, `source = "synthetic"`, `SYNTHETIC_VERSION = "life9-synthetic-v2"`; not cached (about 10 ms) |
| `earth_inputs()` | Earth and the Sun in chain form, `source = "earth_reference"`: the calibration case |
| `inputs_for(seed, source)` | dispatch |

**Which planet is chosen.**
- The planet is the one `bridge.planet_of` picks: the most links up to `groups`.
- Without any climb, it is the first temperate terran planet with liquid water.
- A seed with neither raises `NoHabitablePlanet`. World7 seeds 0, 1, 2, 4, 5 and 8 raise it.

**Extra keys beyond the contract:**
- `star`: `end_yr`, `now_yr`, `elapsed_yr`, `frost_line_au`, `hz_inner_au`, `hz_outer_au`, `disc_solids_earth`
  and `disc_gas_earth`.
  - `star.age_yr` is world7's birth epoch, in years after the Big Bang.
  - `star.elapsed_yr` is the star's age when the habitat starts, 8.71e9 yr for 781. Formation uses it.
- `planet`: `id`, `surface`, `gas_earth` and `type`.
- `cloud`: `time_gyr`.
- Top level: `chemistry` (the era plus the molecule counts), `links`, `params`, `planet_index` and
  `truncated_after`.
- `biosphere.oxygen_pal` is the bodies era's `air`.
  - Without a bodies era it is 0 PAL (`new_rule`: no oxygenic life).
  - `biosphere.senses` holds the senses era, or `None`.

**Synthetic planets** (`SYNTHETIC_RANGES`, `new_rule`). The ranges cover world7's temperate terran worlds, not
their frequencies.

| Quantity | Range |
|---|---|
| star mass | log-uniform 0.45-1.2 solar masses |
| cloud formation | 1-10 Gyr |
| star's age at habitat start | 2-9 Gyr |
| metallicity | at least 0.1 Z_sun |
| siblings | 1-5 |
| sibling mass | log-uniform 0.1-20 M_E |
| sibling orbit | log-uniform 0.2-20 times the planet's |

**Deviations:**
- `reached` is the rung of the prefix cut after `senses` (`groups` for 781). The full `world7.run` reaches
  `language`.
- `chain.py` runs its own copy of `scan_world`'s prefix, so that it can keep the level rollouts. The eras equal
  `scan_world`'s; this is checked for seeds 781 and 34.

**Open issues:**
- `chain.py` depends on private APIs: `world7._params`, `world7._Live` and `bridge._climb_to_senses`.
- Bump `CHAIN_VERSION` whenever the output changes. Otherwise stale caches are reused.
- The bodies level's guild biomasses (producers, grazers, predators) are not exported. So the founder diet is a
  proxy (section 2.8).

### 2.3 `formation.py` (CPU): turning the chain into a `PlanetSpec`
`build_planet(inputs: dict, seed: int) -> PlanetSpec` returns a dataclass with these fields:
- `to_dict()`, `provenance: dict[field, (tag, note)]` and `tensors(device) -> dict` (per-world scalars)
- `check_spec(spec) -> list[str]`, which returns the failures

| Field (SI) | Rule |
|---|---|
| `star_mass_kg`, `star_luminosity_w`, `star_age_yr` | chain |
| `star_radius_m` | new_rule: main-sequence R = R_sun M^0.8 (M < 1 solar mass) |
| `star_teff_k` | derived: T_sun (L / R²)^0.25 in solar units |
| `par_fraction` | derived: Planck integral of B_λ(T_eff) over 400-700 nm divided by σT⁴/π (numerical) |
| `orbit_m`, `year_s` | chain orbit; year from Kepler's third law with the star's mass |
| `insolation_w_m2` | derived: S0 x flux_earth |
| `planet_mass_kg` | chain |
| `core_mass_fraction` | derived, new_rule: all Fe goes to the metal core; Mg and Si go to MgO + SiO2 in the mantle and crust; core = Fe / (Fe + MgO + SiO2) by mass (0.233 for 781) |
| `radius_m`, `core_radius_m`, `mean_density_kg_m3` | derived: uncompressed layer densities (iron 7,870, silicate 3,300, reference), then compression scaled to the Earth calibration (Earth's bulk density 5,514 against its uncompressed density, new_rule) |
| `gravity_m_s2`, `escape_velocity_m_s` | derived: g = GM/R², v = sqrt(2GM/R) (the single radius rule used everywhere) |
| `ocean_mass_kg` | new_rule: volatile delivery. The chain's planet accreted 0 ice, yet world7 says water is `liquid`. Water mass fraction = 2.3e-4 x (Z/Z_sun)^0 by default (the Earth value; config `water_mass_fraction`). It is labelled new_rule, and the choice is reported. |
| `atmosphere_mass_kg`, `partial_pressure_pa{N2,O2,CO2,Ar,H2O}`, `surface_pressure_pa` | new_rule plus chain plus derived. N2 and Ar are outgassed in proportion to Earth's column per unit g (P_N2 = 78 kPa x (g/g_E)^0 x the volatile scale, stated; Earth's dry-air mole fractions x 101325 Pa). O2 = chain `oxygen_pal` x 21.2 kPa (PAL means present atmospheric level). CO2 comes from the greenhouse calibration below. H2O at the surface = 0.77 x e_sat(T_s) (Magnus). P = Σp is the surface air (μ, cp, γ, ρ, c use it). Water vapour is not well mixed, so M_atm = (P - p_H2O) 4πR² / g + `vapour_column_kg_m2` x 4πR²; the vapour comes out of the planet's water and the rest is the ocean. |
| `vapour_scale_height_m`, `vapour_column_kg_m2` | derived plus reference plus new_rule: H_w = R_v T_s² / (L_v Γ) (Clausius-Clapeyron at constant relative humidity), Γ = 6.5 K/km (US Std. Atm. 1976) x g/g_E (new_rule); column = p_H2O μ_w / (R T_s) x H_w. Earth: 2.3 km and 22 kg/m², observed 24.6 kg/m² (Trenberth et al. 2005); 781: 3.05 km and 29.2 kg/m². climate.py starts `vapour` from this column. |
| `air_molar_mass_kg_mol`, `air_cp_j_kg_k`, `air_gamma`, `scale_height_m`, `air_density_kg_m3`, `sound_speed_m_s` | derived: mixture rules; H = RT/(μg); ρ = Pμ/(RT); c = sqrt(γRT/μ) |
| `albedo` | reference: 0.30 (the albedo built into world7's T_eq) |
| `t_eq_k` | derived: (S(1 - A)/(4σ))^0.25, which must match the chain's t_eq within 1 K |
| `t_surface_target_k` | chain: t_eq (of the era flux) + greenhouse, exactly the value world7 rounds to its whole-K T_s |
| `greenhouse_tau` | derived: a grey atmosphere, T_s = T_eq (1 + 0.75τ)^0.25 solved for τ (about 1.13 for 781) |
| `co2_ref_pa` | new_rule: the CO2 partial pressure at which τ is calibrated; forcing ΔF = 5.35 ln(p/p_ref) W/m² (Myhre 1998) changes τ. The forcing that separates τ from Earth's is read as CO2 around Earth's 278 ppm (5130 Pa for 781). |
| `co2_capped`, `tau_residual` | new_rule **CO2 cap**: the log law is a fit near present concentrations (Byrne & Goldblatt 2014), so `co2_ref_pa` is at most 1e4 Pa (`FormationRules.co2_max_pa`). The cap is the common case, not an edge case: 216 of 300 synthetic planets and 21 of the 26 habitable world7 planets of seeds 12-59 hit it (781 does not). For them `co2_capped` is true, `co2_ref_pa` is the cap, not a calibration, and `tau_residual` > 0 is the part of τ of no named gas that still holds the chain's T_s (0 when not capped). Consumers must not read a capped `co2_ref_pa` as calibrated; for example, the biosphere's f_CO2 is 2.07 at the cap. |
| `tau_per_ln_co2` | derived: dτ/d ln p_CO2 such that the grey forcing is 5.35 W/m² per e-fold at the calibration point (0.0625 for 781). At fixed T_eq it gives a Planck-only response of about 1.27 K per CO2 doubling for 781, against Earth's about 3 K with water-vapour and other feedbacks (IPCC AR6 likely range 2.5-4 K). |
| `tidally_locked`, `rotation_period_s`, `obliquity_rad`, `day_length_s` | derived and new_rule: the lock radius (Kasting et al. 1993: a_lock = 0.027 (P0 t / Q)^(1/6) M^(1/3) au with P0 = 13.5 h, Q = 100 and t the star's age) decides `tidally_locked`. If locked, rotation = year, obliquity 0 and the day is infinite (permanent day side). If not, rotation period and obliquity are drawn on the formation RNG from documented ranges (new_rule). |
| `relief_m` | new_rule: habitat relief = 600 m x (9.81/g) (relief limited by rock strength scales as 1/g) |
| `crust` | derived plus reference, dict species to mass fraction (see `materials.CRUST_SPECIES`). Bulk silicate ratios (Mg/Si, Fe/Si, C/O) come from the cloud. Trace ore elements (Cu, Sn, Ca, Na, K) are Earth crust abundances (reference) scaled by Z / Z_sun (new_rule, Z_sun = 0.0134). |
| `radiogenic_heat_w_m2` | reference plus new_rule: Earth's 0.047 W/m² radiogenic flux scaled by Z/Z_sun and by the decay since formation (half-lives of K-40, U-238, U-235, Th-232) |

`check_spec`:
- the layer masses sum to the planet mass within 1e-6
- g equals GM/R²
- T_eq matches the chain within 1 K
- the grey τ reproduces the chain's T_s within 0.5 K
- every field has a provenance tag
- M_atm is the dry air plus the vapour column, and ocean + vapour is the planet's water
- `co2_capped` agrees with `tau_residual` > 0, and the CO2 part of τ plus `tau_residual` is τ
- the crust keys are `materials.CRUST_SPECIES` in that order

`to_dict()` is strict JSON: a non-finite number (the infinite day of a locked planet) becomes `null`, and
`tidally_locked` says why. `tensors()` keeps the infinity.

#### As built
The table above was brought up to date in version 2 and matches the code. The details:

**Formulas beyond the table:**
- **Star radius.** R = R_sun M^0.8 below one solar mass and M^0.57 above. The second law is `new_rule`; it is
  needed because synthetic stars reach 1.2 solar masses. The two laws meet at M = 1.
- **Compression.** c = (1 + K' x)^(1/K'), from Murnaghan's equation of state.
  - x = x_E (M/M_E)^(2/3) (ρ0 c / (ρ0_E c_E))^(4/3) is the hydrostatic scaling P ~ G M^(2/3) ρ^(4/3).
  - K' = 4 (`reference`).
  - x_E is set so that Earth gets its own c_E: its solid density, ocean removed, over its uncompressed density.
  - It matches Venus and Mars within 2-3 %. It matches Zeng et al. 2016, stated for 1-8 M_E. Planet 781, at
    0.61 M_E, is an extrapolation.
  - The core takes the same factor.
- **N2 and Ar.** Earth's dry mole fractions x 101325 Pa: N2 is 79.12 kPa, not the rounded 78 kPa in the table.
- **CO2 calibration.**
  - First, the forcing that separates the planet's τ from Earth's, both at the planet's T_s:
    dF = σ T_s⁴ [1/(1 + 0.75 τ_E) − 1/(1 + 0.75 τ)].
  - Then p_CO2 = 28.17 Pa x exp(dF / 5.35).
  - τ_E = 0.838 is world7's Earth, 287.6 K at flux 1.
- **CO2 slope.** `tau_per_ln_co2` = 5.35 (1 + 0.75 τ)² / (0.75 σ T_s⁴).
- **Radiogenic heat.** Earth's 0.047 W/m² x Z/Z_sun x the decay factor x silicate mass per area relative to Earth.
  - For 781: 0.227 x 0.583 x 0.877 x 0.047 = 0.0054 W/m².
  - The size factor is beyond the contract. A fixed heat production per kg gives a flux proportional to
    M_mantle / R².
- **Lock radius.** It uses the star's real age at habitat start (8.71 Gyr for 781), giving 0.747 au.
- **Rotation of planets that are not locked.** The period is log-uniform in 12-48 h. The obliquity is uniform in
  0-35°.

**`FormationRules`** (all `new_rule` unless marked):

| Rule | Default |
|---|---|
| `water_mass_fraction` | 2.3e-4 |
| `water_metal_exponent` | 0 |
| `volatile_scale` | 1 |
| `volatile_g_exponent` | 0 |
| `relative_humidity` | 0.77 |
| `lapse_rate_k_m` | 6.5e-3 x g/g_E |
| `relief_m` at `relief_g` | 600 m at 9.81 m/s² |
| `lock_p0_h`, `lock_q` | 13.5 h, 100 (reference: Kasting et al. 1993) |
| `rotation_h` | 12-48 h |
| `obliquity_deg` | 0-35° |
| `par_nm` | 400-700 nm (reference: McCree 1972) |
| `ms_radius_exponents` | 0.8 and 0.57 |
| `k_prime` | 4 (reference) |
| `co2_max_pa` | 1e4 Pa |

**`check_spec` results.** It passes for:
- 781
- the Earth twin
- synthetic seeds 0-39 (tests) and 0-299 (review)
- world7 seeds 3, 6, 7, 9, 10 and 11
- all 26 habitable world7 seeds in 12-59

The tightest margin is seed 34: 0.013 K inside the 0.5 K gate.

**The Earth twin** (`earth_inputs`):

| Quantity | Value |
|---|---|
| radius | 6,354 km |
| g | 9.874 m/s² |
| CO2 | 28.17 Pa |
| vapour column | 22.2 kg/m² (observed 24.6) |
| P | 102.58 kPa = 101,318 Pa dry + 1,265 Pa H2O |
| tidally locked | no |

**Deviations from the original contract, with reasons:**

| Deviation | Reason |
|---|---|
| `star_age_yr` is the star's age at habitat start, not its birth epoch. | The lock radius and the radiogenic decay need the age. |
| The vapour column is separate from the well-mixed air. | The first version counted vapour as well mixed, about 5-7 times too much water in M_atm. |
| The Earth twin's dry partial pressures are not scaled down for its H2O. | The US Standard Atmosphere is dry air, and 278 ppm is a dry mole fraction. Scaling would also move 781's O2 off the section 3 value. So P differs from g M_atm / A by about 1 %. |
| `chain_t_eq_k` comes from the era flux. | The level flux missed world7's rounding by up to 0.03 K and failed seed 34. |
| No local crust fallback. | `crust_fractions` always calls `materials.crust_fractions` and raises if the species order differs. A silent fallback had a 13-species mix that no other module indexes by. |
| There are extra fields. | Fields beyond the table: `star_birth_yr`, `star_lifetime_yr`, `flux_earth`, `metallicity`, `water_mass_fraction`, `core_mass_kg`, `mantle_mass_kg`, `uncompressed_density_kg_m3`, `compression`, `ocean_volume_m3`, `ocean_layer_m`, `oxygen_pal`, `oxygen_mole_fraction`, `chain_t_eq_k`, `chain_t_surface_k`, `lock_radius_m`. |
| Helpers are added. | `stack_specs` gives a `[W]` tensor per scalar and `crust [W, S]`. `describe` prints one line per field. |

**Open issues:**
- `world.summary()` should report the share of CO2-capped worlds.
- Four citations were written from memory and are not re-checked. They are labelled in the code:
  - Byrne & Goldblatt 2014 (where the log law departs)
  - Dai 2006 (relative humidity)
  - Turcotte & Schubert (isotope heat values)
  - Stacey & Davis (K' = 4)
- A cold-cache run of `test_formation.py` takes 28 s, because it runs world7 for 781 and 6 seeds. A warm cache
  takes 12 s.

### 2.4 `materials.py` (reference data plus physics, torch helpers)
- `SPECIES`, an ordered tuple of about 30 entries:
  - rocks and minerals: basalt, granite, flint, sandstone, limestone, clay, sand, salt, hematite, magnetite,
    malachite, cassiterite, native_copper
  - organics: wood, plant_fiber, resin, meat, fat, bone, hide, charcoal, ash
  - products: ceramic, lime, copper, tin, bronze, iron, steel, glass
- `CRUST_SPECIES` is the subset that occurs in ground deposits.
- `CLASS[species]` is one of stone, ore, clay, wood, organic, metal, product. It is used only for senses and
  stats, never for rules.
- Per-species property tables, each row with a source string. Values are real reference values:
  - `DENSITY` kg/m³
  - `MOHS` hardness
  - `TOUGHNESS` MPa·m^0.5 (fracture toughness)
  - `BRITTLE` (0 or 1, conchoidal fracture possible)
  - `MELT_K`
  - `CP` J/(kg K)
  - `CONDUCTIVITY` W/(m K)
  - `COMBUSTION_J_KG` (0 if the species is not a fuel)
  - `IGNITION_K`
  - `FOOD_J_KG` (gross energy) and `DIGESTIBLE` per diet type (plant or meat)
  - `COOK_GAIN` (cooking raises digestible energy; meat about 1.3)
- `THERMO`, NIST-JANAF ΔH_f and S° for the oxides, carbonates and gases the transforms need.
  `equilibrium_temperature(reaction)` solves ΔG = ΔH - TΔS = 0.
- `TRANSFORMS`, a list of physically conditioned transformations. Each has:
  - inputs and outputs in kg per kg, balanced by elements (tested with `haishool.truth.reactions` style
    balancing on the formulas)
  - `min_temp_k = max(ΔG = 0 temperature, kinetic or practical onset from the archaeometallurgy reference)`
  - `atmosphere`: `reducing` (charcoal present in the same fire), `smothered` (see below) or `any`
  - `enthalpy_j_kg`, `days`, `source`

  They include:
  - clay to ceramic (sintering at 1,150 K or more)
  - limestone to lime + CO2 (ΔG = 0 at about 1,118 K)
  - wood to charcoal (above 600 K, smothered in a fire bed) *(corrected v3: was "with reducing conditions",
    which needs charcoal before the first charcoal exists)*
  - malachite to copper (reducing, 1,000 K or more)
  - cassiterite to tin (reducing, 1,100 K or more)
  - hematite and magnetite to bloomery iron (reducing, 1,400 K or more)
  - copper + tin (molten) to bronze
  - iron + charcoal to steel (carburising, 1,200 K or more)
  - sand + wood ash to glass (1,500 K or more) *(corrected v3: was "sand + ash or lime, 1,400 K"; sand + lime
    does not melt below the CaO-SiO2 eutectic, about 1,709 K)*

  Fire's own chemistry, wood or fat or charcoal + O2 to CO2 + H2O + heat, is in crafting, using
  `COMBUSTION_J_KG`.
- `props(comp: [..., S] mass fractions, mass) -> dict` (torch, batched) gives hardness, density, toughness,
  melting temperature, cp, fuel energy, food energy and brittleness:
  - hardness is mass-weighted, with the hardest continuous phase capped
  - an alloy uses its own row
  - melting is the minimum over the species present above 10 %
- `crust_fractions(inputs, spec) -> dict` converts the chain's bulk composition into the deposit mix used by
  `formation`.

#### As built
**Species.** There are 31 species. `pyrite` (FeS2) was added; the spark ignition in 2.11 needs it.
- `CRUST_SPECIES` is the first 14: basalt, granite, flint, sandstone, limestone, clay, sand, salt, hematite,
  magnetite, malachite, cassiterite, native_copper, pyrite.
- Always index species by name, through `IDX`.

**Tables.** All the contract tables exist, plus `GRANULAR` (sand 1 and ash 1, everything else 0).
- Each entry is a `Val` carrying its tag and source.
- `NEVER_K = 5000 K` is a `new_rule` sentinel for "does not melt or ignite below any fire". Examples: wood melt,
  and charcoal, which sublimes at 4,098 K.
- `MAKEUP` gives every species' element composition. `ELEMENTS` has 16 symbols: H C N O Na Mg Al Si P S Cl K Ca
  Fe Cu Sn. `element_matrix()` is `[S, E]`.

**`props()` outputs:**
- `mass`, `volume_m3`, `density` (volume-additive)
- `hardness`: the mass-weighted Mohs value, capped at the hardest continuous phase. A phase is continuous at a
  mass fraction of 0.3116 or more (the site percolation threshold), or if it is the dominant phase.
- `granular` and `tool_hardness = hardness x (1 − granular)`
- `toughness`, `brittle`, `melt_k`, `cp`, `heat_capacity_j_k`, `conductivity`
- `fuel_j`, `ignition_k`
- `food_j`, `food_plant_j`, `food_meat_j`, `cook_gain`

Negative or non-finite fractions and masses count as 0, so every output is finite and non-negative.

**Transforms as built** (kg per kg of the first input; the gases are kept apart from the species):

| Name | Atmosphere | min_temp_k | days | Products (per kg basis) | Enthalpy (MJ/kg, + absorbs) |
|---|---|---|---|---|---|
| clay_to_ceramic | any | 1150 | 0.5 | ceramic 0.860, H2O 0.140 | +0.58 |
| limestone_to_lime | any | 1118.5 (ΔG = 0) | 1 | lime 0.560, CO2 0.440 | +1.79 |
| wood_to_charcoal | smothered | 600 | 1 | charcoal 0.246, ash 0.01; takes 0.709 O2; gives 0.553 H2O, 0.901 CO2 | −10.5 |
| malachite_to_copper | reducing | 1000 | 0.25 | copper 0.575; takes 0.109 charcoal; gives CO2, H2O, CO | +0.88 |
| cassiterite_to_tin | reducing | 1100 | 0.25 | tin 0.788; takes 0.159 charcoal; gives CO | +2.37 |
| hematite_to_iron | reducing | 1400 | 0.5 | iron 0.699; takes 0.226 charcoal; gives CO | +3.09 |
| magnetite_to_iron | reducing | 1400 | 0.5 | iron 0.724; takes 0.208 charcoal; gives CO | +2.92 |
| copper_tin_to_bronze | any | 1357.8 | 0.1 | bronze 1.124 from copper 1 + tin 0.124 | 0 |
| native_copper_to_copper | any | 1357.8 | 0.1 | copper 1.0 | 0 |
| iron_to_steel | reducing | 1200 | 2 | steel 1.008; takes 0.008 charcoal | 0 |
| sand_ash_to_glass | any | 1500 | 1 | glass 1.591 from sand 1 + ash 1; gives CO2 0.409 | +0.84 |

**Thermochemistry checks.** Equilibrium temperatures from `THERMO`:

| Reaction | ΔG = 0 at |
|---|---|
| CaCO3 → CaO + CO2 | 1,118.5 K |
| Fe2O3 + 3C | 907.5 K |
| Fe3O4 + 4C | 961.9 K |
| SnO2 + 2C | 923.8 K |
| malachite + 2C | 292.5 K |

- `_from_reaction` uses ΔG = 0 as a lower bound only when ΔH > 0 and ΔS > 0. It raises for reactions favoured
  only below their ΔG = 0 temperature, or never.
- Element closure through the float32 tensors is below 1e-6.

**Fire chemistry.** `burn(species)` takes fuels only (`FUELS`) and raises for anything else. A carbonate-ash book
of meat would create Ca and K and destroy N.
- `FUELS` = wood, plant_fiber, resin, fat, charcoal.
- Lower heating values: wood 18.6, plant_fiber 16.4, resin 36.2, fat 37.0, charcoal 32.8 MJ/kg.
- Ignition: wood 573 K, fibre 533 K, resin 623 K, fat 616 K, charcoal 623 K.
- `gas_to_air(t)` and `transform_tensors()["air_out"]` give each transform's net exchange with the air. The CO
  from a smelt is burnt to CO2 above the bed, because the climate ledger has no CO.

**Atmospheres** (`fire_atmosphere`, `atmosphere_ok`):
- `reducing`: there is charcoal in the fire bed or mixed into the item.
- `smothered`: the fire has no forced air, and its bed of wood plus charcoal is at least `SMOTHER_COVER = 1.0`
  times the item's mass.
- **Precedence.** A transform whose conditions hold beats ignition. A fuel item above its `IGNITION_K` whose own
  transform does not apply burns as fuel.

**Crust mix** (`crust_fractions`, `CRUST_RULE`):
- It starts from Earth's surface lithology shares, `EARTH_SHARE` (`new_rule`):

  | Rock | Share |
  |---|---|
  | granite | 0.25 |
  | sandstone | 0.15 |
  | limestone | 0.13 |
  | sand | 0.10 |
  | clay | 0.08 |
  | basalt | 0.07 |
  | flint | 0.01 |
  | salt | 0.01 |

- The upper continental crust supplies Fe, Cu, Sn and S (Rudnick & Gao 2003).
- These are scaled by the cloud's ratios relative to solar:
  - basalt x Mg/Si
  - felsic and silica rocks ÷ Mg/Si
  - iron ores x Fe/Si
  - limestone x C/O x other/Si
  - salt and pyrite x other/Si
  - Cu and Sn ores x other/Si x Z/Z_sun
- Iron ore is 10 % of crustal Fe, split 2:1 between hematite and magnetite.
- Crustal Cu is split 80 % malachite, 20 % native copper.

For 781:

| Species | Mass fraction |
|---|---|
| granite | 0.313 |
| sandstone | 0.188 |
| sand | 0.125 |
| basalt | 0.119 |
| clay | 0.116 |
| limestone | 0.109 |
| flint | 0.0125 |
| salt | 0.0113 |
| hematite | 3.6e-3 |
| magnetite | 1.7e-3 |
| pyrite | 1.3e-3 |
| malachite | 1.0e-5 |
| native copper | 1.4e-6 |
| cassiterite | 6.9e-7 |

**Deviations, with reasons:**
- **Pyrite** was added (see above).
- **wood_to_charcoal is `smothered`, not `reducing`.** Charring needs restricted air, not a reductant, and the
  contract's rule needed charcoal before the first charcoal exists. Folding "no forced air" into `reducing` would
  have broken bloomery smelting with bellows.
- **Charring is modelled as partial combustion:** 4 C6H9O4 + 13 O2 → 12 C + 18 H2O + 12 CO2. Dry wood has too
  little oxygen to turn all its hydrogen into water, so pure pyrolysis cannot balance with the ledger gases. The
  charring enthalpy is derived from the wood's heating value by Hess's law.
- **Glass is sand + wood ash only**, 1 kg of ash per kg of sand. The melt is SiO2 62.9 %, CaO 26.4 %, K2O 10.7 %.
  The onset was raised from the contract's 1,400 K to 1,500 K for this lime-rich melt.
- **Ca, Na, K and S scale with the cloud's other/Si, not Z/Z_sun.** These elements sit in the chain's "other"
  bin, and a lower Z makes less rock, not rock poorer in Ca. Cu and Sn keep the contract's Z/Z_sun factor on top.
- **native_copper_to_copper was added**, so the native ore can become the copper product.
- **Alloys.** Bronze is 89 % Cu / 11 % Sn (the Incropera table row). Steel is 0.8 % C.
- **Basalt Mohs is 6.5**, the top of its 6-7 range. It is tagged `new_rule`: it was chosen so that basalt x 1.1
  ≥ flint (7.0), which the knapping gate needs.
- **Loose grains.** Sand keeps its grain hardness of 7 for abrasion. Tools use `tool_hardness`, so a sand pile
  is no hammer. Clay is cohesive (`GRANULAR` 0).

**`new_rule` choices** (beyond the deviations):

| Choice | Value |
|---|---|
| charcoal | pure C |
| wood | C6H9O4 with 1 % ash |
| ash | 75 % CaCO3 + 25 % K2CO3 |
| melting and ignition | count only species above 10 % |
| `SMOTHER_COVER` | 1.0 |
| crafting's CO | burns to CO2 above the bed |
| mixing enthalpies of bronze and steel | neglected |
| K2CO3 + SiO2 enthalpy | from the soda analogue |

Digestibility for (plant-eater, meat-eater):

| Item | Plant-eater | Meat-eater |
|---|---|---|
| meat | 0.4 | 0.9 |
| fat | 0.4 | 0.9 |
| bone | 0 | 0.4 |
| hide | 0 | 0.3 |
| plant_fiber | 0.3 | 0 |

**Open issues:**
- **Values tagged as estimates** (no reference value was found):
  - toughness of clay, salt, the ore minerals, tin, soft tissues, charcoal and lime
  - Mohs hardness of wood, organics, sandstone, bronze, steel and lime
  - conductivity of several species
  - cp of resin and fibre
  - ignition of resin and fat
- **The 1,500 K glass onset** has no liquidus citation. A K2O-CaO-SiO2 liquidus check (Morey, Kracek & Bowen
  1930) would make it firm.
- **The kaolinite and mullite rows are approximate.** They set only the clay-firing enthalpy, and the 1,150 K
  onset overrides their 462 K ΔG = 0 temperature.
- **Ores are physically rare.** For 781, malachite is 1.0e-5 and cassiterite 6.9e-7 of the deposit mass. Only
  the patch concentration in `globe` makes them findable. Bronze is practically out of reach on low-Z worlds.
- **No tissue pyrolysis.** There is no materials transform for meat, hide and bone in a fire. `crafting` chars
  them to the soil instead (section 2.11).
- **CUDA.** The CUDA comparison tests skip on a CPU host. They should run on the cluster.
- **`state_hash`** hashes native-endian bytes. All target hosts are little-endian.

**`items.py`** holds the pools of section 2.11 (data only):
- `ItemPool`, `FirePool`
- `spawn`, `remove`, `spawn_fire`, `remove_fire`
- `species_mass`, `element_mass`
- `holder_code` (n K + k)
- `state_dict`, `from_state`, `state_hash` (SHA-256 of row-major bytes; 7 ms for a `[16, 4096]` pool)

How `spawn` behaves:
- It fills the lowest free slots in request order.
- It returns −1 for a full world or an invalid request: non-finite or empty `comp`, mass ≤ 0, or a negative
  temperature. An invalid request takes no slot, so the item ledger cannot leak.

### 2.5 `globe.py` (torch)
- `Globe(G, device)` holds:
  - `centers [C,3]` (unit vectors), `area [C]` (steradians, summing to 4π), `lat [C]`, `lon [C]`
  - `east [C,3]` and `north [C,3]` (the local tangent basis)
  - `nbr [C,8]` (neighbour indices across the cube faces)
  - `cell_of(p [...,3]) -> long [...]`: an exact face projection, vectorised
- `move(p, heading, dist_m, radius_m) -> (p2, heading2)` moves along a great circle, keeping the heading relative
  to local north.
- `angle(p, q)` gives the geodesic angle.
- `bearing(p, q, heading) -> rad`: the direction of q seen from p, relative to p's heading.
- `make_terrain(globe, W, gen, relief_m [W], water_volume_m3 [W], radius_m) -> dict` returns:
  - `elevation_m [W,C]`: spherical noise from random great-circle faults plus smooth random plane waves on the
    sphere, scaled to the relief
  - `sea_level_m [W]`: solved by bisection so that the ocean volume equals the water volume, scaled to the
    habitat (the ocean fraction of the real planet's water budget is kept)
  - `depth_m [W,C]` and `land [W,C]`
- `make_deposits(globe, W, gen, terrain, crust: [W,S] fractions) -> stock_kg_m2 [W,C,S]`: accessible ground stock
  per species.
  - Felsic rocks favour high ground and mafic rocks low ground.
  - Ores come in patches; sand lies on coasts; salt lies in dry low cells.
  - Clay starts small, and weathering adds it later.

#### As built
**Grid.** An equiangular gnomonic cube-sphere (Ronchi et al. 1996). Cell index c = f G² + i G + j.
- The geometry is built in float64 on the CPU and stored in float32 on the device, so every device sees the same
  grid.
- Cell sides are great-circle arcs, and areas are exact.
- `area64` keeps the float64 areas.
- `Globe(48)` builds in about 0.44 s.

**Extra attributes:**
- `nbr_valid`
- `nbr4`
- `corners`, `vertices`
- `spacing`, `min_spacing`
- `lap_radius`

**Neighbours.** The 24 cells at the 8 cube corners have only 7 distinct neighbours. Their missing diagonal slot
holds the cell itself, and `nbr_valid` marks it.

**`cell_of`** returns the gnomonic quad that contains the point, not the Voronoi-nearest centre. The two agree
in about 95 % of cases. Otherwise the assigned centre is at most 0.28 cell widths farther away.

**Conventions:**
- A heading is a compass angle from local north, clockwise seen from outside: d = cos(h) north + sin(h) east.
- At a pole the basis follows lon = atan2(y, x).
- `move` returns headings in [0, 2π). `bearing` returns [−π, π).
- `slerp` is safe for antipodal points.

**Operators:**
- **`laplacian`.** The unit-sphere Laplacian as a diamond finite-volume scheme (Coudière, Vila & Villedieu 1999):
  conservative and stable.
  - Its low eigenvalues converge at second order.
  - It is supraconvergent: diffusion converges at second order, but the pointwise error on face-rim cells does
    not shrink (about 0.6 for Y_2^0).
  - Use it inside `diffuse` or in divergence form, not as a pointwise curvature.
  - The plain two-point flux was rejected, because its error grows with G on this non-orthogonal grid.
- **`diffuse(field, kappa, steps=None)`.** Explicit sub-steps sized by a power-iteration spectral radius,
  accumulated in float64.
  - It conserves sum(area f / κ), so a per-cell κ = D dt / C works.
  - A 1-day land step takes 56 sub-steps at G = 48. The count grows as G².
  - It raises if `steps` is too small to be stable.
- **Others:**
  - `smooth(field, length_rad)` is independent of G.
  - `hops(source, max_hops)`.
  - `distance(source, max_rad)`: the geodesic distance by vector propagation (Danielsson 1980).
  - `at_vertices(field)`.

**Terrain** (`make_terrain`):
- Elevation is 0.5 x 128 random great-circle faults (tanh scarp, width 0.05 rad) plus 0.5 x 64 plane waves, each
  at unit standard deviation.
  - The waves have k log-uniform in 1-24 rad⁻¹ and amplitude k^−0.5, giving a degree variance proportional to
    l⁻² (Turcotte 1987).
  - The result is rescaled to [0, relief_m]. The datum 0 is the lowest ground, at the habitat radius.
- **Sea level.** `sea_level()` is a float64 bisection on the exact spherical-shell volume, 80 steps. The
  curvature term is 6 % at 600 m on a 10 km sphere.
- **Returned keys:** `elevation_m`, `sea_level_m`, `depth_m`, `land`, `coast` (land with an ocean neighbour),
  `ocean_fraction`, `ocean_volume_m3` (float64) and `inland_rad` (the distance to the coast, independent of G).
- **Water budget.** `habitat_water_volume(...)` turns the real planet's ocean into the habitat volume
  (section 1).

**Deposits** (`make_deposits(..., species, accessible_kg_m2=2670, clay_start=0.1, report=None)`):
- **The land carries the crust mix.**
  - Each land cell holds a column of accessible_kg_m2 x sum(crust): 1 m of rock at the 2,670 kg/m³ Bouguer
    density. Clay counts at a tenth of its share.
  - The column is split among the species by `LAND_WEIGHTS` x log-normal patchiness (ln std 0.5), with float64
    Sinkhorn matrix scaling.
  - The land-area mean of each species is accessible_kg_m2 x crust[s], with an inventory error of 1e-10 or less.
- **The sea floor** holds only basalt, limestone, clay and sand.
  - Their shares are renormalised among themselves, with `MARINE_WEIGHTS`.
  - A shelf weight exp(−depth / (0.05 x relief)) favours the shallows.
  - The sea floor is kept out of the land inventory.
- **Patterns by kind** (`LAND_WEIGHTS`):
  - felsic on high ground; mafic, sediment, carbonate and clay on low ground
  - sand within exp(−d / 0.05 rad) of the coast, plus an inland floor of 0.05
  - salt only on the share of a cell at least 0.08 rad inland
  - ores in 12 patches per kind, radius 0.02-0.06 rad, centred on land
    - each patch is normalised to carry π r² at every G
    - copper favours uplands, tin follows granite and magnetite favours low ground
- **Support widening.** When a species has too little area (less than 2 times its land share), its support widens
  to low land, then to all land. With no land at all, the species is absent and a `RuntimeWarning` is raised.
  `report` receives the support levels and the inventory errors.
- **Speed.** Deposits at G = 48, W = 16 take about 0.3 s.

**Deviations:**
- The Laplacian scheme, `cell_of` and the corner neighbours (see above).
- `make_deposits` takes a `species` argument.
- The land inventory is normalised over the land area, not the whole sphere. With the old rule, land columns were
  about 1.9 times too heavy and the land mix was too felsic. As a result, ore per land cell is about half what
  the first version gave.
- Deposit rules are written in distances (rad), not cell hops. The hop rules changed with G.

**`new_rule` list.** Every one is in `PROVENANCE` as `NAME = value`:

| Group | Constants |
|---|---|
| terrain | `FAULTS` 128, `FAULT_WIDTH_RAD` 0.05, `WAVES` 64, `WAVE_K` (1, 24), `FAULT_SHARE` 0.5 |
| water | `EARTH_FILL` 0.143 (calibrated) |
| deposits | `ACCESSIBLE_DEPTH_M` 1, `CLAY_START` 0.1, `ORE_PATCHES` 12, `ORE_RADIUS_RAD` (0.02, 0.06), `DEPOSIT_NOISE` 0.5, `NOISE_WAVES` 32, `NOISE_K` (2, 24) |
| coast, salt and support | `SHELF_SCALE` 0.05, `SAND_SCALE_RAD` 0.05, `SAND_FLOOR` 0.05, `SALT_MIN_DIST_RAD` 0.08, `SUPPORT_MARGIN` 2 |
| weights | `LAND_WEIGHTS`, `MARINE_WEIGHTS`, `DEPOSIT_KIND` |

**Open issues:**
- **Salt is a placeholder.** It goes to low inland ground, because aridity from the climate does not exist when
  deposits are made.
- **A world with no land has no ore.** Ore patches are centred on land.
- **`coast` depends on G.** It is a one-cell band. Rules that must not depend on G use `inland_rad`.
- **Host syncs.** `diffuse` calls `float(kappa.max())` and `float(kappa.min())` even when `steps` is given. That
  is two host syncs per call on a GPU.
- **GPU randomness.** `torch.multinomial` and the random draws give different values on a GPU.
- **Shelf share.** The shelf covers 9-12 % of the ocean, against Earth's continental shelf of about 9 %. That
  figure (Harris et al. 2014) was quoted from memory.

### 2.6 `climate.py` (torch)
State: `T [W,C]` (K, surface air), `vapour [W,C]` (kg/m², column water), `soil [W,C]` (kg/m², bucket of 150 mm
at most), `snow [W,C]`, and the global atmosphere as moles per m² of column `gas [W,4]` (N2, O2, CO2, Ar), with
partial pressures p_i = n_i g μ_i (n_i in mol/m², μ_i the molar mass) *(corrected v3: wording)*.
- **Insolation** `insolation(globe, P, day) -> [W,C]` W/m², the daily mean.
  - A locked planet gets S max(0, cos θ_substellar).
  - A rotating planet gets the standard daily-mean formula from latitude and the declination given by
    obliquity and orbital phase.
- **Energy balance (Budyko-Sellers form):** C dT/dt = I(1 - α) - σT⁴/(1 + 0.75τ) + D ∇²_angular T.
  - τ = τ0 + k ln(pCO2/p_ref), with k chosen so that ΔF = 5.35 ln matches at the calibration point.
    This grey slope alone is a Planck-only response, about 1.27 K per CO2 doubling for 781 (Earth's is about
    3 K with feedbacks). Add an H2O term, for example τ depending on e_sat(T), where realistic sensitivity
    matters. When `co2_capped` is true, `p_ref` is the cap and `tau_residual` is part of τ0.
  - `vapour` starts from `PlanetSpec.vapour_column_kg_m2`.
  - D = 0.55 W m⁻² K⁻¹ x (P / 1 bar) (heat transport scales with the atmosphere's mass, new_rule).
  - C is 2.1e8 J m⁻² K⁻¹ for ocean (a 50 m mixed layer) and 1e7 for land.
  - The scheme is implicit or sub-stepped so that it is stable at a 1-day step (tested).
  - Elevation lapse: Γ = g / cp.
  - Albedo per cell: ocean 0.06, ice and snow 0.6 (below 263 K), bare land 0.25, vegetation 0.15. The global
    planetary value is calibrated with a cloud term so that a planet without life reaches the chain's T_s
    within 3 K (the gate).
- **Water.** Evaporation is proportional to e_sat(T) (Clausius-Clapeyron, Magnus form) where there is water.
  Vapour diffuses. Precipitation happens where vapour is above saturation, with a background rate. The soil
  bucket fills from rain, and the overflow runs off back to the ocean. Snow forms below 273 K. The water ledger
  closes.
- **Carbon-silicate cycle** (Walker, Hays & Kasting 1981):
  - Weathering F_w = F0 exp((T - 288)/13.7) (runoff/runoff0)^0.65 removes CO2.
  - Volcanic outgassing F_v follows from `radiogenic_heat_w_m2` (new_rule proportionality).
  - Both are tiny at sim time scales, and they are reported.
- `step(state, ...)` returns diagnostics: precipitation, evaporation, runoff and pressures.

#### As built
**State** (`ClimateState`):
- `T`, `vapour`, `soil`, `snow` `[W, C]` (float32)
- `gas [W, 4]` (float64)
- `ocean [W]`: the ocean as a global-mean column, float64 kg/m²
- `cloud_albedo [W]`
- `runoff_mean [W, C]`
- ledger counters (`water0`, `water_external`, `gas_external`, `co2_outgassed`, `co2_weathered`)
- `day`

States are values. `step` returns a new state. `take_water`, `give_water` and `add_gas` rebind fields to new
tensors, so an earlier state or `state_dict` never changes.

**One day** (`step`):
1. **Insolation and sun angle.**
   - A locked planet gets S max(0, cos θ), with the substellar point fixed at lon 0, lat 0.
   - A rotating planet gets Berger's (1978) daily mean, with the declination asin(sin(obliquity) sin(2π (day +
     0.5) / year)). The orbit is circular, with the vernal equinox at day 0.
   - μ is the insolation-weighted daily mean of cos(zenith).
2. **Evaporation.** E = min(1, C_E U dt / H_w(T)) x (v_sat − v)⁺ x wet.
   - C_E = 1.15e-3 and U = 5 m/s.
   - H_w(T) is formation's vapour scale height, scaled as T².
   - wet = β (1 − f_snow) on land, with Manabe's β = soil / (0.75 x 150) capped at 1. On the ocean wet = 1 − f_ice.
   - Land evaporation is limited by the soil water.
3. **Radiation**, linearised backward Euler, stable at any step:
   T1 = T + dt (I (1 − α_p) − ε σ T⁴) / (C + dt 4 ε σ T³), with ε = 1 / (1 + 0.75 τ).
4. **Transport** on θ = T + (g/c_p) h, where h is the ground above the sea:
   - Sellers diffusion D ∇²θ through `globe.diffuse`.
   - Then an implicit, exactly energy-conserving exchange with a uniform free troposphere: C dθ = γ dt (H − θ_e).
     - θ_e = θ + L_v q / c_p, with q at the relative humidity r x wet.
     - H is set so that the exchange conserves energy.
     - γ = w ρ c_p C_H U, with w = Λ² / (1 + Λ²) and Λ = N H_p / (Ω R), using the real planet's R and Ω.
   - Vapour is mixed by the same D and γ over the air column's heat capacity.
5. **Precipitation.**
   - Background rate: vapour x (1 − e^(−1/8.9)).
   - Plus everything above the saturation column.
   - Snow forms on land below 273.15 K. Melt is 4 kg/m² per K per day.
   - Snow above 1,000 kg/m² leaves as glacier discharge to the ocean.
   - Bucket overflow runs off to the ocean.
6. **Carbonate-silicate cycle.**
   - Weathering on land: F0 exp((T − 288)/13.7) (r̄ / r0)^0.65, where r̄ is the one-year running-mean runoff.
   - F0 balances Earth's outgassing per m² of land.
   - Outgassing: Earth's 0.26 Gt CO2/yr x radiogenic / 0.047 W/m².

**Albedo and clouds:**
- **Surface:**
  - ocean: Briegleb et al. 1986 at μ: 0.024 at overhead sun, 0.14 at μ = 0.3
  - bare land 0.25, vegetation 0.15 (the plant cover from the biosphere)
  - snow and ice 0.6; the ice ramps over 258-268 K; snow covers the ground fully at 10 kg/m²
- **One cloud and air layer** (adding method): α_p = a_c + t² α_s / (1 − a_c α_s), with
  t = (1 − a_c)(1 − 0.22). The air absorbs 22 % (Stephens et al. 2012).
- **The ground** gets I t / (1 − a_c α_s). That is the light for plants and eyes.

**Greenhouse:**
- τ = τ_rest + k_CO2 ln(p_CO2 / co2_ref) + max(−τ_w,ref, k_w ln(v / v_ref)).
- τ_rest is the spec's `greenhouse_tau`, which already holds `tau_residual` for a capped planet.
- k_w gives the IPCC AR6 ratio of water-vapour plus lapse-rate feedback to Planck feedback, 1.30 / 3.22.
- τ_w,ref = 0.5 min(τ, τ_Earth) (Schmidt et al. 2010).
- The log form cannot run away.

**Spin-up** (`spin_up`):
- First, accelerated 60-day chunks: every cell at C = 1e7 and the year-mean insolation.
- Then real-C chunks with seasons, each as long as the longest year (at most 730 days).
- After each chunk:
  - The equilibrium is projected: T_eq = T̄ + N̄ / λ, with λ = Planck x (1 − 1.30/3.22).
  - The cloud albedo moves by λ (T_eq − T_s) / Ī / (dα_p / da_c).
- The water ledger is untouched.

**`ClimateRules`:**

| Rule | Value | Tag |
|---|---|---|
| `heat_ocean`, `heat_land`, `heat_spin` | 2.1e8, 1e7, 1e7 J m⁻² K⁻¹ | reference, new_rule, new_rule |
| `d_per_bar` | 0.55 W m⁻² K⁻¹ per bar | new_rule (contract) |
| `air_absorption` | 0.22 | reference |
| albedos | ocean 0.06 (or Briegleb), ice 0.6, land 0.25, vegetation 0.15 | reference |
| `ice_k`, `ice_ramp_k` | 263 K, 10 K | reference, new_rule |
| `bucket_kg_m2`, `beta_share`, `soil_start` | 150, 0.75, 0.5 | reference, reference, new_rule |
| `transfer_coeff`, `wind_m_s` | 1.15e-3, 5 m/s | reference, new_rule |
| `residence_days` | 8.9 d | reference |
| `degree_day_kg_m2_k`, `glacier_kg_m2`, `snow_mask_kg_m2` | 4, 1000, 10 | reference, new_rule, new_rule |
| `wv_lr_feedback`, `planck_feedback`, `water_share` | 1.30, 3.22 W m⁻² K⁻¹, 0.5 | reference, reference, reference+new_rule |
| `outgas_earth_kg_co2_yr` | 0.26e12 | reference (Gerlach 2011) |
| `runoff0_kg_m2_yr` | 22 mm/yr | derived from the model's own Earth |
| `runoff_mean_days` | 365.25 | new_rule |
| `weathering_k`, `weathering_runoff_exp` | 13.7 K, 0.65 | reference (WHAK 1981; Berner 1994) |

**Values:**
- **781**, probe:
  - WTG γ is 6.31 W m⁻² K⁻¹ (w about 1); D is 0.489.
  - After a short spin-up, the global mean over 120 days is 287.85 K, against the 287.67 K target.
  - Substellar maximum 316.4 K, antistellar minimum 274.6 K; day-side mean 299.2 K, night-side mean 276.5 K.
  - Planetary albedo 0.282, cloud albedo 0.236.
  - Evaporation 3.43 mm/day.
  - Vapour 28.8 kg/m², against the spec's 29.2.
  - Land runoff 203 mm/yr. No snow.
  - The builder measured 0.016 K spin-up error at G = 24.
  - A CO2 doubling warms 781 by 1.71 K with the water term, against 1.27 K Planck-only.
- **Earth twin**, probe, one year after spin-up:
  - γ is 0.176 (w about 0.04).
  - Global mean 287.74 K.
  - Equator ±15° 297.8 K; 60-90°N 271.2 K; 60-90°S 265.7 K.
  - Land runoff 75 mm/yr.
  - Weathering over outgassing 1.03.
- **CPU cost.**
  - At G = 48, W = 16, one step takes about 0.27 s, mostly the 50 diffusion sub-steps.
  - At G = 16 a step takes 2.5-6 ms.

**Deviations, with reasons:**
- **The WTG exchange was added.** The contract's Sellers D alone left a locked planet's mean 23 K below T_s, with
  its substellar point near 390 K. No cloud calibration can fix that.
- **The cloud albedo is calibrated per world in `spin_up`**, not fixed.
- **The H2O greenhouse term is on by default.** The contract asked for it where sensitivity matters.
- **The gas column and the ocean store are float64.**
- **Latent heat is not a separate term.** It sits inside D (the Budyko-Sellers convention) and inside the
  moist-static-energy exchange.
- **New rules:** glacier discharge, the running-mean runoff and the derived runoff0.
  - The daily bucket overflow is spiky, and the 0.65 power law is concave.
  - With Earth's observed runoff as runoff0, the model's Earth weathered only 13 % of its outgassing.
- **The WHAK pCO2^0.3 factor is omitted** (contract). The 0.65 runoff exponent is cited to Berner 1994, not WHAK.
- **D uses the spec's surface pressure**, constant through the run, not the evolving gas column.
- **The water used by photosynthesis** (one H2O per C, about 0.1 % of precipitation) is not booked.

**Open issues:**
- **Sea level is not re-solved.** The land mask is fixed while the ocean store changes slightly. `world.py` could
  call `globe.sea_level`.
- **Daily-mean insolation only.** There is no diurnal cycle (see section 5).
- **Snow** has no sublimation and no latent heat of fusion.
- **On a GPU**, `index_add_` in the scatters and ledgers is not bit-reproducible.
- See section 5 for the equator-pole contrast, the low runoff and the climate sensitivity.

### 2.7 `biosphere.py` (torch)
Per cell: `plant_c` (kg C/m², living), `wood_c` (the woody share, which can be collected), `litter_c`, and
`nutrient` (an N/P proxy, kg/m²).
- **Production:** NPP = LUE x PAR_abs x f_T x f_W x f_CO2 x f_N - maintenance respiration.
  - LUE = 1.8 g C per MJ of absorbed PAR (reference, Monteith-type).
  - PAR = surface insolation x `par_fraction`.
  - fAPAR = 1 - exp(-plant_c / 0.5).
  - f_T is a bell between 268 and 318 K with its peak at 298 K.
  - f_W = soil / 150.
  - f_CO2 = (p/(p + 30 Pa)) / (28/(28 + 30)).
- **Turnover to litter:** leaves 1/yr, wood 0.05/yr.
- **Decomposition:** k = 0.3/yr x Q10^((T-288)/10) x moisture, with Q10 = 2.
- **Gas exchange:** each mol of C fixed takes 1 mol CO2 and releases 1 mol O2; respiration and decomposition
  reverse this. Gases go into the climate state's `gas`.
- **Ocean:** phytoplankton as a production term limited by light and nutrients, for the carbon and O2 ledgers.
  It is not food for land individuals.
- `harvest(bio, w, cell, kg_dry_requested) -> kg_given` is a scatter that never goes negative.
  `collect_wood(...)` is analogous.

#### As built
**State** (`BioState`, float32 `[W, C]`):
- `plant_c`: living soft tissue (leaves, fine roots, herbs; the forage)
- `wood_c`: living wood. The two pools are disjoint.
- `litter_c`
- `nutrient`: mineral N, kg N/m²
- `phyto_c`
- `t_acclim`, `day`
- float64 ledger counters

**One day** (`step`, after `climate.step`):
- **GPP** = LUE x fAPAR x PAR x f_T x f_W x f_CO2 x f_N.
  - PAR = the ground shortwave x `par_fraction`.
  - f_T is the beta function of Yan & Hunt 1999 (zero at 268 and 318 K, 1 at 298 K).
  - f_N = N / (N + 4 g N/m²).
- **Maintenance, per pool, acclimated:**
  - R_leaf = r_N plant_c / 29 x q and R_wood = r_N wood_c / 330 x q.
  - q = 2^((T − T_acclim)/10), where T_acclim is the cell's one-year running-mean temperature.
  - r_N = 25.24 kg C per kg N per yr is derived so that a steady-state plant has NPP/GPP = 0.47 (Waring et al.
    1998).
  - GPP pays the soft tissue first, then the wood.
  - What is unpaid burns the pool's own tissue, then the other pool's, wood first.
- **Growth.** NPP is split 61 % to soft tissue and 39 % to wood (Malhi et al. 2011). It takes N at C:N 29 and
  330. Growth stops when the mineral N is empty.
- **Resprouting.** Where the plant could photosynthesise, soft tissue below 0.0782 x wood (its steady-state
  ratio) is rebuilt from the wood with a 30-day e-folding.
- **Seed floor.** Soft tissue stays at 1 g C/m² or more on land that gets light during the year. The carbon comes
  from the air through the gas exchange.
- **Turnover:** soft tissue 1/yr, wood 0.05/yr. The N returns to the mineral pool at litterfall.
- **N fixation:** at most 0.39 g N/m²/yr (Vitousek et al. 2013). It falls to 0 at the cell's starting N.
- **Decomposition:** 0.3/yr x 2^((T − 288)/10) x soil / 150. On ocean cells (corpses only) the soil factor is
  dropped.
- **Ocean production:** e_o x PAR x 1.066^(min(T, 293.15) − 291.15) x (1 − ice).
  - e_o = 6.17e-5 kg C per MJ PAR, calibrated to Earth's 48.5 Pg C/yr (Field et al. 1998). Earth's mean nutrient
    limitation is inside it.
  - The stock turns over in 7 days. 0.33 % of the loss is buried; the rest returns to CO2.
- **Gas exchange:** 1 mol CO2 for 1 mol O2. Guards keep any day from fixing more than half the CO2 column, and
  slow decomposition when O2 runs short.

**Scatters:**
- `harvest(bio, w, cell, kg_dry, carbon_fraction=0.47)` returns float64. It returns exactly what left the pool,
  which can be more than was asked. 85 % of the eaten tissue's N goes back to the cell.
- `collect_wood` books carbon at materials' wood carbon fraction, 0.4927.
- `add_litter`.
- `cover(bio, rules)` is fAPAR, for the climate's albedo. The rules are required.

**Spin-up** (`spin_up`):
- Two 365-day segments, with the chain's air held fixed.
- Each segment ends with a steady-state jump:
  - wood, and before the last segment also the soft tissue, go to their per-cell steady state from the window's
    mean fluxes
  - litter goes to the window year's exact periodic steady state
- The water ledger is reset only with `reset_water=True`.

**`BioRules` with sources:**

| Rule | Value |
|---|---|
| `lue_g_c_mj` | 1.8 (Monteith 1977) |
| `fapar_kg_c` | 0.5 (contract) |
| `t_min_k`, `t_opt_k`, `t_max_k` | 268, 298, 318 K |
| `co2_half_pa`, `co2_ref_pa` | 30, 28 Pa |
| `leaf_turnover_yr`, `wood_turnover_yr` | 1, 0.05 |
| `decomposition_yr` | 0.3 |
| `q10` | 2 |
| `acclimation_days` | 365.25 |
| `wood_allocation` | 0.39 |
| `npp_gpp` | 0.47 |
| `cn_leaf`, `cn_wood` | 29, 330 (Sitch et al. 2003) |
| `nutrient_start_kg_m2`, `nutrient_half_kg_m2` | 0.03, 0.004 |
| `n_fixation_kg_m2_yr` | 3.9e-4 |
| `excreted_n_share` | 0.85 (Haynes & Williams 1993) |
| `seed_c_kg_m2`, `seed_floor_kg_m2` | 0.05, 1e-3 |
| `resprout_days` | 30 |
| `plant_carbon_fraction` | 0.47 |
| `eppley_base`, `eppley_t_k`, `eppley_t_max_k` | 1.066, 291.15, 293.15 K |
| `phyto_turnover_days` | 7 |
| `burial_share` | 0.0033 |

**Values**, probe, after `spin_up` plus one year:

| Quantity | 781 | Earth twin |
|---|---|---|
| NPP, global mean (kg C/m²/yr) | 0.088 | 0.139 |
| GPP | 0.187 | 0.300 |
| ocean production | 0.062 (0.20 before the Eppley cap) | 0.083 |
| plant / wood / litter (kg C/m²) | 0.054 / 0.664 / 0.261 | 0.089 / 0.888 / 0.490 |
| air CO2 after 1 year | x 1.0000014 | x 1.011 |
| carbon ledger error (kg C/m²) | 1.7e-7 | 1.4e-8 |
| O2 ledger error (mol/m²) | −1.4e-5 | −1.2e-6 |

Other measured behaviour:
- **Grazing (781, G = 12).** Cells grazed down to 0-10 % recover to 54-59 % of their soft tissue in 30 days and
  to 83 % in 480 days. Their wood settles at 0.80-0.83 of its old stock.

**Deviations, with reasons:**
- **Disjoint pools, with each pool paying its own maintenance.** With soft tissue paying for the wood, a grazed
  cell burnt its last leaves, reached fAPAR 0 and never regrew.
- **Acclimated maintenance respiration.** With a fixed 288 K reference and Q10 2, warm cells had no steady state.
  They went through decadal boom-bust cycles, and no spin-up was possible. This changes the contract's
  respiration form.
- **Resprouting, the seed floor, N return from grazers and N fixation are added.** Without them, grazed land
  stripped for good, or stalled on N.
- **NPP = GPP − R_m, with no separate growth respiration.** LUE 1.8 is read as gross.
- **Wood allocation is 0.39**, the wood share of Malhi et al. 2011. The first value, 0.3, was the canopy share,
  misquoted.
- **The Eppley factor is capped at 20 °C** (the VGPM optimum). Uncapped, 781's warm ocean produced 5 times
  Earth's ocean mean.
- **Slow pools are put at their steady state at the end of the spin-up.** Without that, Earth's CO2 fell 15 % in
  the first year.
- **Plants are seeded only on land lit during the year.** A locked planet's night side stays bare, so its NPP is
  exactly 0.
- **Litter added by other modules brings carbon only.** Bodies carry no N ledger.

**Open issues:**
- **Earth's CO2 still drifts about 0.5-1 %/yr** in the first years after spin-up. The causes are the decadal
  leaf-wood mode and the seasonal mean-field steady state. 781 does not drift, because its CO2 column is about
  240 times Earth's: 186 kg C/m² against 0.78.
- **No ocean carbon buffer.** Any biosphere imbalance reaches the air one to one (section 5).
- **A cell stripped of both soft tissue and wood** regrows from the 1 g seed floor at about 0.4 %/day: years to a
  full stand.
- **One fixed allocation.** Dry marginal cells can still boom and bust over decades. An adaptive allocation
  (grass versus trees) would be the real fix.
- **No dispersal** beyond the seed floor on lit land.

### 2.8 `creatures.py` (torch): bodies, energetics, life history
State `[W,N]`:
- `alive`, `pos [W,N,3]`, `heading` (rad from local north), `mass_kg`, `reserve_j` (fat), `water_kg`, `health`
- `age_d`, `cooldown_d`, `uid`, `parent`, `founder`, `generation`
- `inv [W,N,K]` (item indices, -1 empty; K = 3), `calls [W,N,D]`

The genome holds the brain (section 2.10) plus genes:
- `adult_mass_kg`
- `diet` in [0,1] (0 plant, 1 meat; digestibility trades off linearly)
- `insulation` in [0.2, 3], `speed` in [0.3, 1.2], `acuity` (vision), `hearing`, `voice`
- `swim` in [0,1], `litter` (offspring count, 1-6)

Energetics per day (reference allometry; each formula cites its source in code):
- **Basal:** 3.4 W x M^0.75 (Kleiber, mammal-like) x 86400.
- **Locomotion:** 10.7 M^-0.316 J kg⁻¹ m⁻¹ x M x distance (Taylor, Heglund & Maloiy 1982). Climbing costs
  M g Δh / 0.25. Water costs drag 0.5 ρ_w C_d A v² x distance, with A = 0.1 M^(2/3).
- **Thermoregulation:** below T_lc = 303 K - 8 x insulation, extra power C_th (T_lc - T) with
  C_th = 0.174 M^0.5 / insulation W/K (Herreid & Kessel 1967) *(corrected v3: was 1.0, about 5.7 times the
  published minimal conductance, mis-tagged reference)*. Above 310 K, evaporative water loss = excess / 2.43e6 J/kg.
- **Brain:** `neuron_cost_w` x active units (new_rule, default 0.002 W per unit).
- **Oxygen:** aerobic capacity a = pO2 / (pO2 + 3 kPa) scales the top speed and strike power. If pO2 is below
  1 kPa, health falls (hypoxia).
- **Water turnover:** 0.12 M^0.82 kg/day (Nagy & Peterson 1988), plus evaporative loss. Drinking is possible
  where `soil` is above 50 kg/m² or the cell is land with a lake. Ocean water raises salt load and does not
  rehydrate unless `swim` is above 0.5.
- **Food:**
  - Plant intake is at most 0.08 M^0.75 kg dry per day, with energy 18.5e6 J/kg x (0.75 - 0.5 diet)
    digestible.
  - Meat comes from corpse items: 7e6 J/kg x (0.4 + 0.5 diet) x cooking gain, if cooked.
  - Fat in the reserve is at most 0.3 M x 39.5e6 J.
- **Respiration** adds O2 consumed = J / 4.5e5 J/mol and CO2 = 0.85 x that, into the gas ledger.
- **Life history** (Calder 1984 and Peters 1983 allometry):
  - maximum lifespan 11.6 yr x M^0.20
  - maturity 0.6 yr x M^0.27
  - gestation 65 d x M^0.25
  - litter mass 0.1 M^0.92
  - juvenile growth follows von Bertalanffy, paid from the reserve
  - death by starvation (reserve below 0 and lean tissue lost), dehydration (water below 0.8 x the norm),
    injury, age or hypoxia
- **A dead body becomes items:** meat 0.45 M, fat from the reserve, bone 0.1 M, hide 0.06 M. Its carbon goes into
  the item ledger, and decay returns it to litter and CO2.
- **Reproduction** is asexual by default. It happens when the individual is mature, its reserve is above half
  the maximum and its cooldown is 0. The offspring pay their tissue energy from the parent's reserve. The genome
  mutates (brain section 2.10, genes at a scale of 3 %).

#### As built
**Extra state fields:**
- `frame_kg`: the largest lean mass reached, the starvation reference
- `harm`: the cause code of the last health loss
- `struck`: health lost to strikes since the last `physiology`
- `loud`
- `genome`, `wh_live`, `brain_state`, `baseline`
- `next_uid [W]`
- `ledger` (float64)

`state_hash` covers them all.

**Genes** (`BODY_SPECS`):

| Gene | Range | Mutation |
|---|---|---|
| `adult_mass_kg` | 10 g to 5 t | 3 % log-normal |
| `diet` | 0-1 | 3 % of the range |
| `insulation` | 0.2-3 | 3 % of the range |
| `speed` | 0.3-1.2 | 3 % of the range |
| `acuity`, `hearing` | 0-5 (world7's sensor scale) | 3 % of the range |
| `voice` | 0-1.5 | 3 % of the range |
| `swim` | 0-1 | 3 % of the range |
| `litter` | 1-6 | ±1 with probability 0.03 |

`GENE_SPECS = {**brain.BRAIN_SPECS, **BODY_SPECS}`.

**Body model** (`new_rule`):
- The lean body is meat 0.84 + bone 0.10 + hide 0.06 by mass (`BODY_MIX`). The 0.84 counts the viscera as
  meat-like tissue.
- From the materials makeup this gives:
  - `TISSUE_J_KG` 7.216e6 J/kg
  - `TISSUE_C_KG` 0.1557 kg C/kg
  - `WATER_FRACTION` 0.668
- The reserve is fat, with carbon `FAT_C_PER_J` 1.921e-8 kg C/J (RQ 0.72).
- Tissue built from fat costs `GROWTH_J_KG` 8.10e6 J/kg: a 12 % growth heat.

**Energetics as built:**
- **Basal:** 3.4 M^0.75 W. The constant is shared with `brain.KLEIBER_W`, so the two cannot drift apart.
- **Travel:**
  - Top speed v = 1.53 M^0.24 (the trot-gallop transition, Heglund et al. 1974) x speed gene x a.
  - Distance = speed output x v x `MOVE_S` (7,200 s).
  - The turn is at most π per day.
  - Land cost = 10.7 M^−0.316 x M x d x (1 + swim).
  - Climb = M g rise / 0.25. The rise is summed on land over the same bilinear surface sight uses, one sample
    per smallest cell spacing, at most 256 samples.
  - Swim = 0.5 ρ C_d A v² d / (0.25 η), with C_d 0.04 on the Meeh area and η from 0.33 (paddling) to 0.85
    (lift), set by the swim gene.
  - Speed in water is capped where drag power reaches the aerobic ceiling.
- **Thermoregulation:**
  - C_th = 0.174 M^0.5 / insulation W/K below T_lc = 303 − 8 x insulation.
  - The power is capped at the aerobic ceiling, 10 x basal x a. The uncovered share costs health as "cold".
  - Above 310 K, the day's respired heat plus C_th (T − 310) is evaporated at 2.43e6 J/kg.
- **Other costs:**
  - brain: 0.002 W x k
  - senses: `SENSE_W` M^0.75 (2 acuity² + hearing²), with `SENSE_W` = 0.0286. It is calibrated so 781's founders
    spend 7.4 % of basal, as the chain's senses era does.
  - calling: acoustic power at 1 % efficiency for 600 s a day
- **Respiration.** O2 = J / 4.5e5 J/mol. CO2 is the carbon actually oxidised.
- **Water.**
  - Turnover is 0.12 M^0.82 kg/day.
  - `drink` fills to the norm plus the day's turnover, plus the expected evaporation at rest when `t_air_k` is
    given.
  - It drinks from a lake, else from soil above 50 kg/m² (shared among drinkers), else from the sea, which only
    rehydrates when swim > 0.5.
- **Food.**
  - Plants: intake at most 0.08 M^0.75 kg dry per day, energy 18.5e6 x (0.75 − 0.5 diet) J/kg. The C fraction
    is 0.47. It books exactly what the harvester gave.
  - Items: the best food item held or within 5 m. Digestible = `materials.DIGESTIBLE` interpolated in the diet x
    `COOK_GAIN` if cooked, capped at the gross energy.
  - An item counts as cooked once its peak temperature has reached 343.15 K.
- **Life history:** as the contract.
  - For 30 kg: lifespan 8,365 d, maturity 1.5 yr.
  - Von Bertalanffy growth reaches 90 % of adult mass at maturity. It is paid from reserve above 1/4 of the
    maximum.
- **Deaths.** Causes: starvation (lean mass below 0.6 of `frame_kg`), dehydration (water below 0.8 x the norm),
  injury, age, hypoxia, or cold.
  - Age death is deterministic, at the maximum lifespan.
  - A body becomes meat 0.45 M, bone 0.10 M, hide 0.06 M and fat (reserve / 39.5e6) items. The viscera, 0.39 M,
    go to litter.
- **Reproduction.** Ready parents are mature, have reserve above half the maximum, cooldown 0, and can spare the
  litter's water (water − litter water ≥ 0.8 x norm).
  - At capacity they are served in a random order.
  - The parent pays the tissue (8.10 MJ/kg), the child's reserve (half its maximum) and the water.
  - Cooldown = 65 d x M^0.25.
  - `brain.mutate` and `brain.install` build the child: live weights = genome Wh, state 0, baseline 0.
- **Founders** (`spawn_founders`, `founder_genes`):
  - They are placed on land cells by area, jittered inside their own cell.
  - They start as young adults (age = maturity) at half their maximum reserve.

  For 781:

  | Gene | Value | Tag |
  |---|---|---|
  | `adult_mass_kg` | 30 kg | new_rule |
  | `diet` | 0.034 | new_rule proxy |
  | `insulation` | 1.875 = (303 − T_s)/8 | derived |
  | `acuity`, `hearing`, `voice` | 2.04, 0.685, 0.08 | chain senses era |
  | `speed`, `swim`, `litter` | 1, 0, 2 | new_rule |

  781's O2 of 2.5 kPa gives a = 0.455.

**Death times** (1 kg, insulation 1, after the C_th fix):

| Condition | Result |
|---|---|
| no water | dehydration in 2 d |
| fed at 200 K | starvation in 15 d |
| unfed at 296 K | starvation in 31 d |
| 220 K at 1.5 kPa O2 | cold in 3 d |
| 0.4 kPa O2 | hypoxia in 1 d |
| fed at 270 K | alive at 60 d |
| watered at 320 K | alive at 30 d |

A 30 kg animal dies of dehydration in 3 d and unfed in 73 d.

**Speed.** One day of observe plus move, eat, drink, physiology, deaths and reproduce takes about 0.4 s on the
CPU for W = 4, N = 1024, G = 48.

**Deviations, with reasons:**
- **C_th = 0.174, not 1.0.** The published value. With 1.0, cold was about 5.7 times too costly.
- **CO2 is the carbon actually oxidised**, not 0.85 x O2. Fat alone gives RQ 0.72, and a grazer's overall RQ
  comes to about 0.95. This makes the carbon ledger close exactly.
- **Item digestibility is capped at the gross energy.** The contract's 0.9 x 1.3 = 1.17 would digest more energy
  than the meat holds.
- **Above 310 K the respired heat is evaporated too.** No dry heat loss is possible once the air is at body
  temperature.
- **Swimming cost** divides the drag work by muscle and propulsive efficiency.
- **Cold is a death cause.** It applies when heat loss exceeds the aerobic ceiling.
- **Extra costs:** sensor upkeep, the swim gene's land penalty, and calling. Without them those genes drift free.
- **Drinking fills the norm plus the day's turnover.** Without it, animals below about 1 kg lose a lethal 20 %
  between two daily drinks.
- **Founder diet is the bodies era's `predator_share`** (0.034 for 781), the predators' share of all biomass.
  The better value, predators / (grazers + predators) = 0.10, needs guild biomasses that `chain.py` does not
  export.
- **Healing** needs health > 0, no strike that day, no hypoxia or cold, and being fed and watered. Before this, a
  lethal strike could be healed the same day.
- **`harm`** names the largest of the day's harms.
- **State stays float32** whatever dtype the environment fields have. The climate's pressures are float64.

**`new_rule` list:**
- `FOUNDER_MASS_KG` 30, `FOUNDER_RESERVE` 0.5
- `BODY_MIX`
- `MOVE_S` 7,200 s, `MAX_TURN` π
- `SWIM_LAND_PENALTY` 1
- `HYPOXIA_PER_DAY` 2, `COLD_PER_DAY` 1, `HEAL_PER_DAY` 0.05
- `VB_MATURE_FRACTION` 0.9, `GROWTH_FLOOR` 0.25, `CHILD_RESERVE` 0.5
- `REACH_M` 5 m
- `CALL_S` 600 s
- `SENSES_DEFAULT` (eyes 1, ears 1, talkers 0.1 when there is no senses era)
- the contract's thresholds, retagged `new_rule`: `T_LC0`, `AEROBIC_HALF_PA`, `HYPOXIA_PA`, `DEHYDRATION_LETHAL`,
  `DRINK_SOIL_MIN`, `SWIM_SALT`, `PLANT_DIG0`, `PLANT_INTAKE`, `GESTATION_D`, `RESERVE_MAX_FRACTION`, `CORPSE`

**Open issues:**
- **T_lc is not derived.** T_lc = 303 − 8 x insulation is mass-independent. The published conductance would give
  T_lc = T_body − basal / C: about 290 K at 1 kg and 264 K at 30 kg.
- **Evaporative cooling switches on as a step at 310 K.** Real mammals ramp it up from the upper critical
  temperature.
- **Small corpses and water.** A very small animal (below about 0.5 kg) that dies of dehydration can hold less
  water than its corpse items need. The rest is drawn from the environment; the ledger still closes.
- **Founder diet** stays a proxy until `chain.py` exports the guild biomasses.

### 2.9 `senses.py` (torch)
`observe(world) -> x [W,N,in_dim]` is the only input to the brain. Sectors S = 8 around the heading, out to
range r_v:
- Vision range r_v = base x acuity x light, with light from the cell's insolation and a floor at night or on the
  dark side. The horizon on the habitat globe is real: d = sqrt(2 R h), with h the eye height above the local
  ground plus the terrain lines of sight sampled along great circles.
- Per sector:
  - nearest conspecific proximity
  - the largest heavier animal (threat proxy, by mass ratio)
  - the largest lighter animal (prey proxy)
  - plant carbon sampled along the sector
  - fresh water
  - meat items
  - stone/ore/clay/wood richness (4 channels)
  - fire heat
  - the call vector (D = 8 channels): spherical spreading 1/r² plus atmospheric absorption, a speed of sound
    from `sound_speed_m_s`, no sound if P < 1 kPa, and occlusion by terrain x 0.5
- Self: reserve fraction, water fraction, health, age fraction, mass/adult, local T (scaled), local plant, local
  water, light, slope ahead, the held items' properties (hardness, sharpness, mass, fuel, food, temperature for
  each slot), being in water, and bias.

#### As built
**Layout.** `IN_DIM = 182` = 8 sectors x 19 channels + 30 self channels.
- Sector channels, in order: kin, threat, prey, plant, water, meat, stone, ore, clay, wood, fire, call0-call7.
- Self channels:
  - reserve, water, health, age, mass, temp, plant, fresh, light, slope
  - 3 slots x (hardness, sharp, mass, fuel, food, temp)
  - in_water, bias
- `PLANT_INDEX = 158` is the local plant input that the innate forage wiring reads.

**Vision:**
- r_v = 1,000 m x acuity x light, with light = 0.05 + 0.95 min(1, I / 340 W/m²).
- Eyes are 0.2 M^(1/3) m high: 0.62 m at 30 kg.
- **Line of sight.** Each sight line is tested at 8 points against the bilinear surface (the ground, or sea level
  over the ocean). The chord sags R θ² t(1 − t)/2, so the horizon is real.
- **Fires** are seen by their own light out to 1,000 m x acuity.
- **Other individuals:**
  - kin: proximity of the nearest within a mass ratio of 2
  - threat: proximity x (1 − m_self / m_other) for the largest heavier one
  - prey: proximity x m_other / m_self for the largest lighter one
- **Field channels** are sampled at 1/16, 1/8, 1/4, 1/2 and 1 of r_v along the sector's centre line, and
  averaged:
  - plant: cover 1 − exp(−plant_c / 0.5)
  - water: soil ≥ 50 kg/m² on land, or a lake
  - stone, ore, clay, wood: richness 1 − exp(−x / x_typ), where x_typ is the world's typical deposit of the class
- **meat:** proximity x (1 − exp(−E / 10 MJ)).
- **fire:** proximity x (T_fire − T_air) / 1,000 K.

**Hearing:**
- Pressure at 1 m: 2 Pa x loud x voice x (M/30)^0.5 x ρ / 1.204. It spreads as 1/r.
- Absorption: 5 dB/km in Earth air (ISO 9613-1), scaled by 1/(ρ c³). That is how the planet's speed of sound
  enters.
- Terrain between halves the pressure.
- The threshold is 20 µPa / hearing.
- Weight = the sensation level / 60 dB, clamped to [0, 1].
- **Sector call vectors.** Σ w v / max(1, Σ w): the weighted sum while the summed weight is at most 1, else the
  weighted mean. Loudness saturates, the content does not.
- No sound below 1 kPa.
- **Which pairs get a line of sight.** Only pairs that are audible without occlusion, or in sight range. With
  1,024 founders on G = 48, voice 0.5 and hearing 2, this is about 89,000 of 1.05 million pairs, and `observe`
  takes 0.06 s on the CPU.

**Deviations, with reasons:**
- **Uneven field samples.** The real horizon over level ground is about 200 m, so most of an evenly sampled range
  would be hidden.
- **The bilinear sight surface.** With cell steps, every cell acted as a plateau that hid everything beyond
  111 m. A one-cell feature now counts at about 1/4 of its height, for sight and for climbing alike.
- **Beyond the curvature horizon, sound gets the 0.5 occlusion.** There is no diffraction model.
- **Calls are mixed as above**, not as a plain clamped sum. A clamped sum saturated at ±1 in any group, and the
  content was lost.
- **Richness is relative to the world's typical deposit.** Divided by the 2,670 kg/m² column, ores were about
  1e-3: invisible.

**`new_rule` list:**
- `VISION_BASE_M` 1,000 m, `LIGHT_FLOOR` 0.05, `EYE_K` 0.2
- heights: item 0.1 m, fire 1 m, field 0.5 m
- `KIN_RATIO` 2
- `SL_FULL_DB` 60
- the scales: `MEAT_SCALE_J` 10 MJ, `FIRE_SCALE_K` 1,000 K, `WOOD_SCALE_C` 5 kg C/m², `SLOPE_PROBE_M` 100 m,
  temperature (T − 288.15)/30, held items (hardness/10, tanh(mass/1 kg), tanh(J/10 MJ), tanh(ΔT/500 K))

**Open issues:**
- **Dense pair tensors.** `observe` builds `[W, N, N]` pair tensors, about 67 MB each at W = 16, N = 1024. Chunk
  over worlds if the capacity grows.
- **The sight pre-mask is range-only**, so pairs in range but beyond the horizon are still tested.
- **`surface_at` reads `globe._frame`**, a private attribute.
- **Light is the daily mean.** A rotating planet has no night within a tick (section 5).

### 2.10 `brain.py` (torch): bigger brains
This extends `haishool/life9/brain.py` with configurable dimensions:
- The default maximum is **hidden = 128** (four times the flat prototype), with 48 units at the start. The active
  units `k` are a gene (2..hidden) and each costs metabolism.
- Outputs are turn, speed, loudness and the vocal vector (D = 8), plus `A` action logits:

  rest, forage, eat_meat, drink, strike, collect, drop, knap, combine, make_fire, feed_fire, heat_item, cook,
  share

  The action is sampled from softmax(logits / temperature gene) with the world generator. `collect` takes its
  class from 4 extra logits (stone, ore, clay, wood).
- Within a life, Hebbian plasticity is modulated by the individual's own outcome:
  (Δreserve + energy invested in offspring or growth) / (one day of basal energy, 3.4 M^0.75 x 86400 J)
  + 10 Δwater/M + 8 Δhealth, minus its running mean, which moves at 0.05 per day. The learning rate is per day.
  *(corrected v3: the reserve term was Δreserve / 1e6 J. That made the modulator a sign function above a few kg,
  counted a birth as a loss, and carried the flat prototype's per-tick rates into a per-day tick.)*
- Founders get innate output biases toward `forage` with a positive gain on the local plant input.
  - This is derived: world7 bodies give grazer lineages.
  - It is labelled `chain`.
  - It is the only innate wiring.

#### As built
**Output layout** (`Layout`, `OUT_DIM` 29):

| Index | Output |
|---|---|
| 0 | turn |
| 1 | speed |
| 2 | loud |
| 3-10 | vocal (8) |
| 11-24 | the 14 action logits (`FORAGE_OUT` = 12) |
| 25-28 | the 4 collect-class logits |

**The step** (`think`):
- new = tanh(x Wx + h Wh_live + b) on units < k. Units beyond k are exactly zero and are not computed.
- out = new Wo + bo.
- Only units up to max(k) are evaluated. That costs one host sync, unless the caller passes `width ≥ max(k)`.
- Narrow-dtype weights are cast to float32 per chunk.

**Actions.** Gumbel-max sampling of softmax(logits / temperature) with the passed generator.
- The action is drawn first, then the collect class.
- A row with nothing allowed means "rest only", in both `action_probs` and `sample_actions`.

**Learning:**
- `outcome(d_reserve, d_water, M, d_health, invested_j, basal_j, water_margin_kg)` follows the corrected
  contract.
  - A fasting day is about −1, and a full grazing day about +2.8, at any body mass.
  - *(integration round)* The world passes as `d_reserve` the change of body energy: the reserve change less
    the lean tissue burnt that day (`burnt_kg x TISSUE_J_KG`). Starvation covers a negative reserve with lean
    tissue, so the reserve change alone read 0 exactly when fasting hurt most.
  - *(integration round)* With `water_margin_kg` (the world passes the lethal margin 0.2 x the norm) the water
    term is 8 Δwater / margin: losing the whole margin weighs as much as losing all health, both being death.
    The contract's 10 Δwater / M valued the 2-day water margin at −1.3, a fifth of one grazing day.
- `modulate` gives tanh(outcome − baseline). The baseline moves at 0.05 per day: a 20-day time constant.
  - *(integration round)* The world starts the baseline at an individual's first outcome (founders and
    newborns), so the first weeks carry no systematic anti-Hebbian bias.
- `hebbian`: dW[i, j] = η x mod x before_i x after_j on the active block, clipped to ±3.
  - Live weights may be float32 or bfloat16.
  - bfloat16 needs a generator for unbiased stochastic rounding.

**Time scale.** One tick is one day, and the flat prototype's per-tick rates are restated per day:
- `TICK_TO_DAY` = 600 / 8,365 = 0.0717.
  - The flat life is 600 ticks.
  - A 30 kg Calder lifespan is 8,365 days.
- η0 = 0.01 x 0.0717 = 7.17e-4 per day. The η gene's bounds are 0 to 0.0143, with a step of 3.59e-4 per
  generation.
- `baseline_rate` stays 0.05 per day. It is deliberately not rescaled by life length: a 279-day time constant
  would credit seasons to behaviour.

**Genes and mutation:**
- **Weights.** Wx, Wh and Wo get N(0, (0.1 x founder sd)²), with the founder sd taken at the parent's own fan-in:
  Wx 1/sqrt(in_dim), Wh 0.5/sqrt(k), Wo 1/sqrt(k).
  - One mutation moves the logits by about 0.2 of the founder spread at k = 16, 48 and 128 alike.
- **Biases.** b and bo get N(0, 0.05²).
- **k.** It moves by exactly ±1 with probability 0.05, within [2, min(hidden, the genome's units)].
- **Clip.** Weight genes are clipped to ±3.
- **Temperature gene.** It starts at 1, with bounds 0.1-10 and a 3 % log-normal step.
- **Unknown genes.** `mutate` raises `KeyError` for a genome entry without a `GeneSpec`, so no body gene silently
  stops evolving.

**Initial weights.**
- The founder sd uses k0 = `initial_hidden`: Wh 0.5/sqrt(k0) and Wo 1/sqrt(k0). Wx stays 1/sqrt(in_dim).
- **Innate wiring:**
  - It is one clean relay unit (unit 0).
  - Its input is 2.0 on `PLANT_INDEX` and zero elsewhere. Its recurrent row and column are zero.
  - Its output is 2.0 to the forage logit, plus a forage bias of 1.0.
  - So, at temperature 1 and ignoring the random weights, P(forage) is about 0.17 without plants and about 0.58
    at full local plant.

**Birth.** `install(genome, wh_live, state, w, c, child, baseline)` writes the whole child:
- It raises if any key is missing.
- It sets live weights to the genome's Wh, and zeroes the state and the baseline.
- Nothing learned is inherited.

**Speed and memory** (in_dim 180 in the benchmark; `senses` gives 182):
- `think` for 2,048 individuals at hidden 128 with 48 active takes 2.2-2.8 ms on the CPU with 24 threads.
- `hebbian` takes 1.7 ms with float32 live weights.
- At full scale (16 worlds x 1,024):
  - genome plus live weights take about 3.9 GB in float32
  - about 2.5 GB with bfloat16 genes and float32 live weights (the recommended setting)

**Deviations:**
- The outcome and rates, as corrected in the contract above.
- Mutation is relative to the fan-in. The flat absolute 0.05 grew with k and penalised big brains on top of
  their metabolic price.
- k steps by exactly ±1. The flat prototype drew from {−1, 0, +1}.
- The innate magnitudes 1.0 / 2.0 / 2.0 are `new_rule`. The contract gives only their sign and tag.

**`new_rule` list:**
- `weight_mutation_rel` 0.1, `bias_mutation_sd` 0.05
- `REF_MASS_KG` 30
- `baseline_rate` 0.05 per day, not rescaled
- the init scales at k0
- the innate magnitudes
- "nothing allowed means rest"

**Open issues:**
- **`world.py` duties:** pass `invested_j` to `brain.outcome` and call `install(..., baseline=...)`. On an
  accelerator, refresh any cached `width` after every install; the check runs only on the CPU.
- **`REF_MASS_KG` = 30 kg** is tied to the founders' mass.
- **Config clash.** The flat `Config9` now has `initial_hidden` 16. A planet config should set hidden 128 and
  `initial_hidden` 48.
- **bfloat16 live weights** make Hebbian learning about 13 times slower, because of the stochastic rounding.
- **No GPU run yet.**

### 2.11 `crafting.py` (torch): items, fires and their physics
- **Item pool** `[W,I]` (I = 4,096 by default):
  - `alive`, `comp [W,I,S]` (mass fractions), `mass`, `temp_k`, `sharp` (0-1), `pos [W,I,3]`, `holder`
    (slot index or -1)
  - assembly fields `head_mass`, `handle_len_m`, `bond` (0-1)
  - process fields `peak_k`, `hot_days`, `age_d`
- **Fire pool** `[W,F]` (F = 256): `alive`, `pos`, `fuel_kg [W,F,S]`, `temp_k`, `air` (effort).
- **Actions:**
  - `collect`: from the cell's deposit stock or wood, and the yield rises with the tool's hardness against the
    rock
  - `drop`
  - `knap`: a held hammer against a brittle held core whose hardness ≤ the hammer's x 1.1 removes mass to a
    debris item and raises `sharp` toward a toughness-dependent limit
  - `combine`: head + handle (+ fiber or resin binder raises `bond`)
  - `make_fire`: fuel item(s) + ignition. Ignition needs friction work (effort, against the fuel's
    `IGNITION_K`) or flint + pyrite sparks, and O2 mole fraction ≥ 0.15 (Belcher et al. 2010). Without it the
    fire fails.
  - `feed_fire`
  - `heat_item`: an item in a fire. Its temperature relaxes toward the fire temperature by conduction (cp,
    mass), and the `TRANSFORMS` apply when their conditions hold.
  - `cook`: meat in a fire gets `COOK_GAIN`
  - `share`: give a held item to the nearest relative within reach
- **Fire temperature:** T_fire = T_air + ΔT0 x sqrt(x_O2/0.21) x (fuel factor: wood 1.0, charcoal 1.35)
  x (1 + 0.3 air), with ΔT0 = 812 K. This is a calibrated approximation (new_rule): an open wood fire is about
  1,100 K at Earth O2. A charcoal fire is about 1,500 K at a third of full forced air, and 1,713 K at full effort
  for the hour fanned. *(corrected v3: ΔT0 was 1000 K, which contradicts the contract's own 1,100 K anchor and
  gave a 2,043 K forced charcoal fire, above iron's melting point.)* Burning consumes fuel at a rate set by the
  fire's size, releases COMBUSTION_J_KG, and uses O2 and makes CO2 in the gas ledger.
- **What tools change (physics only):**
  - strike damage = ½ m_head v² with v = v_arm (1 + handle_len/0.6 m), scaled by sharp and hardness against the
    target's toughness
  - butchering yield per day = base + sharp x hardness term
  - chopping and digging yields are ratios of hardness
  - fire warms everyone within 20 m (it cuts thermoregulation)
  - cooking gain
  - hide + fiber worn as a held item adds insulation
- **Items decay:** meat and fat spoil with Q10 temperature dependence (to litter carbon), and wood rots slowly.

#### As built
**Conventions:**
- Pools are changed in place.
- `inv [W, N, K]` is kept in step with `holder = n K + k`. `check_inventory` verifies the pair.
- `act [W, N]` masks the actors.
- `effort` is already scaled by aerobic capacity.
- Actions return per-actor outcomes and per-world float64 ledger flows.
- One generator is used, by `collect` only.

**Actions:**

| Action | Rule as built |
|---|---|
| `collect(choice)` | First picks up the nearest loose item within 5 m whose largest species falls in the chosen group (`PICKUP_RULE`). Else it digs a lump: a species drawn in proportion to the cell's stock of the class. Mass = 0.1 M x effort x `dig_yield`, with the tool's `tool_hardness` against the deposit's. Wood uses `chop_yield`. An item hotter than 333.15 K is never taken by hand: an actor holding another item rakes it out onto the ground, and a bare-handed actor digs instead. |
| `drop` | puts a held item on the ground |
| `knap` | Works if the core is at least half brittle and its hardness ≤ 1.1 x the hammer's working hardness. Removes 0.2 x effort of the knappable mass as a debris flake. sharp → lim − (lim − s) exp(−share/0.05), with lim = 1/(1 + (K_IC/2)²): flint 0.703, basalt 0.372, glass 0.877, resin 0.941. On an assembly it works only the brittle species. `body_mass_kg` is required. |
| `combine` | Merges head, handle and binder. bond = 0.2 + 0.8 x binds x (1 − exp(−m_binder / (0.05 m_head))). The handle length comes from a rod of aspect 15, shortened by min(1, K_IC/2). Heat content is conserved. |
| `make_fire` | Needs x_O2 ≥ 0.15 and at least 10 g of fuel. Friction lights it if T_air + 0.1 x effort x 75 W (M/70)^0.75 / (8 x 0.005 m x k_fuel) ≥ the fuel's ignition point; the smallest body that can drill-light wood is about 11.5 kg. A spark lights it with an item at least half pyrite, plus a striker of working hardness ≥ 6.25, plus a fuel igniting at ≤ 550 K. |
| `feed_fire(fan)` | Fuel goes into the nearest fire's bed. `fan` is forced air for the next hour. By default a feeder does not fan, and an actor with nothing to feed fans at its effort. |
| `heat_item` | Lays the held item in the nearest fire. `fire_step` then heats, transforms or burns it. |
| `cook` | Holds the item at roasting distance for an hour. It meets T_air + 0.2 (T_fire − T_air): about 450 K by a wood fire, 507 K by a charcoal fire. Cooked when the peak reaches 343.15 K. |
| `share` | Gives a held item to the nearest living parent, child or sibling with a free hand within 5 m. |
| `fire_step` | Runs 48 sub-steps per day (see below). |
| `decay` | Rots items (see below). |

**Fire physics** (`fire_step`, per sub-step: kill, temperature, burn the beds, then the items):
- **A fire dies** below 10 g of fuel or at x_O2 < 0.15.
- **Temperature** follows the corrected contract formula. The fuel factor is mass-weighted.
- **Burning:**
  - The bed is a compact pile of envelope V = Σ m_s / (φ_s ρ_s), with packing φ: wood 0.4, fibre 0.07, charcoal
    0.55, others 1.
  - Flaming fuels regress at 0.50 mm/min x x_O2/0.21 x (1 + air) (EN 1995-1-2, hardwood).
  - Glowing char burns at the diffusion limit, about 0.1 mm/min.
  - Wood burns in two stages: flaming leaves char with the `wood_to_charcoal` yield (0.246), and the char then
    glows away. The total heat is the wood's heating value.
- **Fanning.** Forced air acts only for the first `FAN_S` = 3,600 s of a call. Then it resets to 0.
- **Items in the bed** (within 0.5 m of a fire):
  - **Heating.** They relax toward T_fire with τ = m cp / (h A) + r² / (π² α), where h = 10 W/m²K plus the
    radiative term at emissivity 0.9.
  - **Heat budget.** All items in one fire take at most `LOAD_SHARE` = 0.3 of the heat its bed released in that
    sub-step. Where they need more, heating and transform progress scale down.
  - **Transforms** run at (basis kg + main product kg / yield) / days per day, for the time spent above
    `min_temp_k`. The atmosphere comes from `materials.atmosphere_ok`. The charcoal reductant is taken from the
    item, then from the bed.
  - **Smothered charcoal** is protected from ignition.
  - **Meat, hide and bone above 573 K** dry, char and shrink at the flaming rate. Their mass goes to `to_soil_kg`
    and `charred_kg`.
  - **Fuel items** above their ignition point move into the bed and burn, unless their own transform applies.
- **Outputs:** air exchange `air_kg [W, 3]` (O2, CO2, H2O), `heat_j`, `item_heat_j`, `burnt_kg`,
  `transformed_kg`, `to_soil_kg`, `charred_kg`, `mean_hrr_w` and `fire_pos`.

**Tool physics:**

| Function | Formula |
|---|---|
| `strike_energy` | ½ m_head (v_arm (1 + L / 0.6 m))², v_arm 5 m/s. A bare limb strikes with 0.022 M at hardness 2.5. |
| `strike_damage` | E x min(1, H_tool / H_target) x (1 + 2 sharp) / (1 + K_target / 1 MPa m^0.5), in J |
| `dig_yield` | min(1, H_tool / H_target) |
| `chop_yield` | dig_yield x (0.25 + 0.75 sharp) |
| `butcher_yield` | 0.3 + 0.7 x sharp x min(1, H_tool / H_bone) |
| `working_hardness` | For an assembly: the species of at least half the head mass whose mass is closest to the head mass. |
| `warmth_within` | 0.3 HRR / (4π d²), d ≥ 1 m, within 20 m (W/m²) |
| `worn_insulation` | +1 x min(1, hide / (0.06 M)) x min(1, fibre / (0.05 hide)) |

**Decay.** Daily rates at 288 K:

| Item | Rate |
|---|---|
| meat | 0.1 /day |
| fat | 0.03 /day |
| hide | 0.05 /day |
| bone | 0.1 /yr |
| wood | 0.1 /yr |
| fibre | 0.3 /yr |

- Q10 is 2, capped at 310.15 K.
- There is no decay below 263.15 K or above 333.15 K.
- The carbon goes to the item's cell as litter.

**Key values:**
- **Fire temperatures** at x_O2 0.21 and T_air 288 K:
  - open wood 1,100 K
  - open charcoal 1,384 K
  - charcoal at air 1/3: 1,494 K
  - fully fanned charcoal: 1,713 K
  - wood at x_O2 0.15: 974 K
- **Fire lifetimes:**
  - charcoal 3, 10 and 20 kg: 11, 17 and 22 h
  - wood 2, 7 and 30 kg: 7, 11 and 19 h
  - 1 kg of loose fibre: under 1 h
- **A small fire cannot do big work.** A 50 g charcoal fire under a 100 kg limestone boulder heats it to only
  292 K and calcines nothing.
- **Ceramic.** Clay in a 10 kg charcoal fire becomes ceramic within a day. An open wood fire alone (1,100 K) does
  not fire clay; its embers, fanning or charcoal do.
- **Copper.** Malachite in a 1,384 K charcoal bed for 6 h: more than half becomes copper. Without charcoal,
  none does.
- **Element closure** through every transform: below 2e-5 kg.
- **Speed.** `fire_step` takes 0.63 s on the CPU for W = 4, I = 4,096, F = 256.

**Deviations, with reasons:**
- **ΔT0 = 812 K** (corrected in the contract).
- **Two-stage wood combustion.** It gives the ember beds that make wood fires last hours and makes them reducing.
- **`collect` also picks up loose items.** Otherwise nothing could take an item back from a fire, a knapping flake
  or a dropped tool. The contract's 14 actions have no pick-up.
- **Raking and the 333 K handling limit.** Without raking, smothered charcoal could never be recovered.
- **The fire's heat budget** (`LOAD_SHARE`). Without it, a 50 g fire could calcine a 100 kg boulder: energy was
  created.
- **Fanning lasts one hour.** One action used to make a forge for the whole day.
- **Tissue chars to the soil**, not to char plus gases. No species holds bone's Ca and P, and the air ledger has
  no N or S. The volatiles reach the air later, through the soil carbon.
- **Cooking happens at roasting distance.** In the bed, meat reached 1,208 K and kept its full value.
- **Transforms progress continuously** with time above the onset. `hot_days` is a diagnostic.
- **The burn rate is 0.50 mm/min** (hardwood: the wood species is oak), not the softwood 0.65.

**`new_rule` list:**
- reach and geometry: `REACH_M` 5, `FIRE_RADIUS_M` 0.5, `NEAR_FIRE_M` 1
- fire: `FIRE_DT0_K` 812, `AIR_GAIN` 0.3, `FUEL_FACTOR`, `MIN_FIRE_KG` 0.01, `PACKING`, `LOAD_SHARE` 0.3,
  `FAN_S` 3,600, `ROAST_SHARE` 0.2
- ignition: `FRICTION_EFFICIENCY` 0.1, `DRILL_RADIUS_M` 0.005, `PYRITE_MIN` 0.5, `SPARK_TINDER_K` 550
- knapping: `BRITTLE_MIN` 0.5, `KNAP_LOSS` 0.2, `KNAP_EDGE_SHARE` 0.05, `K_EDGE` 2
- assemblies: `HEAD_CORE` 0.5, `BOND_FIT` 0.2, `BINDER_SHARE` 0.05, `HANDLE_ASPECT` 15, `K_ROD` 2
- tools: `V_ARM` 5, `CUT_GAIN` 2, `K_SOFT` 1, `BUTCHER_BASE` 0.3, `CHOP_BLUNT` 0.25, `CARRY_FRACTION` 0.1,
  `ARM_POWER_W` 75, `ACTION_S` 3,600
- garment: `GARMENT_INSULATION` 1, `FIBER_PER_HIDE` 0.05
- most decay rates
- `PICKUP_RULE`, `WOOD_CHAR_RULE`, `TISSUE_CHAR_RULE`

**Open issues:**
- **Fires and the 1-day tick.** A fire made on day d is usually gone by day d + 1 unless it holds about 25 kg of
  fuel or more.
  - Charcoal making needs the item raked out on a day boundary while the bed is still deep. The test uses a
    60 kg bed.
  - `world.py` must order the day so that `heat_item`, `feed_fire` and `cook` see the fires made that tick.
- **`cook`'s budget** is not deducted from `fire_step`'s budget for the same hour.
- **Raked items stay hot.** A raked item lands at the actor's feet. If the actor stands within 0.5 m of a fire,
  the item is still in it.
- **Transform heat does not cool the fire.** The fire's temperature is the calibrated formula.
- **Wood carries ash in from outside.** Wood collected from plants carries 1 % carbonate ash (Ca, K) that the
  biosphere does not hold. The item ledger gains it.
- **No head marker.** Knapping an assembly whose brittle species is the handle knaps the handle; items carry no
  head marker.
- **Wanted from the owners of other modules:** a materials transform for tissue pyrolysis, a stored head
  hardness and a per-fire fan-time field.
- **Citations from memory**, so the figures are approximate:
  - Kofman 2010 (wood packing)
  - MacCarty, Still & Ogle 2010 (the 10-20 % pot share behind `LOAD_SHARE`)
- **Estimates.** Many tool and decay numbers are tagged estimates.

### 2.12 `world.py`, `__main__.py`, `viewer_globe.html`
- `PlanetWorld(config, seeds: list[int], device)`:
  - builds the chain inputs, the specs and the globe, then terrain, deposits, climate (spun up), biosphere
    (spun up) and founders
  - `step()` runs one day: climate, biosphere, observe, think, act, crafting, physiology, deaths to corpses,
    plasticity, births, decay
  - also `summary()`, `ledgers()` (each with its error), `state_dict()` / `from_state()` and `frame(w)` (a
    globe frame for the viewer)
- `python -m haishool.life9.planet describe --seed 781` prints the PlanetSpec with its provenance.
  `run --seeds 781,85 --days N --device cuda --out DIR [--view-every K]` and `bench` are the other commands.
- The viewer is a three.js globe: terrain relief (exaggeration labelled), the ocean, ice, vegetation colour,
  day/night or substellar shading, individuals, fires, items and calls, with a timeline.

#### As built
`world.py`, `__main__.py` and `view.py` are being written in parallel. The viewer module is `view.py`, not
`viewer_globe.html`. This record does not describe them.

The modules below leave these duties to them:

| From | `world.py` must |
|---|---|
| formation | Report the share of CO2-capped worlds in `summary()`. Merge every module's provenance into the export. |
| globe | Make the habitat water with `habitat_water_volume(spec.ocean_mass_kg, spec.radius_m, spec.gravity_m_s2, spec.relief_m, habitat_radius_m)`. Pass `report=` to `make_deposits` and report the support levels. Re-solve the sea level with `sea_level` if the ocean store moves. |
| climate and biosphere | Each day call `climate.step(..., plant_cover=biosphere.cover(bio, B))`, then `biosphere.step`. Route creature water through `take_water` and `give_water`, and gases through `add_gas`. |
| creatures | Route `faeces_c` and `litter_c` to litter at the cell. Route `water_loss_kg` and the deaths' `water_kg` to vapour or soil. Subtract `soil_taken_kg_m2` from the soil. Send `o2_mol` and `co2_mol` to the gas column. Pass `t_air_k` to `drink`, or watered animals above 310 K dry out. Call `injure` before `physiology` on the same day. Write `cr.loud` and `cr.calls` from the brain each tick. Count the water held in food items as a water store. |
| brain | Pass `invested_j` to `outcome`. Call `install(..., baseline=...)`. Keep any cached width ≥ max(k) after every install. Set hidden 128 and `initial_hidden` 48. |
| crafting | Book `air_kg` and `to_soil_kg` of `cook` and `fire_step` at the fire's cell. Map the brain's `feed_fire` to `fan`. Keep held items' positions current (`sync_held`). Book the 1 % wood ash that enters items. |
| crafting and creatures | **Three conversions are not defined anywhere yet.** They need a stated rule: `strike_damage` (J) to `injure`'s health units; fire warmth (`warmth_within`, W/m²) to `physiology`'s `heat_k` (K); and `butcher_yield`, which no eating function uses yet. |

**Integration round (world review fixes).** `world.py` states its rules in `WORLD_PROVENANCE`; the ones that
change the day:

- **Order:** observe, think, then every action where the individual looked (crafting, strikes, eating,
  drinking), fire_step and decay, and the day's travel last; physiology and deaths at the night's position.
  Before, everyone moved 5-10 km before acting, so the senses said nothing about the cell acted on.
- **Time budget** (`world.time_budget`): a 12 h active day. The focus action takes an hour
  (`crafting.ACTION_S`) unless it is forage, drink or rest; travel takes speed x the travel time. Everyone but
  the resters grazes for the time left (forage: the whole day). `rest` stays put: no travel, no grazing.
  One action a day had been an energy trap: a forage day gives at most 13.9 MJ against 5-7 MJ of costs.
- **Water:** grazing brings the forage's water (`creatures.FORAGE_WATER_KG` = 3 kg per kg dry) from the cell's
  soil; everyone at fresh water drinks each day (`world.background_drink`); respiration makes metabolic water
  (2 (O2 − CO2) mol, `creatures` PROVENANCE `metabolic_water`); water above the norm is excreted.
- **Travel** (`world.travel`): the day's travel time lets the top economical speed cover twice Garland's mean
  daily movement 1.04 M^0.25 km (about 23 min at 30 kg); land animals stop at the shore.
- **Founders** start on habitable cells (land, drinkable soil, air between their lower critical temperature
  and 310 K).
- **Reach:** strike, share, eat_meat and the fire actions first walk to the nearest target the actor sees.
- **Settlement:** float64 per-cell carries replace the settlement with the global ocean and CO2 stores.
- **Ledgers:** the deposit stock is float64 and a ledger of its own (`deposits`); `oxygen_mobile` compares the
  climate's record with the creatures' own O2 ledger; `oxygen_atoms` checks the O of respired O2 against
  the CO2 of oxidised fat plus metabolic water. The aggregates `carbon`, `oxygen` and `water` are for
  information only, since the air's columns and the ocean dominate their scales.
- **Gates:** `world.gates()` checks the ledgers (1e-5), founders after 30 days (half alive) and newborns (half
  live 30 days).

Measured at G = 12 with 128 founders and 256 slots per world (all 14 actions, 200 days):
- The synthetic planets 13 and 7 and the Earth twin fill their 256 slots.
- On 781, 87 of 128 founders + 128 born are alive at day 200 (30 founders). About half of its founders die
  of thirst in the first 30 days, because only about 5 % of its land is both wet and mild.
- Newborns survive 30 days at 0.58-0.98.
- At G = 48 (781 and 85, 256 founders, short spin-ups), 57 % and 79 % of the founders are alive at day 40.

**Open:**
- Lactation or other parental provisioning, which belongs to the spec owner.
- `share` still finds relatives by pedigree, not by perception.
- Fire residues and cooking budgets are per world, not per fire.
- Evolution needs runs of more than 1,000 days. Maturity is 549 days at 30 kg, gestation 152 days and learning
  about 1e-3 per day.

## 3. Values expected for planet 781 (tests)

| Quantity | Expected |
|---|---|
| star | 0.617 solar masses; L 0.1449 L_sun; born 3.5026e9 yr after the Big Bang; lifetime 4.26e10 yr |
| planet | 0.609 Earth masses at 0.40488 au; flux 0.884; t_eq 246.9 K; greenhouse 40.8 K; T_s 288 K; water liquid |
| cloud (mass fractions) | O 1.50e-3, C 4.58e-4, Fe 1.87e-4, Mg 1.82e-4, Si 1.47e-4, N 1.37e-4, Z 3.05e-3 |
| biosphere O2 | 0.118 PAL, about 2.5 kPa. This is below the fire limit, so 781 cannot make fire until oxygen rises. That is a real outcome, not a bug. |
| year | about 119.8 days |
| tidal lock | locked (lock radius about 0.747 au) *(corrected v3: was 0.67 au, which assumed a 4.5 Gyr star; this star is 8.71 Gyr old when the habitat starts)* |
| core mass fraction | about 0.233 |

### As built (values from the code)

| Quantity | Value |
|---|---|
| star | 1.2268e30 kg (0.617 M_sun); L 5.548e25 W (0.144924 L_sun); age at habitat start 8.71e9 yr; R 4.728e8 m; T_eff 4,320 K; PAR fraction 0.247 (Sun 0.366) |
| orbit | 0.404877 au; year 119.80 d; insolation 1,203.2 W/m²; locked, lock radius 0.747 au |
| interior | core fraction 0.2331; compression 1.247; radius 5,674.4 km; core radius 2,742.8 km; mean density 4,755 kg/m³; g 7.544 m/s²; escape velocity 9,253 m/s |
| water | ocean 8.371e20 kg (global layer 2,069 m); vapour scale height 3,054 m; vapour column 29.23 kg/m² |
| air | P 88,971 Pa: N2 79,119, O2 2,505, CO2 5,131, Ar 946, H2O 1,271 Pa; x_O2 0.0282 (fire limit 0.15); μ 0.029033 kg/mol; cp 1,018.4 J/(kg K); γ 1.3912; scale height 10.92 km; ρ 1.080 kg/m³; c 338.5 m/s; M_atm 4.716e18 kg |
| climate | albedo 0.30; T_eq 246.86 K (chain 246.87); target T_s 287.672 K; τ 1.1256; co2_ref 5,130.5 Pa, not capped; tau_per_ln_co2 0.06248 |
| surface | relief 780.2 m; radiogenic 0.00543 W/m² |
| founders | aerobic capacity a = 0.455 at 2.5 kPa O2 |

## 4. Gates (tests and `world.gates()`)

- `check_spec` passes for 781 and for 20 synthetic planets.
- **Climate without life:** the global mean reaches the chain's T_s within 3 K after spin-up, the water ledger
  closes, and the scheme is stable for 1,000 days.
- **Biosphere:** NPP is positive where there is light, water and warmth, and zero on the dark side of a locked
  planet. The carbon and O2 ledgers close.
- **Creatures:** a fed, watered individual in mild weather keeps its reserve. Starvation, dehydration, cold and
  age kill in the right order of time. Energy is conserved between food, reserve, costs, offspring and corpses.
- **Crafting:**
  - Knapping flint with basalt raises sharpness.
  - Knapping basalt with wood does nothing.
  - Clay heated above 1,150 K long enough becomes ceramic.
  - Malachite with charcoal above 1,000 K becomes copper; without charcoal it does not.
  - No fire is possible at an O2 mole fraction below 0.15.
  - Mass and elements balance through every transform.
- **Determinism:** same seed, same hash on the CPU; exact resume on the CPU.
- **GPU:** CPU and GPU statistics agree within noise (`scripts/life9_bench_grid.py` style).

### As built (gate status)

| Gate | Status | Evidence |
|---|---|---|
| `check_spec` | passes | 781, Earth twin, synthetic 0-39 (tests) and 0-299 (review), all 26 habitable world7 seeds in 12-59 |
| climate reaches T_s within 3 K | passes | `test_gate_locked_planets_...`, `test_gate_rotating_planets_...`; 781 at 0.016 K (G = 24); the rotating fixtures within 1 K |
| water ledger, 1,000 days stable | passes | `test_gate_stable_for_1000_days_...` (locked) and `test_gate_rotating_stable_for_1000_days_...` (Earth and synthetic 7; Earth also weathers what it outgasses) |
| NPP positive where warm, wet and lit | passes | `test_gate_npp_positive_...`. The test selects 280-300 K, soil > 75 and shortwave > 150; at 300-310 K with a dry bucket a seedling's maintenance correctly outgrows GPP on dim days. |
| NPP zero on the night side | passes | `test_gate_npp_zero_on_the_night_side_...` |
| carbon and O2 ledgers | pass | `test_gate_carbon_and_oxygen_ledgers_close` |
| creatures keep their reserve; death order | passes | `test_fed_and_watered_...`, `test_dehydration_cold_starvation_and_age_kill_in_that_order` (2 d < 15 d < 31 d < lifespan) |
| creature energy conserved | passes | `test_energy_carbon_and_water_ledgers_close_through_a_busy_run` |
| crafting gates (6) | pass | `test_flint_knapped_by_basalt_sharpens`, `test_basalt_knapped_by_wood_does_nothing`, `test_clay_above_1150_k_...`, `test_malachite_with_charcoal_...`, `test_no_fire_below_the_oxygen_limit`, `test_every_transform_runs_and_balances_...` |
| CPU determinism and resume | passes | every module has a determinism and exact-resume test |
| GPU agrees within noise | **not run** | no GPU was used in this phase; the CUDA tests skip on the CPU |
| `world.gates()` | built: ledgers, founders after 30 days, newborns 30 days | `test_founders_are_viable_for_thirty_days` (G = 12, 781 and synthetic 13: 13 passes the gate; 781 keeps about half of its founders, near the gate) |

## 5. Known limitations

These are true of the code today. Some are choices, some are gaps.

**Climate**
1. **The equator-to-pole contrast is too small.** Earth's annual mean is 297.8 K at the equator and 266-271 K
   poleward of 60°: a contrast of 27-32 K. The observed contrast is 40-60 K.
   - Cloud amount is uniform.
   - Sea ice is instantaneous and melts every summer.
   - Only the ocean albedo depends on the sun angle.
   - The global calibration hides this.
2. **Land runoff is low.** The model's Earth runs off 75 mm/yr from land, against the observed 251 mm/yr. Land
   evaporation is potential evaporation without stomatal control. `runoff0` is calibrated to the model's own
   Earth (22 mm/yr), so the carbonate balance is right relative to the model, not to Earth.
3. **The climate is less sensitive than Earth's.** The grey model gives 781 1.71 K per CO2 doubling with the
   water term, and 1.27 K Planck-only. Earth's is about 3 K.
4. **No diurnal cycle.** Insolation is the daily mean. A rotating planet has no night within a tick, for plants
   or for eyes. Only a locked planet's dark side and polar night are dark.
5. **The CO2 cap is the common case.** It applies on 72 % of synthetic planets and 81 % of habitable world7
   planets in seeds 12-59.
   - There, `co2_ref_pa` is the 1e4 Pa cap, not a calibration.
   - Part of τ belongs to no named gas.
   - The plants' CO2 factor is 2.07, against 1 at Earth.
6. **Snow** has no sublimation and no latent heat of fusion. Deep snow leaves by a glacier rule.
7. **Sea level does not move** as the ocean store changes. There are no lakes or rivers: drinking is from soil
   ≥ 50 kg/m², or from the sea for swimmers.

**Carbon and life**

8. **There is no ocean carbon buffer.** Any biosphere imbalance (grazing, fires, growth) reaches the air one to
   one. Earth's CO2 drifts about 1 % per year after spin-up. 781 does not, because its CO2 column is about 240
   times Earth's.
9. **Plants do not disperse.** Only a 1 g C/m² seed floor keeps lit land alive.
   - A stripped cell takes years to regrow.
   - A locked planet's night side stays bare.
   - One fixed 39 % wood allocation lets dry marginal cells boom and bust over decades.
10. **The founders' body is a choice, not a chain value.** The 30 kg adult mass is `new_rule`. The diet (0.034)
    is a proxy, about a third of the predators' share among consumers (0.10), because `chain.py` does not export
    the guild biomasses.

**Planet 781 in particular**

11. **781 cannot make fire.** Its O2 is 0.118 PAL: a mole fraction of 0.028, against the 0.15 limit. So on 781
    there is no cooking, charcoal, ceramic, lime, glass or metal. Its crafting is limited to collecting,
    knapping, combining, dropping and sharing.
    - Its aerobic capacity of 0.455 roughly halves top speed and the effort of every action.
    - This is the chain's real outcome, not a bug.
    - It means 781 cannot exercise the fire half of the engine. Use other worlds for that.
12. **Ores are rare.** For 781 malachite is 1e-5 and cassiterite 7e-7 of the deposit mass, concentrated in
    patches. Bronze is practically out of reach on low-metallicity worlds.

**Habitat scale**

13. **Sight is short and travel is long.** Sight reaches about 223 m between 30 kg animals on level ground. A day's
    travel can reach 24.6 km, 0.4 of the habitat's circumference. Sight and climbing see a one-cell terrain
    feature at about a quarter of its height.
14. **Fires and days do not fit well.** Most fires burn out within a day, while actions happen once a day. Making
    charcoal needs a deep bed and raking on a day boundary.

**Integration gaps** (until `world.py` lands)

15. **Three conversions have no rule yet:**
    - strike energy (J) to health
    - fire warmth (W/m²) to a temperature shift (K)
    - butchering yield, which is not wired into eating
16. **Two water and element flows need world.py:** the water held in food items must become a store of the
    world's water ledger, and the 1 % wood ash entering items from plants must be booked.
17. **One ledger across all modules** does not exist yet. Each module closes its own.

**Evidence quality**

18. **Some citations were written from memory and are flagged in the code:**
    - Byrne & Goldblatt 2014
    - Dai 2006
    - Turcotte & Schubert 2002 (isotope heats)
    - Stacey & Davis 2008
    - Harris et al. 2014 (shelf share)
    - Kofman 2010 (wood packing)
    - MacCarty, Still & Ogle 2010 (`LOAD_SHARE`)
19. **Many numbers are estimates, tagged as such:**
    - toughness of ores and tissues
    - transform durations
    - decay rates
    - knapping and assembly constants
    - packing of fibre and charcoal
    - the 1,500 K glass onset, which has no liquidus citation
20. **Planet 781's radius** (0.61 M_E) rests on an extrapolation of the compression rule, checked only against
    Venus and Mars.

**Hardware and cost**

21. **Nothing has run on a GPU.** On CUDA or ROCm, `index_add_` and scatters are not bit-reproducible, and
    `torch.multinomial` differs. Host syncs remain:
    - `int(k.max())` in `think`, unless `width` is passed
    - `float(kappa.max())` in `globe.diffuse`
22. **Cost at full size:**
    - The climate takes about 0.27 s per day at G = 48, W = 16 on the CPU, and the diffusion sub-steps grow as G².
    - `senses` builds dense `[W, N, N]` pair tensors.
    - Genome plus live weights take about 3.9 GB in float32 at 16 x 1,024.
23. **A few rules are coarse:**
    - Evaporative cooling switches on as a step at 310 K.
    - The lower critical temperature does not depend on body mass.
    - Age death is deterministic, with no hazard curve.
