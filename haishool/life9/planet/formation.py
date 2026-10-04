"""From the chain to a planet: :func:`build_planet` turns chain inputs into a :class:`PlanetSpec`
(PLANET-SPEC section 2.3). CPU, standard library plus NumPy; ``tensors`` hands the result to torch.

Every field of a spec has a provenance entry ``(tag, note)``. A tag is one of ``chain``,
``derived``, ``reference``, ``new_rule`` or a ``+`` join of them when a field combines several
(``derived+new_rule``: a physical formula applied under a stated modelling choice).

The rules, in order (SI units throughout):

star       mass, luminosity, age from the chain; radius R = R_sun M^0.8 (M < 1; M^0.57 above,
           new_rule); T_eff = T_sun (L / R^2)^(1/4); PAR share of the black body (400-700 nm,
           Gauss-Legendre integral of the Planck function).
orbit      chain semi-major axis; year from Kepler's third law with G M_star; insolation S0 x flux.
interior   all Fe to a metal core, Mg and Si to MgO + SiO2 (new_rule): core fraction
           Fe / (Fe + MgO + SiO2) by mass from the cloud. Uncompressed density from iron 7870 and
           silicate 3300 kg/m^3; self-compression c = rho / rho0 from Murnaghan's equation of
           state at a characteristic pressure P ~ G M^(2/3) rho^(4/3) with K' = 4, calibrated so
           that Earth gets its own c_E (Earth's solid density over its uncompressed density;
           new_rule). The core takes the same factor. The ocean is a liquid-water shell on the
           solid sphere. One radius rule everywhere: g = G M / R^2, v_esc = sqrt(2 G M / R).
atmosphere N2 and Ar at Earth's (dry) partial pressures x volatile scale x (g / g_E)^k (k = 0 by
           default); O2 = chain oxygen_pal x Earth's O2; CO2 from the greenhouse calibration; H2O at
           the surface = relative humidity x Magnus e_sat(T_s). P = sum (the surface air: mu, cp,
           gamma, rho, sound speed). Water vapour is not well mixed: its column is the surface vapour
           density x the vapour scale height H_w = R_v T^2 / (L_v Gamma) of a troposphere with lapse
           rate Gamma = 6.5 K/km x g / g_E (about 2.3 km and 22 kg/m^2 for Earth, observed 24.6).
           M_atm = (P - p_H2O) 4 pi R^2 / g + vapour column x 4 pi R^2 (iterated with the radius);
           the vapour comes out of the planet's water, the rest is the ocean.
climate    albedo 0.30 (world7's); T_eq = (S (1 - A) / (4 sigma))^(1/4); grey atmosphere
           T_s = T_eq (1 + 0.75 tau)^(1/4) solved for tau at the chain's t_eq + greenhouse (t_eq of
           the era flux, so the target is exactly what world7 rounds to its whole-K T_s).
           CO2: the forcing that separates this tau from Earth's (both at this planet's T_s) is
           read as CO2 forcing 5.35 ln(p / p_E) W/m^2 (Myhre et al. 1998) around Earth's
           pre-industrial 278 ppm: world7's greenhouse rule is the carbonate-silicate thermostat
           (haishool/cosmos/planets.py greenhouse), so the excess greenhouse is CO2 (new_rule),
           up to 1e4 Pa, where the logarithmic law stops being a fit. Beyond it (``co2_capped``,
           the common case: 216 of 300 synthetic planets, 21 of the 26 habitable world7 planets of
           seeds 12-59; 781 is not capped) tau still holds the chain's T_s and ``tau_residual`` is
           the part of tau of no named gas. A capped co2_ref_pa is a stated limit, not a
           calibration: biosphere's f_CO2 = (p / (p + 30)) / (28 / 58) is 2.07 there (1 at Earth).
rotation   Kasting, Whitmire & Reynolds 1993 (Icarus 101, 108) lock radius
           a_lock = 0.027 (P0 t / Q)^(1/6) M^(1/3) au (P0 13.5 h, Q 100, t the star's age in yr,
           M in solar masses). Inside it: rotation = year, obliquity 0, day infinite. Outside:
           rotation period and obliquity drawn on the formation RNG (new_rule ranges).
surface    relief 600 m x 9.81 / g (new_rule); crust deposit mix (``materials.crust_fractions``);
           radiogenic heat flux = Earth's 0.047 W/m^2 x Z / Z_sun x the decay of K-40, U-238,
           U-235, Th-232 since formation (relative to Earth's 4.567 Gyr) x silicate mass per area
           relative to Earth.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np
import torch

from . import constants as C
from . import materials

FORMATION_VERSION = "life9-formation-v2"
TAGS = ("chain", "derived", "reference", "new_rule")
GASES = ("N2", "O2", "CO2", "Ar", "H2O")
#: salt of the formation RNG (rotation and obliquity of planets that are not locked)
FORMATION_SALT = 0x666F726D  # "form"


@dataclass(frozen=True)
class FormationRules:
    """The modelling choices (new_rule) of :func:`build_planet`, with the spec's defaults."""
    water_mass_fraction: float = C.EARTH_OCEAN_MASS_FRACTION  # (ocean + vapour) / planet mass (Earth's ocean value)
    water_metal_exponent: float = 0.0       # x (Z / Z_sun)^this
    volatile_scale: float = 1.0             # N2 and Ar relative to Earth's partial pressures
    volatile_g_exponent: float = 0.0        # x (g / g_E)^this (1 would be Earth's column per area)
    relative_humidity: float = 0.77         # near-surface mean (Earth about 0.75-0.8: Dai 2006, J. Climate 19, 3589)
    lapse_rate_k_m: float = C.EARTH_LAPSE_RATE  # tropospheric lapse rate at g_E, x g / g_E (vapour scale height)
    relief_m: float = 600.0                 # habitat relief at g = relief_g
    relief_g: float = 9.81
    lock_p0_h: float = 13.5                 # Kasting et al. 1993: initial rotation period
    lock_q: float = 100.0                   # Kasting et al. 1993: tidal dissipation factor
    rotation_h: tuple = (12.0, 48.0)        # log-uniform draw for planets that are not locked
    obliquity_deg: tuple = (0.0, 35.0)      # uniform draw for planets that are not locked
    par_nm: tuple = (400.0, 700.0)          # photosynthetically active band (McCree 1972)
    ms_radius_exponents: tuple = (0.8, 0.57)  # main-sequence R ~ M^a below, M^b above 1 M_sun
    k_prime: float = 4.0                    # Murnaghan dK/dP of the compression (reference)
    co2_max_pa: float = 1.0e4               # the CO2 calibration stops here (the log law's domain)


# ---------------------------------------------------------------------------------------------
# formulas


def main_sequence_radius_m(mass_msun: float, exponents=(0.8, 0.57)) -> float:
    """Main-sequence radius: R_sun M^0.8 below one solar mass, M^0.57 above (textbook
    approximation, continuous at 1; new_rule)."""
    a, b = exponents
    return C.R_SUN * mass_msun ** (a if mass_msun < 1.0 else b)


def effective_temperature_k(luminosity_w: float, radius_m: float) -> float:
    """Stefan-Boltzmann in solar units: T_sun (L / R^2)^(1/4)."""
    return C.T_SUN * ((luminosity_w / C.L_SUN) / (radius_m / C.R_SUN) ** 2) ** 0.25


def planck_lambda(lam_m, t_k):
    """Spectral radiance B_lambda(T) in W m^-2 sr^-1 m^-1."""
    lam = np.asarray(lam_m, dtype=np.float64)
    x = C.H_PLANCK * C.C_LIGHT / (lam * C.K_B * t_k)
    return 2.0 * C.H_PLANCK * C.C_LIGHT ** 2 / lam ** 5 / np.expm1(x)


def band_fraction(t_k: float, lo_m: float, hi_m: float, nodes: int = 64) -> float:
    """Share of a black body's radiance between two wavelengths: the Planck integral
    (Gauss-Legendre, ``nodes`` points) divided by sigma T^4 / pi."""
    x, w = np.polynomial.legendre.leggauss(nodes)
    lam = 0.5 * (hi_m - lo_m) * x + 0.5 * (hi_m + lo_m)
    radiance = 0.5 * (hi_m - lo_m) * float(np.sum(w * planck_lambda(lam, t_k)))
    return radiance / (C.SIGMA * t_k ** 4 / math.pi)


def par_fraction(t_k: float, band_nm=(400.0, 700.0)) -> float:
    """Photosynthetically active share (400-700 nm) of a black body's energy flux."""
    return band_fraction(t_k, band_nm[0] * 1e-9, band_nm[1] * 1e-9)


def kepler_period_s(orbit_m: float, star_mass_kg: float) -> float:
    return 2.0 * math.pi * math.sqrt(orbit_m ** 3 / (C.G * star_mass_kg))


def lock_radius_au(star_mass_msun: float, age_yr: float, p0_h: float = 13.5, q: float = 100.0) -> float:
    """Tidal-lock radius of Kasting, Whitmire & Reynolds 1993 (Icarus 101, 108, section 7):
    0.027 (P0 t / Q)^(1/6) M^(1/3) au, P0 in hours, t in years, M in solar masses."""
    return 0.027 * (p0_h * age_yr / q) ** (1.0 / 6.0) * star_mass_msun ** (1.0 / 3.0)


def equilibrium_temperature_k(insolation_w_m2: float, albedo: float) -> float:
    return (insolation_w_m2 * (1.0 - albedo) / (4.0 * C.SIGMA)) ** 0.25


def grey_tau(t_surface_k: float, t_eq_k: float) -> float:
    """Grey-atmosphere optical depth with T_s = T_eq (1 + 0.75 tau)^(1/4)."""
    return ((t_surface_k / t_eq_k) ** 4 - 1.0) / 0.75


def grey_surface_k(t_eq_k: float, tau: float) -> float:
    return t_eq_k * (1.0 + 0.75 * tau) ** 0.25


def e_sat_pa(t_k: float) -> float:
    """Saturation vapour pressure over water, Magnus form (Alduchov & Eskridge 1996, J. Appl.
    Meteor. 35, 601): 610.94 Pa exp(17.625 t / (t + 243.04)), t in degrees C."""
    t = t_k - 273.15
    return 610.94 * math.exp(17.625 * t / (t + 243.04))


def vapour_scale_height_m(t_k: float, lapse_k_m: float) -> float:
    """e-folding height of water vapour in a troposphere of lapse rate Gamma: Clausius-Clapeyron,
    d ln e_sat / dT = L_v / (R_v T^2), with dT/dz = -Gamma gives H_w = R_v T^2 / (L_v Gamma)
    (constant relative humidity; e.g. Pierrehumbert 2010, Principles of Planetary Climate, 2.4)."""
    r_v = C.R_GAS / C.molar_mass("H2O")
    return r_v * t_k ** 2 / (C.L_VAPORIZATION * lapse_k_m)


def vapour_column_kg_m2(p_h2o_pa: float, t_k: float, scale_height_m: float) -> float:
    """Column water vapour of an exponential profile: surface vapour density p M_w / (R T) x H_w."""
    return p_h2o_pa * C.molar_mass("H2O") / (C.R_GAS * t_k) * scale_height_m


def earth_tau() -> float:
    """Grey optical depth of world7's Earth (254.6 K + 33 K at flux 1, albedo 0.30): the CO2
    calibration's reference point."""
    from haishool.cosmos import planets as level3
    return grey_tau(level3.T_EQ_EARTH + level3.GREENHOUSE_EARTH, equilibrium_temperature_k(C.S0, C.EARTH_ALBEDO))


def co2_tau(p_co2_pa: float, t_surface_k: float) -> float:
    """The tau that Earth's tau plus the CO2 forcing 5.35 ln(p / p_E) W/m^2 (Myhre et al. 1998)
    gives at surface temperature T_s: sigma T_s^4 [1/(1+0.75 tau_E) - 1/(1+0.75 tau)] = forcing."""
    p_e = C.EARTH_AIR_MOLE_FRACTION["CO2"] * C.P_STANDARD
    inv = 1.0 / (1.0 + 0.75 * earth_tau()) - 5.35 * math.log(p_co2_pa / p_e) / (C.SIGMA * t_surface_k ** 4)
    return (1.0 / inv - 1.0) / 0.75


def uncompressed_density(core_mass_fraction: float) -> float:
    """Mass-weighted volumes of iron and silicate at zero pressure."""
    return 1.0 / (core_mass_fraction / C.RHO_IRON + (1.0 - core_mass_fraction) / C.RHO_SILICATE)


def earth_compression() -> float:
    """Earth's solid-body density over its uncompressed density (the calibration, new_rule)."""
    ocean = C.EARTH_OCEAN_MASS_FRACTION * C.M_EARTH
    solid_volume = 4.0 / 3.0 * math.pi * C.R_EARTH ** 3 - ocean / C.RHO_WATER
    return (C.M_EARTH - ocean) / solid_volume / uncompressed_density(C.EARTH_CORE_MASS_FRACTION)


def compression(mass_earth: float, rho0: float | None = None, k_prime: float = 4.0) -> float:
    """Self-compression factor c = rho / rho0 of a rocky planet, calibrated on Earth (new_rule).

    Murnaghan's equation of state at a characteristic interior pressure: c = (1 + K' P / K0)^(1/K')
    with P / K0 = x_E (M / M_E)^(2/3) (rho0 c / (rho0_E c_E))^(4/3), the hydrostatic scaling
    P ~ G M^(2/3) rho^(4/3). x_E follows from Earth's c_E = (1 + K' x_E)^(1/K'), so K0 and the
    pressure prefactor drop out; K' = 4 is the usual pressure derivative of the bulk modulus of
    silicates and iron (Stacey & Davis 2008, Physics of the Earth). Solved by fixed-point
    iteration. It gives radii within about 2 % of Venus (0.815 M_E) and Mars (0.107 M_E, 3 %), and
    of the Earth-like fit R = (1.07 - 0.21 CMF) M^(1/3.7) of Zeng et al. 2016 (ApJ 819, 127), which
    is stated for 1-8 M_E; below 1 M_E (planet 781 at 0.61 M_E) only Venus and Mars check it."""
    rho0_e = uncompressed_density(C.EARTH_CORE_MASS_FRACTION)
    rho0 = rho0_e if rho0 is None else rho0
    c_e = earth_compression()
    x_e = (c_e ** k_prime - 1.0) / k_prime
    c = 1.0
    for _ in range(200):
        x = x_e * mass_earth ** (2.0 / 3.0) * (rho0 * c / (rho0_e * c_e)) ** (4.0 / 3.0)
        c_next = (1.0 + k_prime * x) ** (1.0 / k_prime)
        if abs(c_next - c) < 1e-15:
            return c_next
        c = c_next
    return c


def core_mass_fraction(cloud: dict) -> float:
    """All Fe to the metal core, Mg and Si to MgO + SiO2: Fe / (Fe + MgO + SiO2) by mass."""
    fe, mg, si = cloud["iron"], cloud["magnesium"], cloud["silicon"]
    mgo = mg * C.molar_mass("MgO") / C.element_mass("Mg")
    sio2 = si * C.molar_mass("SiO2") / C.element_mass("Si")
    return fe / (fe + mgo + sio2)


def number_ratio(cloud: dict, a: str, b: str) -> float:
    """Atom ratio N_a / N_b of two cloud elements (names as in the cloud dict)."""
    sym = {"carbon": "C", "nitrogen": "N", "oxygen": "O", "neon": "Ne", "magnesium": "Mg", "silicon": "Si",
           "iron": "Fe", "hydrogen": "H", "helium": "He"}
    return (cloud[a] / C.element_mass(sym[a])) / (cloud[b] / C.element_mass(sym[b]))


def radiogenic_decay_factor(age_yr: float) -> float:
    """Present radiogenic heat of a body of ``age_yr`` relative to Earth's today, for the same
    starting inventory: sum over K-40, U-238, U-235, Th-232 of Earth's present share x
    exp(-ln 2 (age - age_Earth) / half-life)."""
    dt = age_yr - C.EARTH_AGE_YR
    return sum(share * math.exp(-math.log(2.0) * dt / C.HALF_LIFE_YR[iso])
               for iso, share in C.EARTH_RADIOGENIC_SHARE.items())


# ---------------------------------------------------------------------------------------------
# the crust


def crust_fractions(inputs: dict, spec=None) -> dict:
    """The deposit mix of a planet: ``materials.crust_fractions`` (PLANET-SPEC 2.4), mass fractions
    over ``materials.CRUST_SPECIES`` (the [W, S] order globe and crafting index by)."""
    crust = {str(k): float(v) for k, v in materials.crust_fractions(inputs, spec).items()}
    if tuple(crust) != tuple(materials.CRUST_SPECIES):
        raise ValueError(f"crust species {tuple(crust)} differ from materials.CRUST_SPECIES")
    return crust


# ---------------------------------------------------------------------------------------------
# the spec


@dataclass(frozen=True)
class PlanetSpec:
    """One planet, SI units. ``provenance[field] = (tag, note)``."""
    seed: int
    source: str
    # star
    star_mass_kg: float
    star_luminosity_w: float
    star_age_yr: float
    star_birth_yr: float
    star_lifetime_yr: float
    star_radius_m: float
    star_teff_k: float
    par_fraction: float
    # orbit and light
    orbit_m: float
    year_s: float
    flux_earth: float
    insolation_w_m2: float
    # bulk and interior
    metallicity: float
    planet_mass_kg: float
    core_mass_fraction: float
    water_mass_fraction: float
    core_mass_kg: float
    mantle_mass_kg: float
    ocean_mass_kg: float
    atmosphere_mass_kg: float
    uncompressed_density_kg_m3: float
    compression: float
    radius_m: float
    core_radius_m: float
    mean_density_kg_m3: float
    gravity_m_s2: float
    escape_velocity_m_s: float
    ocean_volume_m3: float
    ocean_layer_m: float
    # atmosphere
    oxygen_pal: float
    partial_pressure_pa: dict
    surface_pressure_pa: float
    oxygen_mole_fraction: float
    air_molar_mass_kg_mol: float
    air_cp_j_kg_k: float
    air_gamma: float
    scale_height_m: float
    air_density_kg_m3: float
    sound_speed_m_s: float
    vapour_scale_height_m: float
    vapour_column_kg_m2: float
    # climate calibration
    albedo: float
    chain_t_eq_k: float
    t_eq_k: float
    chain_t_surface_k: float
    t_surface_target_k: float
    greenhouse_tau: float
    co2_ref_pa: float
    co2_capped: bool
    tau_residual: float
    tau_per_ln_co2: float
    # rotation
    lock_radius_m: float
    tidally_locked: bool
    rotation_period_s: float
    obliquity_rad: float
    day_length_s: float
    # surface
    relief_m: float
    crust: dict
    radiogenic_heat_w_m2: float
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Strict-JSON-ready dict: a non-finite number (the infinite day of a locked planet) becomes
        None; ``tidally_locked`` says why."""
        d = asdict(self)
        d["provenance"] = {k: list(v) for k, v in self.provenance.items()}
        return {k: None if isinstance(val, float) and not math.isfinite(val) else val for k, val in d.items()}

    def scalar_names(self) -> tuple[str, ...]:
        """Names of the per-world scalars :meth:`tensors` returns (partial pressures as
        ``p_<gas>_pa``; ``crust`` is the one vector)."""
        names = []
        for f in fields(self):
            if f.name in ("seed", "source", "provenance", "crust"):
                continue
            if f.name == "partial_pressure_pa":
                names += [f"p_{gas}_pa" for gas in self.partial_pressure_pa]
            else:
                names.append(f.name)
        return tuple(names)

    def scalar(self, name: str):
        """The value of one of :meth:`scalar_names`."""
        if name.startswith("p_") and name.endswith("_pa") and name[2:-3] in self.partial_pressure_pa:
            return self.partial_pressure_pa[name[2:-3]]
        return getattr(self, name)

    def tensors(self, device="cpu", dtype=torch.float32) -> dict[str, torch.Tensor]:
        """Per-world scalars as 0-d tensors (bools as 0/1) plus ``crust`` as a [S] vector in the
        order of ``self.crust``. :func:`stack_specs` stacks them over worlds."""
        out = {}
        for name in self.scalar_names():
            out[name] = torch.tensor(float(self.scalar(name)), dtype=dtype, device=device)
        out["crust"] = torch.tensor([float(v) for v in self.crust.values()], dtype=dtype, device=device)
        return out


def stack_specs(specs, device="cpu", dtype=torch.float32) -> dict[str, torch.Tensor]:
    """``{name: [W] tensor}`` over worlds (``crust``: [W, S]; ``crust_species`` gives the order)."""
    specs = list(specs)
    if not specs:
        raise ValueError("no specs to stack")
    species = tuple(specs[0].crust)
    if any(tuple(s.crust) != species for s in specs):
        raise ValueError("specs have different crust species")
    per = [s.tensors(device, dtype) for s in specs]
    return {key: torch.stack([p[key] for p in per]) for key in per[0]}


def crust_species(spec: PlanetSpec) -> tuple[str, ...]:
    return tuple(spec.crust)


# ---------------------------------------------------------------------------------------------
# building


def build_planet(inputs: dict, seed: int, rules: FormationRules | None = None) -> PlanetSpec:
    """The :class:`PlanetSpec` of chain inputs (``chain.chain_inputs`` or ``synthetic_inputs``).
    ``seed`` seeds the formation RNG (rotation and obliquity when the planet is not locked)."""
    r = rules or FormationRules()
    star_in, planet_in, cloud, bio = inputs["star"], inputs["planet"], inputs["cloud"], inputs["biosphere"]
    v: dict = {"seed": int(seed), "source": str(inputs.get("source", "unknown"))}
    prov: dict = {}
    chain_note = f"{v['source']} seed {inputs.get('seed')}"

    def put(name, value, tag, note):
        v[name] = value
        prov[name] = (tag, note)

    put("seed", int(seed), "chain", f"world seed; formation RNG salt {FORMATION_SALT:#x}")
    put("source", v["source"], "chain", f"inputs source ({chain_note})")
    # ---- star
    m_sun = float(star_in["mass_msun"])
    put("star_mass_kg", m_sun * C.M_SUN, "chain", f"star.mass_msun {m_sun} x M_sun ({chain_note})")
    put("star_luminosity_w", float(star_in["luminosity_lsun"]) * C.L_SUN, "chain",
        f"star.luminosity_lsun {star_in['luminosity_lsun']:.6g} x L_sun 3.828e26 W")
    put("star_age_yr", float(star_in["elapsed_yr"]), "chain",
        "star.elapsed_yr: the star's age when the habitat starts (end of the planet's last chain era minus the star's birth)")
    put("star_birth_yr", float(star_in["age_yr"]), "chain", "star.age_yr: years after the Big Bang at the star's birth")
    put("star_lifetime_yr", float(star_in["lifetime_yr"]), "chain", "star.lifetime_yr = 1e10 yr M / L (world7)")
    put("star_radius_m", main_sequence_radius_m(m_sun, r.ms_radius_exponents), "new_rule",
        f"main-sequence R = R_sun M^{r.ms_radius_exponents[0]} below 1 M_sun, M^{r.ms_radius_exponents[1]} above")
    put("star_teff_k", effective_temperature_k(v["star_luminosity_w"], v["star_radius_m"]), "derived",
        "T_sun (L / R^2)^(1/4) in solar units (Stefan-Boltzmann)")
    put("par_fraction", par_fraction(v["star_teff_k"], r.par_nm), "derived",
        f"Planck integral of B_lambda(T_eff) over {r.par_nm[0]:g}-{r.par_nm[1]:g} nm / (sigma T^4 / pi), 64-node Gauss-Legendre")
    # ---- orbit
    put("orbit_m", float(planet_in["orbit_au"]) * C.AU, "chain", f"planet.orbit_au {planet_in['orbit_au']:.6g} x au")
    put("year_s", kepler_period_s(v["orbit_m"], v["star_mass_kg"]), "derived", "Kepler III: 2 pi sqrt(a^3 / (G M_star))")
    put("flux_earth", float(planet_in["flux_earth"]), "chain", "planet.flux_earth = L / a^2 (earth = 1)")
    put("insolation_w_m2", C.S0 * v["flux_earth"], "derived", "S0 1361 W/m^2 (Kopp & Lean 2011) x flux_earth")
    # ---- bulk composition and interior
    z = float(cloud["metallicity"])
    put("metallicity", z, "chain", "cloud.metallicity (stars level), mass fraction of metals")
    m_e = float(planet_in["mass_earth"])
    mass = m_e * C.M_EARTH
    put("planet_mass_kg", mass, "chain", f"planet.mass_earth {m_e:.6g} x M_earth")
    cmf = core_mass_fraction(cloud)
    put("core_mass_fraction", cmf, "derived+new_rule",
        "new_rule: all cloud Fe to the metal core, Mg and Si to MgO + SiO2 (mantle and crust); "
        "core = Fe / (Fe + MgO + SiO2) by mass")
    wmf = r.water_mass_fraction * (z / C.Z_SUN) ** r.water_metal_exponent
    put("water_mass_fraction", wmf, "new_rule",
        f"volatile delivery: {r.water_mass_fraction:g} (Earth's ocean / mass) x (Z/Z_sun)^{r.water_metal_exponent:g}; "
        "world7's planet accreted no ice inside the frost line yet its water is liquid")
    water = wmf * mass
    rho0 = uncompressed_density(cmf)
    comp = compression(m_e, rho0, r.k_prime)
    put("uncompressed_density_kg_m3", rho0, "derived+reference",
        "1 / (f / 7870 + (1 - f) / 3300): iron (CRC Handbook) and zero-pressure mantle rock")
    put("compression", comp, "derived+new_rule",
        f"Murnaghan (1 + K' P/K0)^(1/K'), K' {r.k_prime:g}, P ~ G M^(2/3) rho^(4/3), calibrated on Earth's "
        f"c_E = {earth_compression():.5f} (its solid density over its uncompressed density)")
    # world7's surface temperature before its whole-K rounding (chain t_eq_k is of the era flux)
    t_target = float(planet_in["t_eq_k"]) + float(planet_in["greenhouse_k"])
    # partial pressures that do not depend on g
    p_o2 = float(bio["oxygen_pal"]) * C.EARTH_AIR_MOLE_FRACTION["O2"] * C.P_STANDARD
    p_h2o = r.relative_humidity * e_sat_pa(t_target)
    # greenhouse calibration (independent of the radius)
    albedo = C.EARTH_ALBEDO
    t_eq = equilibrium_temperature_k(v["insolation_w_m2"], albedo)
    tau = grey_tau(t_target, t_eq)
    tau_earth = earth_tau()
    olr = C.SIGMA * t_target ** 4
    forcing = olr * (1.0 / (1.0 + 0.75 * tau_earth) - 1.0 / (1.0 + 0.75 * tau))
    p_co2_earth = C.EARTH_AIR_MOLE_FRACTION["CO2"] * C.P_STANDARD
    p_co2_log = p_co2_earth * math.exp(forcing / 5.35)
    capped = p_co2_log > r.co2_max_pa
    p_co2 = r.co2_max_pa if capped else p_co2_log
    # radius, gravity, dry air and vapour, iterated to a fixed point (the atmosphere is ~1e-6 of M);
    # the vapour comes out of the planet's water, the rest is the ocean
    dry = vap = 0.0
    for _ in range(100):
        ocean = water - vap
        solid = mass - water - dry
        rho_solid = rho0 * comp
        r_solid = (3.0 * solid / (4.0 * math.pi * rho_solid)) ** (1.0 / 3.0)
        radius = (r_solid ** 3 + 3.0 * (ocean / C.RHO_WATER) / (4.0 * math.pi)) ** (1.0 / 3.0)
        g = C.G * mass / radius ** 2
        area = 4.0 * math.pi * radius ** 2
        vol = r.volatile_scale * (g / C.G_STANDARD) ** r.volatile_g_exponent * C.P_STANDARD
        pp = {"N2": C.EARTH_AIR_MOLE_FRACTION["N2"] * vol, "O2": p_o2, "CO2": p_co2,
              "Ar": C.EARTH_AIR_MOLE_FRACTION["Ar"] * vol, "H2O": p_h2o}
        pressure = sum(pp.values())
        h_vap = vapour_scale_height_m(t_target, r.lapse_rate_k_m * g / C.G_STANDARD)
        column = vapour_column_kg_m2(p_h2o, t_target, h_vap)
        dry_next, vap_next = (pressure - p_h2o) * area / g, column * area
        done = abs(dry_next - dry) + abs(vap_next - vap) <= 1e-13 * mass
        dry, vap = dry_next, vap_next
        if done:
            break
    ocean = water - vap
    core = cmf * solid
    put("core_mass_kg", core, "derived", "core_mass_fraction x (planet - water - atmosphere)")
    put("mantle_mass_kg", solid - core, "derived", "silicate mantle and crust: (1 - core fraction) x solid mass")
    put("ocean_mass_kg", ocean, "new_rule+derived", "water_mass_fraction x planet mass, less the vapour in the air")
    put("atmosphere_mass_kg", dry + vap, "derived",
        "dry air (P - p_H2O) 4 pi R^2 / g plus the vapour column x 4 pi R^2 (water vapour is not well mixed)")
    put("radius_m", radius, "derived+new_rule",
        "solid sphere of density uncompressed x compression, plus the ocean as a liquid-water shell (1000 kg/m^3)")
    put("core_radius_m", (3.0 * core / (4.0 * math.pi * C.RHO_IRON * comp)) ** (1.0 / 3.0), "derived+new_rule",
        "core of iron 7870 kg/m^3 x the same compression factor")
    put("mean_density_kg_m3", mass / (4.0 / 3.0 * math.pi * radius ** 3), "derived", "M / (4/3 pi R^3)")
    put("gravity_m_s2", C.G * mass / radius ** 2, "derived", "G M / R^2")
    put("escape_velocity_m_s", math.sqrt(2.0 * C.G * mass / radius), "derived", "sqrt(2 G M / R)")
    put("ocean_volume_m3", ocean / C.RHO_WATER, "derived", "ocean mass / 1000 kg/m^3")
    put("ocean_layer_m", v["ocean_volume_m3"] / (4.0 * math.pi * radius ** 2), "derived",
        "global equivalent layer: ocean volume / 4 pi R^2")
    # ---- atmosphere
    put("oxygen_pal", float(bio["oxygen_pal"]), "chain", "biosphere.oxygen_pal (bodies era air, present atmospheric level)")
    put("partial_pressure_pa", dict(pp), "derived+new_rule+reference+chain",
        f"N2, Ar: Earth's (US Std. Atm. 1976 mole fractions x 101325 Pa) x volatile scale {r.volatile_scale:g} "
        f"x (g/g_E)^{r.volatile_g_exponent:g} (new_rule); "
        "O2: oxygen_pal x Earth's 21.2 kPa (chain); CO2: co2_ref_pa; "
        f"H2O: relative humidity {r.relative_humidity:g} (new_rule) x Magnus e_sat(T_s)")
    put("surface_pressure_pa", pressure, "derived", "sum of the partial pressures")
    x = {gas: p / pressure for gas, p in pp.items()}
    mu = sum(x[gas] * C.molar_mass(gas) for gas in pp)
    cp_molar = sum(x[gas] * C.CP_MOLAR_298[gas] for gas in pp)
    put("oxygen_mole_fraction", x["O2"], "derived", "p_O2 / P")
    put("air_molar_mass_kg_mol", mu, "derived", "sum x_i mu_i (molar masses from atomic_weights.json)")
    put("air_cp_j_kg_k", cp_molar / mu, "derived+reference", "sum x_i Cp_i (NIST-JANAF, 298 K) / mu")
    put("air_gamma", cp_molar / (cp_molar - C.R_GAS), "derived", "Cp / (Cp - R), ideal gas")
    put("scale_height_m", C.R_GAS * t_target / (mu * g), "derived", "R T_s / (mu g)")
    put("air_density_kg_m3", pressure * mu / (C.R_GAS * t_target), "derived", "P mu / (R T_s)")
    put("sound_speed_m_s", math.sqrt(v["air_gamma"] * C.R_GAS * t_target / mu), "derived", "sqrt(gamma R T_s / mu)")
    put("vapour_scale_height_m", h_vap, "derived+reference+new_rule",
        f"R_v T_s^2 / (L_v Gamma) (Clausius-Clapeyron at constant relative humidity), L_v {C.L_VAPORIZATION:g} J/kg, "
        f"Gamma = {r.lapse_rate_k_m * 1e3:g} K/km (US Std. Atm. 1976) x g / g_E (new_rule: the lapse rate scales with g "
        "at a fixed composition, like the dry adiabat g / cp)")
    put("vapour_column_kg_m2", column, "derived+reference+new_rule",
        "p_H2O mu_w / (R T_s) x vapour_scale_height_m: the planet-mean column water vapour (climate.py's initial "
        f"vapour); the same rule gives Earth about 22 kg/m^2 against the observed {C.EARTH_VAPOUR_COLUMN:g} "
        "(Trenberth, Fasullo & Smith 2005)")
    # ---- climate calibration
    put("albedo", albedo, "reference", "0.30, the Bond albedo world7 builds into T_eq 254.6 K (haishool/cosmos/planets.py:958)")
    put("chain_t_eq_k", float(planet_in["t_eq_k"]), "chain",
        "planet.t_eq_k = 254.6 K era_flux^(1/4), world7's unrounded t_eq of the surface era (planets.equilibrium_temperature)")
    put("t_eq_k", t_eq, "derived", "(S (1 - A) / (4 sigma))^(1/4)")
    put("chain_t_surface_k", float(planet_in["t_surface_k"]), "chain",
        "planet.t_surface_k, world7's whole-K surface temperature")
    put("t_surface_target_k", t_target, "chain",
        "planet.t_eq_k + planet.greenhouse_k: exactly the value world7.temperature_of rounds to t_surface_k")
    put("greenhouse_tau", tau, "derived", "grey atmosphere: T_s = T_eq (1 + 0.75 tau)^(1/4) solved for tau")
    put("co2_ref_pa", p_co2, "derived+new_rule",
        f"Earth's pre-industrial 278 ppm x 101325 Pa x exp(dF / 5.35) (Myhre et al. 1998), dF = sigma T_s^4 "
        f"[1/(1+0.75 tau_E) - 1/(1+0.75 tau)] = {forcing:.4g} W/m^2 with tau_E = {tau_earth:.5f} (world7's Earth 287.6 K); "
        f"new_rule: the greenhouse beyond Earth's is CO2 (world7's thermostat), at most {r.co2_max_pa:g} Pa "
        "(the log law is a fit near present concentrations; Byrne & Goldblatt 2014, GRL 41, 152, show it departing "
        f"at 1e4-1e5 ppm); the log law asks {p_co2_log:.4g} Pa" + (", capped (see co2_capped)" if capped else ""))
    put("co2_capped", bool(capped), "derived+new_rule",
        f"the log law asks more than co2_max_pa {r.co2_max_pa:g} Pa: co2_ref_pa is the cap, not a calibration "
        "(the common case: 216 of 300 synthetic planets, 21 of 26 world7 planets of seeds 12-59), and "
        "tau_residual > 0")
    tau_residual = tau - co2_tau(p_co2, t_target) if capped else 0.0
    put("tau_residual", tau_residual, "derived+new_rule",
        "greenhouse_tau minus the tau that Earth's tau plus 5.35 ln(co2_ref_pa / p_E) W/m^2 gives at T_s: the "
        "greenhouse of no named gas that holds the chain's T_s beyond the CO2 cap (0 when not capped)")
    slope = 5.35 * (1.0 + 0.75 * tau) ** 2 / (0.75 * olr)
    planck = t_target * 0.75 * slope * math.log(2.0) / (4.0 * (1.0 + 0.75 * tau))
    put("tau_per_ln_co2", slope, "derived",
        "dtau / dln p_CO2 such that the grey forcing equals 5.35 W/m^2 per e-fold at the calibration point; at fixed "
        f"T_eq it gives a Planck-only response of {planck:.3g} K per CO2 doubling (no water-vapour or ice feedback; "
        "Earth's is about 3 K, IPCC AR6 likely 2.5-4 K): climate.py must add an H2O term (tau with e_sat(T)) "
        "where realistic sensitivity matters")
    # ---- rotation
    age = v["star_age_yr"]
    a_lock = lock_radius_au(m_sun, age, r.lock_p0_h, r.lock_q)
    put("lock_radius_m", a_lock * C.AU, "derived+reference",
        f"Kasting et al. 1993: 0.027 (P0 t / Q)^(1/6) M^(1/3) au, P0 {r.lock_p0_h:g} h, Q {r.lock_q:g}, t = star_age_yr")
    locked = float(planet_in["orbit_au"]) < a_lock
    put("tidally_locked", bool(locked), "derived", "orbit inside the lock radius")
    if locked:
        put("rotation_period_s", v["year_s"], "derived", "locked: rotation = year")
        put("obliquity_rad", 0.0, "derived", "locked: obliquity 0")
        put("day_length_s", math.inf, "derived", "locked: permanent day side, infinite solar day")
    else:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, FORMATION_SALT]))
        lo, hi = r.rotation_h
        p_rot = math.exp(rng.uniform(math.log(lo), math.log(hi))) * 3600.0
        obl = math.radians(rng.uniform(*r.obliquity_deg))
        put("rotation_period_s", p_rot, "new_rule", f"log-uniform {lo:g}-{hi:g} h on the formation RNG")
        put("obliquity_rad", obl, "new_rule", f"uniform {r.obliquity_deg[0]:g}-{r.obliquity_deg[1]:g} deg on the formation RNG")
        put("day_length_s", p_rot * v["year_s"] / abs(v["year_s"] - p_rot), "derived",
            "solar day of prograde rotation: P_rot P_orb / (P_orb - P_rot)")
    # ---- surface
    put("relief_m", r.relief_m * r.relief_g / v["gravity_m_s2"], "new_rule",
        f"{r.relief_m:g} m x {r.relief_g:g} / g (relief held by rock strength scales as 1/g)")
    put("crust", {}, "derived+reference+new_rule", materials.CRUST_RULE)
    mantle_earth = (1.0 - C.EARTH_CORE_MASS_FRACTION) * (1.0 - C.EARTH_OCEAN_MASS_FRACTION) * C.M_EARTH
    size = (v["mantle_mass_kg"] / mantle_earth) / (radius / C.R_EARTH) ** 2
    decay = radiogenic_decay_factor(age)
    put("radiogenic_heat_w_m2", C.EARTH_RADIOGENIC_FLUX * (z / C.Z_SUN) * decay * size, "reference+new_rule",
        f"Earth's 0.047 W/m^2 x Z/Z_sun {z / C.Z_SUN:.4g} (new_rule) x decay {decay:.4g} (K-40, U-238, U-235, Th-232 over "
        f"{(age - C.EARTH_AGE_YR) / 1e9:+.3g} Gyr against Earth) x silicate mass per area {size:.4g} relative to Earth")
    spec = PlanetSpec(**v, provenance=prov)
    return replace(spec, crust=crust_fractions(inputs, spec))


# ---------------------------------------------------------------------------------------------
# checks and text


def check_spec(spec: PlanetSpec) -> list[str]:
    """The failures of a spec (an empty list when it passes)."""
    bad: list[str] = []
    layers = spec.core_mass_kg + spec.mantle_mass_kg + spec.ocean_mass_kg + spec.atmosphere_mass_kg
    if abs(layers - spec.planet_mass_kg) > 1e-6 * spec.planet_mass_kg:
        bad.append(f"layer masses sum to {layers:.9g} kg, planet {spec.planet_mass_kg:.9g} kg")
    g = C.G * spec.planet_mass_kg / spec.radius_m ** 2
    if abs(spec.gravity_m_s2 - g) > 1e-9 * g:
        bad.append(f"g {spec.gravity_m_s2} differs from G M / R^2 {g}")
    if abs(spec.escape_velocity_m_s - math.sqrt(2 * g * spec.radius_m)) > 1e-6 * spec.escape_velocity_m_s:
        bad.append("escape velocity differs from sqrt(2 G M / R)")
    if abs(spec.t_eq_k - spec.chain_t_eq_k) > 1.0:
        bad.append(f"T_eq {spec.t_eq_k:.3f} K is more than 1 K from the chain's {spec.chain_t_eq_k:.3f} K")
    t_grey = grey_surface_k(spec.t_eq_k, spec.greenhouse_tau)
    if abs(t_grey - spec.chain_t_surface_k) > 0.5 + 1e-9:
        bad.append(f"grey T_s {t_grey:.3f} K is more than 0.5 K from the chain's {spec.chain_t_surface_k} K")
    if abs(t_grey - spec.t_surface_target_k) > 1e-6:
        bad.append(f"grey T_s {t_grey:.6f} K misses the target {spec.t_surface_target_k:.6f} K")
    for f in fields(spec):
        if f.name == "provenance":
            continue
        entry = spec.provenance.get(f.name)
        if entry is None or len(entry) != 2 or not entry[1]:
            bad.append(f"{f.name} has no provenance")
        elif any(part not in TAGS for part in str(entry[0]).split("+")):
            bad.append(f"{f.name} has an unknown provenance tag {entry[0]!r}")
    for name in spec.scalar_names():
        value = spec.scalar(name)
        if isinstance(value, bool):
            continue
        if math.isnan(value) or (math.isinf(value) and name != "day_length_s"):
            bad.append(f"{name} is not finite ({value})")
        elif value < 0:
            bad.append(f"{name} is negative ({value})")
    if spec.tidally_locked and (not math.isinf(spec.day_length_s) or spec.rotation_period_s != spec.year_s
                                or spec.obliquity_rad != 0.0):
        bad.append("a locked planet must rotate once a year, without obliquity, with an infinite day")
    if abs(sum(spec.partial_pressure_pa.values()) - spec.surface_pressure_pa) > 1e-9 * spec.surface_pressure_pa:
        bad.append("surface pressure is not the sum of the partial pressures")
    area = 4 * math.pi * spec.radius_m ** 2
    p_h2o = spec.partial_pressure_pa.get("H2O", 0.0)
    column = vapour_column_kg_m2(p_h2o, spec.t_surface_target_k, spec.vapour_scale_height_m)
    if abs(spec.vapour_column_kg_m2 - column) > 1e-9 * max(column, 1e-30):
        bad.append("vapour column differs from p_H2O mu_w / (R T_s) x H_w")
    atm = (spec.surface_pressure_pa - p_h2o) * area / spec.gravity_m_s2 + spec.vapour_column_kg_m2 * area
    if abs(spec.atmosphere_mass_kg - atm) > 1e-6 * spec.atmosphere_mass_kg:
        bad.append("atmosphere mass differs from (P - p_H2O) 4 pi R^2 / g + vapour column x 4 pi R^2")
    water = spec.water_mass_fraction * spec.planet_mass_kg
    if abs(spec.ocean_mass_kg + spec.vapour_column_kg_m2 * area - water) > 1e-9 * water:
        bad.append("ocean plus vapour differ from water_mass_fraction x planet mass")
    if spec.co2_capped != (spec.tau_residual > 0.0):
        bad.append(f"co2_capped {spec.co2_capped} but tau_residual {spec.tau_residual}")
    tau_co2 = co2_tau(spec.co2_ref_pa, spec.t_surface_target_k)
    if abs(tau_co2 + spec.tau_residual - spec.greenhouse_tau) > 1e-9 * max(spec.greenhouse_tau, 1.0):
        bad.append(f"CO2 tau {tau_co2:.6f} + residual {spec.tau_residual:.6f} is not greenhouse_tau {spec.greenhouse_tau:.6f}")
    crust_total = sum(spec.crust.values())
    if not spec.crust or abs(crust_total - 1.0) > 1e-6 or min(spec.crust.values()) < 0:
        bad.append(f"crust fractions must be non-negative and sum to 1 (sum {crust_total})")
    if tuple(spec.crust) != tuple(materials.CRUST_SPECIES):
        bad.append("crust species differ from materials.CRUST_SPECIES")
    if not 0.0 < spec.par_fraction < 1.0:
        bad.append(f"par_fraction {spec.par_fraction} outside (0, 1)")
    if not 0.0 < spec.core_radius_m < spec.radius_m:
        bad.append("core radius must lie within the planet")
    year = kepler_period_s(spec.orbit_m, spec.star_mass_kg)
    if abs(spec.year_s - year) > 1e-9 * year:
        bad.append("year differs from Kepler's third law")
    return bad


_UNITS = {"_kg_m3": "kg/m^3", "_kg_m2": "kg/m^2", "_kg_mol": "kg/mol", "_j_kg_k": "J/(kg K)", "_w_m2": "W/m^2", "_m_s2": "m/s^2",
          "_m_s": "m/s", "_m3": "m^3", "_kg": "kg", "_w": "W", "_yr": "yr", "_m": "m", "_s": "s", "_k": "K",
          "_pa": "Pa", "_rad": "rad"}


def describe(spec: PlanetSpec) -> str:
    """One line per field: name, value, unit, provenance tag and note."""
    lines = [f"PlanetSpec seed {spec.seed} ({spec.source}), {FORMATION_VERSION}"]
    for f in fields(spec):
        if f.name == "provenance":
            continue
        value = getattr(spec, f.name)
        unit = next((u for suffix, u in _UNITS.items() if f.name.endswith(suffix)), "")
        if isinstance(value, dict):
            shown = ", ".join(f"{k} {val:.4g}" for k, val in value.items())
        elif isinstance(value, float):
            shown = f"{value:.6g}"
        else:
            shown = str(value)
        tag, note = spec.provenance.get(f.name, ("?", ""))
        lines.append(f"{f.name:28s} {shown} {unit}  [{tag}] {note}")
    return "\n".join(lines)
