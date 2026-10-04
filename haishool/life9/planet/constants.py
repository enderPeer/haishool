"""Physical, astronomical and chemical reference constants of the planet engine (SI units).

Every value here is ``reference`` provenance (PLANET-SPEC section 0): published data with its
source next to it. Nothing in this module is a modelling choice; those live in ``formation``.

* Physical constants: CODATA 2018 (Tiesinga et al. 2021, Rev. Mod. Phys. 93, 025010). Since the
  2019 SI redefinition ``K_B``, ``N_A``, ``H_PLANCK`` and ``C_LIGHT`` are exact, and ``R_GAS`` and
  ``SIGMA`` follow from them exactly.
* Astronomical values: IAU 2015 Resolution B3 nominal solar values (Prsa et al. 2016, AJ 152, 41),
  IAU 2012 Resolution B2 (the au), Kopp & Lean 2011 (GRL 38, L01706) for the solar constant.
* Molar masses: the repo's table ``data/truth-v5/atomic_weights.json`` (IUPAC standard atomic
  weights, conventional values) through :func:`molar_mass`.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------------- CODATA 2018
G = 6.67430e-11                 # m^3 kg^-1 s^-2, Newtonian constant of gravitation
K_B = 1.380649e-23              # J/K, Boltzmann constant (exact)
N_A = 6.02214076e23             # 1/mol, Avogadro constant (exact)
H_PLANCK = 6.62607015e-34       # J s, Planck constant (exact)
C_LIGHT = 299792458.0           # m/s, speed of light in vacuum (exact)
R_GAS = K_B * N_A               # J/(mol K), molar gas constant = 8.314462618... (exact)
SIGMA = 2 * math.pi ** 5 * K_B ** 4 / (15 * H_PLANCK ** 3 * C_LIGHT ** 2)  # W m^-2 K^-4 = 5.670374419e-8

# ---------------------------------------------------------------------------------- astronomy
S0 = 1361.0                     # W/m^2, total solar irradiance at 1 au (Kopp & Lean 2011)
L_SUN = 3.828e26                # W, IAU 2015 B3 nominal solar luminosity
GM_SUN = 1.3271244e20           # m^3/s^2, IAU 2015 B3 nominal solar mass parameter
GM_EARTH = 3.986004e14          # m^3/s^2, IAU 2015 B3 nominal terrestrial mass parameter
#: masses from the nominal GM and this module's G (CODATA 2018), so that G M is the IAU GM
#: exactly: 1.988410e30 kg and 5.972168e24 kg (the GM is known far better than G)
M_SUN = GM_SUN / G
M_EARTH = GM_EARTH / G
R_SUN = 6.957e8                 # m, IAU 2015 B3 nominal solar radius
T_SUN = 5772.0                  # K, IAU 2015 B3 nominal solar effective temperature
AU = 1.495978707e11             # m, astronomical unit (IAU 2012 B2, exact)
R_EARTH = 6.371e6               # m, Earth's mean radius (IUGG; Moritz 2000)

# ---------------------------------------------------------------------------------- time
YEAR_S = 31557600.0             # s, Julian year (365.25 d; IAU)
DAY_S = 86400.0                 # s

# ---------------------------------------------------------------------------------- Earth references
G_STANDARD = 9.80665            # m/s^2, standard gravity (3rd CGPM 1901)
P_STANDARD = 101325.0           # Pa, standard atmosphere (mean sea-level pressure)
#: dry-air mole fractions near the surface (U.S. Standard Atmosphere 1976; CO2 here is
#: the pre-industrial 278 ppm of IPCC TAR 2001, ch. 6, the reference C0 of Myhre et al. 1998)
EARTH_AIR_MOLE_FRACTION = {"N2": 0.78084, "O2": 0.209476, "Ar": 0.00934, "CO2": 278e-6}
#: Earth's Bond albedo, the value world7 builds into T_eq = 254.6 K (haishool/cosmos/planets.py:958;
#: CERES-era measurements give 0.29-0.30, Stephens et al. 2015, Rev. Geophys. 53, 141)
EARTH_ALBEDO = 0.30
#: present-day solar photospheric metallicity Z (Asplund, Grevesse, Sauval & Scott 2009,
#: ARA&A 47, 481), the chain's own Z_SUN (haishool/evo/stars.py:290)
Z_SUN = 0.0134
#: solar photospheric abundances log eps = log10(N_X / N_H) + 12 (Asplund et al. 2009, Table 1)
SOLAR_LOG_EPS = {"H": 12.00, "He": 10.93, "C": 8.43, "N": 7.83, "O": 8.69, "Ne": 7.93, "Na": 6.24,
                 "Mg": 7.60, "Al": 6.45, "Si": 7.51, "S": 7.12, "Ar": 6.40, "K": 5.03, "Ca": 6.34,
                 "Fe": 7.50, "Cu": 4.19, "Sn": 2.04}
EARTH_AGE_YR = 4.5673e9         # yr, age of the solar system (Connelly et al. 2012, Science 338, 651)
EARTH_CORE_MASS_FRACTION = 1.932e24 / M_EARTH  # 0.3235: PREM core mass (Dziewonski & Anderson 1981, PEPI 25, 297)
EARTH_BULK_DENSITY = M_EARTH / (4.0 / 3.0 * math.pi * R_EARTH ** 3)  # kg/m^3, 5513.4
EARTH_OCEAN_MASS_FRACTION = 2.3e-4  # ocean 1.4e21 kg / M_EARTH (Charette & Smith 2010, Oceanography 23(2), 112)

# ---------------------------------------------------------------------------------- materials
RHO_IRON = 7870.0               # kg/m^3, iron at 293 K (CRC Handbook of Chemistry and Physics, 7.874 g/cm^3)
RHO_SILICATE = 3300.0           # kg/m^3, zero-pressure upper-mantle rock (pyrolite; olivine Fo90 about
                                # 3.3 g/cm^3, e.g. Stacey & Davis 2008, Physics of the Earth, 4th ed.)
RHO_WATER = 1000.0              # kg/m^3, liquid water (CRC: 999.97 at 277 K)
L_VAPORIZATION = 2.501e6        # J/kg, latent heat of vaporisation of water at 273.15 K (Rogers & Yau 1989,
                                # A Short Course in Cloud Physics, 3rd ed., table 2.1)
#: Earth's mean tropospheric lapse rate, K/m (U.S. Standard Atmosphere 1976, 0-11 km)
EARTH_LAPSE_RATE = 6.5e-3
#: Earth's global, annual mean column water vapour (precipitable water), kg/m^2 (Trenberth,
#: Fasullo & Smith 2005, Clim. Dyn. 24, 741: 24.6 mm from NVAP / ERA-40); the check of the
#: vapour column rule in formation
EARTH_VAPOUR_COLUMN = 24.6
#: ideal-gas molar heat capacities at 298.15 K, J/(mol K) (NIST-JANAF Thermochemical Tables,
#: Chase 1998, J. Phys. Chem. Ref. Data Monograph 9)
CP_MOLAR_298 = {"N2": 29.124, "O2": 29.376, "Ar": 20.786, "CO2": 37.129, "H2O": 33.588}

# ---------------------------------------------------------------------------------- radiogenic heat
#: half-lives in years (NUBASE2016, Audi et al. 2017, Chinese Phys. C 41, 030001)
HALF_LIFE_YR = {"K40": 1.248e9, "U238": 4.468e9, "U235": 7.04e8, "Th232": 1.40e10}
#: share of Earth's present radiogenic heat per isotope: heat production per kg of isotope
#: (U-238 9.46e-5, U-235 5.69e-4, Th-232 2.64e-5, K-40 2.92e-5 W/kg) times the mantle content
#: (U 31 ppb, Th 124 ppb, K 310 ppm; U-235/U 0.0071, K-40/K 1.19e-4), Turcotte & Schubert 2002,
#: Geodynamics 2nd ed., ch. 4 (their mantle total 7.39e-12 W/kg)
_H_ISOTOPE = {"U238": 31e-9 * 0.9928 * 9.46e-5, "U235": 31e-9 * 0.0071 * 5.69e-4,
              "Th232": 124e-9 * 2.64e-5, "K40": 310e-6 * 1.19e-4 * 2.92e-5}
EARTH_RADIOGENIC_SHARE = {k: v / sum(_H_ISOTOPE.values()) for k, v in _H_ISOTOPE.items()}
#: Earth's radiogenic surface flux: about 24 TW (U + Th 20 TW from geoneutrinos, KamLAND
#: Collaboration 2011, Nature Geosci. 4, 647; K about 4 TW) over 5.10e14 m^2
EARTH_RADIOGENIC_FLUX = 0.047   # W/m^2

# ---------------------------------------------------------------------------------- molar masses
ATOMIC_WEIGHTS_PATH = Path(__file__).resolve().parents[3] / "data" / "truth-v5" / "atomic_weights.json"


@lru_cache(maxsize=1)
def atomic_weights() -> dict[str, float]:
    """Standard atomic weights in g/mol by element symbol (the repo's IUPAC table)."""
    with open(ATOMIC_WEIGHTS_PATH, encoding="utf-8") as f:
        return {k: float(v) for k, v in json.load(f).items()}


@lru_cache(maxsize=None)
def molar_mass(formula: str) -> float:
    """Molar mass in kg/mol of a formula such as ``"CO2"``, ``"Cu2CO3(OH)2"`` or ``"Fe"``."""
    from haishool.truth.formula import parse
    weights = atomic_weights()
    return sum(weights[el] * n for el, n in parse(formula).items()) / 1000.0


def element_mass(symbol: str) -> float:
    """Molar mass of one element in kg/mol."""
    return atomic_weights()[symbol] / 1000.0


def solar_number_ratio(a: str, b: str) -> float:
    """Solar photospheric number ratio N_a / N_b (Asplund et al. 2009)."""
    return 10.0 ** (SOLAR_LOG_EPS[a] - SOLAR_LOG_EPS[b])
