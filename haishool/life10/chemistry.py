"""Conserved, spatial precursor chemistry for life10.

This is a *coarse-grained research model*, not an ab-initio origin-of-life
simulation.  Formulae and reaction stoichiometry are exact; energies, barriers,
rates and the collapsed precursor routes below are declared phenomenological
rules, not measured reaction data.  Balancing a route does not demonstrate its
chemical feasibility.  No species is assigned replication, fitness or a copying
motif.  Organic precursors must be made from the declared initial inventory.

Amounts are micromoles, stored in float64 on a periodic two-dimensional grid.
Every forward/reverse extent is bounded by its reactants and local energy.  A
photon-driven forward reaction requires photons even if the cell is warm.
Reaction energy enters heat or stored chemical energy.  Optional temperature
cycles exchange heat with a declared bath whose signed input is recorded.
Diffusion and thermal transport are conservative convex neighbour exchanges.
CPU initialization uses an explicitly seeded generator; no random draws occur
inside a chemistry step.  GPU trajectories can differ in floating-point rounding.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Mapping, Sequence

import torch


ELEMENTS = ("C", "H", "N", "O", "P")
R_GAS = 8.31446261815324
UMOL_PER_MOL = 1_000_000.0
POLYMER_MONOMERS = ("glycine", "alanine")
CONDENSATION_WATER = "H2O"
CONDENSATION_WATER_ELEMENTS = (0, 2, 0, 1, 0)
# A fixed coarse-grained stored energy per linkage, used by the polymer layer.
BOND_ENERGY_J_PER_UMOL = 0.5


@dataclass(frozen=True)
class Species:
    name: str
    formula: str
    elements: tuple[int, int, int, int, int]
    chemical_j_per_umol: float


# Energies are a declared model potential with the simple initial gases as its
# reference.  They are not tabulated heats of formation or free energies.
SPECIES = (
    Species("H2O", "H2O", (0, 2, 0, 1, 0), 0.0),
    Species("CO2", "CO2", (1, 0, 0, 2, 0), 0.0),
    Species("N2", "N2", (0, 0, 2, 0, 0), 0.0),
    Species("H2", "H2", (0, 2, 0, 0, 0), 0.05),
    Species("NH3", "NH3", (0, 3, 1, 0, 0), 0.02),
    Species("H3PO4", "H3PO4", (0, 3, 0, 4, 1), 0.0),
    Species("HCN", "HCN", (1, 1, 1, 0, 0), 0.25),
    Species("formaldehyde", "CH2O", (1, 2, 0, 1, 0), 0.20),
    Species("glycolaldehyde", "C2H4O2", (2, 4, 0, 2, 0), 0.40),
    Species("ribose", "C5H10O5", (5, 10, 0, 5, 0), 1.00),
    Species("adenine", "C5H5N5", (5, 5, 5, 0, 0), 1.25),
    Species("acetaldehyde", "C2H4O", (2, 4, 0, 1, 0), 0.40),
    Species("glycine", "C2H5NO2", (2, 5, 1, 2, 0), 0.45),
    Species("alanine", "C3H7NO2", (3, 7, 1, 2, 0), 0.65),
    # Unallocated initial atoms remain explicit and cannot disappear.  They do
    # not automatically turn into useful molecules during the simulation.
    Species("C_atom", "C", (1, 0, 0, 0, 0), 0.0),
    Species("H_atom", "H", (0, 1, 0, 0, 0), 0.0),
    Species("N_atom", "N", (0, 0, 1, 0, 0), 0.0),
    Species("O_atom", "O", (0, 0, 0, 1, 0), 0.0),
    Species("P_atom", "P", (0, 0, 0, 0, 1), 0.0),
    Species("O2", "O2", (0, 0, 0, 2, 0), 0.0),
    Species("acetic_acid", "C2H4O2", (2, 4, 0, 2, 0), 0.40),
    Species("hexanoic_acid", "C6H12O2", (6, 12, 0, 2, 0), 1.00),
)
SPECIES_INDEX = {sp.name: i for i, sp in enumerate(SPECIES)}


@dataclass(frozen=True)
class Reaction:
    name: str
    reactants: tuple[tuple[str, int], ...]
    products: tuple[tuple[str, int], ...]
    prefactor_per_s: float
    activation_j_per_mol: float
    photon_j_per_umol: float = 0.0
    explanation: str = ""

    @property
    def delta_energy_j_per_umol(self) -> float:
        return sum(SPECIES[SPECIES_INDEX[s]].chemical_j_per_umol * n
                   for s, n in self.products) - sum(
            SPECIES[SPECIES_INDEX[s]].chemical_j_per_umol * n
            for s, n in self.reactants)


# Fixed aggregate routes.  In particular, ribose/adenine production is not a
# claim that an unconstrained formose/HCN mixture selectively makes them.
# Those selectivities are explicit model assumptions to test and vary by version.
REACTIONS = (
    Reaction("carbon_photoreduction", (("CO2", 1), ("H2", 2)),
             (("formaldehyde", 1), ("H2O", 1)), 1.0, 9_000.0, 0.15,
             "Assumed aggregate light-powered carbon reduction."),
    Reaction("nitrogen_reduction", (("N2", 1), ("H2", 3)),
             (("NH3", 2),), 0.4, 12_000.0, 0.0,
             "Assumed effective nitrogen-reduction route; catalysis not resolved."),
    Reaction("cyanide_formation", (("formaldehyde", 1), ("NH3", 1)),
             (("HCN", 1), ("H2O", 1), ("H2", 1)), 0.8, 8_000.0, 0.10,
             "Assumed light-driven collapsed precursor route."),
    Reaction("glycolaldehyde_assembly", (("formaldehyde", 2),),
             (("glycolaldehyde", 1),), 0.4, 7_000.0),
    Reaction("ribose_assembly", (("glycolaldehyde", 1), ("formaldehyde", 3)),
             (("ribose", 1),), 0.1, 8_000.0),
    Reaction("adenine_assembly", (("HCN", 5),), (("adenine", 1),),
             0.08, 8_000.0),
    Reaction("acetaldehyde_formation", (("formaldehyde", 2), ("H2", 1)),
             (("acetaldehyde", 1), ("H2O", 1)), 0.3, 8_000.0),
    Reaction("glycine_formation", (("formaldehyde", 1), ("HCN", 1), ("H2O", 1)),
             (("glycine", 1),), 0.5, 7_000.0,
             explanation="Assumed overall amino-acid route; intermediates omitted."),
    Reaction("alanine_formation", (("acetaldehyde", 1), ("HCN", 1), ("H2O", 1)),
             (("alanine", 1),), 0.5, 7_000.0,
             explanation="Assumed overall amino-acid route; intermediates omitted."),
    Reaction("water_photolysis", (("H2O", 2),), (("H2", 2), ("O2", 1)),
             0.3, 8_000.0, 0.15,
             "Assumed aggregate light-powered water splitting."),
    Reaction("acetic_acid_formation", (("acetaldehyde", 2), ("O2", 1)),
             (("acetic_acid", 2),), 0.3, 7_000.0),
    Reaction("hexanoic_acid_formation", (("acetic_acid", 1), ("formaldehyde", 4), ("H2", 4)),
             (("hexanoic_acid", 1), ("H2O", 4)), 0.06, 9_000.0,
             explanation="Assumed collapsed amphiphile route, not a validated prebiotic pathway."),
)


def _check_catalog() -> None:
    for r in REACTIONS:
        inventory = [0] * len(ELEMENTS)
        for sign, side in ((-1, r.reactants), (1, r.products)):
            for name, n in side:
                if n <= 0:
                    raise ValueError(f"Invalid stoichiometry in {r.name}")
                for j, atoms in enumerate(SPECIES[SPECIES_INDEX[name]].elements):
                    inventory[j] += sign * n * atoms
        if any(inventory):
            raise ValueError(f"Unbalanced reaction {r.name}: {inventory}")
        if r.photon_j_per_umol and r.photon_j_per_umol < r.delta_energy_j_per_umol:
            raise ValueError(f"Insufficient photon energy in {r.name}")


_check_catalog()


@dataclass(frozen=True)
class ChemicalConfig:
    grid_size: int = 8
    batch_size: int = 1
    dt_s: float = 1.0
    temperature_k: float = 300.0
    heat_capacity_j_per_k: float = 10.0
    photon_input_j: float = 0.1  # joules per cell per second, external input
    diffusion_fraction: float = 0.10  # fraction exchanged per step, <= 1
    thermal_diffusion_fraction: float = 0.05
    max_reaction_fraction: float = 0.20
    reference_amount_umol: float = 1.0
    temperature_cycle_amplitude_k: float = 0.0
    temperature_cycle_period_s: float = 86400.0
    thermostat: bool = False

    def __post_init__(self) -> None:
        if self.grid_size < 1 or self.batch_size < 1:
            raise ValueError("Grid and batch size must be positive")
        for key in ("dt_s", "temperature_k", "heat_capacity_j_per_k",
                    "reference_amount_umol", "temperature_cycle_period_s"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("photon_input_j", "temperature_cycle_amplitude_k"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.temperature_cycle_amplitude_k >= self.temperature_k:
            raise ValueError("Temperature cycle must remain above zero kelvin")
        for key in ("diffusion_fraction", "thermal_diffusion_fraction",
                    "max_reaction_fraction"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must lie in [0, 1]")


@dataclass
class ChemicalState:
    amount_umol: torch.Tensor  # [B, H, W, S]
    heat_j: torch.Tensor  # [B, H, W]
    photon_j: torch.Tensor  # [B, H, W]
    cumulative_input_j: torch.Tensor  # [B], all external energy, including bath
    cumulative_photon_input_j: torch.Tensor
    cumulative_bath_input_j: torch.Tensor  # signed, heat can leave the system
    initial_energy_j: torch.Tensor
    initial_elements_umol: torch.Tensor  # [B, len(ELEMENTS)]
    reaction_extent_umol: torch.Tensor  # [B, R, 2] forward/reverse lifetime totals
    steps: int = 0

    def checkpoint(self) -> dict:
        return {f.name: (getattr(self, f.name).detach().cpu().clone()
                         if isinstance(getattr(self, f.name), torch.Tensor)
                         else getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_checkpoint(cls, data: Mapping, device: str | torch.device = "cpu") -> "ChemicalState":
        return cls(**{f.name: (data[f.name].to(device=device, dtype=torch.float64).clone()
                               if isinstance(data[f.name], torch.Tensor) else data[f.name])
                      for f in fields(cls)})


class ChemicalEngine:
    def __init__(self, config: ChemicalConfig | None = None,
                 device: str | torch.device = "cpu"):
        self.config = config or ChemicalConfig()
        self.device = torch.device(device)
        self.composition = torch.tensor([s.elements for s in SPECIES],
                                        dtype=torch.float64, device=self.device)
        self.chemical_energy = torch.tensor([s.chemical_j_per_umol for s in SPECIES],
                                            dtype=torch.float64, device=self.device)

    def _batch_value(self, value: float | Sequence[float] | torch.Tensor) -> torch.Tensor:
        v = torch.as_tensor(value, dtype=torch.float64, device="cpu")
        if v.ndim == 0:
            v = v.expand(self.config.batch_size).clone()
        if v.shape != (self.config.batch_size,) or not torch.isfinite(v).all() or (v < 0).any():
            raise ValueError("Inventories must be finite nonnegative amounts, one per batch")
        return v

    def initialize(self, seed: int, inventory_umol: Mapping[str, float | Sequence[float]]
                   | None = None, temperature_k: float | torch.Tensor | None = None) -> ChemicalState:
        """Distribute exactly the declared inventory, never adding organics.

        ``inventory_umol`` specifies total micromoles per world, not per cell.
        The seed supplies small independent spatial heterogeneity.  An empty
        inventory remains empty even under illumination.
        """
        c = self.config
        amounts = torch.zeros(c.batch_size, c.grid_size, c.grid_size, len(SPECIES),
                              dtype=torch.float64)
        g = torch.Generator(device="cpu").manual_seed(seed)
        for name in sorted(inventory_umol or {}):
            if name not in SPECIES_INDEX:
                raise ValueError(f"Unknown chemical species: {name}")
            total = self._batch_value(inventory_umol[name])
            spatial = 0.75 + 0.5 * torch.rand(c.batch_size, c.grid_size, c.grid_size,
                                             generator=g, dtype=torch.float64)
            spatial /= spatial.sum(dim=(1, 2), keepdim=True)
            amounts[..., SPECIES_INDEX[name]] = spatial * total[:, None, None]
        amounts = amounts.to(self.device)
        temperatures = self._temperature_field(c.temperature_k if temperature_k is None else temperature_k,
                                               amounts.shape[:3])
        heat = (temperatures * c.heat_capacity_j_per_k).clone()
        zeros = torch.zeros(c.batch_size, dtype=torch.float64, device=self.device)
        state = ChemicalState(amounts, heat, torch.zeros_like(heat), zeros.clone(),
                              zeros.clone(), zeros.clone(), zeros.clone(),
                              torch.zeros(c.batch_size, len(ELEMENTS), dtype=torch.float64,
                                          device=self.device),
                              torch.zeros(c.batch_size, len(REACTIONS), 2,
                                          dtype=torch.float64, device=self.device))
        state.initial_energy_j.copy_(self.energy_total(state))
        state.initial_elements_umol.copy_(self.elemental_totals(state))
        return state

    def initialize_from_elements(self, seed: int,
                                 elemental_budget_umol: Mapping[str, float | Sequence[float]]
                                 | Sequence[Mapping[str, float]] | torch.Tensor,
                                 temperature_k: float | torch.Tensor | None = None) -> ChemicalState:
        """A declared initial molecular partition, not an evolving chemistry step.

        Allocate H3PO4, CO2, H2O, N2 and H2 in that fixed order, bounded by the
        exact supplied atom inventory.  Unallocated atoms remain explicit.  No
        phosphorus is supplied if the source inventory contains none.  The
        partition and its initial stored energy must be saved in run manifests.
        """
        if isinstance(elemental_budget_umol, torch.Tensor):
            if elemental_budget_umol.shape != (self.config.batch_size, len(ELEMENTS)):
                raise ValueError("Element budget tensor must have shape [batch_size, len(ELEMENTS)]")
            elemental_budget_umol = {e: elemental_budget_umol[:, i].detach().cpu()
                                     for i, e in enumerate(ELEMENTS)}
        elif not isinstance(elemental_budget_umol, Mapping):
            if len(elemental_budget_umol) != self.config.batch_size:
                raise ValueError("Element budget list must contain one dictionary per batch")
            unsupported = set().union(*(set(row) - set(ELEMENTS) for row in elemental_budget_umol))
            if unsupported:
                raise ValueError(f"Unsupported elements: {sorted(unsupported)}")
            elemental_budget_umol = {e: [row.get(e, 0.0) for row in elemental_budget_umol]
                                     for e in ELEMENTS}
        unknown = set(elemental_budget_umol) - set(ELEMENTS)
        if unknown:
            raise ValueError(f"Unsupported elements: {sorted(unknown)}")
        remaining = {e: self._batch_value(elemental_budget_umol.get(e, 0.0))
                     for e in ELEMENTS}
        inventory = {}
        for name in ("H3PO4", "CO2", "H2O", "N2", "H2"):
            sp = SPECIES[SPECIES_INDEX[name]]
            amount = torch.stack([remaining[e] / n for e, n in zip(ELEMENTS, sp.elements)
                                  if n]).amin(dim=0)
            inventory[name] = amount
            for e, n in zip(ELEMENTS, sp.elements):
                remaining[e] = remaining[e] - amount * n
        for e in ELEMENTS:
            inventory[f"{e}_atom"] = remaining[e].clamp_min(0)
        return self.initialize(seed, inventory, temperature_k=temperature_k)

    def _temperature_field(self, value: float | torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        temperature = torch.as_tensor(value, dtype=torch.float64, device=self.device)
        if temperature.ndim == 1 and temperature.shape[0] == self.config.batch_size:
            temperature = temperature[:, None, None]
        temperature = torch.broadcast_to(temperature, shape)
        if not torch.isfinite(temperature).all() or (temperature <= 0).any():
            raise ValueError("Temperature must be finite and above zero kelvin")
        return temperature

    def temperature(self, state: ChemicalState) -> torch.Tensor:
        return state.heat_j / self.config.heat_capacity_j_per_k

    def elemental_totals(self, state: ChemicalState) -> torch.Tensor:
        return state.amount_umol.sum(dim=(1, 2)) @ self.composition

    def energy_total(self, state: ChemicalState) -> torch.Tensor:
        chemical = (state.amount_umol * self.chemical_energy).sum(dim=(1, 2, 3))
        return chemical + (state.heat_j + state.photon_j).sum(dim=(1, 2))

    def energy_residual(self, state: ChemicalState) -> torch.Tensor:
        return self.energy_total(state) - state.initial_energy_j - state.cumulative_input_j

    def _diffuse(self, field: torch.Tensor, fraction: float,
                 permeability: float | torch.Tensor | None = None) -> torch.Tensor:
        if not fraction:
            return field
        if permeability is not None:
            p = torch.as_tensor(permeability, dtype=torch.float64, device=self.device)
            if p.ndim == 3 and field.ndim == 4:
                p = p[..., None]
            p = torch.broadcast_to(p, field.shape)
            if not torch.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                raise ValueError("Permeability must be finite and lie in [0, 1]")
            x_p, y_p = torch.roll(p, -1, 1), torch.roll(p, -1, 2)
            x_face = 2 * p * x_p / (p + x_p).clamp_min(1e-300)
            y_face = 2 * p * y_p / (p + y_p).clamp_min(1e-300)
            x_back, y_back = torch.roll(x_face, 1, 1), torch.roll(y_face, 1, 2)
            f = fraction / 4.0
            # The same face coefficient is used by both cells sharing a face.
            # This convex form is nonnegative and conservative, including walls.
            return (field * (1 - f * (x_face + x_back + y_face + y_back))
                    + f * (x_face * torch.roll(field, -1, 1)
                           + x_back * torch.roll(field, 1, 1)
                           + y_face * torch.roll(field, -1, 2)
                           + y_back * torch.roll(field, 1, 2)))
        neighbours = (torch.roll(field, 1, 1) + torch.roll(field, -1, 1)
                      + torch.roll(field, 1, 2) + torch.roll(field, -1, 2))
        return (1.0 - fraction) * field + (fraction / 4.0) * neighbours

    def _reaction(self, state: ChemicalState, r: Reaction, reaction_i: int,
                  reverse: bool, catalyst: float | torch.Tensor) -> None:
        """Bounded mass-action Euler extent with Arrhenius activation.

        For thermal routes, the reverse barrier is chosen so kf/kr equals
        exp(-delta_E/RT).  This is an idealized energy-only detailed-balance
        relation; concentration/activity and entropy models are simplified.
        Light-driven routes are out of equilibrium and have a separate photon
        requirement.  Catalysts accelerate both directions identically.
        """
        c = self.config
        reactants, products = (r.products, r.reactants) if reverse else (r.reactants, r.products)
        delta = (-1 if reverse else 1) * r.delta_energy_j_per_umol
        photon_cost = 0.0 if reverse else r.photon_j_per_umol
        # A light-driven forward activation is paid by the declared photon
        # supply.  Thermal uphill activation adds the energy difference.
        barrier = r.activation_j_per_mol + (0 if photon_cost else max(0, delta) * UMOL_PER_MOL)
        temperature = self.temperature(state).clamp_min(1e-12)
        catalyst = torch.as_tensor(catalyst, dtype=torch.float64, device=self.device)
        if catalyst.ndim == 1 and catalyst.shape[0] == c.batch_size:
            catalyst = catalyst[:, None, None]
        if not torch.isfinite(catalyst).all() or (catalyst < 0).any():
            raise ValueError("Catalysis multiplier must be finite and nonnegative")
        rate = r.prefactor_per_s * torch.exp(-barrier / (R_GAS * temperature)) * catalyst
        limit = torch.full_like(state.heat_j, float("inf"))
        log_activity = torch.zeros_like(state.heat_j)
        present = torch.ones_like(state.heat_j, dtype=torch.bool)
        for name, n in reactants:
            amount = state.amount_umol[..., SPECIES_INDEX[name]]
            limit = torch.minimum(limit, amount / n)
            present &= amount > 0
            log_activity += n * torch.log((amount / c.reference_amount_umol).clamp_min(1e-300))
        extent = rate * c.dt_s * c.reference_amount_umol * torch.exp(log_activity.clamp(max=80.0))
        # A few ulps of unused inventory avoid a negative remainder from an
        # upward-rounded division/multiplication when the limit is exhausted.
        extent = torch.minimum(extent, limit * c.max_reaction_fraction * (1 - 1e-14))
        extent = torch.where(present, extent, torch.zeros_like(extent))
        if photon_cost:
            extent = torch.minimum(extent, state.photon_j / photon_cost * (1 - 1e-14))
        heat_per_extent = photon_cost - delta
        if heat_per_extent < 0:
            extent = torch.minimum(extent, state.heat_j / -heat_per_extent * (1 - 1e-14))
        for name, n in reactants:
            state.amount_umol[..., SPECIES_INDEX[name]] -= n * extent
        for name, n in products:
            state.amount_umol[..., SPECIES_INDEX[name]] += n * extent
        state.photon_j -= photon_cost * extent
        state.heat_j += heat_per_extent * extent
        state.reaction_extent_umol[:, reaction_i, int(reverse)] += extent.sum(dim=(1, 2))

    @torch.no_grad()
    def step(self, state: ChemicalState, illumination: float | torch.Tensor = 1.0,
             catalysts: Mapping[str, float | torch.Tensor] | None = None,
             catalyst_multiplier: float | torch.Tensor = 1.0,
             permeability: float | torch.Tensor | None = None,
             temperature_k: float | torch.Tensor | None = None) -> ChemicalState:
        """Advance one fixed-law step in place, without importing matter."""
        c = self.config
        light = torch.as_tensor(illumination, dtype=torch.float64, device=self.device)
        if not torch.isfinite(light).all() or (light < 0).any():
            raise ValueError("Illumination must be finite and nonnegative")
        if light.ndim == 1 and light.shape[0] == c.batch_size:
            light = light[:, None, None]
        added_photons = torch.broadcast_to(light * c.photon_input_j * c.dt_s, state.photon_j.shape)
        state.photon_j += added_photons
        added = added_photons.sum(dim=(1, 2))
        state.cumulative_input_j += added
        state.cumulative_photon_input_j += added
        if temperature_k is not None or c.thermostat or c.temperature_cycle_amplitude_k:
            if temperature_k is None:
                phase = 2.0 * math.pi * (state.steps + 1) * c.dt_s / c.temperature_cycle_period_s
                temperature_k = c.temperature_k + c.temperature_cycle_amplitude_k * math.sin(phase)
            target = c.heat_capacity_j_per_k * self._temperature_field(temperature_k, state.heat_j.shape)
            bath = (target - state.heat_j).sum(dim=(1, 2))
            state.heat_j.copy_(target)
            state.cumulative_bath_input_j += bath
            state.cumulative_input_j += bath
        state.amount_umol = self._diffuse(state.amount_umol, c.diffusion_fraction, permeability)
        state.heat_j = self._diffuse(state.heat_j, c.thermal_diffusion_fraction)
        for i, r in enumerate(REACTIONS):
            cat = torch.as_tensor((catalysts or {}).get(r.name, 1.0), device=self.device,
                                  dtype=torch.float64) * torch.as_tensor(catalyst_multiplier,
                                                                        device=self.device,
                                                                        dtype=torch.float64)
            self._reaction(state, r, i, False, cat)
            self._reaction(state, r, i, True, cat)
        state.steps += 1
        return state

    def checkpoint(self, state: ChemicalState) -> dict:
        return {"config": asdict(self.config), "state": state.checkpoint()}

    @classmethod
    def load_checkpoint(cls, data: Mapping, device: str | torch.device = "cpu") -> tuple["ChemicalEngine", ChemicalState]:
        engine = cls(ChemicalConfig(**data["config"]), device)
        return engine, ChemicalState.from_checkpoint(data["state"], device)
