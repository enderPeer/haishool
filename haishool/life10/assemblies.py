"""Spatial peptide-like condensation, hydrolysis and reversible scaffold adsorption.

These are explicitly phenomenological laws, not a molecular dynamics calculation.
Every residue is supplied by chemical reactions; joining spends energy and releases
water. No sequence is given copying ability and no whole sequence is ever copied.
Scaffolds bias local adsorption through a generic binary affinity approximation;
this is a toy template mechanism, not evidence of real glycine/alanine replication.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random

import torch

from .streams import planet_streams, tuples

from .chemistry import (BOND_ENERGY_J_PER_UMOL, ELEMENTS, POLYMER_MONOMERS,
                        SPECIES, SPECIES_INDEX)


@dataclass(frozen=True)
class AssemblyConfig:
    quantum_umol: float = 0.1
    join_probability: float = 0.035
    hydrolysis_probability: float = 0.006
    adsorption_probability: float = 0.03
    release_probability: float = 0.06
    hop_probability: float = 0.05
    affinity_gain: float = 0.6
    catalytic_gain: float = 2.0
    max_length: int = 32
    max_chains: int = 2048
    compartment_threshold_umol: float = 0.05


def folded_contacts(sequence):
    # Coarse hydrophobic-contact score with glycine-like flexibility. No motif.
    flexible = sequence.count(0) / max(1, len(sequence))
    contacts = sum(sequence[i] == sequence[j] == 1
                   for i in range(len(sequence)) for j in range(i + 3, len(sequence)))
    return contacts * (0.25 + flexible) / max(1, len(sequence) ** 2)


class Assemblies:
    def __init__(self, batch, grid, seed, config=None, profile="normal"):
        self.config = config or AssemblyConfig()
        if self.config.quantum_umol <= 0 or self.config.max_length < 2:
            raise ValueError("invalid assembly resolution")
        self.batch, self.grid, self.profile = batch, grid, profile
        self.stream_seeds, self.rngs = planet_streams(batch, seed, 'assemblies')
        self.chains = []
        self.next_ids = [1] * batch
        self.counts = [0] * batch
        self.events = {"joined": 0, "hydrolysed": 0, "adsorbed": 0,
                       "released": 0, "scaffold_release": 0, "capacity_hits": 0,
                       "length_hits": 0}
        self.events_by_planet = [{key: 0 for key in self.events} for _ in range(batch)]

    @property
    def rng(self):
        if self.batch != 1:
            raise ValueError("Use a planet-specific rngs entry")
        return self.rngs[0]

    @rng.setter
    def rng(self, value):
        if self.batch != 1:
            raise ValueError("Use a planet-specific rngs entry")
        self.rngs[0] = value

    def _event(self, batch, name, amount=1):
        self.events[name] += amount
        self.events_by_planet[batch][name] += amount

    def _new(self, batch, cell, sequence, tick):
        row = {"id": self.next_ids[batch], "batch": batch, "cell": list(cell),
               "sequence": list(sequence), "born": tick, "scaffold": None,
               "assembled_on": None}
        self.next_ids[batch] += 1
        self.counts[batch] += 1
        self.chains.append(row)
        return row

    def compartment_mask(self, state):
        if self.profile == "no_compartments" or "hexanoic_acid" not in SPECIES_INDEX:
            return torch.zeros_like(state.heat_j, dtype=torch.bool)
        return state.amount_umol[..., SPECIES_INDEX["hexanoic_acid"]] >= self.config.compartment_threshold_umol

    def step(self, state):
        cfg, q = self.config, self.config.quantum_umol
        idx = [SPECIES_INDEX[n] for n in POLYMER_MONOMERS]
        water = SPECIES_INDEX["H2O"]
        masks = self.compartment_mask(state).detach().cpu()
        by_cell = {}
        for row in self.chains:
            by_cell.setdefault((row["batch"], *row["cell"]), []).append(row)
        # All chemical tensors stay on device; sparse molecular graph events use a
        # deterministic CPU RNG. This boundary is measured in the benchmarks.
        for batch in range(self.batch):
            rng = self.rngs[batch]
            for x in range(self.grid):
                for y in range(self.grid):
                    key = (batch, x, y)
                    local = by_cell.get(key, [])
                    if self.counts[batch] >= cfg.max_chains:
                        self._event(batch, "capacity_hits")
                        continue
                    cat = 0.0 if self.profile == "no_catalysis" else sum(folded_contacts(r["sequence"]) for r in local)
                    if rng.random() >= min(0.5, cfg.join_probability * (1 + cfg.catalytic_gain * cat)):
                        continue
                    amounts = [float(state.amount_umol[batch, x, y, i]) for i in idx]
                    if sum(amounts) < 2 * q:
                        continue
                    choices = [i for i, value in enumerate(amounts) if value >= q]
                    if not choices:
                        continue
                    first = rng.choices(choices, weights=[amounts[i] for i in choices])[0]
                    amounts[first] -= q
                    choices = [i for i, value in enumerate(amounts) if value >= q]
                    if not choices:
                        continue
                    second = rng.choices(choices, weights=[amounts[i] for i in choices])[0]
                    cost = BOND_ENERGY_J_PER_UMOL * q
                    if float(state.photon_j[batch, x, y]) < cost:
                        continue
                    state.amount_umol[batch, x, y, idx[first]] -= q
                    state.amount_umol[batch, x, y, idx[second]] -= q
                    state.amount_umol[batch, x, y, water] += q
                    state.photon_j[batch, x, y] -= cost
                    new = self._new(batch, (x, y), [first, second], state.steps)
                    by_cell.setdefault(key, []).append(new)
                    self._event(batch, "joined")
        identities = {(r["batch"], r["id"]): r for r in self.chains}
        for row in list(self.chains):
            batch, (x, y), seq = row["batch"], row["cell"], row["sequence"]
            rng = self.rngs[batch]
            # Hydrolysis cuts one uniformly selected bond. Water is consumed, and
            # the bond's stored energy becomes heat. Both pieces remain material.
            if rng.random() < min(1.0, cfg.hydrolysis_probability * (len(seq) - 1)) \
                    and float(state.amount_umol[batch, x, y, water]) >= q:
                cut = rng.randrange(1, len(seq))
                pieces = [seq[:cut], seq[cut:]]
                if all(len(p) > 1 for p in pieces) and self.counts[batch] >= cfg.max_chains:
                    self._event(batch, "capacity_hits")
                    continue
                state.amount_umol[batch, x, y, water] -= q
                state.heat_j[batch, x, y] += BOND_ENERGY_J_PER_UMOL * q
                self.chains.remove(row)
                self.counts[batch] -= 1
                identities.pop((batch, row["id"]), None)
                for piece in pieces:
                    if len(piece) == 1:
                        state.amount_umol[batch, x, y, idx[piece[0]]] += q
                    else:
                        fragment = self._new(batch, (x, y), piece, state.steps)
                        identities[(batch, fragment["id"])] = fragment
                self._event(batch, "hydrolysed")
                continue
            scaffold = identities.get((batch, row["scaffold"]))
            if scaffold is None or scaffold["batch"] != batch or scaffold["cell"] != row["cell"]:
                row["scaffold"] = None
                scaffold = None
            elif rng.random() < cfg.release_probability:
                row["scaffold"] = None
                scaffold = None
                self._event(batch, "released")
                self._event(batch, "scaffold_release")
            if self.profile != "no_template" and row["scaffold"] is None \
                    and rng.random() < cfg.adsorption_probability:
                nearby = [r for r in self.chains if r["batch"] == batch and r["cell"] == row["cell"]
                          and r["id"] != row["id"] and len(r["sequence"]) >= len(seq)]
                if nearby:
                    parent = rng.choice(nearby)
                    row["scaffold"] = parent["id"]
                    row["assembled_on"] = parent["id"]
                    scaffold = parent
                    self._event(batch, "adsorbed")
            if len(seq) >= cfg.max_length:
                self._event(batch, "length_hits")
            elif rng.random() < cfg.join_probability:
                available = [float(state.amount_umol[batch, x, y, i]) for i in idx]
                weights = list(available)
                if scaffold is not None and self.profile != "no_template":
                    target = scaffold["sequence"][len(seq) % len(scaffold["sequence"])]
                    # Affinity changes rates, not identity: mismatched residues can bind.
                    weights[target] *= 1 + cfg.affinity_gain
                choices = [i for i, amount in enumerate(available) if amount >= q]
                cost = BOND_ENERGY_J_PER_UMOL * q
                if choices and float(state.photon_j[batch, x, y]) >= cost:
                    choice = rng.choices(choices, weights=[weights[i] for i in choices])[0]
                    state.amount_umol[batch, x, y, idx[choice]] -= q
                    state.amount_umol[batch, x, y, water] += q
                    state.photon_j[batch, x, y] -= cost
                    seq.append(choice)
                    self._event(batch, "joined")
            if row["scaffold"] is None and rng.random() < cfg.hop_probability \
                    * (0.1 if bool(masks[batch, x, y]) else 1.0):
                dx, dy = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
                row["cell"] = [(x + dx) % self.grid, (y + dy) % self.grid]
        # A scaffold that breaks or diffuses away ceases to guide its adsorbates.
        live = {(r["batch"], r["id"]): r for r in self.chains}
        for row in self.chains:
            parent = live.get((row["batch"], row["scaffold"]))
            if parent is None or parent["batch"] != row["batch"] or parent["cell"] != row["cell"]:
                row["scaffold"] = None

    def elemental_totals(self, device="cpu"):
        total = torch.zeros(self.batch, len(ELEMENTS), dtype=torch.float64, device=device)
        monomers = [SPECIES[SPECIES_INDEX[n]] for n in POLYMER_MONOMERS]
        water = SPECIES[SPECIES_INDEX["H2O"]]
        for row in self.chains:
            formula = [sum(monomers[i].elements[e] for i in row["sequence"])
                       - (len(row["sequence"]) - 1) * water.elements[e] for e in range(len(ELEMENTS))]
            total[row["batch"]] += torch.tensor(formula, device=device, dtype=torch.float64) * self.config.quantum_umol
        return total

    def energy_total(self, device="cpu"):
        total = torch.zeros(self.batch, dtype=torch.float64, device=device)
        mono = [SPECIES[SPECIES_INDEX[n]].chemical_j_per_umol for n in POLYMER_MONOMERS]
        water = SPECIES[SPECIES_INDEX["H2O"]].chemical_j_per_umol
        for row in self.chains:
            n = len(row["sequence"])
            total[row["batch"]] += (sum(mono[i] for i in row["sequence"])
                                    + (n - 1) * (BOND_ENERGY_J_PER_UMOL - water)) * self.config.quantum_umol
        return total

    def summary(self):
        lengths = [len(r["sequence"]) for r in self.chains]
        return {"chains": len(lengths), "residues": sum(lengths), "longest": max(lengths, default=0),
                "distinct_sequences": len({tuple(r["sequence"]) for r in self.chains}),
                "fold_contacts": sum(folded_contacts(r["sequence"]) for r in self.chains),
                "events": dict(self.events),
                "per_planet": [{"batch": b, "chains": self.counts[b],
                                "events": dict(self.events_by_planet[b]),
                                "censored": bool(self.events_by_planet[b]["capacity_hits"]
                                                 or self.events_by_planet[b]["length_hits"])}
                               for b in range(self.batch)],
                "capacity_scope": "per_planet", "life_verdict": "not_established",
                "censored": bool(self.events["capacity_hits"] or self.events["length_hits"])}

    def state_dict(self):
        return {"schema": "life10-assemblies-per-planet-v1", "config": asdict(self.config),
                "batch": self.batch, "grid": self.grid, "profile": self.profile,
                "chains": self.chains, "next_ids": list(self.next_ids),
                "events": dict(self.events), "events_by_planet": [dict(e) for e in self.events_by_planet],
                "stream_seeds": list(self.stream_seeds), "rngs": [rng.getstate() for rng in self.rngs]}

    @classmethod
    def from_dict(cls, data):
        if "rngs" not in data and data["batch"] != 1:
            raise ValueError("Legacy shared-RNG batch checkpoints require their frozen original implementation")
        obj = cls(data["batch"], data["grid"], [0] * data["batch"],
                  AssemblyConfig(**data["config"]), data["profile"])
        obj.chains, obj.events = data["chains"], dict(data["events"])
        obj.counts = [sum(r["batch"] == b for r in obj.chains) for b in range(obj.batch)]
        obj.next_ids = list(data.get("next_ids", [data.get("next_id", 1)]))
        obj.events_by_planet = [dict(e) for e in data.get("events_by_planet", [obj.events])]
        obj.stream_seeds = list(data.get("stream_seeds", obj.stream_seeds))
        for rng, state in zip(obj.rngs, data.get("rngs", [data.get("rng")])):
            rng.setstate(tuples(state))
        return obj
