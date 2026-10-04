"""life9 configuration. Every field is a modelling assumption and is recorded with each run."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

CHANNELS = ("normal", "masked", "scrambled", "deaf")


@dataclass(frozen=True)
class Config9:
    # batch and space
    worlds: int = 8                  # independent worlds simulated together (paired replicates)
    capacity: int = 128              # agent slots per world (a hard cap, reported when reached)
    founders: int = 48
    size: float = 48.0               # torus side in x and y (world units)
    grid: int = 64                   # terrain cells per side
    relief: float = 6.0              # height of the highest point above the lowest
    octaves: int = 4
    los_samples: int = 6
    # bodies and senses
    eye_height: float = 0.6
    sight: float = 6.0
    height_sight_gain: float = 0.6   # sight x (1 + gain x ground height / relief): high ground sees further
    predator_detect: float = 0.6     # predators are seen only within this fraction of sight (stealth)
    sectors: int = 8                 # egocentric sectors around the heading
    max_turn: float = 0.8            # radians per tick
    max_speed: float = 1.0
    reach: float = 1.2
    # energy
    initial_energy: float = 30.0
    max_energy: float = 60.0
    base_metabolism: float = 0.04
    neuron_cost: float = 0.0015      # per active hidden unit per tick: brain size is paid for
    move_cost: float = 0.06          # x speed^2
    climb_cost: float = 0.15         # x height gained
    lifespan: int = 600
    # food (fertile valleys, poor hills)
    food_patches: int = 32
    food_capacity: float = 30.0
    food_regrowth: float = 0.25      # per tick on the lowest ground; x (0.25 + 0.75 fertility)
    bite: float = 2.0
    # sound
    vocal_dims: int = 4              # a call is a continuous vector, not one of four symbols
    call_cost: float = 0.12          # x loudness
    call_range: float = 16.0
    sound_occlusion: float = 0.5     # sound behind a hill keeps this fraction (sight keeps none)
    silence: float = 0.05            # loudness below this is silence (free, not heard)
    channel: str = "normal"          # normal | masked (presence only) | scrambled (random content) | deaf
    oracle: bool = False             # diagnostic only: callers who see a predator emit a fixed vector
    # predators: ambush hunters with limited stamina
    predators: int = 2
    predator_speed: float = 1.1
    predator_sight: float = 8.0
    predator_strike: float = 0.9
    predator_damage: float = 0.55
    predator_stamina: int = 12
    predator_rest: int = 20
    # brain (capacity is a heritable, paid-for gene, up to `hidden`)
    hidden: int = 128                # maximum brain units (raised from 32 in update 9, version 2)
    initial_hidden: int = 16
    plasticity: bool = True          # reward-modulated Hebbian change within a life; never inherited
    weight_clip: float = 3.0
    # reproduction and inheritance
    reproduction_threshold: float = 40.0
    offspring_energy: float = 15.0
    birth_cost: float = 3.0
    maturity: int = 30
    birth_cooldown: int = 40
    mutation_scale: float = 0.05
    hidden_mutation_rate: float = 0.05

    def __post_init__(self):
        if self.channel not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}")
        if not 2 <= self.initial_hidden <= self.hidden:
            raise ValueError("initial_hidden must lie within 2 and hidden")
        if not 1 <= self.founders <= self.capacity:
            raise ValueError("founders must lie within 1 and capacity")

    @property
    def in_dim(self):
        return self.sectors * (3 + self.vocal_dims) + 6

    @property
    def out_dim(self):
        return 4 + self.vocal_dims       # turn, speed, eat, loudness, vocal vector

    def to_dict(self):
        return asdict(self)

    def but(self, **changes):
        return replace(self, **changes)
