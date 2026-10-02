"""Bounded experimental ecology parameters; time is measured in simulation ticks.

life8 additions to Matrix 2cf045a: ``organ_flip_rate`` (binary voice/hearing organs
mutate at their own small rate instead of the full ``mutation_rate``), ``log``
("full" keeps every per-tick observation/transition record, "compact" keeps only
births, deaths, signals sent/heard, tool and social events) and
``birth_transfer_in_reward`` (an intervention control only: True restores Matrix's
bug of booking the birth transfer against the parent's unrelated action).

``channel`` is a signal-channel intervention for discovery measurements: "normal"
delivers the (possibly noisy) symbol; "scrambled" replaces every delivered symbol by
a uniform random one from a dedicated RNG kept in the state; "masked" delivers the
message but hides its symbol from the controller. Costs, range and delivery are
identical in all three.
"""
from dataclasses import asdict, dataclass
import math

LOG_MODES = ("full", "compact")
CHANNELS = ("normal", "scrambled", "masked")


@dataclass(frozen=True)
class Config:
    width: float = 32.
    height: float = 24.
    population: int = 24
    food_patches: int = 60
    initial_objects: int = 36
    population_limit: int = 128
    object_limit: int = 512
    learning: bool = True
    tools: bool = True
    communication: bool = True
    culture: bool = True
    mutation_rate: float = .08
    mutation_scale: float = .12
    initial_energy: float = 40.
    max_energy: float = 60.
    base_metabolism: float = .045
    action_cost: float = .08
    food_energy: float = 4.
    food_capacity: float = 24.
    food_regrowth: float = .10
    interaction_range: float = 1.5
    signal_noise: float = .08
    message_ttl: int = 8
    signal_cost: float = .25
    culture_cost: float = .35
    safe_touch_temperature: float = 40.
    thermal_touch_conductance: float = .08
    thermal_injury_per_energy: float = .02
    maturity_age: int = 35
    max_age: int = 600
    reproduction_threshold: float = 34.
    offspring_energy: float = 12.
    reproduction_cost: float = 4.
    birth_cooldown: int = 45
    automatic_reproduction: bool = True
    inventory_limit: int = 4
    memory_limit: int = 32
    organ_flip_rate: float = .005
    log: str = "full"
    birth_transfer_in_reward: bool = False
    channel: str = "normal"
    # bridge (round-7 hand-off): ambient light scales every individual's sight radius; 1.0 = Matrix
    visibility: float = 1.
    # genome (heritable behaviour, see genome.py): "frozen" = newborns start blank as in
    # Matrix (the control); "evolving" = innate controller biases, learning rate and
    # exploration are inherited with mutation and seed the newborn's controller.
    genome: str = "frozen"
    reproduction: str = "asexual"  # "sexual": a nearby willing partner, both pay, genes recombine
    age_effects: bool = False      # juvenile/senescent ability curve, heritable lifespan and maturity
    bias_mutation_rate: float = .05
    bias_mutation_scale: float = .25
    # living-world ecology (round 8, see docs/life8/README.md "Living-world ecology").
    # Every field is neutral at its default; LIVING below is the measured preset.
    bloom_interval: int = 0          # ticks between short-lived rich patches (0 = none)
    bloom_amount: float = 16.        # food units in a new bloom
    bloom_ttl: int = 25              # ticks before an uneaten bloom spoils
    bloom_richness: float = 2.       # energy per food unit at a bloom, x food_energy
    bloom_partners: int = 1          # individuals within reach needed for a bloom bite (2 = cooperative)
    predators: int = 0               # hazards that chase and wound the nearest individual
    predator_speed: float = .6       # world units per tick (a default organism steps about 0.87)
    predator_mass: float = 1.5       # wound scales with predator_mass / (predator_mass + victim body_mass)
    predator_damage: float = .3      # health lost per strike at equal masses
    predator_sight: float = 6.       # how far a predator finds prey
    predator_visibility: float = .6  # prey see a predator within this fraction of their sight radius
    predator_cooldown: int = 4       # ticks between strikes of one predator
    permanent_injury: float = 0.     # share of every wound that lowers the health ceiling for good
    health_ability: bool = False     # health scales eating, striking and sight by (0.5 + 0.5 health)
    strike_injury: float = .08       # health per unit damage of an organism-on-organism strike
    injury_mass_scaling: bool = False  # strike wounds x attacker body_mass / target body_mass
    call_power: bool = False         # voice power is its own heritable gene: reach = call_reach x power
    call_reach: float = 8.           # call reach at power 1 (sight at sensing 1 is 5)
    kin_credit: float = 0.           # sender reward += kin_credit x relatedness x hearer's reward
    kin_window: int = 8              # ticks after hearing during which the hearer's reward is credited
    receiver_features: bool = False  # heard bearing, message age band, sender kinship band, own live call
    reciprocity_features: bool = False  # directed per-peer energy tallies, nearest neighbour's balance band
    ambient_temperature: float = 20.  # objects cool toward this (Celsius); 20 = Matrix

    def __post_init__(self):
        integer_fields = ("population", "food_patches", "initial_objects", "population_limit", "object_limit",
                          "message_ttl", "maturity_age", "max_age", "birth_cooldown", "inventory_limit", "memory_limit",
                          "bloom_interval", "bloom_ttl", "bloom_partners", "predators", "predator_cooldown", "kin_window")
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        flag_fields = ("learning", "tools", "communication", "culture", "automatic_reproduction",
                       "birth_transfer_in_reward", "age_effects",
                       "health_ability", "injury_mass_scaling", "call_power", "receiver_features", "reciprocity_features")
        if self.genome not in ("frozen", "evolving") or self.reproduction not in ("asexual", "sexual"):
            raise ValueError("genome must be frozen|evolving and reproduction asexual|sexual")
        if self.log not in LOG_MODES:
            raise ValueError("log must be one of " + ", ".join(LOG_MODES))
        if self.channel not in CHANNELS:
            raise ValueError("channel must be one of " + ", ".join(CHANNELS))
        for name in flag_fields:
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name, value in asdict(self).items():
            if name not in integer_fields and name not in flag_fields and name not in ("log", "channel", "genome", "reproduction"):
                if name == "ambient_temperature":
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not -80 <= value <= 200:
                        raise ValueError("ambient_temperature must lie within -80 and 200 C")
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name} must be finite and nonnegative")
        if min(self.width, self.height, self.food_energy, self.max_energy) <= 0:
            raise ValueError("world dimensions and energy units must be positive")
        if not 0 <= self.mutation_rate <= 1 or not 0 <= self.signal_noise <= 1 or self.mutation_scale > 1 \
                or not 0 <= self.organ_flip_rate <= 1 or not self.bias_mutation_rate <= 1:
            raise ValueError("probabilities/scales must lie within zero and one")
        if self.population > self.population_limit or self.initial_objects > self.object_limit:
            raise ValueError("initial counts exceed capacity")
        if self.population_limit > 4096 or self.object_limit > 20000 or self.food_patches > 4096:
            raise ValueError("configuration exceeds prototype compute bounds")
        if self.max_age < 1 or self.inventory_limit < 1 or self.memory_limit < 1 or self.message_ttl < 1:
            raise ValueError("lifetime and memory/inventory limits must be positive")
        if self.initial_energy > self.max_energy or self.offspring_energy > self.max_energy:
            raise ValueError("initial/offspring energy exceeds max_energy")
        if self.permanent_injury > 1 or self.predators > 64 or self.kin_credit > 4 or self.bloom_ttl < 1:
            raise ValueError("permanent_injury <= 1, predators <= 64, kin_credit <= 4, bloom_ttl >= 1")
        if self.reproduction_threshold < self.offspring_energy + self.reproduction_cost:
            raise ValueError("reproduction threshold cannot fund offspring and birth cost")

    def ecology_on(self):
        """True when any living-world mechanism is switched on (the state then carries an ``ecology`` record)."""
        return bool(self.bloom_interval or self.predators or self.permanent_injury or self.call_power
                    or self.kin_credit or self.receiver_features or self.reciprocity_features)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("configuration must be an object")
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(f"unknown or invalid configuration: {exc}") from exc


# The living-world preset (measured in docs/life8/README.md, "Living-world ecology"):
# food-regulated (30 patches), rich short-lived blooms to approach, predators to flee,
# calls that carry further than sight with their own heritable power, a kin credit
# route for senders, the receiver and reciprocity features, real injuries, and the
# evolving genome. Nothing here rewards any symbol.
LIVING = {"food_patches": 30, "bloom_interval": 8, "bloom_amount": 16., "bloom_ttl": 25, "bloom_richness": 2.,
          "predators": 2, "predator_speed": .6, "predator_mass": 1.5, "predator_damage": .25, "predator_sight": 5.,
          "predator_visibility": .6, "predator_cooldown": 15, "permanent_injury": .25, "health_ability": True,
          "strike_injury": .2, "injury_mass_scaling": True, "call_power": True, "call_reach": 8., "kin_credit": 1.,
          "kin_window": 8, "receiver_features": True, "reciprocity_features": True, "genome": "evolving"}


def living_config(**overrides):
    """Config(**LIVING) with ``overrides`` replacing single fields."""
    return Config(**{**LIVING, **overrides})
