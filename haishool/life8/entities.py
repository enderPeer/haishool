"""Persistent individual and material records, using only JSON-serializable state."""
from dataclasses import asdict, dataclass, field


@dataclass
class Anatomy:
    body_mass: float = 1.
    speed: float = 1.
    sensing: float = 1.
    manipulation: float = 1.
    metabolism: float = 1.
    voice: int = 1
    hearing: int = 1

    @property
    def sensor_range(self):
        return 2.5 + 2.5 * self.sensing

    @property
    def move_distance(self):
        return self.speed / (1 + .15 * self.body_mass)

    @property
    def metabolic_factor(self):
        return self.metabolism * self.body_mass * (1 + .15 * self.sensing ** 2 + .1 * self.manipulation + .04 * (self.voice + self.hearing))

    def to_dict(self):
        # Same result as dataclasses.asdict for these scalar genes, without its recursion cost.
        return {"body_mass": self.body_mass, "speed": self.speed, "sensing": self.sensing,
                "manipulation": self.manipulation, "metabolism": self.metabolism,
                "voice": self.voice, "hearing": self.hearing}


@dataclass
class Agent:
    id: int
    parent_id: int | None
    position: list[float]
    age: int
    energy: float
    health: float
    anatomy: Anatomy = field(default_factory=Anatomy)
    inventory: list[int] = field(default_factory=list)
    controller: dict = field(default_factory=dict)
    memory: dict = field(default_factory=lambda: {"messages": [], "social": {}, "observations": []})
    last_action: str = "rest"
    last_outcome: float = 0.
    born_tick: int = 0
    cooldown: int = 0
    lifespan: int = 600
    # Heritable behaviour and life history (see genome.py): innate controller biases,
    # learning rate, exploration, lifespan and maturity genes. {} = the default genome.
    genome: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["anatomy"] = Anatomy(**data["anatomy"])
        data.setdefault("genome", {})  # life8 states written before the genome existed
        return cls(**data)


@dataclass
class Artifact:
    id: int
    material: str
    position: list[float]
    mass: float = 1.
    sharpness: float = .1
    bond_strength: float = 0.
    temperature: float = 20.
    components: list[int] = field(default_factory=list)
    holder: int | None = None
    maker_id: int | None = None
    properties: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class FoodPatch:
    id: int
    position: list[float]
    amount: float
    capacity: float
    hardness: float = 0.
    regrowth: float = .1
    opening: float = 0.

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)
