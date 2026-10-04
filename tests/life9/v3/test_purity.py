"""Section 0 and 9 of PLANET-V3-SPEC for the whole v3 package: no forbidden mechanism in any v3 source file, and
behaviourally: two worlds that differ only in their bodies' lineage labels sense exactly the same, act exactly the
same and end the day in the same state (labels aside)."""
from __future__ import annotations

import ast
import inspect
import re
import textwrap
from pathlib import Path

import torch

from haishool.life9.v3 import body as B
from haishool.life9.v3 import world3 as w3

PKG = Path(w3.__file__).resolve().parent
SOURCES = sorted(PKG.glob("*.py"))

# (pattern, flags, what it would be): the section-9 mechanisms and their version-2 names
FORBIDDEN = (
    (r"\bACTIONS\b", 0, "a named action list (version 2's ACTIONS)"),
    (r"\b[A-Z_]*ACTIONS?\s*=\s*[\(\[]", 0, "a named action list"),
    (r"[\(\[]\s*\"(rest|forage|eat|drink|flee|hunt|mate|sleep|groom|share|strike|approach)\"\s*,", re.I,
     "a tuple of purposeful action names"),
    (r"innate", re.I, "an innate bias or default"),
    (r"forage", re.I, "the forage bias / background grazing"),
    (r"background_drink", re.I, "version 2's background drinking"),
    (r"time_budget", re.I, "version 2's time budget"),
    (r"pedigree", re.I, "kinship by pedigree"),
    (r"relatedness", re.I, "kinship by relatedness"),
    (r"seed_floor(?!_kg_m2=0\.0\))", re.I, "version 2's seed floor (only its removal, set to 0, may appear)"),
    (r"lifespan", re.I, "a supplied lifespan"),
    (r"maturity", re.I, "a supplied age of maturity"),
    (r"gestation", re.I, "a supplied gestation"),
    (r"allometr", re.I, "an allometric life-history law"),
    (r"kleiber", re.I, "the Kleiber basal rate as a law"),
    (r"reward", re.I, "a reward constant"),
    (r"default_outputs", 0, "a supplied policy"),
    (r"founder_habitat", 0, "habitability-biased founding"),
    (r"(?<![\"'])\bsynthetic_fallback\b", 0, "a synthetic planet in place of a world7 seed (a rescue the protocol "
                                            "removed; test_protocol checks the run path)"),
    (r"source\s*==\s*[\"']auto[\"']", 0, "version 2's automatic choice of the chain inputs"),
    (r"one_action", 0, "a one-action-per-day rule"),
    (r"\bis_(food|kin|prey|predator|edible|enemy|mate)\b", 0, "a class label"),
)


def test_the_package_is_there():
    names = {p.name for p in SOURCES}
    for need in ("world3.py", "__main__.py", "body.py", "brain3.py", "senses3.py", "manipulate.py", "patch.py",
                 "formation3.py", "optics.py"):
        assert need in names, need


def test_no_forbidden_mechanism_in_any_v3_source():
    hits = []
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        for pattern, flags, what in FORBIDDEN:
            for m in re.finditer(pattern, text, flags):
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{path.name}:{line}: {m.group(0)!r} ({what})")
    assert not hits, "\n".join(hits)


def test_the_senses_and_the_brain_never_receive_lineage():
    """World3 builds the senses' input from body.sense_view only, and sense_view carries no lineage id."""
    fn = ast.parse(textwrap.dedent(inspect.getsource(w3.World3.sense))).body[0]
    if isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant):
        fn.body = fn.body[1:]                             # the docstring may name what is left out
    src = ast.unparse(fn)
    for word in ("uid", "founder", "parent", "generation", "lineage", "age_bouts"):
        assert word not in src, word
    view = B.sense_view(B.found(1, 4, 2, 100.0, torch.Generator().manual_seed(0), in_dim=4, hidden=8, brain=False))
    assert not set(view) & set(B.LINEAGE_FIELDS) and "age_bouts" not in view


def _world():
    cfg = w3.V3Config(G=8, patches=2, patch_m=256.0, cells=16, capacity=96, founders=32, items=128, fires=8,
                      hidden=16, k_founder=(2, 8), bouts=4, climate_fast_chunks=1, climate_slow_chunks=1,
                      bio_spin_days=20, bio_spin_segments=1, veg_spin_days=20, veg_spin_bouts=2, source="earth")
    return w3.World3(cfg, [0])


def _permute_labels(world, gen):
    """Permute every lineage label (uid, parent, founder, generation) among each arena's living bodies."""
    b = world.b
    for a in range(b.shape[0]):
        idx = b.alive[a].nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(idx.numel(), generator=gen)]
        for name in B.LINEAGE_FIELDS:
            t = getattr(b, name)
            t[a, idx] = t[a, perm].clone()


def test_permuted_lineage_labels_give_identical_senses_and_the_same_world(monkeypatch):
    base = _world()
    base.run(1)                                           # some bodies are children by now
    one = w3.World3.from_state(base.state_dict())
    two = w3.World3.from_state(base.state_dict())
    _permute_labels(two, torch.Generator().manual_seed(5))
    assert any(not torch.equal(getattr(one.b, n), getattr(two.b, n)) for n in B.LINEAGE_FIELDS)
    seen = {id(one): [], id(two): []}
    real = w3.World3.sense

    def spy(self, scene, **kw):
        x = real(self, scene, **kw)
        seen[id(self)].append(x.clone())
        return x

    monkeypatch.setattr(w3.World3, "sense", spy)
    one.run(1)
    two.run(1)
    assert len(seen[id(one)]) == one.config.bouts and len(seen[id(two)]) == len(seen[id(one)])
    for x1, x2 in zip(seen[id(one)], seen[id(two)]):
        assert torch.equal(x1, x2)
    # and everything that is not a label came out the same
    for name in B.STATE_FIELDS:
        if name in B.LINEAGE_FIELDS:
            continue
        assert torch.equal(getattr(one.b, name), getattr(two.b, name)), name
    for k in one.b.genome:
        assert torch.equal(one.b.genome[k], two.b.genome[k]), k
    assert torch.equal(one.b.wh_live, two.b.wh_live) and torch.equal(one.ps.plant, two.ps.plant)
    assert torch.equal(one.items.mass, two.items.mass)


def test_the_rules_in_effect_have_no_floor_and_no_target():
    """Section 0 for the version-2 functions World3 calls (rules in effect, not only source text): the coarse global
    biosphere runs without its seed rain, and the patches' nitrogen fixation follows growth, not a starting stock."""
    from haishool.life9.v3 import patch as pt
    w = _world()
    assert w.Bp["rules"].seed_floor_kg_m2 == 0.0
    src = inspect.getsource(pt.step)
    assert "nutrient_start" not in src and "bnf_share" in src
