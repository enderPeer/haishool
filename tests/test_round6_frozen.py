"""Round 6 is frozen: the version-5 models were trained on these rollouts.

Corrections to a level are options (``rules=7``); without them every rollout and every line must
stay exactly as it was when ``tests/data/round6-frozen.json`` was written.
"""
import hashlib
import importlib
import json
import random
from pathlib import Path

import pytest

FROZEN = json.loads((Path(__file__).parent / "data" / "round6-frozen.json").read_text())["hashes"]


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


@pytest.mark.parametrize("key", sorted(FROZEN))
def test_default_output_is_unchanged(key):
    sim, seed = key.split(":")
    m = importlib.import_module(f"haishool.cosmos.{sim}")
    r = m.simulation().run(int(seed), **m.random_params(random.Random(int(seed))))
    assert _h([r.params, r.steps, r.summary]) == FROZEN[key]["rollout"], f"{key}: the default rollout changed"
    assert _h([ln.text for ln in m.lines(r)]) == FROZEN[key]["lines"], f"{key}: the default lines changed"
