import json
from argparse import Namespace
import pytest

from haishool.life10 import __main__ as cli


def args(tmp_path):
    return Namespace(out=tmp_path, threads=1, samples=1000000, steps=3, repeat=0,
                     profile="normal", grid=2, seeds=[0], device="cpu", chains=None, every=1)


def test_no_star_outcome_is_completed_and_archived(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "build_chain", lambda *a: {"cosmic_seed": 0, "outcome": "no_star",
                                                      "planets": [], "errors": []})
    cli.run(args(tmp_path))
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["status"] == "complete" and m["chain_outcomes"] == {"0": "no_star"}
    assert json.loads((tmp_path / "chains.json").read_text())[0]["outcome"] == "no_star"


def test_every_formed_planet_is_run_even_when_hot_or_cold(tmp_path, monkeypatch):
    def chain(*a):
        return {"cosmic_seed": 0, "outcome": "planets_formed", "errors": [], "planets": [
            {"id": i, "type": "gas" if i else "rocky", "temperature_k": t, "flux_earth": .5,
             "atom_inventory": {"C": 10, "H": 200, "N": 10, "O": 40}, "chemistry_seed": i}
            for i, t in enumerate([60., 1000.])]}
    monkeypatch.setattr(cli, "build_chain", chain)
    cli.run(args(tmp_path))
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["status"] == "complete" and m["planets"] == 2
    assert m["checkpoint_roundtrip_verified"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert {p["temperature_k"] for p in summary["planets"]} == {60., 1000.}


@pytest.mark.parametrize("temperature", [0., -10., float("nan")])
def test_invalid_temperatures_are_rejected_without_repair(temperature):
    with pytest.raises(ValueError, match="temperature"):
        cli.initialize_batch([{"temperature_k": temperature}], cli.ChemicalConfig(), "cpu")


def test_existing_failed_archive_is_never_overwritten(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text('{"status":"failed","original":true}')
    before = p.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        cli.main(["run", "--seeds", "0", "--out", str(tmp_path)])
    assert p.read_bytes() == before
