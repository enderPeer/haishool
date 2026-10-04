"""Report every run and paired controls, without selecting successful seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import write_json


def repeated_population_planets(records):
    """A repeated population count, not persistence of individual molecules."""
    if len(records) < 2 or records[-1].get("step", 0) <= records[-2].get("step", 0):
        return []
    eligible = []
    for record in records[-2:]:
        found = {}
        for planet in record.get("planets", []):
            key = (planet["seed"], planet["planet"])
            if key in found:
                raise ValueError(f"Duplicate planet in sampled summary: {key}")
            found[key] = planet.get("chains", 0) >= 10 and planet.get("longest", 0) >= 4
        eligible.append({key for key, passed in found.items() if passed})
    return [{"seed": seed, "planet": planet}
            for seed, planet in sorted(eligible[0] & eligible[1])]


def pair_invalid_reasons(normal, control):
    reasons = []
    for label, row in (("normal", normal), ("control", control)):
        if row["status"] != "complete":
            reasons.append(label + "_incomplete")
        if row["censored"]:
            reasons.append(label + "_censored")
        if row["chain_errors"]:
            reasons.append(label + "_chain_errors")
        required = row["requested_steps"]
        executed = row["steps"]
        no_planets = row["planets"] == 0 and row["stop_reason"] == "no_formed_planets"
        if required is None or executed != (0 if no_planets else required):
            reasons.append(label + "_step_budget_mismatch")
    for field in ("source_sha256", "chain_inputs_sha256"):
        if not normal[field] or not control[field] or normal[field] != control[field]:
            reasons.append(field + "_mismatch_or_missing")
    for field in ("seeds", "repeat", "physical_config", "device", "requested_steps", "steps"):
        if normal[field] != control[field]:
            reasons.append(field + "_mismatch")
    return reasons


def collect(root):
    root = Path(root)
    rows, groups, errors = [], {}, []
    for path in sorted(root.rglob("manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("schema") != "life10-run-v1":
                continue
            summary_path = path.parent / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            sequences = summary.get("assemblies", {})
            records = []
            history = path.parent / "summary.jsonl"
            if history.exists():
                records = [json.loads(l) for l in history.read_text().splitlines() if l.strip()]
            # A population screen is a measurement definition, not a reward,
            # selection criterion, or evidence that individual molecules persisted.
            repeated = repeated_population_planets(records)
            sampled_censorship = any(r.get("assemblies", {}).get("censored", False)
                                     or r.get("compartments", {}).get("censored", False)
                                     for r in records + [summary])
            row = {"path": str(path.parent.resolve()), "status": manifest["status"],
                   "seeds": manifest["seeds"], "repeat": manifest["config"]["repeat"],
                   "profile": manifest["config"]["profile"], "device": manifest["device"],
                   "systems": manifest.get("systems", 0), "planets": manifest.get("planets", 0),
                   "outcomes": manifest.get("chain_outcomes", {}),
                   "steps": manifest.get("executed_steps", 0), "chains": sequences.get("chains", 0),
                   "longest": sequences.get("longest", 0), "events": sequences.get("events", {}),
                   "compartments": summary.get("compartments", {}),
                   "repeated_population_screen": bool(repeated),
                   "repeated_population_planets": repeated,
                   "censored": manifest.get("censored", False) or sampled_censorship,
                   "chain_errors": manifest.get("chain_errors", []),
                   "chain_inputs_sha256": manifest.get("chain_inputs_sha256"),
                   "physical_config": {k: v for k, v in manifest["config"].items() if k != "profile"},
                   "requested_steps": manifest["config"].get("steps"),
                   "stop_reason": manifest.get("stop_reason"),
                   "ledger": manifest.get("ledger_worst_relative", {}),
                   "source_sha256": manifest["source"]["sha256"],
                   "life_verdict": "not_established"}
            rows.append(row)
            # Group siblings by lane, then validate identity explicitly. Including
            # seeds/repeat in the key would silently omit mismatched controls.
            key = path.parent.parent.as_posix()
            groups.setdefault(key, {}).setdefault(row["profile"], []).append(row)
            if manifest.get("chain_errors") or manifest["status"] != "complete":
                errors.append({"path": row["path"], "chain_errors": manifest.get("chain_errors"),
                               "error": manifest.get("error")})
        except (ValueError, KeyError, OSError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    paired = []
    for lane, group in groups.items():
        normals = group.get("normal", [])
        for profile, controls in group.items():
            if profile == "normal":
                continue
            for normal in normals:
                for control in controls:
                    reasons = pair_invalid_reasons(normal, control)
                    if len(normals) > 1 or len(controls) > 1:
                        reasons.append("duplicate_profile_in_lane")
                    valid = not reasons
                    paired.append({"lane": lane, "repeat": normal["repeat"], "control": profile,
                                   "normal_path": normal["path"], "control_path": control["path"],
                                   "valid": valid, "invalid_reasons": reasons,
                                   "chain_difference": normal["chains"] - control["chains"] if valid else None,
                                   "longest_difference": normal["longest"] - control["longest"] if valid else None})
    return {"schema": "life10-report-v1", "runs": rows, "paired_controls": paired,
            "errors": errors, "completed": sum(r["status"] == "complete" for r in rows),
            "source_identities": sorted({r["source_sha256"] for r in rows}),
            "repeated_population_screen_passes": sum(r["repeated_population_screen"] for r in rows),
            "life_verdict": "not_established",
            "limits": ["Short bounded pilot, not an estimate of real-world abiogenesis probability.",
                       "Rates/energies, scaffold affinities and compartment dynamics are phenomenological.",
                       "Prebiotic cosmic prefix has declared coarse state-allocation boundaries.",
                       "A repeated population screen does not track individual molecular persistence, reproduction, heredity or biological life."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = collect(args.root)
    write_json(args.out, result)
    print(json.dumps({k: result[k] for k in ("completed", "repeated_population_screen_passes", "life_verdict", "errors")}))


if __name__ == "__main__":
    main()
