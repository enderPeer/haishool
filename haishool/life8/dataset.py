"""Export actual local experience; do not turn observer summaries into invented facts."""
import json
from pathlib import Path

from .checkpoint import RUN_SCHEMA, file_digest, write_json


def export_transitions(run, out):
    run, out = Path(run).resolve(), Path(out).resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != RUN_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("dataset export requires a completed verified run")
    if manifest.get("config", {}).get("log", "full") != "full":
        raise ValueError("compact-log runs keep no transition records; rerun with log='full' to export experience")
    for name in ("events.jsonl", "checkpoint.json"):
        if file_digest(run / name) != manifest.get("files", {}).get(name):
            raise ValueError("run artifact hash mismatch: " + name)
    if out.exists() or out.with_suffix(out.suffix + ".manifest.json").exists():
        raise FileExistsError("dataset output already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (run / "events.jsonl").open(encoding="utf-8") as source, out.open("x", encoding="utf-8") as target:
        for line in source:
            event = json.loads(line)
            if event.get("type") != "transition":
                continue
            record = {key: event[key] for key in ("tick", "agent", "observation", "action", "reward", "next_observation", "done")}
            record["world_seed"] = manifest["seed"]
            record["executed"] = event.get("executed", True)
            record["reward_components"] = event.get("reward_components", {})
            record["source_sha256"] = manifest["source"]["sha256"]
            record["timing"] = event.get("timing")
            target.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
            count += 1
    receipt = {"schema": "life8-experience-v2", "examples": count, "sha256": file_digest(out),
               "source_run_manifest_sha256": file_digest(run / "manifest.json"),
               "source": manifest["source"], "world_seed": manifest["seed"],
               "scope": "Local observations, selected actions, execution flags and measured outcomes; no old factual corpus or technology labels added. An organism killed before its turn has executed=false.",
               "split_guidance": "Keep entire world seeds and their continuation branches in one split; neighboring transitions are not independent test cases."}
    write_json(out.with_suffix(out.suffix + ".manifest.json"), receipt)
    return {"path": str(out), "examples": count, "sha256": receipt["sha256"]}
