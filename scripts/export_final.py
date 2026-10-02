"""Export completed checkpoints and reports without re-opening sealed evaluations."""
import hashlib
import json
import tarfile
from pathlib import Path

import torch

from haishool.student import load
from haishool.final_run import ask_many

ROOT = Path("runs/final-v5")
OUT = Path("exports/final-v5")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    assert json.loads((ROOT / "status.json").read_text())["phase"] == "complete"
    summary = json.loads((ROOT / "summary.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"artifacts": {}, "reports": {}, "verification": "float32 tensors equal original; exported files loaded and generated on Adler"}
    for size, selected in summary["models"].items():
        source = Path(selected["selected"]) / "student.pt"
        assert sha(source) == selected["sha256"]
        sealed = json.loads((ROOT / f"{size}-sealed.json").read_text())
        assert sealed["fingerprint"]["checkpoint"] == selected["sha256"]
        for phase, source in (("selected", source), ("production", Path(selected["production"]))):
            original = torch.load(source, map_location="cpu", weights_only=True)
            target = OUT / f"haishool-v5-{size}-{phase}.pt"
            torch.save({k: original[k] for k in ("model_state", "gpt_config", "itos")}, target)
            exported = torch.load(target, map_location="cpu", weights_only=True)
            assert original["gpt_config"] == exported["gpt_config"] and original["itos"] == exported["itos"]
            assert all(torch.equal(t, exported["model_state"][k]) for k, t in original["model_state"].items())
            del exported, original
            model, vocab = load(target, "cuda")
            # These are sampled previously trained old questions, never sealed questions.
            from haishool.truth import split_line
            pairs = [split_line(s) for s in (source.parent / "old-recall.txt").read_text().splitlines()[:3]]
            got = ask_many(model, vocab, [p for p, _ in pairs], "cuda", max_new=64)
            assert all(isinstance(s, str) and s for s in got)
            manifest["artifacts"][target.name] = {
                "sha256": sha(target), "bytes": target.stat().st_size,
                "source": str(source), "source_sha256": sha(source),
                "config": model.config.to_json(), "phase": phase,
                "smoke": [{"prompt": p, "expected": g, "got": a} for (p, g), a in zip(pairs, got)],
                "sealed_report": f"{size}-sealed.json" if phase == "selected" else None}
            del model
            torch.cuda.empty_cache()
    reports = [p for p in ROOT.rglob("*.json") if "cache" not in p.parts]
    with tarfile.open(OUT / "reports.tar.gz", "w:gz") as archive:
        for p in sorted(reports):
            name = str(p.relative_to(ROOT)).replace("\\", "/")
            archive.add(p, arcname=name)
            manifest["reports"][name] = sha(p)
        archive.add("data/final-v5/report.json", arcname="dataset-report.json")
        manifest["reports"]["dataset-report.json"] = sha(Path("data/final-v5/report.json"))
    manifest["report_archive_sha256"] = sha(OUT / "reports.tar.gz")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"exported": list(manifest["artifacts"]), "reports": len(manifest["reports"])}))


if __name__ == "__main__":
    main()
