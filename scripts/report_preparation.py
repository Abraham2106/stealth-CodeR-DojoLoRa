"""Publish only aggregate evidence and hashes; raw reviews and source remain local."""
import argparse
import json
from collections import Counter
from pathlib import Path

from coder.common import write_json
from coder.model import PROFILES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    root, destination = Path(args.data), Path(args.output)
    collection = json.loads((root / "raw/collection-report.json").read_text(encoding="utf-8"))
    prepared_path = root / "prepared/preparation-report.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8")) if prepared_path.exists() else None
    inventory = json.loads((root / "raw/inventory.json").read_text(encoding="utf-8"))
    metadata = {"collection": {k: v for k, v in collection.items() if k != "errors"},
                "collection_error_count": len(collection["errors"]), "preparation": prepared,
                "model_profiles": PROFILES, "training_executed": False,
                "deployment_4gb_verified": False,
                "artifact_location": "Local data/raw and data/prepared; excluded from Git",
                "license_distribution": dict(Counter(str(r["license"]) for r in inventory))}
    write_json(destination / "preparation-summary.json", metadata)
    write_json(destination / "source-manifest.json", inventory)
    if prepared:
        write_json(destination / "split-hashes.json", {"hashes": prepared["hashes"],
                   "raw_record_hash": prepared["raw_record_hash"], "sizes": prepared["sizes"]})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(figsize=(7, 4))
    values = [collection["bot_inline_comments"], collection["human_inline_comments"]]
    axes.bar(["Bots (weak supervision)", "Humans"], values, color=["#5576aa", "#58906c"])
    axes.set(ylabel="Inline review comments", title=f"Dojo corpus · {collection['prs']} PRs · {collection['repositories']} repositories")
    axes.set_ylim(0, max(values) * 1.2 + 1)
    for i, count in enumerate(values):
        axes.text(i, count + max(values) * .025, str(count), ha="center")
    figure.text(.02, .02, "Public GitHub acquisition; counts do not establish comment correctness.", fontsize=9)
    figure.tight_layout(rect=(0, .08, 1, 1))
    for extension in ("png", "pdf", "svg"):
        figure.savefig(destination / f"corpus.{extension}", dpi=300)
    svg = destination / "corpus.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n", encoding="utf-8")
    plt.close(figure)
    write_json(destination / "corpus-plot-data.json", {"categories": ["bots", "humans"], "counts": values,
                                                       "denominator": collection["inline_comments"]})
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
