"""Plot observed A imitation metrics; never assert code-review correctness."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/training-A"))
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text(encoding="utf-8"))
    history = json.loads((args.run / "history.json").read_text(encoding="utf-8"))
    if not report.get("training_executed") or not report.get("fresh_reload_verified"):
        raise ValueError("Observed training and reload evidence required")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True, exist_ok=True)
    baseline, candidate = report["baseline_test_reloaded"], report["candidate_test_reloaded"]
    if (baseline["examples"], baseline["completion_tokens"]) != (candidate["examples"], candidate["completion_tokens"]):
        raise ValueError("Baseline and candidate denominators differ")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot([h["epoch"] for h in history], [h["validation"]["mean_token_nll"] for h in history], marker="o")
    validation = history[0]["validation"]
    axes[0].set(xlabel="Epoch", ylabel="Completion token NLL", xticks=[h["epoch"] for h in history],
                title=f"Validation · n={validation['examples']} · {validation['completion_tokens']} tokens")
    values = [baseline["mean_token_nll"], candidate["mean_token_nll"]]
    axes[1].bar(["Frozen backbone", "LoRA A"], values, color=["#8797a9", "#5576aa"])
    axes[1].set(ylabel="Completion token NLL", title=f"Matched test · n={candidate['examples']}")
    fig.text(.02, .02, "One seed; bot imitation. Lower NLL does not establish technically correct reviews.", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1))
    for extension in ("png", "pdf", "svg"):
        path = args.output / f"imitation.{extension}"
        fig.savefig(path, dpi=300)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)
    (args.output / "plot-data.json").write_text(json.dumps({"baseline": baseline, "candidate": candidate, "history": history}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
