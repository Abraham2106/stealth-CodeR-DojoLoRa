"""Local PR preparation and guarded inference; never posts GitHub reviews."""

import argparse
import datetime as dt
import json
from pathlib import Path

from .common import commit, git, write_json
from .context import Graph, Retrieval, snapshot
from .data import valid_anchor
from .github import GitHub


def prepare_context(repo, base, head, cache, identity=None, cutoff=None, pr=None):
    repo, cache = Path(repo), Path(cache)
    base, head = commit(repo, base), commit(repo, head)
    _, sources, standards, excluded = snapshot(repo, base)
    graph = Graph(cache / "graph.sqlite")
    coverage = graph.update(sources, base)
    paths = git(repo, "diff", "--name-only", "-z", base, head, binary=True).decode("utf-8", errors="replace").strip("\0").split("\0")
    diff = git(repo, "diff", "--no-ext-diff", "--unified=3", base, head, "--")
    graph_context = graph.context(paths)
    graph.close()
    rag_context = []
    if (cache / "retrieval.sqlite").exists() and identity:
        retrieval = Retrieval(cache / "retrieval.sqlite")
        rag_context = retrieval.retrieve(diff, identity, cutoff, exclude_pr=pr)
        retrieval.close()
    return {"base_sha": base, "head_sha": head, "diff": diff, "changed_paths": paths,
            "graph": graph_context, "rag": rag_context,
            "standards": [{"path": path, "content": body[:4000], "source_commit": base} for path, body in standards.items()][:6],
            "coverage": {**coverage, "excluded": excluded}, "context_cutoff": cutoff}


def review(context, checkpoint):
    import torch
    from transformers import AutoTokenizer
    from .model import QUALITY, RISK, SharedReviewer, load_base, tokenize
    config_path = Path(checkpoint) / "config.json"
    if not config_path.exists():
        raise ValueError("A complete trained checkpoint is required; use --prepare-only")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config.get("trained") or not config.get("calibration"):
        raise ValueError("Trained weights and validation calibration are required")
    tokenizer = AutoTokenizer.from_pretrained(Path(checkpoint) / "tokenizer", local_files_only=True)
    base = load_base(config["profile"], config["base_revision"], local_only=True)
    reviewer = SharedReviewer(base, tokenizer, config["profile"], config["base_revision"])
    reviewer.restore(checkpoint)
    reviewer.model.eval()
    with torch.inference_mode():
        tokens = tokenize(tokenizer, context, base.device)
        generated = reviewer.generate(tokens)
        text = tokenizer.decode(generated[0, tokens["input_ids"].shape[1]:], skip_special_tokens=True)
        candidates = json.loads(text)
        if not isinstance(candidates, list):
            raise ValueError("Review output failed JSON-array schema")
        risk_input = tokenize(tokenizer, context, base.device, "risk")
        logits = reviewer.classify(risk_input)["risk_logits"]
        risk = RISK[int(logits.argmax(dim=-1)[0])]
        retained = []
        for c in candidates:
            if not isinstance(c, dict) or not isinstance(c.get("body"), str) or not isinstance(c.get("line"), int):
                continue
            if not valid_anchor(context["diff"], c.get("path"), c["line"], c.get("side", "RIGHT")):
                continue
            classified = reviewer.classify(tokenize(tokenizer, {**context, "comment": c["body"]}, base.device, "quality"))
            temperature = config["calibration"]["quality_temperature"]
            probability = torch.softmax(classified["quality_logits"] / temperature, dim=-1)[0, QUALITY.index("useful")].item()
            if probability >= config["calibration"]["quality_threshold"]:
                retained.append({**c, "quality_score": probability})
        return {"status": "reviewed", "comments": retained, "risk": risk,
                "human_review_required": risk == "high", "raw_candidate_count": len(candidates)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", type=int)
    parser.add_argument("--github-repo", help="owner/name; inferred from origin when omitted")
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--cache", default=".cache/review")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.pr and args.offline:
            raise ValueError("Offline mode requires explicit --base/--head, without --pr")
        identity = args.github_repo
        cutoff = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if args.pr:
            if args.base or args.head:
                raise ValueError("Choose --pr or --base/--head")
            if not identity:
                origin = git(args.repo, "remote", "get-url", "origin")
                identity = origin.removesuffix(".git").split("github.com/")[-1].split("github.com:")[-1]
            value = GitHub(Path(args.cache) / "api").get(f"repos/{identity}/pulls/{args.pr}")
            args.base, args.head = value["base"]["sha"], value["head"]["sha"]
            git(args.repo, "fetch", "--no-tags", "origin", args.base, args.head)
        if not args.base or not args.head:
            raise ValueError("Supply --pr or both --base and --head")
        context = prepare_context(args.repo, args.base, args.head, args.cache, identity, cutoff, args.pr)
        result = {"status": "prepared", "context": context, "training_executed": False,
                  "comments": None, "risk": None, "human_review_required": None,
                  "note": "No predictions produced in preparation mode"}
        if not args.prepare_only:
            if not args.checkpoint:
                raise ValueError("Supply trained --checkpoint or select --prepare-only")
            result.update(review(context, args.checkpoint))
        write_json(args.output, result)
        print(json.dumps({"output": args.output, "status": result["status"], "coverage": context["coverage"]["supported_files"]}))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"review_pr: {error}\n")


if __name__ == "__main__":
    main()
