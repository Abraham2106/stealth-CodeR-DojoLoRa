"""Public Dojo corpus, historical examples, weak labels, and leakage audits."""

import argparse
import datetime as dt
import json
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .common import digest, git, manifest, write_json
from .context import Graph, Retrieval, snapshot
from .github import GitHub

ORG = "DojoCodingLabs"
OUTCOME_FIELDS = {"merged", "merged_at", "state", "approved", "risk_label", "quality_label",
                  "review_state", "label", "outcome", "accepted", "resolved", "target"}


def collect(root="data", workers=4):
    root = Path(root)
    api = GitHub(root / "raw/api")
    repositories = [r for r in api.pages(f"orgs/{ORG}/repos?type=public") if not r["private"]]
    inventory = [{"repo": r["full_name"], "fork": r["fork"], "archived": r["archived"],
                  "language": r["language"], "license": (r.get("license") or {}).get("spdx_id"),
                  "retrieval": "GitHub REST API 2022-11-28; own organization PRs only",
                  "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "reuse_status": "unreviewed; public availability is not a reuse grant"}
                 for r in repositories]
    write_json(root / "raw/inventory.json", inventory)
    jobs, errors = [], []
    for repo in inventory:
        name = repo["repo"]
        pulls = api.pages(f"repos/{name}/pulls?state=all&sort=created&direction=asc")
        inline = api.pages(f"repos/{name}/pulls/comments?sort=created&direction=asc")
        discussion = api.pages(f"repos/{name}/issues/comments?sort=created&direction=asc")
        by_inline, by_discussion = {}, {}
        for c in inline:
            number = int(c["pull_request_url"].rsplit("/", 1)[-1])
            by_inline.setdefault(number, []).append(c)
        for c in discussion:
            number = int(c["issue_url"].rsplit("/", 1)[-1])
            by_discussion.setdefault(number, []).append(c)
        for p in pulls:
            jobs.append((repo, p, by_inline.get(p["number"], []), by_discussion.get(p["number"], [])))

    def acquire(job):
        repo, summary, inline, discussion = job
        name, number = repo["repo"], summary["number"]
        dest = root / "raw/prs" / name.split("/")[1] / f"{number}.json"
        if dest.exists():
            previous = json.loads(dest.read_text(encoding="utf-8"))
            if previous.get("collection_schema") == 2:
                return previous
        pr = api.get(f"repos/{name}/pulls/{number}")
        commits = api.pages(f"repos/{name}/pulls/{number}/commits")
        reviews = api.pages(f"repos/{name}/pulls/{number}/reviews")
        files = api.pages(f"repos/{name}/pulls/{number}/files")
        timeline = api.pages(f"repos/{name}/issues/{number}/timeline")
        body = pr.get("body") or ""
        # GitHub references in copied upstream release notes aren't local issues.
        linked_numbers = set(map(int, re.findall(
            r"(?i)\b(?:fix(?:es|ed)?|clos(?:e[sd]?|ing)|resolv(?:e[sd]?|ing)|related issue|issue)\s*[: ]\s*#(\d+)\b", body)))
        linked_numbers.update(map(int, re.findall(r"https://github\.com/" + re.escape(name) + r"/issues/(\d+)", body)))
        external_references = []
        for event in timeline:
            issue = (event.get("source") or {}).get("issue")
            if issue:
                if issue.get("repository_url") == f"https://api.github.com/repos/{name}":
                    linked_numbers.add(issue["number"])
                else:
                    external_references.append(issue.get("html_url"))
        linked_issues = []
        unresolved_references = []
        for linked in sorted(linked_numbers):
            try:
                issue = api.get(f"repos/{name}/issues/{linked}")
                if "pull_request" not in issue:
                    linked_issues.append(issue)
            except Exception as e:
                if getattr(e, "code", None) == 404:
                    unresolved_references.append({"number": linked, "reason": "not-a-visible-local-issue"})
                else:
                    errors.append({"repo": name, "pr": number, "linked_issue": linked, "error": type(e).__name__})
        historical_base = None
        if commits:
            first = api.get(f"repos/{name}/commits/{commits[0]['sha']}")
            if len(first["parents"]) == 1:
                historical_base = first["parents"][0]["sha"]
        value = {"collection_schema": 2, "identity": f"{name}#{number}", "repository": repo, "pr": pr,
                 "commits": commits, "reviews": reviews, "inline_comments": inline,
                 "discussion_comments": discussion, "linked_issues": linked_issues,
                 "files": files, "historical_base": historical_base,
                 "timeline": timeline, "external_issue_references": external_references,
                 "unresolved_local_issue_references": unresolved_references,
                 "base_provenance": "first currently available PR commit's single parent; verify ancestry for each revision",
                 "limitations": ["Original opening base/head may be unavailable after rebase or force push",
                                 "REST does not expose review-thread resolution; unknown is not false"]}
        write_json(dest, value)
        print(f"Collected {value['identity']}: {len(inline)} inline comments", flush=True)
        return value

    values = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(acquire, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                values.append(future.result())
            except Exception as e:
                job = futures[future]
                errors.append({"repo": job[0]["repo"], "pr": job[1]["number"], "error": type(e).__name__, "message": str(e)[:300]})
    inline = [c for value in values for c in value["inline_comments"]]
    stats = {"repositories": len(inventory), "prs": len(values), "expected_prs": len(jobs),
             "inline_comments": len(inline), "bot_inline_comments": sum(c["user"]["type"] == "Bot" for c in inline),
             "human_inline_comments": sum(c["user"]["type"] != "Bot" for c in inline),
             "errors": errors, "complete": len(values) == len(jobs) and not errors,
             "training_executed": False}
    write_json(root / "raw/collection-report.json", stats)
    write_json(root / "raw/artifact-manifest.json", manifest(root / "raw"))
    return stats


def weak_risk(record):
    reviews = record["reviews"]
    # These are workflow proxies, not measured defect probabilities.
    if any(r["state"] == "CHANGES_REQUESTED" for r in reviews):
        return "medium", "changes-requested proxy; not defect severity"
    if any(r["state"] == "APPROVED" for r in reviews):
        return "low", "explicit approval proxy; does not prove safety"
    return None, "No defensible outcome label; manual risk annotation required"


def weak_quality(comment):
    # REST resolution is unavailable. A maintainer's thumbs-up remains a noisy proxy.
    if (comment.get("reactions") or {}).get("+1", 0) > 0:
        return "useful", "positive reaction; voter identity/technical truth not established"
    return None, "No explicit quality evidence; ignored does not mean low quality"


def partition(examples):
    """PR grouping plus duplicate connected components; future duplicates embargo earlier groups."""
    groups = {}
    for e in examples:
        groups.setdefault(e["group"], []).append(e)
    parent = {g: g for g in groups}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    seen = {}
    for group, items in groups.items():
        for e in items:
            fingerprint = e.get("duplicate_key") or digest(e["input"])
            if fingerprint in seen:
                parent[find(group)] = find(seen[fingerprint])
            seen[fingerprint] = group
    ordered = sorted(groups, key=lambda g: (min(e.get("pr_created_at", e["timestamp"]) for e in groups[g]), g))
    n = len(ordered)
    first, second = math.floor(n * .70), math.floor(n * .85)
    initial = {g: 0 if i < first else 1 if i < second else 2 for i, g in enumerate(ordered)}
    component_split = {}
    for g in ordered:
        component_split[find(g)] = max(component_split.get(find(g), 0), initial[g])
    # Moving an earlier duplicate forward would break chronology: embargo it instead.
    splits = {"train": [], "validation": [], "test": [], "embargo": []}
    names = ("train", "validation", "test")
    for g in ordered:
        assigned = component_split[find(g)]
        name = names[assigned] if initial[g] == assigned else "embargo"
        splits[name].extend(groups[g])
    # Reviews on training PRs may occur after a validation/test PR opens.
    # Embargo those late observations instead of leaking future supervision.
    for name, next_names in (("train", ("validation", "test")), ("validation", ("test",))):
        boundaries = [e.get("pr_created_at", e["timestamp"]) for next_name in next_names for e in splits[next_name]]
        if boundaries:
            boundary = min(boundaries)
            late_groups = {e["group"] for e in splits[name] if e.get("context_cutoff", e["timestamp"]) >= boundary}
            splits["embargo"].extend(e for e in splits[name] if e["group"] in late_groups)
            splits[name] = [e for e in splits[name] if e["group"] not in late_groups]
    return splits


def audit(splits):
    owners, fingerprints, dates = {}, {}, {}
    for split in ("train", "validation", "test"):
        rows = splits[split]
        dates[split] = [e.get("pr_created_at", e["timestamp"]) for e in rows]
        for e in rows:
            group = e["group"]
            if group in owners and owners[group] != split:
                raise ValueError("PR group leakage")
            owners[group] = split
            fingerprint = e.get("duplicate_key") or digest(e["input"])
            if fingerprint in fingerprints and fingerprints[fingerprint] != split:
                raise ValueError("Duplicate leakage")
            fingerprints[fingerprint] = split
            forbidden = OUTCOME_FIELDS.intersection(e["input"])
            if forbidden:
                raise ValueError(f"Oracle metadata in inputs: {forbidden}")
            for item in e["input"].get("rag", []):
                if item["timestamp"] >= e.get("context_cutoff", e["timestamp"]):
                    raise ValueError("Future RAG document")
                if item.get("provenance", {}).get("pr") == e["pr"]:
                    raise ValueError("Same-PR review leaked through RAG")
    nonempty = [dates[s] for s in ("train", "validation", "test") if dates[s]]
    for before, after in zip(nonempty, nonempty[1:]):
        if max(before) > min(after):
            raise ValueError("Chronological split violation")
    return {"passed": True, "sizes": {s: len(v) for s, v in splits.items()},
            "hashes": {s: digest(v) for s, v in splits.items()},
            "limits": "Exact grouping/content checks do not certify semantic independence or label truth"}


def repository_clone(root, repo, revisions):
    destination = Path(root) / "snapshots" / repo.split("/")[1]
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / ".git").exists():
        git(destination, "init")
        git(destination, "remote", "add", "origin", f"https://github.com/{repo}.git")
    for revision in sorted(set(revisions)):
        if not revision or not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise ValueError("Expected an immutable GitHub SHA")
        try:
            git(destination, "cat-file", "-e", f"{revision}^{{commit}}")
        except subprocess.CalledProcessError:
            git(destination, "fetch", "--no-tags", "--depth=1", "origin", revision)
    return destination


def prepare(root="data", with_context=False):
    root = Path(root)
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "raw/prs").glob("*/*.json"))]
    if not records:
        raise ValueError("Collect the public corpus first")
    rag = Retrieval(root / "prepared/retrieval.sqlite")
    for record in records:
        repo, pr = record["repository"]["repo"], record["pr"]["number"]
        for comment in record["inline_comments"]:
            rag.add(f"{repo}:review:{comment['id']}", repo, comment.get("updated_at") or comment["created_at"], comment["body"],
                    {"pr": pr, "author": comment["user"]["login"], "type": comment["user"]["type"], "source": comment["html_url"]})
    examples, annotations, exclusions = [], [], []
    for record in records:
        repo, pr = record["repository"]["repo"], record["pr"]["number"]
        risk, risk_reason = weak_risk(record)
        comments = [c for c in record["inline_comments"] if not c.get("in_reply_to_id")]
        # B1 requires an independent PR snapshot and manual/outcome labels.
        annotations.append({"task": "risk", "group": record["identity"], "weak_label": risk,
                            "reason": risk_reason, "url": record["pr"]["html_url"],
                            "introduced_defect_label": None, "human_label": None})
        for comment in comments:
            revision = comment.get("original_commit_id")
            base = record.get("historical_base")
            line = comment.get("original_line")
            quality, reason = weak_quality(comment)
            annotation = {"task": "quality", "group": record["identity"], "comment_id": comment["id"],
                          "weak_label": quality, "reason": reason, "human_label": None,
                          "url": comment["html_url"], "risk_human_label": None}
            annotations.append(annotation)
            if comment.get("updated_at", comment["created_at"]) != comment["created_at"]:
                exclusions.append({"group": record["identity"], "comment_id": comment["id"], "reason": "edited-review-body-original-version-unavailable"})
                continue
            if not with_context:
                exclusions.append({"group": record["identity"], "comment_id": comment["id"], "reason": "historical-context-not-built"})
                continue
            if not base or not revision or not line:
                exclusions.append({"group": record["identity"], "comment_id": comment["id"], "reason": "unrecoverable-revision-or-line"})
                continue
            try:
                clone = repository_clone(root, repo, [base, revision])
                # Both shallow roots can be disconnected. Verify true ancestry with API compare.
                comparison = GitHub(root / "raw/api").get(f"repos/{repo}/compare/{base}...{revision}")
                if comparison.get("merge_base_commit", {}).get("sha") != base:
                    raise ValueError("Historical base is not an ancestor of this comment revision")
                graph = Graph(root / "prepared/graphs" / repo.split("/")[1] / f"{base}.sqlite")
                existing_revision = graph.db.execute("SELECT value FROM metadata WHERE key='revision'").fetchone()
                existing_parser = graph.db.execute("SELECT value FROM metadata WHERE key='parser_version'").fetchone()
                existing_policy = graph.db.execute("SELECT value FROM metadata WHERE key='snapshot_policy'").fetchone()
                reuse = existing_revision == (base,) and existing_parser == ("0.25.2",) and existing_policy == ("exclude-vendor-v1",)
                _, sources, standards, unsupported = snapshot(clone, base, sources_required=not reuse)
                if reuse:
                    coverage = {"files": graph.db.execute("SELECT count(*) FROM files").fetchone()[0], "changed": 0,
                                "supported_files": graph.db.execute("SELECT count(*) FROM files WHERE language IS NOT NULL").fetchone()[0],
                                "parse_errors": graph.db.execute("SELECT count(*) FROM files WHERE parse_error=1").fetchone()[0],
                                "unresolved_calls": graph.db.execute("SELECT count(*) FROM calls WHERE resolved_path IS NULL").fetchone()[0]}
                else:
                    coverage = graph.update(sources, base)
                context = graph.context([comment["path"]])
                graph.close()
                diff = git(clone, "diff", "--no-ext-diff", "--unified=3", base, revision, "--")
                if not valid_anchor(diff, comment["path"], line, comment.get("side") or "RIGHT"):
                    raise ValueError("Review anchor not present in historical diff")
                cutoff = comment["created_at"]
                base_time = git(clone, "show", "-s", "--format=%cI", base)
                if dt.datetime.fromisoformat(base_time) > dt.datetime.fromisoformat(cutoff.replace("Z", "+00:00")):
                    raise ValueError("Base commit postdates this review")
                standard_context = [{"path": p, "content": s[:4000], "source_commit": base} for p, s in standards.items()][:6]
                input_data = {"diff": diff, "graph": context,
                              "rag": rag.retrieve(diff + " " + comment["path"], repo, cutoff, exclude_pr=pr),
                              "standards": standard_context,
                              "path": comment["path"], "line": line, "side": comment.get("side") or "RIGHT"}
                common = {"group": record["identity"], "repo": repo, "pr": pr,
                          "timestamp": record["pr"]["created_at"], "context_cutoff": cutoff,
                          "base_sha": base, "head_sha": revision, "input": input_data,
                          "provenance": {"author": comment["user"]["login"], "author_type": comment["user"]["type"],
                                         "source": comment["html_url"], "license": record["repository"]["license"],
                                         "supervision": "weak-bot" if comment["user"]["type"] == "Bot" else "human-unverified",
                                         "graph_coverage": coverage, "omitted_files": len(unsupported),
                                         "historical_base_verified": "first-commit parent and compare ancestry; opening base not certified"}}
                # Split timestamp is PR creation; retrieval cutoff is this review's revision event.
                common["timestamp"] = cutoff
                common["pr_created_at"] = record["pr"]["created_at"]
                examples.append({**common, "task": "A", "target": comment["body"],
                                 "duplicate_key": digest({"diff": diff, "comment": comment["body"]})})
                quality_input = {**input_data, "comment": comment["body"]}
                examples.append({**common, "task": "B2", "input": quality_input, "target": quality,
                                 "label_reason": reason, "duplicate_key": digest({"diff": diff, "comment": comment["body"], "task": "B2"})})
                examples.append({**common, "task": "B1", "target": risk, "label_reason": risk_reason,
                                 "duplicate_key": digest({"diff": diff, "task": "B1"})})
                print(f"Prepared {record['identity']} comment {comment['id']}", flush=True)
            except Exception as e:
                exclusions.append({"group": record["identity"], "comment_id": comment["id"], "reason": str(e)[:300]})
    rag.close()
    unique = {}
    for e in examples:
        unique.setdefault((e["group"], e["task"], e["duplicate_key"]), e)
    examples = list(unique.values())
    splits = partition(examples)
    report = audit(splits)
    for name, rows in splits.items():
        write_json(root / "prepared" / f"{name}.json", rows)
    write_json(root / "prepared/annotation-queue.json", annotations)
    write_json(root / "prepared/exclusions.json", exclusions)
    report.update({"training_executed": False, "historical_context_built": with_context,
                   "tasks": {task: sum(e["task"] == task for e in examples) for task in ("A", "B1", "B2")},
                   "labeled": {task: sum(e["task"] == task and e["target"] is not None for e in examples) for task in ("A", "B1", "B2")},
                   "annotation_queue": len(annotations), "excluded": len(exclusions),
                   "label_warning": "Bot imitation and workflow labels are not independently verified technical quality"})
    report["label_support"] = {task: {str(label): sum(e["task"] == task and e["target"] == label for e in examples)
                                     for label in ("low", "medium", "high", None) if task == "B1"} if task == "B1" else
                               {str(label): sum(e["task"] == task and e["target"] == label for e in examples)
                                for label in ("useful", "low_quality", None)} for task in ("B1", "B2")}
    report["raw_record_hash"] = digest(records)
    write_json(root / "prepared/preparation-report.json", report)
    write_json(root / "prepared/artifact-manifest.json", manifest(root / "prepared"))
    return report


def valid_anchor(diff, path, line, side="RIGHT"):
    current, old_line, new_line = None, 0, 0
    for text in diff.splitlines():
        if text.startswith("diff --git "):
            match = re.match(r"diff --git a/(.*?) b/(.*)", text)
            current = match.group(2) if match else None
        elif text.startswith("@@"):
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", text)
            if match:
                old_line, new_line = map(int, match.groups())
        elif current == path and not text.startswith(("+++", "---", "\\")):
            if text.startswith((" ", "+", "-")):
                candidate = new_line if side == "RIGHT" else old_line
                exists = not text.startswith("-") if side == "RIGHT" else not text.startswith("+")
                if exists and candidate == line:
                    return True
                if not text.startswith("+"):
                    old_line += 1
                if not text.startswith("-"):
                    new_line += 1
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "prepare", "audit"))
    parser.add_argument("--root", default="data")
    parser.add_argument("--with-context", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.action == "collect":
        report = collect(args.root, args.workers)
    elif args.action == "prepare":
        report = prepare(args.root, args.with_context)
    else:
        report = audit({s: json.loads((Path(args.root) / "prepared" / f"{s}.json").read_text(encoding="utf-8"))
                        for s in ("train", "validation", "test", "embargo")})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
