"""Bounded SFT of LoRA A only. B and technical-quality claims remain untrained."""

import json
import math
import random
import re
import time
from pathlib import Path

from .common import digest, manifest, verify_manifest, write_json
from .model import PROFILES, SharedReviewer, load_base, prompt


def focused_hunk(diff, path, line, side):
    """The training/inference unit is a changed hunk; target coordinates aren't inputs."""
    current, header, chunks = None, "", []
    for value in diff.splitlines(keepends=True):
        if value.startswith("diff --git "):
            match = re.match(r"diff --git a/(.*?) b/(.*)\n?", value)
            current = match.group(2).rstrip("\n") if match else None
            header = value
        elif value.startswith(("--- ", "+++ ", "index ", "new file ", "deleted file ")):
            header += value
        elif value.startswith("@@"):
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", value)
            if current == path and match:
                old_start, old_count, new_start, new_count = match.groups()
                start, count = (int(old_start), int(old_count or 1)) if side == "LEFT" else (int(new_start), int(new_count or 1))
                chunks.append({"match": start <= line < start + count, "text": header + value})
            else:
                chunks.append({"match": False, "text": ""})
        elif chunks:
            chunks[-1]["text"] += value
    selected = [c["text"] for c in chunks if c["match"]]
    if len(selected) != 1:
        raise ValueError("Target must belong to exactly one recoverable changed hunk")
    return selected[0]


def prepare_sample(example, tokenizer, max_sequence=4096, max_completion=1024):
    raw = example["input"]
    path, line, side = raw["path"], raw["line"], raw.get("side", "RIGHT")
    context = {"diff": focused_hunk(raw["diff"], path, line, side), "changed_paths": [path],
               "graph": raw.get("graph", []), "rag": raw.get("rag", []), "standards": raw.get("standards", [])}
    target = json.dumps([{"path": path, "line": line, "side": side, "body": example["target"]}], ensure_ascii=False)
    completion = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(completion) > max_completion:
        raise ValueError("Human/bot completion exceeds configured limit; not truncated")
    changes = []
    for level in range(4):
        prefix = tokenizer.apply_chat_template(prompt(context), tokenize=True, add_generation_prompt=True,
                                               enable_thinking=False, return_dict=False)
        if len(prefix) + len(completion) <= max_sequence:
            break
        if level == 0:
            context["graph"] = [{**g, **({"source": g["source"][:1500]} if "source" in g else {})} for g in context["graph"][:8]]
            context["rag"] = [{**r, "content": r["content"][:1000]} for r in context["rag"][:3]]
            context["standards"] = [{**r, "content": r["content"][:1000]} for r in context["standards"][:3]]
            changes.append("bounded-context-snippets")
        elif level == 1:
            context["graph"] = context["graph"][:3]
            context["rag"] = context["rag"][:1]
            context["standards"] = context["standards"][:1]
            changes.append("reduced-context-count")
        else:
            raise ValueError("Prompt plus intact completion still exceeds sequence limit")
    return {"ids": prefix + completion, "labels": [-100] * len(prefix) + completion,
            "prefix": prefix, "context": context, "target": target, "group": example["group"],
            "changes": changes, "base_sha": example["base_sha"], "head_sha": example["head_sha"]}


def batch(sample, device):
    import torch
    return {"input_ids": torch.tensor([sample["ids"]], device=device),
            "attention_mask": torch.ones(1, len(sample["ids"]), dtype=torch.long, device=device),
            "labels": torch.tensor([sample["labels"]], device=device)}


def timed_remaining(started_at, hours=6, reserve_seconds=300):
    remaining = hours * 3600 - (time.time() - started_at)
    if remaining <= reserve_seconds:
        raise TimeoutError("GPU execution budget exhausted; preserve artifacts before continuing")
    return remaining


def token_nll(reviewer, samples, started_at, disable_adapter=False):
    import torch
    from contextlib import nullcontext
    reviewer.model.eval()
    total, count = 0.0, 0
    reviewer.switch("A")
    for sample in samples:
        timed_remaining(started_at)
        tokens = batch(sample, reviewer.model.get_base_model().device)
        with reviewer.model.disable_adapter() if disable_adapter else nullcontext():
            with torch.inference_mode():
                loss = reviewer.model(**tokens, use_cache=False).loss
        denominator = int((tokens["labels"][:, 1:] != -100).sum())
        total += float(loss) * denominator
        count += denominator
    if not count:
        raise ValueError("No supervised evaluation tokens")
    return {"mean_token_nll": total / count, "completion_tokens": count, "examples": len(samples)}


def train_a(data_directory, output_directory, gpu_started_at, profile="qwen35", max_steps=2000, epochs=3,
            learning_rate=2e-4, accumulation=4, seed=42, max_sequence=4096, evaluate_quantized=True):
    import torch
    from transformers import AutoTokenizer
    if max_steps > 2000 or epochs > 3 or max_steps < 2 or epochs < 1:
        raise ValueError("This authorized run allows at most 2000 steps / 3 epochs")
    if not torch.cuda.is_available():
        raise RuntimeError("Requested training requires a real CUDA GPU")
    timed_remaining(gpu_started_at)
    root, output = Path(data_directory), Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    audit_report = json.loads((root / "preparation-report.json").read_text(encoding="utf-8"))
    if not audit_report.get("passed"):
        raise ValueError("Data audit has not passed")
    raw_splits = {s: json.loads((root / f"{s}.json").read_text(encoding="utf-8")) for s in ("train", "validation", "test")}
    for split, rows in raw_splits.items():
        if digest(rows) != audit_report["hashes"][split]:
            raise ValueError(f"Frozen {split} hash mismatch")
    config = PROFILES[profile]
    tokenizer = AutoTokenizer.from_pretrained(config["id"], revision=config["revision"])
    splits, exclusions = {}, []
    for name, rows in raw_splits.items():
        splits[name] = []
        for row in rows:
            if row["task"] != "A":
                continue
            try:
                splits[name].append(prepare_sample(row, tokenizer, max_sequence))
            except ValueError as error:
                exclusions.append({"split": name, "group": row["group"], "reason": str(error)})
        if not splits[name]:
            raise ValueError(f"No usable A examples in {name}; refusing training/test reuse")
    provenance = {"profile": config, "epochs_limit": epochs, "steps_limit": max_steps,
                  "gpu_hours_limit": 6, "external_spend_limit": 0, "gpu_started_at": gpu_started_at,
                  "seed": seed, "learning_rate": learning_rate, "gradient_accumulation": accumulation,
                  "max_sequence": max_sequence, "split_hashes": audit_report["hashes"],
                  "example_counts": {s: len(v) for s, v in splits.items()},
                  "protocol": "positive-only bot-review imitation over changed hunks; no technical-quality oracle",
                  "B_trained": False, "deployment_4gb_verified": False}
    from .evaluation import preflight
    provenance["runtime"] = preflight()
    write_json(output / "provenance.json", provenance)
    write_json(output / "excluded-examples.json", exclusions)
    write_json(output / "training-input-manifest.json", {s: [{"group": e["group"], "length": len(e["ids"]), "changes": e["changes"]} for e in examples]
                                                          for s, examples in splits.items()})
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    timed_remaining(gpu_started_at)
    base = load_base(profile, config["revision"], quantized=False, local_only=False)
    reviewer = SharedReviewer(base, tokenizer, profile, config["revision"])
    reviewer.switch("A")
    parameters = [p for p in reviewer.model.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("Adapter A has no trainable parameters")
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "trainable_parameters": sum(p.numel() for p in parameters),
                      "counts": provenance["example_counts"]}), flush=True)
    baseline_validation = token_nll(reviewer, splits["validation"], gpu_started_at, disable_adapter=True)
    baseline_test = token_nll(reviewer, splits["test"], gpu_started_at, disable_adapter=True)
    # Authorized smoke: real forward/backward/save/reload, within the same run budget.
    reviewer.model.train()
    smoke_tokens = batch(splits["train"][0], base.device)
    smoke_before = {n: p.detach().cpu().clone() for n, p in reviewer.model.named_parameters() if p.requires_grad}
    smoke_optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    smoke_loss = reviewer.model(**smoke_tokens, use_cache=False).loss
    if not torch.isfinite(smoke_loss):
        raise RuntimeError("Non-finite smoke loss")
    smoke_loss.backward()
    smoke_optimizer.step()
    smoke_optimizer.zero_grad(set_to_none=True)
    changed = any(not torch.equal(smoke_before[n], p.detach().cpu()) for n, p in reviewer.model.named_parameters() if n in smoke_before)
    if not changed:
        raise RuntimeError("Smoke step failed to update adapter A")
    reviewer.save(output / "smoke", trained=False)
    reviewer.restore(output / "smoke", require_trained=False)
    print("Smoke forward/backward/save/reload passed", flush=True)
    # Reset the adapter to its initial state; the smoke step is not hidden training data repetition.
    with torch.no_grad():
        for name, parameter in reviewer.model.named_parameters():
            if name in smoke_before:
                parameter.copy_(smoke_before[name].to(parameter.device))
    del smoke_optimizer, smoke_before, smoke_tokens, smoke_loss
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=.01)
    best, history, global_step = math.inf, [], 0
    for epoch in range(epochs):
        timed_remaining(gpu_started_at)
        reviewer.switch("A")
        reviewer.model.train()
        indices = list(range(len(splits["train"])))
        random.Random(seed + epoch).shuffle(indices)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for position, index in enumerate(indices):
            timed_remaining(gpu_started_at)
            tokens = batch(splits["train"][index], base.device)
            loss = reviewer.model(**tokens, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; stop instead of corrupting checkpoint")
            loss.backward()
            losses.append(float(loss.detach()))
            end_batch = (position + 1) % accumulation == 0 or position + 1 == len(indices)
            if end_batch:
                actual_batch = accumulation if (position + 1) % accumulation == 0 else (position + 1) % accumulation
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.div_(actual_batch)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 5 == 0:
                    print(json.dumps({"epoch": epoch + 1, "step": global_step, "training_loss": sum(losses[-actual_batch:]) / actual_batch}), flush=True)
                    reviewer.save(output / "latest", trained=False)
                    torch.save({"optimizer": optimizer.state_dict(), "epoch": epoch, "position": position,
                                "global_step": global_step, "torch_rng": torch.get_rng_state(),
                                "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate()}, output / "latest/resume.pt")
                if global_step >= max_steps - 1:
                    break
        validation = token_nll(reviewer, splits["validation"], gpu_started_at)
        history.append({"epoch": epoch + 1, "steps": global_step, "train_loss": sum(losses) / len(losses), "validation": validation})
        if validation["mean_token_nll"] < best:
            best = validation["mean_token_nll"]
            reviewer.save(output / "checkpoint", trained=False)
        write_json(output / "history.json", history)
        print(json.dumps(history[-1]), flush=True)
        if global_step >= max_steps - 1:
            break
    reviewer.restore(output / "checkpoint", require_trained=False)
    candidate_test = token_nll(reviewer, splits["test"], gpu_started_at)
    report = {"baseline_validation": baseline_validation, "baseline_test": baseline_test,
              "candidate_test_bf16": candidate_test, "selected_by": "validation token NLL only",
              "steps": global_step, "smoke_optimizer_steps": 1, "total_optimizer_steps": global_step + 1,
              "training_executed": True, "trained_adapters": ["A"],
              "B_trained": False, "deployment_4gb_verified": False,
              "quality_claim": "NLL measures imitation of bots, not correctness or useful code review"}
    config_path = output / "checkpoint/config.json"
    saved = json.loads(config_path.read_text())
    saved.update({"trained_adapters": ["A"], "A_trained": True, "B_trained": False,
                  "trained": False, "stage": "A-trained-B-untrained"})
    write_json(config_path, saved)
    write_json(output / "checkpoint/artifact-manifest.json", manifest(output / "checkpoint"))
    # Real reload into a fresh instance, with only one backbone at a time.
    del optimizer, parameters, reviewer, base
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    timed_remaining(gpu_started_at)
    fresh_base = load_base(profile, config["revision"], quantized=evaluate_quantized, local_only=True)
    fresh = SharedReviewer(fresh_base, tokenizer, profile, config["revision"])
    fresh.restore(output / "checkpoint", require_trained=False)
    report["fresh_reload_verified"] = True
    report["quantized_evaluation"] = evaluate_quantized
    report["baseline_test_reloaded"] = token_nll(fresh, splits["test"], gpu_started_at, disable_adapter=True)
    report["candidate_test_reloaded"] = token_nll(fresh, splits["test"], gpu_started_at)
    predictions = []
    measurements = []
    for sample in splits["test"]:
        timed_remaining(gpu_started_at)
        inputs = {"input_ids": torch.tensor([sample["prefix"]], device=fresh_base.device),
                  "attention_mask": torch.ones(1, len(sample["prefix"]), dtype=torch.long, device=fresh_base.device)}
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        generation_start = time.perf_counter()
        with torch.inference_mode():
            generated = fresh.generate(inputs, max_new_tokens=512)
        torch.cuda.synchronize()
        measurements.append({"seconds": time.perf_counter() - generation_start,
                             "input_tokens": len(sample["prefix"]),
                             "output_tokens": generated.shape[1] - len(sample["prefix"]),
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                             "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        text = tokenizer.decode(generated[0, len(sample["prefix"]):], skip_special_tokens=True)
        try:
            result = json.loads(text)
            valid = isinstance(result, list)
        except json.JSONDecodeError:
            result, valid = None, False
        predictions.append({"group": sample["group"], "generated": text, "json_array_valid": valid,
                            "reference": sample["target"], "technical_quality_human_label": None})
    report["generation"] = {"denominator": len(predictions), "json_array_valid": sum(p["json_array_valid"] for p in predictions)}
    report["inference_measurements"] = measurements
    report["elapsed_gpu_session_seconds"] = time.time() - gpu_started_at
    write_json(output / "predictions.json", predictions)
    write_json(output / "report.json", report)
    write_json(output / "artifact-manifest.json", manifest(output))
    verify_manifest(output, json.loads((output / "artifact-manifest.json").read_text()))
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-started-at", type=float, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="qwen35")
    args = parser.parse_args()
    try:
        report = train_a(args.data, args.output, args.gpu_started_at, args.profile)
        import shutil
        archive = shutil.make_archive(str(Path(args.output).resolve()), "zip", args.output)
        print(json.dumps({"archive": archive, "report": report}), flush=True)
    except Exception as error:
        failure = {"status": "failed", "type": type(error).__name__, "reason": str(error),
                   "checkpoints_may_exist": str(Path(args.output) / "checkpoint")}
        write_json(Path(args.output) / "failure.json", failure)
        print(json.dumps(failure), flush=True)
        raise


if __name__ == "__main__":
    main()
