"""Evaluation primitives; no evaluator silently fits a model or uses a paid judge."""

import copy
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

from .common import write_json


def preflight():
    packages = {}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "marimo", "tree-sitter"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    gpu = None
    if shutil.which("nvidia-smi"):
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=20)
        gpu = result.stdout.strip() if result.returncode == 0 else "nvidia-smi failed"
    cuda = None
    if packages.get("torch"):
        import torch
        cuda = {"available": torch.cuda.is_available(), "runtime": torch.version.cuda}
    disk = shutil.disk_usage(Path.cwd())
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except FileNotFoundError:
        revision = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "gpu_observed": gpu, "cuda": cuda, "disk_free_bytes": disk.free,
            "code_revision": revision, "training_executed": False, "deployment_4gb_verified": False}


def risk_rule(added, deleted, files):
    lines = added + deleted
    return "high" if lines > 500 or files > 10 else "medium" if lines > 100 or files > 3 else "low"


def classification_metrics(gold, predicted, labels):
    from sklearn.metrics import classification_report, confusion_matrix
    if not gold or len(gold) != len(predicted):
        raise ValueError("Nonempty matched gold/prediction lists are required")
    if any(x not in labels for x in [*gold, *predicted]):
        raise ValueError("Unknown classification label")
    return {"denominator": len(gold), "labels": labels,
            "report": classification_report(gold, predicted, labels=labels, output_dict=True, zero_division=0),
            "confusion_gold_rows_pred_columns": confusion_matrix(gold, predicted, labels=labels).tolist()}


def fit_feature_baseline(features, labels):
    """Explicit future fitting entrypoint. Never called during preparation or notebook load."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(set(labels)) < 2 or len(features) != len(labels):
        raise ValueError("Matched training-only features and at least two classes required")
    estimator = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", random_state=42))
    return estimator.fit(features, labels)


def base_generation(reviewer, tokens, max_new_tokens=256):
    """Matched-context A baseline, using the existing backbone with adapters disabled."""
    import torch
    with reviewer.lock, reviewer.model.disable_adapter(), torch.inference_mode():
        reviewer.cache = None
        return reviewer.model.get_base_model().generate(**tokens, do_sample=False,
                                                       max_new_tokens=max_new_tokens, use_cache=True)


def select_quality_threshold(validation_probabilities, gold, target_precision=.9):
    """Select only from validation; return no threshold when the target cannot be met."""
    if len(gold) != len(validation_probabilities) or not gold:
        raise ValueError("Matched validation-only quality observations required")
    if any(label not in ("useful", "low_quality") for label in gold):
        raise ValueError("Unknown labels must be annotated before calibration")
    candidates = []
    for threshold in sorted(set(validation_probabilities)):
        accepted = [label for probability, label in zip(validation_probabilities, gold) if probability >= threshold]
        precision = sum(label == "useful" for label in accepted) / len(accepted)
        if precision >= target_precision:
            candidates.append((len(accepted), threshold, precision))
    if not candidates:
        return {"threshold": None, "reason": "Validation precision target not achieved"}
    accepted, threshold, precision = max(candidates)
    return {"threshold": threshold, "validation_precision": precision, "accepted": accepted,
            "denominator": len(gold), "limits": "Weak labels/small support cannot guarantee production precision"}


def ablations(context):
    result = {}
    for graph in (False, True):
        for rag in (False, True):
            value = copy.deepcopy(context)
            if not graph:
                value["graph"] = []
            if not rag:
                value["rag"] = []
                value["standards"] = []
            result[f"graph_{int(graph)}_rag_{int(rag)}"] = value
    return result


def human_review_sheet(examples, predictions):
    if len(examples) != len(predictions):
        raise ValueError("Matched examples/predictions required")
    return [{"example_id": e["group"], "provenance": e["provenance"], "candidate": p,
             "correctness": None, "actionability": None, "supported_by_context": None,
             "duplicate_or_nit": None, "reviewer": None} for e, p in zip(examples, predictions)]


def measure_inference(operation, repetitions=5):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU inference measurement requires CUDA")
    if repetitions < 1:
        raise ValueError("Positive repetitions required")
    operation()  # explicit inference warm-up, never an optimizer step
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    elapsed = []
    for _ in range(repetitions):
        start = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
    return {"repetitions": repetitions, "latency_seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "limits": "Allocator memory excludes driver/display memory; final 4GB test must run on the RTX 2050"}


def export_classification_figure(metrics, directory, run_id):
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(6, 5))
    matrix = metrics["confusion_gold_rows_pred_columns"]
    axes.imshow(matrix, cmap="Blues")
    axes.set(xticks=range(len(metrics["labels"])), yticks=range(len(metrics["labels"])),
             xticklabels=metrics["labels"], yticklabels=metrics["labels"],
             xlabel="Predicted", ylabel="Gold", title=f"{run_id} · n={metrics['denominator']}")
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            axes.text(j, i, str(count), ha="center", va="center")
    figure.tight_layout()
    for extension in ("png", "pdf", "svg"):
        figure.savefig(directory / f"confusion.{extension}", dpi=300)
    plt.close(figure)
    write_json(directory / "plot-data.json", {"run_id": run_id, **metrics})
