"""Small shared artifact and Git helpers."""

import hashlib
import json
import subprocess
import uuid
from pathlib import Path


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def git(repo, *args, binary=False):
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=True,
    )
    return result.stdout if binary else result.stdout.decode("utf-8", errors="replace").strip()


def commit(repo, ref):
    # Resolve to an immutable commit; never interpolate refs in a shell command.
    if not ref or ref.startswith("-"):
        raise ValueError("Invalid Git revision")
    return git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")


def manifest(directory):
    directory = Path(directory)
    return {
        str(p.relative_to(directory)).replace("\\", "/"): digest(p.read_bytes())
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "artifact-manifest.json"
    }


def verify_manifest(directory, hashes):
    directory = Path(directory).resolve()
    for name, expected in hashes.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f"Missing or unsafe artifact: {name}")
        if digest(path.read_bytes()) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
