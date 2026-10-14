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


def git_blobs(repo, identities):
    """Read immutable blobs in bounded batches rather than launching Git per file."""
    identities = list(dict.fromkeys(identities))
    output = {}
    for start in range(0, len(identities), 64):
        requested = identities[start:start + 64]
        result = subprocess.run(["git", "-C", str(repo), "cat-file", "--batch"],
                                input=("\n".join(requested) + "\n").encode(),
                                capture_output=True, check=True, timeout=60)
        offset = 0
        for identity in requested:
            end = result.stdout.index(b"\n", offset)
            header = result.stdout[offset:end].decode().split()
            if len(header) != 3 or header[0] != identity or header[1] != "blob":
                raise ValueError("Unexpected Git batch blob response")
            size = int(header[2])
            offset = end + 1
            output[identity] = result.stdout[offset:offset + size]
            if len(output[identity]) != size:
                raise ValueError("Truncated Git batch blob")
            offset += size + 1
    return output


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
