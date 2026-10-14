"""Read-only GitHub acquisition with resumable, private-on-disk responses."""

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .common import digest, write_json


class GitHub:
    def __init__(self, cache="data/raw/api", offline=False):
        self.cache = Path(cache)
        self.offline = offline
        self.token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if not self.token and not offline and shutil.which("gh"):
            token = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
            if token.returncode == 0:
                self.token = token.stdout.strip()

    def get(self, path):
        if not path.startswith(("repos/", "orgs/", "search/")):
            raise ValueError("Unsupported read-only GitHub endpoint")
        cache = self.cache / f"{digest(path)}.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        if self.offline:
            raise FileNotFoundError(f"Uncached GitHub response: {path}")
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "CodeR-preparation/0.1",
                   "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(f"https://api.github.com/{path}", headers=headers)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    value = json.load(response)
                write_json(cache, value)
                return value
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503) or attempt == 2:
                    raise
                time.sleep(min(10, 2 ** attempt))
        raise RuntimeError("Unreachable request state")

    def pages(self, path):
        separator = "&" if "?" in path else "?"
        result = []
        for page in range(1, 1001):
            values = self.get(f"{path}{separator}per_page=100&page={page}")
            if not isinstance(values, list):
                raise ValueError("Expected a paginated GitHub list")
            result.extend(values)
            if len(values) < 100:
                return result
        raise RuntimeError("Pagination limit exceeded; refusing a truncated corpus")
