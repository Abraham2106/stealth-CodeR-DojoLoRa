import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CLITests(unittest.TestCase):
    def test_offline_real_git_context_and_no_fake_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            repo.mkdir()
            def git(*args):
                return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()
            git("init")
            git("config", "user.name", "CPU fixture")
            git("config", "user.email", "fixture@example.invalid")
            (repo / "a.py").write_text("def f():\n    return 1\n")
            (repo / "CONTRIBUTING.md").write_text("Check boundaries.\n")
            git("add", ".")
            git("commit", "-m", "base")
            base = git("rev-parse", "HEAD")
            (repo / "a.py").write_text("def f():\n    return 2\n")
            git("add", ".")
            git("commit", "-m", "head")
            head = git("rev-parse", "HEAD")
            output = Path(directory) / "result.json"
            result = subprocess.run([sys.executable, "-m", "coder.cli", "--repo", str(repo), "--base", base,
                                     "--head", head, "--offline", "--prepare-only", "--cache", str(Path(directory) / "cache"),
                                     "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(output.read_text())
            self.assertEqual(value["status"], "prepared")
            self.assertIsNone(value["comments"])
            self.assertIsNone(value["risk"])
            self.assertEqual(value["context"]["base_sha"], base)
            self.assertEqual(value["context"]["graph"][0]["source"], "def f():\n    return 1")


if __name__ == "__main__":
    unittest.main()
