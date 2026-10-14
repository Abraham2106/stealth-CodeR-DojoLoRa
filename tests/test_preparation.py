import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from coder.common import digest, manifest, verify_manifest, write_json
from coder.context import Graph, Retrieval, parse_source
from coder.data import audit, partition, valid_anchor, weak_quality
from coder.evaluation import ablations, risk_rule


class PreparationTests(unittest.TestCase):
    def test_real_modules_repeated_node_traversal_does_not_crash(self):
        import argparse
        import inspect
        import typing
        for module in (argparse, inspect, typing, argparse):
            source = Path(module.__file__).read_bytes()
            symbols, calls, imports, refs, error = parse_source(Path(module.__file__).name, source)
            self.assertGreater(len(symbols), 10)
            self.assertGreater(len(calls), 10)

    def test_graph_incremental_matches_rebuild_and_removes_stale_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            incremental = Graph(Path(directory) / "incremental.sqlite")
            source = {"a.py": b"def callee():\n    return 1\ndef caller():\n    return callee()\n",
                      "b.ts": b"export function hello() { return 1; }\nhello();\n"}
            self.assertEqual(incremental.update(source)["changed"], 2)
            self.assertEqual(incremental.update(source)["changed"], 0)
            source["a.py"] = b"def replacement():\n    return 2\n"
            del source["b.ts"]
            incremental.update(source)
            clean = Graph(Path(directory) / "clean.sqlite")
            clean.update(source)
            for table in ("files", "symbols", "calls", "imports", "refs"):
                self.assertEqual(incremental.db.execute(f"SELECT * FROM {table} ORDER BY path").fetchall(),
                                 clean.db.execute(f"SELECT * FROM {table} ORDER BY path").fetchall())
            incremental.close()
            clean.close()

    def test_rag_excludes_future_and_same_pr_and_handles_fts_syntax(self):
        with tempfile.TemporaryDirectory() as directory:
            rag = Retrieval(Path(directory) / "rag.sqlite")
            rag.add("past", "org/repo", "2026-01-01T00:00:00Z", "auth token", {"pr": 1})
            rag.add("future", "org/repo", "2026-03-01T00:00:00Z", "auth token", {"pr": 2})
            rag.add("same", "org/repo", "2026-01-02T00:00:00Z", "auth token", {"pr": 3})
            result = rag.retrieve('auth:* OR "token', "org/repo", "2026-02-01T00:00:00Z", exclude_pr=3)
            self.assertEqual([e["id"] for e in result], ["past"])
            rag.close()

    def test_import_dependency_and_alias_call_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = Graph(Path(directory) / "graph.sqlite")
            graph.update({"a.py": b"from b import helper as h\ndef run():\n    return h()\n",
                          "b.py": b"def helper():\n    return 1\n",
                          "a.ts": b"import { helper as h } from './b';\nfunction run() { return h(); }\n",
                          "b.ts": b"export function helper() { return 1; }\n"})
            matches = graph.db.execute("SELECT resolved_path,resolved_symbol FROM calls ORDER BY path").fetchall()
            self.assertEqual(matches, [("b.py", "helper"), ("b.ts", "helper")])
            self.assertIn("imports", graph.context(["a.py"])[0])
            graph.close()

    def test_grouped_chronological_splits_and_duplicate_embargo(self):
        examples = [{"group": str(i), "timestamp": f"2026-01-{i+1:02d}T00:00:00Z", "input": {"diff": str(i)},
                     "pr": i, "target": "comment"} for i in range(20)]
        examples[0]["input"] = examples[-1]["input"]
        result = partition(examples)
        self.assertTrue(audit(result)["passed"])
        self.assertEqual(len(result["embargo"]), 1)
        result["validation"].append(result["test"][0])
        with self.assertRaises(ValueError):
            audit(result)

    def test_future_rag_and_oracle_are_rejected(self):
        example = {"group": "one", "timestamp": "2026-01-01T00:00:00Z", "pr": 1,
                   "input": {"rag": [{"timestamp": "2026-01-02T00:00:00Z"}]}}
        with self.assertRaises(ValueError):
            audit({"train": [example], "validation": [], "test": [], "embargo": []})
        example["input"] = {"merged": True}
        with self.assertRaises(ValueError):
            audit({"train": [example], "validation": [], "test": [], "embargo": []})

    def test_diff_anchor_and_unknown_quality(self):
        diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n-old\n+new\n unchanged\n"
        self.assertTrue(valid_anchor(diff, "a.py", 1))
        self.assertTrue(valid_anchor(diff, "a.py", 1, "LEFT"))
        self.assertFalse(valid_anchor(diff, "a.py", 10))
        self.assertIsNone(weak_quality({"body": "ignored"})[0])

    def test_manifest_integrity_and_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            write_json(Path(directory) / "head.json", {"trained": False})
            hashes = manifest(directory)
            verify_manifest(directory, hashes)
            (Path(directory) / "head.json").write_text("tampered")
            with self.assertRaises(ValueError):
                verify_manifest(directory, hashes)
            with self.assertRaises(ValueError):
                verify_manifest(directory, {"../outside": "fake"})

    def test_ablations_do_not_mutate_original(self):
        original = {"graph": ["g"], "rag": ["r"], "standards": ["s"]}
        self.assertEqual(ablations(original)["graph_0_rag_0"]["standards"], [])
        self.assertEqual(original["graph"], ["g"])
        self.assertEqual(risk_rule(40, 10, 1), "low")
        self.assertEqual(risk_rule(400, 200, 1), "high")


if __name__ == "__main__":
    unittest.main()
