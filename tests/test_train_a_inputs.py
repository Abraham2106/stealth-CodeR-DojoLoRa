import unittest
from coder.train_a import focused_hunk, prepare_sample


class TinyTokenizer:
    eos_token_id = 999
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))
    def apply_chat_template(self, messages, **kwargs):
        return list(range(len(str(messages))))


class TrainingInputTests(unittest.TestCase):
    def test_hunk_supervision_masks_prompt_and_keeps_coordinates_only_in_target(self):
        diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n@@ -40 +40 @@\n-before\n+after\n"
        example = {"group": "repo#1", "base_sha": "base", "head_sha": "head", "target": "Check the new value.",
                   "input": {"diff": diff, "path": "a.py", "line": 40, "side": "RIGHT", "graph": [], "rag": []}}
        sample = prepare_sample(example, TinyTokenizer())
        self.assertNotIn("+new", sample["context"]["diff"])
        self.assertIn("+after", sample["context"]["diff"])
        self.assertNotIn("line", sample["context"])
        self.assertEqual(sample["labels"][:len(sample["prefix"])], [-100] * len(sample["prefix"]))
        self.assertEqual(sample["labels"][-1], 999)

    def test_missing_hunk_is_rejected(self):
        with self.assertRaises(ValueError):
            focused_hunk("", "a.py", 1, "RIGHT")

    def test_tokenizer_mapping_is_normalized_to_token_ids(self):
        from collections import UserDict
        class MappingTokenizer(TinyTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return UserDict(input_ids=[7, 8, 9])
        example = {"group": "repo#1", "base_sha": "base", "head_sha": "head", "target": "Review",
                   "input": {"diff": "diff --git a/a.py b/a.py\n@@ -1 +1 @@\n-a\n+b\n",
                             "path": "a.py", "line": 1, "side": "RIGHT"}}
        sample = prepare_sample(example, MappingTokenizer())
        self.assertEqual(sample["prefix"], [7, 8, 9])
        self.assertEqual(sample["labels"][:3], [-100, -100, -100])


if __name__ == "__main__":
    unittest.main()
