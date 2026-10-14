"""Tiny CPU-only inference tests: no downloads, optimizer, backward, or training."""

import tempfile
import unittest
import subprocess
import sys
import json
from pathlib import Path


class SharedModelTests(unittest.TestCase):
    def test_one_backbone_switch_and_reload_in_fresh_instance(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from coder.model import SharedReviewer
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                             head_dim=16, max_position_embeddings=64, tie_word_embeddings=True)
        torch.manual_seed(123)
        first = SharedReviewer(Qwen3ForCausalLM(config), adapter_init=False)
        first.model.eval()
        inputs = {"input_ids": torch.tensor([[0, 0, 3, 4], [3, 4, 0, 0]]),
                  "attention_mask": torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])}
        base = first.model.get_base_model()
        pointer = base.model.embed_tokens.weight.data_ptr()
        self.assertEqual(pointer, base.lm_head.weight.data_ptr())
        first.cache = object()
        with torch.inference_mode():
            before = first.classify(inputs)
        self.assertIsNone(first.cache)
        self.assertEqual(first.active, "B")
        self.assertEqual(pointer, first.model.get_base_model().model.embed_tokens.weight.data_ptr())
        self.assertEqual(before["risk_logits"].shape, (2, 3))
        self.assertEqual(before["quality_logits"].shape, (2, 2))
        first.switch("A")
        self.assertEqual(first.active, "A")
        with tempfile.TemporaryDirectory() as directory:
            first.save(directory, trained=False)
            self.assertFalse((Path(directory) / "model.safetensors").exists())
            torch.manual_seed(123)
            second = SharedReviewer(Qwen3ForCausalLM(config), adapter_init=False)
            with self.assertRaises(ValueError):
                second.restore(directory)
            second.restore(directory, require_trained=False)
            with torch.inference_mode():
                after = second.classify(inputs)
            self.assertTrue(torch.allclose(before["risk_logits"], after["risk_logits"], atol=1e-6))
            self.assertTrue(torch.allclose(before["quality_logits"], after["quality_logits"], atol=1e-6))
            for name, value in first.model.state_dict().items():
                if "lora_" in name:
                    self.assertTrue(torch.equal(value, second.model.state_dict()[name]))
            child_code = '''
import json, sys, torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from coder.model import SharedReviewer
config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
    head_dim=16, max_position_embeddings=64, tie_word_embeddings=True)
torch.manual_seed(123)
model = SharedReviewer(Qwen3ForCausalLM(config), adapter_init=False)
model.restore(sys.argv[1], require_trained=False)
inputs = {"input_ids": torch.tensor([[0, 0, 3, 4], [3, 4, 0, 0]]),
          "attention_mask": torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])}
with torch.inference_mode():
    output = model.classify(inputs)
print(json.dumps({k: v.tolist() for k, v in output.items()}))
'''
            fresh = subprocess.run([sys.executable, "-c", child_code, directory], capture_output=True, text=True, timeout=120)
            self.assertEqual(fresh.returncode, 0, fresh.stderr)
            reloaded = json.loads(fresh.stdout.strip().splitlines()[-1])
            self.assertTrue(torch.allclose(before["risk_logits"], torch.tensor(reloaded["risk_logits"]), atol=1e-6))

    def test_qwen35_text_only_shared_head(self):
        import torch
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        from coder.model import SharedReviewer
        config = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, layer_types=["full_attention"], max_position_embeddings=64,
            tie_word_embeddings=True)
        reviewer = SharedReviewer(Qwen3_5ForCausalLM(config))
        reviewer.model.eval()
        with torch.inference_mode():
            output = reviewer.classify({"input_ids": torch.tensor([[3, 4]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)})
        self.assertEqual(output["risk_logits"].shape, (1, 3))
        self.assertFalse(hasattr(reviewer.model.get_base_model(), "visual"))


if __name__ == "__main__":
    unittest.main()
