import unittest
import numpy as np
from benchmark import run_once
from tiny_llm import TinyTransformer, tokenize


class TinyLlmTests(unittest.TestCase):
    def test_cache_grows_one_position(self):
        model = TinyTransformer()
        cache = model.new_cache(2)
        logits = model.step(np.array([1, 2]), cache)
        self.assertEqual(logits.shape, (2, 256))
        self.assertEqual(cache[0][0].shape[2], 1)

    def test_kv_estimate_and_tokenizer(self):
        model = TinyTransformer()
        self.assertGreater(model.kv_cache_bytes(4, 64), model.kv_cache_bytes(1, 64))
        self.assertEqual(tokenize("a", 8, 4).shape, (4, 8))

    def test_parallel_prefill_matches_token_by_token_reference(self):
        model = TinyTransformer()
        token_ids = tokenize("parallel prefill", 7, 2)

        reference_cache = model.new_cache(2)
        for position in range(token_ids.shape[1]):
            reference_logits = model.step(token_ids[:, position], reference_cache)

        prefill_cache = model.new_cache(2)
        prefill_logits = model.prefill(token_ids, prefill_cache)

        np.testing.assert_allclose(prefill_logits, reference_logits, rtol=1e-5, atol=1e-6)
        for prefill_layer, reference_layer in zip(prefill_cache, reference_cache):
            np.testing.assert_allclose(prefill_layer[0], reference_layer[0], rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(prefill_layer[1], reference_layer[1], rtol=1e-5, atol=1e-6)

    def test_one_output_token_has_no_decode_metrics(self):
        model = TinyTransformer()
        result = run_once(model, tokenize("one token", 4, 1), output_length=1)

        self.assertIsNone(result["time_per_output_token_ms"])
        self.assertIsNone(result["decode_tokens_per_second"])
        self.assertGreater(result["prefill_tokens_per_second"], 0.0)


if __name__ == "__main__": unittest.main()
