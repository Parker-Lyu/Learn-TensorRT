import unittest
import numpy as np
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


if __name__ == "__main__": unittest.main()
