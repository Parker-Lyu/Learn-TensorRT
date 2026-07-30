from __future__ import annotations

import dataclasses
import hashlib
import json
import math

import numpy as np


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    hidden_size: int = 64
    heads: int = 4
    layers: int = 2
    mlp_size: int = 128
    seed: int = 20260714

    @property
    def revision(self) -> str:
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True).encode()
        return "tiny-transformer-" + hashlib.sha256(payload).hexdigest()[:12]


class TinyTransformer:
    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config
        rng = np.random.default_rng(config.seed)
        scale = 1.0 / math.sqrt(config.hidden_size)
        self.embedding = rng.normal(0, scale, (config.vocab_size, config.hidden_size)).astype(np.float32)
        self.layers = []
        for _ in range(config.layers):
            self.layers.append(tuple(rng.normal(0, scale, shape).astype(np.float32) for shape in [
                (config.hidden_size, config.hidden_size), (config.hidden_size, config.hidden_size),
                (config.hidden_size, config.hidden_size), (config.hidden_size, config.hidden_size),
                (config.hidden_size, config.mlp_size), (config.mlp_size, config.hidden_size)]))
        self.lm_head = rng.normal(0, scale, (config.hidden_size, config.vocab_size)).astype(np.float32)

    def new_cache(self, batch: int):
        head_dim = self.config.hidden_size // self.config.heads
        return [[np.empty((batch, self.config.heads, 0, head_dim), np.float32),
                 np.empty((batch, self.config.heads, 0, head_dim), np.float32)]
                for _ in range(self.config.layers)]

    def step(self, token_ids: np.ndarray, cache) -> np.ndarray:
        x = self.embedding[token_ids]
        batch = x.shape[0]
        heads = self.config.heads
        head_dim = self.config.hidden_size // heads
        for index, (wq, wk, wv, wo, w1, w2) in enumerate(self.layers):
            q = (x @ wq).reshape(batch, heads, head_dim)
            k = (x @ wk).reshape(batch, heads, 1, head_dim)
            v = (x @ wv).reshape(batch, heads, 1, head_dim)
            cache[index][0] = np.concatenate([cache[index][0], k], axis=2)
            cache[index][1] = np.concatenate([cache[index][1], v], axis=2)
            scores = np.einsum("bhd,bhtd->bht", q, cache[index][0]) / math.sqrt(head_dim)
            scores -= scores.max(axis=-1, keepdims=True)
            attention = np.exp(scores)
            attention /= attention.sum(axis=-1, keepdims=True)
            context = np.einsum("bht,bhtd->bhd", attention, cache[index][1]).reshape(batch, -1)
            x = x + context @ wo
            x = x + np.tanh(x @ w1) @ w2
        return x @ self.lm_head

    def weight_bytes(self) -> int:
        arrays = [self.embedding, self.lm_head]
        arrays.extend(weight for layer in self.layers for weight in layer)
        return sum(array.nbytes for array in arrays)

    def kv_cache_bytes(self, batch: int, sequence_length: int) -> int:
        return (self.config.layers * 2 * batch * self.config.heads * sequence_length *
                (self.config.hidden_size // self.config.heads) * 4)


def tokenize(text: str, length: int, batch: int) -> np.ndarray:
    encoded = list(text.encode("utf-8")) or [0]
    sequence = np.array([encoded[index % len(encoded)] for index in range(length)], np.int64)
    return np.tile(sequence, (batch, 1))
