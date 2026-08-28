"""Static-shape behavior of cache-aware attention layers."""

from __future__ import annotations

import torch

from tabicl._model.kv_cache import KVCache
from tabicl._model.layers import InducedSelfAttentionBlock


def _block() -> InducedSelfAttentionBlock:
    return InducedSelfAttentionBlock(
        d_model=4,
        nhead=1,
        dim_feedforward=8,
        num_inds=2,
    )


def test_cache_forward_restores_skip_batches_without_skipping_attention(monkeypatch):
    block = _block()
    src = torch.zeros(2, 3, 4)
    src[1] = block.skip_value
    calls = []

    def fake_attention(src, *args, **kwargs):
        calls.append(src.shape)
        return torch.ones_like(src)

    monkeypatch.setattr(block, "induced_attention_with_cache", fake_attention)
    out = block.forward_with_cache(src, KVCache(), 0, train_size=2, store_cache=True)

    assert calls == [src.shape]
    torch.testing.assert_close(out[0], torch.ones_like(out[0]))
    torch.testing.assert_close(out[1], torch.full_like(out[1], block.skip_value))


def test_cache_forward_keeps_static_path_when_every_batch_is_skipped(monkeypatch):
    block = _block()
    src = torch.full((2, 3, 4), block.skip_value)
    calls = []

    def fake_attention(src, *args, **kwargs):
        calls.append(src.shape)
        return torch.ones_like(src)

    monkeypatch.setattr(block, "induced_attention_with_cache", fake_attention)
    out = block.forward_with_cache(src, KVCache(), 0, train_size=2, store_cache=True)

    assert calls == [src.shape]
    torch.testing.assert_close(out, src)
