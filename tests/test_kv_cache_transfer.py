from unittest.mock import patch

import torch

from tabicl._model.kv_cache import KVCache, KVCacheEntry, TabICLCache


def test_cache_transfer_chunks_tensors_larger_than_fixed_budget():
    key = torch.arange(3 * 11 * 5, dtype=torch.float32).reshape(3, 11, 5)
    value = key + 1000
    cache = TabICLCache(
        col_cache=KVCache(kv={0: KVCacheEntry(key=key, value=value)}),
        row_repr=key.clone(),
        train_shape=(3, 11, 5),
    )

    # One index along the widest dimension is 3 * 5 * 4 = 60 bytes, so
    # a 121-byte budget deterministically produces chunks of width two.
    with patch("tabicl._model.kv_cache.torch.cat", wraps=torch.cat) as cat:
        moved = cache.to("cpu", max_chunk_bytes=121)

    assert cat.call_count == 3
    torch.testing.assert_close(moved.col_cache.kv[0].key, key)
    torch.testing.assert_close(moved.col_cache.kv[0].value, value)
    torch.testing.assert_close(moved.row_repr, key)


def test_cache_transfer_skips_chunking_when_tensor_fits_budget():
    key = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    cache = KVCacheEntry(key=key, value=key.clone())

    with patch("tabicl._model.kv_cache.torch.cat", wraps=torch.cat) as cat:
        moved = cache.to("cpu", max_chunk_bytes=key.numel() * key.element_size())

    cat.assert_not_called()
    torch.testing.assert_close(moved.key, key)
    torch.testing.assert_close(moved.value, key)
