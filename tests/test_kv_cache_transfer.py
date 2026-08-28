from unittest.mock import patch

import torch

from tabicl._model.kv_cache import KVCache, KVCacheEntry, TabICLCache
from tabicl._sklearn.base import TabICLBaseEstimator


def test_cache_transfer_chunks_tensors_larger_than_fixed_budget():
    key = torch.arange(3 * 11 * 5, dtype=torch.float32).reshape(3, 11, 5)
    value = key + 1000
    cache = TabICLCache(
        col_cache=KVCache(kv={0: KVCacheEntry(key=key, value=value)}),
        row_repr=key.clone(),
        train_shape=(3, 11, 5),
    )

    # 165 float32 values at 121 bytes per transfer produce six chunks.
    with patch("tabicl._model.kv_cache.torch.cat", wraps=torch.cat) as cat:
        moved = cache.to("cpu", max_chunk_bytes=121)

    assert cat.call_count == 3
    torch.testing.assert_close(moved.col_cache.kv[0].key, key)
    torch.testing.assert_close(moved.col_cache.kv[0].value, value)
    torch.testing.assert_close(moved.row_repr, key)


def test_chunked_transfer_preserves_non_contiguous_logical_order():
    tensor = torch.arange(5 * 7, dtype=torch.float32).reshape(5, 7).T
    assert not tensor.is_contiguous()
    cache = KVCacheEntry(key=tensor, value=tensor + 100)

    moved = cache.to("cpu", max_chunk_bytes=5 * tensor.element_size())

    torch.testing.assert_close(moved.key, tensor)
    torch.testing.assert_close(moved.value, tensor + 100)


def test_cache_transfer_skips_chunking_when_tensor_fits_budget():
    key = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    cache = KVCacheEntry(key=key, value=key.clone())

    with patch("tabicl._model.kv_cache.torch.cat", wraps=torch.cat) as cat:
        moved = cache.to("cpu", max_chunk_bytes=key.numel() * key.element_size())

    cat.assert_not_called()
    torch.testing.assert_close(moved.key, key)
    torch.testing.assert_close(moved.value, key)


def test_pickle_state_excludes_cache_when_save_kv_cache_is_false():
    estimator = object.__new__(TabICLBaseEstimator)
    estimator.model_kv_cache_ = {"none": TabICLCache(row_repr=torch.ones(2, 3))}
    estimator._save_model_weights = False
    estimator._save_training_data = True
    estimator._save_kv_cache = False

    state = estimator.__getstate__()

    assert "model_kv_cache_" not in state
