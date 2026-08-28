import types

import torch

from tabicl._model.tabicl import TabICL


class _RawQuantileDeviceSpy:
    def __init__(self):
        self.requested_device = None

    def to(self, device):
        self.requested_device = device
        return torch.tensor([[[1.0, 2.0, 3.0]]], device=device)


class _Distribution:
    def __init__(self, quantiles):
        self.quantiles = quantiles


class _HostQuantilePostprocessor:
    alpha_levels = torch.tensor([0.25, 0.5, 0.75])

    def __call__(self, quantiles):
        return _Distribution(quantiles)


def test_regression_moves_accelerator_output_to_quantile_postprocessor_device():
    model = TabICL(
        max_classes=0,
        num_quantiles=3,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=1,
        col_num_inds=2,
        col_feature_group=False,
        row_num_blocks=1,
        row_nhead=1,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=1,
    )
    raw_quantiles = _RawQuantileDeviceSpy()

    def forward_with_cache(self, **_kwargs):
        return raw_quantiles

    model.forward_with_cache = types.MethodType(forward_with_cache, model)
    del model._modules["quantile_dist"]
    object.__setattr__(model, "quantile_dist", _HostQuantilePostprocessor())

    result = model.predict_stats_with_cache(
        X_test=torch.zeros(1, 1, 1),
        use_cache=True,
        store_cache=False,
        output_type="mean",
    )

    assert raw_quantiles.requested_device == torch.device("cpu")
    torch.testing.assert_close(result, torch.tensor([[2.0]]))
