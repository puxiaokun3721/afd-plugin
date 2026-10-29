# SPDX-License-Identifier: Apache-2.0
"""Weight admission fails before any fused operator can consume a layer."""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from afd_plugin.compat.npu.mega_moe_abi import MegaMoeStaticConfig  # noqa: E402
from afd_plugin.model_executor.npu.async_cam_w4a8 import W4A8LayerWeights  # noqa: E402
from afd_plugin.v1.worker.npu.async_cam_megamoe import (  # noqa: E402
    MegaMoeLayeredExecutor,
)


@pytest.fixture
def weight_contract():
    config = MegaMoeStaticConfig(1, 1, 1, 1, 1, 1, 1024, 512, 1, 16, 16, 1, 4)
    layer = W4A8LayerWeights(
        layer_idx=2,
        w13=torch.zeros((1, 1024, 128), dtype=torch.int32),
        w2=torch.zeros((1, 512, 128), dtype=torch.int32),
        w13_scale=torch.zeros((1, 1024), dtype=torch.int64),
        w2_scale=torch.zeros((1, 1, 1024), dtype=torch.int64),
        w13_bias=torch.zeros((1, 1024), dtype=torch.float32),
        w2_bias=torch.zeros((1, 1024), dtype=torch.float32),
        per_channel=True,
        swiglu_limit=7.0,
        routed_scaling_factor=1.0,
    )
    return config, layer


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("layer_idx", 4, "layer IDs"),
        ("output_dtype", torch.float16, "BF16"),
        ("routed_scaling_factor", 2.5, "applied once"),
        ("swiglu_limit", float("nan"), "finite and nonnegative"),
        ("swiglu_limit", float("inf"), "finite and nonnegative"),
        ("swiglu_limit", -1.0, "finite and nonnegative"),
    ],
)
def test_invalid_layer_contract_rejected_before_operator_load(
    weight_contract, field, value, message
):
    config, layer = weight_contract
    with pytest.raises(ValueError, match=message):
        MegaMoeLayeredExecutor([replace(layer, **{field: value})], config)


def test_missing_remote_layer_is_rejected(weight_contract):
    config, _ = weight_contract
    with pytest.raises(ValueError, match="cover every"):
        MegaMoeLayeredExecutor([], config)


def test_distinct_layer_limits_do_not_allow_distinct_compute_geometry(weight_contract):
    config, layer = weight_contract
    other_layer = replace(
        layer,
        layer_idx=3,
        swiglu_limit=10.0,
        w13=torch.zeros((1, 1024, 64), dtype=torch.int32),
        w2=torch.zeros((1, 256, 128), dtype=torch.int32),
        w13_scale=torch.zeros((1, 512), dtype=torch.int64),
        w13_bias=torch.zeros((1, 512), dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="static compute tiling"):
        MegaMoeLayeredExecutor([layer, other_layer], replace(config, layer_count=2))
