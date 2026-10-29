# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Persistent layer tables and bounded fused MegaMoE work submission."""

from __future__ import annotations

import math
from collections import deque

import torch

from afd_plugin.compat.npu.cam_async_megamoe import MegaMoeContext
from afd_plugin.compat.npu.mega_moe_abi import MegaMoeStaticConfig
from afd_plugin.compat.npu.ops import ensure_mega_moe_ops_available
from afd_plugin.model_executor.npu.async_cam_w4a8 import (
    AsyncCAMW4A8Executor,
    W4A8LayerWeights,
)

MAX_PENDING_WORK_ITEMS = 2
SWIGLU_ACTIVATION_CODE = 0.0


class MegaMoeLayeredExecutor:
    """Keep six L*E lists stable; the device work descriptor chooses the layer."""

    def __init__(
        self, layers: list[W4A8LayerWeights], config: MegaMoeStaticConfig
    ) -> None:
        layers = sorted(layers, key=lambda layer: layer.layer_idx)
        if len(layers) != config.layer_count:
            raise ValueError(
                "MegaMoE weight list must cover every configured MoE layer"
            )
        self.layer_ids = tuple(layer.layer_idx for layer in layers)
        if (
            len(set(self.layer_ids)) != len(layers)
            or self.layer_ids[0] < 0
            or self.layer_ids[-1] >= config.layer_id_capacity
        ):
            raise ValueError("MegaMoE requires unique layer IDs inside the device LUT")
        for layer in layers:
            AsyncCAMW4A8Executor._validate_layer(layer)
            if not layer.per_channel or layer.output_dtype != torch.bfloat16:
                raise ValueError("MegaMoE requires per-channel W4A8 with BF16 output")
            if layer.w13.shape != (
                config.local_expert_count,
                config.hidden_size,
                2 * config.intermediate_size // 8,
            ):
                raise ValueError(
                    "MegaMoE layer shape differs from static compute tiling"
                )
            if not math.isfinite(layer.swiglu_limit) or layer.swiglu_limit < 0:
                raise ValueError("MegaMoE swiglu_limit must be finite and nonnegative")
            if layer.routed_scaling_factor != 1.0:
                raise ValueError(
                    "MegaMoE routed scaling must be applied once "
                    "in Attention topk weights"
                )
            if layer.w13.device != layers[0].w13.device:
                raise ValueError("MegaMoE layers must share a device")
        ensure_mega_moe_ops_available()
        # This binding reads the original NPU storage descriptor. It splits
        # existing grouped-NZ bytes into expert carriers, without re-quantizing
        # weights or interpreting packed INT32 bytes as an ND INT4 matrix.
        from afd_plugin import _C_ascend

        self.w13: list[torch.Tensor] = []
        self.w2: list[torch.Tensor] = []
        self.weight_scale13: list[torch.Tensor] = []
        self.weight_scale2: list[torch.Tensor] = []
        self.compensation13: list[torch.Tensor] = []
        self.compensation2: list[torch.Tensor] = []
        mapping = [-1] * config.layer_id_capacity
        parameters: list[list[float]] = []
        for slot, layer in enumerate(layers):
            mapping[layer.layer_idx] = slot
            self.w13.extend(_C_ascend.raw_extract_nz_experts(layer.w13))
            self.w2.extend(_C_ascend.raw_extract_nz_experts(layer.w2))
            for expert in range(config.local_expert_count):
                self.weight_scale13.append(layer.w13_scale[expert].reshape(-1))
                self.weight_scale2.append(layer.w2_scale[expert].reshape(-1))
                self.compensation13.append(layer.w13_bias[expert].reshape(-1))
                self.compensation2.append(layer.w2_bias[expert].reshape(-1))
            parameters.append([layer.swiglu_limit, SWIGLU_ACTIVATION_CODE, 0.0, 0.0])
        device = layers[0].w13.device
        self.layer_id_to_slot = torch.tensor(mapping, dtype=torch.int64, device=device)
        self.layer_params = torch.tensor(parameters, dtype=torch.float32, device=device)
        self._completions: deque[torch.npu.Event] = deque()

    def submit(self, context: MegaMoeContext) -> None:
        if len(self._completions) == MAX_PENDING_WORK_ITEMS:
            self._completions.popleft().synchronize()
            context.check_health()
        context.schedule()
        context.ffn(
            self.layer_id_to_slot,
            self.w13,
            self.w2,
            self.weight_scale13,
            self.weight_scale2,
            self.compensation13,
            self.compensation2,
            self.layer_params,
        )
        completion = torch.npu.Event()
        completion.record()
        self._completions.append(completion)

    def drain(self, context: MegaMoeContext) -> None:
        while self._completions:
            self._completions.popleft().synchronize()
        context.synchronize()
        context.check_health()
