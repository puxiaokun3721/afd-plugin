# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Persistent buffers and stream ordering for the four A3 MegaMoE operators."""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock
from typing import TYPE_CHECKING

import torch
import torch_npu  # noqa: F401 -- registers the NPU device/stream implementation

from afd_plugin.compat.npu.mega_moe_abi import (
    ABI_VERSION,
    COMPLETION_FIELDS,
    CONTROL_ABORT_INDEX,
    CONTROL_ELEMENTS,
    CONTROL_STOP_INDEX,
    STATE_HEADER_FIELDS,
    TASK_FIELDS,
    WORK_FIELDS,
    ErrorCode,
    MegaMoeStaticConfig,
)
from afd_plugin.compat.npu.ops import ensure_mega_moe_ops_available

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class MegaMoeContext:
    """Own one physical source slot, or one serial FFN work-item context.

    HCCL windows belong to the connector's process group, not to this object.
    The connector must keep that group alive until ``synchronize`` succeeds.
    A control copy stream is independent of the possibly blocking task stream.
    Device qualification must verify its progress on the target CANN runtime.
    """

    def __init__(
        self,
        config: MegaMoeStaticConfig,
        group_name: str,
        device: torch.device | str,
    ) -> None:
        ensure_mega_moe_ops_available()
        self.config = config
        self.group_name = group_name
        self.device = torch.device(device)
        if self.device.type != "npu":
            raise ValueError("A3 MegaMoE requires an NPU device")
        self.config_values = list(config.config_values)
        self.layout = config.layout
        self.topology = torch.tensor(
            config.topology, dtype=torch.int32, device=self.device
        )
        self.comm_args = torch.zeros(1, dtype=torch.float16, device=self.device)
        initial_state = [0] * (STATE_HEADER_FIELDS + config.attn_dp_size)
        initial_state[0:2] = [ABI_VERSION, config.session_epoch]
        self.state = torch.tensor(initial_state, dtype=torch.int64, device=self.device)
        self.control = torch.zeros(
            CONTROL_ELEMENTS, dtype=torch.int32, device=self.device
        )
        self.completion = torch.zeros(
            COMPLETION_FIELDS, dtype=torch.int64, device=self.device
        )
        self.work_desc = torch.zeros(WORK_FIELDS, dtype=torch.int64, device=self.device)
        self.source_desc = torch.zeros(
            (config.attn_tp_size, TASK_FIELDS), dtype=torch.int64, device=self.device
        )
        if config.is_attention:
            self.route_index = torch.empty(
                self.layout.padded_tokens * config.topk,
                dtype=torch.int32,
                device=self.device,
            )
            self.source_counts = torch.empty(
                (self.layout.num_chunks, self.layout.expert_stride),
                dtype=torch.int32,
                device=self.device,
            )
            self.send_scratch = torch.empty(
                config.send_scratch_bytes, dtype=torch.uint8, device=self.device
            )
            self.ffn_scratch = torch.empty(0, dtype=torch.uint8, device=self.device)
        else:
            self.route_index = torch.empty(0, dtype=torch.int32, device=self.device)
            self.source_counts = torch.empty(0, dtype=torch.int32, device=self.device)
            self.send_scratch = torch.empty(0, dtype=torch.uint8, device=self.device)
            self.ffn_scratch = torch.empty(
                config.ffn_scratch_bytes, dtype=torch.uint8, device=self.device
            )
        self._enqueue_lock = Lock()
        self._control_lock = Lock()
        self._execution_stream = torch.npu.current_stream(self.device)
        self._control_stream = torch.npu.Stream(device=self.device)
        self._control_stream.wait_stream(self._execution_stream)
        self._control_shadow = torch.zeros(
            CONTROL_ELEMENTS, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._handoff_event = None

    @contextmanager
    def _ordered_launch(self) -> Iterator[None]:
        # No event on the normal same-stream path. Cross-stream Attention
        # callbacks retain the single-slot ordering without a host synchronize.
        with self._enqueue_lock:
            current = torch.npu.current_stream(self.device)
            if current != self._execution_stream:
                event = torch.npu.Event()
                event.record(self._execution_stream)
                current.wait_event(event)
                self._handoff_event = event
                self._execution_stream = current
            yield

    def dispatch_send(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        token_mask: torch.Tensor,
        task_meta: torch.Tensor,
    ) -> None:
        if not self.config.is_attention:
            raise ValueError("dispatch_send is an Attention operation")
        with self._ordered_launch():
            for tensor in (x, expert_ids, token_mask, task_meta):
                tensor.record_stream(self._execution_stream)
            torch.ops.afd_ascend.afd_mega_moe_dispatch_send(
                x,
                expert_ids,
                token_mask,
                task_meta,
                self.comm_args,
                self.topology,
                self.control,
                self.state,
                self.route_index,
                self.source_counts,
                self.send_scratch,
                group_name=self.group_name,
                config=self.config_values,
            )

    def schedule(self) -> None:
        if self.config.is_attention:
            raise ValueError("schedule is an FFN operation")
        with self._ordered_launch():
            torch.ops.afd_ascend.afd_mega_moe_scheduler(
                self.comm_args,
                self.topology,
                self.control,
                self.state,
                self.work_desc,
                self.source_desc,
                group_name=self.group_name,
                config=self.config_values,
            )

    def ffn(
        self,
        layer_id_to_slot: torch.Tensor,
        w13: Sequence[torch.Tensor],
        w2: Sequence[torch.Tensor],
        weight_scale13: Sequence[torch.Tensor],
        weight_scale2: Sequence[torch.Tensor],
        compensation13: Sequence[torch.Tensor],
        compensation2: Sequence[torch.Tensor],
        layer_params: torch.Tensor,
    ) -> None:
        if self.config.is_attention:
            raise ValueError("ffn is an FFN operation")
        with self._ordered_launch():
            torch.ops.afd_ascend.afd_mega_moe_ffn(
                self.work_desc,
                self.source_desc,
                layer_id_to_slot,
                w13,
                w2,
                weight_scale13,
                weight_scale2,
                compensation13,
                compensation2,
                layer_params,
                self.comm_args,
                self.topology,
                self.control,
                self.state,
                self.ffn_scratch,
                self.completion,
                group_name=self.group_name,
                config=self.config_values,
            )

    def combine_recv(
        self,
        topk_weights: torch.Tensor,
        token_mask: torch.Tensor,
        task_meta: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        if not self.config.is_attention:
            raise ValueError("combine_recv is an Attention operation")
        with self._ordered_launch():
            # These tensors can originate on the other microbatch's stream.
            # Ordering the kernels alone does not protect their allocator slabs
            # after the connector releases its pending transaction references.
            for tensor in (topk_weights, token_mask, task_meta, out):
                tensor.record_stream(self._execution_stream)
            torch.ops.afd_ascend.afd_mega_moe_combine_recv(
                topk_weights,
                token_mask,
                task_meta,
                self.route_index,
                self.comm_args,
                self.topology,
                self.control,
                self.state,
                out,
                self.completion,
                group_name=self.group_name,
                config=self.config_values,
            )

    def _publish_control(self, index: int, value: int) -> None:
        with self._control_lock:
            # Preserve the host buffer until the previous nonblocking DMA ends.
            # This synchronization never targets the blocking task stream.
            self._control_stream.synchronize()
            self._control_shadow[index] = value
            with torch.npu.stream(self._control_stream):
                self.control.copy_(self._control_shadow, non_blocking=True)

    def request_stop(self) -> None:
        self._publish_control(CONTROL_STOP_INDEX, 1)

    def abort(self, error_code: int = ErrorCode.ABORTED) -> None:
        if not 1 <= int(error_code) <= (1 << 31) - 1:
            raise ValueError("abort error code must be a positive int32")
        self._publish_control(CONTROL_ABORT_INDEX, int(error_code))

    def check_health(self) -> None:
        """Observe only fatal status, outside the per-work-item data path."""
        with self._control_lock, torch.npu.stream(self._control_stream):
            fatal = int(self.state[3:4].to(device="cpu").tolist()[0])
        if fatal:
            raise RuntimeError(f"A3 MegaMoE device protocol failed: error_code={fatal}")

    def synchronize(self) -> None:
        """Drain after STOP or after the caller completed its transaction."""
        self._execution_stream.synchronize()
        self._control_stream.synchronize()
        self.check_health()
