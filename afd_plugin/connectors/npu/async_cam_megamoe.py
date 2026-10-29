# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A3 prefill MegaMoE transport, with one Attention slot per rank."""

from __future__ import annotations

import json
import os
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import timedelta
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import PrefixStore, ProcessGroup, Store

from afd_plugin.compat.npu.cam_async_megamoe import MegaMoeContext
from afd_plugin.compat.npu.mega_moe_abi import MegaMoeStaticConfig
from afd_plugin.compat.npu.ops import ensure_mega_moe_ops_available
from afd_plugin.config import AFDConfig
from afd_plugin.config_utils import coerce_extra_positive_int
from afd_plugin.connectors.base import AFDConnectorBase
from afd_plugin.connectors.metadata import AFDA2FTransferPayload, AFDTransferContext
from afd_plugin.connectors.npu.async_cam import AFDAsyncExtraInfo, build_async_topology
from afd_plugin.distributed import (
    ProcessGroupRendezvousContext,
    create_hccl_process_group_options,
    init_afd_process_group,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

AFD_MEGAMOE_GROUP_NAME = "afd_async_cam_megamoe_v1"
DEFAULT_RECV_ROUTE_CAPACITY = 4096
DEFAULT_TASK_TIMEOUT_SECONDS = 300
HANDSHAKE_TIMEOUT = timedelta(minutes=5)
CONTROL_POLL_SECONDS = 0.05
CONTROL_JOIN_SECONDS = 5
HEARTBEAT_INTERVAL_SECONDS = 1
HEARTBEAT_TIMEOUT_SECONDS = 300
PREFILL_ADMISSION_ERROR = 1


@dataclass(frozen=True)
class MegaMoeExtraInfo(AFDAsyncExtraInfo):
    """Q bounds F private scratch; it does not shrink the A communication slot."""

    recv_route_capacity: int = DEFAULT_RECV_ROUTE_CAPACITY
    source_token_capacity: int | None = None
    task_timeout_seconds: int = DEFAULT_TASK_TIMEOUT_SECONDS

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> MegaMoeExtraInfo:
        if raw is not None and not isinstance(raw, Mapping):
            raise TypeError("MegaMoE connector_extra_config must be a mapping")
        values = dict(raw or {})
        task_timeout = coerce_extra_positive_int(
            values.pop("task_timeout_seconds", DEFAULT_TASK_TIMEOUT_SECONDS),
            field_name="task_timeout_seconds",
        )
        receive_capacity = coerce_extra_positive_int(
            values.pop("recv_route_capacity", DEFAULT_RECV_ROUTE_CAPACITY),
            field_name="recv_route_capacity",
        )
        source_capacity = values.pop("source_token_capacity", None)
        if source_capacity is not None:
            source_capacity = coerce_extra_positive_int(
                source_capacity, field_name="source_token_capacity"
            )
        base = AFDAsyncExtraInfo.from_mapping(values)
        return cls(
            **vars(base),
            recv_route_capacity=receive_capacity,
            source_token_capacity=source_capacity,
            task_timeout_seconds=task_timeout,
        )

    def to_mapping(self) -> dict[str, Any]:
        values = super().to_mapping()
        values["recv_route_capacity"] = self.recv_route_capacity
        values["task_timeout_seconds"] = self.task_timeout_seconds
        if self.source_token_capacity is not None:
            values["source_token_capacity"] = self.source_token_capacity
        return values


@dataclass
class MegaMoeTaskCompletion:
    started_at: float
    event: torch.npu.Event | None = None


@dataclass(frozen=True)
class PendingMegaMoeTask:
    stage_idx: int
    topk_weights: torch.Tensor
    token_mask: torch.Tensor
    task_meta: torch.Tensor
    completion: MegaMoeTaskCompletion


class AsyncCamMegaMoeConnector(AFDConnectorBase):
    """Independent MegaMoE protocol; no legacy DispatchRecv/CombineSend payload."""

    @classmethod
    def parse_extra_config(cls, raw: Mapping[str, Any] | None) -> MegaMoeExtraInfo:
        return MegaMoeExtraInfo.from_mapping(raw)

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
        *,
        rendezvous_context: ProcessGroupRendezvousContext | None = None,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        extra = cast(MegaMoeExtraInfo, self.extra_info)
        model = vllm_config.model_config.hf_config
        self.topology = build_async_topology(
            afd_config, role_rank, num_routed_experts=model.n_routed_experts
        )
        self.tp_size = extra.attn_ranks_per_dp
        self.attn_size = self.topology.attn_size
        self.ffn_size = self.topology.ffn_size
        self.world_rank = self.topology.world_rank
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # All TP members store disjoint shards, including SP padding. Reserve
        # the whole scheduler batch: ubatch imbalance cannot exceed this bound.
        source_capacity = extra.source_token_capacity or (
            (max_tokens + self.tp_size - 1) // self.tp_size
        )
        # The supported DSV4 decoder constructs a MoE at every layer (see
        # AFDDeepseekV4DecoderLayer); dense-layer models are rejected upstream.
        self.layer_ids = tuple(range(model.num_hidden_layers))
        self.static_config = MegaMoeStaticConfig(
            session_epoch=1,
            physical_rank=self.world_rank,
            attn_dp_size=self.attn_size // self.tp_size,
            attn_tp_size=self.tp_size,
            ffn_ep_size=self.ffn_size,
            local_expert_count=self.topology.expert_per_rank,
            hidden_size=model.hidden_size,
            intermediate_size=model.moe_intermediate_size,
            topk=model.num_experts_per_tok,
            source_token_capacity=source_capacity,
            recv_route_capacity=extra.recv_route_capacity,
            layer_count=len(self.layer_ids),
            layer_id_capacity=model.num_hidden_layers,
        )
        self.group_name = ""
        self.cam_pg: ProcessGroup | None = None
        self.context: MegaMoeContext | None = None
        self._rendezvous_context = rendezvous_context
        self._pending: PendingMegaMoeTask | None = None
        self._sequence = 0
        self._task_lock = Lock()
        self._domain_error: str | None = None
        self._task_completions: deque[MegaMoeTaskCompletion] = deque()
        self._task_timeout_seconds = extra.task_timeout_seconds
        self._prefill_admitted = False
        self._management_store: Store | None = None
        self._control_stop = Event()
        self._control_thread: Thread | None = None

    @property
    def is_initialized(self) -> bool:
        return self.context is not None

    def _handshake(self, store: Store) -> None:
        if self._rendezvous_context is not None:
            self._rendezvous_context.retain_store(store)
        handshake = PrefixStore(AFD_MEGAMOE_GROUP_NAME, store)
        handshake.set_timeout(HANDSHAKE_TIMEOUT)
        self._management_store = handshake
        if self.world_rank == 0:
            handshake.set(
                "epoch",
                str((int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1)) or 1),
            )
        epoch = int(handshake.get("epoch").decode())
        self.static_config = replace(self.static_config, session_epoch=epoch)
        config = self.static_config
        # Rank and role differ intentionally; every memory/compute contract
        # must agree before any process creates a communicator or publishes Ready.
        signature = dict(vars(config))
        signature.pop("physical_rank")
        signature.update(
            prefill_only=True,
            quant="per_channel_w4a8",
            hccl_buffer_size=cast(MegaMoeExtraInfo, self.extra_info).hccl_buffer_size,
            activation=str(self.vllm_config.model_config.dtype),
            layer_ids=self.layer_ids,
            torch_version=torch.__version__,
            task_timeout_seconds=self._task_timeout_seconds,
        )
        encoded = json.dumps(signature, sort_keys=True)
        handshake.set(f"config/{self.world_rank}", encoded)
        keys = [f"config/{rank}" for rank in range(config.world_size)]
        handshake.wait(keys, HANDSHAKE_TIMEOUT)
        if any(handshake.get(key).decode() != encoded for key in keys):
            raise ValueError("MegaMoE startup configuration differs between A/F ranks")

    def init_afd_connector(self) -> None:
        if self.is_initialized:
            return
        ensure_mega_moe_ops_available()
        self.cam_pg = init_afd_process_group(
            backend="hccl",
            init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
            world_size=self.static_config.world_size,
            rank=self.world_rank,
            group_name=AFD_MEGAMOE_GROUP_NAME,
            timeout=HANDSHAKE_TIMEOUT,
            pg_options=create_hccl_process_group_options(
                cast(MegaMoeExtraInfo, self.extra_info).hccl_buffer_size
            ),
            on_rendezvous=self._handshake,
        )
        backend = self.cam_pg._get_backend(torch.device("npu"))
        self.group_name = str(backend.get_hccl_comm_name(self.world_rank))
        self.context = MegaMoeContext(
            self.static_config, self.group_name, torch.device("npu", self.local_rank)
        )
        if self._rendezvous_context is not None:
            self._rendezvous_context.bind(self.cam_pg)
        self._control_thread = Thread(
            target=self._monitor_domain_abort, name="afd-megamoe-control", daemon=True
        )
        self._control_thread.start()

    def validate_ffn_resources(
        self, layer_ids: tuple[int, ...], layer_params: tuple[tuple[float, ...], ...]
    ) -> None:
        """All F resources agree before startup publishes any receiver ready."""
        if layer_ids != self.layer_ids:
            raise ValueError("MegaMoE loaded layer IDs differ from model topology")
        store = self._management_store
        if store is None:
            raise RuntimeError("MegaMoE resource check requires the startup Store")
        encoded = json.dumps((layer_ids, layer_params))
        store.set(f"resources/{self.world_rank}", encoded)
        keys = [
            f"resources/{rank}"
            for rank in range(self.attn_size, self.static_config.world_size)
        ]
        deadline = time.monotonic() + HANDSHAKE_TIMEOUT.total_seconds()
        while not store.check(keys):
            if store.check(["abort"]):
                raise RuntimeError(
                    "MegaMoE FFN resource initialization aborted by a peer"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("MegaMoE FFN resource initialization timed out")
            self._control_stop.wait(CONTROL_POLL_SECONDS)
        if any(store.get(key).decode() != encoded for key in keys):
            raise ValueError("MegaMoE FFN layer maps or activation parameters differ")

    def abort_domain(self, reason: str) -> None:
        """Publish a sticky host failure; every idle F observes it independently."""
        with self._task_lock:
            if self._domain_error is None:
                self._domain_error = reason
        try:
            if self._management_store is not None:
                self._management_store.compare_set("abort", "", reason)
        finally:
            if self.context is not None:
                self.context.abort(PREFILL_ADMISSION_ERROR)

    def _monitor_domain_abort(self) -> None:
        store, context = self._management_store, self.context
        assert store is not None and context is not None
        torch.npu.set_device(self.local_rank)
        observed = {
            rank: (b"", time.monotonic())
            for rank in range(self.static_config.world_size)
        }
        heartbeat, next_heartbeat = 0, 0.0
        try:
            while not self._control_stop.wait(CONTROL_POLL_SECONDS):
                if store.check(["abort"]):
                    self.abort_domain(store.get("abort").decode())
                    return
                now = time.monotonic()
                if now < next_heartbeat:
                    continue
                # A failed DS/CR can complete its event without writing output.
                # Observe fatal before retiring completion records, including
                # on an otherwise idle F whose peer never published Ready.
                context.check_health()
                with self._task_lock:
                    while (
                        self._task_completions
                        and self._task_completions[0].event is not None
                        and self._task_completions[0].event.query()
                    ):
                        self._task_completions.popleft()
                    stalled = (
                        bool(self._task_completions)
                        and now - self._task_completions[0].started_at
                        > self._task_timeout_seconds
                    )
                if stalled:
                    self.abort_domain("Attention task completion deadline exceeded")
                    return
                heartbeat += 1
                store.set(f"heartbeat/{self.world_rank}", str(heartbeat))
                next_heartbeat = now + HEARTBEAT_INTERVAL_SECONDS
                for rank, (previous, last_change) in observed.items():
                    key = f"heartbeat/{rank}"
                    current = store.get(key) if store.check([key]) else b""
                    if current != previous:
                        observed[rank] = (current, now)
                    elif now - last_change > HEARTBEAT_TIMEOUT_SECONDS:
                        self.abort_domain(f"rank {rank} management heartbeat stopped")
                        return
        except Exception as exc:
            # Store loss and device fatal both poison the host connector. If
            # publication itself fails, abort_domain's finally still attempts
            # the local management copy; later data/model calls raise sticky
            # error instead of accepting an unwritten result.
            try:
                self.abort_domain(str(exc))
            except Exception:
                return

    def raise_if_domain_failed(self) -> None:
        """Check only host error state; safe on every DS/CR without a device sync."""
        with self._task_lock:
            reason = self._domain_error
        if reason is not None:
            raise RuntimeError(f"MegaMoE domain failed: {reason}")

    def complete_attention_step(self) -> None:
        """Fence one complete model step before its result may reach the caller."""
        self.raise_if_domain_failed()
        runtime = self.context
        if runtime is None or not self.static_config.is_attention:
            raise RuntimeError("MegaMoE step completion requires initialized Attention")
        try:
            # execute_model's sampling/CPU-result boundary normally already
            # observes completion. Explicitly cover asynchronous-output variants
            # as well; synchronize includes a fatal check, once per model step.
            runtime.synchronize()
            self.raise_if_domain_failed()
            if self._management_store is not None and self._management_store.check(
                ["abort"]
            ):
                reason = self._management_store.get("abort").decode()
                self.abort_domain(reason)
                self.raise_if_domain_failed()
        except Exception as exc:
            self.abort_domain(str(exc))
            raise

    def admit_prefill(self, is_prefill: bool) -> None:
        """Called after the runner's TP-wide admission, before the first Send."""
        self.raise_if_domain_failed()
        if not is_prefill:
            self.abort_domain("decode/mixed batch violates prefill-only admission")
            raise RuntimeError(
                "cam_async_megamoe supports pure prefill only; "
                "decode/mixed batch rejected"
            )
        self._prefill_admitted = True

    def send_attn_output(
        self, hidden_states: torch.Tensor, context: AFDTransferContext, **kwargs: Any
    ) -> None:
        self.raise_if_domain_failed()
        runtime = self.context
        if runtime is None or not self.static_config.is_attention:
            raise RuntimeError("MegaMoE Send requires an initialized Attention rank")
        if not self._prefill_admitted:
            raise RuntimeError("MegaMoE Send requires runner prefill admission")
        if self._pending is not None:
            raise RuntimeError(
                "MegaMoE single slot requires CombineRecv before the next Send"
            )
        mask = kwargs.get("token_mask")
        if mask is None:
            raise ValueError("MegaMoE requires the stage/TP-derived token_mask")
        ids = kwargs["topk_ids"].to(dtype=torch.int32).contiguous()
        weights = kwargs["topk_weights"].to(dtype=torch.float32).contiguous()
        rows = hidden_states.shape[0]
        if rows != context.metadata.total_tokens:
            raise ValueError("MegaMoE input extent differs from transfer metadata")
        self._sequence += 1
        task_meta = torch.tensor(
            self.static_config.make_task_meta(
                self._sequence,
                self.role_rank // self.tp_size,
                context.metadata.layer_idx,
                context.metadata.stage_idx,
                rows,
            ),
            dtype=torch.int64,
            device=hidden_states.device,
        )
        completion = MegaMoeTaskCompletion(time.monotonic())
        with self._task_lock:
            self._task_completions.append(completion)
        try:
            runtime.dispatch_send(hidden_states.contiguous(), ids, mask, task_meta)
        except Exception as exc:
            self.abort_domain(str(exc))
            raise
        self._pending = PendingMegaMoeTask(
            context.metadata.stage_idx, weights, mask, task_meta, completion
        )

    def recv_ffn_output(
        self, ref_tensor: torch.Tensor, ubatch_idx: int = 0, **kwargs: Any
    ) -> torch.Tensor:
        self.raise_if_domain_failed()
        pending = self._pending
        if self.context is None or pending is None or pending.stage_idx != ubatch_idx:
            raise RuntimeError("MegaMoE CombineRecv has no matching pending microbatch")
        output = torch.empty_like(ref_tensor, dtype=torch.bfloat16)
        self.context.combine_recv(
            pending.topk_weights, pending.token_mask, pending.task_meta, output
        )
        completion = torch.npu.Event()
        completion.record()
        with self._task_lock:
            pending.completion.event = completion
        self._pending = None
        return output

    def request_stop(self) -> None:
        if self.context is not None:
            self.context.request_stop()

    def close(self) -> None:
        if self._pending is not None:
            self.abort_domain("Attention shutdown before CombineRecv")
            raise RuntimeError(
                "MegaMoE has an unconsumed Attention slot; "
                "retain communication resources"
            )
        if self.context is not None:
            self.context.synchronize()
            self.context.check_health()
        self._control_stop.set()
        if self._control_thread is not None:
            self._control_thread.join(timeout=CONTROL_JOIN_SECONDS)
            if self._control_thread.is_alive():
                raise RuntimeError(
                    "MegaMoE management thread did not stop; retain communicator"
                )
        if self.cam_pg is not None:
            dist.destroy_process_group(self.cam_pg)
        self.cam_pg = None
        self.context = None
        self._pending = None
        if self._rendezvous_context is not None:
            self._rendezvous_context.invalidate()

    def recv_attn_output(
        self, ubatch_idx: int = 0, **kwargs: Any
    ) -> AFDA2FTransferPayload:
        raise RuntimeError(
            "MegaMoE uses fused Scheduler/FFN work items, not DispatchRecv payloads"
        )

    def send_ffn_output(
        self, ffn_output: torch.Tensor, context: AFDTransferContext, **kwargs: Any
    ) -> None:
        raise RuntimeError(
            "MegaMoE FFN performs fused writeback; CombineSend is not its interface"
        )
