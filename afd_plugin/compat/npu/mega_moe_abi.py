# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host metadata for A3 MegaMoE ABI v1; independent of torch and vLLM.

Byte offsets are the wire contract shared with afd_mega_moe/abi.h. This module
does not emulate the device protocol, fences, or numerical kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

ABI_VERSION = 1
CONFIG_FIELDS = 24
TASK_FIELDS = 16
WORK_FIELDS = 32
STATE_HEADER_FIELDS = 32
COMPLETION_FIELDS = 8
RECORD_BYTES = 512
CONTROL_ELEMENTS = 128
CONTROL_STOP_INDEX = 0
CONTROL_ABORT_INDEX = 32
MAX_TICKET = (1 << 31) - 1
MAX_INT64 = (1 << 63) - 1
PERMUTE_TOKENS = 1024
PERMUTE_CORES = 128
PREFIX_RESERVED_BYTES = 10 * 1024 * 1024
TAIL_RESERVED_BYTES = 512 * 1024
CONTROL_SCRATCH_BYTES = 64 * 1024
W4_ROUTE_CACHE_BYTES = 16032


class RunState(IntEnum):
    IDLE = 0
    PUBLISHED = 1
    CLAIMED = 2
    RUNNING = 3
    COMPLETE = 4
    STOPPED = 5
    ERROR = 6


class CompletionStatus(IntEnum):
    PENDING = 0
    SUCCESS = 1
    STOP = 2
    ERROR = 3


class ErrorCode(IntEnum):
    NONE = 0
    INVALID_CONFIG = 1
    WINDOW_TOO_SMALL = 2
    UNSUPPORTED_WEIGHT_LAYOUT = 3
    INVALID_LAYER = 4
    TASK_ID_MISMATCH = 5
    SEQUENCE_GAP = 6
    ROUTE_OUT_OF_RANGE = 7
    WORKSPACE_OVERFLOW = 8
    PEER_FAILED = 9
    ABORTED = 10
    INVALID_STATE = 11


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class MegaMoeWindowLayout:
    chunk_tokens: int
    num_chunks: int
    padded_tokens: int
    expert_stride: int
    x_base: int
    scale_base: int
    d_base: int
    done_base: int
    inbox_stride: int
    tail_bytes: int
    required_bytes: int

    def ready_offset(self, source_index: int) -> int:
        return PREFIX_RESERVED_BYTES + source_index * self.inbox_stride

    def counts_offset(self, source_index: int, tile_index: int) -> int:
        return (
            self.ready_offset(source_index)
            + 2 * RECORD_BYTES
            + 4 * tile_index * self.expert_stride
        )


@dataclass(frozen=True)
class MegaMoeStaticConfig:
    session_epoch: int
    physical_rank: int
    attn_dp_size: int
    attn_tp_size: int
    ffn_ep_size: int
    local_expert_count: int
    hidden_size: int
    intermediate_size: int
    topk: int
    source_token_capacity: int
    recv_route_capacity: int
    layer_count: int
    layer_id_capacity: int

    def __post_init__(self) -> None:
        integer_fields = (
            self.session_epoch,
            self.physical_rank,
            self.attn_dp_size,
            self.attn_tp_size,
            self.ffn_ep_size,
            self.local_expert_count,
            self.hidden_size,
            self.intermediate_size,
            self.topk,
            self.source_token_capacity,
            self.recv_route_capacity,
            self.layer_count,
            self.layer_id_capacity,
        )
        if any(type(value) is not int for value in integer_fields):
            raise ValueError("MegaMoE static fields must be integers")
        if not 0 < self.session_epoch <= MAX_INT64:
            raise ValueError("session_epoch must be a positive int64")
        if not 1 <= self.attn_dp_size <= 384 or not 1 <= self.attn_tp_size <= 128:
            raise ValueError("unsupported Attention DP/TP size")
        if not 1 <= self.ffn_ep_size <= 128 or not 1 <= self.local_expert_count <= 128:
            raise ValueError("unsupported FFN EP/local expert count")
        if self.num_experts > 1024:
            raise ValueError("native prologue supports at most 1024 global experts")
        if self.world_size > 384 or not 0 <= self.physical_rank < self.world_size:
            raise ValueError("invalid HCCL world size or physical rank")
        if not 1024 <= self.hidden_size <= 8192 or self.hidden_size % 512:
            raise ValueError("hidden_size must be 512-aligned in [1024, 8192]")
        if not 512 <= self.intermediate_size <= 3072 or self.intermediate_size % 256:
            raise ValueError("intermediate_size must be 256-aligned in [512, 3072]")
        if not 1 <= self.topk <= min(16, self.num_experts):
            raise ValueError("topk must be in [1, min(16, global experts)]")
        if not 1 <= self.source_token_capacity <= MAX_TICKET // self.topk:
            raise ValueError("source route capacity exceeds the native int32 index")
        if not 1 <= self.recv_route_capacity <= MAX_TICKET:
            raise ValueError("recv_route_capacity must be a positive int32")
        if not 1 <= self.layer_count <= self.layer_id_capacity <= 65536:
            raise ValueError("invalid layer count or global-layer LUT capacity")
        route_cache_bytes = (
            align_up(4 * self.attn_tp_size * align_up(self.local_expert_count, 8), 32)
            + align_up(4 * self.attn_tp_size * self.local_expert_count, 32)
            + align_up(4 * self.attn_tp_size, 32)
        )
        if route_cache_bytes > W4_ROUTE_CACHE_BYTES:
            raise ValueError(
                "source/expert route cache overlaps the native W4 UB workspace"
            )
        if max(self.layout.required_bytes, self.ffn_scratch_bytes) > MAX_INT64:
            raise ValueError("MegaMoE memory plan exceeds int64 byte addressing")

    @property
    def attn_rank_count(self) -> int:
        return self.attn_dp_size * self.attn_tp_size

    @property
    def world_size(self) -> int:
        return self.attn_rank_count + self.ffn_ep_size

    @property
    def is_attention(self) -> bool:
        return self.physical_rank < self.attn_rank_count

    @property
    def num_experts(self) -> int:
        return self.ffn_ep_size * self.local_expert_count

    @property
    def layout(self) -> MegaMoeWindowLayout:
        chunk = min(self.source_token_capacity, PERMUTE_TOKENS)
        tiles = (self.source_token_capacity + chunk - 1) // chunk
        padded = chunk * tiles
        expert_stride = align_up(self.num_experts + 1, 128)
        x_base = PREFIX_RESERVED_BYTES
        scale_base = align_up(
            x_base + padded * self.topk * (self.hidden_size + 512), 512
        )
        d_base = align_up(scale_base + 4 * padded * self.topk, 512)
        done_base = align_up(d_base + 2 * padded * self.topk * self.hidden_size, 512)
        inbox_stride = 2 * RECORD_BYTES + 4 * tiles * expert_stride
        tail = max(RECORD_BYTES * self.world_size, TAIL_RESERVED_BYTES)
        attention_bytes = done_base + RECORD_BYTES * self.ffn_ep_size + tail
        ffn_bytes = PREFIX_RESERVED_BYTES + self.attn_rank_count * inbox_stride + tail
        return MegaMoeWindowLayout(
            chunk,
            tiles,
            padded,
            expert_stride,
            x_base,
            scale_base,
            d_base,
            done_base,
            inbox_stride,
            tail,
            align_up(max(attention_bytes, ffn_bytes), 512),
        )

    @property
    def max_recv_rounds(self) -> int:
        routes = self.attn_tp_size * self.layout.chunk_tokens * self.topk
        return max(
            1, (routes + self.recv_route_capacity - 1) // self.recv_route_capacity
        )

    @property
    def send_scratch_bytes(self) -> int:
        return CONTROL_SCRATCH_BYTES + align_up(
            PERMUTE_CORES * align_up(self.num_experts, 8) * 4, 512
        )

    @property
    def ffn_native_scratch_bytes(self) -> int:
        sources, experts = self.attn_tp_size, self.local_expert_count
        tiles, rounds = self.layout.num_chunks, self.max_recv_rounds
        rows, hidden, inner = (
            self.recv_route_capacity,
            self.hidden_size,
            self.intermediate_size,
        )
        table_bytes = (
            PERMUTE_CORES * align_up(sources * experts, 8) * 4
            + PERMUTE_TOKENS * self.topk * 4
            + tiles * sources * sources * experts * 4
            + tiles * rounds * 3 * sources * experts * 4
        )
        return align_up(
            align_up(table_bytes, 512)
            + 2 * align_up(4 * rows, 512)
            + 4 * rows * max(2 * inner, hidden)
            + rows * max(hidden, inner)
            + 4 * sources * align_up(experts, 16),
            512,
        )

    @property
    def ffn_scratch_bytes(self) -> int:
        return (
            self.ffn_native_scratch_bytes
            + align_up(4 * self.attn_tp_size * self.layout.expert_stride, 512)
            + align_up(
                4
                * self.layout.num_chunks
                * self.attn_tp_size
                * self.local_expert_count,
                512,
            )
            + CONTROL_SCRATCH_BYTES
        )

    @property
    def config_values(self) -> tuple[int, ...]:
        return (
            ABI_VERSION,
            1,
            1,
            self.session_epoch,
            self.physical_rank,
            self.attn_dp_size,
            self.attn_tp_size,
            self.ffn_ep_size,
            self.local_expert_count,
            self.hidden_size,
            self.intermediate_size,
            self.topk,
            self.source_token_capacity,
            self.recv_route_capacity,
            self.layer_count,
            self.layer_id_capacity,
            0,
            self.layout.required_bytes,
            1,
            1,
            0,
            0,
            0,
            0,
        )

    @property
    def topology(self) -> tuple[tuple[int, int, int, int], ...]:
        attention = tuple(
            (0, rank // self.attn_tp_size, rank % self.attn_tp_size, rank)
            for rank in range(self.attn_rank_count)
        )
        ffn = tuple(
            (1, -1, rank, self.attn_rank_count + rank)
            for rank in range(self.ffn_ep_size)
        )
        return attention + ffn

    def make_task_meta(
        self,
        seq: int,
        dp: int,
        layer: int,
        mb: int,
        tokens: int,
        *,
        comm_round_idx: int = 0,
        comm_round_count: int = 1,
        token_base: int = 0,
    ) -> tuple[int, ...]:
        if not self.is_attention or dp != self.physical_rank // self.attn_tp_size:
            raise ValueError("task source must belong to the specified Attention DP")
        if not 1 <= seq <= MAX_TICKET:
            raise ValueError("publish ticket exhausted or invalid; rebuild the domain")
        if not 0 <= layer < self.layer_id_capacity or not 0 <= mb <= MAX_TICKET:
            raise ValueError("invalid task layer or microbatch")
        if not 0 <= tokens <= self.source_token_capacity:
            raise ValueError("task tokens exceed the complete-microbatch source window")
        if (comm_round_idx, comm_round_count, token_base) != (0, 1, 0):
            raise ValueError("ABI v1 supports only the default communication round")
        return (
            ABI_VERSION,
            self.session_epoch,
            seq,
            seq,
            dp,
            layer,
            mb,
            tokens,
            self.physical_rank,
            0,
            1,
            0,
            tokens,
            1,
            0,
            0,
        )
