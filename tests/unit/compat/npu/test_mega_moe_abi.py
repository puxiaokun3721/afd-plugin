# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Public host ABI/capacity contracts. These are not device protocol tests."""

from dataclasses import replace

import pytest

from afd_plugin.compat.npu.mega_moe_abi import (
    MAX_TICKET,
    PREFIX_RESERVED_BYTES,
    MegaMoeStaticConfig,
)


@pytest.fixture
def config() -> MegaMoeStaticConfig:
    return MegaMoeStaticConfig(
        session_epoch=17,
        physical_rank=0,
        attn_dp_size=2,
        attn_tp_size=4,
        ffn_ep_size=8,
        local_expert_count=4,
        hidden_size=1024,
        intermediate_size=512,
        topk=2,
        source_token_capacity=1025,
        recv_route_capacity=1024,
        layer_count=3,
        layer_id_capacity=8,
    )


def test_config_serializes_the_versioned_operator_contract(config):
    assert config.config_values == (
        1,
        1,
        1,
        17,
        0,
        2,
        4,
        8,
        4,
        1024,
        512,
        2,
        1025,
        1024,
        3,
        8,
        0,
        config.layout.required_bytes,
        1,
        1,
        0,
        0,
        0,
        0,
    )
    assert config.topology[5] == (0, 1, 1, 5)
    assert config.topology[11] == (1, -1, 3, 11)


def test_complete_microbatch_window_has_distinct_tiles_and_disjoint_regions(config):
    layout = config.layout
    assert (layout.chunk_tokens, layout.num_chunks, layout.padded_tokens) == (
        1024,
        2,
        2048,
    )
    assert layout.x_base == PREFIX_RESERVED_BYTES
    assert layout.scale_base - layout.x_base == 2048 * 2 * (1024 + 512)
    assert layout.d_base - layout.scale_base == 4 * 2048 * 2
    assert layout.done_base - layout.d_base == 2 * 2048 * 2 * 1024
    assert layout.required_bytes == layout.done_base + 8 * 512 + layout.tail_bytes
    assert (
        layout.ready_offset(1) >= layout.counts_offset(0, 1) + 4 * layout.expert_stride
    )
    assert (
        layout.counts_offset(1, 0) - layout.counts_offset(0, 0) == layout.inbox_stride
    )


def test_receiver_round_capacity_does_not_shrink_source_window(config):
    smaller = replace(config, recv_route_capacity=128)
    assert smaller.layout == config.layout
    assert smaller.max_recv_rounds == 64
    assert config.max_recv_rounds == 8
    assert smaller.ffn_scratch_bytes < config.ffn_scratch_bytes


def test_all_roles_share_window_layout_but_keep_source_identity(config):
    ffn = replace(config, physical_rank=8)
    assert ffn.layout == config.layout
    assert not ffn.is_attention
    with pytest.raises(ValueError, match="Attention DP"):
        ffn.make_task_meta(1, 0, 3, 0, 128)


@pytest.mark.parametrize("tokens", [0, 1, 1024, 1025])
def test_task_descriptor_encodes_zero_and_tail_source_lengths(config, tokens):
    assert config.make_task_meta(9, 0, 7, 1, tokens) == (
        1,
        17,
        9,
        9,
        0,
        7,
        1,
        tokens,
        0,
        0,
        1,
        0,
        tokens,
        1,
        0,
        0,
    )


@pytest.mark.parametrize("seq", [0, -1, MAX_TICKET + 1])
def test_ticket_cannot_wrap_in_a_live_domain(config, seq):
    with pytest.raises(ValueError, match="rebuild the domain"):
        config.make_task_meta(seq, 0, 7, 1, 128)


@pytest.mark.parametrize(
    "round_fields",
    [dict(comm_round_idx=1), dict(comm_round_count=2), dict(token_base=1024)],
)
def test_first_tier_rounds_are_rejected_before_dispatch(config, round_fields):
    with pytest.raises(ValueError, match="default communication round"):
        config.make_task_meta(1, 0, 7, 1, 128, **round_fields)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (dict(hidden_size=512), "hidden_size"),
        (dict(intermediate_size=513), "intermediate_size"),
        (dict(source_token_capacity=MAX_TICKET), "native int32"),
        (dict(ffn_ep_size=128, local_expert_count=128), "1024 global experts"),
        (dict(attn_tp_size=128, local_expert_count=32), "route cache"),
        (dict(recv_route_capacity=0), "recv_route_capacity"),
        (dict(layer_id_capacity=2), "layer count"),
    ],
)
def test_unsupported_shapes_fail_before_allocating_windows(config, changes, message):
    with pytest.raises(ValueError, match=message):
        replace(config, **changes)
