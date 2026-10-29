# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A3 binding contracts and actual torch_npu raw-NZ storage validation.

Run on 910C after building the custom operators:
  AFD_RUN_ASCEND_OP_RUNTIME=1 SOC_VERSION=910c python -m pytest -q \
    tests/npu/a3_camasync_megamoe_binding.py

Meta checks validate registration and argument contracts only. Raw-NZ checks
require actual NPU allocation/format conversion and do not emulate device memory.
Numerical and distributed execution belong to a3_camasync_megamoe_precision.py.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

from afd_plugin.compat.npu.mega_moe_abi import (  # noqa: E402
    COMPLETION_FIELDS,
    CONTROL_ELEMENTS,
    STATE_HEADER_FIELDS,
    TASK_FIELDS,
    WORK_FIELDS,
    MegaMoeStaticConfig,
)
from afd_plugin.compat.npu.ops import ensure_mega_moe_ops_available  # noqa: E402

pytestmark = pytest.mark.npu


@pytest.fixture(scope="module", autouse=True)
def configured_npu() -> Iterator[None]:
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        pytest.skip(
            "requires AFD_RUN_ASCEND_OP_RUNTIME=1 and a compiled 910C extension"
        )
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    old_format = torch.npu.config.allow_internal_format
    torch.npu.config.allow_internal_format = True
    ensure_mega_moe_ops_available()
    yield
    torch.npu.synchronize()
    torch.npu.config.allow_internal_format = old_format


@pytest.fixture
def binding_config() -> MegaMoeStaticConfig:
    return MegaMoeStaticConfig(
        session_epoch=1,
        physical_rank=0,
        attn_dp_size=1,
        attn_tp_size=1,
        ffn_ep_size=1,
        local_expert_count=2,
        hidden_size=1024,
        intermediate_size=512,
        topk=2,
        source_token_capacity=16,
        recv_route_capacity=32,
        layer_count=2,
        layer_id_capacity=8,
    )


@pytest.fixture
def meta_arguments(binding_config: MegaMoeStaticConfig):
    config = binding_config
    layout = config.layout
    tokens = 7

    def tensor(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    comm = tensor((1,), torch.float16)
    topology = tensor((config.world_size, 4), torch.int32)
    control = tensor((CONTROL_ELEMENTS,), torch.int32)
    state = tensor((STATE_HEADER_FIELDS + config.attn_dp_size,), torch.int64)
    task = tensor((TASK_FIELDS,), torch.int64)
    mask = tensor((tokens,), torch.bool)
    route = tensor((layout.padded_tokens * config.topk,), torch.int32)
    work = tensor((WORK_FIELDS,), torch.int64)
    sources = tensor((config.attn_tp_size, TASK_FIELDS), torch.int64)
    completion = tensor((COMPLETION_FIELDS,), torch.int64)
    w13, w2, s13, s2, b13, b2 = [], [], [], [], [], []
    for _ in range(config.layer_count * config.local_expert_count):
        w13.append(
            tensor((config.hidden_size, config.intermediate_size // 4), torch.int32)
        )
        w2.append(
            tensor((config.intermediate_size, config.hidden_size // 8), torch.int32)
        )
        s13.append(tensor((2 * config.intermediate_size,), torch.int64))
        s2.append(tensor((config.hidden_size,), torch.int64))
        b13.append(tensor((2 * config.intermediate_size,), torch.float32))
        b2.append(tensor((config.hidden_size,), torch.float32))
    return {
        "dispatch_send": [
            tensor((tokens, config.hidden_size), torch.bfloat16),
            tensor((tokens, config.topk), torch.int32),
            mask,
            task,
            comm,
            topology,
            control,
            state,
            route,
            tensor((layout.num_chunks, layout.expert_stride), torch.int32),
            tensor((config.send_scratch_bytes,), torch.uint8),
        ],
        "scheduler": [comm, topology, control, state, work, sources],
        "ffn": [
            work,
            sources,
            tensor((config.layer_id_capacity,), torch.int64),
            w13,
            w2,
            s13,
            s2,
            b13,
            b2,
            tensor((config.layer_count, 4), torch.float32),
            comm,
            topology,
            control,
            state,
            tensor((config.ffn_scratch_bytes,), torch.uint8),
            completion,
        ],
        "combine_recv": [
            tensor((tokens, config.topk), torch.float32),
            mask,
            task,
            route,
            comm,
            topology,
            control,
            state,
            tensor((tokens, config.hidden_size), torch.bfloat16),
            completion,
        ],
    }


@pytest.fixture
def operators():
    return {
        "dispatch_send": torch.ops.afd_ascend.afd_mega_moe_dispatch_send,
        "scheduler": torch.ops.afd_ascend.afd_mega_moe_scheduler,
        "ffn": torch.ops.afd_ascend.afd_mega_moe_ffn,
        "combine_recv": torch.ops.afd_ascend.afd_mega_moe_combine_recv,
    }


@pytest.mark.parametrize(
    "operation", ["dispatch_send", "scheduler", "ffn", "combine_recv"]
)
def test_meta_accepts_preallocated_v1_contract(
    binding_config, meta_arguments, operators, operation
):
    assert (
        operators[operation](
            *meta_arguments[operation],
            group_name="meta-contract",
            config=list(binding_config.config_values),
        )
        is None
    )


@pytest.mark.parametrize(
    "operation", ["dispatch_send", "scheduler", "ffn", "combine_recv"]
)
def test_meta_rejects_unknown_wire_version(
    binding_config, meta_arguments, operators, operation
):
    config = list(binding_config.config_values)
    config[0] += 1
    with pytest.raises(RuntimeError, match="invalid A3 MegaMoE v1 config"):
        operators[operation](
            *meta_arguments[operation], group_name="meta", config=config
        )


@pytest.mark.parametrize(
    ("operation", "argument_index", "message"),
    [
        ("dispatch_send", 10, "send_scratch has incorrect shape"),
        ("scheduler", 4, "work_desc has incorrect shape"),
        ("ffn", 14, "ffn_scratch has incorrect shape"),
        ("combine_recv", 9, "completion has incorrect shape"),
    ],
)
def test_meta_rejects_truncated_persistent_output(
    binding_config, meta_arguments, operators, operation, argument_index, message
):
    arguments = meta_arguments[operation]
    arguments[argument_index] = arguments[argument_index][:-1]
    with pytest.raises(RuntimeError, match=message):
        operators[operation](
            *arguments, group_name="meta", config=list(binding_config.config_values)
        )


def test_meta_ffn_rejects_incomplete_layer_major_weight_table(
    binding_config, meta_arguments, operators
):
    arguments = meta_arguments["ffn"]
    arguments[3] = arguments[3][:-1]
    with pytest.raises(RuntimeError, match="all six TensorLists"):
        operators["ffn"](
            *arguments, group_name="meta", config=list(binding_config.config_values)
        )


@pytest.mark.parametrize(
    ("experts", "rows", "columns"), [(2, 1024, 1024), (1, 512, 1024)]
)
def test_raw_nz_experts_preserve_physical_slabs(experts, rows, columns):
    from afd_plugin import _C_ascend

    # Same byte packing and cast order as the .26 W4A8 model loader.
    generator = torch.Generator().manual_seed(17)
    values = torch.randint(
        -8, 8, (experts, rows, columns), dtype=torch.int8, generator=generator
    )
    pairs = values.reshape(-1, 2)
    packed = ((pairs[:, 1] << 4) | (pairs[:, 0] & 15)).reshape(
        experts, rows, columns // 2
    )
    grouped = torch_npu.npu_format_cast(packed.npu(), 29).view(torch.int32).contiguous()
    views = _C_ascend.raw_extract_nz_experts(grouped)
    assert len(views) == experts
    slab_bytes = rows * columns // 2
    for expert, view in enumerate(views):
        assert view.shape == (rows, columns // 8)
        assert view.stride() == (columns // 8, 1)
        assert view.dtype == torch.int32
        assert view.device == grouped.device
        assert view.data_ptr() == grouped.data_ptr() + expert * slab_bytes
        assert view.untyped_storage().data_ptr() == grouped.untyped_storage().data_ptr()


def test_raw_nz_experts_reject_plain_nd_carrier():
    from afd_plugin import _C_ascend

    dense = torch.empty((2, 1024, 128), dtype=torch.int32, device="npu")
    with pytest.raises(
        RuntimeError, match="weight storage must preserve grouped INT4 NZ bytes"
    ):
        _C_ascend.raw_extract_nz_experts(dense)
