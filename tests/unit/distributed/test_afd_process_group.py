# SPDX-License-Identifier: Apache-2.0
"""Wire protocol announcements precede every async CAM HCCL domain."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.distributed.afd_process_group import (  # noqa: E402
    validate_afd_wire_protocol,
)


def test_mixed_cam_protocols_fail_before_domain_creation():
    store = torch.distributed.HashStore()
    store.set("afd_wire_protocol/rank/1", "afd_async_cam_megamoe_v1")
    with pytest.raises(RuntimeError, match="incompatible CAM wire protocols"):
        validate_afd_wire_protocol(
            store, rank=0, world_size=2, protocol="afd_async_cam"
        )


def test_matching_protocol_is_announced_to_peers():
    store = torch.distributed.HashStore()
    validate_afd_wire_protocol(
        store, rank=0, world_size=1, protocol="afd_async_cam_megamoe_v1"
    )
    assert store.get("afd_wire_protocol/rank/0") == b"afd_async_cam_megamoe_v1"
