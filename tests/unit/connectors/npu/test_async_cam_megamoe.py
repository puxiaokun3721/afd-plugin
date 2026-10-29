# SPDX-License-Identifier: Apache-2.0
"""Host configuration/admission seams; no simulated device qualification."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm")

from afd_plugin.config import parse_afd_config  # noqa: E402
from afd_plugin.connectors.npu.async_cam_megamoe import (  # noqa: E402
    AsyncCamMegaMoeConnector,
    MegaMoeExtraInfo,
)
from tests.unit.fakes.connector import MegaMoeRuntimeProbe  # noqa: E402
from tests.unit.fakes.vllm_config import megamoe_config  # noqa: E402


@pytest.mark.parametrize("capacity", [1, 8, 257, 1024, 8192])
def test_receive_round_capacity_does_not_shrink_source_window(capacity):
    source = megamoe_config(torch.bfloat16, recv_route_capacity=capacity)
    connector = AsyncCamMegaMoeConnector(0, 0, source, parse_afd_config(source), 0)
    assert connector.static_config.source_token_capacity == 8
    assert connector.static_config.recv_route_capacity == capacity


def test_extra_config_round_trips_explicit_capacities():
    values = {
        "dynamicQuant": 1,
        "attn_ranks_per_dp": 4,
        "source_token_capacity": 2048,
        "recv_route_capacity": 512,
    }
    parsed = MegaMoeExtraInfo.from_mapping(values)
    assert MegaMoeExtraInfo.from_mapping(parsed.to_mapping()) == parsed


def test_extra_config_rejects_unknown_protocol_option():
    with pytest.raises(ValueError, match="unknown"):
        MegaMoeExtraInfo.from_mapping({"comm_round_count": 2})


def test_decode_admission_fails_before_communication_initialization():
    source = megamoe_config(torch.bfloat16)
    connector = AsyncCamMegaMoeConnector(0, 0, source, parse_afd_config(source), 0)
    with pytest.raises(RuntimeError, match="pure prefill"):
        connector.admit_prefill(False)
    assert not connector.is_initialized


def test_fused_connector_rejects_legacy_receive_payload_interface():
    source = megamoe_config(torch.bfloat16, role="ffn")
    connector = AsyncCamMegaMoeConnector(0, 0, source, parse_afd_config(source), 0)
    with pytest.raises(RuntimeError, match="fused Scheduler/FFN"):
        connector.recv_attn_output()


def test_completed_model_step_rejects_device_fatal_and_keeps_error_sticky():
    source = megamoe_config(torch.bfloat16)
    connector = AsyncCamMegaMoeConnector(0, 0, source, parse_afd_config(source), 0)
    runtime = MegaMoeRuntimeProbe("dispatch failed before Ready")
    connector.context = runtime
    with pytest.raises(RuntimeError, match="dispatch failed before Ready"):
        connector.complete_attention_step()
    runtime.error = None
    with pytest.raises(RuntimeError, match="domain failed"):
        connector.complete_attention_step()


def test_prefill_admission_cannot_clear_a_peer_failure():
    source = megamoe_config(torch.bfloat16)
    connector = AsyncCamMegaMoeConnector(0, 0, source, parse_afd_config(source), 0)
    connector.abort_domain("peer rejected descriptor")
    with pytest.raises(RuntimeError, match="peer rejected descriptor"):
        connector.admit_prefill(True)
