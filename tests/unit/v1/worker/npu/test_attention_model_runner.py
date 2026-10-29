# SPDX-License-Identifier: Apache-2.0
"""MegaMoE device errors must not become successful model outputs."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from afd_plugin.config import parse_afd_config  # noqa: E402
from afd_plugin.connectors.npu.async_cam_megamoe import (  # noqa: E402
    AsyncCamMegaMoeConnector,
)
from afd_plugin.v1.worker.npu import attention_model_runner  # noqa: E402
from tests.unit.fakes.connector import MegaMoeRuntimeProbe  # noqa: E402
from tests.unit.fakes.vllm_config import megamoe_config  # noqa: E402


@pytest.fixture
def runner(monkeypatch):
    source = megamoe_config(torch.bfloat16)
    connector = AsyncCamMegaMoeConnector(0, 0, source, parse_afd_config(source), 0)
    result = object.__new__(attention_model_runner.AFDNPUAttentionModelRunner)
    result.connector, result.prof = connector, None
    result._afd_is_graph_replaying, result._afd_live_execution = False, False
    monkeypatch.setattr(
        attention_model_runner, "step_afd_npu_profiler", lambda prof: None
    )
    monkeypatch.setattr(
        attention_model_runner.NPUModelRunner,
        "execute_model",
        lambda self, scheduler_output, intermediate_tensors: "model-output",
    )
    return result


def test_model_step_cannot_return_output_after_device_fatal(runner):
    runner.connector.context = MegaMoeRuntimeProbe("device fatal after CombineRecv")
    with pytest.raises(RuntimeError, match="device fatal after CombineRecv"):
        runner.execute_model(None)


def test_model_step_returns_output_after_healthy_completion(runner):
    runner.connector.context = MegaMoeRuntimeProbe()
    assert runner.execute_model(None) == "model-output"
