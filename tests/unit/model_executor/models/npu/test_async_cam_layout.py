# SPDX-License-Identifier: Apache-2.0
"""Masks describe actual stage tokens in global TP coordinates."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.model_executor.models.npu.async_cam_layout import (  # noqa: E402
    build_cam_token_mask,
)


@pytest.mark.parametrize(
    "rows,start,actual,expected",
    [
        (4, 0, 7, [True, True, True, True]),
        (4, 4, 7, [True, True, True, False]),
        (4, 4, 1, [False, False, False, False]),
        (1, 0, 1, [True]),
        (0, 0, 0, []),
    ],
)
def test_mask_preserves_real_rows_and_excludes_tp_padding(
    rows, start, actual, expected
):
    mask = build_cam_token_mask(rows, start, actual, torch.device("cpu"))
    assert mask.tolist() == expected
    assert mask.dtype == torch.bool


def test_legacy_payload_without_real_token_metadata_has_no_mask():
    assert build_cam_token_mask(4, 0, None, torch.device("cpu")) is None
