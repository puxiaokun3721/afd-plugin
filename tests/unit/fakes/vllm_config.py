# SPDX-License-Identifier: Apache-2.0
"""Small config builders for constructor/configuration module seams."""

from types import SimpleNamespace


def megamoe_config(dtype, *, role="attention", **extra):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=dtype,
            hf_config=SimpleNamespace(
                hidden_size=1024,
                moe_intermediate_size=512,
                num_hidden_layers=2,
                n_routed_experts=16,
                num_experts_per_tok=2,
            ),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        additional_config={
            "afd": {
                "connector": "cam_async_megamoe",
                "role": role,
                "async_dp": True,
                "num_attention_ranks": 4,
                "num_ffn_ranks": 2,
                "compute_gate_on_attention": True,
                "connector_extra_config": {
                    "dynamicQuant": 1,
                    "attn_ranks_per_dp": 2,
                    **extra,
                },
            }
        },
    )
