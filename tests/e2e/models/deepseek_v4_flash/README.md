# DSV4 A3 MegaMoE prefill acceptance

The additional scenario `afd-dsv4-flash-megamoe-prefill-dp2tp4-ep8` uses the
existing 16-device deployment: Attention DP2/TP4, FFN EP8/TP1, eager mode,
8192 scheduler tokens, two token-split microbatches and per-channel W4A8.
It selects `cam_async_megamoe` and fixes receiver capacity Q to 4096.
The existing `afd-dsv4-flash-async-cam-dp2tp4-ep8` scenario retains its full
256-token arithmetic-answer checks.

The MegaMoE case submits the same ten concurrent arithmetic prompts with
`max_tokens=1`, temperature zero and thinking disabled. It requires exactly one
generated token, the same token as a native reference, finite log probabilities,
and absolute log-probability difference at most 0.05. This tolerance is an
initial acceptance threshold, pending A3 qualification. Every response and HTTP
latency is saved; the output prints median/min/max first-token latency. This
includes HTTP and scheduling overhead and is not a GPU kernel timing.

Set the usual `AFD_E2E_BACKEND=npu`, `AFD_E2E_DEVICES`, `AFD_NPU_E2E_MODEL`,
`HCCL_IF_IP` and `HCCL_SOCKET_IFNAME`, plus
`AFD_NPU_MEGAMOE_PREFILL_REFERENCE` pointing to a JSON reference. Then run:

```bash
pytest -q tests/e2e/models/deepseek_v4_flash/test_async_cam_npu.py -k megamoe
```

Capture the reference on a native service using the same checkpoint revision,
per-channel W4A8 weights and SwiGLU clipping semantics. Do not use a baseline
known to ignore the checkpoint's clipping limit as the numerical oracle.
The JSON schema is:

```json
{
  "model_revision": "actual checkpoint revision or manifest hash",
  "quantization": "W4A8-per-channel",
  "cases": [
    {
      "prompt": "Compute 12 + 7. Reply with just the number.",
      "token": "actual native token string",
      "logprob": -0.0123
    }
  ]
}
```

The values above illustrate the schema; they are not reference results. Supply
all ten prompts, with the first operand running from 12 through 21. For each
native `/v1/chat/completions` request use `max_tokens=1`, `temperature=0`,
`logprobs=true`, `top_logprobs=5` and
`chat_template_kwargs={"thinking": false}`. Copy the token and logprob from
`choices[0].logprobs.content[0]`. Record the native deployment and checkpoint
manifest alongside the reference. The case refuses a missing reference and
does not relabel a successful HTTP request as accuracy acceptance.

For performance comparison run the native service and this scenario with the
same prompts, concurrency, checkpoint and warmup policy, retaining both raw
latency artifacts. Operator numerical/throughput tests and protocol-control
tests live under `tests/npu/a3_camasync_megamoe_*.py`; this HTTP test does not
replace their hot-expert, multi-tile or receiver-round coverage. A3 build,
device execution and performance results have not been collected on the
Windows development host.
