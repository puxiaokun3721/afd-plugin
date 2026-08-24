# [RFC]: Asynchronous Attention-FFN Disaggregation for Prefill on Ascend NPU

> Status: Draft for discussion
>
> Target baseline: vLLM 0.26.0 and the current `afd-plugin` main branch
>
> Current implementation: `CAMAsyncAFDConnector`
>
> Detailed design: [RFC_ASYNC_AFD_ON_ASCEND_NPU_DESIGN_codex.md](./RFC_ASYNC_AFD_ON_ASCEND_NPU_DESIGN_codex.md)

## Motivation

Attention-FFN Disaggregation (AFD) separates the stateful Attention path from
the stateless FFN/MoE path so that the two roles can use different resource
ratios, parallel strategies, and execution policies. Synchronous and
asynchronous AFD address different inefficiencies.

Synchronous AFD primarily improves FFN efficiency by changing the
Attention-to-FFN resource ratio. More Attention workers can feed fewer FFN
workers, raising the FFN batch size and improving weight reuse. This is
especially useful during decode, where a small FFN batch is commonly
memory-bandwidth bound. Systems such as MegaScale-Infer and FastAFD study this
resource-balancing benefit.

That resource separation does not, by itself, remove synchronization stalls.
In a colocated or synchronous AFD execution, MoE dispatch and combine create
barriers between Attention data-parallel replicas. Online requests have
different prompt lengths and arrive at different times, so Attention DP replicas
do not finish a layer at the same time. A fast replica waits for the slowest
replica even when the FFN side already has enough ready work to execute. The
result is straggler-induced idle time caused by load imbalance rather than
by the speed of a single communication operation.

This RFC proposes an asynchronous AFD execution model for prefill on Ascend
NPU. The readiness unit is one Attention DP replica. A replica becomes ready
only after all participating Attention ranks have published the dispatch shards
required to form its complete payload. The participant set may use tensor,
context, sequence, or combined parallelism. Once the replica is ready, the FFN
side may consume and execute it without waiting for other Attention DP
replicas. Routing metadata travels in band with the data. The FFN runtime
consumes ready work items from the connector instead of waiting for all
Attention DP replicas to advance through each MoE layer in a globally
synchronized execution wave.

Prefill is the initial scope because one ready Attention DP replica normally
provides enough tokens to run FFN efficiently. Decode has a different
trade-off: consuming one small DP contribution immediately may reduce waiting
but leave FFN memory-bandwidth bound. Dynamic re-batching across multiple ready
DP replicas is a promising direction, but its latency/throughput policy is still
under exploration and is not standardized by this RFC.

### Goals

- Enable FFN workers to consume each complete Attention DP replica payload
  independently, removing global barriers across Attention DP replicas while
  preserving per-request and per-layer correctness.
- Define a maintainable semantic contract between the model runtime,
  `CAMAsyncAFDConnector`, and the asynchronous dispatch/combine operators.
- Make the CAM asynchronous operator sources, Python bindings, build
  integration, and tests available through `afd-plugin`.
- Define reproducible correctness, reliability, compatibility, and performance
  acceptance criteria for the vLLM 0.26.0 plugin baseline.

### Non-goals

- The initial scope is asynchronous AFD for prefill. This RFC does not define
  a decode scheduling or dynamic re-batching policy; decode may be proposed as
  a follow-up extension.
- This RFC does not deprecate or replace synchronous AFD connectors such as
  `CAMP2pAFDConnector`. Synchronous and asynchronous AFD address different
  bottlenecks and may coexist.
- This RFC does not introduce new TP/EP topology rules, expert placement
  policies, or Attention/FFN resource allocation strategies.
- This RFC does not redesign MoE routing or eliminate the underlying workload
  imbalance. It reduces idle time caused by global synchronization barriers
  when such imbalance occurs.
- This RFC defines the semantic behavior of asynchronous AFD, but does not
  standardize the current operator flag layout, peer-accessible memory offsets,
  or fixed implementation constants as a public ABI.

## Proposed change

### Architecture

The proposed execution path has four logical stages per MoE layer:

```text
Attention DP replica                       FFN ranks

Attention + routing
        |
        | async dispatch (data + routing metadata)
        v
  dispatch_send  ----------------------->  dispatch_recv
                                                 |
                                                 | routed FFN/MoE compute
                                                 v
  combine_recv  <-------------------------  combine_send
        |
        v
weighted routed result + Attention-side shared expert result
```

The current implementation uses `CAMAsyncAFDConnector` and four custom
operators:

- `async_dispatch_send`: publish Attention activations and routing metadata;
- `async_dispatch_recv`: select a ready Attention DP replica and materialize
  expert-grouped FFN inputs;
- `async_combine_send`: publish processed routed expert outputs;
- `async_combine_recv`: reconstruct token order and apply top-k weights.

The operators use peer-accessible device memory and in-band metadata. The
connector has no separate DP control plane. The FFN model runner blocks on the
next ready connector work item and obtains the actual layer identity from that
item.

### Asynchronous work-item contract

The RFC defines semantic readiness and completion rather than the current flag
layout:

1. Each participating Attention rank must finish writing its dispatch data and
   routing metadata before publishing its ready signal.
2. An Attention DP replica is ready for FFN consumption only after all of its
   dispatch participants have published their ready signals.
3. FFN ranks may consume any ready Attention DP replica without waiting for all
   replicas. The selection policy must not starve a ready replica indefinitely.
4. A work item identifies at least the source Attention DP replica and model
   layer.
5. Attention may consume a combine result only after every participating FFN
   rank has completed the matching work item.

Exact ready/completion flags, memory-ordering primitives, and buffer layouts
remain operator implementation details.

### Shared-expert ownership

The current operators model one synthetic shared expert per FFN rank, transfer
shared expert inputs and outputs, and add the shared result during combine.
That behavior is not proposed as the long-term contract.

The CAM asynchronous operators released as source use a routed-expert-only
protocol:

- CAM dispatch/combine handles routed experts only;
- routed expert IDs use a zero-based range without a synthetic shared slot;
- Attention computes the shared expert branch locally or overlaps it with the
  routed AFD path;
- the framework adds the shared result after routed combine.

The migration must also revise expert indexing, completion conditions, output
schemas, window sizing, and metadata lengths. It should be tracked as a child
design/implementation issue under this RFC.

### Operator source release

The current plugin installs an Ascend operator `.run` package and an
`umdk_cam_op_lib` wheel. The source release will include the four operators'
host-side code, tiling implementations, Ascend C device kernels, Python/PyTorch
bindings, supported build integration, and integration with connector-managed
HCCL communication resources required by `afd-plugin`.

The source-release plan includes:

1. import the operator source with provenance and license review;
2. add the four Python/PyTorch bindings and FakeTensor (meta) implementations;
3. reconcile operator packaging and the `umdk_cam_op_lib` namespace;
4. integrate the supported Ascend build into the plugin packaging flow;
5. remove bundled binaries only after source-built artifacts pass equivalent
   correctness, installation, and performance tests;
6. publish supported CANN, `torch_npu`, Python, SoC, dtype, model, and topology
   matrices rather than relying on implicit constants.

Open sourcing the operators is an enabling deliverable of the architecture,
not the sole purpose of this RFC.

### Staged delivery

Each phase should be reviewable independently. Child issues and PRs can be
linked under the corresponding phase as implementation proceeds.

#### Phase 0 — Finalize semantic and source boundaries

- agree on Attention DP replica readiness and the work-item identity, including
  source Attention DP replica and model layer;
- confirm the routed-expert-only CAM protocol and Attention-side shared expert
  ownership;
- define the responsibilities of the model runtime, `CAMAsyncAFDConnector`,
  and the four asynchronous operators;
- confirm connector ownership of HCCL communicator initialization,
  peer-accessible communication-buffer setup, lifetime, and teardown;
- settle source provenance, license, package, and namespace requirements.

This phase is complete when the community agrees on the architecture and
source integration boundaries needed to begin reviewable implementation work.

#### Phase 1 — Release the routed-expert-only CAM operator source code

- publish the operators' host-side code, tiling implementations, and Ascend C
  device kernels;
- add Python/PyTorch bindings and FakeTensor (meta) implementations;
- integrate with connector-managed HCCL communication resources;
- document and validate supported operator configurations and
  communication-buffer capacity requirements;
- add reproducible source builds and operator-level correctness tests.

This phase is complete when all four operators build from source and pass their
documented operator-level tests without transferring shared expert data.

#### Phase 2 — Integrate source-built operators into `afd-plugin`

- switch `CAMAsyncAFDConnector` to the source-built operators;
- compute the shared expert branch on Attention and merge it with the routed
  result in the model integration;
- validate asynchronous prefill AFD end to end on the vLLM 0.26.0 plugin
  baseline;
- retain the binary path only where needed for controlled migration or parity
  comparison.

This phase is complete when the source-built path passes end-to-end correctness
without changing the non-AFD or synchronous AFD paths.

#### Phase 3 — Qualify the feature and complete source migration

- validate timeout, shutdown, failure cleanup, uneven DP progress, and repeated
  execution;
- document the supported vLLM, vLLM-Ascend, CANN, `torch_npu`, Python, SoC,
  dtype, model, and topology combinations;
- perform reproducible plugin-native performance characterization;
- remove or deprecate bundled binary artifacts only after source-built parity
  and migration requirements are satisfied.

This phase is complete when the supported configurations have reproducible
correctness, reliability, compatibility, and performance results, and the
binary migration policy is documented.

### Decode follow-up

Asynchronous AFD for decode remains a possible follow-up direction. Its
scheduling, dynamic re-batching, and latency/throughput policies are outside
the scope of this RFC and require a separate design discussion.

### Decisions requested from the community

This RFC asks for feedback on the following decisions:

1. Should barrier removal at the Attention-DP-group granularity be the stable
   execution model for Ascend NPU asynchronous prefill AFD?
2. Is the division of responsibility between the model runtime,
   `CAMAsyncAFDConnector`, the connector-driven FFN execution loop, and the four operator
   primitives maintainable?
3. Should CAM operator sources and bindings live in `afd-plugin`, and what
   package/namespace migration policy should be used?
4. Is a routed-expert-only CAM operator boundary, with shared expert computation owned
   by Attention, acceptable as the long-term protocol?
5. Which parts of the current async-DP engine behavior should become upstream
   extension points rather than remain plugin compatibility patches?

## Plugin boundary

### Upstream-owned

- Request scheduling, request state, and the standard worker lifecycle.
- Native Attention and MoE computation outside the AFD transfer boundary.
- Base NPU model runner behavior unrelated to AFD.
- KV-cache, sampling, and output lifecycle.
- Standard process group, forward context, and DP behavior for non-AFD paths.

### Plugin-owned

- AFD configuration, topology validation, and connector selection.
- `CAMAsyncAFDConnector` and its semantic work-item contract.
- AFD-specific Attention/FFN worker and model runner subclasses or integration
  hooks, including the connector-driven FFN execution path.
- Model-side routed dispatch/combine integration, including Attention-side
  shared expert computation and result merging.
- HCCL communicator initialization, peer-accessible communication-buffer
  setup, lifetime, and teardown for the CAM connector backend.
- CAM operator source, bindings, build integration, and backend-specific tests.
- Supported-configuration matrices, reproducible recipes, and qualification
  artifacts.

### Compatibility helpers

- Narrow adapters for pinned vLLM and vLLM-Ascend process group, runner,
  forward context, and platform APIs.

### Compatibility patches

- The current async-DP engine patch that disables native DP wave coordination
  for AFD Attention execution.
- The current forward context patch that avoids constructing synchronized
  native DP metadata for this asynchronous path.

Every patch must remain version-gated and carry an upstream or removal plan.
This RFC does not propose unrelated changes to vLLM core.

### Explicit class paths

- `afd_plugin.connectors.npu.async_cam.CAMAsyncAFDConnector`
- `afd_plugin.v1.worker.npu.attention_worker.AFDNPUAttentionWorker`
- `afd_plugin.v1.worker.npu.ffn_worker.AFDNPUFFNWorker`
- corresponding NPU model runners and model-side asynchronous CAM forward
  helpers

## Risks and alternatives

### Risks

- **Deadlock and stale state**: the current device-side waits are unbounded;
  rank failure or an invalid state transition can block all progress.
- **Ordering and window reuse**: independent DP progress can overwrite a window
  unless ownership and completion are explicit.
- **Fairness**: repeatedly selecting the first ready DP replica may starve a slow
  or unlucky group.
- **Framework/operator drift**: changing operator metadata without a versioned
  semantic contract can silently break the connector.
- **Compatibility cost**: async execution currently relies on plugin patches to
  vLLM engine and forward context behavior.
- **Maintenance and licensing**: source import must resolve provenance,
  third-party code, and differences between existing license headers.

### Alternatives considered

1. **Keep colocated or synchronous AFD execution.** This is simpler and remains
   a valid fallback, but it does not remove cross-DP straggler barriers.
2. **Tune only the Attention:FFN rank ratio.** This improves FFN efficiency but
   addresses a different bottleneck from asynchronous barrier removal.
3. **Use more fixed microbatches behind a synchronous barrier.** This may hide
   some communication but still couples progress to a global wave and cannot
   consume arbitrary ready DP replicas.
4. **Keep distributing binary operators.** This minimizes near-term build work
   but prevents community debugging, adaptation, review, and contribution.
5. **Wait for a complete decode design.** Prefill already has a distinct and
   useful execution model; coupling it to unresolved decode scheduling would
   delay review without improving the prefill contract.

## Feedback period

Suggested feedback period: two weeks after the RFC is posted. The issue should
remain open as the umbrella tracker, while accepted implementation work moves
to linked child issues and PRs. Performance results may be added when the
vLLM/`afd-plugin` reproduction is complete.

## CC list

TBD before posting. Suggested reviewer groups:

- `afd-plugin` maintainers;
- vLLM-Ascend runtime and MoE owners;
- CAM/Ascend C operator owners;
- NPU distributed communication owners;
- contributors working on Async GPU Connector and ModelRunner V2;
- authors of the plugin-native performance reproduction.

## References

- [MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism](https://arxiv.org/abs/2504.02263)
- [Step-3 is Large yet Affordable](https://arxiv.org/abs/2507.19427)
- [Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony](https://arxiv.org/abs/2505.08944)
- [FastAFD](https://haoailab.com/blogs/fastafd/)
- [NVIDIA Groq 3 LPX heterogeneous AFD](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform)
- [afd-plugin AFD paper index](https://github.com/vllm-project/afd-plugin/issues/32)
- [afd-plugin project roadmap](https://github.com/vllm-project/afd-plugin/issues/155)
- [Async GPU Connector RFC](https://github.com/vllm-project/afd-plugin/issues/233)
