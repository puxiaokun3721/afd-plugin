# Asynchronous AFD for Prefill on Ascend NPU: Detailed Design Draft

> Status: Non-normative working draft
>
> Companion RFC: [RFC_ASYNC_AFD_ON_ASCEND_NPU_codex.md](./RFC_ASYNC_AFD_ON_ASCEND_NPU_codex.md)
>
> Baseline: vLLM 0.26.0, current `afd-plugin` main, and the current
> `CAMAsyncAFDConnector` implementation

## 1. Purpose

This document separates implementation-level design discussion from the RFC
issue body. It records the current architecture, the intended protocol
contract, known gaps, and planned convergence work. It is not yet an approved
specification and must be updated when the latest operator bindings and HCCL
communicator integration become available.

## 2. Problem statement

An Attention DP replica processes a workload whose cost depends on request count,
prompt length, padding, and attention implementation. Different groups therefore
finish the same model layer at different times. A synchronous MoE dispatch wave
turns that natural variance into a global barrier:

```text
time -------------------------------------------------------------->

DP0 Attention ---- ready ----------- wait ---------- FFN ----------
DP1 Attention ----------- ready ---- wait ---------- FFN ----------
DP2 Attention ------------------ ready ------------- FFN ----------
                                  ^
                                  all groups align here
```

The async prefill design changes the scheduling unit from a global wave to one
complete Attention DP replica:

```text
time -------------------------------------------------------------->

DP0 Attention ---- ready -> FFN(DP0) -> next layer ----------------
DP1 Attention ----------- ready -> FFN(DP1) -> next layer ---------
DP2 Attention ------------------ ready -> FFN(DP2) -> next layer --
```

This removes cross-DP waiting. It does not make a partial participant set safe:
all Attention ranks whose dispatch shards form the selected replica's payload
must publish a consistent work item.

## 3. Terminology and identities

- **Attention rank**: a process/device executing the Attention-side model.
- **FFN rank**: a process/device owning routed experts and executing FFN/MoE.
- **Attention DP replica**: one independently progressing request batch.
- **Attention dispatch participants**: the Attention ranks whose dispatch shards
  collectively form one complete payload for an Attention DP replica. The
  participant set is derived from the replica's tensor, context, sequence, or
  combined parallel layout.
- **Work item**: one routed MoE layer invocation for one Attention DP replica.
- **Internal chunk**: one contiguous local expert range used inside the
  connector/operator implementation when a routed work item cannot fit in one
  internal buffer. It is not part of the model runner contract.
- **Dispatch window**: peer-accessible storage holding activations, expert
  counts, address maps, and readiness state.
- **Combine window**: peer-accessible storage holding FFN results and completion
  state on the Attention side.

The current implementation allocates independent communication storage for
each Attention DP replica. Multiple execution stages of the same DP replica
reuse that replica's storage. The connector/operator implementation is
responsible for serializing that reuse so a later stage cannot overwrite
unfinished work. Execution stage is framework-side scheduling state in the
current implementation; it is not carried in operator BatchInfo and is not
part of the stable work-item identity. This allocation strategy is an
implementation detail, not part of the public semantic contract; a future
implementation may instead allocate per-stage storage without changing the
work-item behavior.

A stable work-item identity must be sufficient to reject stale or mismatched
data. The logical identity should include:

```text
(source_attention_dp, layer)
```

The current operator metadata also carries an internal expert range. It does
not carry an explicit execution-stage or generation identifier; framework-side
ordering currently provides that association. Whether future concurrency
requires a stronger identity remains an open question.

## 4. Component responsibilities

### 4.1 Attention runtime

- owns requests, scheduler execution, KV cache, and Attention computation;
- computes routed top-k IDs and weights;
- converts the model's parallel token layout into one disjoint CAM dispatch
  shard per participating Attention rank;
- publishes dispatch work and queues the original routing payload needed by
  combine;
- waits only for the matching work item's final FFN completion;
- restores the model token layout after combine;
- in the target design, computes the shared expert branch locally and adds it to
  the routed result.

### 4.2 `CAMAsyncAFDConnector`

- creates and owns the cross-role process group and communication resources;
- translates framework tensors and topology into operator calls;
- carries transfer state between dispatch receive and combine send;
- validates token layout, dtype, shape, topology, and capacity preconditions;
- exposes semantic work items to the model runners;
- owns timeout, shutdown, and protocol error translation at the framework
  boundary.

### 4.3 FFN runtime

- has no request scheduler and no separate DP metadata control plane;
- blocks on a ready connector work item;
- selects the layer using in-band metadata;
- executes routed expert compute for the returned local expert counts;
- observes one complete semantic work item rather than operator-internal
  capacity handling.

### 4.4 Async operator primitives

- implement peer-accessible memory movement and device-side synchronization;
- publish data before readiness flags with an explicit memory-ordering contract;
- transform token-major inputs into expert-major FFN inputs and reverse the
  transformation during combine;
- close internal capacity handling and empty-rank completion without exposing
  them to the model runner;
- expose capacity and protocol errors instead of waiting indefinitely;
- do not own model-level shared expert semantics in the target design.

## 5. Current data path

### 5.1 Dispatch send

For each Attention token, `async_dispatch_send` currently:

1. reads the routed top-k expert IDs;
2. maps each routed expert to its FFN rank and local expert slot;
3. sends the activation at most once to each destination FFN rank, even when
   several selected experts are on the same rank;
4. records one address-map entry for every expert occurrence so dispatch recv
   can expand the deduplicated transfer;
5. publishes per-expert counts, source/layer metadata, and a final readiness
   value after all payload data is visible.

The current implementation also assigns every token to a synthetic shared
expert slot. That branch is removed by the target protocol.

### 5.2 Dispatch receive

Each FFN rank scans Attention DP replicas from a rotating starting point. A
replica is eligible only after all of its dispatch participants are ready. The
receiver then:

1. aggregates expert counts across the selected participant set;
2. selects a contiguous range of unconsumed routed experts whose total output
   fits `maxTokenNum`;
3. follows the address maps to materialize expert-major activations;
4. returns routed counts, layer/source metadata, and the inclusive expert range;
5. marks the consumed experts in the dispatch window;
6. clears the dispatch readiness state only after the final expert range.

The scheduling policy is currently scanning/round-robin-like rather than a
formally specified fairness algorithm.

### 5.3 Combine send

The FFN side uses dispatch-receive metadata to map each processed expert range
back to the originating Attention ranks. It writes routed outputs into the
Attention combine window. Completion is published only when the final local
expert range has been sent.

### 5.4 Combine receive

The Attention side waits for final completion from every FFN rank, then uses
the original routed expert IDs and weights to reconstruct each token result.
The current implementation also reads and adds the shared expert result. The
target implementation returns only the weighted routed result; framework code
adds the Attention-side shared branch.

## 6. State machine

The logical lifecycle for one source work item is:

```text
FREE
  |
  | Attention writes payload + metadata
  v
DISPATCH_READY
  |
  | connector/operator materializes the complete routed work item
  v
FFN_COMPUTING
  |
  | every FFN rank publishes completion
  v
COMBINE_READY
  |
  | Attention reads result and clears completion
  v
FREE
```

Required properties:

- only the producer changes `FREE -> DISPATCH_READY`;
- readiness is written after payload visibility;
- only the consumer marks dispatch progress;
- any internal partial progress is resolved before semantic completion is
  exposed to the model runner;
- clearing occurs after the corresponding data has been consumed;
- reuse cannot make an old nonzero flag look like completion for a new work
  item.

## 7. Operator-internal capacity handling

### 7.1 Capacity invariant

The current operator may select whole-expert ranges to fit its internal output
capacity. This mechanism is private to the connector/operator boundary and is
not part of the model runner work-item contract. Because one expert is not
split across ranges, the following remains an operator correctness
precondition:

```text
maxTokenNum >= max_e count(topk_ids == e)
```

where the count covers the full set of dispatch participants for one replica
and one dispatch.
If top-k IDs are unique per token, a sufficient static bound is the maximum
global token count of that Attention DP replica. If duplicate IDs are allowed,
the occurrence bound must include them.

### 7.2 Encapsulation boundary

The model runner must not loop over expert ranges or interpret
`start_expert` / `end_expert`. If those fields are used by the low-level
operators, the connector/operator implementation must close all range progress
before exposing completion for the semantic work item. Different FFN ranks may
use different internal ranges without changing the Attention-side contract.

### 7.3 Invalid capacity

If one expert exceeds `maxTokenNum`, the current source can derive an invalid
expert range. The target behavior is a deterministic protocol error containing
the offending expert count and capacity. A device-visible error field or a
host-checkable status output is preferable to an unbounded device wait.

## 8. Operator-internal empty-rank behavior

Expert routing can leave an FFN rank with no routed tokens. Removing shared
experts makes this more common. The operator implementation must still:

1. receive and validate the work-item metadata;
2. skip expert compute without manufacturing a fake model token;
3. execute or emulate final combine publication;
4. allow Attention to distinguish successful zero work from a missing rank.

This behavior is an operator correctness condition. It must not require the
model runner to manufacture a fake model token or implement a separate
zero-token control path.

## 9. Removing shared experts from the protocol

The current local expert indexing is effectively:

```text
[shared slot 0, routed slots 1..N]
```

The operator source release uses:

```text
[routed slots 0..N-1]
```

This changes more than tensor schemas:

- chunk discovery cannot use `start_expert == 0` as a not-found sentinel;
- the expert scan must begin at routed expert zero;
- final-chunk checks become `end_expert == expert_per_rank - 1`;
- dispatch/combine windows no longer allocate shared payload regions;
- dispatch recv no longer produces shared activations/scales/counts;
- combine send no longer accepts a shared result;
- combine recv no longer searches shared placement or adds a shared result;
- empty routed ranks must still publish completion;
- model code owns overlap and addition of the shared branch.

A schema/version transition is required if old binary packages and new
source-built operators may coexist during migration.

## 10. HCCL communicator integration boundary

Current `afd-plugin` creates an HCCL process group and passes its communicator
name to the binary binding. The available operator source reads a communication
argument structure containing peer-accessible memory addresses and currently
carries XCCL/LCCL naming and initialization assumptions. The latest source will
replace or adapt that communicator-management layer.

The stable operator requirement should be expressed as:

- rank and world topology are explicit;
- each participant can resolve peer-accessible dispatch/combine windows;
- publication and visibility ordering is defined;
- lifetime is owned by connector initialization and close;
- communicator creation is not hidden in per-layer operator calls.

`CAMAsyncAFDConnector` owns HCCL communicator initialization,
peer-accessible communication-buffer setup, lifetime, and teardown. The
operators consume connector-provided communicator handles or peer-window
descriptors without owning those resources. The adapter used to resolve those
resources remains an implementation choice below the connector's semantic
protocol.

## 11. Backpressure, fairness, and failure

These areas are incomplete in the current implementation and must be resolved
before a production claim:

### Backpressure

- document the number of concurrent windows per source Attention DP replica;
- serialize same-replica stage reuse so Attention cannot overwrite an outstanding
  work item;
- allow future per-stage window allocation without changing the semantic
  protocol;
- define whether a full window blocks, queues, or returns retryable status;
- include any operator-internal partial progress in occupancy accounting.

### Fairness

- define whether the FFN selector is strict round robin, oldest-ready, or
  another bounded-starvation policy;
- specify how an incomplete TP group affects scanning;
- expose queueing/wait metrics per source Attention DP replica.

### Failure

- replace indefinite polling with timeout/error propagation where supported;
- distinguish peer failure, invalid metadata, capacity violation, and shutdown;
- make `close()` safe when another rank is waiting;
- clear or invalidate stale windows during restart;
- document whether partial-work replay is supported or the request fails.

## 12. Runtime and operator interfaces

The framework-facing interfaces should carry semantic data rather than current
magic constants. Candidate fields include:

- source Attention DP replica, dispatch participant identity, and parallel
  topology;
- model layer identity;
- routed expert count and experts per FFN rank;
- top-k and hidden size;
- dtype and dynamic quantization mode;
- protocol/schema version;
- HCCL communicator handle or resolved peer-window descriptor.

Low-level operator interfaces may additionally carry internal capacity and
expert-range fields. The connector must not expose those fields as model runner
control flow.

Hard constraints remain valid only when justified by the kernel and validated
at startup. Unsupported dtypes, top-k values, capacities, or layouts must fail
before entering a device-side wait.

## 13. Validation plan

### Operator correctness

- reference token routing and weighted combine golden implementation;
- BF16 and FP16, with dynamic quantization modes supported by the target;
- duplicate destination ranks within one token's top-k;
- balanced and maximally skewed expert routing;
- zero-token local rank;
- multiple expert-range chunks;
- exact-boundary `maxTokenNum` and one-expert overflow rejection;
- multiple dispatch-participant layouts, including tensor, context, sequence,
  and combined parallelism, across multiple Attention DP replicas;
- repeated window reuse across layers and stages.

### Framework correctness

- ready DP replicas complete without waiting for unrelated slow replicas;
- delayed DP replicas are not starved;
- original token layout and top-k weights are restored exactly;
- Attention-side shared computation matches the baseline model;
- shutdown and worker failure do not hang indefinitely;
- non-AFD and synchronous connector paths remain isolated.

### Performance

Compare colocated/synchronous/asynchronous configurations using the same:

- model and quantization;
- prompt-length and arrival distribution;
- hardware and Attention:FFN resource budget;
- scheduler limits and accuracy settings;
- TTFT/throughput SLO.

Report at least:

- request and token throughput;
- median and tail TTFT;
- Attention and FFN active/idle time;
- synchronization wait and exposed dispatch/combine time;
- FFN batch size and arithmetic intensity proxy;
- NPU utilization and memory/communication bandwidth;
- per-DP ready-to-consume queueing delay.

## 14. Rollout

The rollout follows the companion RFC's staged delivery:

1. finalize the semantic work-item, routed-expert-only protocol, component ownership,
   and source-integration boundaries;
2. release the four routed-expert-only operators, bindings, integration with
   connector-managed HCCL communication resources, source build, and operator
   tests;
3. integrate the source-built operators into `CAMAsyncAFDConnector`, with
   shared expert computation and result merging on Attention;
4. qualify correctness, reliability, compatibility, and performance before
   completing the migration away from bundled binary artifacts.

Operator-internal capacity handling and empty-rank completion are validated
within the source-release phase without becoming model runner control flow.

Decode async remains a separate follow-up design after the prefill protocol and
validation criteria are stable.

## 15. Open questions

- Does future concurrent multi-stage execution require an explicit stage or
  generation identifier, or is framework-side ordering sufficient?
- Which timeout and error status can be implemented without adding a new host
  synchronization point?
- What fairness policy best preserves SLOs under highly skewed Attention DP
  progress?
- Should the source-built Torch namespace remain `umdk_cam_op_lib` during a
  compatibility period or move directly to an AFD-owned namespace?
- Which operator build artifacts belong in the Python wheel versus an Ascend
  vendor package?
- Which current engine/forward context patches can be replaced by upstream
  extension points?
- What public model/SoC/topology combination is the first performance support
  target after reproduction?
