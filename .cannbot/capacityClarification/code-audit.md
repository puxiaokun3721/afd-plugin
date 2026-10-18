# Capacity clarification: implementation audit

2026-10-16. Reviewed current working tree, preserving previous dispatch reuse changes. Source baseline: DeepEP-Ascend 3b25377d04b24fc6154698ded78a2bcb2c59afff. No historical QA materials edited.

## Scope and method

Read `.cannbot/环境信息.md`, Ascend C tiling skill and operator-code-review skill, including `02_ascendc_overflow_patterns.md`. Applied host/kernel width, accumulation, representability and per-stage allocation checks to AFD Context, owner constructors, send/recv kernels and native EP entry. No target SDK API behavior changed. Current machine has no confirmed CANN/NPU compiler/runtime.

## Confirmed field mapping

| Concept | Current ABI / implementation | Scope |
|---|---|---|
| maxBS | Context.num_token_capacity, AFDBuffer constructor parameter | rank-local full input allocation extent; positive int host bound |
| B_m | AFDTaskInfo.num_tokens | this transaction logical row extent including masked holes, <= maxBS; may be zero |
| C_m | AFDLane.capacity, lane_specs[][5] | registered rank/physical-slot communication window, 1..32768 |
| C_max | Context.num_max_tokens_per_rank | max of registered lane capacities; legacy native per-call name; sender JIT/workspace bound, not maxBS |
| Q | prepare_recv(q), recv JIT Q | valid expert rows in one computation piece |
| Q_alloc | prepare_recv(q_alloc), recv JIT QA | padded physical output capacity |
| J_DP | AFDTaskInfo.num_comm_rounds | host/framework supplied common DP round count max(1,max_m ceil(B_m/C_m)); sender locally checks lower bound |

Native EP `csrc/buffers/ep.hpp:135-142` gets num_tokens from x rows and bounds it by num_max_tokens_per_rank. AFD's native-equivalent call is one communication window, so retaining this name for C_max is intentional; full input allocation independently uses num_token_capacity.

## Changes

- `csrc/buffers/afd.hpp`: removed production and local receiver fixture DP capacity summations/32768 rejection. Retained unique physical slot, DP member/source/capacity identity across its multiple slots; distinct DP members may differ in C_m. Added production table length <= INT_MAX check to match JIT integer template representation, already present in local fixture.
- Added checked byte accumulation for per-source F input reservation, own-source A return reservation, fixture payload and combined memory budgets. Sizes use uint64 arithmetic and must fit INT64_MAX tensor size representation; this is not a new token or DP limit. Retained explicit actual memory budget and transport reserve checks.
- `layout/afd/context.hpp`: added named kMaxLaneTokens=32768 implementation-family bound; documented full allocation/actual extent/window/common rounds without changing ABI sizes/offsets.
- `impls/afd/dispatch_recv.hpp`: used the named bound for existing per-lane checks only. No data traversal, stage, DMA or protocol algorithm changed.

## What required no algorithm change

- Send kernel checks B_m <= num_token_capacity and J_DP >= ceil(B_m/C_m), loops J_DP times, uses begin=min(B_m,r*C_m), length=min(C_m,B_m-begin). Hence 256k/32k gives 8 local windows; an exhausted/zero-length member still publishes zero-count Inbox/Ready through common J_DP. All-zero DP uses one control round, not J_DP=0. Per-window route workspaces use C_max and retain existing bounded tile processing.
- F input bytes already sum every registered physical source slot using uint64 offsets, independent of DP grouping. A return bytes already sum its own physical slots. Two 32k owners therefore require two source regions per F; no unchanged-total-memory claim.
- Recv validates each lane and each Inbox, and requires same DP/microbatch/round/J_DP for a selected entry. Total/count/physical accumulation is uint64. Active's current-piece counts remain <= Q and padded rows <= Q_alloc. `prepare_recv` retains L*local_experts <= UINT32_MAX for Segment-count representation. Increasing source volume does not increase Q or output tensor rows.
- No AFD ffnScheduler/combineSend/combineRecv implementations exist in csrc/kernels/afd, csrc/buffers/afd.hpp or impls/afd. Active structures are scheduler-output ABI only; final scheduling/ACK/return coverage is still a documented implementation gap, not completed functionality.

## Protocol reasoning and limitations

Alternating bits are scoped to physical channels/slots, not token counts. Increasing another source's C_m changes source-region reservation and work amount, not bit sequence or credit ownership. A member cannot reuse its input until all required F acknowledgements; extra Q pieces delay acknowledgement until references finish. All DP members must use the same J_DP and publish exhausted tails. Sender alone cannot verify other ranks' B_m or J_DP. Framework/scheduler must compute/provide common J_DP and validate full membership; receiver only validates selected entry consistency. Removing a short member early would deadlock the existing protocol and is not allowed.

Pending whole-window route totals must be uint64 in future scheduler code; per-lane route counts remain bounded by C_m*min(K,local_experts), and per-piece Active remains Q-bounded. No pooled 32k temporary array was found in implemented kernels.

## Validation

Read and traced production and fixture construction, send round loop, recv per-lane/piece bounds, Active layout and native per-call capacity. `git -C D:/Code/refs/DeepEP-Ascend diff --check` passed. No C++/Bisheng build, NPU execution, graph replay or real transport/visibility test performed. CPU capacity regression results are recorded by the test task separately and cannot certify device code.
