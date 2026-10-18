# combineSend implementation record

Pinned native reference: DeepEP-Ascend 3b25377d04b24fc6154698ded78a2bcb2c59afff.

Implementation: `deep_ep/include/deep_ep/impls/afd/combine_send.hpp`.

## Native reuse and necessary differences
- Keeps original combine x-direct branch: each BF16 route references registered x GM directly. No Q*H send mirror, no weights, no local-rank HiddenReducer; A performs final weighted combine.
- Uses existing native HcommJetty/HcommPeer SQ encoder, wrapped SQ copy, CQ completion, doorbell and drain through the AFD sender's shared put_at_offset/flush_writes. Bounded 32-request batches; checks original SQ/CQ assumptions. Does not introduce a second WQE head-accounting mechanism.
- Two metadata stages prefetch the next tile while current tile is consumed. Compact 4-int route metadata replaces upstream K+2 rank-partial representation; native SIMT load_meta_issue_wqe cannot be copied unchanged because destination identity and completion differ.
- Parallel preflight validates row metadata, member ownership, addresses and counts before any remote write. A UB lane-to-entry/member map avoids per-row Active table scans. Requires dispatchRecv-generated metadata to remain immutable through GMM and send; no full duplicate route/hash cache is added.
- Reuses current recv_load/store/read/join helpers and native vector fill_ubuf. Error paths converge all local AIVs and mark ABORTED; domain-level remote cancellation remains deployment integration responsibility.

## Protocol
NONE waits for all selected entry member credits. HELD continues without reacquisition. All AIV payload queues drain and join before Completion entry/member/delta snapshots. Final entries publish Done to every member, then Consumed to every member; zero-route entries participate. Completion READY is written last and continuation retains HELD. Post-ACK code only accesses immutable Active/Completion/context, never Inbox.

Workspace is recv_workspace_bytes(NC), UB is sizeof(CombineSendUB<NA,L,T>), which includes 8*L bytes of lane maps in addition to peer cache and two metadata stages. Host must account this explicit L term; no Q-sized map.

## Skill usage and validation boundary
Read local ascendc-operator-code-gen SKILL.md and mandatory GUIDE.md, data-copy/sync references. Its standalone op_host/template packaging is intentionally replaced by the user's explicitly required existing DeepEP JIT organization; low-level asc_* API use follows checked repository source rather than treating generic TPipe skill examples as A5 API proof.

CPU `python -m unittest discover -s D:/Code/refs/DeepEP-Ascend/tests/afd -p test_combine_send_cpu.py -v`: 5 tests pass. These include the actual scheduler ARMED / send READY source-contract and 32B ABI check, plus independent protocol/address oracles (greater-than-4GiB offsets, padding, invalid routes, held continuation, zero-route, delayed drain), not C++ execution or precision evidence. CANN compilation, URMA x registration, cross-transport payload-before-Done visibility, timeout domain abort, A5 multi-card correctness and performance remain unverified locally.
