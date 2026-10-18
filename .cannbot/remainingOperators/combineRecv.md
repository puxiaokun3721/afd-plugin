# combineRecv implementation record

Scope: `deep_ep/include/deep_ep/impls/afd/combine_recv.hpp`, six CPU semantic/protocol tests in `tests/afd/test_combine_recv_cpu.py`, and the per-operator design implementation note. Baseline native commit is 3b25377d04b24fc6154698ded78a2bcb2c59afff. Other operators, host/JIT, and shared layouts are owned by the parent/other workers.

## Reuse and necessary differences

- Native `combine_reduce_epilogue_impl` contiguous token-per-core partition and names (`vec_core_idx`, `num_tokens_per_core`, `token_idx`) retained.
- Native `RingReducer::load_next_token` DMA shape, NORMAL/NOTALLOC cache hints, and input stage lifetime are retained as the basis of a two-stage bounded chunk loader. No whole-K hidden cache or FP32 GM output mirror.
- Every route must have its own FP32 multiplication. Native rank-partial last-slot dedup, count=1 raw-copy and count=2 BF16 shortcut cannot preserve this contract. They are deliberately not reused. FP32 Cast/Muls/Add runs in fixed K order and a single final BF16 CAST_RINT writes output. This matches the approved design, but exact SDK overloads need compilation.
- Metadata validation loads a K-wide expert vector once per token and validates on UB, distributed across cores before outputs. It does not scalar-load hidden values.
- The native collective barrier cannot serve disjoint A/F roles. Existing AFD load_control/store_control/local_join are reused by including dispatch_send.hpp; reverse credit is a UBMEM control write, not an Hcomm payload SQE, so no jetty argument is needed here.
- Full-F Done polling, credit store completion before Ledger EMPTY, and final full-F Consumed wait preserve alternating-bit closure. Per-lane rounds follow actual B and C; exhausted members still execute common zero-length rounds. Errors never forge credit or ACK; runtime must abort the domain on recv_status=ERROR.

## Interface and bounds

Template `<E,N,K,H,HC,NC,Timeout>`; HC is a positive 128 multiple <=H; H is a 128 multiple. Kernel parameter order follows design with byte-typed registered buffer. `sizeof(CombineRecvUB<K,HC>)=14HC+align32(4K)+align32(8K)+256`. Workspace `combine_recv_workspace_bytes(NC)=256+align32(32NC)`, independent from sender scratch. Reverse directory comes from Context.reserved_address, shared AFDReturnDirectory/AFDReturnLane. Per-slot return allocation uses C*K*return_stride, independent of DP aggregate; user output indexing uses full local token_begin.

## Skills and verification boundary

Read Ascend C operator-code-gen, its GUIDE, data-structure/resource/data-copy/vector/sync/constraint references and row template, API-best-practices, plus compile-debug prerequisites. The existing DeepEP direct JIT architecture and native DMA take precedence over unrelated template wrappers. No new generic ascend-kernel project was created. This is an adapted existing-repository implementation, not a claim that the generic skill's full CANN compile/install workflow ran.

Bare LocalTensor constructor is documented in local basic-data-structures-api and also appears in ops-transformer arch35 CPU debug stubs; that is not target-device compilation evidence. High-level BF16 Cast, FP32 in-place Muls/Add, mixed low-level event/high-level vector code, cache visibility, launch resource partition, and hardware graph replay require A5/CANN verification. CANN compile-debug prerequisites are absent in this Windows environment, so no compiler or NPU execution is claimed.

Executed `python -m unittest discover -s D:/Code/refs/DeepEP-Ascend/tests/afd -p test_combine_recv_cpu.py -v`: 6/6 pass. These test arithmetic and protocol oracles only, not C++ execution. Cases: K=1/2/8/32 weighted every route, inactive NaN holes, FP32 accumulation vs BF16 shortcut, zero-weight NaN propagation, delayed Done/final ACK and credit-before-bank-release, heterogeneous actual sizes through eight shared rounds. Native equivalent numerical/performance behavior remains unmeasured.

Replay hardening: task_info is a single explicit uint32-bypass snapshot through recv_read, not scalar cached pointer reads. token_mask is read into the existing status scratch in exact-length tiles of at most32 bytes, independently for parallel validation and reduction. This avoids full-B mask copies and scalar DCache replay hazards. The tail DMA byte-length family (1..31B) is part of the target-SDK/A5 test gate; no read extends past the allocation.
