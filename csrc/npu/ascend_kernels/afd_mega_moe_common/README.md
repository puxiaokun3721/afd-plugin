# A3 CAMAsync MegaMoE kernel sources

This directory belongs to the four `afd_mega_moe_*` custom operators. The shared
host/device ABI is `include/afd_mega_moe/abi.h`; the full design and acceptance
matrix are in `docs/design/a3_camasync_megamoe/` at the repository root.

## Pinned native code

- ops-transformer: `8d46ac50deabb544f59f56f5b82a995159907c03`, CANN Open Software
  License 2.0, copied under `native/` with original relative include paths.
- CATLASS: `769cd40a8716b28650b6bebb08db4834eea4462f`, copied as the required header
  dependency closure under `vendor/catlass/include/`; its license is retained.
- `source_manifest.json` records SHA-256 hashes of the original Git blobs and
  local files, upstream paths, revisions and whether a file is adapted. Source
  bytes are read with binary Git output; checkout/archive line-ending filters
  are not used. Native sources use LF. CATLASS has mixed line endings in the
  pinned Git tree, so `vendor/** -text` preserves its exact bytes on every host.
  SDK headers are not vendored.

`op_kernel/native_ffn.h` specializes the pinned native arch22 wrapper for A3
W4A8, NZ weights, BF16 output, the original 128x256x1024 / 128x256x256 tiling,
1:2 AIC/AIV mix, per-channel FIXPIPE scales and original epilogue pipelines.
This is not a GEMM reimplementation.

The adapted native files are delimited with `// ### PATCH START/END` comments:

| File | Required AFD difference |
| --- | --- |
| `mega_moe_kernel_a3.hpp` | Independent F ordinal and T source count; receiver entry; global-E count staging; selected-layer pointers; source-local hot-expert round offset; AFD private-buffer alignment |
| `utils/hccl_shmem.hpp` | Map selected DP source indices to physical A ranks using the stable source snapshot; direct MC2 HCCL context |
| `utils/block_epilogue_w4a8post_pertoken_swiglu.hpp` | Compensation list layer offset and bounded activation temporary UB view |
| `utils/block_epilogue_w4a8post_pertoken_v2.hpp` | Compensation list layer offset |
| `utils/weight_l2_prefetch.hpp` | Apply the same layer offset to cold weight prefetch |

Native permute, quantization, MSD, block MMAD, GMM scheduling, activation math,
remote-store epilogue math, and unpermute/top-k accumulation remain the pinned
implementations. A local CR tiling helper only fixes the native zero divisor
when fewer tokens than AIV cores are present.

## Kernel boundaries

1. DispatchSend validates the task and routes, runs the native prologue for
   each <=1024-token source tile, and publishes the complete source task to
   every F. Tile-local route indices and original top-k-expanded INT8 rows are
   retained. The mask is fused through the native prologue.
2. Scheduler runs on one AIV. It scans DPs fairly and claims a task only after
   all T Ready records match. Every F chooses independently; no F-F collective
   appears in the new execution path.
3. FFN validates every source/count row before releasing any AIC into compute.
   It stages the current tile's full global-E count vectors, computes immutable
   source-local prefixes, then uses native cumsum/cache/GMM/MSD/epilogue methods.
   `native_receiver.inc` replaces only the synchronous communication front and
   back ends. Logical Q determines second-level receive rounds; allocating the
   private scale buffers at 512-byte boundaries also permits Q=8 or Q=257
   without changing Q or source offsets. There is no source publication round.
4. CombineRecv waits for every F Done on AIV0, releases all AIVs together, and
   runs native unpermute. Only then does it set A state to IDLE and permit the
   next microbatch to reuse the physical communication window.

F's native scratch allocations use actual task tile count, bounded by the ABI
capacity formula. Count staging and immutable prefixes are placed after the
capacity-sized native allocation. Activation UB is limited to its native
minimum temporary requirement so the high-address route cache remains intact;
`ValidateConfig` checks that the route cache fits the remaining 16032 bytes.

## Protocol ordering and errors

Ready and Done have a separate 32-bit commit word at byte 0 and a 64-bit body
starting at byte 128 of a 512-byte record. A writer invalidates the commit,
drains body/payload writes, then publishes a strictly increasing positive
ticket. Readers use MTE for changing control data, read the commit twice, and
ignore records from another session epoch/ABI. Session startup must agree on a
fresh unique epoch. No post-publication clearing of peer windows is performed.

FFN has one mixed-core barrier after uniform validation and one after native
compute drains. It never lets AIV abort while AIC is waiting on native flags.
Zero-route work skips compute and still returns Done. A preflight failure or
an ABORT observed after compute produces ERROR Done and local fatal state;
invalid communication contexts only produce local errors. STOP is handled by
Scheduler before claiming work. CR likewise makes the all-vector success/error
choice at a single barrier, so independent control-stream updates cannot split
its participants.

There is no device deadline or watchdog in these kernels. The framework's
management path may publish sticky ABORT; an already running native compute
body drains before honoring it. Hardware execution stalls require the runtime
or external supervisor's device failure handling.

## Validation boundary

Header dependency closure, source hashes, interface order, tiling integration,
workspace formulas and patch scope can be checked on the development host.
CANN/Bisheng compilation, real HCCL memory ordering, NPU precision, graph replay,
resource coexistence and performance still require the A3 test machine. Native
code reuse is not itself a hardware validation or a performance measurement.
