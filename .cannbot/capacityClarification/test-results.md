# Capacity clarification validation — 2026-10-16

Scope: tests/afd and implementation usage docs in DeepEP-Ascend only. Historical
QA reports were not overwritten. Previous batching/pipeline fixtures preserved.

## Executed locally

- `python -m unittest discover -s tests/afd -p 'test_*cpu.py' -v` from
  `D:/Code/refs/DeepEP-Ascend`: **33 passed** (27 pre-existing plus six capacity cases).
- AST parse of all `tests/afd/*.py`: **15 files passed**.
- Existing ABI parser was updated to ignore C++ comments after new capacity
  comments made it reject otherwise unchanged struct declarations. This does
  not change expected sizes/offsets or bypass layout verification.

New CPU cases cover same-DP 2x32k/64k, per-rank256k/8rounds, heterogeneous
capacities and early-zero members, mask holes/all-masked logical extents,
TP nonowner zero publication, independent Decode slots vs member count,
all-registered-slot memory, hot-F524288routes/16Q slices and padding,
full-B token indices through existing receiver byte oracle, and single-owner
regression through actual C++ workspace arithmetic. F payload increase for one
additional32k slot at hiddenBF16=128,K=R=8 is exactly17825856bytes; control and
directory growth is separate. No model test proves DMA visibility or liveness.

## Prepared, not executed

`python tests/afd/run_capacity_a5.py --device 0` exercises actual compiled local
receiver host construction: single32k, same-DP2x32k, heterogeneous32k/16k,
Q_alloc fixed at32888, full token capacity262144, and per-lane32769 rejection.
This is host/allocator validation on a real NPU, not a kernel/transport test.

No CANN C++ compilation, NPU kernel, HCCL construction, live multi-round send,
ACK/credit closure, scheduler, combine, graph replay or performance run occurred.
The full-transaction tests are specifications/oracles because downstream
scheduler/combine operations remain unimplemented. Do not fabricate their ACKs
and report an end-to-end protocol pass.
