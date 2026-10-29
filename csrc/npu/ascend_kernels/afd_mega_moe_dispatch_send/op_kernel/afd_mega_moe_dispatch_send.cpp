// SPDX-License-Identifier: Apache-2.0
#include "op_kernel/protocol.h"
#include "moe_permute_prologue/moe_permute_prologue.hpp"
using namespace AfdMegaMoe;

extern "C" __global__ __aicore__ void afd_mega_moe_dispatch_send(
    GM_ADDR x, GM_ADDR expert_ids, GM_ADDR token_mask, GM_ADDR task_meta,
    GM_ADDR comm_args, GM_ADDR topology, GM_ADDR control, GM_ADDR state,
    GM_ADDR route_index, GM_ADDR source_counts, GM_ADDR send_scratch,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AfdMegaMoe::TilingData);
    GET_TILING_DATA_WITH_STRUCT(AfdMegaMoe::TilingData, tiling, tilingGM);
    const int64_t *config = tiling.config;
    const auto layout = MakeWindowLayout(config);
    const uint32_t core = VectorIndex(), cores = VectorCount();
    const int32_t logical = LogicalRank(config, topology);
    int64_t task[TASK_FIELDS];
    int64_t error = NoError;
    Peers peers;
    {
        ControlIo io;
        io.ReadArray(task_meta, task, TASK_FIELDS);
        error = ValidateTask(task, config, config[CfgRank]);
        if (!ValidateConfig(config, CONFIG_FIELDS) || logical < 0 ||
            logical >= config[CfgDp] * config[CfgTp] || !peers.Valid(config)) { error = InvalidConfig; }
        if (error == NoError && (task[TaskDp] != logical / config[CfgTp] ||
            task[TaskTokens] != tiling.inputTokens)) { error = TaskIdMismatch; }
        if (error == NoError && (!StateSessionValid(io, state, config) ||
            io.Read<int64_t>(state + StateRun * 8) != Idle ||
            io.Read<int64_t>(state + StateFatal * 8) != NoError)) { error = InvalidState; }
        if (error == NoError && (task[TaskSeq] != io.Read<int64_t>(state + StateLastCombined * 8) + 1 ||
            task[TaskTicket] <= io.Read<int64_t>(state + StateTicket * 8))) {
            error = SequenceGap;
        }
        if (core == 0 && !io.ValidTopology(config, topology)) { error = InvalidConfig; }
        if (Aborting(io, control)) { error = Aborted; }
        // Validate disjoint route ranges before any native prologue can index
        // the expert-count table. Inactive rows may contain padding IDs.
        if (error == NoError) {
            error = io.CheckRoutes(expert_ids, token_mask, task[TaskTokens], config[CfgTopK],
                                   config[CfgFfn] * config[CfgExperts], core, cores);
        }
        io.Write<int64_t>(send_scratch + core * ERROR_STRIDE, error);
        AscendC::SyncAll<true>();
        for (uint32_t index = 0; index < cores; ++index) {
            const int64_t other = io.Read<int64_t>(send_scratch + index * ERROR_STRIDE);
            if (other != NoError) { error = other; }
        }
        if (error != NoError) {
            if (core == 0) { SetFailure(io, state, error); }
            return;
        }
        // Counts for source tiles beyond B are zero, so uneven TP source
        // lengths and all-zero sources need no receiver-side special format.
        for (uint64_t tile = core; tile < layout.numChunks; tile += cores) {
            io.Zero(source_counts + tile * layout.expertStride * 4, layout.expertStride * 4);
        }
        AscendC::SyncAll<true>();
    }
    {
        using Prologue = MoePermute::MoePermutePrologue<2, MoePermute::MoePermuteProloguePolicy,
                                                       AscendC::bfloat16_t, int8_t>;
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
        for (uint64_t tokenBase = 0, tile = 0; tokenBase < task[TaskTokens];
             tokenBase += layout.chunkTokens, ++tile) {
            const uint64_t tokens = Min(layout.chunkTokens, task[TaskTokens] - tokenBase);
            typename Prologue::Params params(tokens, config[CfgHidden], config[CfgTopK],
                                              config[CfgFfn] * config[CfgExperts]);
            Prologue prologue(resource, params);
            GlobalTensor<AscendC::bfloat16_t> input;
            GlobalTensor<int32_t> ids, rows, counts;
            GlobalTensor<int8_t> expanded;
            input.SetGlobalBuffer(reinterpret_cast<__gm__ AscendC::bfloat16_t *>(x) + tokenBase * config[CfgHidden]);
            ids.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expert_ids) + tokenBase * config[CfgTopK]);
            rows.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(route_index) + tokenBase * config[CfgTopK]);
            counts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(source_counts) + tile * layout.expertStride);
            expanded.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(peers.Local() + layout.xBase +
                tile * layout.chunkTokens * config[CfgTopK] * (config[CfgHidden] + 512)));
            prologue(input, ids, expanded, rows, counts, send_scratch + CONTROL_SCRATCH_BYTES,
                     token_mask + tokenBase);
            AscendC::SyncAll<true>();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    AscendC::SyncAll<true>();
    {
        ControlIo io;
        for (uint32_t ffn = core; ffn < config[CfgFfn]; ffn += cores) {
            const int32_t rank = PhysicalRank(topology, config[CfgDp] * config[CfgTp] + ffn);
            GM_ADDR inbox = peers.At(rank) + layout.reservedBytes + logical * layout.inboxStride;
            io.Write<int32_t>(inbox, 0);
            io.Copy(inbox + 2 * RECORD_BYTES, source_counts, layout.numChunks * layout.expertStride * 4);
            io.WriteArray(inbox + RECORD_BYTES, task, TASK_FIELDS);
            Publish(io, inbox, task, config[CfgRank], logical % config[CfgTp], Pending, NoError, 0);
        }
        AscendC::SyncAll<true>();
        if (core == 0) {
            io.Write<int64_t>(state + StateDp * 8, task[TaskDp]);
            io.Write<int64_t>(state + StateSeq * 8, task[TaskSeq]);
            io.Write<int64_t>(state + StateTicket * 8, task[TaskTicket]);
            io.Write<int64_t>(state + StateRun * 8, Published);
        }
    }
}
