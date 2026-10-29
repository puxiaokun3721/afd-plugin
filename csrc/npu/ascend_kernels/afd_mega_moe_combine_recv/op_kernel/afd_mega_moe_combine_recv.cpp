// SPDX-License-Identifier: Apache-2.0
#include "op_kernel/protocol.h"
#include "unpermute/mc2_mega_moe_moe_token_unpermute.h"
using namespace AfdMegaMoe;
namespace {
// Native tiling, with a nonzero split divisor for B < the AIV count.
__aicore__ inline MoeTokenUnpermuteTilingData UnpermuteTiling(uint32_t tokens, uint32_t hidden,
                                                           uint32_t topk, uint32_t cores)
{
    MoeTokenUnpermuteTilingData tiling{};
    tiling.hidden_size = hidden;
    tiling.top_k = topk;
    tiling.num_out_tokens = tokens * topk;
    tiling.hidden_splited_length = hidden;
    tiling.hidden_splited_num = 1;
    tiling.tokens_core_length = tokens / cores;
    tiling.tokens_core_remain = tokens % cores;
    tiling.tokens_splited_length = Max(1, Min(tiling.tokens_core_length, 600));
    tiling.tokens_splited_num = tiling.tokens_core_length / tiling.tokens_splited_length;
    tiling.tokens_splited_remain = tiling.tokens_core_length % tiling.tokens_splited_length;
    tiling.buffer_num = 4;
    return tiling;
}
}
extern "C" __global__ __aicore__ void afd_mega_moe_combine_recv(
    GM_ADDR topk_weights, GM_ADDR token_mask, GM_ADDR task_meta, GM_ADDR route_index,
    GM_ADDR comm_args, GM_ADDR topology, GM_ADDR control, GM_ADDR state,
    GM_ADDR out, GM_ADDR completion, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AfdMegaMoe::TilingData);
    GET_TILING_DATA_WITH_STRUCT(AfdMegaMoe::TilingData, tiling, tilingGM);
    const int64_t *config = tiling.config;
    const auto layout = MakeWindowLayout(config);
    const uint32_t core = VectorIndex(), cores = VectorCount();
    int64_t task[TASK_FIELDS];
    Peers peers;
    {
        ControlIo io;
        io.ReadArray(task_meta, task, TASK_FIELDS);
        if (core == 0) {
            int64_t error = ValidateTask(task, config, config[CfgRank]);
            const int32_t logical = LogicalRank(config, topology);
            if (!io.ValidTopology(config, topology) || logical < 0 ||
                logical >= config[CfgDp] * config[CfgTp] || !peers.Valid(config)) { error = InvalidConfig; }
            if (error == NoError && (task[TaskDp] != logical / config[CfgTp] ||
                task[TaskTokens] != tiling.inputTokens || !StateSessionValid(io, state, config) ||
                io.Read<int64_t>(state + StateRun * 8) != Published ||
                io.Read<int64_t>(state + StateSeq * 8) != task[TaskSeq] ||
                io.Read<int64_t>(state + StateTicket * 8) != task[TaskTicket])) { error = InvalidState; }
            const int64_t fatal = io.Read<int64_t>(state + StateFatal * 8);
            if (fatal != NoError) { error = fatal; }
            for (uint32_t ffn = 0; error == NoError && ffn < config[CfgFfn]; ++ffn) {
                GM_ADDR done = peers.Local() + layout.doneBase + ffn * RECORD_BYTES;
                const int32_t producer = PhysicalRank(topology, config[CfgDp] * config[CfgTp] + ffn);
                for (;;) {
                    if (Aborting(io, control)) { error = Aborted; break; }
                    int32_t ticket;
                    int64_t body[12];
                    if (!ReadRecord(io, done, ticket, body) || body[0] != config[CfgEpoch] ||
                        body[10] != ABI_VERSION || ticket < task[TaskTicket]) { continue; }
                    if (ticket != task[TaskTicket] || !RecordMatches(body, task, producer, ffn)) {
                        error = TaskIdMismatch;
                    } else if (body[6] != Success || body[7] != NoError) {
                        error = body[7] == NoError ? PeerFailed : body[7];
                    }
                    break;
                }
            }
            if (error != NoError) { SetFailure(io, state, error); }
        }
        // Only AIV0 polls; all vectors enter or skip unpermute together even
        // when an independent stream updates the ABORT control word.
        AscendC::SyncAll<true>();
        const int64_t error = io.Read<int64_t>(state + StateFatal * 8);
        if (error != NoError) {
            // Shape comes from validated host tiling, never malformed task
            // metadata. The caller still observes fatal/completion ERROR.
            for (uint64_t row = core; row < tiling.inputTokens; row += cores) {
                io.Zero(out + row * config[CfgHidden] * 2, config[CfgHidden] * 2);
            }
            AscendC::SyncAll<true>();
            if (core == 0) { CompleteTask(io, completion, task, Error, error, 0); }
            return;
        }
    }
    for (uint64_t begin = 0, tile = 0; begin < task[TaskTokens]; begin += layout.chunkTokens, ++tile) {
        const uint32_t tokens = Min(layout.chunkTokens, task[TaskTokens] - begin);
        auto unpermuteTiling = UnpermuteTiling(tokens, config[CfgHidden], config[CfgTopK], cores);
        KernelMoeTokenUnpermute<AscendC::bfloat16_t, int32_t, float, true> unpermute;
        unpermute.Init(peers.Local() + layout.dBase +
                tile * layout.chunkTokens * config[CfgTopK] * config[CfgHidden] * 2,
            route_index + begin * config[CfgTopK] * 4,
            topk_weights + begin * config[CfgTopK] * 4,
            out + begin * config[CfgHidden] * 2, &unpermuteTiling, true);
        unpermute.Process();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll<true>();
    }
    AscendC::SyncAll<true>();
    if (core == 0) {
        ControlIo io;
        if (Aborting(io, control)) {
            SetFailure(io, state, Aborted);
            CompleteTask(io, completion, task, Error, Aborted, 0);
            return;
        }
        const int64_t routes = io.CountActive(token_mask, task[TaskTokens]) * config[CfgTopK];
        CompleteTask(io, completion, task, Success, NoError, routes);
        io.Write<int64_t>(state + StateLastCombined * 8, task[TaskSeq]);
        io.Write<int64_t>(state + StateRun * 8, Idle);
    }
}
