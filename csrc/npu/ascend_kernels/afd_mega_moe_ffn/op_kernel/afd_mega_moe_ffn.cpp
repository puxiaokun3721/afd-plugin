// SPDX-License-Identifier: Apache-2.0
#include "op_kernel/protocol.h"
#include "op_kernel/native_ffn.h"
using namespace AfdMegaMoe;
namespace {
constexpr uint64_t VALIDATION_HEADER = 4096;
__aicore__ inline void InvalidateScalar(GM_ADDR address, uint32_t bytes)
{
    AscendC::GlobalTensor<uint8_t> global;
    for (uint32_t offset = 0; offset < bytes; offset += 32) {
        global.SetGlobalBuffer(address + offset);
        AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                         AscendC::DcciDst::CACHELINE_OUT>(global);
    }
}
}
extern "C" __global__ __aicore__ void afd_mega_moe_ffn(
    GM_ADDR work_desc, GM_ADDR source_desc, GM_ADDR layer_id_to_slot,
    GM_ADDR w13, GM_ADDR w2, GM_ADDR weight_scale13, GM_ADDR weight_scale2,
    GM_ADDR compensation13, GM_ADDR compensation2, GM_ADDR layer_params,
    GM_ADDR comm_args, GM_ADDR topology, GM_ADDR control, GM_ADDR state,
    GM_ADDR ffn_scratch, GM_ADDR completion, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(AfdMegaMoe::TilingData);
    GET_TILING_DATA_WITH_STRUCT(AfdMegaMoe::TilingData, tiling, tilingGM);
    const int64_t *config = tiling.config;
    const auto layout = MakeWindowLayout(config);
    InvalidateScalar(work_desc, WORK_FIELDS * 8);
    InvalidateScalar(source_desc, config[CfgTp] * TASK_FIELDS * 8);
    auto work = reinterpret_cast<__gm__ int64_t *>(work_desc);
    int64_t task[TASK_FIELDS]{};
    task[TaskAbi] = ABI_VERSION;
    task[TaskEpoch] = work[WorkEpoch];
    task[TaskSeq] = work[WorkSeq];
    task[TaskTicket] = work[WorkTicket];
    task[TaskDp] = work[WorkDp];
    task[TaskLayer] = work[WorkLayer];
    task[TaskMicrobatch] = work[WorkMicrobatch];
    task[TaskCommRoundCount] = 1;
    task[TaskInputLayout] = INPUT_LAYOUT_EXPANDED;
    const int32_t ffn = LogicalRank(config, topology) - config[CfgDp] * config[CfgTp];
    Peers peers;
    if (work[WorkKind] != Work) {
        if ASCEND_IS_AIV {
            if (VectorIndex() == 0) {
                ControlIo io;
                if (work[WorkKind] == StopWork) {
                    CompleteTask(io, completion, task, Stop, NoError, 0);
                    io.Write<int64_t>(state + StateRun * 8, Stopped);
                } else {
                    const int64_t error = work[WorkError] == NoError ? InvalidState : work[WorkError];
                    if (io.ValidTopology(config, topology) && peers.Valid(config) &&
                        ffn >= 0 && ffn < config[CfgFfn] && task[TaskDp] >= 0 &&
                        task[TaskDp] < config[CfgDp] && task[TaskTicket] > 0 && task[TaskTicket] <= MAX_TICKET) {
                        PublishDone(io, peers, topology, config, task, ffn, Error, error, 0);
                    }
                    CompleteTask(io, completion, task, Error, error, 0);
                    SetFailure(io, state, error);
                }
            }
        }
        return;
    }
    GM_ADDR validation = ffn_scratch + FfnScratchBytes(config) - CONTROL_SCRATCH_BYTES;
    if ASCEND_IS_AIV {
        const uint32_t core = VectorIndex(), cores = VectorCount();
        ControlIo io;
        int64_t error = NoError;
        uint64_t routes = 0, maxTokens = 0;
        int64_t slot = -1;
        if (!ValidateConfig(config, CONFIG_FIELDS) || !StateSessionValid(io, state, config) || !peers.Valid(config) || ffn < 0 || ffn >= config[CfgFfn] ||
            work[WorkAbi] != ABI_VERSION || work[WorkEpoch] != config[CfgEpoch] || work[WorkFfn] != ffn ||
            work[WorkSources] != config[CfgTp] || work[WorkExperts] != config[CfgExperts] ||
            work[WorkCapacity] != config[CfgRecvCapacity] || work[WorkCommRound] != 0 ||
            work[WorkCommRoundCount] != 1 || work[WorkTokenBase] != 0 ||
            work[WorkInputLayout] != INPUT_LAYOUT_EXPANDED || work[WorkSlot] != 0 || work[WorkFlags] != 0 ||
            task[TaskDp] < 0 || task[TaskDp] >= config[CfgDp] || task[TaskSeq] < 1 || task[TaskSeq] > MAX_TICKET ||
            task[TaskTicket] < 1 || task[TaskTicket] > MAX_TICKET ||
            tiling.ffnScratchBytes < FfnScratchBytes(config)) { error = InvalidConfig; }
        if (error == NoError) {
            if (task[TaskLayer] < 0 || task[TaskLayer] >= config[CfgLayerCapacity]) { error = InvalidLayer; }
            else {
                slot = io.Read<int64_t>(layer_id_to_slot + task[TaskLayer] * 8);
                if (slot < 0 || slot >= config[CfgLayers] ||
                    io.Read<float>(layer_params + slot * 16 + 4) != 0.0f) { error = InvalidLayer; }
            }
        }
        if (error == NoError && (io.Read<int64_t>(state + StateFatal * 8) != NoError ||
            io.Read<int64_t>(state + StateRun * 8) != Claimed ||
            io.Read<int64_t>(state + StateSeq * 8) != task[TaskSeq] ||
            io.Read<int64_t>(state + StateTicket * 8) != task[TaskTicket])) { error = InvalidState; }
        if (core == 0 && !io.ValidTopology(config, topology)) { error = InvalidConfig; }
        if (Aborting(io, control)) { error = Aborted; }
        if (error == NoError) {
            // Validate every selected source, then check count pages before any
            // mixed-core flag can be produced. All AIVs reach the same barrier.
            for (uint32_t source = 0; source < config[CfgTp]; ++source) {
                int64_t sourceTask[TASK_FIELDS];
                io.ReadArray(source_desc + source * TASK_FIELDS * 8, sourceTask, TASK_FIELDS);
                const int32_t physical = PhysicalRank(topology, task[TaskDp] * config[CfgTp] + source);
                const int64_t sourceError = ValidateTask(sourceTask, config, physical);
                if (sourceError != NoError) { error = sourceError; break; }
                if (!SameTask(task, sourceTask)) { error = TaskIdMismatch; break; }
                maxTokens = Max(maxTokens, sourceTask[TaskTokens]);
                for (uint64_t tile = 0; tile < layout.numChunks; ++tile) {
                    if ((source * layout.numChunks + tile) % cores != core) { continue; }
                    const uint64_t begin = tile * layout.chunkTokens;
                    const uint64_t tokens = sourceTask[TaskTokens] > begin ?
                        Min(layout.chunkTokens, sourceTask[TaskTokens] - begin) : 0;
                    GM_ADDR counts = peers.Local() + layout.reservedBytes +
                        (task[TaskDp] * config[CfgTp] + source) * layout.inboxStride +
                        2 * RECORD_BYTES + tile * layout.expertStride * 4;
                    const int64_t countError = io.CheckCounts(counts, layout.expertStride,
                        config[CfgFfn] * config[CfgExperts], ffn * config[CfgExperts], config[CfgExperts],
                        tokens * config[CfgTopK], routes);
                    if (countError != NoError) { error = countError; }
                }
            }
            if (work[WorkTiles] != (maxTokens + PERMUTE_TOKENS - 1) / PERMUTE_TOKENS) {
                error = TaskIdMismatch;
            }
        }
        const int64_t local[2] = {error, static_cast<int64_t>(routes)};
        io.WriteArray(validation + core * ERROR_STRIDE, local, 2);
        AscendC::SyncAll<true>();
        if (core == 0) {
            routes = 0;
            for (uint32_t index = 0; index < cores; ++index) {
                int64_t other[2];
                io.ReadArray(validation + index * ERROR_STRIDE, other, 2);
                if (other[0] != NoError) { error = other[0]; }
                routes += other[1];
            }
            const int64_t header[4] = {error, static_cast<int64_t>(routes), slot, static_cast<int64_t>(maxTokens)};
            io.WriteArray(validation + VALIDATION_HEADER, header, 4);
            if (error == NoError) { io.Write<int64_t>(state + StateRun * 8, Running); }
        }
    }
    // Only two mixed barriers surround the body. AIC never enters GMM while an
    // AIV can take a validation/abort exit; native flags 0..10 drain unchanged.
    AscendC::SyncAll<false>();
    InvalidateScalar(validation + VALIDATION_HEADER, 32);
    auto checked = reinterpret_cast<__gm__ int64_t *>(validation + VALIDATION_HEADER);
    if (checked[0] == NoError && checked[1] > 0) {
        RunNativeFfn(tiling, checked[3], checked[2], task[TaskDp], ffn, source_desc,
            w13, w2, weight_scale13, weight_scale2, compensation13, compensation2,
            layer_params, ffn_scratch);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIV {
        if (VectorIndex() == 0) {
            ControlIo io;
            int64_t result[4];
            io.ReadArray(validation + VALIDATION_HEADER, result, 4);
            int64_t error = result[0];
            if (Aborting(io, control)) { error = Aborted; }
            const int64_t status = error == NoError ? Success : Error;
            if (io.ValidTopology(config, topology) && peers.Valid(config) &&
                ffn >= 0 && ffn < config[CfgFfn] && task[TaskDp] >= 0 && task[TaskDp] < config[CfgDp] &&
                task[TaskTicket] > 0 && task[TaskTicket] <= MAX_TICKET) {
                PublishDone(io, peers, topology, config, task, ffn, status, error, result[1]);
            }
            CompleteTask(io, completion, task, status, error, result[1]);
            if (error == NoError) {
                io.Write<int64_t>(state + (StateCompletedBase + task[TaskDp]) * 8, task[TaskSeq]);
                io.Write<int64_t>(state + StateRun * 8, Complete);
            } else { SetFailure(io, state, error); }
        }
    }
}
