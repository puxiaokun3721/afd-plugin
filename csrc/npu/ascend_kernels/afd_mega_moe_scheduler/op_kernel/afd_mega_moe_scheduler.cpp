// SPDX-License-Identifier: Apache-2.0
#include "op_kernel/protocol.h"
using namespace AfdMegaMoe;

extern "C" __global__ __aicore__ void afd_mega_moe_scheduler(
    GM_ADDR comm_args, GM_ADDR topology, GM_ADDR control, GM_ADDR state,
    GM_ADDR work_desc, GM_ADDR source_desc, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AfdMegaMoe::TilingData);
    GET_TILING_DATA_WITH_STRUCT(AfdMegaMoe::TilingData, tiling, tilingGM);
    if (VectorIndex() != 0) { return; }
    const int64_t *config = tiling.config;
    const auto layout = MakeWindowLayout(config);
    const int32_t logical = LogicalRank(config, topology);
    const int32_t attention = config[CfgDp] * config[CfgTp];
    const int32_t ffn = logical - attention;
    Peers peers;
    ControlIo io;
    int64_t work[WORK_FIELDS]{};
    work[WorkAbi] = ABI_VERSION;
    work[WorkEpoch] = config[CfgEpoch];
    work[WorkFfn] = ffn;
    work[WorkDp] = -1;
    int64_t fatal = io.Read<int64_t>(state + StateFatal * 8);
    if (!io.ValidTopology(config, topology) || !StateSessionValid(io, state, config) || ffn < 0 || ffn >= config[CfgFfn] || !peers.Valid(config)) {
        fatal = InvalidConfig;
    }
    const int64_t previousRun = io.Read<int64_t>(state + StateRun * 8);
    if (fatal == NoError && previousRun != Idle && previousRun != Complete && previousRun != Stopped) {
        fatal = InvalidState;
    }
    if (fatal != NoError) {
        work[WorkKind] = ErrorWork;
        work[WorkError] = fatal;
        io.WriteArray(work_desc, work, WORK_FIELDS);
        SetFailure(io, state, fatal);
        return;
    }
    uint32_t cursor = io.Read<int64_t>(state + StateCursor * 8) % config[CfgDp];
    for (;;) {
        if (Aborting(io, control)) {
            work[WorkKind] = ErrorWork;
            work[WorkError] = Aborted;
            io.WriteArray(work_desc, work, WORK_FIELDS);
            SetFailure(io, state, Aborted);
            return;
        }
        if (io.Read<int32_t>(control + CONTROL_STOP_INDEX * 4) != 0) {
            work[WorkKind] = StopWork;
            io.WriteArray(work_desc, work, WORK_FIELDS);
            io.Write<int64_t>(state + StateRun * 8, Stopped);
            return;
        }
        for (uint32_t scanned = 0; scanned < config[CfgDp]; ++scanned) {
            const uint32_t dp = (cursor + scanned) % config[CfgDp];
            const int64_t expected = io.Read<int64_t>(state + (StateCompletedBase + dp) * 8) + 1;
            int64_t selected[TASK_FIELDS]{};
            uint64_t maxTokens = 0;
            int64_t error = NoError;
            bool ready = true;
            for (uint32_t source = 0; source < config[CfgTp]; ++source) {
                GM_ADDR record = peers.Local() + layout.reservedBytes +
                    (dp * config[CfgTp] + source) * layout.inboxStride;
                int32_t ticket;
                int64_t body[12];
                if (!ReadRecord(io, record, ticket, body) || body[0] != config[CfgEpoch] ||
                    body[10] != ABI_VERSION || body[1] < expected) { ready = false; break; }
                int64_t task[TASK_FIELDS];
                io.ReadArray(record + RECORD_BYTES, task, TASK_FIELDS);
                if (ticket != io.Read<int32_t>(record)) { ready = false; break; }
                const int32_t physical = PhysicalRank(topology, dp * config[CfgTp] + source);
                error = ValidateTask(task, config, physical);
                if (error == NoError && (body[1] != expected || task[TaskSeq] != expected)) { error = SequenceGap; }
                if (error == NoError && (task[TaskDp] != dp || task[TaskTicket] != ticket ||
                    !RecordMatches(body, task, physical, source) || body[6] != Pending || body[7] != NoError)) {
                    error = TaskIdMismatch;
                }
                if (source == 0) {
                    for (uint32_t field = 0; field < TASK_FIELDS; ++field) { selected[field] = task[field]; }
                } else if (error == NoError && !SameTask(selected, task)) { error = TaskIdMismatch; }
                io.WriteArray(source_desc + source * TASK_FIELDS * 8, task, TASK_FIELDS);
                maxTokens = Max(maxTokens, task[TaskTokens]);
                if (error != NoError) { break; }
            }
            if (!ready) { continue; }
            work[WorkDp] = dp;
            work[WorkSeq] = selected[TaskSeq];
            work[WorkTicket] = selected[TaskTicket];
            work[WorkLayer] = selected[TaskLayer];
            work[WorkMicrobatch] = selected[TaskMicrobatch];
            work[WorkSources] = config[CfgTp];
            work[WorkTiles] = (maxTokens + PERMUTE_TOKENS - 1) / PERMUTE_TOKENS;
            work[WorkExperts] = config[CfgExperts];
            work[WorkCapacity] = config[CfgRecvCapacity];
            work[WorkCommRoundCount] = 1;
            work[WorkInputLayout] = INPUT_LAYOUT_EXPANDED;
            work[WorkError] = error;
            // Commit kind last; following FFN executes on the same stream.
            io.WriteArray(work_desc, work, WORK_FIELDS);
            io.Write<int64_t>(work_desc + WorkKind * 8, error == NoError ? Work : ErrorWork);
            io.Write<int64_t>(state + StateCursor * 8, (dp + 1) % config[CfgDp]);
            io.Write<int64_t>(state + StateDp * 8, dp);
            io.Write<int64_t>(state + StateSeq * 8, selected[TaskSeq]);
            io.Write<int64_t>(state + StateTicket * 8, selected[TaskTicket]);
            if (error == NoError) { io.Write<int64_t>(state + StateRun * 8, Claimed); }
            else { SetFailure(io, state, error); }
            return;
        }
        cursor = (cursor + 1) % config[CfgDp];
    }
}
