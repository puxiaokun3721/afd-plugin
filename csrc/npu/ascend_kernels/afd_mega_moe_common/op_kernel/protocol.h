// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the AFD plugin project.
#ifndef AFD_MEGA_MOE_PROTOCOL_H
#define AFD_MEGA_MOE_PROTOCOL_H
#include "kernel_operator.h"
#include "afd_mega_moe/abi.h"
#include "afd_mega_moe/tiling.h"
#include "utils/hccl_shmem.hpp"

namespace AfdMegaMoe {
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
constexpr uint32_t IO_BYTES = 8192;
constexpr uint32_t ERROR_STRIDE = 32;

// Used only before/after native compute. No UB or event is live across that
// boundary. Every GM access to a changing record goes through MTE, not DCache.
class ControlIo {
public:
    __aicore__ inline ControlIo() { pipe_.InitBuffer(buffer_, IO_BYTES); }
    __aicore__ inline ~ControlIo() { AscendC::PipeBarrier<PIPE_ALL>(); }
    template <typename T>
    __aicore__ inline T Read(GM_ADDR address)
    {
        auto local = buffer_.Get<T>();
        GlobalTensor<T> global;
        global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(address));
        AscendC::DataCopyPad(local, global, {1, sizeof(T), 0, 0}, {});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        return local.GetValue(0);
    }
    template <typename T>
    __aicore__ inline void Write(GM_ADDR address, T value)
    {
        auto local = buffer_.Get<T>();
        local.SetValue(0, value);
        Store(address, local, sizeof(T));
    }
    template <typename T>
    __aicore__ inline void ReadArray(GM_ADDR address, T *values, uint32_t count)
    {
        auto local = buffer_.Get<T>();
        GlobalTensor<T> global;
        global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(address));
        AscendC::DataCopyPad(local, global, {1, static_cast<uint16_t>(count * sizeof(T)), 0, 0}, {});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        for (uint32_t index = 0; index < count; ++index) { values[index] = local.GetValue(index); }
    }
    template <typename T>
    __aicore__ inline void WriteArray(GM_ADDR address, const T *values, uint32_t count)
    {
        auto local = buffer_.Get<T>();
        for (uint32_t index = 0; index < count; ++index) { local.SetValue(index, values[index]); }
        Store(address, local, count * sizeof(T));
    }
    __aicore__ inline bool ValidTopology(const int64_t *config, GM_ADDR topology)
    {
        if (!ValidateConfig(config, CONFIG_FIELDS)) { return false; }
        const uint32_t attention = config[CfgDp] * config[CfgTp];
        const uint32_t world = attention + config[CfgFfn];
        auto local = buffer_.Get<int32_t>();
        GlobalTensor<int32_t> global;
        global.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(topology));
        AscendC::DataCopyPad(local, global, {1, static_cast<uint16_t>(world * 16), 0, 0}, {});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        uint32_t seen[12]{}; // 384-rank protocol maximum
        for (uint32_t logical = 0; logical < world; ++logical) {
            const int32_t role = logical < attention ? 0 : 1;
            const int32_t dp = logical < attention ? logical / config[CfgTp] : -1;
            const int32_t index = logical < attention ? logical % config[CfgTp] : logical - attention;
            const int32_t physical = local.GetValue(logical * 4 + 3);
            if (local.GetValue(logical * 4) != role || local.GetValue(logical * 4 + 1) != dp ||
                local.GetValue(logical * 4 + 2) != index || physical < 0 || physical >= world) { return false; }
            const uint32_t bit = uint32_t{1} << (physical % 32);
            if ((seen[physical / 32] & bit) != 0) { return false; }
            seen[physical / 32] |= bit;
        }
        return true;
    }
    __aicore__ inline int64_t CheckRoutes(GM_ADDR ids, GM_ADDR mask, uint64_t tokens,
                                         uint32_t topk, uint32_t experts, uint32_t core, uint32_t cores)
    {
        constexpr uint32_t TOKEN_PAGE = 64;
        constexpr uint32_t MASK_OFFSET = 4096; // 64 tokens * max top-k 16 * int32
        auto localIds = buffer_.Get<int32_t>();
        auto localMask = buffer_.Get<uint8_t>()[MASK_OFFSET];
        for (uint64_t begin = core * TOKEN_PAGE; begin < tokens; begin += cores * TOKEN_PAGE) {
            const uint32_t count = Min(TOKEN_PAGE, tokens - begin);
            GlobalTensor<int32_t> globalIds;
            GlobalTensor<uint8_t> globalMask;
            globalIds.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ids) + begin * topk);
            globalMask.SetGlobalBuffer(mask + begin);
            AscendC::DataCopyPad(localIds, globalIds, {1, static_cast<uint16_t>(count * topk * 4), 0, 0}, {});
            AscendC::DataCopyPad(localMask, globalMask, {1, static_cast<uint16_t>(count), 0, 0}, {});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
            for (uint32_t token = 0; token < count; ++token) {
                if (localMask.GetValue(token) == 0) { continue; }
                for (uint32_t k = 0; k < topk; ++k) {
                    const int32_t expert = localIds.GetValue(token * topk + k);
                    if (expert < 0 || expert >= experts) { return RouteOutOfRange; }
                }
            }
        }
        return NoError;
    }
    __aicore__ inline uint64_t CountActive(GM_ADDR mask, uint64_t tokens)
    {
        uint64_t active = 0;
        auto local = buffer_.Get<uint8_t>();
        for (uint64_t begin = 0; begin < tokens; begin += IO_BYTES) {
            const uint32_t count = Min(IO_BYTES, tokens - begin);
            GlobalTensor<uint8_t> global;
            global.SetGlobalBuffer(mask + begin);
            AscendC::DataCopyPad(local, global, {1, static_cast<uint16_t>(count), 0, 0}, {});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
            for (uint32_t token = 0; token < count; ++token) { active += local.GetValue(token) != 0; }
        }
        return active;
    }
    __aicore__ inline int64_t CheckCounts(GM_ADDR address, uint32_t stride,
                                         uint32_t experts, uint32_t first, uint32_t localExperts,
                                         uint64_t maximum, uint64_t &localRoutes)
    {
        auto local = buffer_.Get<int32_t>();
        GlobalTensor<int32_t> global;
        global.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(address));
        AscendC::DataCopyPad(local, global, {1, static_cast<uint16_t>(stride * 4), 0, 0}, {});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        uint64_t total = 0;
        for (uint32_t expert = 0; expert < stride; ++expert) {
            const int32_t count = local.GetValue(expert);
            if (count < 0 || (expert >= experts && count != 0)) { return RouteOutOfRange; }
            total += count;
            if (expert >= first && expert < first + localExperts) { localRoutes += count; }
        }
        return total <= maximum ? NoError : RouteOutOfRange;
    }
    __aicore__ inline void Copy(GM_ADDR destination, GM_ADDR source, uint64_t bytes)
    {
        auto local = buffer_.Get<uint8_t>();
        for (uint64_t offset = 0; offset < bytes; offset += IO_BYTES) {
            const uint16_t length = static_cast<uint16_t>(Min(IO_BYTES, bytes - offset));
            GlobalTensor<uint8_t> src, dst;
            src.SetGlobalBuffer(source + offset);
            dst.SetGlobalBuffer(destination + offset);
            AscendC::DataCopyPad(local, src, {1, length, 0, 0}, {});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            AscendC::DataCopyPad(dst, local, {1, length, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        }
    }
    __aicore__ inline void Zero(GM_ADDR destination, uint64_t bytes)
    {
        auto local = buffer_.Get<uint8_t>();
        AscendC::Duplicate(buffer_.Get<int32_t>(), int32_t{0}, IO_BYTES / sizeof(int32_t));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        for (uint64_t offset = 0; offset < bytes; offset += IO_BYTES) {
            GlobalTensor<uint8_t> dst;
            dst.SetGlobalBuffer(destination + offset);
            AscendC::DataCopyPad(dst, local,
                {1, static_cast<uint16_t>(Min(IO_BYTES, bytes - offset)), 0, 0});
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
    }
private:
    template <typename T>
    __aicore__ inline void Store(GM_ADDR address, LocalTensor<T> local, uint32_t bytes)
    {
        GlobalTensor<T> global;
        global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(address));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::DataCopyPad(global, local, {1, static_cast<uint16_t>(bytes), 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
    }
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_;
};

__aicore__ inline uint32_t VectorIndex()
{
    return get_block_idx() + get_subblockid() * get_block_num();
}
__aicore__ inline uint32_t VectorCount() { return get_block_num() * get_subblockdim(); }
__aicore__ inline int32_t PhysicalRank(GM_ADDR topology, uint32_t logical)
{
    return reinterpret_cast<__gm__ int32_t *>(topology)[logical * 4 + 3];
}
__aicore__ inline int32_t LogicalRank(const int64_t *config, GM_ADDR topology)
{
    const uint32_t world = config[CfgDp] * config[CfgTp] + config[CfgFfn];
    for (uint32_t index = 0; index < world; ++index) {
        if (PhysicalRank(topology, index) == config[CfgRank]) { return index; }
    }
    return -1;
}
class Peers {
public:
    __aicore__ inline Peers()
    {
        context_ = reinterpret_cast<__gm__ HcclOpResParam *>(AscendC::GetHcclContext<HCCL_GROUP_ID_0>());
    }
    __aicore__ inline GM_ADDR Local() const { return reinterpret_cast<GM_ADDR>(context_->localWindowsIn); }
    __aicore__ inline GM_ADDR At(int32_t physical) const
    {
        return physical == context_->localUsrRankId ? Local() : reinterpret_cast<GM_ADDR>(
            reinterpret_cast<__gm__ HcclRankRelationResV2 *>(context_->remoteRes[physical].nextDevicePtr)->windowsIn);
    }
    __aicore__ inline bool Valid(const int64_t *config) const
    {
        return context_ != nullptr && context_->localUsrRankId == config[CfgRank] &&
            context_->rankSize == config[CfgDp] * config[CfgTp] + config[CfgFfn] &&
            context_->winSize >= MakeWindowLayout(config).requiredBytes;
    }
private:
    __gm__ HcclOpResParam *context_;
};

__aicore__ inline int64_t ValidateTask(const int64_t *task, const int64_t *config, int32_t rank)
{
    if (task[TaskAbi] != ABI_VERSION || task[TaskEpoch] != config[CfgEpoch] ||
        task[TaskDp] < 0 || task[TaskDp] >= config[CfgDp] || task[TaskSourceRank] != rank ||
        task[TaskTokens] < 0 || task[TaskTokens] > config[CfgSourceCapacity] || task[TaskMicrobatch] < 0 ||
        task[TaskTotalTokens] != task[TaskTokens] || task[TaskInputLayout] != INPUT_LAYOUT_EXPANDED ||
        task[TaskCommRound] != 0 || task[TaskCommRoundCount] != 1 || task[TaskTokenBase] != 0 ||
        task[TaskFlags] != 0 || task[TaskReserved] != 0 || task[TaskSeq] < 1 || task[TaskSeq] > MAX_TICKET ||
        task[TaskTicket] < 1 || task[TaskTicket] > MAX_TICKET) { return TaskIdMismatch; }
    if (task[TaskLayer] < 0 || task[TaskLayer] >= config[CfgLayerCapacity]) { return InvalidLayer; }
    return NoError;
}
__aicore__ inline bool SameTask(const int64_t *left, const int64_t *right)
{
    return left[TaskEpoch] == right[TaskEpoch] && left[TaskSeq] == right[TaskSeq] &&
        left[TaskTicket] == right[TaskTicket] && left[TaskDp] == right[TaskDp] &&
        left[TaskLayer] == right[TaskLayer] && left[TaskMicrobatch] == right[TaskMicrobatch] &&
        left[TaskCommRound] == right[TaskCommRound] && left[TaskCommRoundCount] == right[TaskCommRoundCount];
}
__aicore__ inline bool Aborting(ControlIo &io, GM_ADDR control)
{
    return io.Read<int32_t>(control + CONTROL_ABORT_INDEX * sizeof(int32_t)) != 0;
}
__aicore__ inline bool StateSessionValid(ControlIo &io, GM_ADDR state, const int64_t *config)
{
    return io.Read<int64_t>(state + StateAbi * 8) == ABI_VERSION &&
        io.Read<int64_t>(state + StateEpoch * 8) == config[CfgEpoch];
}
__aicore__ inline void SetFailure(ControlIo &io, GM_ADDR state, int64_t error)
{
    io.Write<int64_t>(state + StateFatal * 8, error);
    io.Write<int64_t>(state + StateRun * 8, Failed);
}
__aicore__ inline void CompleteTask(ControlIo &io, GM_ADDR completion, const int64_t *task,
                                  int64_t status, int64_t error, int64_t routes)
{
    const int64_t fields[COMPLETION_FIELDS] = {ABI_VERSION, status, error, task[TaskEpoch],
        task[TaskSeq], task[TaskDp], task[TaskLayer], routes};
    io.WriteArray(completion, fields, COMPLETION_FIELDS);
}
// A first invalidates the old ticket, then commits a new immutable record only
// after every payload write drained. A reader checks the ticket twice so an
// old ticket can never authenticate a body being replaced for the next task.
__aicore__ inline void Publish(ControlIo &io, GM_ADDR record, const int64_t *task,
                              int64_t producer, int64_t producerIndex,
                              int64_t status, int64_t error, int64_t routes)
{
    io.Write<int32_t>(record, 0);
    int64_t body[(RECORD_BYTES - RECORD_BODY_OFFSET) / 8]{};
    body[(RecordEpoch - RECORD_BODY_OFFSET) / 8] = task[TaskEpoch];
    body[(RecordSeq - RECORD_BODY_OFFSET) / 8] = task[TaskSeq];
    body[(RecordDp - RECORD_BODY_OFFSET) / 8] = task[TaskDp];
    body[(RecordLayer - RECORD_BODY_OFFSET) / 8] = task[TaskLayer];
    body[(RecordMicrobatch - RECORD_BODY_OFFSET) / 8] = task[TaskMicrobatch];
    body[(RecordProducerRank - RECORD_BODY_OFFSET) / 8] = producer;
    body[(RecordStatus - RECORD_BODY_OFFSET) / 8] = status;
    body[(RecordError - RECORD_BODY_OFFSET) / 8] = error;
    body[(RecordRoutes - RECORD_BODY_OFFSET) / 8] = routes;
    body[(RecordLayout - RECORD_BODY_OFFSET) / 8] = INPUT_LAYOUT_EXPANDED;
    body[(RecordAbi - RECORD_BODY_OFFSET) / 8] = ABI_VERSION;
    body[(RecordProducerIndex - RECORD_BODY_OFFSET) / 8] = producerIndex;
    io.WriteArray(record + RECORD_BODY_OFFSET, body, (RECORD_BYTES - RECORD_BODY_OFFSET) / 8);
    io.Write<int32_t>(record, static_cast<int32_t>(task[TaskTicket]));
}
__aicore__ inline bool ReadRecord(ControlIo &io, GM_ADDR record, int32_t &ticket, int64_t *body)
{
    ticket = io.Read<int32_t>(record);
    if (ticket <= 0) { return false; }
    io.ReadArray(record + RECORD_BODY_OFFSET, body, 12);
    return ticket == io.Read<int32_t>(record);
}
__aicore__ inline bool RecordMatches(const int64_t *body, const int64_t *task,
                                   int64_t producer, int64_t producerIndex)
{
    return body[0] == task[TaskEpoch] && body[1] == task[TaskSeq] &&
        body[2] == task[TaskDp] && body[3] == task[TaskLayer] && body[4] == task[TaskMicrobatch] &&
        body[5] == producer && body[9] == INPUT_LAYOUT_EXPANDED && body[10] == ABI_VERSION &&
        body[11] == producerIndex;
}
__aicore__ inline void PublishDone(ControlIo &io, const Peers &peers, GM_ADDR topology,
                                 const int64_t *config, const int64_t *task,
                                 int32_t ffnIndex, int64_t status, int64_t error, int64_t routes)
{
    const auto layout = MakeWindowLayout(config);
    for (uint32_t source = 0; source < config[CfgTp]; ++source) {
        const int32_t rank = PhysicalRank(topology, task[TaskDp] * config[CfgTp] + source);
        Publish(io, peers.At(rank) + layout.doneBase + ffnIndex * RECORD_BYTES, task,
                config[CfgRank], ffnIndex, status, error, routes);
    }
}
} // namespace AfdMegaMoe
#endif
