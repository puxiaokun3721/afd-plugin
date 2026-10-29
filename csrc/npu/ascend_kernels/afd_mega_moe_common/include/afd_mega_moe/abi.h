// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the AFD plugin project.
#ifndef AFD_MEGA_MOE_ABI_H
#define AFD_MEGA_MOE_ABI_H

#include <cstddef>
#include <cstdint>

#if defined(__CCE_AICORE__)
#define AFD_MOE_INLINE __aicore__ inline
#else
#define AFD_MOE_INLINE inline
#endif

namespace AfdMegaMoe {
constexpr uint64_t CONFIG_FIELDS = 24;
constexpr uint64_t TASK_FIELDS = 16;
constexpr uint64_t WORK_FIELDS = 32;
constexpr uint64_t STATE_HEADER_FIELDS = 32;
constexpr uint64_t COMPLETION_FIELDS = 8;
constexpr uint64_t RECORD_BYTES = 512;
constexpr uint64_t RECORD_BODY_OFFSET = 128;
constexpr uint64_t CONTROL_BYTES = 512;
constexpr uint64_t PREFIX_RESERVED_BYTES = 10ULL * 1024 * 1024;
constexpr uint64_t TAIL_RESERVED_BYTES = 512ULL * 1024;
constexpr uint64_t CONTROL_SCRATCH_BYTES = 64ULL * 1024;
constexpr uint64_t PERMUTE_TOKENS = 1024;
constexpr uint64_t PERMUTE_CORES = 128;
constexpr uint64_t W4_ROUTE_CACHE_BYTES = 16032;
constexpr int64_t MAX_TICKET = 0x7fffffff;
constexpr int64_t ABI_VERSION = 1;
constexpr int64_t INPUT_LAYOUT_EXPANDED = 1;
constexpr int64_t QUANT_W4A8_PER_CHANNEL = 1;

enum ConfigIndex : uint32_t {
    CfgAbiVersion = 0, CfgInputLayout = 1, CfgQuantMode = 2,
    CfgEpoch = 3, CfgRank = 4, CfgDp = 5, CfgTp = 6, CfgFfn = 7,
    CfgExperts = 8, CfgHidden = 9, CfgIntermediate = 10, CfgTopK = 11,
    CfgSourceCapacity = 12, CfgRecvCapacity = 13, CfgLayers = 14,
    CfgLayerCapacity = 15, CfgFlags = 16, CfgWindowBytes = 17,
    CfgInflight = 18, CfgCommRounds = 19,
};
enum TaskIndex : uint32_t {
    TaskAbi = 0, TaskEpoch = 1, TaskSeq = 2, TaskTicket = 3,
    TaskDp = 4, TaskLayer = 5, TaskMicrobatch = 6, TaskTokens = 7,
    TaskSourceRank = 8, TaskCommRound = 9, TaskCommRoundCount = 10,
    TaskTokenBase = 11, TaskTotalTokens = 12, TaskInputLayout = 13,
    TaskFlags = 14, TaskReserved = 15,
};
enum WorkIndex : uint32_t {
    WorkAbi = 0, WorkEpoch = 1, WorkKind = 2, WorkError = 3,
    WorkDp = 4, WorkSeq = 5, WorkTicket = 6, WorkLayer = 7,
    WorkMicrobatch = 8, WorkSources = 9, WorkTiles = 10,
    WorkFfn = 11, WorkExperts = 12, WorkCapacity = 13,
    WorkCommRound = 14, WorkCommRoundCount = 15, WorkTokenBase = 16,
    WorkInputLayout = 17, WorkSlot = 18, WorkFlags = 19,
};
enum StateIndex : uint32_t {
    StateAbi = 0, StateEpoch = 1, StateRun = 2, StateFatal = 3,
    StateCursor = 4, StateDp = 5, StateSeq = 6, StateTicket = 7,
    StateLastCombined = 8, StateCompletedBase = 32,
};
enum RunState : int64_t {
    Idle = 0, Published = 1, Claimed = 2, Running = 3,
    Complete = 4, Stopped = 5, Failed = 6,
};
enum WorkType : int64_t { EmptyWork = 0, Work = 1, StopWork = 2, ErrorWork = 3 };
enum CompletionStatus : int64_t { Pending = 0, Success = 1, Stop = 2, Error = 3 };
enum ErrorCode : int64_t {
    NoError = 0, InvalidConfig = 1, WindowTooSmall = 2,
    UnsupportedWeightLayout = 3, InvalidLayer = 4, TaskIdMismatch = 5,
    SequenceGap = 6, RouteOutOfRange = 7, WorkspaceOverflow = 8,
    PeerFailed = 9, Aborted = 10, InvalidState = 11,
};
enum RecordOffset : uint64_t {
    RecordCommit = 0, RecordEpoch = 128, RecordSeq = 136,
    RecordDp = 144, RecordLayer = 152, RecordMicrobatch = 160,
    RecordProducerRank = 168, RecordStatus = 176, RecordError = 184,
    RecordRoutes = 192, RecordLayout = 200, RecordAbi = 208,
    RecordProducerIndex = 216,
};
constexpr uint32_t CONTROL_STOP_INDEX = 0;
constexpr uint32_t CONTROL_ABORT_INDEX = 32;

AFD_MOE_INLINE uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}
AFD_MOE_INLINE uint64_t Max(uint64_t a, uint64_t b) { return a > b ? a : b; }
AFD_MOE_INLINE uint64_t Min(uint64_t a, uint64_t b) { return a < b ? a : b; }

struct WindowLayout {
    uint64_t chunkTokens;
    uint64_t numChunks;
    uint64_t paddedTokens;
    uint64_t expertStride;
    uint64_t xBase;
    uint64_t scaleBase;
    uint64_t dBase;
    uint64_t doneBase;
    uint64_t inboxStride;
    uint64_t reservedBytes;
    uint64_t tailBytes;
    uint64_t requiredBytes;
};

// The host validates the bounded config before either side computes offsets.
AFD_MOE_INLINE WindowLayout MakeWindowLayout(const int64_t *config)
{
    WindowLayout layout{};
    const uint64_t sources = config[CfgDp] * config[CfgTp];
    const uint64_t ffn = config[CfgFfn];
    const uint64_t topk = config[CfgTopK];
    const uint64_t hidden = config[CfgHidden];
    const uint64_t capacity = config[CfgSourceCapacity];
    layout.chunkTokens = Min(capacity, PERMUTE_TOKENS);
    layout.numChunks = (capacity + layout.chunkTokens - 1) / layout.chunkTokens;
    layout.paddedTokens = layout.chunkTokens * layout.numChunks;
    layout.expertStride = AlignUp(ffn * config[CfgExperts] + 1, 128);
    layout.reservedBytes = PREFIX_RESERVED_BYTES;
    layout.tailBytes = Max(RECORD_BYTES * (sources + ffn), TAIL_RESERVED_BYTES);
    layout.xBase = layout.reservedBytes;
    layout.scaleBase = AlignUp(layout.xBase + layout.paddedTokens * topk * (hidden + 512), 512);
    layout.dBase = AlignUp(layout.scaleBase + 4 * layout.paddedTokens * topk, 512);
    layout.doneBase = AlignUp(layout.dBase + 2 * layout.paddedTokens * topk * hidden, 512);
    layout.inboxStride = 2 * RECORD_BYTES + 4 * layout.numChunks * layout.expertStride;
    const uint64_t attentionBytes = layout.doneBase + RECORD_BYTES * ffn + layout.tailBytes;
    const uint64_t ffnBytes = layout.reservedBytes + sources * layout.inboxStride + layout.tailBytes;
    layout.requiredBytes = AlignUp(Max(attentionBytes, ffnBytes), 512);
    return layout;
}

AFD_MOE_INLINE uint64_t MaxRecvRounds(const int64_t *config)
{
    const auto layout = MakeWindowLayout(config);
    const uint64_t routes = config[CfgTp] * layout.chunkTokens * config[CfgTopK];
    return Max(1, (routes + config[CfgRecvCapacity] - 1) / config[CfgRecvCapacity]);
}
AFD_MOE_INLINE uint64_t SendScratchBytes(const int64_t *config)
{
    return CONTROL_SCRATCH_BYTES + AlignUp(PERMUTE_CORES * AlignUp(
        config[CfgFfn] * config[CfgExperts], 8) * 4, 512);
}
AFD_MOE_INLINE uint64_t FfnNativeScratchBytes(const int64_t *config)
{
    const auto layout = MakeWindowLayout(config);
    const uint64_t s = config[CfgTp], e = config[CfgExperts];
    const uint64_t q = config[CfgRecvCapacity];
    const uint64_t h = config[CfgHidden], i = config[CfgIntermediate];
    const uint64_t tableBytes = PERMUTE_CORES * AlignUp(s * e, 8) * 4 +
        PERMUTE_TOKENS * config[CfgTopK] * 4 +
        layout.numChunks * s * s * e * 4 +
        layout.numChunks * MaxRecvRounds(config) * 3 * s * e * 4;
    // Logical Q need not be aligned. Align the native scale/C storage bases
    // independently so small budgets and tail rounds retain legal DMA addresses.
    const uint64_t bytes = AlignUp(tableBytes, 512) + 2 * AlignUp(4 * q, 512) +
        4 * q * Max(2 * i, h) + q * Max(h, i) +
        4 * s * AlignUp(e, 16);
    return AlignUp(bytes, 512);
}
AFD_MOE_INLINE uint64_t FfnScratchBytes(const int64_t *config)
{
    const auto layout = MakeWindowLayout(config);
    return FfnNativeScratchBytes(config) +
        AlignUp(4 * config[CfgTp] * layout.expertStride, 512) +
        AlignUp(4 * layout.numChunks * config[CfgTp] * config[CfgExperts], 512) +
        CONTROL_SCRATCH_BYTES;
}

// Bounds keep all arithmetic below INT64_MAX, and native route indices int32.
AFD_MOE_INLINE bool ValidateConfig(const int64_t *config, uint64_t count)
{
    if (config == nullptr || count != CONFIG_FIELDS || config[CfgAbiVersion] != ABI_VERSION ||
        config[CfgInputLayout] != INPUT_LAYOUT_EXPANDED ||
        config[CfgQuantMode] != QUANT_W4A8_PER_CHANNEL || config[CfgEpoch] <= 0) {
        return false;
    }
    if (config[CfgDp] < 1 || config[CfgDp] > 384 || config[CfgTp] < 1 || config[CfgTp] > 128 ||
        config[CfgFfn] < 1 || config[CfgFfn] > 128 || config[CfgExperts] < 1 ||
        config[CfgExperts] > 128 || config[CfgFfn] * config[CfgExperts] > 1024) {
        return false;
    }
    const int64_t world = config[CfgDp] * config[CfgTp] + config[CfgFfn];
    if (world > 384 || config[CfgRank] < 0 || config[CfgRank] >= world ||
        config[CfgHidden] < 1024 || config[CfgHidden] > 8192 || config[CfgHidden] % 512 != 0 ||
        config[CfgIntermediate] < 512 || config[CfgIntermediate] > 3072 ||
        config[CfgIntermediate] % 256 != 0 || config[CfgTopK] < 1 || config[CfgTopK] > 16 ||
        config[CfgTopK] > config[CfgFfn] * config[CfgExperts]) {
        return false;
    }
    if (config[CfgSourceCapacity] < 1 || config[CfgSourceCapacity] > MAX_TICKET / config[CfgTopK] ||
        config[CfgRecvCapacity] < 1 || config[CfgRecvCapacity] > MAX_TICKET ||
        config[CfgLayers] < 1 || config[CfgLayers] > 65536 ||
        config[CfgLayerCapacity] < config[CfgLayers] || config[CfgLayerCapacity] > 65536 ||
        config[CfgFlags] != 0 || config[CfgInflight] != 1 || config[CfgCommRounds] != 1) {
        return false;
    }
    for (uint64_t idx = 20; idx < CONFIG_FIELDS; ++idx) {
        if (config[idx] != 0) { return false; }
    }
    const uint64_t sources = config[CfgTp], experts = config[CfgExperts];
    const uint64_t routeCacheBytes = AlignUp(4 * sources * AlignUp(experts, 8), 32) +
        AlignUp(4 * sources * experts, 32) + AlignUp(4 * sources, 32);
    if (routeCacheBytes > W4_ROUTE_CACHE_BYTES) { return false; }
    return config[CfgWindowBytes] >= 0 &&
        static_cast<uint64_t>(config[CfgWindowBytes]) == MakeWindowLayout(config).requiredBytes;
}
} // namespace AfdMegaMoe
#endif
