// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the AFD plugin project
#ifndef AFD_MEGA_MOE_HOST_H
#define AFD_MEGA_MOE_HOST_H

#include <algorithm>
#include <initializer_list>
#include <string>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/hccl/hccl_tiling.h"
#include "ops_log.h"
#include "afd_mega_moe/tiling.h"

namespace AfdMegaMoe::Host {
enum class Operation { DispatchSend, Scheduler, Ffn, CombineRecv };
constexpr size_t GROUP_ATTRIBUTE = 0;
constexpr size_t CONFIG_ATTRIBUTE = 1;
constexpr uint32_t ALL_TO_ALL_OP_TYPE = 8;
constexpr uint64_t RUNTIME_WORKSPACE_BYTES = 16ULL * 1024 * 1024;

template <typename Context>
inline const int64_t *ReadConfig(Context *context)
{
    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) { return nullptr; }
    const auto *values = attrs->template GetAttrPointer<gert::ContinuousVector>(CONFIG_ATTRIBUTE);
    if (values == nullptr || values->GetSize() != CONFIG_FIELDS) { return nullptr; }
    const auto *config = reinterpret_cast<const int64_t *>(values->GetData());
    return ValidateConfig(config, values->GetSize()) ? config : nullptr;
}

inline bool ShapeEquals(const gert::Shape &shape, std::initializer_list<int64_t> expected)
{
    if (shape.GetDimNum() != expected.size()) { return false; }
    size_t index = 0;
    for (int64_t dim : expected) {
        if (shape.GetDim(index++) != dim) { return false; }
    }
    return true;
}

inline bool CheckInput(gert::TilingContext *context, size_t index,
                       std::initializer_list<int64_t> shape, size_t element = 0)
{
    const auto *tensor = context->GetDynamicInputTensor(index, element);
    if (tensor == nullptr || !ShapeEquals(tensor->GetOriginShape(), shape)) {
        OPS_LOG_E(context->GetNodeName(), "input %zu element %zu has invalid shape", index, element);
        return false;
    }
    return true;
}

inline bool ValidateInputs(gert::TilingContext *context, Operation operation, const int64_t *c)
{
    const int64_t world = c[CfgDp] * c[CfgTp] + c[CfgFfn];
    const size_t comm = operation == Operation::Scheduler ? 0 : (operation == Operation::Ffn ? 10 : 4);
    if (!CheckInput(context, comm, {1}) || !CheckInput(context, comm + 1, {world, 4}) ||
        !CheckInput(context, comm + 2, {CONTROL_BYTES / sizeof(int32_t)})) { return false; }
    if (operation == Operation::Scheduler) { return true; }
    if (operation == Operation::Ffn) {
        if (!CheckInput(context, 0, {WORK_FIELDS}) || !CheckInput(context, 1, {c[CfgTp], TASK_FIELDS}) ||
            !CheckInput(context, 2, {c[CfgLayerCapacity]}) || !CheckInput(context, 9, {c[CfgLayers], 4})) {
            return false;
        }
        const int64_t lists = c[CfgLayers] * c[CfgExperts];
        for (int64_t i = 0; i < lists; ++i) {
            if (!CheckInput(context, 3, {c[CfgHidden], 2 * c[CfgIntermediate]}, i) ||
                !CheckInput(context, 4, {c[CfgIntermediate], c[CfgHidden]}, i) ||
                !CheckInput(context, 5, {2 * c[CfgIntermediate]}, i) ||
                !CheckInput(context, 6, {c[CfgHidden]}, i) ||
                !CheckInput(context, 7, {2 * c[CfgIntermediate]}, i) ||
                !CheckInput(context, 8, {c[CfgHidden]}, i)) { return false; }
        }
        for (size_t input = 3; input <= 8; ++input) {
            if (context->GetDynamicInputTensor(input, lists) != nullptr) { return false; }
        }
        return true;
    }
    const auto *first = context->GetInputShape(0);
    if (first == nullptr || first->GetOriginShape().GetDimNum() != 2) { return false; }
    const int64_t tokens = first->GetOriginShape().GetDim(0);
    if (tokens < 0 || tokens > c[CfgSourceCapacity]) { return false; }
    if (operation == Operation::DispatchSend) {
        return CheckInput(context, 0, {tokens, c[CfgHidden]}) &&
            CheckInput(context, 1, {tokens, c[CfgTopK]}) && CheckInput(context, 2, {tokens}) &&
            CheckInput(context, 3, {TASK_FIELDS});
    }
    const auto layout = MakeWindowLayout(c);
    return CheckInput(context, 0, {tokens, c[CfgTopK]}) && CheckInput(context, 1, {tokens}) &&
        CheckInput(context, 2, {TASK_FIELDS}) &&
        CheckInput(context, 3, {static_cast<int64_t>(layout.paddedTokens * c[CfgTopK])});
}

inline void SetShape(gert::Shape *shape, std::initializer_list<int64_t> dimensions)
{
    shape->SetDimNum(dimensions.size());
    size_t index = 0;
    for (int64_t dim : dimensions) { shape->SetDim(index++, dim); }
}

template <Operation OP>
inline ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *c = ReadConfig(context);
    if (c == nullptr) { return ge::GRAPH_FAILED; }
    const auto layout = MakeWindowLayout(c);
    SetShape(context->GetOutputShape(0), {static_cast<int64_t>(STATE_HEADER_FIELDS) + c[CfgDp]});
    if constexpr (OP == Operation::DispatchSend) {
        SetShape(context->GetOutputShape(1), {static_cast<int64_t>(layout.paddedTokens * c[CfgTopK])});
        SetShape(context->GetOutputShape(2), {static_cast<int64_t>(layout.numChunks),
                                            static_cast<int64_t>(layout.expertStride)});
        SetShape(context->GetOutputShape(3), {static_cast<int64_t>(SendScratchBytes(c))});
    } else if constexpr (OP == Operation::Scheduler) {
        SetShape(context->GetOutputShape(1), {WORK_FIELDS});
        SetShape(context->GetOutputShape(2), {c[CfgTp], TASK_FIELDS});
    } else if constexpr (OP == Operation::Ffn) {
        SetShape(context->GetOutputShape(1), {static_cast<int64_t>(FfnScratchBytes(c))});
        SetShape(context->GetOutputShape(2), {COMPLETION_FIELDS});
    } else {
        const auto *weights = context->GetInputShape(0);
        if (weights == nullptr || weights->GetDimNum() != 2) { return ge::GRAPH_FAILED; }
        SetShape(context->GetOutputShape(1), {weights->GetDim(0), c[CfgHidden]});
        SetShape(context->GetOutputShape(2), {COMPLETION_FIELDS});
    }
    return ge::GRAPH_SUCCESS;
}

template <Operation OP>
inline ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT64);
    if constexpr (OP == Operation::DispatchSend) {
        context->SetOutputDataType(1, ge::DT_INT32);
        context->SetOutputDataType(2, ge::DT_INT32);
        context->SetOutputDataType(3, ge::DT_UINT8);
    } else {
        context->SetOutputDataType(1, OP == Operation::Ffn ? ge::DT_UINT8 :
                                     (OP == Operation::CombineRecv ? ge::DT_BF16 : ge::DT_INT64));
        context->SetOutputDataType(2, ge::DT_INT64);
    }
    return ge::GRAPH_SUCCESS;
}

template <Operation OP>
inline ge::graphStatus Tiling(gert::TilingContext *context)
{
    const auto *config = ReadConfig(context);
    if (config == nullptr || !ValidateInputs(context, OP, config)) {
        OPS_LOG_E(context->GetNodeName(), "invalid A3 MegaMoE v1 config or tensor shapes");
        return ge::GRAPH_FAILED;
    }
    auto *tiling = context->GetTilingData<TilingData>();
    if (tiling == nullptr) { return ge::GRAPH_FAILED; }
    *tiling = TilingData{};
    std::copy_n(config, CONFIG_FIELDS, tiling->config);
    const auto *group = context->GetAttrs()->GetAttrPointer<char>(GROUP_ATTRIBUTE);
    if (group == nullptr || group[0] == '\0') { return ge::GRAPH_FAILED; }
    AscendC::Mc2CcTilingConfig mc2(group, ALL_TO_ALL_OP_TYPE,
                                "AlltoAll=level0:fullmesh;level1:pairwise");
    mc2.GetTiling(tiling->mc2InitTiling);
    mc2.GetTiling(tiling->mc2CcTiling1);
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    tiling->aicNum = platform.GetCoreNumAic();
    tiling->aivNum = platform.GetCoreNumAiv();
    if constexpr (OP == Operation::Scheduler) {
        tiling->aivNum = 1;
        context->SetBlockDim(platform.CalcTschBlockDim(1, 0, 1));
    } else if constexpr (OP == Operation::Ffn) {
        if (tiling->aicNum == 0 || tiling->aivNum != 2 * tiling->aicNum) {
            OPS_LOG_E(context->GetNodeName(), "MegaMoE requires A3 mixed AIC:AIV = 1:2");
            return ge::GRAPH_FAILED;
        }
        context->SetBlockDim(tiling->aicNum);
        context->SetScheduleMode(1);
    } else {
        if (tiling->aivNum == 0 || tiling->aivNum > PERMUTE_CORES) { return ge::GRAPH_FAILED; }
        context->SetBlockDim(platform.CalcTschBlockDim(tiling->aivNum, 0, tiling->aivNum));
        context->SetScheduleMode(1);
        tiling->inputTokens = context->GetInputShape(0)->GetOriginShape().GetDim(0);
    }
    tiling->sendScratchBytes = SendScratchBytes(config);
    tiling->ffnScratchBytes = FfnScratchBytes(config);
    context->SetTilingKey(1);
    auto *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) { return ge::GRAPH_FAILED; }
    workspace[0] = RUNTIME_WORKSPACE_BYTES;
    return ge::GRAPH_SUCCESS;
}
}  // namespace AfdMegaMoe::Host
#endif
