// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-2.0
// W4A8 specialization of ops-transformer mega_moe_arch22.h (8d46ac50).
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// See ../native/LICENSE. AFD changes are the receiver entry and Params wiring.
#ifndef AFD_MEGA_MOE_NATIVE_FFN_H
#define AFD_MEGA_MOE_NATIVE_FFN_H
#include "kernel_operator.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_muls.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "utils/select_helper.hpp"
#include "utils/dispatch_policy_custom.hpp"
#include "mega_moe_kernel_a3.hpp"
#include "afd_mega_moe/tiling.h"

namespace AfdMegaMoe {
using namespace Catlass;
using NativeArch = Arch::AtlasA2;
using NativeA = Gemm::GemmType<AscendC::int4b_t, layout::RowMajor>;
using NativeB = Gemm::GemmType<AscendC::int4b_t, layout::zN>;
using NativeC = Gemm::GemmType<AscendC::float16_t, layout::RowMajor>;
using NativeD1 = Gemm::GemmType<int8_t, layout::RowMajor>;
using NativeD2 = Gemm::GemmType<AscendC::bfloat16_t, layout::RowMajor>;
using NativePolicy = Gemm::MmadDispatchPolicyFor_t<AscendC::int4b_t, false, 1, 2, 2, 2, 1, false, true>;
using NativeTileCopy = Gemm::Tile::QuantTileCopy<NativeArch, NativeA, NativeB, NativeC, void,
                                               Gemm::Tile::ScaleGranularity::PER_CHANNEL>;
using NativeMmad = Gemm::Block::BlockMmad<NativePolicy, GemmShape<128, 256, 1024>, GemmShape<128, 256, 256>,
                                        NativeA, NativeB, NativeC, void, NativeTileCopy>;
using NativeScale = Gemm::GemmType<uint64_t, layout::VectorLayout>;
using NativeTokenScale = Gemm::GemmType<float, layout::VectorLayout>;
using NativeMul = Epilogue::Tile::TileElemWiseMuls<NativeArch, Gemm::GemmType<float, layout::RowMajor>, 0>;
using NativeCopy1 = Epilogue::Tile::TileCopy<NativeArch, NativeC, NativeScale, NativeTokenScale, NativeD1>;
using NativeCopy2 = Epilogue::Tile::TileCopy<NativeArch, NativeC, NativeScale, NativeTokenScale, NativeD2>;
using NativeEpilogue1 = Epilogue::Block::BlockEpilogue<
    Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantSwigluQuant<2>, NativeC, NativeTokenScale,
    NativeD1, NativeMul, NativeCopy1>;
using NativeEpilogue2 = Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2PerTokenDequant<2>,
    NativeC, NativeTokenScale, NativeD2, NativeCopy2>;
using NativeEpilogue3 = Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantV2<2>,
    NativeC, NativeTokenScale, NativeD2, NativeCopy2, Epilogue::Block::IS_A2_T<false>>;
using NativeKernel = Gemm::Kernel::MegaMoeKernel<NativeMmad, Gemm::Block::GemmIdentityBlockSwizzle<9, 1>,
    int64_t, NativeEpilogue1, NativeEpilogue2, NativeEpilogue3>;

// ### PATCH START: one selected DP, receiver-only dispatch, selected layer
__aicore__ inline void RunNativeFfn(const TilingData &tiling, uint32_t tokens, uint32_t layerSlot,
    uint32_t dp, int32_t ffn, GM_ADDR sourceDesc, GM_ADDR w13, GM_ADDR w2,
    GM_ADDR scale13, GM_ADDR scale2, GM_ADDR compensation13, GM_ADDR compensation2,
    GM_ADDR layerParams, GM_ADDR scratch)
{
    const int64_t *config = tiling.config;
    const uint32_t h = config[CfgHidden], i = config[CfgIntermediate], q = config[CfgRecvCapacity];
    const uint32_t e = config[CfgExperts], s = config[CfgTp];
    const auto layout = MakeWindowLayout(config);
    const auto parameters = reinterpret_cast<__gm__ float *>(layerParams) + layerSlot * 4;
    typename NativeKernel::Params params{
        GemmCoord{tokens, 2 * i, h}, s, e, e, q, config[CfgTopK], tiling.aivNum,
        AscendC::GetHcclContext<HCCL_GROUP_ID_0>(), nullptr,
        layout::RowMajor{tokens, h}, layout::RowMajor{tokens, i},
        w13, LayoutBInitializer<layout::zN, AscendC::int4b_t>::create(h, 2 * i), compensation13,
        w2, LayoutBInitializer<layout::zN, AscendC::int4b_t>::create(i, h), compensation2,
        scale13, layout::VectorLayout{2 * i}, scale2, layout::VectorLayout{h}, nullptr,
        layout::RowMajor{q, i}, layout::RowMajor{static_cast<uint32_t>(tokens * config[CfgTopK]), h},
        nullptr, nullptr, nullptr, nullptr, nullptr, scratch, nullptr, nullptr, nullptr,
        0, parameters[0], static_cast<uint32_t>(parameters[1]), parameters[2], parameters[3]};
    params.recvRoundsMax = MaxRecvRounds(config);
    params.afdAsync = true;
    params.afdFfnIndex = ffn;
    params.afdLayerBase = layerSlot * e;
    params.afdGlobalExperts = config[CfgFfn] * e;
    params.afdChunkTokens = layout.chunkTokens;
    params.afdSourceDesc = sourceDesc;
    params.afdCountStage = scratch + FfnNativeScratchBytes(config);
    params.afdSourcePrefixes = params.afdCountStage + AlignUp(4 * s * layout.expertStride, 512);
    params.afdXBase = layout.xBase;
    params.afdDBase = layout.dBase;
    params.afdInboxStride = layout.inboxStride;
    params.afdInboxBase = layout.reservedBytes;
    params.afdDp = dp;
    NativeKernel kernel(params);
    if ASCEND_IS_AIC { kernel(params); }
    if ASCEND_IS_AIV { kernel.ProcessAfd(params); }
}
// ### PATCH END: one selected DP, receiver-only dispatch, selected layer
} // namespace AfdMegaMoe
#endif
