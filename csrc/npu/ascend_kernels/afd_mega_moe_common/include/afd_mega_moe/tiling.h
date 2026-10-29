// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the AFD plugin project
#ifndef AFD_MEGA_MOE_TILING_H
#define AFD_MEGA_MOE_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "afd_mega_moe/abi.h"

namespace AfdMegaMoe {
// One fixed, versioned POD for all four entries. Config is host metadata;
// task identity and selected layer remain device inputs, never tiling attrs.
struct TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    int64_t config[24];
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t inputTokens;
    uint64_t sendScratchBytes;
    uint64_t ffnScratchBytes;
};
}  // namespace AfdMegaMoe
#endif
