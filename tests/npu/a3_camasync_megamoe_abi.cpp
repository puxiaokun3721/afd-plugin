// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the AFD plugin project.
// Host-only wire/layout probe. No CANN headers or NPU are required.
#include <iostream>
#include "afd_mega_moe/abi.h"

int main()
{
    int64_t config[AfdMegaMoe::CONFIG_FIELDS]{};
    while (std::cin >> config[0]) {
        for (uint64_t i = 1; i < AfdMegaMoe::CONFIG_FIELDS; ++i) {
            if (!(std::cin >> config[i])) { return 2; }
        }
        if (!AfdMegaMoe::ValidateConfig(config, AfdMegaMoe::CONFIG_FIELDS)) {
            std::cout << "invalid\n";
            continue;
        }
        const auto layout = AfdMegaMoe::MakeWindowLayout(config);
        std::cout << layout.chunkTokens << ' ' << layout.numChunks << ' '
                  << layout.paddedTokens << ' ' << layout.expertStride << ' '
                  << layout.xBase << ' ' << layout.scaleBase << ' '
                  << layout.dBase << ' ' << layout.doneBase << ' '
                  << layout.inboxStride << ' ' << layout.tailBytes << ' '
                  << layout.requiredBytes << ' '
                  << AfdMegaMoe::SendScratchBytes(config) << ' '
                  << AfdMegaMoe::FfnNativeScratchBytes(config) << ' '
                  << AfdMegaMoe::FfnScratchBytes(config) << ' '
                  << AfdMegaMoe::MaxRecvRounds(config) << '\n';
    }
    return 0;
}
