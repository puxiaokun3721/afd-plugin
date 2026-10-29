// SPDX-License-Identifier: Apache-2.0
// Included inside the vendored MegaMoeKernel class. All compute, MSD, GMM
// scheduling and the fused remote-store epilogue remain the native methods.
public:
    CATLASS_DEVICE void ProcessAfd(Params const &params)
    {
        const int64_t chunks = NumChunks(params.problemShape.m());
        for (int64_t tile = 0; tile < chunks; ++tile) {
            cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ChunkCumsumPtr(params, tile)));
            chunkCumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ChunkCumsumPtr(params, tile)));
            preSumBeforeRank.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspaceInfo.ptrSumBeforeRank));
            peermemInfo.offsetA = params.afdXBase + tile * static_cast<uint64_t>(params.afdChunkTokens) *
                params.topK * (params.problemShape.k() + ALIGN_512);
            peermemInfo.offsetD = params.afdDBase + tile * static_cast<uint64_t>(params.afdChunkTokens) *
                params.topK * params.problemShape.k() * sizeof(ElementD2);

            // Inbox rows contain the full global-E count vector. Preserve the
            // source-local prefix before taking this F's expert columns; hot
            // expert subrounds must not use a merged receive-row offset on A.
            auto counts = resource.ubBuf.template GetBufferByByte<int32_t>(0);
            auto prefixes = resource.ubBuf.template GetBufferByByte<int32_t>(
                AlignUp(paddedExpertNumAligned * sizeof(int32_t), BYTE_PER_BLK));
            for (uint32_t source = coreIdx; source < params.EP; source += coreNum) {
                AscendC::GlobalTensor<int32_t> inbox;
                inbox.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(shmem() + params.afdInboxBase +
                    (static_cast<uint64_t>(params.afdDp) * params.EP + source) * params.afdInboxStride +
                    1024 + tile * static_cast<uint64_t>(paddedExpertNumAligned) * sizeof(int32_t)));
                AscendC::DataCopyPad(counts, inbox,
                    {1, static_cast<uint16_t>(paddedExpertNumAligned * sizeof(int32_t)), 0, 0}, {});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                int32_t prefix = 0;
                const uint32_t firstExpert = params.afdFfnIndex * params.expertPerRank;
                for (uint32_t expert = 0; expert < firstExpert; ++expert) { prefix += counts.GetValue(expert); }
                for (uint32_t expert = 0; expert < params.expertPerRank; ++expert) {
                    prefixes.SetValue(expert, prefix);
                    prefix += counts.GetValue(firstExpert + expert);
                }
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::DataCopyPad(tokenPerExpert[source * paddedExpertNumAligned], counts,
                    {1, static_cast<uint16_t>(paddedExpertNumAligned * sizeof(int32_t)), 0, 0});
                AscendC::DataCopyPad(preSumBeforeRank[source * params.expertPerRank], prefixes,
                    {1, static_cast<uint16_t>(params.expertPerRank * sizeof(int32_t)), 0, 0});
                AscendC::GlobalTensor<int32_t> immutablePrefixes;
                immutablePrefixes.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.afdSourcePrefixes) +
                    (tile * params.EP + source) * params.expertPerRank);
                AscendC::DataCopyPad(immutablePrefixes, prefixes,
                    {1, static_cast<uint16_t>(params.expertPerRank * sizeof(int32_t)), 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<true>();
            if (coreIdx == 0) {
                GetCumsumForMMAIV(tokenPerExpert, cumsumMM, params.expertPerRank, params.afdFfnIndex, params.EP);
            }
            AscendC::SyncAll<true>();
            CacheRouteTablesToUb(params);
            const int64_t rounds = ComputeNumRecvRounds(params);
            for (int64_t round = 0; round < rounds; ++round) {
                if (rounds > 1) {
                    BuildRoundTables(params, tile, round);
                    AscendC::SyncAll<true>();
                    SetRoundTableBuffers(params, tile, round);
                    CacheRoundTablesToUb(params);
                }
                ProcessRecvRound(params, tile);
            }
        }
    }
