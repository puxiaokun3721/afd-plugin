// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the AFD plugin project
#include "aclnn_afd_mega_moe_combine_recv.h"
#include "aclnnInner_afd_mega_moe_combine_recv.h"

// Same MTE resource provider as the existing A3 CAM operators.
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *, int32_t);
namespace { constexpr int32_t HCCL_SERVER_MTE = 1; }
extern "C" {
aclnnStatus aclnnAfdMegaMoeCombineRecvGetWorkspaceSize(
    const aclTensor *topk_weights,
    const aclTensor *token_mask,
    const aclTensor *task_meta,
    const aclTensor *route_index,
    const aclTensor *comm_args,
    const aclTensor *topology,
    const aclTensor *control,
    char *group_name,
    const aclIntArray *config,
    const aclTensor *state,
    const aclTensor *out,
    const aclTensor *completion,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerAfdMegaMoeCombineRecvGetWorkspaceSize(
        topk_weights, token_mask, task_meta, route_index, comm_args, topology, control, group_name, config, state, out, completion, workspaceSize, executor);
}
aclnnStatus aclnnAfdMegaMoeCombineRecv(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) { NnopbaseSetHcclServerType(executor, HCCL_SERVER_MTE); }
    return aclnnInnerAfdMegaMoeCombineRecv(workspace, workspaceSize, executor, stream);
}
}
