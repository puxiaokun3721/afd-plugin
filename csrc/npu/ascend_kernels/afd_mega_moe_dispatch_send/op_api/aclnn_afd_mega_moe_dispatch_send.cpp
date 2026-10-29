// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the AFD plugin project
#include "aclnn_afd_mega_moe_dispatch_send.h"
#include "aclnnInner_afd_mega_moe_dispatch_send.h"

// Same MTE resource provider as the existing A3 CAM operators.
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *, int32_t);
namespace { constexpr int32_t HCCL_SERVER_MTE = 1; }
extern "C" {
aclnnStatus aclnnAfdMegaMoeDispatchSendGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *expert_ids,
    const aclTensor *token_mask,
    const aclTensor *task_meta,
    const aclTensor *comm_args,
    const aclTensor *topology,
    const aclTensor *control,
    char *group_name,
    const aclIntArray *config,
    const aclTensor *state,
    const aclTensor *route_index,
    const aclTensor *source_counts,
    const aclTensor *send_scratch,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerAfdMegaMoeDispatchSendGetWorkspaceSize(
        x, expert_ids, token_mask, task_meta, comm_args, topology, control, group_name, config, state, route_index, source_counts, send_scratch, workspaceSize, executor);
}
aclnnStatus aclnnAfdMegaMoeDispatchSend(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) { NnopbaseSetHcclServerType(executor, HCCL_SERVER_MTE); }
    return aclnnInnerAfdMegaMoeDispatchSend(workspace, workspaceSize, executor, stream);
}
}
