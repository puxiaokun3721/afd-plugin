// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the AFD plugin project
#include "aclnn_afd_mega_moe_scheduler.h"
#include "aclnnInner_afd_mega_moe_scheduler.h"

// Same MTE resource provider as the existing A3 CAM operators.
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *, int32_t);
namespace { constexpr int32_t HCCL_SERVER_MTE = 1; }
extern "C" {
aclnnStatus aclnnAfdMegaMoeSchedulerGetWorkspaceSize(
    const aclTensor *comm_args,
    const aclTensor *topology,
    const aclTensor *control,
    char *group_name,
    const aclIntArray *config,
    const aclTensor *state,
    const aclTensor *work_desc,
    const aclTensor *source_desc,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerAfdMegaMoeSchedulerGetWorkspaceSize(
        comm_args, topology, control, group_name, config, state, work_desc, source_desc, workspaceSize, executor);
}
aclnnStatus aclnnAfdMegaMoeScheduler(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) { NnopbaseSetHcclServerType(executor, HCCL_SERVER_MTE); }
    return aclnnInnerAfdMegaMoeScheduler(workspace, workspaceSize, executor, stream);
}
}
