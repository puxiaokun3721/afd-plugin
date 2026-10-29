// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the AFD plugin project
#ifndef ACLNN_AFD_MEGA_MOE_SCHEDULER_H
#define ACLNN_AFD_MEGA_MOE_SCHEDULER_H
#include "aclnn/acl_meta.h"
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default")))
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
    aclOpExecutor **executor);
__attribute__((visibility("default")))
aclnnStatus aclnnAfdMegaMoeScheduler(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif
