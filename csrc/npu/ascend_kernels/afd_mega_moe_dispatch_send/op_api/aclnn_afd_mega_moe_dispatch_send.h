// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the AFD plugin project
#ifndef ACLNN_AFD_MEGA_MOE_DISPATCH_SEND_H
#define ACLNN_AFD_MEGA_MOE_DISPATCH_SEND_H
#include "aclnn/acl_meta.h"
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default")))
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
    aclOpExecutor **executor);
__attribute__((visibility("default")))
aclnnStatus aclnnAfdMegaMoeDispatchSend(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif
