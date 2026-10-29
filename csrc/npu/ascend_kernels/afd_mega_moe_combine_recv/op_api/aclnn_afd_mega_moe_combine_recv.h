// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the AFD plugin project
#ifndef ACLNN_AFD_MEGA_MOE_COMBINE_RECV_H
#define ACLNN_AFD_MEGA_MOE_COMBINE_RECV_H
#include "aclnn/acl_meta.h"
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default")))
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
    aclOpExecutor **executor);
__attribute__((visibility("default")))
aclnnStatus aclnnAfdMegaMoeCombineRecv(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif
