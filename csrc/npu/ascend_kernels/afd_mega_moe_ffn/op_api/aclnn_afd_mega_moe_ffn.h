// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the AFD plugin project
#ifndef ACLNN_AFD_MEGA_MOE_FFN_H
#define ACLNN_AFD_MEGA_MOE_FFN_H
#include "aclnn/acl_meta.h"
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default")))
aclnnStatus aclnnAfdMegaMoeFfnGetWorkspaceSize(
    const aclTensor *work_desc,
    const aclTensor *source_desc,
    const aclTensor *layer_id_to_slot,
    const aclTensorList *w13,
    const aclTensorList *w2,
    const aclTensorList *weight_scale13,
    const aclTensorList *weight_scale2,
    const aclTensorList *compensation13,
    const aclTensorList *compensation2,
    const aclTensor *layer_params,
    const aclTensor *comm_args,
    const aclTensor *topology,
    const aclTensor *control,
    char *group_name,
    const aclIntArray *config,
    const aclTensor *state,
    const aclTensor *ffn_scratch,
    const aclTensor *completion,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
__attribute__((visibility("default")))
aclnnStatus aclnnAfdMegaMoeFfn(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif
