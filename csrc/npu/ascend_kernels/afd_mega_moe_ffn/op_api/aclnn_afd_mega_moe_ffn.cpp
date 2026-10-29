// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the AFD plugin project
#include "aclnn_afd_mega_moe_ffn.h"
#include "aclnnInner_afd_mega_moe_ffn.h"

// Same MTE resource provider as the existing A3 CAM operators.
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *, int32_t);
namespace { constexpr int32_t HCCL_SERVER_MTE = 1; }
extern "C" {
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
    aclOpExecutor **executor)
{
    return aclnnInnerAfdMegaMoeFfnGetWorkspaceSize(
        work_desc, source_desc, layer_id_to_slot, w13, w2, weight_scale13, weight_scale2, compensation13, compensation2, layer_params, comm_args, topology, control, group_name, config, state, ffn_scratch, completion, workspaceSize, executor);
}
aclnnStatus aclnnAfdMegaMoeFfn(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) { NnopbaseSetHcclServerType(executor, HCCL_SERVER_MTE); }
    return aclnnInnerAfdMegaMoeFfn(workspace, workspaceSize, executor, stream);
}
}
