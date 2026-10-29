// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the AFD plugin project
#include "afd_mega_moe_host.h"

namespace ops {
class AfdMegaMoeCombineRecv : public OpDef {
public:
    explicit AfdMegaMoeCombineRecv(const char *name) : OpDef(name)
    {
        this->Input("topk_weights").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("token_mask").ParamType(REQUIRED)
            .DataType({ge::DT_BOOL}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("task_meta").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("route_index").ParamType(REQUIRED)
            .DataType({ge::DT_INT32}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("comm_args").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("topology").ParamType(REQUIRED)
            .DataType({ge::DT_INT32}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("control").ParamType(REQUIRED)
            .DataType({ge::DT_INT32}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("state").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out").ParamType(REQUIRED)
            .DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("completion").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("group_name").String();
        this->Attr("config").ListInt();
        this->AICore().AddConfig("ascend910_93");
        this->MC2().HcclGroup({"group_name"});
        this->SetInferShape(AfdMegaMoe::Host::InferShape<AfdMegaMoe::Host::Operation::CombineRecv>)
            .SetInferDataType(AfdMegaMoe::Host::InferDataType<AfdMegaMoe::Host::Operation::CombineRecv>);
        this->AICore().SetTiling(AfdMegaMoe::Host::Tiling<AfdMegaMoe::Host::Operation::CombineRecv>);
    }
};
OP_ADD(AfdMegaMoeCombineRecv);
}  // namespace ops
