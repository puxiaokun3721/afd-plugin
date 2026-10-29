// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the AFD plugin project
#include "afd_mega_moe_host.h"

namespace ops {
class AfdMegaMoeScheduler : public OpDef {
public:
    explicit AfdMegaMoeScheduler(const char *name) : OpDef(name)
    {
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
        this->Output("work_desc").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("source_desc").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("group_name").String();
        this->Attr("config").ListInt();
        this->AICore().AddConfig("ascend910_93");
        this->MC2().HcclGroup({"group_name"});
        this->SetInferShape(AfdMegaMoe::Host::InferShape<AfdMegaMoe::Host::Operation::Scheduler>)
            .SetInferDataType(AfdMegaMoe::Host::InferDataType<AfdMegaMoe::Host::Operation::Scheduler>);
        this->AICore().SetTiling(AfdMegaMoe::Host::Tiling<AfdMegaMoe::Host::Operation::Scheduler>);
    }
};
OP_ADD(AfdMegaMoeScheduler);
}  // namespace ops
