// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the AFD plugin project
#include "afd_mega_moe_host.h"

namespace ops {
class AfdMegaMoeFfn : public OpDef {
public:
    explicit AfdMegaMoeFfn(const char *name) : OpDef(name)
    {
        this->Input("work_desc").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("source_desc").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("layer_id_to_slot").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w13").ParamType(DYNAMIC)
            .DataType({ge::DT_INT4}).Format({ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
        this->Input("w2").ParamType(DYNAMIC)
            .DataType({ge::DT_INT4}).Format({ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
        this->Input("weight_scale13").ParamType(DYNAMIC)
            .DataType({ge::DT_UINT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight_scale2").ParamType(DYNAMIC)
            .DataType({ge::DT_UINT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("compensation13").ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("compensation2").ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("layer_params").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
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
        this->Output("ffn_scratch").ParamType(REQUIRED)
            .DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("completion").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("group_name").String();
        this->Attr("config").ListInt();
        this->AICore().AddConfig("ascend910_93");
        this->MC2().HcclGroup({"group_name"});
        this->SetInferShape(AfdMegaMoe::Host::InferShape<AfdMegaMoe::Host::Operation::Ffn>)
            .SetInferDataType(AfdMegaMoe::Host::InferDataType<AfdMegaMoe::Host::Operation::Ffn>);
        this->AICore().SetTiling(AfdMegaMoe::Host::Tiling<AfdMegaMoe::Host::Operation::Ffn>);
    }
};
OP_ADD(AfdMegaMoeFfn);
}  // namespace ops
