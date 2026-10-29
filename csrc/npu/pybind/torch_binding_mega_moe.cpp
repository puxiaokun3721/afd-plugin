// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the AFD plugin project
// A3 prefill adapters. Control buffers are explicit mutable arguments; no host
// reads of task identity, selected layer, completion, or communication memory.
#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_extension/op_api_common.h"
#include "afd_mega_moe/abi.h"
#include "afd_mega_moe_dispatch_send/op_api/aclnn_afd_mega_moe_dispatch_send.h"
#include "afd_mega_moe_scheduler/op_api/aclnn_afd_mega_moe_scheduler.h"
#include "afd_mega_moe_ffn/op_api/aclnn_afd_mega_moe_ffn.h"
#include "afd_mega_moe_combine_recv/op_api/aclnn_afd_mega_moe_combine_recv.h"

namespace afd_plugin::mega_moe {
using namespace AfdMegaMoe;

// The .26 W4A8 loader stores packed INT4 NZ bytes in INT32 carriers and stores
// hardware dequant scale bits in INT64 carriers. Never send these lists through
// the general ConvertType(TensorList), which would register INT32 / INT64.
struct TypedTensorList {
  at::TensorList tensors;
  aclDataType dtype;
};

aclTensorList *ConvertType(const TypedTensorList &list) {
  static const auto create_tensor = GET_OP_API_FUNC(aclCreateTensor);
  static const auto create_list = GET_OP_API_FUNC(aclCreateTensorList);
  TORCH_CHECK(create_tensor && create_list, "ACL tensor constructors unavailable");
  std::vector<const aclTensor *> tensors;
  tensors.reserve(list.tensors.size());
  for (const auto &tensor : list.tensors) {
    aclTensor *converted = nullptr;
    if (list.dtype == ACL_INT4) {
      const int64_t rows = tensor.size(0);
      const int64_t cols = tensor.size(1) * 8;
      const std::array<int64_t, 2> logical{rows, cols};
      const std::array<int64_t, 2> strides{cols, 1};
      // NZ storage dimensions are in INT4 elements (64 nibbles per 32B).
      const std::array<int64_t, 4> storage{cols / 64, rows / 16, 16, 64};
      converted = create_tensor(logical.data(), logical.size(), ACL_INT4,
          strides.data(), 0, ACL_FORMAT_FRACTAL_NZ, storage.data(),
          storage.size(), tensor.data_ptr());
    } else {
      const int64_t storage = tensor.numel();
      converted = create_tensor(tensor.sizes().data(), tensor.dim(), ACL_UINT64,
          tensor.strides().data(), 0, ACL_FORMAT_ND, &storage, 1,
          tensor.data_ptr());
    }
    TORCH_CHECK(converted, "cannot construct MegaMoE typed tensor descriptor");
    tensors.push_back(converted);
  }
  auto *result = create_list(tensors.data(), tensors.size());
  TORCH_CHECK(result, "cannot construct MegaMoE typed tensor list");
  return result;
}

namespace {
constexpr int64_t INT4_PER_INT32 = 8;
constexpr int64_t NZ_ROWS = 16;
constexpr int64_t NZ_INT4_COLUMNS = 64;

void check_tensor(const at::Tensor &value, const at::Tensor &anchor,
                  at::ScalarType dtype, std::initializer_list<int64_t> shape,
                  const char *name, bool raw_nz = false) {
  TORCH_CHECK(value.defined(), name, " must be defined");
  TORCH_CHECK(value.device() == anchor.device(), name, " must be on the same device");
  TORCH_CHECK(!value.requires_grad(), name, ": MegaMoE supports inference only");
  TORCH_CHECK(value.scalar_type() == dtype, name, " has incorrect dtype");
  TORCH_CHECK(value.sizes() == at::IntArrayRef(shape), name, " has incorrect shape");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(raw_nz || value.is_meta() || IsOpInputBaseFormat(value), name,
              " must use a base storage format");
  if (raw_nz && !value.is_meta()) {
    const auto &desc = vllm_ascend::NPUBridge::GetNpuStorageImplDesc(value);
    const auto &storage = desc.storage_sizes_;
    const size_t rank = storage.size();
    const int64_t rows = value.size(0), cols = value.size(1) * INT4_PER_INT32;
    TORCH_CHECK((desc.data_type_.itemsize() == 1 || desc.data_type_.itemsize() == 4) &&
        desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ && rank >= 4 &&
        storage[rank - 4] == cols / NZ_INT4_COLUMNS &&
        storage[rank - 3] == rows / NZ_ROWS && storage[rank - 2] == NZ_ROWS &&
        storage[rank - 1] * static_cast<int64_t>(desc.data_type_.itemsize()) == 32,
        name, " must carry raw INT4 NZ bytes; use raw_extract_nz_experts at load time");
    const uint64_t last_byte = static_cast<uint64_t>(value.storage_offset() + value.numel()) * sizeof(int32_t);
    TORCH_CHECK(value.storage_offset() % value.numel() == 0 &&
        last_byte <= value.storage().nbytes(), name, " has invalid raw expert slab bounds");
  }
}

void check_common(const at::Tensor &comm, const at::Tensor &topology,
                  const at::Tensor &control, const at::Tensor &state,
                  c10::string_view group, at::IntArrayRef config) {
  TORCH_CHECK(ValidateConfig(config.data(), config.size()),
              "invalid A3 MegaMoE v1 config (geometry, capacities, UB or communication window)");
  TORCH_CHECK(!group.empty(), "group_name must not be empty");
  TORCH_CHECK(comm.is_privateuseone() || comm.is_meta(), "MegaMoE requires an NPU tensor");
  const int64_t world = config[CfgDp] * config[CfgTp] + config[CfgFfn];
  check_tensor(comm, comm, at::kHalf, {1}, "comm_args");
  check_tensor(topology, comm, at::kInt, {world, 4}, "topology");
  check_tensor(control, comm, at::kInt, {CONTROL_BYTES / sizeof(int32_t)}, "control");
  check_tensor(state, comm, at::kLong,
               {static_cast<int64_t>(STATE_HEADER_FIELDS) + config[CfgDp]}, "state");
}

// Equivalent to the repository's EXEC_NPU_CMD path, with workspace and input
// owners captured by the launch closure. The asynchronous OpCommand must never
// observe a released workspace or temporary TensorList carrier.
template <typename... Args>
void launch(const char *name, std::vector<at::Tensor> owners, Args &...args) {
  const std::string workspace_name = std::string(name) + "GetWorkspaceSize";
  const auto workspace_address = GetOpApiFuncAddr(workspace_name.c_str());
  const auto operator_address = GetOpApiFuncAddr(name);
  TORCH_CHECK(workspace_address && operator_address, name,
              " is absent from the installed AFD custom operator library");
  auto init_memory = reinterpret_cast<InitHugeMemThreadLocal>(GetOpApiFuncAddr("InitHugeMemThreadLocal"));
  auto uninit_memory = reinterpret_cast<UnInitHugeMemThreadLocal>(GetOpApiFuncAddr("UnInitHugeMemThreadLocal"));
  auto release_memory = reinterpret_cast<ReleaseHugeMem>(GetOpApiFuncAddr("ReleaseHugeMem"));
  if (init_memory) { init_memory(nullptr, false); }
  uint64_t workspace_size = 0;
  uint64_t *size_pointer = &workspace_size;
  aclOpExecutor *executor = nullptr;
  aclOpExecutor **executor_pointer = &executor;
  auto converted = ConvertTypes(args..., size_pointer, executor_pointer);
  auto workspace_function = ConvertToOpApiFunc(converted, workspace_address);
  const auto status = call(workspace_function, converted);
  if (status != 0) {
    ReleaseConvertTypes(converted);
    if (release_memory) { release_memory(nullptr, false); }
    if (uninit_memory) { uninit_memory(nullptr, false); }
    TORCH_CHECK(false, name, " GetWorkspaceSize failed: ", aclGetRecentErrMsg());
  }
  at::Tensor workspace;
  if (workspace_size != 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)},
                          owners.front().options().dtype(at::kByte));
  }
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  auto handler = [converted, workspace, workspace_size, stream, executor,
                  operator_address, release_memory, owners = std::move(owners)]() mutable -> int {
    using Operator = int (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    const auto operation = reinterpret_cast<Operator>(operator_address);
    const int result = operation(workspace.defined() ? workspace.data_ptr() : nullptr,
                                 workspace_size, executor, stream);
    ReleaseConvertTypes(converted);
    if (release_memory) { release_memory(nullptr, false); }
    TORCH_CHECK(result == 0, "MegaMoE device launch failed: ", aclGetRecentErrMsg());
    return result;
  };
  at_npu::native::OpCommand command;
  command.Name(name);
  command.SetCustomHandler(handler);
  command.Run();
  if (uninit_memory) { uninit_memory(nullptr, false); }
}

std::vector<char> group_chars(c10::string_view group) {
  std::vector<char> result(group.begin(), group.end());
  result.push_back('\0');
  return result;
}
}  // namespace

template <bool EXECUTE>
void dispatch_send(const at::Tensor &x, const at::Tensor &expert_ids,
    const at::Tensor &token_mask, const at::Tensor &task_meta,
    const at::Tensor &comm_args, const at::Tensor &topology, const at::Tensor &control,
    at::Tensor state, at::Tensor route_index, at::Tensor source_counts,
    at::Tensor send_scratch, c10::string_view group_name, at::IntArrayRef config) {
  check_common(comm_args, topology, control, state, group_name, config);
  TORCH_CHECK(x.dim() == 2 && x.size(0) <= config[CfgSourceCapacity], "x token capacity exceeded");
  const int64_t tokens = x.size(0);
  const auto layout = MakeWindowLayout(config.data());
  check_tensor(x, comm_args, at::kBFloat16, {tokens, config[CfgHidden]}, "x");
  check_tensor(expert_ids, comm_args, at::kInt, {tokens, config[CfgTopK]}, "expert_ids");
  check_tensor(token_mask, comm_args, at::kBool, {tokens}, "token_mask");
  check_tensor(task_meta, comm_args, at::kLong, {TASK_FIELDS}, "task_meta");
  check_tensor(route_index, comm_args, at::kInt,
               {static_cast<int64_t>(layout.paddedTokens * config[CfgTopK])}, "route_index");
  check_tensor(source_counts, comm_args, at::kInt,
               {static_cast<int64_t>(layout.numChunks), static_cast<int64_t>(layout.expertStride)}, "source_counts");
  check_tensor(send_scratch, comm_args, at::kByte,
               {static_cast<int64_t>(SendScratchBytes(config.data()))}, "send_scratch");
  if constexpr (EXECUTE) {
    auto chars = group_chars(group_name);
    char *group = chars.data();
    launch("aclnnAfdMegaMoeDispatchSend",
      {x, expert_ids, token_mask, task_meta, comm_args, topology, control,
       state, route_index, source_counts, send_scratch},
      x, expert_ids, token_mask, task_meta, comm_args, topology, control,
      group, config, state, route_index, source_counts, send_scratch);
  }
}

template <bool EXECUTE>
void scheduler(const at::Tensor &comm_args, const at::Tensor &topology,
    const at::Tensor &control, at::Tensor state, at::Tensor work_desc,
    at::Tensor source_desc, c10::string_view group_name, at::IntArrayRef config) {
  check_common(comm_args, topology, control, state, group_name, config);
  check_tensor(work_desc, comm_args, at::kLong, {WORK_FIELDS}, "work_desc");
  check_tensor(source_desc, comm_args, at::kLong, {config[CfgTp], TASK_FIELDS}, "source_desc");
  if constexpr (EXECUTE) {
    auto chars = group_chars(group_name);
    char *group = chars.data();
    launch("aclnnAfdMegaMoeScheduler",
      {comm_args, topology, control, state, work_desc, source_desc},
      comm_args, topology, control, group, config, state, work_desc, source_desc);
  }
}

template <bool EXECUTE>
void ffn(const at::Tensor &work_desc, const at::Tensor &source_desc,
    const at::Tensor &layer_id_to_slot, at::TensorList w13, at::TensorList w2,
    at::TensorList weight_scale13, at::TensorList weight_scale2,
    at::TensorList compensation13, at::TensorList compensation2,
    const at::Tensor &layer_params, const at::Tensor &comm_args,
    const at::Tensor &topology, const at::Tensor &control,
    at::Tensor state, at::Tensor ffn_scratch, at::Tensor completion,
    c10::string_view group_name, at::IntArrayRef config) {
  check_common(comm_args, topology, control, state, group_name, config);
  check_tensor(work_desc, comm_args, at::kLong, {WORK_FIELDS}, "work_desc");
  check_tensor(source_desc, comm_args, at::kLong, {config[CfgTp], TASK_FIELDS}, "source_desc");
  check_tensor(layer_id_to_slot, comm_args, at::kLong, {config[CfgLayerCapacity]}, "layer_id_to_slot");
  check_tensor(layer_params, comm_args, at::kFloat, {config[CfgLayers], 4}, "layer_params");
  check_tensor(ffn_scratch, comm_args, at::kByte,
               {static_cast<int64_t>(FfnScratchBytes(config.data()))}, "ffn_scratch");
  check_tensor(completion, comm_args, at::kLong, {COMPLETION_FIELDS}, "completion");
  const int64_t count = config[CfgLayers] * config[CfgExperts];
  TORCH_CHECK(w13.size() == count && w2.size() == count && weight_scale13.size() == count &&
      weight_scale2.size() == count && compensation13.size() == count && compensation2.size() == count,
      "all six TensorLists must contain L * local_experts entries in layer-major order");
  for (int64_t i = 0; i < count; ++i) {
    check_tensor(w13[i], comm_args, at::kInt, {config[CfgHidden], 2 * config[CfgIntermediate] / INT4_PER_INT32}, "w13", true);
    check_tensor(w2[i], comm_args, at::kInt, {config[CfgIntermediate], config[CfgHidden] / INT4_PER_INT32}, "w2", true);
    check_tensor(weight_scale13[i], comm_args, at::kLong, {2 * config[CfgIntermediate]}, "weight_scale13");
    check_tensor(weight_scale2[i], comm_args, at::kLong, {config[CfgHidden]}, "weight_scale2");
    check_tensor(compensation13[i], comm_args, at::kFloat, {2 * config[CfgIntermediate]}, "compensation13");
    check_tensor(compensation2[i], comm_args, at::kFloat, {config[CfgHidden]}, "compensation2");
  }
  if constexpr (EXECUTE) {
    auto chars = group_chars(group_name);
    char *group = chars.data();
    TypedTensorList w13_int4{w13, ACL_INT4}, w2_int4{w2, ACL_INT4};
    TypedTensorList s13_uint64{weight_scale13, ACL_UINT64}, s2_uint64{weight_scale2, ACL_UINT64};
    std::vector<at::Tensor> owners{work_desc, source_desc, layer_id_to_slot, layer_params,
        comm_args, topology, control, state, ffn_scratch, completion};
    for (auto list : {w13, w2, weight_scale13, weight_scale2, compensation13, compensation2}) {
      owners.insert(owners.end(), list.begin(), list.end());
    }
    launch("aclnnAfdMegaMoeFfn", std::move(owners),
      work_desc, source_desc, layer_id_to_slot, w13_int4, w2_int4, s13_uint64, s2_uint64,
      compensation13, compensation2, layer_params, comm_args, topology, control,
      group, config, state, ffn_scratch, completion);
  }
}

template <bool EXECUTE>
void combine_recv(const at::Tensor &topk_weights, const at::Tensor &token_mask,
    const at::Tensor &task_meta, const at::Tensor &route_index,
    const at::Tensor &comm_args, const at::Tensor &topology, const at::Tensor &control,
    at::Tensor state, at::Tensor out, at::Tensor completion,
    c10::string_view group_name, at::IntArrayRef config) {
  check_common(comm_args, topology, control, state, group_name, config);
  TORCH_CHECK(topk_weights.dim() == 2 && topk_weights.size(0) <= config[CfgSourceCapacity],
              "topk_weights token capacity exceeded");
  const int64_t tokens = topk_weights.size(0);
  const auto layout = MakeWindowLayout(config.data());
  check_tensor(topk_weights, comm_args, at::kFloat, {tokens, config[CfgTopK]}, "topk_weights");
  check_tensor(token_mask, comm_args, at::kBool, {tokens}, "token_mask");
  check_tensor(task_meta, comm_args, at::kLong, {TASK_FIELDS}, "task_meta");
  check_tensor(route_index, comm_args, at::kInt,
               {static_cast<int64_t>(layout.paddedTokens * config[CfgTopK])}, "route_index");
  check_tensor(out, comm_args, at::kBFloat16, {tokens, config[CfgHidden]}, "out");
  check_tensor(completion, comm_args, at::kLong, {COMPLETION_FIELDS}, "completion");
  if constexpr (EXECUTE) {
    auto chars = group_chars(group_name);
    char *group = chars.data();
    launch("aclnnAfdMegaMoeCombineRecv",
      {topk_weights, token_mask, task_meta, route_index, comm_args, topology, control, state, out, completion},
      topk_weights, token_mask, task_meta, route_index, comm_args, topology, control,
      group, config, state, out, completion);
  }
}

// Load-time only. Original .26 loader: INT8[E,K,N/2] -> NZ -> view(INT32).
// Splitting its logical first dimension through torch_npu can trigger layout
// conversion. Instead retain the NPU StorageImpl and create metadata-only views
// of physically contiguous expert slabs. These are opaque carriers: do not use
// normal Torch arithmetic or npu_format_cast on them.
std::vector<at::Tensor> raw_extract_nz_experts(const at::Tensor &group) {
  TORCH_CHECK(group.is_privateuseone() && group.scalar_type() == at::kInt &&
      group.dim() == 3 && group.is_contiguous() && !group.requires_grad(),
      "raw_extract_nz_experts requires contiguous NPU INT32[E,K,N/8] weights");
  const auto &desc = vllm_ascend::NPUBridge::GetNpuStorageImplDesc(group);
  const int64_t experts = group.size(0), rows = group.size(1), cols = group.size(2) * INT4_PER_INT32;
  TORCH_CHECK(experts > 0 && rows > 0 && cols > 0 && rows % NZ_ROWS == 0 && cols % NZ_INT4_COLUMNS == 0,
      "packed NZ expert dimensions must be positive and aligned to 16 x 64 INT4");
  const auto &storage = desc.storage_sizes_;
  const int64_t storage_itemsize = desc.data_type_.itemsize();
  TORCH_CHECK((storage_itemsize == 1 || storage_itemsize == 4) &&
      desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ && storage.size() == 5 &&
      storage[0] == experts && storage[1] == cols / NZ_INT4_COLUMNS &&
      storage[2] == rows / NZ_ROWS && storage[3] == NZ_ROWS &&
      storage[4] * storage_itemsize == 32 && group.storage_offset() == 0,
      "weight storage must preserve grouped INT4 NZ bytes; ND or ambiguous carriers are unsupported");
  const int64_t slab = rows * cols / INT4_PER_INT32;
  TORCH_CHECK(group.storage().nbytes() >= static_cast<uint64_t>(experts * slab * sizeof(int32_t)),
      "grouped NZ weight storage is truncated");
  std::vector<at::Tensor> result;
  result.reserve(experts);
  for (int64_t expert = 0; expert < experts; ++expert) {
    auto impl = group.unsafeGetTensorImpl()->shallow_copy_and_detach(
        group.unsafeGetTensorImpl()->version_counter(), true);
    const std::array<int64_t, 2> shape{rows, cols / INT4_PER_INT32};
    const std::array<int64_t, 2> strides{cols / INT4_PER_INT32, 1};
    impl->set_sizes_and_strides(shape, strides);
    impl->set_storage_offset(expert * slab);
    result.emplace_back(std::move(impl));
  }
  return result;
}

void register_weight_helpers(pybind11::module_ &module) {
  module.def("raw_extract_nz_experts", &raw_extract_nz_experts,
      "Zero-copy raw NZ expert carriers for A3 MegaMoE; load-time use only");
}
}  // namespace afd_plugin::mega_moe

TORCH_LIBRARY_FRAGMENT(afd_ascend, ops) {
  ops.def("afd_mega_moe_dispatch_send(Tensor x, Tensor expert_ids, Tensor token_mask, Tensor task_meta, "
    "Tensor comm_args, Tensor topology, Tensor control, Tensor(a!) state, Tensor(b!) route_index, "
    "Tensor(c!) source_counts, Tensor(d!) send_scratch, *, str group_name, int[] config) -> ()");
  ops.def("afd_mega_moe_scheduler(Tensor comm_args, Tensor topology, Tensor control, Tensor(a!) state, "
    "Tensor(b!) work_desc, Tensor(c!) source_desc, *, str group_name, int[] config) -> ()");
  ops.def("afd_mega_moe_ffn(Tensor work_desc, Tensor source_desc, Tensor layer_id_to_slot, Tensor[] w13, "
    "Tensor[] w2, Tensor[] weight_scale13, Tensor[] weight_scale2, Tensor[] compensation13, "
    "Tensor[] compensation2, Tensor layer_params, Tensor comm_args, Tensor topology, Tensor control, "
    "Tensor(a!) state, Tensor(b!) ffn_scratch, Tensor(c!) completion, *, str group_name, int[] config) -> ()");
  ops.def("afd_mega_moe_combine_recv(Tensor topk_weights, Tensor token_mask, Tensor task_meta, Tensor route_index, "
    "Tensor comm_args, Tensor topology, Tensor control, Tensor(a!) state, Tensor(b!) out, "
    "Tensor(c!) completion, *, str group_name, int[] config) -> ()");
}

#define AFD_REGISTER_MEGA_MOE(KEY, EXECUTE) \
TORCH_LIBRARY_IMPL(afd_ascend, KEY, ops) { \
  ops.impl("afd_mega_moe_dispatch_send", &afd_plugin::mega_moe::dispatch_send<EXECUTE>); \
  ops.impl("afd_mega_moe_scheduler", &afd_plugin::mega_moe::scheduler<EXECUTE>); \
  ops.impl("afd_mega_moe_ffn", &afd_plugin::mega_moe::ffn<EXECUTE>); \
  ops.impl("afd_mega_moe_combine_recv", &afd_plugin::mega_moe::combine_recv<EXECUTE>); \
}
AFD_REGISTER_MEGA_MOE(PrivateUse1, true)
AFD_REGISTER_MEGA_MOE(AutogradPrivateUse1, true)
AFD_REGISTER_MEGA_MOE(Meta, false)
#undef AFD_REGISTER_MEGA_MOE
