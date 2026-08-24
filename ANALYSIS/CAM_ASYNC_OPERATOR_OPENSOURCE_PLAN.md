# CAM Async 算子开源落地清单

> 记录日期：2026-08-18
> 目标：把 CAMAsync 连接器依赖的四个异步 MoE 通信算子从闭源 whl/.run 二进制，改为在 AFD 仓库内源码编译，为后续开源到社区做准备。

## 1. 背景与目标

- AFD 的 `CAMAsyncAFDConnector`（异步、NPU）依赖四个自定义算子，当前以**二进制形式**打包在仓库里：
  - `afd_plugin/connectors/npu/bin/CAM_ascend910_93_openEuler_aarch64.run`（kernel 包）
  - `afd_plugin/connectors/npu/bin/umdk_cam_op_lib-*.whl`（torch 绑定包）
- AFD 运行时在 [afd_plugin/compat/npu/ops.py](afd_plugin/compat/npu/ops.py) 里通过 `torch.ops.umdk_cam_op_lib.*` 访问这四个算子：
  - `async_dispatch_send`
  - `async_dispatch_recv`
  - `async_combine_send`
  - `async_combine_recv`
- 这四个算子的**源码由用户本人开发**，现已拉到本地 `D:\Code\mine\cam-aynsc`（gitcode: `puxiaokun3721/cam_async`）。
- 最终目标：算子源码落入 AFD 仓库 `csrc/npu/` 下，走源码编译，替换掉 LFS 里的二进制。

## 2. 算子源码仓库概况（cam-aynsc / XCCL）

- 仓库名在 README 里叫 **XCCL**，定位是"从 umdk/cam build 框架派生的 bare skeleton + 四个 A3 MoE 算子"。
- 单分支 `master`，单 commit `b3c8b34`。
- 目录结构：
  - `src/comm_operator/ascend_kernels/` —— 四个算子的 kernel 源码（完整）
  - `src/comm_operator/pybind/` —— torch 绑定层（**空骨架，见 §5**）
  - `src/third_party/catlass/` —— git submodule（**对这四个算子无用，见 §3**）
  - `build/` —— msopgen + `npu_op_*` 的构建脚本

## 3. 关于 catlass 的结论（已核实）

- catlass 是挂在 XCCL 仓库上的 submodule（`gitcode.com/cann/catlass.git`），是华为 CANN 的 GEMM 库。
- 它只对 **MoE GEMM 类算子**是必需的（build 脚本原文："only MoE GEMM-style kernels need it"，且是 `if(DEFINED ENV{CATLASS_HOME_PATH})` 的条件可选依赖）。
- **这四个算子（dispatch/combine × send/recv）是纯通信算子，源码里零 `#include` catlass**（已 grep 验证，源码只 include `kernel_operator.h`、`register/op_def_registry.h`、`tiling/...`、`lib/matmul_intf.h`，以及仓库自带的 `utils/op_kernel/op_def.h`、`comm_args.h`、`error_log.h`）。
- **结论：移植这四个算子到 AFD 时，完全不需要 catlass。**

## 4. 移植清单

### 4.1 kernel 层（源码完整，可照搬）

| 源（cam-aynsc） | 目标（vllm-afd-plugin） | 说明 |
| --- | --- | --- |
| `src/comm_operator/ascend_kernels/cam_moe_distribute_dispatch_send/` | `csrc/npu/cam_async/dispatch_send/`（或保留原目录名） | op_host + op_kernel |
| `src/comm_operator/ascend_kernels/cam_moe_distribute_dispatch_recv/` | 同上模式 | op_host + op_kernel |
| `src/comm_operator/ascend_kernels/cam_moe_distribute_combine_send/` | 同上模式 | op_host + op_kernel |
| `src/comm_operator/ascend_kernels/cam_moe_distribute_combine_recv/` | 同上模式 | op_host + op_kernel |
| `src/comm_operator/ascend_kernels/utils/op_host/error_log.h` | `csrc/npu/cam_async/utils/` | 编译必需 |
| `src/comm_operator/ascend_kernels/utils/op_kernel/comm_args.h` | 同上 | 编译必需 |
| `src/comm_operator/ascend_kernels/utils/op_kernel/op_def.h` | 同上 | 编译必需 |
| `src/comm_operator/ascend_kernels/operator_registry.json` | `csrc/npu/cam_async/` | 已含四 op，`requires_shmem: false` |
| `src/comm_operator/ascend_kernels/cmake_files/` | 参照 AFD 现有 `csrc/npu/a2e/` 改造 | op_host / op_kernel 的 CMake |

### 4.2 build 层（`.run` 打包）

- 参照 AFD 现有 [csrc/npu/build_aclnn.sh](csrc/npu/build_aclnn.sh) 与 [setup.py](setup.py) 的 `BuildAscendOps`，把 XCCL 的 msopgen + `npu_op_*` 构建流程接进来，产出 `.run`。
- `SOC_VERSION` 对齐：0.19 时代是 `ascend910_9391`（910C），0.26 是 `ascend910_93`，注意两套命名差异。
- **不需要 catlass**（见 §3）。

### 4.3 pybind 层（`.whl` 打包）—— ⚠️ 源码缺失，待补

- `src/comm_operator/pybind/` 下**框架代码是有的**：`NPUBridge.cpp`、`NPUStorageImpl.cpp`、`pytorch_extension/`、`setup.py`。
- 但 `pybind.cpp` 是**空骨架**：`PYBIND11_MODULE` 和 `TORCH_LIBRARY` 两块都没有注册那四个 op。
- **缺失的东西**（待用户从其他仓库/本地找回来）：
  1. 四个 `async_dispatch_send/recv`、`async_combine_send/recv` 的 autograd 声明 + 实现（`functions.h` 声明 + 对应 `<op_name>.cpp` 定义）；
  2. 真正调用 kernel 的 host 端 C++ 代码；
  3. `pybind.cpp` 里的 `m.def(...)` 与 `TORCH_LIBRARY` 注册；
  4. `setup.py` 里 `sources` 补充对应 `.cpp`。
- **命名空间决策**（待定，见 §6）：
  - 方案 A：保持 `umdk_cam_op_lib`，Python 层零改动，但 `umdk` 前缀带历史包袱；
  - 方案 B：改成 AFD 自有命名空间（如 `afd_ascend.cam_*`），需改 [ops.py](afd_plugin/compat/npu/ops.py) 和连接器。

### 4.4 替换 AFD 里的二进制

- 删掉 [afd_plugin/connectors/npu/bin/](afd_plugin/connectors/npu/bin/) 下的 `.run` 和 `.whl`（当前是 LFS blob）。
- 改 [afd_plugin/compat/npu/ops.py](afd_plugin/compat/npu/ops.py) 的算子发现逻辑（若换命名空间）。
- 改 [setup.py](setup.py)，让源码编译替代 LFS 二进制。

### 4.5 合规与工程

- 当前 XCCL 文件头是 `SPDX-License-Identifier: MIT` + `Copyright (c) Huawei Technologies Co., Ltd.`；AFD 仓库统一是 `Apache-2.0` + `Copyright contributors to the AFD plugin project`。移植时要确认主体许可证并统一文件头。
- `LICENSES/` 里有 GPL-2.0、MIT、Third Party Notice——GPL-2.0 与 Apache-2.0 不兼容，需甄别这些是"第三方组件许可证"还是"算子主体许可证"，主体不能带 GPL。
- README（XCCL 那个）与实际内容有脱节（README 说 registry empty、说 skeleton，但实际已含四算子），开源前要刷一遍。

## 5. 验证方案

- 在 0.19 官方镜像 `quay.io/ascend/vllm-ascend:v0.19.1rc1-a3-openeuler` 里：
  1. 先用现有 whl 跑通 PCP8/DP 拓扑，得到基线结果；
  2. 再用源码编译出的 `.run` + `.whl` 替换，跑同样的 benchmark；
  3. **要求逐位一致**（开源算子的"验证合格"标准）。
- 验证入口：`python -c "from afd_plugin.compat.npu import ensure_afd_ascend_ops_loaded"`。

## 6. 待办 / 待决策

- [ ] **找回 pybind 绑定层源码**（四个 op 的 autograd + host 端 C++ 实现），这是最大的不确定项。
- [ ] 决定命名空间：保留 `umdk_cam_op_lib` vs 改 AFD 自有命名空间。
- [ ] 决定 catlass submodule 是否从 XCCL 仓库里摘除（对这四个算子无用）。
- [ ] 决定 kernel 源码目录在 AFD 里的落点与命名（`csrc/npu/cam_async/` vs 其他）。
- [ ] 确认 `lib/matmul_intf.h` 在目标 CANN（8.5.1 / 9.0.1）里的可用性。
- [ ] 0.19 与 0.26 的 `SOC_VERSION` 命名对齐（`ascend910_9391` vs `ascend910_93`）。
- [ ] 开源后二进制去留策略（是否保留 whl 做快速安装）。

## 7. 关键文件索引

- AFD 侧算子发现：[afd_plugin/compat/npu/ops.py](afd_plugin/compat/npu/ops.py)
- AFD 侧二进制：[afd_plugin/connectors/npu/bin/](afd_plugin/connectors/npu/bin/)
- AFD 侧 build：[setup.py](setup.py) + [csrc/npu/build_aclnn.sh](csrc/npu/build_aclnn.sh)
- AFD 现有算子参照：[csrc/npu/a2e/](csrc/npu/a2e/) + [csrc/npu/e2a/](csrc/npu/e2a/)
- XCCL 算子源码：`D:\Code\mine\cam-aynsc\src\comm_operator\ascend_kernels\`
- XCCL build：`D:\Code\mine\cam-aynsc\build\`
