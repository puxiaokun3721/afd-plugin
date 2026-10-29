# A3 CAMAsync MegaMoE 仓库与框架接入设计

状态：保留评审时的接入设计；首版代码已生成，实际文件、启动入口、调整及验证结果见 [实施说明](implementation.md)，设备验收尚未完成。整体目标、内存布局和算法见 [主设计](README.md)，算子参数、descriptor 字段和同步协议见 [算子与协议接口](operators.md)，验收要求见 [测试与验收](validation.md)。

本文的接入目标是 afd-plugin `587e1a9b21c86c467d99b551fa2b5ddd61c9212e`（`upstream/release/v0.26.0rc2`），原生代码来源是 ops-transformer `8d46ac50deabb544f59f56f5b82a995159907c03` 的 MegaMoE arch22。设计最初保存在 0.30 main，本次实现在独立 0.26 worktree；下文既有源码接口和行号仍按固定基线解读。

## 1. 接入边界

增加独立后端 `cam_async_megamoe` 和 `AsyncCamMegaMoeConnector`，复用 0.26 的 Attention/FFN 角色划分、role-rank 解析、HCCL 建组、Store 生命周期和 `control_plane=None` 的 connector-driven 运行方式。新协议不能与旧 `CAMAsyncAFDConnector` 互通，不能复用旧的 ready/count 编码，不能把 MegaMoE 装进旧 `DispatchRecv → layered GMM → CombineSend` 路径。

四个仓内算子为：

| 角色 | Ascend 算子名 | 拟定 PyTorch 名称 | 框架职责 |
| --- | --- | --- | --- |
| A | `AfdMegaMoeDispatchSend` | `afd_ascend::afd_mega_moe_dispatch_send` | 整个 MB 的 prologue、完整发布、建立 pending context |
| F | `AfdMegaMoeScheduler` | `afd_ascend::afd_mega_moe_scheduler` | 设备阻塞轮询完整 DP，产出一个设备工作项 |
| F | `AfdMegaMoeFfn` | `afd_ascend::afd_mega_moe_ffn` | 设备选层，融合拉取/计算/写回，发布 Done |
| A | `AfdMegaMoeCombineRecv` | `afd_ascend::afd_mega_moe_combine_recv` | 等待所有 F，按原生 row index/top-k 权重还原，释放 A 槽 |

首版是 A3 prefill、BF16 输入、动态 INT8 激活、per-channel W4A8。一个工作项仅包含一个 DP 的全部 TP 成员、一个 layer、一个 microbatch。各 F 独立选 DP，无 F leader，不增加逐任务 F-F collective。第一层通信分轮仅保留 `comm_round_idx/count/token_base=0/1/0`；F 内 `recv_route_capacity` 限制的第二层分轮必须实现。

## 2. 文件划分与依赖方向

### 2.1 拟新增的代码目录

下面是实施计划，尚不是仓库现状；目录与入口名称用于限定职责，内部辅助文件可在实现评审时细化。

```text
csrc/npu/ascend_kernels/
  afd_mega_moe_common/                 # 共享 ABI、native arch22 计算及通信适配
    README.md                         # 来源 SHA、文件清单、许可证、AFD 差异清单
    op_host/                          # 共享 shape/容量/tiling 辅助，避免重复注册
    op_kernel/                        # 保留相对 include 层级的 native 子集及 AFD 适配
  afd_mega_moe_dispatch_send/
    op_host/  op_kernel/  op_api/
  afd_mega_moe_scheduler/
    op_host/  op_kernel/  op_api/
  afd_mega_moe_ffn/
    op_host/  op_kernel/  op_api/
  afd_mega_moe_combine_recv/
    op_host/  op_kernel/  op_api/
csrc/npu/pybind/torch_binding_mega_moe.cpp
afd_plugin/
  connectors/npu/async_cam_megamoe.py
  compat/npu/cam_async_megamoe.py
  v1/worker/npu/async_cam_megamoe.py
```

`afd_mega_moe_common` 是共享源码目录，不是第五个算子。四个独立算子目录各自拥有自己的注册、shape/tiling 入口和 ACLNN API；共享目录不注册原生 `MegaMoe`，不引入另一个可调用算子来串起通信和计算。

首版 vendoring 清单以如下已定位文件为起点，构建时按 include 闭包补全并固定哈希；不是只拷贝一个主 kernel 后重新编写缺失依赖：

| 原生相对路径（ops-transformer） | 新目录中的用途 |
| --- | --- |
| `mc2/mega_moe/op_kernel/arch22/mega_moe_kernel_a3.hpp`、`mega_moe_arch22.h` | 核心类/W4模板；保留计算函数，标出外层协议和参数拆分差异 |
| `arch22/moe_permute_prologue/`、`arch22/unpermute/`（均在上行op_kernel下） | A Send/Combine的原生实现；保留mask/rowIdx语义 |
| `arch22/utils/block_mmad_w4a4.hpp`、`block_mmad_preload_async_fixpipe_quant.hpp` | 原生GMM/预取和流水，不另写数学等价替代 |
| `arch22/utils/block_epilogue_w4a8post_pertoken_swiglu.hpp`、`block_epilogue_w4a8post_pertoken_v2.hpp` | MSD补偿、激活量化、写回 |
| `arch22/utils/copy_l0c_to_gm_custom.hpp`、`copy_gm_to_l1_custom.hpp`、`weight_l2_prefetch.hpp` 及其公共头闭包 | 保留include顺序、custom specialization和prefetch，不被另一个版本的CATLASS模板覆盖 |
| `arch22/utils/hccl_shmem.hpp`、`mc2/common/op_kernel/moe_distribute_base.h` 等context定义 | 仅保留A3所需部分并增加MC2 context/角色映射适配，禁用全EP事务屏障 |
| `mc2/mega_moe/op_host/op_tiling/arch22/` 中计算规格/容量辅助 | 抽取纯计算tiling，不复制完整原生op注册 |

上表的 `arch22/...` 均相对于 `mc2/mega_moe/op_kernel/`，不能扁平复制破坏相对include。CATLASS来源按 `cmake/third_party/catlass.cmake:10` 固定 `769cd40a8716b28650b6bebb08db4834eea4462f`（v2.0.0）；实施时引入需要的头文件闭包及许可证到common的vendor子目录，或使用内容完全相同且已核验的仓内公共依赖。不得静默使用机器上任意版本的CATLASS。原生MegaMoE文件使用CANN Open Software License Agreement Version 2.0，保留其文件头及对应LICENSE，不改写为本仓MIT头。导入清单记录原路径、源SHA、文件哈希、本地路径、保留/修改原因和第三方版本。

工作区已有未跟踪的 `csrc/npu/ascend_kernels/afd_mega_moe_native/`、`csrc/npu/pybind/torch_binding_mega_moe_native.cpp`、`afd_plugin/compat/npu/mega_moe.py` 等 A5 工作。它们不属于此方案，不覆盖、不重命名，也不作为 A3 的构建前提。新的 pybind 文件和 Python wrapper 使用上表名称以避免冲突。

### 2.2 Python 责任划分

| 计划文件或现有接入点 | 责任与必要变化 |
| --- | --- |
| 新 `connectors/npu/async_cam_megamoe.py` | `AsyncCamMegaMoeConnector`；解析后端配置，构建拓扑/窗口容量，管理通信域及 task sequence；实现 A 的 send/recv 公共语义和 F 的新工作项接口；不承载模型计算 |
| 新 `compat/npu/cam_async_megamoe.py` | `MegaMoeContext`、四算子薄 wrapper、静态布局/张量契约；把 Python 对象展开成底层扁平 tensor/attrs，不在 wrapper 中读取设备 layer |
| 新 `v1/worker/npu/async_cam_megamoe.py` | `MegaMoeLayeredExecutor`；接收全部层权重、完成 load-time 格式适配、持有 TensorList/层映射和 F scratch，调用融合 FFN；不自行建通信域 |
| `compat/npu/ops.py` | 沿用私有 OPP 与 `_C_ascend` 装载，新增检查四个 MegaMoE op 的可用性函数 |
| `config.py` | 注册后端值、允许对应 async-DP 模式；将公共异步能力判断与旧 CAM 专属选择分开 |
| `connectors/factory.py` | 注册新 connector；调整 rendezvous context 的类型准入，不能只加名字而保留旧类限制 |
| `compat/npu/feature_validation.py` | 对新后端验证 A3、prefill、W4 per-channel、TP/EP/CP、动态量化、容量和图模式范围；旧 layered-GMM 开关仍只属于旧后端 |
| `v1/worker/npu/ffn_model_runner.py` | 增加独立 MegaMoE executor/work item/startup 分支；保留现有 FFN daemon 生命周期 |
| `v1/worker/npu/ffn_worker.py` | 扩展 loop-started 和 shutdown 的后端准入；新阻塞轮询必须先 STOP/drain/join，再 close 通信域，不能走旧非 CAM 的先 close 分支 |
| `v1/worker/npu/attention_model_runner.py` | 新后端参与 async-DP/启动协调及 0.26 Attention microbatch 路径；不能依赖旧 connector 的具体类强转 |
| `model_executor/models/npu/deepseek_v4.py` | 复用已加载的 routed expert 提取；新后端进入 Attention gate/shared-expert 路径，并将权重交给 MegaMoE executor |
| `model_executor/models/npu/deepseek_v4_async_cam_forward.py` 及其共用调度 helper | 保持模型数学和 token split，按 connector 公共 A 端契约发起 DS/CR；落实上一 MB 的 CR 完成后才能复用发送槽 |

优先复用已有不可变拓扑类型、rank 解析和建组 helper；需要抽取的共享能力保持小而明确。`AsyncCamMegaMoeConnector` 不继承旧 CAM connector 来获得旧 DR payload 接口，也不伪造 `hidden_states/group_list/token_nums_rankid_layeridx` 给旧 executor 消费。若基类有抽象的 F 数据接口，实施时以明确的后端工作项能力分支处理，不把融合 FFN 的工作项冒充已经完成接收的数据。

## 3. 0.26 构建、打包及装载

### 3.1 现有链路与新增点

0.26 的链路为 `setup.py → csrc/npu/build_aclnn.sh → scripts/compile_ascend_proj.sh → msopgen/Ascend CMake → 自定义 OPP run 包`，另由 `pybind/CMakeLists.txt` 构建 `afd_plugin._C_ascend`。不另建 ops-transformer Python 扩展，不依赖运行时从参考 checkout 导入代码。

| 现有文件 | 已核对行为 | 计划变化 |
| --- | --- | --- |
| `setup.py:43-88,108-119` | 在 Ascend 环境构建自定义算子，调用 `build_aclnn.sh`；传入 `AFD_SOC_VERSION`；打包 `_cann_ops_custom` 和 `_C_ascend` | 继续使用这条链路；除非打包验证发现缺项，不另加安装入口 |
| `csrc/npu/build_aclnn.sh` | `910c/ascend910_93*` 映射到 `ascend910_93`；安装生成的 run 包到私有目录 | 新四算子跟随 A3 选择；不加入 A5 算子集 |
| `ascend_kernels/operator_registry.json` | 按 SOC 列出目录，元数据控制 `requires_shmem` | 将四个算子目录加入 `ascend910_93`；不注册 common；首版没有外部 SHMEM 依赖时标记 `requires_shmem=false` |
| `scripts/select_ops.py` | 按注册表解析算子和 SOC，支持选择子集 | 保持四入口均可选；共享目录复制由构建 staging 处理 |
| `scripts/compile_ascend_proj.sh` | 总是复制 `utils`；逐 op 将 `op_host/op_kernel/op_api` 内容复制到生成工程 | 检测任一 MegaMoE 入口被选择时，额外复制 common 一次，并保留其内部 include 层级 |
| `ascend_kernels/cmake_files/op_host/CMakeLists.txt` | 递归收集 host `*.cpp`，参与 codegen、proto、tiling、opapi 构建 | 审核 common host 源文件的收集边界；不得把原生完整 op 注册 CPP 直接加入递归扫描 |
| `ascend_kernels/cmake_files/op_kernel/CMakeLists.txt` | 对生成目录收集 kernel，沿现有 Ascend 编译参数构建 | 四个入口唯一；公共模板为 include；新增 include path/编译选项需以 A3 native 依赖实际需求为准 |
| `csrc/npu/pybind/CMakeLists.txt:46-65` | `_C_ascend` 聚合 pybind，旧 async binding 仅 A3 构建 | 同一 A3 条件下加入 `torch_binding_mega_moe.cpp`；不新建 Python 扩展模块 |
| `MANIFEST.in` | 递归包含 `csrc` | common 位于该范围；同时检查 wheel 内真正包含已安装 OPP，不以源码可见代替运行包验证 |

这里的 `HcclShmem` 是原生 MegaMoE 对 HCCL 窗口访问的封装名称，并不自动意味着需要引入外部 SHMEM 工程。若未来改用其他通信分配器，再显式修改依赖和 `requires_shmem`。

现有 staging 会把不同算子目录的内容汇入生成工程，直接复制完整 native 工程有重名覆盖、原生 `MegaMoe` 重复注册和多份非 inline 定义的风险。计划采用下述约束：

1. common 只有一份经清单约束的 native 子集，保留模板所需相对路径；四个入口只 include common。
2. 原生 kernel 总入口 `.cpp`、原生算子 schema 注册 `.cpp` 不作为额外入口被自动扫描；所需 host helper 以头文件或单一明确编译单元提供。
3. kernel 入口、tiling key、host 注册名称和导出符号分别属于四个新 op；shared helper 使用独立 namespace，避免与已有 CAM 算子冲突。
4. 运行 `--ops` 的部分构建时，只要选中任一新 op 就能找到 common；未选择新 op 时不引入新依赖。

### 3.2 PyTorch 和 ACLNN 边界

沿用 `torch_binding_cam_async.cpp` 的注册方式：`TORCH_LIBRARY_FRAGMENT(afd_ascend, ...)`、`PrivateUse1`、`AutogradPrivateUse1`、`Meta`。新文件负责四个 schema、tensor/device/dtype/shape 检查和 ACLNN 调用。对持久状态、窗口控制表、workspace 的修改必须在 schema alias/mutation 契约中表达；不能把有远端副作用的算子伪装成可删除的纯函数。

`Meta` 只推导静态容量形状、返回值和 mutability，不模拟到达情况，不选择 layer，不将 task count 变成 Python 整数。底层算子只接收扁平 tensor 与静态 attrs；`MegaMoeContext` 不穿过 ACLNN ABI。共同参数包含 `comm_args: Tensor`、`topology: int32[A_total+F_total,4]`、host 不可变的固定 ABI `config: int[]` 和 `group_name: str`；`config` 对应 CANN `ListInt` 属性和 ACLNN `const aclIntArray*`。精确参数顺序和 descriptor 布局由 [算子接口](operators.md) 统一，不能在 wrapper 和 host tiling 内各维护一套偏移常量。

`comm_args` 沿用旧 CAM MC2 入口的 placeholder：0.26 `async_cam.py:321` 分配 `torch.empty((1,), dtype=torch.float16, device=device)`，旧 pybind 也明确要求 Half。它不是 Python 填写的 peer 指针表。四入口通过 MC2/KFC 通信上下文取得实际 peer/window，由 common 的通信适配统一包装到 native `HcclShmem` 访问语义；`topology` 只存 rank/角色映射，不能绕过通信上下文自行构造设备裸地址。

0.26 `pybind/pytorch_extension/op_api_common.h:230-291,365-376` 的通用 TensorList 转换按 Torch scalar_type 构造 ACL dtype。裸传 packed INT32 权重和 INT64 scale 不会自动变成 ACL_INT4/ACL_UINT64。新 pybind 内需提供局部 typed TensorList/ACL descriptor adapter，参考 native `TensorListWrapper`，显式保存 logical shape、NZ storage shape/format、INT4/UINT64 类型以及强引用；不改变全仓通用 ConvertType 的语义。GetWorkspaceSize 使用的列表 metadata 与 kernel 的列表 descriptor 必须匹配，并单独验证 descriptor 到实际存储的映射。

`compat/npu/ops.py` 现有 `_ensure_afd_custom_opp_env` 管理私有 vendor 路径、`ASCEND_CUSTOM_OPP_PATH`、`LD_LIBRARY_PATH` 及 `AFD_CUST_OPAPI_LIB_PATH`。新增 `ensure_mega_moe_ops_available`（拟定名称）复用装载过程，但必须逐一确认四个新 op 已注册。仅旧 CAM op 可见不算新后端安装成功。配置和加载应在通信启动前失败，避免部分 rank 已进入 Scheduler 等待。

源码来源说明需保留上游许可证、固定 SHA 和导入清单。所有复制代码的 AFD 差异均使用仓库规定的 `PATCH START/END` 标记；C++ 使用对应的 `//` 注释。记录删除/替换的全 EP count 交换与 barrier、增加的地址映射、层选择和分轮坐标修正，便于后续与同一 native 版本逐项对照。

## 4. 配置、拓扑和协议隔离

### 4.1 新后端配置

0.26 规范键为 `additional_config["afd"]["connector"]`；`afd_connector` 只是 `_ALIASES` 支持的兼容别名。新值为 `cam_async_megamoe`，不能用此字符串覆盖旧 `AFD_ASYNC_CONNECTOR="CAMAsyncAFDConnector"`。

以下仅为待实现的配置结构示例，`recv_route_capacity=4096` 是示例容量，不是已验证的性能推荐值：

```json
{
  "afd": {
    "connector": "cam_async_megamoe",
    "async_dp": true,
    "role": "attention",
    "num_attention_ranks": 8,
    "num_ffn_ranks": 8,
    "compute_gate_on_attention": true,
    "connector_extra_config": {
      "dynamicQuant": 1,
      "attn_ranks_per_dp": 4,
      "async_moe_ubatching": true,
      "async_moe_num_ubatches": 2,
      "recv_route_capacity": 4096
    }
  }
}
```

沿用现有部署的 host/port、HCCL buffer 和 split 配置；F 进程对应设置 `role="ffn"`。新 extra-info 类型使用明确字段和静态校验，禁止把 `recv_route_capacity` 混同于 A 通信槽 token 容量，也不直接复用旧 DR 的专家整块预算语义。

容量由实际模型 H/I/E/K、A 本地单 MB token 上界、TP/SP padding、native 行对齐和协议区大小计算。第一层通信分轮未实现时，整个 MB 放不下 A 窗口必须在启动/发起任务前报错，不能通过减小 Q 宣称解决窗口不足。`comm_round_idx/count/token_base` 是运行 ABI 预留字段，首版不提供开启多通信轮的配置开关。

### 4.2 需要扩展的旧选择器

`.26 config.py:298,322` 以旧 connector 常量判断 async-DP；`feature_validation.py`、A/F runner、DeepSeek 模型也有旧值判断。只注册 factory 名字会漏掉 async-DP patch、Attention gate/shared-expert 路径或错误落入同步 FFN。

实施时把判断分成两类：

- 公共异步能力：async-DP、`control_plane=None`、Attention 拆 MB、gate 位于 A、shared expert 位于 A、Store 启动协调。可以显式支持旧/新两个后端。
- 后端专属能力：旧 layered-GMM 环境开关、旧 `AFDAsyncTransferState`、四小算子、旧 graph capture。只能留在旧后端；MegaMoE 走自己配置和 executor。

`connectors/factory.py:68-76` 还有显式的 `issubclass(connector_cls, CAMAsyncAFDConnector)`：传入 `rendezvous_context` 时，新独立 connector 会被拒绝。这一处必须改为明确支持两个后端，或引入仅承载 rendezvous 生命周期的窄接口。不能为了绕过检查继承旧 DR/CS 实现，也不通过 `getattr/hasattr` 探测可用方法。

### 4.3 通信组和启动握手

rank 组织沿用 A 在前、F 在后的拓扑，但 kernel 不从 DP 编号猜物理 rank：初始化生成固定 source-rank 表和 F-rank 表，每个 DP 的 source 集合恰为全部 TP 成员。F 专家分区、全局专家 ID 和本地 expert ID 由静态拓扑确定。

新后端使用独立 group/startup namespace 和协议 magic/version/layout hash，不与 `afd_async_cam` 旧协议共享序号或控制区。现有 `distributed/afd_process_group.py` 可复用：`ProcessGroupRendezvousContext` 持有 Store 和 group，`create_hccl_process_group_options` 使用每 group 的 buffer 配置，`init_afd_process_group` 在建 group 前暴露 rendezvous callback。公共连接生命周期无需新建全局可变 registry。

启动分两个阶段：

1. **建通信域前的 host 一致性检查。** 基于同一 rendezvous Store 交换协议版本、角色/物理 rank、世界大小、DP/TP/EP/source 表、SOC/runtime、静态模型形状、量化模式、MB 容量和 HCCL buffer 配置。旧/新协议混配或关键字段不一致时全部失败，不能等首个 kernel 超时。此阶段可以使用 Store，不需要 F-F HCCL collective。
2. **建域与模型加载后的资源就绪检查。** 校验实际可用窗口、布局偏移、device tensor、所有层的权重格式和映射、静态 tiling、op 注册/ABI 版本。各 rank 提交就绪或失败状态；受控 warmup 使用正式单调 task sequence，完成 drain 后从下一序号接真实流量，不 reset 已消费序号。若要重新从1开始，必须重建 domain、换 epoch 并清空全域记录。

0.26 `async_cam_startup.py` 的 nonce、故障 key、有限等待和 Store 生命周期可借鉴，但其中旧 connector 类型、旧 graph warmup 和 namespace 不能原样套用。启动时 host rank 0 生成 namespace 不构成逐任务 F leader；任务调度仍由各 F 的设备 Scheduler 独立完成。

MC2/KFC context 的首次资源初始化与每任务访问也要分开验证：沿用旧 CAM 的 `GetHcclContext` 资源访问，不在 Scheduler/Ffn 的工作项路径调用要求全域同序参与的初始化或 collective。启动 warmup 可以协调全部角色；正式运行中各 F 的 DP 选择和 tile/round 数可不同，任何隐含的全域初始化等待都会违背此合同。

首版按统一 HCCL buffer 配置建立 A/F 域。F 逻辑上只需控制窗，不等于 HCCL 会为 F 物理分配更小窗口。native A3 KFC 路径取已有 HCCL winIn/winOut，不由 `HcclShmem` 按角色重新分配。角色不对称的物理内存节约不在本方案已确认能力内，详情见主设计内存章节。

## 5. A 端模型与槽位生命周期

A 端可以保留 `send_attn_output` / `recv_ffn_output` 这组模型可见语义，由新 connector 调用新 DS/CR。Attention 仍计算 gate、top-k IDs/weights 和 shared expert；新 connector 保存 top-k weights、原始 token 布局及输出引用，DS 生成整 MB 的量化展开记录/count/row-index，并最终发布 Ready。

必须补齐 token_mask 的上游来源：0.26 `model_executor/models/npu/async_cam_layout.py:105-129` 的 `CAMDispatchPayload` 没有 mask，SP 分支的本地含 padding shape 也不足以恢复有效性。在模型 stage/prepare 边界，用 `AsyncMoeStage.actual_tokens`、TP 分片的全局 token 起点及实际 split/padding 映射构造 BOOL mask，经新 connector 公共 payload 传入 Send，并由 pending context 持有到 Combine。有效性不能从 x 是否全零或 expert_id 是否为0推断。mask=true 才产生路由；mask=false 的输出为零，布局还原后丢弃 padding。

首版准入明确为纯 prefill：在模型/runner 发起 Send 前检查本次调度包含的请求阶段；decode 或 prefill/decode mixed batch 明确拒绝，不临时切换到旧 CAM 协议。测试服务使用 prefill-only/首 token 请求（`max_tokens=1`），已有完整 decode case 继续归旧后端回归。已有部署若使用独立 prefill 实例，只接入其 prefill 执行段，本方案不新增 PD/KV-transfer 功能。warmup 同样构造合法 prefill batch。

一个 `MegaMoeContext` 管理静态布局和运行 tensor；每个 A 只有一个在途 MB 物理槽。task identity 至少包含 session、DP、task sequence、layer、microbatch 和首版固定通信轮字段。DP 内 TP 成员的身份一致，但有效 token 数允许由 padding/分片契约决定，不能靠错误要求所有 source 的实际 token 数相同来验证完整性。合法零 token 成员也发布相同任务身份的 Ready。

模型调度顺序必须满足：

```text
DS(MB0) → 可提前计算 MB1 Attention → CR(MB0) → DS(MB1)
```

`CR(MB0)` 等待所有 F 的 Done，并完成当前 MB 的 unpermute/top-k 加权以后，才能让 `DS(MB1)` 覆写量化输入、返回 D 区或 row-index。提前计算下一 MB 的 Attention 不表示可以提前发布下一 MB。若未来跨 stream 调度，通过明确 event 依赖建立相同顺序，不以 Python 调用先后假定 device 执行顺序。

1024-token prologue tile 是本地处理粒度：一个 MB 的每个 tile 写不同固定偏移；最后统一 Ready。tile 之间不复用当前 MB 输入/返回空间，不额外发早 ACK。保存 expanded row index 的私有 tensor 和 top-k weights 与通信槽拥有相同任务寿命；待 CR 完成后统一释放或复用。

CR 返回后，按既有 `CAMDispatchLayout` 的 TP/SP 和 padding 契约恢复模型布局，再加 shared-expert 输出。DSV4 原有 routed scaling 因子已进入 top-k 权重的路径应保持一次生效；不能在新 FFN 和 CR 中重复乘。

## 6. F 端新 executor 与工作项

### 6.1 接入 0.26 runner

0.26 目标是 `v1/worker/npu/ffn_model_runner.py`，不是 0.30 中的其他 runner。现有 `execute_connector_driven_step:247` 已适配 `control_plane=None`；`_execute_layered_work_item:347` 则明确是 `recv_attn_output → layered executor → send_ffn_output`。新后端在前者选择独立 MegaMoE 路径，不调用后者。

拟定执行关系为：

```text
FFN daemon / execute_connector_driven_step
  └─ MegaMoE work item 分支
       ├─ AfdMegaMoeScheduler(context tensors) → work_desc + source_desc
       └─ MegaMoeLayeredExecutor(work_desc, source_desc)
            └─ AfdMegaMoeFfn(all-layer weights, context, scratch)
```

`work_desc` 为固定 `int64[32]` 设备 tensor，精确字段见算子文档。Scheduler 只选择完整 DP 并读取 descriptor，不提前构造整个任务的 route tables；Ffn 从选中 DP 的 inbox counts 开始复用 `GetCumsumForMMAIV` 等原生表构造，再按 tile/round 更新紧凑表。Scheduler 与 Ffn 在同一执行 stream 上顺序提交；Ffn 直接读 device descriptor。Python 不调用 `.cpu().tolist()`、`.item()` 读取每项的 layer/count/DP，不动态选择某一层 Python module，也不为当前专家列表重新构造 TensorList。

原 runner 遍历 `_ffn_layer_indices` 的次数不是实际设备选层来源。新路径可复用 daemon 的外层生命周期，但每次只对设备选出的一个工作项执行融合 FFN；不得把 host 的第 i 次提交解释为模型第 i 层。不同 DP 可以处于不同 layer，权重选择由当前 descriptor 决定。

### 6.2 Scheduler 返回与流控

Scheduler 采用阻塞设备轮询，公平扫描 DP，直到 `WORK`、`STOP` 或 `ERROR`。等待期间可发布 IDLE/进度状态，避免 bounded NOP 让 CPU 不断排队空 kernel。正常没有请求属于 IDLE，不应直接按“某任务通信超时”判失败。

选择一个 DP 时，必须验证其全部 TP 成员 Ready、身份一致、descriptor 边界合法，且 sequence 未被本 F 完成。F 的 `completed_seq[dp]` 在融合 kernel 完成所有读写、发布 Done 后推进，防止 A 尚未清 Ready 时重复取同一任务。合法零路由也返回 WORK，由 Ffn 执行零工作完成路径并发送 Done。

`STOP/ERROR` 工作项让随后的 Ffn 安全退出或记录故障，不访问权重/peer payload，也不能伪造成功 Done。host 可低频观察终止/故障状态，不把这一控制路径扩展为每任务 layer/count 同步。不得无限提前提交任务：runner 保持有界提交和完成观察，以现有生命周期控制停止/资源释放。

首版 host 提交窗口建议固定 `MAX_PENDING_WORK_ITEMS=2`：最多排入两组 Scheduler→Ffn，在每组 Ffn 后记录完成 event，第三组入队前等待最早一组 event；只观察设备完成，不拷贝 layer/DP/count。这样下一组已在队列中，host 回收上一组时设备仍可推进。event/背压开销进入整链性能报告；不能用不受限队列掩盖长期资源增长。收到停止请求后不再入队，已有队列按 STOP/ERROR 语义 drain。

阻塞轮询还有一项必须落实的运行时约束：STOP/error 更新不能排在被 Scheduler 阻塞的同一 stream 后方。实现须提供可被轮询端观察的独立控制更新路径，并验证可见性；可以评估专用管理 stream/已映射控制区，但不能仅凭 Python 可写 tensor 就宣称能够中断正在运行的 kernel。watchdog 区分健康 IDLE、任务停滞和 peer 失败，退出期间等待或终止已经提交的 device 工作后才销毁通信域。

### 6.3 Ffn 内部责任

一个 Ffn 调用负责当前 DP/MB/layer 的所有 tile 和接收轮。保留 native W4A8 的 MTE 拉取、MSD、GMM1、epilogue/再量化、GMM2 和回写流水；按 Q 重复使用 F 私有 scratch。外层不再 launch DR/GMM/CS 小算子，也不为每个 receiver round 增加 Python 调度。

Q 仅界定本轮 route 行容量，热点专家可以跨轮；local expert/source 计数和源端 prefix 按算子文档生成。所有轮回写到 A 整 MB D 区的最终位置，最后一次排空 native 本卡同步及远端写，再向该 DP 的每个 A 发布 Done。保留本卡 AIV/AIC flag 和 MTE completion；删除的只是全 EP count allgather/barrier，不能顺手删掉本卡计算完成条件。

不同 F 允许同时选择不同 DP。common 中任何残留的“等待全部 F 进入同一个 work item”的调用都会破坏这一前提，必须在代码审查中逐处排除，而不能只检查外层没有显式 `all_gather`。

## 7. 权重格式、常驻 TensorList 与设备选层

### 7.1 复用模型导出，不复用旧 kernel 的存储假设

0.26 `model_executor/models/npu/deepseek_v4.py:692-753` 已能从加载后的 routed experts 导出所有 W4A8 层；`model_executor/npu/async_cam_w4a8.py:89-139` 验证旧 layered 路径的格式：

| 数据 | 旧 0.26 存储契约 | 新 native A3 需求 |
| --- | --- | --- |
| W13 | contiguous INT32 packed INT4，`[e,H,2I/8]` | 每 expert 逻辑 `[H,2I]` 的 INT4 `FRACTAL_NZ`，并满足 native transpose/interleave 契约 |
| W2 | contiguous INT32 packed INT4，`[e,I,H/8]` | 每 expert 逻辑 `[I,H]` 的 INT4 `FRACTAL_NZ` |
| scale1 | 编码 INT64，per-channel `[e,2I]` | native 每 expert 的编码 scale 向量及 A3 UINT64 ABI |
| scale2 | 编码 INT64，per-channel `[e,1,H]` | native 每 expert 的编码 scale 向量；移除仅用于旧 API 的多余轴 |
| compensation1/2 | FP32 `[e,2I]` / `[e,H]` | 每 expert FP32 向量，与实际输出通道顺序一致 |

native `mega_moe_tiling_arch22.cpp:861-909` 要求 INT4 权重使用 `FRACTAL_NZ`；其 scale/bias list 也是每 expert 一个张量。旧 packed tensor 不能仅 reshape 后宣称满足 native 布局，torch INT32 存储也不能直接被当成无须说明的 native INT4 tensor。

新 executor 在模型加载阶段对每层执行一次格式适配：解读既有 packed INT4 的符号和 nibble 顺序，转换到 native 需要的逻辑 shape、padding、NZ layout、gate/up interleave 和 transpose 约定。若所选 CANN 转换 API 不能直接接收该 packed 表示，则实现明确的 load-time pack/format 转换并单独验收；不能把转换放进每个 work item。

格式转换也必须设计内存所有权：启动时先选择新后端，优先在每层参数加载/量化后处理完成时逐层转换并交接给executor，最多暂存一层的旧/新格式，不要求所有层在NPU上同时保留ND与NZ两份副本。旧参数只有在没有forward/loader/其他view引用且转换验收完成后才能释放；新后端不会走旧FFN forward。若现有loader只能全模型后处理，先估算双份峰值并明确报错，实施相应逐层加载hook后再上线，不能用OOM后的空cache补救。只有逻辑shape、NZ storage、nibble和stride全部相符时才可直接复用已有storage。CPU侧保留可重放的原始权重/转换样本用于验证，转换中间值不长期占据NPU HBM。

scale 的 INT64 存储是编码位模式，native A3 wrapper 使用 UINT64 ABI。该适配是位模式及 ABI 表达的对齐，不是把 scale 数值转换成普通浮点或重新量化。权重输出通道若发生 interleave，关联 scale 和 compensation 必须同顺序调整。

### 7.2 同一层的全部指针一起切换

`MegaMoeLayeredExecutor` 持有所有转换后的张量强引用、固定 TensorList 和 device layer map。采用一致的扁平索引：

```text
flat_index = layer_slot * local_expert_count + local_expert_id
```

W13、W2、scale1、scale2、compensation1、compensation2 六组 TensorList 全部按此索引组织，长度均为 `num_layers * local_expert_count`。model layer ID 可不连续：初始化建立 `model_layer_id → layer_slot` 的设备映射；descriptor 保存模型层 ID，Ffn 在设备入口验证并映射一次，再为六组指针生成同一 slot 的 expert view。

原生计算核心只看到选中层的 e 个 local expert，后续 GMM/epilogue 逻辑不需要感知全部层。native host 的 TensorList 长度等于 e 的检查不能直接应用于全层列表；AFD host 包装验证 `L*e` 和各层一致性，再向可复用 compute tiling 提供单层静态规格。

沿用旧选层“设备 layer map + 全层持久地址”的思路，但不照搬旧 helper 的非法层回退为第 0 层行为。非法 layer、未加载 slot 或任意一组张量数量/形状不一致，应在访问地址前进入 ERROR；不能只切 W13/W2 而继续用上一层的 scale 或 compensation。

权重指针和列表在 executor 生命周期内稳定；切换设备、重新加载权重或销毁模型时，先 drain 工作项再释放。schema 或图池记录了 tensor 地址时，格式转换/重新打包不能在 capture 后悄悄替换底层 storage。

### 7.3 静态层属性与数学一致性

首版要求所有参与层的 H/I、local expert 数、量化格式、输出 dtype、interleave 和计算模板兼容。对不能共用一个 native tiling 的层，初始化明确拒绝；不在设备上猜测可复用性。

`swiglu_limit`、激活类型、routed scaling 和模型的 gate/up 顺序必须从实际 checkpoint/已加载层提取，写入启动校验与数值基线。旧 `async_cam_w4a8.py` 已收集 `swiglu_limit`，但旧 layered 路径是否使用该属性与 native 是否应用截断不能凭名称认定相同。新路径按模型目标数学对齐，并对 native epilogue 实际支持范围做验收；若需要补齐限制参数，应在原生 epilogue 的同一位置适配，不能默默忽略非默认值。

DSV4 的 routed scaling 只保留一个生效位置；当前模型导出的 F factor 为 1 的情形，FFN 不额外乘路由因子。精度比较需使用同一 checkpoint、相同量化与激活数学的参考，不能将已知缺少激活截断的旧路径当作唯一真值。

## 8. 原生计算/tiling 的复用限制

可直接复用的重点是 arch22 W4A8 计算模板、GMM tile/stages/preload、MSD 数值过程、逐 expert 流水和本卡同步；保持性能的首要手段是让这些代码继续在一个融合 kernel 中工作，而不是重新实现等价的小算子。

必须适配的部分包括 native `EP` 的多重含义、peer 地址、count/route 表输入、全 EP 生命周期、整 MB tile 偏移、全层权重选取和多轮 source-row 坐标。不能把 A+F world size 直接作为 native EP 传入，也不能把一个 DP 的 TP 数替代所有 native EP 用途。

tiling 拆成两类信息：

| 信息 | 来源与生命周期 |
| --- | --- |
| H/I/K、e、dtype、W4 layout、Q、core/tile/stages、静态表容量、workspace 上界 | 模型/拓扑/配置确定；初始化或首次静态 tiling 时校验和缓存 |
| 本次 DP、layer slot、source token/count、tile 有效长度、receiver round、source prefix | 由 Scheduler/Ffn 设备 descriptor 和路由表确定；不得为获得这些值同步回 host |

复用原生 host tiling 中纯计算规格选择和容量推导；不能直接沿用其“对称 EP、当前输入 M、每 expert 一张量、一次完整 MegaMoe 调用”的整套校验与 workspace 分区。四入口分别推导自己的输出和临时空间；F scratch 与 A 通信窗分开计量，权重转换后的常驻显存另计。

容量与索引运算在 host 侧使用足够宽的整数并检查溢出；设备端对实际 source counts、route offset 和 Q 范围进行边界验证。peer pointer 从已验证 topology/layout 产生，work item 只能选择合法静态范围，不能直接信任 descriptor 内的任意地址。

## 9. 启动、warmup、图模式和退出

推荐初始化次序为：配置/拓扑验证 → 四 op 装载检查 → Store 协议一致性检查 → HCCL 建组和实际窗口检查 → 模型权重适配/持久列表 → scratch/descriptor/state 初始化 → 所有角色报告资源就绪 → 受控通信与数值 warmup → 进入服务。

eager 是第一道设备验收。0.26 已有 FULL graph 的旧 CAM startup/capture 方法，不表示新四算子、阻塞 Scheduler、动态层选择和远端副作用可以直接 capture。图模式需独立验证：地址稳定、op side effect 不被裁剪、不同 DP/layer 的 replay、STOP/error、warmup sequence 与正式 sequence 隔离，以及 replay 不重复消费同一 Ready。在该门禁通过前，新后端显式拒绝未支持的 graph 配置，而不是静默回退到旧 DR/GMM/CS graph。

退出顺序必须先停止接受新 A 任务，处理已发布任务或进入明确失败状态，让 F 退出阻塞 Scheduler，等待本地在途写回和必要 stream 同步，然后清理 executor/context，最后销毁 HCCL group 并 invalidate rendezvous context。发生失败时不能释放仍被 peer 读取的 A slot 后重用原序号继续服务；重启建立新的 session namespace 和清零状态。

必须修改 .26 `ffn_worker.py:212,254-289` 的具体类型判断：它目前只对旧 CAM 类设置 loop_started、走安全停止路径，新独立 connector 否则会先 `connector.close()` 再 join daemon。新分支必须先通过独立 control 请求 STOP/ABORT，再确认 daemon 和已排队 device 工作结束，最后 close；无法正常 drain 时由域 abort/进程退出收敛，不能提前释放窗口。loop_started 事件同样扩展到新后端，避免启动协调永久等待。

## 10. 实施顺序与验收边界

实施按可独立评审的步骤推进：

1. 固定 native 导入清单、许可证和差异标记；打通四个空壳的 A3 编译/注册/OPP 安装及 wrapper shape/mutation 契约，不接模型流量。
2. 实现 shared ABI、窗口/静态容量与启动握手；验证协议版本、拓扑或量化模式不一致时可终止。
3. 实现权重 load-time 适配和 device 层选择；在单 F 上证明每层六组参数共同切换，转换后权重和激活数学与 reference 一致。
4. 实现 DS/Scheduler/Ffn/CR 的单 DP 事务，再加入热点专家 Q 分轮、多 source prefix 和多 DP 独立选择；保留所有零 token/零 route 路径。
5. 接入 0.26 runner 和 DeepSeek V4 模型，先 eager E2E；最后才评估图模式和性能。

配置解析、factory 准入、启动消息和资源生命周期的 Python seam 可做 CPU 单测；不得用 mock 证明设备 Ready/Done、跨核可见性或融合数学正确。新增 unit test 文件按 source tree 映射，并先归入 `docs/design/module/` 对应模块的 `validation_paths`。NPU 协议/算子行为放 `tests/npu` 或 `tests/e2e`；构建成功、CPU 测试通过、真实 A3 精度通过、图重放通过和性能提升分别报告。

## 11. 源码证据索引

下表用于实施前复核；路径属于指定固定版本，不表示当前 main 同行仍有相同代码。

| 证据 | 固定版本中的位置 | 支持的结论 |
| --- | --- | --- |
| 0.26 依赖 | afd-plugin `pyproject.toml:37` | 目标版本，不迁移到 0.30 |
| 配置规范键/别名 | `afd_plugin/config.py:20-35,44-72,113-149,288-324` | 新值需注册；extra 配置位于 `connector_extra_config`；异步 selector 需更新 |
| factory rendezvous 限制 | `afd_plugin/connectors/factory.py:68-76,119-123` | 新 connector 不能只注册名字 |
| CAM 建组与 control plane | `afd_plugin/connectors/npu/async_cam.py:214-224,291-341` | 可复用角色拓扑/建组生命周期，协议需隔离 |
| Store/PG 生命周期 | `afd_plugin/distributed/afd_process_group.py:26-122` | 复用 context、per-group HCCL options 和建组前 callback |
| 当前 startup | `afd_plugin/v1/worker/npu/async_cam_startup.py:31-66,69-143` | host 握手、namespace 和失败等待的可复用思路 |
| FFN 旧链路 | `afd_plugin/v1/worker/npu/ffn_model_runner.py:158-176,247-377,531` | 新分支应绕过 DR/GMM/CS，不读 host 层号 |
| 旧全层权重契约 | `afd_plugin/model_executor/npu/async_cam_w4a8.py:53-180` | 全层持有、层映射和旧 packed 形状 |
| DSV4 参数导出 | `afd_plugin/model_executor/models/npu/deepseek_v4.py:647-753` | 加载后 routed expert 权重和层属性来源 |
| 既有 op 装载 | `afd_plugin/compat/npu/ops.py`、`csrc/npu/pybind/torch_binding_cam_async.cpp:277-314` | 私有 OPP 环境、统一扩展和四类 dispatcher 注册 |
| native NZ/列表校验 | ops-transformer `mc2/mega_moe/op_host/op_tiling/arch22/mega_moe_tiling_arch22.cpp:861-1099` | INT4 NZ 与每 expert scale/bias 契约 |
| native A3 scale ABI | `mc2/mega_moe/torch_extension/csrc/mega_moe.cpp:266-288` | A3 编码 scale 的 UINT64 tensor-list ABI |
| native W4 模板 | `mc2/mega_moe/op_kernel/arch22/mega_moe_arch22.h:199-270` | 原生计算 tile/layout/epilogue 复用基础 |

当前尚未完成实现、A3 编译、NPU 数值、多 rank 协议或性能验证。设计评审通过后，应在 0.26 基线上实施上述变更，当前 main 中的文档落地不构成 0.30 接口迁移授权。
