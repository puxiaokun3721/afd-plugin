# 内网 agent 执行 prompt：A3 CAMAsync MegaMoE 编译、调试与验证

你接手的是一个已经生成首版源代码、尚未在 CANN/NPU 上编译运行的改造项目。请直接推进环境核对、实际编译、调试、精度、协议和性能验证；发现明确的软件问题可以修复。不要仅输出审查意见或重复设计。每个阶段保存可复现命令和原始结果，不把源码存在、CPU 通过或通信不挂误当成设备数值/性能通过。

## 1. 仓库、版本和资料

- 用户仓库：`https://github.com/puxiaokun3721/afd-plugin.git`。
- 交接分支：`codex/a3-camasync-megamoe`。从交接方给出的发布 commit SHA 启动；记录实际 SHA。代码和本 prompt 均在该分支，不能从 GitHub 默认 main 开始。
- afd-plugin 基线：`587e1a9b21c86c467d99b551fa2b5ddd61c9212e`，来自 `upstream/release/v0.26.0rc2`，目标 vLLM **0.26.0**。不迁移到 0.30，也不擅自用其他版本接口掩盖编译/运行问题。
- 原生 MegaMoE：ops-transformer `8d46ac50deabb544f59f56f5b82a995159907c03` 的 **arch22/A3 W4A8** 路径。
- CATLASS：`769cd40a8716b28650b6bebb08db4834eea4462f`。两者需要的源码/头文件已随插件 vendoring，运行时不应依赖开发者机器的参考仓。
- A5 仅设计参考：`https://github.com/puxiaokun3721/afd-plugin/blob/5ac3177c9a2fcade47051f251ba4b19a5954bbb4/docs/design/a5_async_afd_design.md`。不把 A5 的架构、接口、量化或验证记录套到 A3。

先阅读本仓：

1. `docs/design/a3_camasync_megamoe/README.md`：整体背景、原生流程、窗口与两级分轮。
2. `operators.md`：四算子接口、ABI、Ready/Done、状态机。
3. `integration.md`：0.26 自定义算子打包和框架接入设计。
4. `implementation.md`：**当前实际代码入口、可运行命令、与设计草案的差异、已执行检查**。
5. `validation.md`：完整验收计划。其中评审阶段的 `cases/` 和 `--checkpoint` CLI 是草案，尚不能照抄执行；当前脚本使用 `tests/npu/manifests/`。
6. `tests/e2e/models/deepseek_v4_flash/README.md`：首 token 参考 JSON 与 E2E。

遵循当前仓库 AGENTS.md。开始前保存 HEAD、branch、status、相关软件版本及原有修改，避免覆盖其他人的工作。内网若不能访问 GitHub，使用交接方提供的同一发布 commit 的完整 Git bundle/源码镜像；不要只拷贝一个 diff 而漏掉新文件或 vendored 依赖。

## 2. 项目背景与必须保持的语义

原 CAMAsync 使用 DispatchSend / DispatchRecv / layered FFN / CombineSend / CombineRecv 等小算子。现在改为四个插件内自定义算子：

| 角色 | 新算子 | 主要责任 |
| --- | --- | --- |
| Attention，简称 A | `AfdMegaMoeDispatchSend` | 原生 prologue：mask、动态量化、expert 排序、top-k 展开，写 A 本地窗口，向 F 发布 count/meta/Ready |
| FFN，简称 F | `AfdMegaMoeScheduler` | 独立轮询；一个 DP 的全部 TP source 已完整到达才选择该任务 |
| F | `AfdMegaMoeFfn` | 设备选层；一个 kernel 内完成所选 DP/MB 的全部 tile/Q 轮、远端拉取、MSD、GMM1、SwiGLU/再量化、GMM2 与回写 |
| A | `AfdMegaMoeCombineRecv` | 等全部 F Done，再进行原生 unpermute/top-k 加权合并，成功后才允许复用槽 |

必须保持：

- 仅 A3 / arch22、DeepSeek V4 Flash、BF16 激活、**per-channel W4A8**、prefill、eager。模型维度与量化由实际 checkpoint/加载后的参数确认。
- 默认实际模型目标：A DP2/TP4 + F EP8/TP1，共 16 个进程；每 DP scheduler `max_num_batched_tokens=8192`、两个 AFD microbatch、PCP/DCP=1。不要把它误算成每个 MB 或每个 TP 固定相等的有效 token 数。
- 一个任务仅一个 DP、一个 layer、一个 microbatch；每个 F 独立选 DP，允许不同 F 先处理不同 DP。不增加逐任务 F-F count allgather、leader 或跨 DP/MB 组 batch。
- A 的输入/返回窗口容纳整 MB。当前只实现**第二层分轮**：F 的接收/计算 scratch 用 Q=`recv_route_capacity` 限容，热点专家也可切成多个 Q 轮。
- 第一层通信窗口分轮仅预留 task metadata 字段，首版只接受 `comm_round_idx/count/token_base=0/1/0`。降低 Q 不能解决 A 的整 MB 窗口不足。
- 同一 DP 的多个 MB 共享单物理槽：A 执行 `CR(MB0) → DS(MB1)`，下一个 MB 的 Attention 计算可提前重叠。不改为多通信槽并发发布。
- Dispatch 是 F **read/pull A 窗口**，F 回写结果到 A。首版保留 top-k 复制；输入 expanded record 为量化 token 加 512B 行尾区域。未来去重需要新增 layout/路由协议，不能在本轮顺手替换现有布局。
- 所有 F，包括零路由 F，都必须在完整任务结束后发布 Done；A 不依据局部接收轮提前释放整个 MB。
- 层号在设备端通过 LUT 选择六组常驻 TensorList：W13、W2、两组 scale、两组 compensation。禁止每任务 `.item()` 回读层号/count 后在 Python 选权重。
- 原生计算代码尽量复用，保持 W4A8 tile/stages/preload、MSD、激活公式、epilogue 与 C2V/V2C/GMM flag 依赖。不要为“跑通”退回小算子链并仍称新路径通过。

## 3. 已生成代码在哪里

四个算子目录均含 `op_kernel/`、`op_host/`、`op_api/`：

```text
csrc/npu/ascend_kernels/afd_mega_moe_dispatch_send/
csrc/npu/ascend_kernels/afd_mega_moe_scheduler/
csrc/npu/ascend_kernels/afd_mega_moe_ffn/
csrc/npu/ascend_kernels/afd_mega_moe_combine_recv/
```

FFN 的 cpp 约 172 行，只是 kernel 入口和协议前后处理。真实调用链为：

```text
afd_mega_moe_ffn.cpp → common/op_kernel/native_ffn.h::RunNativeFfn
  AIC → MegaMoeKernel::operator() → GMM1 / GMM2
  AIV → native_receiver.inc::ProcessAfd → ProcessRecvRound
        → pull/MSD、激活/再量化、fused remote store
```

其中 common 为 `csrc/npu/ascend_kernels/afd_mega_moe_common/`：

- `native/mc2/mega_moe/op_kernel/arch22/mega_moe_kernel_a3.hpp`：约 2,212 行，原生核心主体。
- `native/.../utils/block_epilogue_w4a8post_pertoken_swiglu.hpp`、`block_epilogue_w4a8post_pertoken_v2.hpp`：W4A8 后处理。
- `vendor/catlass/include/`：底层矩阵/搬运模板依赖。
- `include/afd_mega_moe/abi.h`、`tiling.h`：共享 ABI、布局及 tiling。
- `op_kernel/protocol.h`、`native_receiver.inc`、`native_ffn.h`：AFD 适配。
- `source_manifest.json`：固定来源、源/目标哈希、修改标记。203 个依赖中 198 个原样保留、5 个适配，另有固定 W4A8 模板包装和两份 LICENSE。

其他入口：

```text
csrc/npu/pybind/torch_binding_mega_moe.cpp
afd_plugin/compat/npu/mega_moe_abi.py
afd_plugin/compat/npu/cam_async_megamoe.py
afd_plugin/connectors/npu/async_cam_megamoe.py
afd_plugin/v1/worker/npu/async_cam_megamoe.py
afd_plugin/v1/worker/npu/{attention_model_runner,ffn_model_runner,ffn_worker}.py
afd_plugin/model_executor/models/npu/deepseek_v4*.py
```

四 op 已接 A3 registry、CMake、现有 msopgen/私有 OPP 和 `_C_ascend`。公共依赖必须保留目录层级，不要扁平化 include。新后端使用 `additional_config.afd.connector="cam_async_megamoe"` 和规范键 `async_dp=true`；继续保留 Attention gate、ModelRunnerV1、eager、CP1、FFN TP1/EP 等准入。

## 4. 已知风险和重点检查项

1. **工具链尚未实际编译**：GE tiling、MC2 资源、ACLNN、Torch-NPU storage descriptor、INT4 NZ / UINT64 TensorList 描述符可能存在版本差异。先真实编译再判断，不能把确定的软件错误归为“需要硬件验证”。
2. **实际权重布局**：现有 0.26 loader 是 INT8-packed → NZ → INT32 view。`raw_extract_nz_experts` 按 storage descriptor 建 expert slab 零拷贝 carrier。验证每个 pointer/offset/shape/stride、所有者生命周期和 scale/compensation 编码；丢失 NZ metadata 时明确拒绝，不能猜 ND 等价。
3. **源行偏移**：多 source、非首个专家、热点专家跨 Q 轮时，远端行地址必须为 immutable source-expert prefix 加本轮源内偏移，不能使用合并接收行号。原生参考自身的多轮结果也需独立真值校验。
4. **内存边界**：Q 可为 1/8/257 等任意正数，AFD scratch 的 table/scale/C 基址单独对齐。核实实际写入大小、尾轮、UB 16032B route-cache 上界、A window 容量与实际 HCCL 分配；不要把 Torch allocator peak 当全部通信内存。
5. **Ready/Done 发布**：512B 独占记录、32-bit 正 ticket、稳定 int64 metadata、双读 commit、epoch 隔离。各 source→F、各 F→A 的 payload→commit 及 commit→payload 可见性必须实测；同卡 event 或 MTE3 drain 不自动证明跨卡 release/acquire。
6. **跨核同步**：FFN 只在统一 preflight 和原生流水 drain 后处理错误/ABORT；禁止 AIV 提前退出而 AIC 仍等 flag。验证零任务、零路由 tile、Q 多轮、tile 间固定 flag 复用及不同 F 不同 DP 顺序。
7. **生命周期/失败**：独立 Store monitor、控制 stream、sticky domain error、STOP→drain/join→释放；Attention 每模型步返回前检查 fatal。CR 出错时清零输出且报错，不返回有效结果。内核没有 device watchdog，硬件卡死由 runtime/外部监督处理，不能释放仍被设备引用的资源。
8. **框架数值语义**：stage actual tokens + TP offset 生成 mask，SP padding 不进有效路由；单 token prefill 合法，decode/mixed 要在 Ready 前一致拒绝；gate/shared experts 在 A，routed scaling 只乘一次，保留每层 `swiglu_limit`。

## 5. 验证顺序和可运行命令

先记录 NPU 型号、CANN/driver/firmware、NNAL（若使用）、Torch/Torch-NPU、vLLM/vLLM-Ascend 的版本/commit、模型 revision 和配置。使用与 0.26 匹配的已安装运行栈；不要升级到最新版来替代兼容性定位。

### 5.1 CPU 与真实构建

```bash
git status --short
git rev-parse HEAD
git merge-base --is-ancestor 587e1a9b21c86c467d99b551fa2b5ddd61c9212e HEAD
python -m pytest tests/unit -q -o addopts=

mkdir -p .cache
g++ -std=c++17 -Wall -Wextra -Werror \
  -I csrc/npu/ascend_kernels/afd_mega_moe_common/include \
  tests/npu/a3_camasync_megamoe_abi.cpp -o .cache/mega_moe_abi_probe
python tests/npu/a3_camasync_megamoe_abi_check.py \
  --probe .cache/mega_moe_abi_probe --output .cache/mega_moe_abi_validation.json

SOC_VERSION=910c AFD_BUILD_ASCEND_OPS=1 \
  pip install -e . -v --no-build-isolation
```

确保 compiler、CANN、Torch-NPU 等环境已按内网规范初始化。保存完整 build log，确认四个新 op 被选中/注册、私有 OPP 被加载；不能只看到 Python editable 安装成功就判算子编译成功。

### 5.2 Binding 与合成精度

```bash
AFD_RUN_ASCEND_OP_RUNTIME=1 SOC_VERSION=910c \
  python -m pytest -q tests/npu/a3_camasync_megamoe_binding.py

python tests/npu/a3_camasync_megamoe_precision.py --dry-run \
  --manifest tests/npu/manifests/a3_megamoe_small.json

HCCL_BUFFSIZE=128 HCCL_OP_EXPANSION_MODE=AIV \
  torchrun --standalone --nproc-per-node=4 \
  tests/npu/a3_camasync_megamoe_precision.py \
  --manifest tests/npu/manifests/a3_megamoe_small.json --output .cache/precision-small.json

HCCL_BUFFSIZE=128 HCCL_OP_EXPANSION_MODE=AIV \
  torchrun --standalone --nproc-per-node=16 \
  tests/npu/a3_camasync_megamoe_precision.py \
  --manifest tests/npu/manifests/a3_megamoe_multitile.json --output .cache/precision-multitile.json

HCCL_BUFFSIZE=128 HCCL_OP_EXPANSION_MODE=AIV \
  torchrun --standalone --nproc-per-node=16 \
  tests/npu/a3_camasync_megamoe_precision.py \
  --manifest tests/npu/manifests/a3_megamoe_distributed.json --output .cache/precision-distributed.json
```

上述 H/I/E/K 为小型合成维度。检查独立 signed-INT4 数学参考、finite、rtol/atol/relative-L2、路由 count、无遗漏/重复、mask、稀疏层号、同层重入、非零 clipping、空 TP/零路由 F。默认合成阈值 rtol=0.04、atol=0.05、relative-L2=0.05 仅为这些构造的初始门槛；失败时先定位，不直接放宽阈值。全零输出、NaN 或未覆盖分支不能通过。

### 5.3 协议、故障和长时间复用

```bash
# stop/abort 各自使用新域；以下命令的 stop 可改为 abort 再执行一次。
HCCL_BUFFSIZE=128 HCCL_OP_EXPANSION_MODE=AIV \
  torchrun --standalone --nproc-per-node=4 \
  tests/npu/a3_camasync_megamoe_control.py stop \
  --manifest tests/npu/manifests/a3_megamoe_small.json --output .cache/control-stop.json
```

再补齐并执行：确认已进入 idle polling 后独立控制流可推进；不同 F 先选不同 DP；单热点专家多轮；大量 CR→DS 复用；stale epoch/ticket、非法拓扑/层号/count、部分 TP 不到达、对端退出、prepare 失败、任务超时、STOP drain。测试不可依赖健康空闲时的错误超时。设备不响应时先保留 log/profile/资源状态，再按内网故障规范处理，不让后台进程污染下轮结果。

### 5.4 实际 checkpoint 精度与 E2E

本阶段**还需要补软件测试工具**：当前脚本未实现 `--checkpoint` 冻结 payload 的旧链/native/新链三方回放、原生 FFN 同工作量性能比较器、完整发布 litmus/故障注入。请基于现有 harness 实现并验证，不能只照跑现成脚本后宣布完整验收。

使用用户当前 per-channel W4A8 DSV4 checkpoint，记录真实 H/I/E/K、MoE 层表、NZ、scale、compensation、激活限制。采集首/中/末层、不同层参数、同层重入的冻结输入/route/weight/output，分别对独立数学参考、正确的 native 路径、新路径比对，另采最终 prefill logits。旧 CAM 若忽略非零 clipping，不能把其输出作为唯一真值。

E2E 先用相同 checkpoint、tokenizer、prompt/template、temperature=0、max_tokens=1、logprobs=true 的 native 服务生成首 token 参考 JSON；格式见 DSV4 guide。然后配置：

```bash
export AFD_E2E_BACKEND=npu
export AFD_E2E_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export AFD_NPU_E2E_MODEL=/actual/path/to/per-channel-w4a8-checkpoint
export AFD_NPU_MEGAMOE_PREFILL_REFERENCE=/actual/path/to/native-first-token-reference.json
export HCCL_IF_IP=<actual-communication-ip>
export HCCL_SOCKET_IFNAME=<actual-interface>
python -m pytest -q -s \
  'tests/e2e/models/deepseek_v4_flash/test_async_cam_npu.py::test_deepseek_v4_flash_async_cam[afd-dsv4-flash-megamoe-prefill-dp2tp4-ep8]'
```

路径/IP 占位符必须替换，设备列表按实际16个可见设备调整。首 token logprob atol=0.05 是初始门槛，HTTP 延迟含调度/网络；这两者都不能替代多层 logits 精度或 native kernel 性能。

### 5.5 性能

正确性通过后运行现有合成链路 benchmark：

```bash
HCCL_BUFFSIZE=128 HCCL_OP_EXPANSION_MODE=AIV \
  torchrun --standalone --nproc-per-node=16 \
  tests/npu/a3_camasync_megamoe_performance.py \
  --manifest tests/npu/manifests/a3_megamoe_distributed.json \
  --warmup 20 --iterations 100 --output .cache/performance.json
```

现有 timing 是完整 A/F 链路，含 Ready 等待；barrier 只在测量边界，生产协议不含此 barrier。进一步实现同 payload 的原生 FFN 热路径及旧链对比，使用实际模型形状/层、均衡/热点路由和多种 P/Q。记录 warmup、采样数、median/p95、分阶段 profile、通信与计算 overlap、内存分配，包含 host TensorList/管理线程/模型步同步成本；不以单次结果宣称匹配原生性能，也不通过删同步/错误检查来换取未经证明的加速。

## 6. 交接时已有证据

- WSL CPU：617 passed、137 skipped、17 subtests passed。缺 Torch/NPU/vLLM 的用例跳过；后续在内网完整运行时数量可能不同，记录实际结果。
- ABI + 构建清单：55 passed。
- 发布前补充 package bytes/manifest 哈希回归用例，定向测试扩为 56 项；上游依赖、许可证按固定 Git blob 逐一二进制核验，提交对象也核对哈希。
- 普通 g++ C++17 ABI probe 编译、Python/C++ 的50组窗口/scratch/轮数对照通过。
- 修改/新增 Python AST、ruff、diff 检查及依赖来源哈希检查通过；两个 E2E 节点可收集。
- **没有 CANN 编译通过、NPU 精度、协议实测、真实模型 E2E 或性能验收证据。**知识查询只是设计参考，可移植导出不能替代原始 receipt，更不证明跨卡内存序。

## 7. 交付要求

持续推进可执行的工作，不在第一处编译错误就停留于分析。按“环境/编译 → binding → 数值 → 协议/故障 → checkpoint/E2E → 性能”阶段保留门禁。修复保持必要性和来源补丁标记；改变 vendored 源码后更新 `source_manifest.json` 的落地 hash 与修改说明，不改来源版本来掩盖差异。

最终提供：实际 commit/软件栈/模型 manifest、编译修复及错误日志、通过/失败/跳过测试矩阵、逐层数值误差、协议/故障证据、profile 和原始性能数据、已知限制及剩余工作。将结论写入 `docs/design/a3_camasync_megamoe/` 下的设备验证报告；较大 profile 和模型数据保存在内网产物目录，报告引用可定位路径，不提交 checkpoint/凭据/用户请求。修复提交到独立验证分支，是否向外网推送按内网策略和用户授权执行。
