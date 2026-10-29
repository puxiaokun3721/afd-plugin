# A3 CAMAsync MegaMoE 首版实施说明

实现位于独立 0.26 分支 `codex/a3-camasync-megamoe`，基线 `587e1a9b21c86c467d99b551fa2b5ddd61c9212e`。目标是 A3、DeepSeek V4、BF16 输入及 per-channel W4A8 prefill。本文记录生成的代码和真实执行的检查；设备编译、精度、可靠性与性能尚未验收。

## 1. 代码入口

| 部分 | 仓内路径 | 行为 |
| --- | --- | --- |
| DispatchSend | `csrc/npu/ascend_kernels/afd_mega_moe_dispatch_send/` | 复用原生 prologue 量化/排序/top-k 复制；发布各 F 的 count、任务描述和 Ready |
| Scheduler | `csrc/npu/ascend_kernels/afd_mega_moe_scheduler/` | 本地轮询完整 DP，独立选择任务，不做逐任务 F-F collective |
| Ffn | `csrc/npu/ascend_kernels/afd_mega_moe_ffn/` | 设备选层，单 kernel 执行全部 tile/Q 轮，复用 W4A8 MSD、GMM1、激活/再量化、GMM2 与远端写回 |
| CombineRecv | `csrc/npu/ascend_kernels/afd_mega_moe_combine_recv/` | 等全部 F Done，复用原生 unpermute/top-k 加权合并；成功后释放单槽 |
| 公共代码 | `csrc/npu/ascend_kernels/afd_mega_moe_common/` | ABI、tiling、协议适配、固定版本原生代码及 CATLASS 依赖 |
| Torch binding | `csrc/npu/pybind/torch_binding_mega_moe.cpp` | 四入口、可变 buffer schema、Meta 校验、INT4 NZ/UINT64 描述符、权重 slab helper |
| 主机布局契约 | `afd_plugin/compat/npu/mega_moe_abi.py` | 不依赖 Torch 的容量/地址计算与 24-field config、16-field task 编码 |
| 持久运行上下文 | `afd_plugin/compat/npu/cam_async_megamoe.py` | buffers、stream 顺序、外部张量 `record_stream`、独立控制流 STOP/ABORT |
| Connector | `afd_plugin/connectors/npu/async_cam_megamoe.py` | 通信域、协议/参数握手、单槽生命周期、域故障传播与 watchdog |
| FFN 执行器 | `afd_plugin/v1/worker/npu/async_cam_megamoe.py` | 六组全层 TensorList、layer LUT、最多两个排队工作项、退出 drain |
| 模型/runner | `afd_plugin/model_executor/models/npu/deepseek_v4*.py`、`afd_plugin/v1/worker/npu/*runner.py`、`ffn_worker.py` | 原有两 microbatch 排布、真实 token mask、prefill 准入及融合 FFN 分支 |

四算子通过现有 `operator_registry.json → compile_ascend_proj.sh → 私有 OPP → _C_ascend` 链路构建，仅加入 `ascend910_93`。公共目录保留原目录层级，避免相对 include 破坏；不注册为第五个计算算子，不依赖运行时访问参考仓。

## 2. 原生代码复用与必要调整

`afd_mega_moe_common/source_manifest.json` 记录每个引入文件的来源 revision、源 SHA256、落地 SHA256 和是否修改。原生代码固定于 ops-transformer `8d46ac50deabb544f59f56f5b82a995159907c03`，CATLASS 固定于 `769cd40a8716b28650b6bebb08db4834eea4462f`；保留双方 LICENSE。

依赖闭包共有 203 个上游文件，其中 198 个与固定 Git blob 逐字节一致、5 个有适配；另有固定 W4A8 模板包装和两份许可证。发布前按实际 Git blob 二进制内容重新核验来源，修正了缓存副本的换行差异。`.gitattributes` 对 native 固定 LF、对 vendor 禁用文本换行转换；CATLASS 固定版本含混合行尾，按各自 Git blob 原字节保存，保证提交和 Linux 检出后的哈希一致。

计算模板保留原 W4A8 L1/L0 tile、stages/preload、MSD 和 C2V/V2C/GMM flag 流水。主要适配为：

- 增加 receiver-only 入口，把 source 轴设为所选 DP 的 T，global expert 轴保留 F×e，远端地址按 source descriptor 的物理 rank 获取。
- 设备 layer LUT 选择 `layer_slot * local_experts`，同步偏移六组权重、scale 与 compensation；原加载权重不重新量化。
- 每 tile 从本 F inbox 整理 `[T,Epad]` count，保存源专家前缀。热点专家的后续 Q 轮使用“源专家前缀 + 本轮在该源专家内的偏移”，不把合并接收行号当成远端源行号。
- 任意正逻辑 Q 可用于切轮；round table 后、两块 per-token scale 和 GMM C 的基址按 512B 对齐。逻辑容量不通过强制 Q%16 来改变。
- A3 W4A8 activation UB 临时区按本分支实际需求收敛，启动前检查剩余 route cache 上界；不改变矩阵计算分块。
- CR 使用原生 unpermute；小 B 时 split divisor 至少为 1。空 source/零路由 F 仍参与最终 Done。

首版仍按 top-k 复制量化记录，A 窗保存整 MB。Q 只限制 F 私有接收/计算 scratch，不缩小 A 输入/返回区。未来去重需新增 input-layout 版本及路由到唯一 token 的映射；当前 layout=1 保持原生读取方式。通信层分轮参数仅接受 `comm_round_idx/count/token_base=0/1/0`。

## 3. 框架合同

配置选择 `additional_config.afd.connector="cam_async_megamoe"`，同时设置规范字段 `async_dp=true`（兼容别名 `async=true`）、Attention gate、eager、ModelRunnerV1、CP=1、FFN TP=1/EP。形状和量化以实际模型及加载后权重为准；不以模型名称推断 H/I/E/K。

`connector_extra_config.recv_route_capacity` 默认 4096，可设置任意正数；`source_token_capacity` 缺省按完整 scheduler batch 的 TP 分片上界保守分配，不假设两个 microbatch 等大；可显式覆盖，双方必须一致。`task_timeout_seconds` 默认 300，限制已发布 A 任务的完成等待。HCCL window 必须满足 ABI 容量校验，F 的实际 HCCL 分配也按统一窗口计算。

A 沿现有流水执行 `CR(MB0) → DS(MB1)`；Context 在切换 stream 时建立依赖，DS/CR 外部张量记录执行 stream。实际 stage token 数和 TP offset 形成 mask，padding 不产生有效路由。decode/mixed batch 在发布 Ready 前通过 TP 一致准入检查拒绝。

F 每次 `Scheduler → Ffn`，设备选 DP/层。Host 不读 layer/count 或拆 Q 轮。六组 TensorList 常驻；原 INT8-packed→NZ→INT32 view 权重使用 storage descriptor 验证后生成零拷贝 expert carrier。若权重已丢失 NZ metadata，明确拒绝，不能猜测其存储布局。

所有 rank 先交换协议 ID，旧 CAM 与 MegaMoE 混配拒绝；再校验 ABI、拓扑、容量和 epoch。F 全部层表及激活参数一致后才发布资源就绪。管理 Store 传播 sticky ABORT，独立控制 stream 使空闲 Scheduler 可以退出；心跳区分活着但空闲与对端失联。A 的任务完成 event 用于 watchdog，不作为跨 MB 重复 wait。STOP 后先 drain/join，再释放通信资源。

完成事件不等于计算成功。管理线程先核对设备 fatal 再回收完成事件；Attention 在每个模型步返回前统一同步 Context、核对 fatal/域 ABORT，防止 CR 已退出但输出无效时返回正常响应。错误 CR 将输出清零并保留 ERROR。该模型步边界同步及六组 TensorList 的 host 描述符成本需要计入链路性能，不以原生 kernel 复用掩盖框架开销。

原生 FFN body 不插入会让 AIC/AIV 单边提前退出的 abort 分支；异常只在统一 preflight 或原生流水 drain 后处理。若设备计算硬件停滞，不能据 host ABORT 宣称已安全释放，仍需进程/NPU 故障恢复。

## 4. 构建与已生成的测试入口

以下 Linux 命令需在匹配 0.26 的 Torch/Torch-NPU/vLLM/vLLM-Ascend 和 A3 CANN 环境执行。本机尚未运行这条设备构建命令：

```bash
SOC_VERSION=910c AFD_BUILD_ASCEND_OPS=1 \
  pip install -e . -v --no-build-isolation
```

纯主机 ABI 对照可单独复现，不依赖 CANN：

```bash
mkdir -p .cache
g++ -std=c++17 -Wall -Wextra -Werror \
  -I csrc/npu/ascend_kernels/afd_mega_moe_common/include \
  tests/npu/a3_camasync_megamoe_abi.cpp -o .cache/mega_moe_abi_probe
python tests/npu/a3_camasync_megamoe_abi_check.py \
  --probe .cache/mega_moe_abi_probe --output .cache/mega_moe_abi_validation.json
```

先检查 binding/Meta 与真实 NZ slab：

```bash
AFD_RUN_ASCEND_OP_RUNTIME=1 SOC_VERSION=910c \
  python -m pytest -q tests/npu/a3_camasync_megamoe_binding.py
```

四算子精度链路使用独立 signed-INT4 数学参考，检查 finite、误差、count、路由无丢失/重复、mask，以及非连续选层。manifest 中的 H/I/E/K 为合成测试维度，不声称等同真实 DSV4 checkpoint：

```bash
# CPU 可执行，只验证配置和容量，不运行设备算子。
python tests/npu/a3_camasync_megamoe_precision.py --dry-run \
  --manifest tests/npu/manifests/a3_megamoe_small.json

# 单机 4 rank：A DP1/TP2 + F EP2，含空 source、小 Q 与激活限制。
HCCL_BUFFSIZE=128 HCCL_OP_EXPANSION_MODE=AIV \
  torchrun --standalone --nproc-per-node=4 \
  tests/npu/a3_camasync_megamoe_precision.py \
  --manifest tests/npu/manifests/a3_megamoe_small.json --output precision-small.json

# 16 rank：DP2/TP4 + EP8，1025-token 尾 tile、热点专家多轮。
HCCL_BUFFSIZE=128 HCCL_OP_EXPANSION_MODE=AIV \
  torchrun --standalone --nproc-per-node=16 \
  tests/npu/a3_camasync_megamoe_precision.py \
  --manifest tests/npu/manifests/a3_megamoe_multitile.json --output precision-multitile.json

# 同样 16 rank 的分散路由/性能；先做一次精度预检，再 warmup 和计时。
HCCL_BUFFSIZE=128 HCCL_OP_EXPANSION_MODE=AIV \
  torchrun --standalone --nproc-per-node=16 \
  tests/npu/a3_camasync_megamoe_performance.py \
  --manifest tests/npu/manifests/a3_megamoe_distributed.json \
  --warmup 20 --iterations 100 --output performance.json

# 分别启动新通信域运行 stop 和 abort。
HCCL_BUFFSIZE=128 HCCL_OP_EXPANSION_MODE=AIV \
  torchrun --standalone --nproc-per-node=4 \
  tests/npu/a3_camasync_megamoe_control.py stop \
  --manifest tests/npu/manifests/a3_megamoe_small.json
```

计时报告是完整 A/F 链路的最慢 rank 时间，包含 Ready 等待；报告各 rank allocator peak，不冒充实际 HCCL window 内存统计。benchmark 的 barrier 只在测量边界，生产算子协议没有这个 barrier。当前脚本尚未实现真实 checkpoint 冻结 payload 的旧链/native 三方回放，也没有原生 FFN 热路径同工作量性能比较器。

E2E 场景为 `afd-dsv4-flash-megamoe-prefill-dp2tp4-ep8`，使用当前 per-channel W4A8 checkpoint、16 设备和 native 首 token 参考 JSON。环境与参考格式见 [E2E README](../../../tests/e2e/models/deepseek_v4_flash/README.md)。运行节点：

```bash
python -m pytest -q \
  'tests/e2e/models/deepseek_v4_flash/test_async_cam_npu.py::test_deepseek_v4_flash_async_cam[afd-dsv4-flash-megamoe-prefill-dp2tp4-ep8]'
```

所有请求 `max_tokens=1`，比较 token/logprob 并记录 HTTP 首 token 延迟。该结果不能替代多层 logits 回放或设备热路径性能验收；旧 256-token 完整生成 case 保持旧 backend 回归。

## 5. 当前验证与待完成门禁

已执行检查（2026-10-29）：

| 检查 | 结果与边界 |
| --- | --- |
| WSL Ubuntu / Python 3.12 `pytest tests/unit -q -o addopts=` | **617 passed、137 skipped、17 subtests passed**；Torch/NPU/vLLM 运行时依赖未安装，相关用例跳过 |
| ABI + 构建清单定向 CPU 测试 | **55 passed**，其中 21 ABI、34 构建/打包清单 |
| 公共 `abi.h` 普通 C++ 编译 | g++ C++17、`-Wall -Wextra -Werror` 通过；不含 CANN kernel 编译 |
| C++ / Python ABI 对照 | **50 组通过**；P=1/1023/1024/1025/8192，Q=1/8/128/257/4096，H=1024/7168；窗口、scratch、最大轮数逐值比较 |
| Python AST / ruff | 新增与修改 Python 文件通过 |
| 源码 provenance / include closure | 固定依赖哈希通过；CANN SDK 头由目标构建环境提供 |

发布前另增加了 packaged bytes 与 `source_manifest.json` 的哈希回归测试；包含该用例的 ABI/构建清单定向测试为 56 项。上表的全量 CPU 结果保留原始执行记录，新增用例及 Git 提交对象另行核验，不据此增加设备通过项。

Windows 首次全量测试中有旧 Linux 信号、POSIX 路径、GBK 解码及未安装插件 entry point 的失败；在 WSL CPU 环境安装本 worktree 的 editable 包（禁用 Ascend build）后全量通过。没有为取得通过而伪造 Linux 接口，也没有调用 CANN 编译器或取得 NPU 测量数据。

必须在设备环境继续验证：实际 CANN/Torch-NPU 接口编译、六表 NZ 描述符与内存共享、非对称 HCCL context 获取、多 rank Ready/Done 的发布可见性、同槽长时间复用、活跃故障/异常注入、真实 checkpoint 全层精度、旧链及原生同 payload 性能。控制脚本仅覆盖 terminal 合同，不证明控制写入时设备已经处于轮询，也不替代发布 litmus。测试门槛详见 [validation.md](validation.md)。
