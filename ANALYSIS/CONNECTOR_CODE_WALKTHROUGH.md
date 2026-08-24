# AFD Connector 代码走读

## 1. Connector 的职责边界

AFD 把一个 MoE layer 的 Attention 与 FFN 放在不同进程/设备上。Connector 的职责不是执行 Attention 或 FFN，而是在两侧之间建立通信资源，并提供四个稳定的数据面动作：

1. Attention: `send_attn_output(hidden_states, context)`
2. FFN: `recv_attn_output(...) -> AFDA2FTransferPayload`
3. FFN: `send_ffn_output(ffn_output, context)`
4. Attention: `recv_ffn_output(ref_tensor, ...) -> Tensor`

公共抽象位于 `afd_plugin/connectors/base.py`。模型代码只依赖这四个动作，因此可以在不修改模型主流程的前提下切换 NCCL、CAMP2P 或异步 CAM 后端。

## 2. 对象关系

```text
AFDTransferMetadata
  - layer_idx
  - stage_idx (ubatch/stage)
  - seq_lens
          |
          v
AFDTransferContext -----------------------+
  - metadata                              |
  - states: backend-specific state        |
          |                               |
          v                               |
AFDA2FTransferPayload                     |
  - hidden_states                         |
  - context ------------------------------+
```

`metadata` 是后端无关的逻辑描述；`states` 是后端自己的回程凭据和路由状态。Attention -> FFN 接收后，FFN runner 必须把同一个 `context` 交还给 `send_ffn_output`，否则 Connector 可能不知道如何把结果送回原 Attention rank。

## 3. 创建与初始化

`AFDConnectorFactory` 在模块加载时注册三个名字，并保存惰性导入函数：

- `P2pNcclAFDConnector`: GPU/NCCL 同步路径
- `CAMP2pAFDConnector`: NPU/CAMP2P 同步路径
- `CAMAsyncAFDConnector`: NPU/CAM 异步 dispatch/combine 路径

Attention 与 FFN 的 model runner 都会读取 AFD 配置，然后调用 `create_connector(...)`。Factory 根据配置中的 connector 名称导入并实例化对应类。之后 runner 调用 `init_afd_connector()`，真正创建 NCCL/HCCL/Gloo process group 和算子缓冲区。

惰性导入很重要：只使用 NPU Connector 时不会因为导入 GPU 后端而引入 CUDA 依赖，反之亦然。

## 4. 同步 Connector 的一次完整交换

```text
Attention scheduler/model runner                  FFN worker thread
           |                                             |
           |-- AFDControlPayload (Gloo/control plane) -->|
           |     token counts, graph/warmup flags        |
           |                                             | allocate/choose shape
layer N    |-- send_attn_output (HCCL/NCCL data) ------->| recv_attn_output
           |                                             | compute_ffn_output
           |<-- recv_ffn_output  (HCCL/NCCL data) -------| send_ffn_output
           |                                             |
layer N+1  | continue Attention                          | wait next receive
```

模型侧 `forward_with_afd` 的流水方式是：进入第 N 层前先收第 N-1 层 FFN 输出，计算第 N 层 Attention，再把结果发送给 FFN。最后一层循环结束后，再补收最后一个 FFN 输出。

同步 Connector 需要控制面的原因是：FFN 在调用接收算子之前必须知道本轮每个 rank/ubatch 的 token 数和图模式状态。Attention runner 构造 `AFDControlPayload`，本地更新一次 Connector 状态后发送；FFN worker 阻塞接收该 payload，再驱动一整个 FFN step。

## 5. CAMP2P Connector

### 5.1 通信组

`CAMP2pAFDConnector.init_afd_connector()` 创建：

- 每个 ubatch 一个 AFD HCCL group：传输 Attention hidden states 与 FFN outputs；
- FFN 侧额外的 `afd_moe` HCCL group：用于 FFN/MoE 内部通信；
- 一个较小的 Gloo `p2p` group：传输 DP token metadata。

CAMP2P 的 HCCL world rank 排列为 FFN ranks 在前、Attention ranks 在后。当前拓扑要求 `attention_size >= ffn_size`。

### 5.2 数据面

Attention 侧：

1. `send_attn_output` 校验 token 维度，创建 `CAMP2PTransferState`。
2. 将 state 放入 vLLM forward context，以便同一 forward 的回程接收复用。
3. 调用注册在 `torch.ops.vllm` 下的 `afd_camp2p_send_attn_output`。
4. 包装 op 内部调用 `torch.ops.afd_ascend.a2e`，并保存返回的 `atten_batch_size`。
5. `recv_ffn_output` 从 forward context 取回 state，调用包装后的 `e2a` 接收结果。

FFN 侧：

1. `recv_attn_output` 根据控制面的 DP metadata 计算本 FFN rank 应收 token 数。
2. 调用 `torch.ops.afd_ascend.a2e` 接收 hidden states，同时保存 active mask、Attention token 数和 endpoint 等回程状态。
3. FFN runner 调用 `compute_ffn_output`。
4. `send_ffn_output` 使用接收阶段保存的 state 调用 `torch.ops.afd_ascend.e2a`。

Attention 侧的 a2e/e2a 外面又包一层 `torch.ops.vllm` custom op，是为了给 `torch.compile`/ACL graph 一个可追踪的算子边界，并提供 fake implementation；不是用 Python monkey patch 替换函数。

## 6. 异步 CAM Connector

`CAMAsyncAFDConnector.control_plane = None`。它由 CAM 的 dispatch/combine 协议携带路由状态，不先走 Gloo token metadata。

```text
Attention                                      FFN connector-driven loop
    | compute Attention + MoE gate                       |
    | topk_ids / topk_weights                            |
    |-- async_dispatch_send ---------------------------->|
    |                                         async_dispatch_recv
    |                                         routed/shared expert compute
    |<-- async_combine_recv -----------------------------|
    |                                         async_combine_send
    | weights + combines returned expert outputs         |
```

Attention 的 `send_attn_output` 要求额外传入 `topk_ids/topk_weights`，并把 `(context, ids, weights)` 按 stage 放进本地 FIFO。随后 `recv_ffn_output` 弹出匹配项，调用 `async_combine_recv` 完成路由、加权与聚合。

FFN worker 发现 `control_plane is None` 后，不等待 metadata，而是不断调用 `execute_connector_driven_step()`。Connector 的 `recv_ffn_work_item()` 调用 `async_dispatch_recv`，从返回的 `TokenNums_Rankid_Layeridx` 读出真实 layer 和 token 数；runner 计算对应 layer 的 expert；最后 `send_ffn_work_item_output()` 调用 `async_combine_send`。

因此这里所谓“异步”的核心不只是 Python API 名字，而是 FFN 的工作由通信到达事件驱动，且 dispatch/combine 同时承担数据搬运和 MoE 路由。

## 7. 建议的阅读顺序

1. `connectors/base.py`：四动作协议与控制面协议。
2. `connectors/metadata.py`：metadata/context/state/payload 的所有权。
3. `connectors/factory.py`：配置如何选择实现。
4. `model_executor/models/deepseek_v2.py`：同步模型调用链。
5. `v1/worker/npu/ffn_worker.py` 与 `ffn_model_runner.py`：FFN 服务循环。
6. `connectors/npu/camp2p.py`：同步昇腾实现。
7. `model_executor/models/npu/deepseek_v2_async_cam_forward.py` 与 `connectors/npu/async_cam.py`：异步实现。

## 8. 走读时应持续追踪的三个问题

- 谁拥有状态：全局 Connector、forward context，还是单次 `AFDTransferContext`？
- 谁决定张量形状：Attention metadata、控制面 DP metadata，还是 CAM recv 返回值？
- 谁驱动 FFN：控制面 payload，还是异步 dispatch 到达？

这三个问题可以解释同步/异步两套 Connector 绝大多数结构差异。

## 9. vLLM Worker 与 ModelRunner 的关系

通常一个加速卡 rank 对应一个 worker 进程，进程中创建一个 Worker 对象。Worker 是设备侧的运行时外壳，ModelRunner 是它持有的模型执行组件：

```text
EngineCore / Scheduler
        |
        | SchedulerOutput
        v
Executor / distributed RPC
        |
        v
Worker（每个设备 rank）
  - 初始化设备与分布式环境
  - 管理显存/KV cache/生命周期
  - 接收上层执行命令
  - self.model_runner
             |
             v
       ModelRunner
         - 加载并持有 model
         - 准备模型输入和 forward context
         - 选择 eager/graph 执行
         - 调用 model.forward
         - 整理模型输出
```

两者是组合关系而不是并列关系：`Worker` owns/contains `ModelRunner`。同一个模型并不是 Worker 和 ModelRunner 各保存一份；实际模型对象一般由 ModelRunner 持有。

AFD Attention 仍保留 scheduler 驱动方式，只是 Worker 将原生 runner 换成 AFD-aware runner。AFD FFN 则没有请求调度和 KV cache：它拒绝默认 `Worker.execute_model(SchedulerOutput)`，由 Worker 启动 connector server loop，收到 Connector 控制消息或 CAM work item 后调用 FFN ModelRunner。因而 AFD FFN 仍是“Worker 管循环、ModelRunner 管计算”，只是触发源从 scheduler 变成了 connector。

## 10. `rank` / `local_rank` 与 `context` / `kwargs`

Runner 从 vLLM `get_world_group()` 取得 `(group.rank, group.local_rank)` 传给 Connector。`rank` 是当前 vLLM 分布式 world 中的进程编号；`local_rank` 是当前节点上的设备编号。两个独立的 Attention/FFN deployment 都可能各自有一个 vLLM `rank=0`，所以它不等于合并后的 AFD world rank。

Connector 使用 `afd_config.afd_role_rank` 表示本角色内的编号，并由后端 topology 计算 AFD `world_rank`。P2P/CAMP2P 与 async CAM 的 combined-world 排列甚至可以不同，因此 AFD 通信组创建必须使用后端算出的 `world_rank`。当前实现中构造参数 `rank` 主要作为调用方身份保存在基类；`local_rank` 会实际用于选择 CUDA/NPU device。

`AFDTransferContext` 是一个有类型、跨调用阶段传递的单次 transfer 上下文：`metadata` 保存 layer/stage/token layout，`states` 保存接收后回发仍需要的后端状态。它通常只是本进程内的 Python 对象，并不代表整个对象会被网络序列化到对端。

`**kwargs` 是单次方法调用的可选、后端专属扩展参数。P2P 不使用它；CAMP2P receive 使用 `layer_idx/max_num_tokens` 作为创建 context 前的提示；async CAM send 使用 `topk_ids/topk_weights` 作为 dispatch 的额外输入，send-FFN 使用 `expand_x_shared`。若某项状态必须从 `recv_attn_output` 一直活到 `send_ffn_output`，应放进 `context.states`，而不应依赖调用方再次通过 kwargs 传入。

## 11. CAMP2P 控制面的频率与关键路径

稳定态下，CAMP2P 控制面按 vLLM execution step 发送一次，而不是每层发送一次。一份 `AFDControlPayload` 已包含该 step 全部 stage/ubatch 的 token metadata；FFN 收到后进入一次 `execute_ffn_step()`，在内部循环所有 layer 和 stage。每层重复的是 A2E/E2A 数据面传输。

CAMP2P 控制面经 CPU/Gloo 同步发送。当前 wire protocol 先发送一个长度 Tensor，再发送 JSON 编码后的 `uint8` payload，并对目标 FFN rank 顺序发送。FFN 后台线程通常已阻塞在 receive，因此没有应用层 request-response 往返，但 Attention 的同步 send 位于 model forward 之前，是稳定态首层数据面开始前的关键路径开销；代码当前也没有用异步 `isend` 与 Attention 计算重叠。其固定延迟可能在小 batch decode 时更显眼，但被一个 step 的所有层共同摊销。warmup/graph capture 会执行额外的 dummy step，因此初始化阶段可能另外发送控制 payload；这仍不是每层发送。

## 11. Connector ExtraInfo 与控制面

`ConnectorExtraInfo` 是 Connector 私有的、带类型的静态配置。公共 AFD 配置（role、connector、host/port、Attention/FFN rank 数等）进入 `AFDConfig`；只有某种通信后端才认识的选项放在 `additional_config["afd"]["connector_extra_config"]`，由具体 Connector 的 `parse_extra_config()` 转换为其 ExtraInfo dataclass。CAMP2P 的例子包括 AIV core 数和 quant mode；async CAM 包括 dynamic quant、每个 DP 的 Attention rank 数和 async MoE ubatching。P2P 当前不支持额外选项并拒绝非空配置。

控制面与搬运 hidden states/FFN output 的数据面分离。同步 Connector 的 `AFDControlPayload` 每个 step 携带各 stage 的 DP token 数以及 graph-capture/warmup 标志。Attention 侧先本地 `update_state_from_dp_metadata()` 并发送 payload；FFN worker 阻塞接收 payload，以它作为工作到达通知，再由 FFN runner 本地更新 Connector 状态并执行 FFN。状态更新可推导动态 tensor shape、选择/预分配稳定 buffer，并对齐图捕获模式。实际激活值仍由 Connector 数据面传输。

控制面是逻辑职责而不是固定传输技术：GPU P2P 使用设备侧 NCCL control group，CAMP2P 使用 CPU/Gloo。Async CAM 没有独立控制面，因为 dispatch 接收结果本身已经携带 token、rank、layer 和路由信息，并直接驱动 FFN work-item loop。
