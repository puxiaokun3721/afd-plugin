# A3 CAMAsync MegaMoE 算子、内存与协议接口

状态：接口设计稿，首版四算子实现已生成；可执行入口与当前验证边界见 [实施说明](implementation.md)，共享 ABI 以仓内 `abi.h` / `mega_moe_abi.py` 为准。范围、固定源码和原生阶段映射见 [总体设计](README.md)。部署与 framework 封装见 [接入设计](integration.md)。设备验收尚未完成。

## 1. 接口分层

四个算子的名字及职责固定如下。Scheduler 不把 token 拉到计算工作区，Ffn 不重新向全 EP 发起 count allgather。

| CANN op | Torch 注册名（拟） | 执行核 | 职责 |
| --- | --- | --- | --- |
| AfdMegaMoeDispatchSend | `afd_ascend::afd_mega_moe_dispatch_send` | AIV | 原生 prologue、整 MB 持久化、A→F count/Ready 发布 |
| AfdMegaMoeScheduler | `afd_ascend::afd_mega_moe_scheduler` | AIV | 本 F 轮询完整 DP、验证 task identity、输出设备工作描述 |
| AfdMegaMoeFfn | `afd_ascend::afd_mega_moe_ffn` | MIX_AIC_1_2 | 设备选层、原生 pull/MSD/GMM/epilogue、第二层分轮、Done |
| AfdMegaMoeCombineRecv | `afd_ascend::afd_mega_moe_combine_recv` | AIV | 等所有 F Done、原生 unpermute/top-k 加权、A 槽回收 |

Python `MegaMoeContext` 是每个 connector 实例拥有的资源对象，持有静态 config、通信域引用、拓扑、状态、workspace 和持久输出；不引入模块级可变全局状态。高层方法可简化为：

```python
context.dispatch_send(x, expert_ids, token_mask, task_meta)
context.schedule()                       # 设备输出 work_desc/source_desc
executor.execute(context)                # Ffn 消费上述描述，设备选层
context.combine_recv(topk_weights, task_meta, out)
```

底层注册接口必须全部展开成 tensor/attribute；不把 Python context、任意字典或裸 Python int 指针穿过 operator ABI。创建通信域、构造 TensorList、加载权重、计算 layout 是初始化流程，不算第五个每任务算子。

## 2. 公共配置与 tensor 合同

### 2.1 静态配置

所有算子共用 `group_name: str`、`config: int[]` 和 `topology: Tensor`。`config` 是 host 不可变的 24 个 int64 属性，CANN 对应 ListInt/`aclIntArray`，启动即验证长度、版本和取值；kernel tiling 保存解码后的命名字段，热循环不解释整数数组。0.26 的 grouped_matmul_layered 已有 ListInt 定义，`op_api_common.h` 已有 IntArrayRef→aclIntArray 转换；v1 保持这一套属性 schema，不另设等价但不同签名的分支。

| config 下标 | 名称 | 合同 |
| --- | --- | --- |
| 0 | abi_version | 1 |
| 1 | input_layout_version | 1：expanded routes |
| 2 | quant_mode | 1：per-channel W4A8/MSD |
| 3 | session_epoch | 启动握手一致，重建通信域时更换；非每任务时间戳 |
| 4 | physical_rank | HCCL domain 中本 rank，不能拿 dp_id 代替 |
| 5,6,7 | attn_dp_size, attn_tp_size, ffn_ep_size | D、T、F；F TP=1 |
| 8 | local_expert_count | e；全局 E=F×e；首版连续均匀专家分配 |
| 9,10,11 | hidden_size, intermediate_size, topk | H、I、K；I 为激活后宽度，W13 输出 2I |
| 12,13 | source_token_capacity, recv_route_capacity | P_cap、Q，单位分别是 token、route 行 |
| 14,15 | layer_count, layer_id_capacity | L、global-layer→slot LUT 长度 |
| 16 | feature_flags | v1=0，未知 bit 拒绝 |
| 17 | planned_window_bytes | layout planner 计算的公共 winIn 大小；必须小于等于实际 context winSize |
| 18 | inflight_per_source | v1=1 |
| 19 | comm_round_capacity | v1=1 |
| 20…23 | reserved | 0 |

角色从 topology 得到。`topology` 为连续 NPU INT32 `[D*T+F,4]`，列为 `(role, dp_id, tp_or_ffn_idx, physical_rank)`。A 行按 `(dp,tp)` 排，F 行按 ffn_idx 排；F 的 dp_id=-1。每个物理 rank 恰好出现一次，A/F 不重叠。首版建组仍可沿用 `[所有A,所有F]`，但 kernel 使用显式映射，避免逻辑编号与物理编号耦合。

`comm_args` 为 FP16 `[1]` placeholder，沿 .26 CAM 的兼容入口传递，由 wrapper 创建，Python 不写 peer 地址。现有 CAM MC2 路径把它作为 ABI placeholder；新算子也从经验证的 MC2/KFC context 取得实际 peer bases/winSize。旧 CAM CommArgs 的结构不能直接当 native HcclShmem 的 context 强转，需公共适配器核对地址表、角色映射及实际窗口边界。初始化遇到不支持的 context 类型立即拒绝，不自行切到 A5 channel 路径。

### 2.2 持久 tensor 与所有者

除注明为 host 属性外，以下均为本 rank NPU tensor。所有 scratch/output 在初始化按上界预分配；每任务不重新分配大 tensor。

| tensor | dtype / shape | 所有者、读写与寿命 |
| --- | --- | --- |
| state | INT64 `[32+D]` | 本 rank connector；四 op 按角色读写；同任务流串行；会话生命周期 |
| control | INT32 `[128]`，512B | 生命周期控制路径写，轮询算子读；与任务流分离，见 §8 |
| work_desc | INT64 `[32]` | F Scheduler 写，Ffn 读；保持到 Ffn 返回 |
| source_desc | INT64 `[T,16]` | Scheduler 对所选 DP 的 TaskMeta 做稳定快照；Ffn 读 |
| route_index | INT32 `[P_pad*K]` | A Send 写全 MB 每 tile 的原生 expandedRowIdx；Combine 读；一个未完成 MB |
| source_counts | INT32 `[J_cap,align128(E+1)]` | A Send 写；随后发到各 F inbox；保留便于诊断/单算子验证 |
| send_scratch | UINT8 `[send_bytes]` | A prologue 的私有临时区，保留原生布局/对齐 |
| ffn_scratch | UINT8 `[ffn_bytes]` | F 私有 table、MSD/GMM/同步工作区；同 F 一次一个任务 |
| completion | INT64 `[8]` | Ffn/Combine 写 status、error、identity、处理行数；只供低频健康检查/测试读取 |
| out | BF16 `[B,H]` | A Combine 写最终本地 token 输出；调用方拥有 |

`P_pad/J_cap` 定义见 README §6.2。scratch 的有效起始地址按原生 region alignment 对齐，不只是检查 `data_ptr % 32`；host allocator 与 kernel 共用 offset 定义。kernel 可能还需要 CANN 自身的辅助 workspace，由 GetWorkspaceSize 单独返回，不能又重复申请一整份 ffn_scratch。

state 的前 32 项是 header：0…8 依次为 `abi_version, session_epoch, run_state, fatal_code, rr_cursor, active_dp, active_task_seq, active_ticket, last_combined_seq`，9…31 保留零；32+d 是 F 的 `completed_seq[d]`。run_state 枚举为 `IDLE=0,PUBLISHED=1,CLAIMED=2,RUNNING=3,COMPLETE=4,STOPPED=5,ERROR=6`，A 的 FREE 对应 IDLE。completion 的0…7依次为 `abi_version,status,error_code,session_epoch,task_seq,dp_id,global_layer_id,routes_processed`，status 为 `PENDING=0,SUCCESS=1,STOP=2,ERROR=3`。未使用字段置零，不依赖残留内存。不要让 A 和 F 用同一个下标写不同类型的动态地址；offset/地址来自静态 planner/context。

`completed_seq[d]` 仅本 F 的 Ffn 正常发完 Done 后更新。Scheduler 只读取并写 active claim/rr_cursor。A 的 `last_combined_seq` 仅 Combine 正常结束后更新。所有状态写者都有明确的同 stream 顺序，不使用跨进程共享 Python 状态。

## 3. TaskMeta、WorkDesc 与通信记录

### 3.1 TaskMetaV1：INT64[16]

| 下标 | 字段 | 解释 |
| --- | --- | --- |
| 0,1 | abi_version, session_epoch | 必须匹配 context |
| 2,3 | task_seq, publish_ticket | 每 DP 按实际 MoE dispatch 次序递增；v1 ticket=task_seq，范围 1…2^31-1 |
| 4,5,6 | dp_id, global_layer_id, microbatch_id | 一个任务的身份；同 DP 全 TP 必须一致 |
| 7 | input_token_count | 本 A 的 B，等于 x 的行数，0≤B≤P_cap；TP 之间可以不同 |
| 8 | source_physical_rank | 校验发布者确属该 DP/TP，不从数据推导任意 peer 地址 |
| 9,10,11 | comm_round_idx, comm_round_count, token_base | v1 只允许 0、1、0 |
| 12 | source_total_tokens | 本 A 完整 MB 的 B；v1 等于字段7 |
| 13 | input_layout_version | 1 |
| 14,15 | flags, reserved | 0 |

task_seq 是通信任务序号，**不是 layer_id**，同一个 layer 多次运行也必须新序号。由 A 侧 DP runner 的确定性 dispatch 顺序生成，全部 TP 在已有相同调度元数据下生成一致值；不在 F 侧猜测，不引入每层全域 host collective。纯 TP padding 的 source 仍发送 B 和 mask；完全空的 source 发送 B=0 的 Ready。

ticket 不允许绕回。接近上限时先 drain、重建 domain/epoch、清空记录后从1开始；首版不设计 modulo alternating bit 复用。64位 task_seq 放在稳定 descriptor body，**不假设远端64位写原子**。

### 3.2 WorkDescV1：INT64[32]

| 下标 | 字段 |
| --- | --- |
| 0…3 | abi_version, session_epoch, kind, error_code |
| 4…8 | dp_id, task_seq, publish_ticket, global_layer_id, microbatch_id |
| 9…13 | num_sources=T, num_tiles, ffn_idx, local_expert_count, recv_route_capacity |
| 14…19 | comm_round_idx, comm_round_count, token_base, input_layout_version, payload_slot=0, flags=0 |
| 20…31 | reserved=0 |

kind：`EMPTY=0`（尚未发布）、`WORK=1`、`STOP=2`、`ERROR=3`。Scheduler 正常返回必为后三种之一；没有逐次短轮询 NOP 被 host 无限制排队。`num_tiles=max_s ceil(B_s/C_cap)`；全空 DP 仍输出 WORK，可由 Ffn 一致走零路由完成路径。某 source 在某 tile 已无 token 时其 count 全零，不访问其 payload。

source_desc 保留每个 source 的 TaskMeta；两者仅是本 F 私有 kernel 间接口，不广播给其他 F。WorkDesc 的 kind 最后写出，并以本卡正确的 MTE/scalar fence 结束 kernel；后续同 stream Ffn 才启动。

### 3.3 远端 Ready/Done 提交记录

每个写者拥有独立 512B 记录，起始按512对齐，commit 位于独立的首个128B块；body 从偏移128开始。commit 是一个非零 INT32 ticket，必要的 DMA 最小搬运粒度内其余字节为保留零值，不与其他写者共享该块。body 使用显式字段 offset、静态大小断言，禁止依赖编译器 struct padding。

- Ready body：epoch、task_seq、source_rank、dp/layer/mb 和 descriptor/count 布局版本。完整 TaskMeta 另有独立512B descriptor 区。
- Done body：epoch、task_seq、dp/layer/mb、ffn_idx、`SUCCESS/ERROR`、error_code、可选诊断 routes_processed。
- counts 不再带 `0x800000` readiness sentinel。count 本身为非负行数，readiness 单独由 commit 表示；padding（含原 E+1 保留位置）置零。
- producer 先写数据和 body、完成规定的本地与远端可见性顺序，再独立提交 commit。consumer 只接受 epoch、ticket、identity 全匹配的稳定记录。
- consumer 不能用一次普通 scalar load 永久缓存 flag；复用当前可验证的 MTE 轮询/缓存维护方法。

此结构避免把64位 metadata 原子性当设计前提，但**不自动证明**32位 commit 的可见性或 payload→commit 次序。必须完成 [知识附录](knowledge_evidence.md) 与 [测试设计](validation.md) 的真实跨设备 litmus。目标工具链若无法保证选定发布原语，先更换经过验证的发布 primitive/记录提交方式再继续；不能靠添加 `volatile` 或一个本卡 SyncAll 宣称修复。

F inbox 为每个 A 预留 `ReadyRecord + TaskMetaRecord + counts[J_cap,E_pad]`。A window 为每个 F 预留 DoneRecord。所有参与者在启动时对 version、epoch、topology、容量、专家布局和量化模式握手一致。不同协议不可复用旧 CAMAsync 的 magic/窗口布局。

ReadyRecord/DoneRecord 共用下表的精确字节布局；TaskMetaRecord 为 TaskMeta 的128B连续内容后补384B零。所有多字节值使用目标 A3 ABI 的小端表示，序列化 helper 与测试共享字段定义，不能以 reinterpret struct 代替字段断言。

| 字节偏移 | 类型/字段 | 写者与取值 |
| --- | --- | --- |
| 0 | uint32 commit_ticket | Ready=A、Done=F 的独占写者，最后提交；0表示未提交 |
| 4…127 | reserved | 全零，不放其他发送者的状态 |
| 128,136 | int64 epoch, task_seq | 同一任务身份 |
| 144,152,160 | int64 dp_id, global_layer_id, microbatch_id | 与 TaskMeta/WorkDesc 一致 |
| 168 | int64 producer_physical_rank | 必须与该记录槽所属 A/F 匹配 |
| 176,184 | int64 status, error_code | Ready：0/0；Done：SUCCESS=1或ERROR=3，与completion使用同一命名状态枚举；Done禁止STOP |
| 192 | int64 routes_processed | Done 的诊断值；Ready 为0；不用于缩小完成参与者 |
| 200,208 | int64 input_layout_version, abi_version | v1均为1 |
| 216 | int64 producer_role_index | Ready 为逻辑 A index=dp*T+tp，Done 为 ffn_idx |
| 224…511 | reserved | 全零 |

### 3.4 字节偏移规划

所有 rank 使用相同 P_cap/E/J_cap/topology 和一个公共 layout planner。记 `align512(x)=ceil(x/512)*512`，`reserved=10MiB`，`tail=max(512*world_size,512KiB)`。A 窗偏移为：

```text
x_base     = reserved
scale_base = align512(x_base + P_pad*K*(H+512))
d_base     = align512(scale_base + 4*P_pad*K)
done_base  = align512(d_base + 2*P_pad*K*H)
W_A_min    = done_base + 512*F + tail
X(tile,row) = peerA + x_base + (tile*C_cap*K+row)*(H+512)
D(tile,row) = peerA + d_base + (tile*C_cap*K+row)*2*H
```

F 窗偏移为：

```text
E_pad        = align128(E+1)                 # 单位为int32元素
inbox_stride = 1024 + 4*J_cap*E_pad
ready(a)     = reserved + a*inbox_stride
meta(a)      = ready(a) + 512
counts(a,j)  = ready(a) + 1024 + 4*j*E_pad
W_F_min      = reserved + A_total*inbox_stride + tail
W_required   = align512(max(W_A_min,W_F_min))
```

本域实际 context winSize 必须统一且≥W_required，底层尾部保留区从实际 winSize 倒排，A/F 所有已分配区域都不得侵入。config 的 planned_window_bytes=W_required；实际更大的 HCCL 窗保留中间空洞，不改变 X/D/inbox 的固定偏移。上式不含另一份 HCCL winOut，也不含 private workspace，实际显存计量见 README。

## 4. 四算子具体签名与行为

以下是**完整参数合同**的 Torch schema 形式；`(a!)` 等为 mutation alias 标记。CANN `GetWorkspaceSize` 按相同顺序接收 tensor、TensorList、属性、可写输出，另加 workspaceSize/executor；执行接口保持标准 `(workspace, workspaceSize, executor, stream)`。CANN/ACLNN 的 const tensor descriptor 不代表其指向的数据不可写。

### 4.1 AfdMegaMoeDispatchSend

```text
afd_mega_moe_dispatch_send(
    Tensor x, Tensor expert_ids, Tensor token_mask, Tensor task_meta,
    Tensor comm_args, Tensor topology, Tensor control,
    Tensor(a!) state, Tensor(b!) route_index,
    Tensor(c!) source_counts, Tensor(d!) send_scratch,
    *, str group_name, int[] config
) -> ()
```

| 输入 | 合同 |
| --- | --- |
| x | 连续 BF16 `[B,H]`；不接受隐藏 stride 或 FP16 自动转换 |
| expert_ids | 连续 INT32 `[B,K]`，全局专家 id `[0,E)`；首版沿模型 top-k 唯一专家语义 |
| token_mask | 连续 BOOL `[B]`；valid=true。padding/无效 token 不产生任何路由 |
| task_meta | §3.1；device tensor，host 提供身份，kernel 检查与实际 tensor 形状一致 |

顺序：

1. 检查当前 A 槽为空/上个 task 已 combine；检查静态参数、任务身份、B、mask/routing 合同。错误不发布正常 Ready。
2. 清空本次所有 unused tile counts；依次对实际 tile 调用原生 prologue。将原来单 tile 复用的 X、rowIdx 改成固定 tile 基址，保留量化和排序实现。
3. 原生 rowIdx 的 tile 内语义保持原样。A Combine 按相同 tile 调用 unpermute，不把所有 tile 拼成一次未经适配的全局 rowIdx。
4. 每个 AIV 完成自己写出的数据，经过本卡 fan-in 确认全 MB payload 和 counts 完成；向所有 F 发送本 A 的全局专家 count 向量与 descriptor。
5. 每个目的 F 的数据发布条件满足后，最后写对应 Ready commit；更新 A state 为 PUBLISHED。

Send 不等待 F 读取，不等待 Done。返回只表示本次发布操作在 A stream 上结束。top-k weights 不作为输入，保留在 A Python 上下文直到 Combine；不存在 F 上再次乘 routed scaling 的逻辑。

### 4.2 AfdMegaMoeScheduler

```text
afd_mega_moe_scheduler(
    Tensor comm_args, Tensor topology, Tensor control,
    Tensor(a!) state, Tensor(b!) work_desc, Tensor(c!) source_desc,
    *, str group_name, int[] config
) -> ()
```

每次返回一个工作项。设备内部 round-robin 扫描全部 DP：

1. 只看本 F inbox；对候选 DP 的 T 个 Ready 检查提交、epoch 与 `task_seq=completed_seq[dp]+1`。已完成旧 ticket 跳过，新序号跳跃视为协议错误。
2. 候选 DP 缺成员时立即扫描下一个 DP，整轮都无 WORK 才继续轮询；不能在第一个缺成员的 DP 上阻塞。完整 TP 集合中，已提交当前 ticket 的 dp/layer/mb/seq/round/layout 必须一致；B 允许不同。不一致直接 ERROR，不能等待不可变 descriptor 自愈，也不能只取首个 TP 的 layer。
3. 满足条件则输出 source_desc、WorkDesc=WORK；推进 rr_cursor，记录本地 active claim。只访问控制 metadata，不读取大 token payload，也不做 F-F 同步。
4. 无任务时保持 IDLE 轮询，检查独立 control 的 STOP/ABORT。空闲没有请求是正常状态，不能用“多久没任务”作为通信错误。

已有 active task 未结束时不得再 schedule；同 F 固定 `Scheduler → Ffn → Scheduler`。协议错误输出 ERROR 并设置本地 fatal 状态，由错误传播/域终止流程处理，不能把缺 TP 当成空路由。可为“已有部分 TP Ready 长期不齐”记录诊断和 host watchdog；超时只可报错，不能跳过后执行部分 DP。

### 4.3 AfdMegaMoeFfn

```text
afd_mega_moe_ffn(
    Tensor work_desc, Tensor source_desc,
    Tensor layer_id_to_slot,
    Tensor[] w13, Tensor[] w2,
    Tensor[] weight_scale13, Tensor[] weight_scale2,
    Tensor[] compensation13, Tensor[] compensation2,
    Tensor layer_params,
    Tensor comm_args, Tensor topology, Tensor control,
    Tensor(a!) state, Tensor(b!) ffn_scratch, Tensor(c!) completion,
    *, str group_name, int[] config
) -> ()
```

量化 tensor 合同：

| 输入 | 合同 |
| --- | --- |
| layer_id_to_slot | INT64 `[layer_id_capacity]`；合法全局 layer 映射到 `[0,L)`，空洞=-1 |
| w13/w2 | 两个长度 L×e 的 TensorList，索引 `slot*e+local_expert`；逻辑 shape `[H,2I]` / `[I,H]`、native INT4 NZ 物理格式 |
| Torch 权重承载 | 通常为 packed INT32 tensor；ACLNN 用与 native 一致的 INT4 view/format 元信息，不能只改 dtype 就当作 ND→NZ |
| weight_scale13/2 | 同样 L×e，每项1D `[2I]` / `[H]`；native UINT64 编码，Torch 可用 INT64 保持位模式，不作数值 cast |
| compensation13/2 | 同样 L×e，每项 FP32 `[2I]` / `[H]`；W4 MSD 补偿，必填，非可任意省略的普通 bias |
| layer_params | FP32 `[L,4]`：swiglu_limit、activation_code、activation_param1/2；首版仅允许模型已支持的 SwiGLU 组合，code 必须可精确表示为整数 |

所有层的 H/I/e、量化模式和计算模板一致；不一致在模型加载时拒绝，首版不做异构层混合 tiling。非 MoE 层不放进 L，但全局 layer_id 可以不连续。

入口处理与计算：

1. 全体 AIC/AIV 在进入第一处混核 wait 前，读取同一不可变 WorkDesc。STOP/ERROR 走全核一致退出，不允许部分核进入正常路径。WORK 检查身份、source 描述、Q 与容量。
2. 解析 layer slot。非法 global id 或 LUT=-1 全核一致错误，禁止沿旧 layered helper 的 clamp-to-zero 行为。选择所有 weight/scale/compensation/prefetch 基址及 layer_params；同一选择用于 GMM1、激活、GMM2、epilogue。
3. 为所选 DP 的每个 tile，从各 source 的 F inbox 逐行收集全 E count 向量到私有连续 `[T,E_pad]` staging，再求全局专家 source prefix、提取本 F e 个专家，复用原生 cumsum 与 UB route cache。staging 的 source stride 才与原生 helper 相同，不能把包含整 MB/control 的 inbox 直接传入。`num_sources=T`、全局专家宽 E 和本 F expert offset 分别传递，不沿用原生 EP 一参多义。**不再次发任何全域 count 消息**。
4. 每 tile 用 Q 切分路由。保留三段 round table、独立 GM table 地址和原生 AIC/AIV 轮数一致性。源端行坐标公式见 README §5.1，读 X 和写 D 使用同一坐标。
5. 复用原生 `ProcessRecvRound`、GMM1/GMM2、W4 epilogue、weight prefetch、flag drain。只改角色/源地址映射、动态层指针与必要 count/offset 适配。
6. 最后一个 tile/round 完成后，确认全部 remote read、remote write 及 AIC/AIV 交接结束，向该 DP 全 T 个 A 写 SUCCESS Done，零路由也执行；最后更新本 F completed_seq 和 completion。

Ffn 没有大 hidden_states 输出，避免把 DR/GMM/CS 间的临时 tensor重新暴露到 Python。输出直接在 A window 中；completion 是小型状态。第二层轮数仅 F 本地已知，不要求各 F 相同。

原生未拆分输入/输出 region 的 overlay 关系必须保留。若抽取代码需要新增最后一次 AIC→AIV 完成确认，须先枚举现有 flag 及 SyncAll 占用，挑不冲突的保留槽并配对消费；禁止省略原生尾部等待或在每个 expert 添加新的全核 barrier。

### 4.4 AfdMegaMoeCombineRecv

```text
afd_mega_moe_combine_recv(
    Tensor topk_weights, Tensor token_mask, Tensor task_meta,
    Tensor route_index, Tensor comm_args, Tensor topology, Tensor control,
    Tensor(a!) state, Tensor(b!) out, Tensor(c!) completion,
    *, str group_name, int[] config
) -> ()
```

topk_weights 为 FP32 `[B,K]`，token_mask 与 Send 相同；out 为 BF16 `[B,H]`。Send 的 x/expert_ids 可以按 stream 生命周期正常释放，但 route_index、mask、weights、task_meta、peer X/D 必须保持到 Combine 完成。

Combine 等本 A 的全部固定 F Done 匹配当前 epoch/seq/identity。任何 F ERROR 立即进入错误路径；不能因零 count、F 未收到某专家或超时而缩小参与者集合。确认完成后按 tile 复用原生 unpermute/加权合并，padding token 输出零，遵守模型路由权重语义。

out 写完及所需本地同步完成后才把槽状态改为可复用，更新 last_combined_seq。下一 Send 在同一 A 通信顺序中排于 Combine 之后；若因 Attention 计算流与通信流分开，需要现有 stream event 建依赖，不插入 host `synchronize()`。Done 记录可保留旧 ticket，下一任务按新 ticket 区分；禁止为了清标志额外引入一次全 F handshake。

## 5. ACLNN、Torch 与资源生命周期

- 每个 op 提供 `aclnnAfdMegaMoeXxxGetWorkspaceSize(...)` 和 `aclnnAfdMegaMoeXxx(...)`，由现有 custom OPP + pybind 打包；TensorList 的 host 签名和 kernel list descriptor 必须一致。
- schema 对 state/scratch/输出明确 mutation；HCCL window 是外部资源，还要将这些 op 标记为有通信副作用，不允许 functionalization/compiler 重排、删除或复制。当前只承诺 eager；Meta/Fake 注册即使存在也不代表图支持。
- 不允许同一个 context 在两条任务 stream 上并行调用 Send、Scheduler 或 Ffn。control 的独立管理路径是专门的协议例外，不是普通 tensor 并发写许可。
- workspace 和 TensorList 地址在 kernel 完成前保持；host 缓存 tiling/executor 时 key 需包含角色、shape/layout、Q、core 参数、context epoch，不能把 task_seq/layer runtime值错误固化在缓存里。
- 所有 `GetWorkspaceSize` 中 shape 检查是 metadata 检查，不 `.cpu()` 读 work_desc；实际 task 的 token/层/round 合法性在 device 检查。
- 停止并 drain 后才能释放通信域/窗口、TensorList 和 workspace。异常失联时通过运行时 abort/进程退出释放，不冒充正常 Done。

## 6. 关键状态机与不变量

```text
A：FREE -> [Send] PUBLISHED -> [Combine wait + unpermute] FREE
F：IDLE -> [Scheduler complete DP] CLAIMED -> [Ffn] RUNNING
        -> [all writes drained + Done] IDLE
任一角色：fatal -> ERROR -> domain abort / drain-and-reinitialize
```

必须成立的不变量：

1. A 只有一次未 combine 的任务；一个 DP 的全部 TP 用同一 task_seq，但每个 A 的输入行数可以不同。
2. F 对某 DP 同一 seq 只执行一次；旧 Ready 即使未清也不会被重新消费。
3. A payload/counts 在所有 F 使用期间不可变；F 不按某一 receiver round 发提前释放 ACK。
4. 第 r 轮读/写的每条 route 都在当前 source tile `[0,C_cap*K)` 内；本 F 对应的合法路由恰好覆盖一次，无遗漏无重写。
5. 已发布 SUCCESS Done 意味着本 F 不再读当前 X、不再写当前 D，并且所用计算 scratch 的所有核间消费者已结束。
6. A Combine 成功意味着所有 F SUCCESS + 最终输出完成；只等 Done 不等于已经释放 route_index/out 的消费依赖。
7. 零路由 F、空 source 也属于固定参与者；完整零 token DP 的任务必须能够完成。
8. 原生 mixed-core flag 的 set/wait 数量、模式与复用顺序一致；所有分支在第一次混核 wait 前统一判定。

## 7. 静态检查、溢出与诊断

启动静态检查包括：A3 arch22、同一受支持 supernode/context、拓扑一一对应、F TP=1、连续均匀专家映射、per-channel W4、BF16、K范围、H/I对齐、原生 W4模板支持的shape、完整权重/scale/补偿、LUT、Q>0、P_cap>0、所有窗口及 scratch 大小。

原生 host 对 H 的范围为1024…8192且512对齐、K≤16、本地专家≤128等；实现要逐项重新核对固定版本的 validator，而不是直接复制对称 EP 的约束到新的 world_size。source 数 T 与 F EP 数各自校验，不能只检查 A+F 属于原生 EP 列表。

还要取 kernel 静态缓冲的约束交集：所分析 `moe_permute_prologue.hpp` 的 `MAX_TOKENS=1024`、`MAX_EXPERTS=1024`，因此首版按全局 E≤1024 admission；不能从 F≤128、e≤128 相乘推导出支持 E=16384。模型预检先确认目标 E，超出范围需另行审查该原生组件的容量扩展。mask 复用其 x_active_mask 分支：无效 route 的 rowIdx=-1，原生 unpermute 跳过无效索引；仍要验证完全无效的输出为零。

所有容量加乘使用 checked uint64；转换成原生 int32 counts/offset 前验证不溢出，DMA 长度还须满足原生字段宽度与对齐上限。不得用截断或 clamp 隐藏超出 Q/P_cap 的输入。Fnum/source数量增大会影响 UB route cache 和三段 round table 的 UB临时量，须延续原生 UB 预算检查，不靠运行时越界试错。

建议命名错误码：`INVALID_CONFIG`、`WINDOW_TOO_SMALL`、`UNSUPPORTED_WEIGHT_LAYOUT`、`INVALID_LAYER`、`TASK_ID_MISMATCH`、`SEQUENCE_GAP`、`ROUTE_OUT_OF_RANGE`、`WORKSPACE_OVERFLOW`、`PEER_FAILED`、`ABORTED`。CANN host错误直接返回失败；设备错误写 completion/fatal record并传播域失败。错误记录含 epoch/seq、dp、source/ffn、layer、tile、round、stage，禁止把地址或内部计数伪装成正常返回值。

## 8. 轮询、停止与失败传播

Scheduler 是阻塞轮询 kernel，A Combine 在有未完成任务时也会阻塞。正常 IDLE 不设“必须有新任务”的超时。worker 低频检查健康状态不参与每任务选层或计算调度。

`control[0]` 定义 STOP_REQUEST，`control[32]` 定义 ABORT_CODE，分别占独立128B块，其他项保留。host shadow 通过受验证的独立 runtime copy/管理路径更新 device control；**绝不能排到正在阻塞的同一 task stream 后面**。必须验证该路径不需要被当前 kernel 占满的计算资源，且 polling 可观察更新。普通 torch 运算排入另一个 stream 并不自动证明能够抢占/唤醒。

STOP 只阻止下一个新工作项：已 CLAIMED 的合法任务完成后再退出。正常 shutdown 先停止新的 A dispatch，drain 已发布任务，再请求 F STOP。Ffn 不在 GMM1/GMM2 的中间随意响应 STOP，避免破坏原生 flag 配对。

ABORT/peer failure 与 STOP 分开处理。失败不能发布 SUCCESS Done，也不能重新用相同槽“重试”。必要时 F 向相关 A 的错误 body 发布 fatal 状态，host 健康路径终止整个 domain；尚无匹配 task 的其他 rank 由域级错误传播唤醒。若控制更新路径在目标运行时无法保证进展，以显式域 abort/worker 退出为失败收敛机制，不能承诺尚未验证的设备内优雅取消。

首次启动 warmup 必须先完成全域建组/布局握手，再由真实 A/F 两侧匹配调用四 op。不能在 F 单方 warmup 中启动等待数据的 Scheduler，也不能沿旧小算子 warmup 顺序误触发新协议。

## 9. 扩展兼容性

v1 拒绝任何非默认第一层分轮字段、非 expanded input layout、未知 feature flag、跨 DP 合批、跨 MB 合批。预留字段是为了以后不丢掉任务身份，不等于这些特性已经可用。

未来 unique-token 输入通过 `input_layout_version=2` 选择独立的 route-gather loader；source offset 在读 X 时变成 token_idx，在写 D 时仍是 route_idx，必须将两者分开。其后的 MSD/GMM/激活仍尽量复用 v1。新增第一层分轮要升级协议状态机与 ACK 语义，并扩大测试矩阵；不以仅修改 count 表的方式暗中打开。
