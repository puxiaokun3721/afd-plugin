# A5 AFD FFN 计算事务与通信内存设计

本文按 Attention 发数据、FFN 选批和计算、Attention 收回结果的顺序描述通信窗口和算子交接。第 3、4 节给出通信内存布局，第 6、7 节解释 Active 和计算事务，第 8 节通过 CPU 例子推导地址。v1 源码索引、实际四卡全景和一条 token 的寻址见[纯 UMDK 2A2F 联调指南](a5_afd_ffn_transaction_2a2f_guide.md)。

第 1–9 节的布局和地址公式源于 CAM `b795d1df` 与 DeepEP owner/scheduler `8757a68d` 的 BF16 事务模式。旧模式逐片生成 Active，见[五算子设计](a5_async_afd_deepep_operator_design.md)。第 10、11 节补充个人 UMDK 分支 `5c3daaa7` 的纯 CAM 四卡实测、Poll 的设备时间和 Attention 成员独立分轮。FP8 仅作为布局预算。 [CAM 事务基点](https://gitcode.com/puxiaokun3721/umdk/commit/b795d1dfc2b7e6da64c48af02d7f00fecbfacaa7) · [DeepEP 布局基点](https://github.com/puxiaokun3721/DeepEP-Ascend/commit/8757a68d5f258c302ab5146df62119b70ea80b6d) · [个人 UMDK 实测提交](https://gitcode.com/puxiaokun3721/umdk/commit/5c3daaa7ccffe4699ceed63b88cba104eb5af4c7)

## 1. 全链路和三个事务边界

```text
Attention 侧 A                                    FFN 侧 F
 x / TopK / weights / mask
          │
          ├── DS 写 records、refs、counts、prefix ───► F Payload
          ├── DS 写 Inbox ──────────────────────────► F Control
          ├── DS 提交 A 本地 Ledger                        │
          └── DS 最后发 InputReady ──────────────────► F Control
                                                           ▼
                                         Poll 观察 NEW_READY
                                           ├── 带时间戳入口写 F 本地 timestamps
                                           └── 轮询报告供调用方读取
                                                           │
                                                Scheduler 一次选批
                                                           │
                                               固定的本地计算事务
                                                           │
                                          DR → FFN → ComputeReady → CS，循环多片
                                                           │
 A Payload ◄────────────────────────────── CS 按 lane/token/topk 写 BF16 route 结果
 A Control ◄────────────────────────────── CS 在单轮完成时写 Done → Consumed
          │
          └── CR 等全 F Done，按原 TopK 权重归并
                  ├── 回写下一代 Credit ──────────────► F Control
                  └── 释放 A 本地 Ledger bank
```

| 名词 | 含义与边界 |
|---|---|
| Attention DP 和 member | DP 是 A 侧一个逻辑数据并行组。member 是该组一个配置好的参与者。F 在本轮收到该 DP/phase 的全部逻辑成员后才能选它，零 route 成员也参与。 |
| 物理槽 `payload_slot_idx` | 某张 A 名下的可复用通信位置，初始化时就绑定窗口。同一 A 的两个槽可以承载两个在途任务。 |
| Lane `l` | 共享目录中唯一标识 `(A 列表下标, 物理槽号)` 的记录。各 F 都有该 lane 的接收窗口，拥有该槽的 A 有该 lane 的返回窗口。DP、member、phase、容量 `C_l` 都由 lane 配置确定。 |
| 通信轮 / 通信事务 | 单个 DP 的一个 MB 在一层的一轮 A→F 发布。每成员的逻辑输入行数 `B_l` 不超过单槽容量 `C_l` 时，一轮可覆盖该成员的整个 MB。超出时分轮。 |
| F 计算事务 | **一张 F** 一次有效选批冻结的完整通信轮集合，可跨 DP、MB、轮号和兼容层，整批只含 P 或只含 D。各 F 可以选出不同集合。 |
| 计算片 | 上述固定集合中的一批完整 `(layer, local_expert)` 组，由一次 DR、FFN 和 CS 完成。一个 expert 组的所有入选行进入同一片。 |
| Record、route、输出行 | Record 是发送到某 F 的去重 token 存储。Route 是一个 token 到一个 expert 的任务，一个 record 可被多条 route 引用。输出行是 DR 按本片 expert 分组分配的行号。 |

符号约定。`S` 为 A 数，`N` 为 F 数，`L` 为注册的 lane 数，`E` 为总 expert 数，`R=E/N` 为每 F 的本地 expert 数，`H` 为 hidden，`K` 为 TopK，`NL` 为注册层数。`A32(x)`、`A512(x)` 分别向上对齐到 32、512 字节。`a` 是 GMM 行对齐，单位为行。以下字节区间用左闭右开写法，`uint32` 占 4 B，`uint64` 占 8 B。

### 轮号和容量是两套计数

A 的同一 DP 事务为所有成员给出共同通信轮数，常用 `W=max(1,max_l ceil(B_l/C_l))`。第 `r` 轮在 lane `l` 上从 `begin=min(B_l,r*C_l)` 开始，长度 `len=min(C_l,B_l-begin)`。`B_l` 和 `len` 包含 mask 为假的行，先耗尽的成员仍发布空轮。F 一次选批只选当前已经发布并到齐的轮，下一轮后来发布时可参加另一个计算事务。

本段描述基线的 DP 共同轮数。成员独立分轮按 [第 11.2 节](#112-各-attention-成员独立分轮) 构造后续轮的等待集合。

## 2. 初始化和地址目录

owner 在每张卡建立两块独立的已注册设备通信内存 `Payload` 与 `Control`。Context、Ledger、Active、Completion 和计算张量是各自的本地存储。Context 描述地址和容量，运行时协议消息在 Control 和 Payload 中。F 的 `combine_x` 是注册用于回包读取的本地计算输出，另算其分配。

Context ABI 为 **2**。Context 固定头 **256 B**，后面是静态目录。各表偏移以 Context 起点为基准。

| 区域 | 大小 | 内容 |
|---|---:|---|
| `AFDContextLayout` 固定头 | 256 B | 18 个 `uint32` 位于 `[0,72)`，8 个 `uint64` 位于 `[72,136)`，`[136,160)` 对齐，`[160,256)` 为 DS 工作区描述。固定头的逐字段偏移见下表。 |
| `AFDLane[L]` | `32L` | 每项为 `src_rank_idx, payload_slot_idx, dp_idx, member_idx, phase, capacity, reserved[2]`，均为 `uint32`。 |
| F `AFDPeer[N]` | `32N` | 每项有通信域 rank 和 endpoint 索引（各 4 B），以及对端 Payload 大小、Control 基址、Control 大小（各 8 B）。 |
| `AFDAddress[L*N]` | `64LN` | 用 `l*N+f` 查表。8 个 `uint64` 依次为 `records, refs, count, prefix, inbox, input_ready, consumed, reserved`。前四项相对 F Payload，随后两项相对 F Control，`consumed` 相对 A Control。 |
| `layer_ids[NL]` | `A32(4NL)` | `uint32` 模型层号表。计算分组里的 `layer_slot` 是此表下标。 |
| `AFDReturnDirectory` | 32 B | 三个 `uint64` 为 A peer 表偏移、return lane 表偏移、返回 record 步长，随后是 4 B 版本号（1）和 4 B 保留字。 |
| A `AFDPeer[S]` | `32S` | 返回目标 A 的 peer 描述，字段与 F peer 表相同。 |
| `AFDReturnLane[L]` | `32L` | 四个 `uint64` 为 `return_offset, done_offset, credit_offset, reserved`。`return_offset`、`done_offset` 分别相对所属 A 的 Payload、Control，`credit_offset` 相对各 F 的 Control。 |

合计 `288+64L+32(N+S)+64LN+A32(4NL)` 字节。`src_rank_idx` 和 `f` 是各自 A/F 列表下标。跨卡寻址使用注册的对端窗口及目录中的偏移。 [Context 字段](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/deep_ep/include/deep_ep/layout/afd/context.hpp#L33-L73) · [Owner 分配](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/csrc/buffers/afd.hpp#L196-L233) · [返回目录](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/deep_ep/include/deep_ep/layout/afd/return.hpp#L6-L14)

| 固定头起点 | 类型 | 依次排列的字段 |
|---:|---|---|
| 0 | `uint32 × 4` | ABI 版本、通信域 rank 数、A 数、lane 数。 |
| 16 | `uint32 × 4` | 全局 expert 数、TopK、hidden 的传输字节数、scale pack 数。 |
| 32 | `uint32 × 4` | 向量核数、本地完整 MB 的输入分配上限、最大 lane 容量 `C_max`、DS 每 tile 的 token 数。 |
| 48 | `uint32 × 4` | 层数、当前 A 在 A 列表中的下标、本地槽数、返回进度模式。 |
| 64 | `uint32 × 2` | channel stride、分轮模式。基线在偏移 68 保留，独立分轮接口在此写 `comm_round_mode`。 |
| 72 | `uint64 × 4` | lane、F peer、地址、层目录相对 Context 的偏移。 |
| 104 | `uint64 × 4` | Context、Payload、Control 的字节数，返回目录相对 Context 的偏移。 |
| 160 | `uint64 × 12` | 10 个 DS 工作区区段偏移，随后是工作区总字节和保留字。 |

## 3. DS 在 F 的通信内存写什么

### 3.1 F Payload 的 lane 窗口

每张 F 的 Payload 都按 lane 表顺序拼接 `L` 个窗口。每个窗口内部是四段连续区间。

```text
F Payload
[ lane 0 ][ lane 1 ] ... [ lane L-1 ]
每个 lane = [ records ][ refs ][ count ][ prefix ]
```

| 每 lane 的区段 | 预留字节 | 内容及读写者 |
|---|---:|---|
| `records` | `A512(C_l*D_in)` | DS 为每个命中本 F 的 token 写一份 record。DR 通过 `record_id*D_in` 找它，`D_in` 是预留步长。 |
| `route_reference` | `A32(4*C_l*min(K,R))` | DS 按本地 expert 排列 `uint32` 引用，每条 route 存 `record_id*K+topk_slot`。DR 除以 `K` 得 record 编号，取余得原 TopK 槽。 |
| `count` | `A32(4R)` | 本 lane 的 `count[e]` 是发给本地 expert `e` 的 route 数。Scheduler 快照，DR 验证。 |
| `prefix` | `A32(4R)` | `prefix[e]` 是截至 `e` 的累计结束下标，供 DR 确定该 expert 的合法 refs 区间 `[e==0?0:prefix[e-1],prefix[e])`。 |

一个 token 在同一张 F 命中多个 expert 时，多个 ref 可以指向同一 record。按不同 F 分别去重。分配时按整条 lane 可能打满该 F 预留，有效条数随路由变化。Record 区的**长度**按 512 B 对齐，lane 起点由前面所有段的长度累加。 [Owner 窗口公式](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/csrc/buffers/afd.hpp#L196-L217)

### 3.2 Record 里每个字段的偏移

设 hidden 的实际字节数为 `X`，BF16 时 `X=2H`；DeepEP FP8 时 `X=H`。设 `SF=2*num_sf_packs` 为 scale 字节数，BF16 用 `SF=0`。以下偏移都从单条 record 起点算。

| 字段 | 起始偏移 | 有效字节 | 用途 |
|---|---:|---:|---|
| hidden | 0 | `X` | DR 搬到对应 GMM 输入行。 |
| scale | `X` | `SF` | 可选的 `int16` pack 原始字节，scale 的语义由量化调用方约定。 |
| `topk_idx` | `I=X+A32(SF)` | `8K` | 全部全局 expert ID。DR 验证选中的 TopK 槽对应正确 F 和 expert。 |
| `topk_weights` | `W=I+A32(8K)` | `4K` | 原始 FP32 权重。回包最终归并使用 A 本地原权重。 |
| `src_token_idx` | `W+4K` | 4 B | 原 token 的轮内编号，保存在 DR 的来源 metadata 中。 |
| `valid_topk_mask` | `W+4K+4` | 4 B | TopK 槽位图。DR 验证该 ref 指向的槽有效。 |

设末字段结束位置为 `Z=W+4K+8`。**DS 实际写 `X+A32(Z-X)` 字节，地址步长 `D_in=A512(Z)`**。步长尾部空隙保留为间距。DR 由 ref 找到 record 和 TopK 槽，再读 record 的轮内 token 编号供回包。 [TokenLayout](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/deep_ep/include/deep_ep/layout/afd/token.hpp#L5-L13)

例如 `H=128`、BF16、`K=2` 时，hidden 占 256 B，TopK ID 从 256 B 开始，权重从 288 B 开始，最后一个字段结束于 304 B。元数据区向上补齐到 64 B，DS 共写 320 B。512 B 是相邻 record 的预留步长。`C_l=4` 时，单个 lane 的 records 区因此是 `4×512=2,048 B`。

### 3.3 F Control 的 128 B 和发布顺序

每条 lane 的 F Control 起点为 `128l`。每个 32 B 信号块都是 `uint32 value` 加 28 B 保留。

| 相对 lane 起点 | 大小 | 字段、读写和含义 |
|---|---:|---|
| `+0` | 64 B | **Inbox**。依次为 `microbatch_idx`（0）、`layer_idx`（4）、`round_idx`（8）、`num_comm_rounds`（12）、`num_round_tokens`（16）、`return_bit`（20）、`num_recv_records`（24）。较新的事务接口在 28 B 偏移增加分轮模式，剩余 32 B 保留。DS 写，Scheduler 快照，DR 校验。`num_round_tokens` 是含 mask 行的轮内输入长度，`num_recv_records` 是本 F 去重后条数。 |
| `+64` | 32 B | **InputReady**。DS 写输入代次位，Poll、scheduler 和 DR 读。 |
| `+96` | 32 B | **Credit**。A 的 CR 写允许 F 使用返回窗口的代次，F 的 scheduler/CS 读。 |

DS 先等待上一轮全 F 的 Consumed 和将要写的 A 本地 Ledger bank 空闲。然后依次完成 **records 传输及排空 → refs/count/prefix 传输及排空 → 写 F Inbox → 提交 A Ledger → 发布 F InputReady**。即使某张 F 收到零 route，也会发布零 count/prefix、Inbox 和 InputReady。F 只有在 Ready 位等于自己期待位时，才接管该轮。 [CAM DS 发布顺序](https://gitcode.com/puxiaokun3721/umdk/blob/b795d1dfc2b7e6da64c48af02d7f00fecbfacaa7/src/cam/comm_operator/ascend_kernels/cam_deepep_dispatch_send/op_kernel/cam_deepep_dispatch_send_rt.h#L570)

## 4. CS 和 CR 在 A 的通信内存如何交接

### 4.1 A Payload 和 A Control

A 的 Payload 按这张 A **自己拥有的 lane** 拼接返回窗口。每条 lane 有 `C_l*K` 个 BF16 返回位置，`D_ret=A512(2H)` 是步长，每个位置实际写 `2H` 字节。合法 TopK 槽对应一个确定的全局 expert 和一个 F，所以不同 F 使用同一返回数组中的不同位置。返回数组按 `(轮内 token, topk_slot)` 索引。

A 的 Control 在每张 A 上都按**全局** `L*N` 预留，每个 `(l,f)` 一对 32 B 控制块，合计 `64LN` 字节。实际信号写到该 lane 的拥有者 A 卡。下面的两个信号地址均相对 A Control。

| 位置 | 大小 | 谁写给谁、何时写 |
|---|---:|---|
| A Payload 的 `return_offset[l]+(t*K+k)*D_ret` | 写 `2H` B | CS 把本片一个有效输出行写到来源 lane、轮内 token `t`、TopK 槽 `k` 的位置。DR 的来源 metadata 提供这些索引。 |
| A Control 的 `64(l*N+f)+32` | 32 B | **Done**。F 的 CS 在本 F 把该通信轮全部有效 route 写完并排空后发布 `return_bit`。A 的 CR 等所有 F 的 Done 匹配。 |
| A Control 的 `64(l*N+f)` | 32 B | **Consumed**。同一 F 对此轮的所有成员发布 Done 后，再向这些成员发布 `input_bit`。A 的 DS 等全 F Consumed，才可以按协议覆盖旧输入窗口。 |
| F Control 的 `128l+96` | 32 B | **Credit**。A 的 CR 归并完本轮后，向所有 F 写 `return_bit^1`，放行返回窗口的下一代。 |

```text
A 端，lane l 的返回窗口
       token 0                         token 1                  token C_l-1
[ k0 | k1 | ... | k(K-1) ]      [ k0 | ... | k(K-1) ]   ...   [ k0 | ... ]
  └ 每格 D_ret B，所属 F 按 route 的 expert 决定

A Control
[ (l=0,f=0) Consumed | Done ] [ (l=0,f=1) Consumed | Done ] ...
                     每对 64 B，F 的身份由表位置确定
```

A 侧窗口预留为 `Σ_本卡lane A512(C_l*K*D_ret)`，owner 最少申请 512 B。BF16 和 FP8 dispatch 预算例的**回包都按 BF16**。 [返回窗口分配](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/csrc/buffers/afd.hpp#L196-L217) · [CS 回包地址](https://gitcode.com/puxiaokun3721/umdk/blob/b795d1dfc2b7e6da64c48af02d7f00fecbfacaa7/src/cam/comm_operator/ascend_kernels/cam_deepep_combine_send/op_kernel/cam_deepep_combine_send_rt.h#L475)

### 4.2 A 本地的任务、账本和进度

这些结构都只在 A 本卡读写。`TaskInfo` 由框架填写，DS 和 CR 在同一任务期间使用同一组字段。`Ledger` 把 F 回来的**轮内行**映射到 A 的原输入行。`ReturnProgress` 给多槽多轮的 CR 公平 sweep 保存游标。

| 本地结构 | 大小及位置 | 字段及作用 |
|---|---|---|
| `AFDTaskInfo` | 每任务 64 B | 前七个 `uint32` 为 `abi_version, src_rank_idx, payload_slot_idx, microbatch_idx, layer_idx, num_tokens, num_comm_rounds`。较新的事务接口在 28 B 偏移增加分轮模式，剩余 32 B 保留。`num_tokens` 是该成员完整 MB 的逻辑行数。 |
| `AFDRoundLedger` | 每槽两项，每项 64 B。第 `r` 轮用 `2*slot+r%2` | 前八个 `uint32` 依次为 `state, input_bit, return_bit, round_idx, token_begin, num_round_tokens, final_round, microbatch_idx`，余 32 B 保留。DS 先写字段再提交 `state=1`。CR 在全部 Credit 写完后清回空闲 `state=0`。 |
| `AFDSendState` | 每槽 32 B，在 A 的 DS workspace | `next_input_bit, next_return_bit, has_published, last_input_bit` 各 4 B，余 16 B 保留。DS 独占推进。 |
| `AFDReturnProgress` | round-step 模式每槽 96 B | 前 64 B 冻结 `TaskInfo`，随后是 `state, next_round, last_input_bit, error_code` 各 4 B，余 16 B 保留。状态为 `IDLE/RUNNING/WAIT_ACK/DONE/ERROR`。每任务启动一次，后续 sweep 继续游标。 |

Ledger 共预留 `128*num_slots` 字节，round-step progress 共 `96*num_slots` 字节。`num_slots` 来自最大登记物理槽号加一。DS/CR 的 status 各为 8 个 32 位字，表达本次调用状态；持久复用状态在 SendState、Ledger 和 ReturnProgress。 [A 本地结构](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/deep_ep/include/deep_ep/layout/afd/context.hpp#L16-L32) · [ReturnProgress](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/deep_ep/include/deep_ep/layout/afd/return.hpp#L15-L24)

CR 校验 Ledger 中的轮、MB、行数和代次，等待**所有 F** 的 Done。每个有效 token 按 `global_token=token_begin+t` 恢复 A 的全局行号，读各 TopK 位置的 BF16 结果，按 A 本地原权重用 FP32 逐 TopK 槽累加，最后转 BF16。A 端 mask 行置零，输出超过有效逻辑范围的容量尾部保留。单轮归并完成后 CR 先发 Credit，再释放 Ledger。整个任务复用槽前，框架还要等 CR 确认最终 Consumed、最后一次 CR 退出以及 DS 发布退出事件。 [CR 归并与 Credit](https://gitcode.com/puxiaokun3721/umdk/blob/b795d1dfc2b7e6da64c48af02d7f00fecbfacaa7/src/cam/comm_operator/ascend_kernels/cam_deepep_combine_recv/op_kernel/cam_deepep_combine_recv_rt.h#L345) · [最终 Consumed 检查](https://gitcode.com/puxiaokun3721/umdk/blob/b795d1dfc2b7e6da64c48af02d7f00fecbfacaa7/src/cam/comm_operator/ascend_kernels/cam_deepep_combine_recv/op_kernel/cam_deepep_combine_recv_rt.h#L438)

### 4.3 一条 lane 的 0→1→0 复用

初始时 F 的 InputReady 为 1，scheduler 首次期待 0，F 的 Credit 为 0。A 的各 F Done 和 Consumed 初值为 1，A 的 DS 首次使用的两个位都是 0。

| 同一槽上的发布代次 | DS 发布给 F | F 完成该轮后写 A | A 完成归并后写 F |
|---:|---|---|---|
| 第一次 | Ready=0，Inbox.return_bit=0 | Done=0，再 Consumed=0 | Credit=1，再释放该 Ledger bank |
| 第二次 | Ready=1，Inbox.return_bit=1 | Done=1，再 Consumed=1 | Credit=0，再释放该 Ledger bank |
| 第三次 | Ready=0，Inbox.return_bit=0 | Done=0，再 Consumed=0 | Credit=1，再释放该 Ledger bank |

以上是代次的逻辑顺序。DS 的下一次输入发布依赖旧输入 Consumed 和相应 Ledger bank 空闲，CS 的下一代返回写入依赖 Credit。这两条依赖使输入和回包可以重叠。一个位可以循环复用，前提是该槽的旧代已经按协议退役。 [Owner 控制位初始化](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/csrc/buffers/afd.hpp#L291-L299)

## 5. F 本地调度内存

F 的 `ffnPoll()` 输出 `int32[L,4]`，共 `16L` 字节。四列是 `[分类, 实际 Ready 位, 期待位, phase]`，分类 `0` 无新 Ready，`1` 新 Ready，`2` 已在 Pending，`-1` 错误。Poll 只观察。旧通信轮已被 CS 退役时，Poll 用旧 Member 快照的 `input_bit^1` 观察新 Ready，scheduler 到下一次选批再接管。

个人 UMDK 分支的 `cam_afd_ffn_poll` 返回同样的状态报告。调用 `cam_afd_ffn_poll_timestamp` 时，首次观测还会写入调用方持有的 F 本地设备缓冲 `timestamps`。该缓冲每 lane 占 64 B，可由后续消费者按 lane 和轮次身份取周期值。Scheduler 使用下表的 workspace 独立扫描和接管。`Pending.first_seen` 是接管时的逻辑 tick，用于现有的排序，设备周期见第 11.1 节。

| Scheduler workspace 区域 | 起始偏移 | 大小与字段 |
|---|---:|---|
| `AFDSchedulerState` | 0 | 32 B。`tick` 为 8 B，随后是 4 B `last_phase`、4 B `has_phase` 和 16 B 保留字。 |
| `Pending[L]` | 32 | 每 lane 96 B。前 32 B 中 `valid/expected_input_bit/grant_state/member_leader` 各 4 B，`first_seen` 8 B，`num_group_members/num_remaining_routes` 各 4 B。随后 64 B 是 Inbox 快照。 |
| `totals[L,R]` | `32+96L` | `A32(4LR)`。接管 Ready 时复制的每 lane 各本地 expert 的 counts。 |
| `consumed[L,R]` | `32+96L+A32(4LR)` | `A32(4LR)`。接管时清零。旧模式逐片用它推进，事务模式的逐片进度在 Active 尾部。 |

Scheduler 只接管期待位匹配的 lane，快照 Inbox 和 counts，并校验轮号、层白名单、计数和容量。到齐键是 `(dp, phase, microbatch, layer, round, total_rounds)`，按该 DP/phase 的逻辑成员集合判断。选择首组时优先考虑和上次不同的就绪 phase，再按首次就绪顺序。随后冻结同 phase 的就绪整轮集合，可来自不同层、DP、MB 和轮号。选中后的 `txn_id` 是本 F 本地编号。 [Scheduler workspace ABI](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/deep_ep/include/deep_ep/layout/afd/scheduler.hpp#L6-L24) · [Scheduler 实现](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/deep_ep/include/deep_ep/impls/afd/ffn_scheduler.hpp#L211)

`phase` 的 0/1 与 P/D 的对应关系由框架配置。默认的 `COMPUTE_AHEAD` 允许有 route 的轮在 Credit 未到时先入选，CS 首次写回该轮前再取许可。`PRECHECK` 在选批时先核对所有成员的 Credit。全轮零 route 时，两种模式都在入选前核对 Credit，避免空轮占住唯一 Active 等待。

个人 UMDK `5c3daaa7` 的固定 AOT Scheduler 用 `kCreditPrecheck=false`，Torch 入口没有切换参数。配套 DeepEP owner 在准备时通过 `prepare_ffn(precheck=True)` 选择预检查的 JIT 模板，默认为 `False`。Credit 约束的是返回窗口代次。算子是否拆 stream 由调用方另行决定。 [两种策略和配置入口](a5_afd_ffn_transaction_2a2f_guide.md#24-qq_alloc-和-credit-策略)

## 6. Active 是什么，当前片和固定事务各放哪里

事务模式的 `active` 张量包含两部分。前缀描述**当前计算片**，尾部保存**整份固定事务**。设前缀长度 `P=64+64L+32LR`。

```text
active
[ 64 B 片头 | L×32 B Entry | L×32 B Member | L×R×32 B Segment ]
                                                               │ 偏移 P
                                                               ▼
[ 32 B 事务头 | L×32 B 固定 Entry | L×32 B 固定 Member | counts | group_counts ]
```

### 6.1 当前计算片前缀

| 区域 | 起始偏移、容量 | 字段含义 |
|---|---|---|
| `AFDActive` | `0`，64 B | 前 40 B 是十个 `uint32`，依次为 `valid, kind, phase, layer_idx, num_entries, num_members, num_segments, num_expanded_tokens, num_padded_tokens, reserved`。后 24 B 是三个 `uint64` 数组偏移，位于 `40/48/56`。事务模式的 `reserved=0x41465431`，片头 `layer_idx=0xffffffff` 表示按成员层槽解释。 |
| `AFDActiveEntry[L]` | `64`，每项 32 B | 8 个 `uint32`。`pending_idx` 是代表 lane；`member_begin,num_members` 指片内成员子区间；`segment_begin,num_segments` 指片内段子区间；`num_rows` 是该轮本片有效行；`is_final` 表示**这一个通信轮在本 F 的剩余行被本片覆盖**；`grant_state` 是返回窗口许可。 |
| `AFDActiveMember[L]` | `64+32L`，每项 32 B | `source_lane_idx,input_bit,return_bit,delta_begin,num_experts,reserved[3]`，均为 `uint32`。`delta_begin` 是 Completion 中本 Member 的本地 expert 计数起点，单位为元素。`reserved[0]` 保存 `layer_slot`。 |
| `AFDActiveSegment[L*R]` | `64+64L`，每项 32 B | `entry_idx,source_lane_idx,local_expert_idx,route_begin,num_routes,dst_row_begin,reserved[2]`，均为 `uint32`。`route_begin` 是该 lane 的 refs 表下标，`dst_row_begin` 是本片输出行号。`reserved[0]` 保存计算组号。 |

`Entry` 是一个被本片处理的通信轮，`Member` 是该轮的一条具体物理 lane，`Segment` 是某 Member 向一个本地 expert 提供的一段连续 route。Entry/Member/Segment 按片内表紧凑存储，输出行按计算组排布。因此表里相邻两个 Segment 的目的行可以不相邻。计算组号 `group_id=layer_slot*R+local_expert_idx` 只决定本地计算分组，源 refs 和回包计数沿用真实 `local_expert_idx`。 [Active 前缀 ABI](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/deep_ep/include/deep_ep/layout/afd/active.hpp#L6-L44)

### 6.2 固定事务尾部

下表偏移都相对整个 `active` 张量的起点。

| 区域 | 起始偏移 | 大小、字段及写入责任 |
|---|---:|---|
| `AFDFfnTransaction` | `P` | 32 B。8 个 `uint32` 为 `state, txn_id, next_group, slice_end, num_entries, num_members, completed_entries, slice_idx`。Scheduler 建立事务。DR 写本片拟议的 `slice_end`。CS 成功后推进 `next_group`、片号和完成轮数。 |
| `AFDFfnEntry[L]` | `P+32` | 每项 32 B。8 个 `uint32` 为 `pending_idx, member_begin, num_members, layer_slot, remaining_rows, grant_state, retired, reserved`。前四项由 scheduler 冻结。CS 更新剩余有效行、许可状态和是否退役。 |
| 固定 `AFDActiveMember[L]` | `P+32+32L` | 每项 32 B，布局与片内 Member 相同。Scheduler 冻结该轮成员的 lane、两个位、层槽和计数映射。 |
| `counts[L,R]` | `P+32+64L` | `A32(4LR)`。按**固定 Member 下标**和真实本地 expert 索引的 route 数。 |
| `group_counts[NL,R]` | `P+32+64L+A32(4LR)` | `A32(4NL*R)`。同层同 expert 跨所有入选 Member 的行数之和，按 `group_id` 索引。 |

总大小是 `P+32+64L+A32(4LR)+A32(4NL*R)`。两个 counts 表由 scheduler 在选批时写入，整份事务期间保持不变。CS 只推进游标和各轮状态。`next_group` 是已提交的下一组起点，`[next_group,slice_end)` 是当前片的组号区间。 [事务尾部 ABI](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/deep_ep/include/deep_ep/layout/afd/ffn_transaction.hpp#L10-L59)

### 6.3 ExecutionState、Completion 和计算缓冲

| F 本地分配 | 字节数 | 数据和使用方 |
|---|---:|---|
| `ExecutionState` | `32+A32(4L)` | 32 B 头为 `stage,error_code,num_entries,reserved[5]`，后接每个片内 Entry 一个 `uint32` grant。DR、computeReady 和 CS 依次推进。 |
| `Completion` | `32+64L+A32(4LR)` | 32 B 头为 `state,error_code,num_entries,num_members,num_deltas,reserved[3]`。偏移 `32` 是 Entry 快照，`32+32L` 是 Member 快照，`32+64L` 是 `uint32` delta 数组。CS 排空后填好。每条 delta 是该 Member 的真实本地 expert 消费量。事务模式用 Active 尾部续片，Completion 提供片级快照。 |
| `recv_x[Q_alloc,H]` | BF16 时 `2*Q_alloc*H` | DR 按目的行写 hidden，FFN 按组读取。 |
| 可选 `recv_sf[Q_alloc,num_sf_packs]` | 连续分配时 `2*Q_alloc*num_sf_packs` | DeepEP FP8 路径的 scale pack 输出。本文的 CAM BF16 事务模式使用空 SF。 |
| `recv_src_metadata[Q_alloc,4]` | `16*Q_alloc` | 四个 `int32` 为 `[source_lane_idx, 轮内 token, topk_slot, 片内 entry_idx]`，由 DR 写，CS 读。 |
| `combine_x[Q_alloc,H]` | `2*Q_alloc*H` | BF16 W2 输出，行顺序与 DR 一致。owner 将这块张量本身注册为 CS 的传输源。 |
| DR scratch | `32*NC+256+2*A32(4*min(LR,Q))` | `NC` 为向量核数。两份索引暂存用于取片和验证。 |
| CS scratch | `32*NC+256` | 片内校验、协同等临时存储。 |
| 旧 per-expert counts 与 psum | `8R` | 两个 `int32[R]` 张量仍分配。事务模式的 scheduler 写零，计算端从本片区间和 `group_counts` 推导跨层分组边界。 |

`ExecutionState.stage` 为 `EMPTY=0, SELECTED=1, INPUT_READY=2, COMPUTE_READY=3, COMMITTED=4, ABORTED=5`。Grant 为 `NONE=0, HELD=1, FINISHED=2`。`Completion.state` 为 `EMPTY=0, ARMED=1, READY=2`。这三套本地状态与通信窗口的交替位各有用途。 [F 本地分配](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/csrc/buffers/afd.hpp#L503-L570) · [Completion 布局](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/deep_ep/include/deep_ep/layout/afd/return.hpp#L25-L34)

FFN 计算写 `combine_x`，调用方在计算之后排入 `ComputeReady`，只对 `WORK` 片标记 `COMPUTE_READY`。同一条 stream 保证数据顺序，跨流用 event/wait 建立依赖。当前 CS 的 `WORK` 路径还显式检查本地状态，缺标记时报 `STATE_ORDER`，整片为 `CONTROL` 时使用 `INPUT_READY`。 [状态与 stream 顺序](a5_afd_ffn_transaction_2a2f_guide.md#25-computeready-和-stream-顺序各保证什么)

## 7. 从一次选批到多片提交

| 步骤 | 读取 | 写入及可见性要求 |
|---|---|---|
| **Poll → Scheduler** | Poll 读 F InputReady 及本地 Pending、Active。Scheduler 另外读 Inbox 和 Payload counts。 | Poll 输出就绪分类，开启时间记录时读取 Inbox 并写入首次观测周期。Scheduler 接管新轮、冻结选中集合和组计数，最后发布事务 `Selected`。这时片头为 NOP，计算尚未开始。忙态重复调用 scheduler 不改变选中集合。 |
| **DR 取片** | 已提交的 `next_group`，固定 Entry/Member、各组总行数和容量。 | 按组号升序跳过空组。一个非空组整体放得下才加入，余量不足就停在该组之前。DR 建当前片，写拟议 `slice_end`，事务进 `Slice`，执行状态进 `SELECTED`。已退役轮从本片 Member 和 Segment 中排除。 |
| **DR 预检与搬运** | 本片 Active、Context、未退役 Member 的 Inbox/Ready、counts/prefix/refs/record。 | 校验代次、Entry 层号、lane 区间、专家和 group 边界，再把有效行搬到 `recv_x`，写来源 metadata。每组 padding hidden/SF 为零，metadata 四列为 -1，片尾之后保持原值。全部输出搬完才发布本地 `ExecutionState.INPUT_READY`。 |
| **FFN → computeReady** | 本片的层槽、计算组行数与位置、对应层和 expert 的权重。 | W2 结果按 DR 相同行位写 `combine_x`。`WORK` 完成后发布 `COMPUTE_READY`。全零 `CONTROL` 跳过计算，保持 `INPUT_READY` 直接交 CS。 |
| **CS 写回和逐轮确认** | 本片、固定事务快照、`combine_x`、metadata、所需 Credit。 | 先验证本片与冻结计数一致。首次写某轮前确保所有成员 Credit 到齐，后续片沿用 `HELD`。写并排空所有 route 回包，填 Completion、扣除各轮剩余行并标记退役。完成轮在这之后独立发 Done 再发 Consumed，零 route 成员也收到确认。 |
| **CS 提交本片** | 已完成的本片写回和各轮剩余量。 | 发布 Completion `READY` 和 `COMMITTED`，再推进 `next_group=slice_end`、`slice_idx`。还有组则回 `Selected`，所有轮完成则发 `Complete`。框架根据此状态决定继续 DR 或进入下一次选批。 |

DR 的实际搬运行公式如下。

```text
ref         = refs[route_begin + i]
record_id   = ref / K
topk_slot   = ref % K
input_addr  = F.Payload + records_offset + record_id * D_in
output_row  = dst_row_begin + i
src_meta    = [source_lane_idx, record.src_token_idx, topk_slot, entry_idx]
```

这里有两个不同的 Ready。**F Control 的 InputReady** 是 A 发布的 per-lane 通信代次位。**ExecutionState.INPUT_READY** 是 DR 在本 F 完成一整片搬运后的状态。前者触发接管通信轮，后者允许 GMM 开始。

事务状态按 `Idle(0) → Selected(1) → Slice(2)` 推进。提交后回到 `Selected(1)` 或进入 `Complete(3)`，错误走 `Aborted(4)`。CS 在发 Consumed 前保留退役信息和快照，后续片跳过这些轮，即使 A 已覆盖它们的旧 Inbox/Payload。Scheduler 在计算事务结束后按旧快照清 Pending、翻期待位，再接管新 Ready。同一 F 的 Poll、scheduler、DR/FFN/CS 应按同流顺序或显式事件衔接。发现无效描述、容量不足或错误状态时应停止推进该域。错误处理无法撤回已发出的远端写入。 [CAM DR 取片](https://gitcode.com/puxiaokun3721/umdk/blob/b795d1dfc2b7e6da64c48af02d7f00fecbfacaa7/src/cam/comm_operator/ascend_kernels/cam_deepep_dispatch_recv/op_kernel/cam_deepep_dispatch_recv_transaction.h#L42) · [CAM CS 确认与进度提交](https://gitcode.com/puxiaokun3721/umdk/blob/b795d1dfc2b7e6da64c48af02d7f00fecbfacaa7/src/cam/comm_operator/ascend_kernels/cam_deepep_combine_send/op_kernel/cam_deepep_combine_send_rt.h#L518) · [Scheduler 退役](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/deep_ep/include/deep_ep/impls/afd/ffn_scheduler.hpp#L168)

## 8. 用一个例子讲完选批、搬运和回包

看 F0。令 `K=3`、本地有 `R=3` 个 expert，层槽 0/1 对应模型层 12/13。各 lane 的单轮容量 `C=4`，计算片 `Q=Q_alloc=12`，行对齐 `a=4`。下表的字节寻址是根据 CPU trace 的路由演算，trace 本身记录事务行为。

| lane | 通信轮 | phase | 发给 F0 的 `counts[e0,e1,e2]` |
|---:|---|---|---|
| 0 | DP0 / MB7 / 第 12 层 / 第 0 轮 | D | `[4,0,0]` |
| 1 | DP0 同一轮的另一成员 | D | `[0,0,0]`。仍发布 Ready，也必须等齐。 |
| 2 | DP1 / MB9 / 第 13 层 / 第 1 轮，共 3 轮 | D | `[4,4,4]` |
| 3 | DP2 / MB5 / 第 12 层 / 第 0 轮 | P | P 同时就绪，待下一个计算事务。 |

DP0 的 lane 0 先到，Poll 能看到新 Ready，scheduler 还在等 lane 1。等两个 D 轮都到齐，一次选批冻结两个 Entry。按层槽顺序得到 `group_counts=[4,0,0,4,4,4]`。

| 计算片 | 组号区间和有效组 | 输出行 | CS 完成后 |
|---|---|---|---|
| 0 | `[0,5)`，即 `(12,e0)`、`(13,e0)`、`(13,e1)` | 依次为 `[0,4)`、`[4,8)`、`[8,12)` | DP0 剩余 0 行，立即对 lane 0、1 发 Done/Consumed。DP1 剩余 4 行。 |
| 1 | `[5,6)`，只有 `(13,e2)` | `[0,4)`，复用本地计算张量 | DP1 剩余 0 行，对 lane 2 发确认。随后计算事务完成，下一次选批可以选 P 或新的就绪轮。 |

跟踪片 1 的一条 route。假设 lane 2 的四条 record 恰好按轮内 token 0、1、2、3 编号，且三个 TopK 槽分别去 e0、e1、e2。则 refs 排成 `[0,3,6,9 | 1,4,7,10 | 2,5,8,11]`，prefix 为 `[4,8,12]`。

1. Scheduler 冻结了 lane 2 的计数，片 1 的 DR 为 e2 生成 `route_begin=8,num_routes=4,dst_row_begin=0`。
2. 第三条 route 查 `refs[8+2]=8`，得到 `record_id=8/3=2`、`topk_slot=8%3=2`，搬到输出行 `2`。四列 metadata 为 `[2,2,2,0]`。最后的 0 是**片 1 的紧凑 Entry 下标**，固定事务中 DP1 是第 1 个 Entry。
3. FFN 用第 13 层 e2 的权重，把结果写到 `combine_x[2]`。CS 将这一行写到 lane 2 的 A 返回起点加 `(2*3+2)*D_ret=8*D_ret`。若 `H=128`，`D_ret=512 B`，相对该 lane 返回窗口的偏移是 `4,096 B`。
4. 假设 Ledger 给出第 1 轮 `token_begin=4`。A 等所有 F Done 后，把这条 TopK 结果作为原输入第 `4+2=6` 行的 TopK 槽 `k=2`参与归并。

片 0 结束后，lane 0、1 即使已经发布新 Ready，也不会加入这份固定事务。对 lane 2 的片 1 不读取它们的旧 Inbox 或 Payload。两个层的 e0 虽然编号相同，分别位于不同的计算组，使用不同权重。 当前 owner 只为一套传输 shape 和格式建立目录，入选层共用 `H/K/R`、dtype、scale 布局与行对齐，具体权重通过 `(layer_slot, local_expert)` 选择。框架还需要确认后端允许这些层的 FFN 形状和权重格式合批。 [CPU 模拟输入](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/tests/afd/ffn_transaction_sim.py#L199) · [完整 JSONL trace](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/tests/afd/ffn_transaction_trace.jsonl)

## 9. 容量规则和 4A4F 字节预算

4A4F 预算按下文的通用公式计算。已跑通的纯 UMDK 2A2F 则按 `C=4096` 预留通信窗口、按 `Q_alloc=8192` 预留 F 本地计算缓冲。图中分开画出了远端可写窗口和每卡本地工作区。 [四卡内存分配图](a5_afd_ffn_transaction_2a2f_guide.md#通信内存预留和本地计算内存)

### 9.1 跨层为什么要按组重新对齐

`Q` 是每片有效行硬上限，`Q_alloc` 是补齐后物理行硬上限。每片同时满足 `Σ n_g ≤ Q` 和 `Σ align_up(n_g,a) ≤ Q_alloc`，每个非空组只出现一片。比如 `Q=100`，两个组分别有 80 和 40 行，物理容量允许时依次取 80、40 两片。两个不同层的同号 expert 各有 2、3 行、`a=4` 时，各自对齐，合计占 8 行。

一个 token 对同一个 expert 最多贡献一条 route。当前配置以 **同 phase 注册 lane 容量和的最大值**作为保守单组上界 `N_bound`。Owner 检查 `Q≥N_bound`、`Q_alloc≥align_up(N_bound,a)`、`Q_alloc≤backend_max_rows`，再核预分配预算。实际组计数也由 scheduler 和 DR 复核。4A4F、每张 A 一个 `C=32K` 的同 phase 槽时，`N_bound=128K`。框架需提供与目标后端匹配的行数、分组数、权重/激活布局、对齐和 workspace 能力；128K 是当前选批范围的容量要求。配置不足时可缩小 `C` 或减少同 phase 注册 lane；更细粒度的容量约束需要整轮选批策略支持。 [Owner 容量校验](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/csrc/buffers/afd.hpp#L473-L503)

### 9.2 实际通信窗口算例

取 4A4F，每张 A 一个槽，`L=4`、`BS=C=32,768`、`H=7,168`、`K=6`、`E=256`、`R=64`。表中的 FP8 是 DeepEP 布局预算，假设每 128 元素一个 FP32 scale，56 个 scale 共 224 B，以 112 个 `int16` pack 搬运。当前 CAM 事务执行路径用 BF16。

| 单条输入 record | CAM BF16 | DeepEP FP8 布局假设 |
|---|---:|---:|
| hidden | 14,336 B | 7,168 B |
| scale | 0 | 224 B |
| TopK ID、权重、token、位图及内部补齐 | 96 B | 96 B |
| DS 实际写入 | 14,432 B | 7,488 B |
| 512 B 对齐后的步长 | **14,848 B** | **7,680 B** |

| 单条 record 的字节区间 | BF16 | FP8 |
|---|---|---|
| Hidden | `[0,14336)` | `[0,7168)` |
| Scale | 0 B | `[7168,7392)` |
| TopK ID（6 个 int64） | `[14336,14384)` | `[7392,7440)` |
| ID 区补齐 | `[14384,14400)` | `[7440,7456)` |
| 权重（6 个 FP32） | `[14400,14424)` | `[7456,7480)` |
| 轮内 token 编号 | `[14424,14428)` | `[7480,7484)` |
| TopK 有效位图 | `[14428,14432)` | `[7484,7488)` |
| 预留步长尾部 | `[14432,14848)` | `[7488,7680)` |

每 lane 的 refs 为 `4*32768*6=786,432 B`，counts 和 prefix 各 `4*64=256 B`。FP8 lane 的 Payload 偏移如下。

```text
0                   251,658,240     252,444,672    252,444,928   252,445,184
|------ records ------|----- refs -----|-- counts --|-- prefix --|

对应的 Control 单独分配 128 B。
```

BF16 同一 lane 的四段起点分别是 `0 / 486,539,264 / 487,325,696 / 487,325,952`，Payload 结束于 `487,326,208`。第 `l` 条 lane 在 F Payload 中以 `l×单 lane Payload 字节数` 为起点，在 F Control 中以 `128l` 为起点。

| 注册窗口口径 | CAM BF16 dispatch | DeepEP FP8 布局假设 |
|---|---:|---:|
| F 每 lane records 段 | 486,539,264 B（464 MiB） | 251,658,240 B（240 MiB） |
| F 每 lane Payload | 487,326,208 B | 252,445,184 B |
| F 每 lane Payload + Control | 487,326,336 B | 252,445,312 B |
| 每张 F 的 Payload，4 条 lane | 1,949,304,832 B | 1,009,780,736 B |
| 每张 F 的 Payload + Control | **1,949,305,344 B（1,859.00 MiB）** | **1,009,781,248 B（963.00 MiB）** |
| 四张 F 合计 | 7,797,221,376 B（约 7.26 GiB） | 4,039,124,992 B（约 3.76 GiB） |
| 每张 A，BF16 返回 Payload + Control | 2,818,573,312 B | 2,818,573,312 B |
| 八张卡的注册 Payload + Control 合计 | 19,071,514,624 B | 15,313,418,240 B |

每张 A 的 Payload 是 `32768*6*14336=2,818,572,288 B`，Control 是 `64*4*4=1,024 B`。因此 FP8 示例只从 F 输入窗口省下空间，八卡合计节省 3.50 GiB。这里合计的是各卡申请的注册窗口大小，进程和整卡 HBM 还包括模型、计算张量、workspace 和 SDK 开销。

### 9.3 把 F 本地分配加回预算

同样是 `L=4,R=64`，再取两个注册层 `NL=2`。以下是与通信窗口分开的分配示例。

| 分配 | 字节数 |
|---|---:|
| Context，每个 rank | 1,856 B |
| F Active 前缀 | 8,512 B |
| F Active 事务尾部 | 1,824 B |
| F Active 全部 | 10,336 B |
| F scheduler workspace | 2,464 B |
| F ExecutionState / Poll | 各 64 B |
| F Completion | 1,312 B |
| F DR / CS scratch，假设一个向量核且 `Q≥LR` | 2,336 B / 288 B |
| F BF16 `recv_x`、`combine_x`，`Q_alloc=131,072` | 各 **1,879,048,192 B（1.75 GiB）** |
| 同一 `Q_alloc` 的四列 metadata | 2,097,152 B（2 MiB） |

以上是已知分配。owner 的 `transport_reserve_bytes` 是预算里为传输留出的估算额，运行期通信库、框架及模型的其他分配需要单独测量。

单纯增加计算事务可选的层数，增加的是本地层目录和组计数表，已有 lane 的 wire Payload、Control 公式不变。如果为了让更多事务同时在途而新增物理槽，则 `L` 增大，各 F 需要为每条新增 lane 预留输入窗口。Q_alloc 取值也要计入 `recv_x` 和 `combine_x` 的成本。 [F 本地分配与返回源注册](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/csrc/buffers/afd.hpp#L503-L570)

## 10. 调用、插桩与验收

### 10.1 纯 UMDK 四卡实测

2026 年 11 月 15 日 19:20（北京时间）回执对应个人 UMDK `codex/cam-ffn-transaction@5c3daaa7ccffe4699ceed63b88cba104eb5af4c7`。实测在同节点 4 张 A5 卡运行，由 UMDK 测试代码借助 CANN HCCL/HCOMM 建通信窗口和 Jetty。Python 加载 `umdk_cam_op_lib`，运行环境中 DeepEP 包和扩展均未安装。使用 BF16、`H=7168`、`C=BS=4096`、`Q=Q_alloc=8192`、`K=2`、`E=4`、`net-layer=0`，两张 A 分属不同 DP 与不同层，每张 A 发一轮。

两张 F 用带时间戳的 Poll 观察并各选一次批。F0 同一事务的两片有 8192 和 4096 行，F1 处理 4096 行并完成另一条零路由通信轮。用例中的 FFN 为按层和按专家设置不同增益的 BF16 计算，然后调用 `ComputeReady`。两张 A 各校验 29,360,128 个不等权回包元素，F0 第二片等到 A0 收完结果才启动。A 侧账本释放，两张 F 的终态 Credit 均为 `[1,1]`。CANN 编译和四卡运行都退出 0，运行输出 `CAM_AFD_2A2F_PASS`。

构建、安装、启动及日志的核对方法见[纯 UMDK 2A2F 联调说明](a5_afd_ffn_transaction_2a2f_guide.md)。[UMDK 运行说明](https://gitcode.com/puxiaokun3721/umdk/blob/5c3daaa7ccffe4699ceed63b88cba104eb5af4c7/test/cam/README.md) · [四卡脚本](https://gitcode.com/puxiaokun3721/umdk/blob/5c3daaa7ccffe4699ceed63b88cba104eb5af4c7/test/cam/afd_ffn_transaction_2a2f.py)

### 10.2 四通信算子 PR 与配套诊断

DS、DR、CS、CR 的上游 PR 目标为 `cam_feature`。四算子候选 `dfc75a2f` 用配套 DeepEP owner 在 A5、BF16、`H=7168` 上验证了跨层多片、同一 DP 长短成员独立分轮、短成员同槽复用及旧调用 eager 兼容。PR 后续整理为 `e15b13ed`，该提交有 CPU 差分和社区门禁结果，A5 实测回执对应 `dfc75a2f`。Poll、Scheduler、ComputeReady 和独立 2A2F 入口在个人 UMDK 分支交付。 [UMDK CAM PR !3918](https://gitcode.com/openeuler/umdk/merge_requests/3918)

在配套 DeepEP-Ascend 工作树中运行下列命令，可重建第 8 节的 CPU 事件 trace。

```powershell
python -B tests/afd/ffn_transaction_sim.py --trace "$env:TEMP/afd-ffn-transaction.jsonl"
```

- **初始化。** 先创建使用同一 lane/layer 目录的 owner。F 调用 `prepare_recv(q,q_alloc,a,ffn_transaction=True,backend_max_rows=...)` 和 `prepare_ffn(...)`。A 准备 CR。本 A 注册了多个槽且任务有多轮时，采用现有的 round-step 和公平 sweep，即使当前只有一个槽在途。DS 与 CR 用原有独立的工作区和执行依赖。
- **调度。** F 用 `ffnPoll()` 观察，到齐由 `ffnScheduler()` 接管。收到 `Selected` 后开始循环 DR → FFN → `computeReady()` → CS，直到事务 `Complete`。A5 诊断 runner 由 Host 同步读取状态并输出 `FFN_TX` 事件，字段包含 F rank、事务号、phase、通信轮键、片号、计算组和逐轮确认。生产接入应使用后端所需的设备侧布局和依赖。
- **验收。** 同一 F 的同一 `txn_id` 只有一次有效选批，包含固定通信轮集合。各 `(layer,expert)` 只在一个片出现。每轮在本 F 完成一次并独立确认，跨层回包逐 MB 与对应层独立 golden 一致。覆盖空轮、成员迟到、P/D 同时到齐、不同 F 选批不同、容量不足和已退役槽复用。
- **证据边界。** 配套提交提供 CPU 模拟、原有 CPU 回归和 C++ 辅助逻辑验证。CANN/Bisheng 构建与加载、A5 可见性与回包顺序、目标 GMM 形状、图模式和性能须分别以对应实机证据验收。 [构建与八卡诊断命令](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/tests/afd/README-ffn-transaction.md) · [诊断 runner](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/8757a68d5f258c302ab5146df62119b70ea80b6d/tests/afd/ffn_transaction_runner.py)

## 11. Poll 设备时间戳与 Attention 成员独立分轮

两项增量都沿用 BF16 计算事务，可独立启用。纯 UMDK 的四卡运行确认了每张 F 能记录两个 lane 非零的设备周期，反复轮询同一轮时周期不变。A 侧独立分轮在四算子候选与配套 DeepEP owner 的四卡运行中验证，见第 10 节。

### 11.1 Poll 记录哪一个时间

Poll 在本 F 的设备上读取通信窗口的 Ready。第一次看到匹配当前期待位的 `NEW_READY` 时，在该 Ready 已经读回以后调用 `asc_get_system_cycle()`，保存原始 `uint64_t` 周期数。此时刻受 Poll 调用间隔影响。输入可以在两次 Poll 之间到达，记录的是 Poll 观测时刻。Attention 发布、Scheduler 接管和首次选中分别是其他事件。 [Ascend 950 系统计数器](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/C-API/sys_var/asc_get_system_cycle.md)

- **单位**。接口保存原始周期。消费方可用同一 F 上经验证的同源计数器读取 `now_cycle`，再算 `now_cycle - system_cycle`。项目的 Ascend 950 S 侧展示口径为周期差除以 1000 得微秒。四卡用例核的是首见周期有效、身份匹配和重复 Poll 保值，跨卡时钟比较与等待时间策略需各自验证。
- **标记**。时间与该 lane 的本次发布代次绑定，单独保存有效标志。同一次到达反复被 Poll 看见时沿用第一次周期。Ready 位在槽复用时重复出现，去重要同时区分发布代次。若使用当前 Pending 保存的上一轮接管 tick，它只用于区分同 lane 的发布，和已有 Scheduler 的排序意义相互独立。
- **提前退役**。某轮在 FFN 计算事务尚有后续片时已经完成，Poll 可按已退役 Member 的输入位翻转识别下一代。即使 Scheduler 尚未清理旧 Pending，也能记录新一轮。旧 Pending、Ready 不匹配和从未被 Poll 观测到的轮不会获得新的时间记录。

`AFDBuffer.ffnPollTimestamp(timestamps)` 和 `cam_afd_ffn_poll_timestamp(context, control, workspace, active, timestamps)` 在现有轮询报告之外更新调用者提供的本卡 `uint8` 持久缓冲。每条 lane 64 B，按下表解码。缓冲绑定一次 owner 生命周期，由调用方在首次使用前调零。调用方以同一 stream 或显式依赖串行读写，保证后续消费者看见已提交的记录。每条 lane 保存最近一次有效观测，用于在对应轮有效期间匹配。需要历史记录时由消费方留存快照。

| 偏移 | 类型 | 字段 |
|---:|---|---|
| 0、4 | `uint32` | 有效标志、当前观测的 `input_bit` |
| 8、16 | `uint64` | 上一次被 Scheduler 接管的 tick、本次首见设备系统周期 |
| 24–51 | 7 个 `uint32` | `microbatch_idx, layer_idx, round_idx, num_comm_rounds, num_round_tokens, return_bit, num_recv_records` |
| 52 | `uint32` | Inbox 的分轮模式 |
| 56–63 | 2 个 `uint32` | 保留 |

去重比较同一 lane 中的 `(上一代接管 tick, 当前输入位)`。该 tick 在当前发布被接管前保持固定，包括旧轮提前退役到 Scheduler 清理 Pending 的间隔。即使 Ready 位在跳过两次 Poll 后重新相同，不同的接管 tick 仍能区分两次发布。读取时间时应检查对应的轮次身份和有效标志，避免把旧记录算给当前轮。普通 `cam_afd_ffn_poll` 返回四列报告，带时间戳的入口另更新 `timestamps`。当前 Scheduler 接口没有传入这个缓冲，组批仍使用原有的 `Pending.first_seen` 逻辑 tick。[CAM 时间戳布局](https://gitcode.com/puxiaokun3721/umdk/blob/5c3daaa7ccffe4699ceed63b88cba104eb5af4c7/src/cam/comm_operator/ascend_kernels/utils/op_kernel/deep_ep/layout/afd/poll_timestamp.h) · [首见采样实现](https://gitcode.com/puxiaokun3721/umdk/blob/5c3daaa7ccffe4699ceed63b88cba104eb5af4c7/src/cam/comm_operator/ascend_kernels/utils/op_kernel/deep_ep/impls/afd/poll_timestamp.h)

**按逻辑轮聚合。** 对同一 DP、phase、microbatch、层、轮的参与成员，最小观测周期表示最早成员开始可见，最大观测周期表示所有成员都曾被看到。缺任一成员观测时，无法计算齐备时间。零路由成员也有 Ready。等齐后的 Credit 约束由现有协议处理。这一节只交付 Poll 周期，组批策略由上层设计。

Scheduler 独立扫描 Ready。若发布发生在该 lane 的 Poll 扫描之后、Scheduler 扫描之前，Scheduler 可以先接管它，该发布没有首见周期。消费方需要处理时间戳缺失。

### 11.2 各 Attention 成员独立分轮

每个 A 以自己的发送行数 `B_i`、窗口容量 `C_i` 得出 `J_i=max(1,ceil(B_i/C_i))`。A 的 DS 和 CR 使用同一个本地总轮数。通过 Inbox 把 `J_i` 告诉各 F，`round_idx+1==J_i` 表示这个成员的最后一轮。共同轮数模式下，短成员继续发空轮。`0` 表示 DP 共同轮数，`1` 表示成员独立轮数。配套 DeepEP owner 在准备通信资源前由所有 rank 调用 `enable_local_rounds()`，设置 Context 的 `comm_round_mode`（偏移 68）。TaskInfo 和 Inbox 的第八个 `uint32`（偏移 28）携带同一模式值，结构尺寸保持一致。发送、调度、回包均须使用同一模式。

个人 UMDK `5c3daaa7` 的固定 2A2F CAM 控制入口支持按 Context 模式和 Scheduler 工作区尺寸选择两种布局。该拓扑共同轮数使用 288 B，独立轮数使用 416 B。纯 UMDK 四卡用例配置为共同轮数且每 A 发一轮。独立分轮及短成员同槽复用的 A5 回执来自第 10.2 节的配套测试。

共同轮数 `W` 由上层提前填入同一 DP 各 A 的 TaskInfo。DS 使用本卡行数和槽容量校验 `W` 能否覆盖本卡输入，F 等齐时校验成员申报的共同轮数。上层从已知的 DP 切分计划得到 `W`，或同步各成员最低所需轮数的最大值。配套诊断器从各进程共用的 CLI 参数得到长度与窗口，联调路径没有为轮数新增集合通信。 [模式配置与轮数来源](a5_afd_ffn_transaction_2a2f_guide.md#23-分轮模式怎样配置谁算总轮数)

| 阶段 | 成员独立分轮 |
|---|---|
| 首轮 | F 等齐本 DP、phase 登记的每个逻辑成员。即使某 A 的整个 microbatch 是 0 行，也发一个 0 行的第 0 轮并声明本轮为本地最后一轮。 |
| 首轮到齐 | F 保存本次任务的成员本地轮数及进度。组内用 DP、phase、microbatch、层、轮识别同一轮，成员总轮数可各自不同。 |
| 后续轮 | 第 `r` 轮只等 `r<J_i` 的成员，校验该成员的轮号和轮数声明。没有路由的数据成员仍发布 Ready。此前已发完的成员退出该轮的等待集合。 |
| 分片和完成 | Scheduler 冻结本轮实际参与的 Member。DR 按冻结成员核对各自 Inbox，按整专家组取片。CS 最后一个计算片完成且回包排空后，才为本轮参与的成员写 Done 再写 Consumed，并推进已完成的轮号。 |
| A 收尾 | 每张 A 的 CR 按自己的轮数收齐所有 F 的 Done，归并本地实际输出，回写 Credit。空轮也走确认。最后一轮完成后，该 A 可按现有槽位复用约束发布下一 microbatch。 |

F 的跨轮记录放在本地 scheduler workspace 后缀，每 F 预分配至多 `L` 个记录位置，每位置占 `32+A32(4L)` 字节。32 B 头保存逻辑任务键和进度，后面用 `uint32[L]` 保存首轮冻结的成员所在 lane 对应的本地总轮数。物理 lane 可以接下一批，旧记录的作用域仍由逻辑任务键决定。一个成员在同一 microbatch 的各轮沿用首轮记录的物理 lane，便于把后续到达和首轮声明关联。F 在 Scheduler 接管 CS 已提交的终态后推进跨轮记录，完成尾轮时回收记录位置。跨轮任务数量受此缓冲区容量约束。

F 的跨轮记录独立于当前计算事务和物理槽。短成员已经退出旧 microbatch 时，长成员可能仍有后续轮。短成员的槽可以发布新 microbatch 的首轮，F 同时保存旧任务的尾轮进度。身份至少绑定 DP、phase、microbatch 和层。本版要求框架在旧任务所有成员的尾轮退役前保持该键唯一。单个 Ready 位只负责槽位发布，无法承担任务身份。

回包地址仍按 Member 的 lane、轮内 token 和 TopK 槽计算。F 只针对冻结进本轮的 Member 写数据和确认。A 对自身发布的轮仍需要等全部 F，包括没有接到这张 A 路由的 F。整个 DP 完成需要涵盖所有成员的本地末轮。最短成员自己的 CR 完成仅表示该成员已经完成。

### 11.3 分轮取舍与验证

等长 TP/SP 发送可由共同的补齐行数直接推导轮数。共用轮数模式下，短 A 需要按剩余轮逐次走空控制握手。独立分轮在发送前省去轮数交换，也省去已结束 A 的空尾轮握手。F 为此保存跨轮状态。缓冲仍按现有整专家组的保守容量上界分配。

下表是复用或补覆盖时的测试设计，已通过的范围见第 10 节。构建、运行和归档使用 `H=7168`、BF16。每次保存源提交、runner、构建产物、实机加载路径和散列、卡映射、参数、退出码和断言对应的原始日志。旧 `H=128` 数据按历史证据保留。

| 目标 | 输入和检查 |
|---|---|
| 基线等长与时间戳 | 同一个 DP 的两个 A 依次发布。记录同一 lane 连续多次 Poll 的首见周期和有效性，新一代 Ready 获得新值。同卡第二次设备读取与较早的首见周期比较。核零路由成员和未观测轮。 |
| 跨窗口边界 | 首跑用 `B=(8193,8192)`、`C=8192`、`Q=Q_alloc=16384`，同 DP、phase、MB、层和每源一个物理槽。共同轮数用同一 fixture 验证短卡有空尾轮。独立轮数检查首轮等齐，两 F 都观察到第二轮仅长卡参与，短卡收齐自己的首轮确认，逐元素回包匹配。资源允许时沿用 `B=(32769,32768)`、`C=32768`、`Q=Q_alloc=65536`。 |
| 零行成员与提前复用 | `B=0` 的成员完成首轮零行握手。`--reuse-short-slot --microbatches 2` 用例在短成员 CR 完成后，用原槽、不同 MB ID 及可辨识输入发布新一批，同时保留长成员旧尾轮。等待集合和 Done／Consumed／Credit 日志按 MB、lane、位、轮号关联。 |
| 错误和回退 | 混用共同轮数和独立轮数的报文、同成员重报不同轮数、跳轮及任务标识重用按发布模式核对失败结果。核对正常共同轮数路径与旧 Poll 入口。非法输入注入和局部仿真分别记录。 |

先验证正确性，再记录所需的设备时间戳和耗时。身份和达成条件均绑定到同一次运行。轮换同一物理槽时至少覆盖三代，确认交替位回到原值仍取得新时间。再覆盖 Poll 跳过旧发布或 Scheduler 先接管的场景，用有效标记和身份核对时间归属。

在 2A2F 且同 phase 两条 lane 的 32K 用例中，保守单专家组上界为 65,536 行。验证前核对实际后端是否允许 `Q_alloc=65536`、所选输入输出缓冲和每卡显存预算。两份 BF16 `[65536,7168]` 计算张量合计约 1.75 GiB；另需计入注册窗口、workspace、输入权重和通信运行时。CPU 回归、构建和跨卡各用例按实际完成情况归档。

[分轮方案对比与验收 trace](../../ANALYSIS/AFD_FFN_POLL_TIMING_LOCAL_ROUNDS.md) 给出两种模式的相同时序、时钟解释和结果判据。
