# A5 异步 AFD v2 原作者复核回复

日期：2026-10-16。范围：对原设计、独立审查意见及 reviewed_v2 的再次静态复核。本记录不修改既有三份文档，不代表平台或性能验收通过。

## 1. 输入、基线与证据边界

| 文件 | 路径 | SHA-256 |
| --- | --- | --- |
| 原稿 | `D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md` | `5f8ff0b610187458334c2b227ce3b4150900777b028e31cb1a3000f742c332d3` |
| 独立审查 | `D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_architecture_review_2026-10-16.md` | `3c95cd7194f66dbb2d0f7bf704486d1a6d2e8fdf56f10648d8e5ff6dda05353e` |
| v2 | `D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design_reviewed_v2.md` | `42a3c05b064b4ca405fec630304d9594ad5e0b5acb0590a99836bfbd53073fe5` |

以下行号对应这些快照；后续修改后应重新定位。复核读取了项目AGENTS.md、本地Ascend C API最佳实践的数据搬运/流水同步资料及代码审查方法论，结合实际源码推演，没有运行A5硬件测试。

实际核对：AFD Plugin HEAD为`7cb0aebf4b036ca6466bf01153ec89282a9eff03`；ops-transformer HEAD为`8d46ac50deabb544f59f56f5b82a995159907c03`。当前工作树代码是本次证据，HEAD本身不能替代工作树差异和运行时算子包核对。

## 2. 总结

审查抓住了真实协议错误，v2主要修正应保留，但不能整体认定已经闭合、可以直接实施。返回ABA修复与credit预检是两件事：前者修复正确性，后者选择执行策略并改变性能重叠。

| 项目 | 判断 |
| --- | --- |
| 返回Done/credit ABA反例 | 同意 |
| 全F逐轮握手修复 | 同意，但依赖执行进度和硬件发布条件 |
| 物理槽Inbox、独立bit提交 | 同意，硬件发布保证仍待验证 |
| credit预检、单Active/workspace | 部分同意，是有效策略而非唯一正确方案 |
| Decode同DP同层多microbatch发布 | 同意修订方向，框架可达性仍需实现 |
| mask与inactive输出 | 同意，首版一维mask是范围选择 |
| 非去重rank紧凑record与专家间接表 | 同意 |
| Pending/Active/RoundLedger、图与退出 | 部分同意，仍有实施接口缺口 |

## 3. 返回交替位与全F闭合

### 3.1 原稿ABA成立

原稿L896、L910、L922同时允许只等参与F、零route不发Done、消费者不清Done交替字。反例：

```text
使用0：F0参与，写Done=0，A消费。
使用1：F0不参与，Done仍为0。
使用2：F0再次参与，预期也是0，A可能接受旧Done。
```

输入全F ACK不会自动证明返回Done是新的：原稿不要求A先等本轮全部输入ACK再检查Done，零route ACK也不要求消费本轮返回credit。credit连续写有序不等于每个F逐次观察了中间位。

此前原作者关于交替位的论证遗漏了返回稀疏参与条件，应修正，不能恢复这组规则。

### 3.2 v2修复成立的条件

v2 L935、L937、L939、L953：每个注册F逐轮接收Ready、消费credit、写Done、再写Consumed，零route不做GMM或payload写入。

再次授权0以前，A必须收齐全部Done=1。因此每个F的旧Done=0已被替换，每个F也已消费中间credit。该证明依赖：有序且无迟到重复写、每次授权只消费一次、初始化排空及硬件发布保证。

协议本身没有必然等待环，但要求：

- A recv不等待整个多轮send结束才启动。
- credit未到的候选留在Pending，scheduler继续扫描其他条目。
- 零route处理也不能阻塞在单个未获credit的条目上。
- 调度保证必要空完成最终取得执行机会。

“无F间barrier”只表示没有F直接互等的集合通信barrier；每个A槽推进仍有全F汇合点，慢F代价必须列出。

## 4. 物理槽Inbox、bit-only提交与复用

同意v2 L741：Inbox按member/slot，logicalMicrobatchId放在稳定header；独立InputReadyControl只含bit。消费者先匹配bit，再按固定物理坐标DMA读取header，避免依赖新bit/新动态指向属于同一原子快照。

同意v2 L939的保守ACK条件：结果写完、Done写完成、输入及route引用结束后才写Consumed；A收齐全F后才覆盖。Pending或Active仍引用Inbox时不得提前ACK。

仍待验证：uint32状态原子可观察性、payload/header先于bit远端可见、ACK对应实际读取结束，以及轮询获取新GM内容。参考源码`D:/Code/refs/ops-transformer/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h:1062`使用32B状态及MTE事件，但不能据此直接宣称完整目标协议已被证明。

## 5. credit预检：保留能力，明确策略取舍

同意审查R6的队头阻塞分析。唯一workspace被已计算但等待credit的结果占住时，不能无成本切换到其他任务。当前参考：

- `D:/Code/mine/vllm-afd-plugin/afd_plugin/v1/worker/npu/ffn_model_runner.py:373`：串行recv/compute/send。
- `D:/Code/mine/vllm-afd-plugin/afd_plugin/v1/worker/npu/ffn_worker.py:170`：服务循环逐步同步。

v2 L1079的预检可以保留MegaMoE任务内wave，源码入口为`D:/Code/refs/ops-transformer/mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:630`。必须统一v2 L992和L1145的条件：

```text
首次计算分段：检查并占用本条目全部成员的返回授权。
后续分段：验证本条目已持有授权，不要求重新取得未占用credit。
最后分段：完成结果、Done及后续提交，释放本地占用。
```

不同意把“禁止compute-ahead”说成单workspace正确性的必然要求。此前用户接受先计算下一轮、在combine_send处等待。相同内存预算有两种合法策略：

| 策略 | 内存 | 行为及代价 |
| --- | --- | --- |
| credit预检 | 单Active/workspace | 有授权才计算，可绕过未授权候选，但可能损失跨轮计算与A归并重叠 |
| 先算后等 | 单Active/workspace | 可提前计算，发送处等待；等待期间接受队头阻塞，不承诺绕过当前任务 |

只有再要求“先算后等且能绕过阻塞任务”时，才需要额外结果buffer/执行机制并重算资源。预检性能是否可接受须实测，不能仅凭安全性判断。

## 6. Decode多microbatch可达性

同意R4和v2修订。当前源码`D:/Code/mine/vllm-afd-plugin/afd_plugin/model_executor/models/npu/deepseek_v2_async_cam_forward.py:447`先recv mb0再send mb1；若F强制凑齐两个mb会形成等待环。

v2 L1256、L1264、L1269、L1822提出的契约正确：最后发布DMA完成后send返回；最终输入ACK由recv收尾等待；send_status不是槽可复用；输入位仍由send唯一推进，recv只读ACK。独立D槽因此允许同一发送流继续发布mb1，F不强制凑满。

仍需实现M=2/3独立task_state、mask/output/event、跨TP/CP一致collective顺序及资源进度。现有send和recv均取全AIV：

- `D:/Code/mine/vllm-afd-plugin/csrc/npu/ascend_kernels/afd_async_dispatch_send/op_host/afd_async_dispatch_send.cpp:198`
- `D:/Code/mine/vllm-afd-plugin/csrc/npu/ascend_kernels/afd_async_combine_recv/op_host/afd_async_combine_recv.cpp:185`

两个stream本身不能证明等待kernel共同推进。尤其P多轮内部闭环依赖真正并发或其他已证明的执行方案。

## 7. mask及非去重布局

### 7.1 Mask

同意v2 L1799：None/一维token mask、send/recv共享稳定view、inactive输出0、全inactive仍发控制消息。二维route mask首版拒绝是业务范围选择，不是硬件必然限制。

原A5 combine检查专家mask再等待：`D:/Code/refs/ops-transformer/mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_mte.h:1461`。其跳过/提前返回不会自动将inactive输出置0，见同文件L1695，目标实现必须补实际写零路径。

### 7.2 非去重record

同意v2 L1433、L1503：ENABLE/DISABLE统一rank紧凑record＋每专家route间接表，再展开为专家连续计算输入。原固定专家跨度见`D:/Code/refs/ops-transformer/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h:822`，不能原样读取新布局。

`memberCommTokenCapacity × min(topk, expertPerRank)`在专家ID不重复且紧凑分配的约束下成立。需要锁定routeReference到recordIndex/专家成员位置的编码；不能把非去重record再次按完整专家列表扇出导致重复计算。原record、ordinal和route表字段是否重复可作为后续优化。

## 8. Pending、Active、RoundLedger与图退出

保留三个生命周期边界：Pending持久排队/续跑、Active当前执行快照、RoundLedger连接A收发进度。引用Inbox的最后使用者结束前不得ACK，不能为了少几个结构整体合并。

仍有三个实施缺口：

1. **授权/Completion ABI**：授权绑定哪个条目、续跑判断、Completion如何区分新旧任务、scheduler消费Completion与吸收新发布的顺序、何时重置Completion和释放Active尚未完整锁定。单writer不足以替代状态转换。
2. **Active.valid贯穿所有节点**：v2 L1627要求abort后device no-op，但L804结构未显式列valid，各小算子API也未展示gate。原GMM/activation不会自动读取它；需明确条件执行、带gate的适配或安全空任务机制。
3. **非法层和一致退出**：当前`D:/Code/mine/vllm-afd-plugin/csrc/npu/ascend_kernels/utils/op_kernel/layered/layered_tensor_addr.h:43`遇非法层回落layer0。目标必须在读取权重和等待wave flag前进入错误路径，并定义A/F及MegaMoE角色一致退出。v2 L1839仍是待定义要求，不是已有能力。

## 9. 对原目标的影响

三类部署场景、A TP/CP与F纯EP、P单槽/D多槽、两级分轮、小返回窗口、单workspace均保留。物理槽Inbox减少P元数据；全F空完成不增加主要payload或空GMM，但增加控制及慢F等待。

MegaMoE任务内wave可以保留；跨返回轮提前计算被v2的预检策略取消，是实质性能变化。Distribute仍可复用量化、搬运和reduce，但接收寻址和大BS路由准备不是小改。

## 10. 决策与阻断项

### 建议保留

- 全F返回闭合修复、零route控制完成。
- 物理槽Inbox与独立bit提交。
- Decode发布完成与槽复用分离、recv收尾等待最终ACK。
- mask闭环、rank紧凑record及专家间接表。
- 源码/安装包对应和硬件准入门禁。

### 建议调整

- credit预检作为显式策略，准确列出与先算后等的取舍，不能静默替代此前用户接受的执行方式。
- 统一首次授权/持有授权续跑/最终释放。
- 补齐Active、Completion、graph gate及错误退出ABI。
- 准确表达全F槽级汇合的慢F代价。

### 应撤回的结论

“单workspace必须禁止提前计算”不成立。单workspace可以提前计算并等待；不能同时无成本承诺绕过当前阻塞任务。审查报告本身已部分区分两者，后续设计不得将预检包装成唯一正确修复。

### 实施前必须定稿

1. 共同协议状态机及所有字段的唯一更新者、前置条件与复用条件。
2. 返回授权占用与Continuation、Completion提交ABI。
3. Decode真实事件图、M=2/3及TP/CP顺序。
4. 每个计算节点的valid/abort行为及非法层处理。
5. record、routeReference、mask和输出地址契约。

### 必须实机验证

1. 跨卡payload/header→bit发布、ACK和Done的实际完成边界。
2. A等待kernel的资源共存、跨stream及graph进度。
3. A/F非对称窗口注册、地址与销毁生命周期。
4. device选层、capture/replay、warmup及一致退出。
5. 全F控制完成和credit预检的吞吐/尾延迟影响。

## 11. 可选简化，非本次必改

v2首版Done完成后才发Consumed。可研究A等待全F Consumed后直接归并，以合并一部分完成通知；必须重新定义授权消费和状态推进，并明确未来提前ACK能力的变化。它不增加payload/workspace，但不能在未证明前直接删除Done。

本轮应先完成共同协议和执行模型定稿，再展开两个后端完整实现。静态推演不能替代A5发布litmus、资源进度与端到端性能验收。
