# A5 / Ascend 950 异步 AFD 架构与实现可行性审查意见

日期：2026-10-16。审查对象为原设计及当时实际检出的工作树。本文件记录静态源码、协议推演及官方API核查意见，不是实机验收报告。

- 原设计：[a5_async_afd_unified_megamoe_distributed_design.md](a5_async_afd_unified_megamoe_distributed_design.md)
- 修订副本：[a5_async_afd_unified_megamoe_distributed_design_reviewed_v2.md](a5_async_afd_unified_megamoe_distributed_design_reviewed_v2.md)
- 本文的“原稿L…”均指原设计，不能直接套用到v2行号。
- 原稿SHA-256：`5f8ff0b610187458334c2b227ce3b4150900777b028e31cb1a3000f742c332d3`。

## 1. 结论与证据等级

双后端方向可行，主要容量表与原生布局一致；当前CAMAsync可以提供DP到齐、专家分轮、层地址视图的参考，但不能直接证明目标A5、两级窗口、多microbatch聚合和图模式已实现。

原稿最重要的确定错误是返回交替位与零路由跳过完成的组合：输入全F ACK成立的安全论证不能直接套用到稀疏返回通道。其次是mask接口、Decode发布调度、非去重布局和单workspace的credit策略未形成完整实施契约。

本文分类：

- **确定错误/不自洽**：可由现有规则构造失败过程，不依赖硬件偶发行为。
- **接口缺口**：某种允许的输入/执行方式没有完整状态或地址契约，实施前必须明确。
- **平台待验证**：源代码及API材料不能证明具体硬件顺序或进度，不宣称已经失败。
- **可选优化**：不作为正确性修复，也不当作首版已支持能力。

v2采用的首版选择是供原作者复核的建议，尤其全F空完成、credit预检和mask范围；不是用户已经批准的最终性能取舍。

## 2. 实际源码基线与资源使用

| 仓库 | 实际HEAD | 与原稿关系 |
| --- | --- | --- |
| AFD Plugin | `7cb0aebf4b036ca6466bf01153ec89282a9eff03` | 原稿仅列PR #408历史快照，未列当前CAM基线；工作树有原有修改 |
| ops-transformer | `8d46ac50deabb544f59f56f5b82a995159907c03` | 一致 |
| vLLM-Ascend（D:/Code/refs/vllm_ascend） | `be0b61da22778b043eb2369b5f922b2a39ddc46d` | 一致 |
| DeepEP-Ascend | `3b25377d04b24fc6154698ded78a2bcb2c59afff` | 一致 |
| DeepGEMM-Ascend | `8491bbb4b8c02a094a2318965f50c70438a3e73c` | 原稿未列 |

原稿PR快照`08b4fc4c15c08b600f25277e0f62a16bdd1d3dcb`在当前AFD本地对象库不存在，本次没有重验该快照，不能将它混作当前CAMAsync实现。

已检查项目AGENTS.md、登记skill清单、项目`.cannbot`与本地skill资源；使用Ascend C代码审查、API检索、数据搬运/同步及性能最佳实践资料。设计/tiling/RegBase/图模式资源按适用范围查阅，未启动模型端到端优化流程。未找到可用本地Ascend devkit；skill中硬编码的A3/CANN8.5环境不当作本机事实。用户最初只读要求下未落盘；本文件及v2是在后续明确授权后新建。

## 3. 按严重程度排序的问题

### R1 / P1：返回通道交替位ABA（确定错误）

原稿位置：[L896](/D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md:896)、[L910](/D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md:910)、[L922](/D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md:922)、[L953](/D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md:953)。

触发：同一物理返回槽逐轮翻位，F参与集合变化，零route F不写Done，旧Done不清除；mb/layer/round在后续推理迭代重复。

```text
使用① bit0：F0参与，写Done0；A消费。
使用② bit1：F0不参与，不改Done；A等待其他F后完成。
使用③ bit0：F0再次参与；A匹配使用①残留Done0，提前读结果。
```

任务描述在跨事务重复时不能排除该误判。框架确实反复使用固定stage和layer，见[Attention forward L430](/D:/Code/mine/vllm-afd-plugin/afd_plugin/model_executor/models/npu/deepseek_v2_async_cam_forward.py:430)。这是目标协议缺陷，不是声称当前CAM已经实现交替位。

credit也不能只靠按序广播解决：慢F可能没有观察到中间1而只看到0。按当前任务防重复会有旧授权误认；要求观察翻转则可能漏掉中间状态。生产者顺序写不保证消费者逐次观察。

最小修复候选：所有F每轮消费credit并提交Done，含零route控制完成；或使用每F返回状态和严格绑定下一实际物理轮的定向授权。后者不能在F0上次完成后就预授其未来任意轮，因为期间F1可能占用相同token/topk返回地址。单独增加每F bit不能解决过早授权。

**v2选择**：全F逐轮返回闭合，零route不执行GMM；Done提交完成后才允许首版Consumed。A等全F Done并完成归并/清理后发下轮credit。通信payload/workspace不增加，控制量及慢F等待增加；无F leader或F间barrier。稀疏返回留作后续协议优化。

### R2 / P1：mask没有收发闭环（接口缺口）

位置：[原稿L1147](/D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md:1147)、[L1156](/D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md:1156)、[L825](/D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md:825)。send接受active_mask，recv未接受且Ledger未明确保存有效route信息。

```text
token t仅topk0有效，topk1被mask。
send只发0 → F完成。
recv等完整K的DataAsFlag：topk1永不到；
或recv只等Done读取完整K：读取残留/未初始化数据。
```

原A5 combine会检查mask再等待topk结果，见[A5 combine L1461](/D:/Code/refs/ops-transformer/mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_mte.h:1461)；原代码skip不能被理解为自动清零。

最小修复：recv引用稳定mask或Ledger有效route bitmap，定义inactive输出。**v2选择**：同一稳定mask传给send/recv；首版None/一维token mask，inactive输出0；二维route mask显式不支持。范围收窄是本次设计选择，原作者需确认是否满足首版业务。增加一个稳定view，避免mask深拷贝，不扩大主要payload/workspace。

### R3 / P1：复合提交块原子性没有依据（平台保证缺口）

位置：[原稿L704](/D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md:704)、[L706](/D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md:706)、[L852](/D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md:852)。控制块同时含mbId与publishBit，消费者依赖二者属于同一快照。

现有API依据不能排除新bit/旧mbId组合；不声称硬件一定撕裂。官方DataCopyPad说明类型/对齐/搬运参数，没有声明跨卡32B原子提交和完整release/acquire语义。

**v2选择**：Inbox按member/slot，mbId放先写完的稳定header；控制仅bit。接收bit后再读header，全F ACK前不能覆盖。Prefill减少M份Inbox重复，Pending仍保留逻辑任务；payload不变。仍需证明uint32状态字的可观察性、远端完成和MTE事件，不能把bit-only误当作全部硬件问题已解决。

### R4 / P1：同DP Decode多mb聚合没有可达发布时序（实现路径缺口）

位置：[原稿L785](/D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md:785)、[L1125](/D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md:1125)、[L1191](/D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md:1191)。

当前[Attention L447](/D:/Code/mine/vllm-afd-plugin/afd_plugin/model_executor/models/npu/deepseek_v2_async_cam_forward.py:447)顺序是send mb0、compute mb1、recv mb0、send mb1。若F等两个mb聚合，mb1永远排在mb0结果之后；不强制聚合也无法借此顺序获得所需同层双mb并存。

**v2选择**：独立D槽，A先发布同层多个mb，再等待相应结果；新增M=2/3事件/状态数组，不用固定stage0/1暗含规则。F聚合当前可执行条目，不强制凑满。payload仍按D多槽原预算，框架调度和并发验证工作增加。P共享单槽继续串行交接，不能复用D发布策略。修订一致性检查另明确：单轮Decode send只等发布DMA完成，不在send内等最终Consumed，否则同一发送流的mb1仍无法到达；最终输入ACK由recv收尾等待。send_status不等于槽可复用，输入位仍由send唯一推进。

### R5 / P2：非去重record排列自相矛盾（确定不自洽）

位置：[原稿L1381](/D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md:1381)、[L1453](/D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md:1453)。前文说紧凑专家段，算法却用tileRankPrefix+rank ordinal。

```text
tile0=[e0,e1]，tile1=[e0,e1]
rank紧凑record=[e0,e1,e0,e1]，counts=[2,2]
若按连续专家段读取，e0会收到e1的一行。
```

原固定专家跨度见[dispatch L822](/D:/Code/refs/ops-transformer/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h:822)。route间接表可避免错配，但必须明确其为唯一读取方式。

**v2选择**：两模式统一rank紧凑record＋每专家route间接表，F展开成专家连续计算输入；不假定payload专家连续。B×min(K,R)容量不变，减少模式间契约差异，增加相对原连续payload读取的索引适配。

### R6 / P2：单workspace无法绕过已计算但等credit的任务（执行契约冲突）

位置：[原稿L1030](/D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md:1030)、[L1094](/D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md:1094)、[L1103](/D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md:1103)。

```text
P0计算结果占住唯一workspace → combine等A0 credit。
D1已就绪且有credit → 后续scheduler不能越过当前图，workspace也不能覆盖。
```

这不必然造成全局死锁，但会队头阻塞。现有[FFN runner L373](/D:/Code/mine/vllm-afd-plugin/afd_plugin/v1/worker/npu/ffn_model_runner.py:373)串行recv/compute/send，[worker L170](/D:/Code/mine/vllm-afd-plugin/afd_plugin/v1/worker/npu/ffn_worker.py:170)还有每步synchronize。

**v2选择**：选择前检查条目所有成员credit，未齐留Pending并扫描其他候选；单Active/workspace不变，放弃跨返回轮compute-ahead，保留MegaMoE任务内部wave。恢复先算后等并绕过任务需要额外结果buffer/提交队列，必须重算内存。

### R7 / P2：运行时证据链不完整（可复现性缺口）

位置：[原稿L1586](/D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md:1586)、[L1629](/D:/Code/mine/vllm-afd-plugin/docs/design/a5_async_afd_unified_megamoe_distributed_design.md:1629)。

当前AFD及DeepGEMM未列入基线；没有锁定950具体型号、驱动固件、CANN、torch_npu和安装算子包。vLLM实际动态加载安装包，见[moe_utils L144](/D:/Code/refs/vllm_ascend/vllm_ascend/ops/fused_moe/moe_utils.py:144)，参考源码提交不能证明部署包相同。

**v2处理**：补源码基线与DeepGEMM边界，把运行时矩阵及API schema列为前置门禁；未虚构本机安装版本或将待验证项改成通过。

## 4. 源码核对结果与继承边界

| 项目 | 源码已有 | 新方案新增 |
| --- | --- | --- |
| CAMAsync到齐 | 连续rank、tpSize、非零token判定 | 显式TP/CP成员表、零token发布 |
| CAMAsync分轮 | TP专家counts聚合，完整专家分轮，部分uint16索引 | 专家内切片、第一层小窗口、32位route和64位偏移 |
| layered | 单device标量层索引及TensorList地址表 | A5 MXFP8、图、非法层错误处理 |
| MegaMoE | read dispatch/write combine、wave、读取去重、原workspace分区 | 来源lane与专家rank拆分、独立A/F协议、切片完成提交 |
| Distribute V2 A5默认MTE | write/write、双半区、固定专家跨度、DataAsFlag | rank去重、线性大BS路由、拆分接口、独立调度 |
| MoeEP Direct | Hcomm UBC_CTP/URMA、rank去重和epilogue | 只借鉴统计/定位，不替换目标MTE引擎 |
| DeepEP | histogram/prefix、rank去重、rank partial | 不能继承其未支持的capture或直接采用不同数值归并 |
| DeepGEMM | 对称workspace、全rank barrier和通算流水 | 不能照搬全组协调作为独立F协议 |

关键源码入口：

- [CAM WaitFromAttn](/D:/Code/mine/vllm-afd-plugin/csrc/npu/ascend_kernels/afd_async_dispatch_recv/op_kernel/afd_async_dispatch_recv.h:500)：零token当前视为未到达；[完整专家分轮](/D:/Code/mine/vllm-afd-plugin/csrc/npu/ascend_kernels/afd_async_dispatch_recv/op_kernel/afd_async_dispatch_recv.h:287)。
- [layered地址视图](/D:/Code/mine/vllm-afd-plugin/csrc/npu/ascend_kernels/utils/op_kernel/layered/layered_tensor_addr.h:43)：非法层当前回落0，需要改变。
- [MegaMoE窗口](/D:/Code/refs/ops-transformer/mc2/mega_moe/op_kernel/arch35/common/mega_moe_peermem.h:136)、[workspace](/D:/Code/refs/ops-transformer/mc2/mega_moe/op_kernel/arch35/common/mega_moe_workspace.h)。保留布局函数不等于保留所有旧容量及source维度。
- [A5默认selector](/D:/Code/refs/ops-transformer/mc2/moe_distribute_dispatch_v2/op_host/op_tiling/arch35/moe_distribute_dispatch_v2_tiling_arch35.cpp:1253)：空comm_alg为fullmesh_v1；不要混入fullmesh_v2/CCU或A3限制。
- [MoeEP Direct传输](/D:/Code/refs/ops-transformer/mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch.h:1191)、[保守窗口规划](/D:/Code/refs/ops-transformer/mc2/common/op_host/op_tiling/moe_ep_window_layout.cpp:82)。
- [DeepEP统计](/D:/Code/refs/DeepEP-Ascend/deep_ep/include/deep_ep/impls/ep/dispatch.hpp:174)、[partial combine](/D:/Code/refs/DeepEP-Ascend/deep_ep/include/deep_ep/impls/ep/combine.hpp:301)、[capture边界](/D:/Code/refs/DeepEP-Ascend/README.md:84)。
- [DeepGEMM全rank同步](/D:/Code/refs/DeepGEMM-Ascend/deep_gemm/include/deep_gemm/comm/mega_moe.hpp:66)。

DataAsFlag固定写1、消费清0与控制交替位是两个机制；原稿L922已经要求清理后再credit，此处不是新缺陷。不得为了增加问题数量重复报错。

## 5. 容量复核

H=7168、K=R=8、EP=8，GiB/MiB为二进制单位。以下原生布局外推与原稿算术一致，不证明支持该shape。

| 口径 | 32k | 256k | 1M |
| --- | ---: | ---: | ---: |
| MegaMoE窗口含68KiB | 3.733463GiB | 29.867252GiB | 119.468815GiB |
| Distribute双半区数据区 | 37.5GiB | 300GiB | 1200GiB |
| MoeEP Direct保守总申请 | 9.359557GiB | 74.875182GiB | 299.500182GiB |
| DeepEP示例主payload | 3.507813GiB | 28.0625GiB | 112.25GiB |

目标32k单槽：MegaMoE A payload=3.725586GiB；Distributed A返回=3.75GiB、8来源F去重接收=1.875GiB；混部F=1020MiB，P DP2TP2各16k时=540MiB。32768计算行的Mega主数据746MiB、dedup4MiB，另加50MiB/flags/平台空间；262144行为5968MiB+32MiB+其他。小算子GMM1/GMM2的32768行BF16输出256/448MiB，不能混作16MiB内部scratch。

v2改变Inbox维度后重算metadata：8源、P单槽、M=2、R=8、C=32k时Inbox dense route=8MiB，dense Active=16MiB；compact Active上限4×maxExpertRowsPerCompute字节。主要payload表未因控制策略改变。

实际传输量、通信预留、普通HBM输入输出、计算workspace、权重及后端额外分配必须分列。1M×7168一份BF16输入约14GiB，输出再约14GiB；小窗口不保证整个进程内存与BS无关。

## 6. 平台待验证门禁

| 门禁 | 已有依据 | 未证明的部分 |
| --- | --- | --- |
| A send/recv进度 | 当前[send tiling](/D:/Code/mine/vllm-afd-plugin/csrc/npu/ascend_kernels/afd_async_dispatch_send/op_host/afd_async_dispatch_send.cpp:198)、[recv tiling](/D:/Code/mine/vllm-afd-plugin/csrc/npu/ascend_kernels/afd_async_combine_recv/op_host/afd_async_combine_recv.cpp:185)均取全AIV | 两个等待kernel能共同推进、资源配额、capture事件 |
| 发布/消费顺序 | 原内核有明确MTE事件；官方规定搬运约束 | 跨卡完成→Ready/Done、状态字原子可见性、UB重读 |
| 非对称窗口 | 框架有[对称初始化](/D:/Code/refs/vllm_ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py:375) | 不同角色不同大小注册与寻址 |
| 图模式 | 通用框架存在图能力 | CAM路径当前[要求eager](/D:/Code/mine/vllm-afd-plugin/afd_plugin/compat/npu/feature_validation.py:186)，新op mutation/alias、warmup、空任务及退出 |
| TP/CP | 当前TP有实际调用 | 每种CP真实owner/hidden/恢复collective及一致顺序 |

官方[950 DataCopyPad](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_UBToGM.md)支持uint32等类型，UB起址32B对齐；具体重载的长度/stride限制须按目标版本确认。它没有声明32B跨卡原子消息。[DCache说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/C-API/cache_ctrl/asc_dcci.md)区分scalar GM缓存与DMA GM↔UB，支持本地DMA轮询方向，但不替代MTE2完成和跨卡协议。

## 7. 简化与性能建议

| 建议 | 成立前提/正确性 | 内存/通信/时延/复杂度影响 | 时机 |
| --- | --- | --- | --- |
| 按槽Inbox、bit-only提交 | 单槽一波输入、全F ACK保护稳定描述 | 减metadata、消除复合指向；payload/workspace不变 | v2首版 |
| Pending与Active保留，route用compact引用 | 最后引用结束前不ACK、不覆盖 | 减拷贝和Active route预留；增加明确引用生命周期 | 首版 |
| 静态身份由注册表推导 | 服务期间不变、单一权威来源 | 小幅减控制字段，减少不一致；不影响主要内存 | 首版 |
| 保留完整RoundLedger | 仅几十轮/mb，send/recv独立进度 | 少量metadata换清晰生命周期；避免引入新本地队列credit | 首版 |
| 全F空完成 | 每F每轮都消费授权，A全F完成后推进 | 增控制及慢F等待，无空GMM或payload增加 | v2首版 |
| credit预检 | 接受放弃跨返回轮提前计算 | workspace不增，避免已计算结果占槽；吞吐需测量 | v2首版 |
| 去除重复ordinal/权重预留 | route列表已唯一表达且权重留A | 少字段/拷贝；512B对齐后record未必缩小 | ABI评估 |
| 去除DataAsFlag | 整轮Done保证全部写完成且A不提前消费 | 15360→14336B/结果，3.75→3.5GiB/32k槽，省轮询/清零 | 后续 |
| 稀疏返回或rank partial | 分别重证授权，或精度允许数值改变 | 少控制/返回数据，但状态/精度复杂度增加 | 后续独立方案 |

不能把Inbox、Pending、Active和RoundLedger整体合并：它们分别代表跨卡发布、持久调度进度、本次执行快照及A本地收发交接。简化对象是重复数据，不是生命周期边界。

## 8. 文档讲解与结构意见

原稿覆盖全面，但原生分析较长，目标闭环出现偏后。v2在文首增加首版决策与完整闭环，保留原章节编号以便与原稿对照；没有大规模移动原生分析，降低本次修订核对成本。

讲解建议：场景→目标闭环→三种内存生命周期→两层分轮→返回状态机→两个后端映射→容量与验证。建议后续制作：输入/返回/workspace三泳道图、原始token到record到专家行到返回token的坐标映射图，以及Decode多槽/P单槽对照。v2已补Decode发布时序和0→1→0全F完成推演。

## 9. 原作者复核与实施前清单

- 复核全F零route完成是否覆盖所有槽使用轮；是否保留了“不产生空GMM、无F leader”的边界。
- 确认全F空完成的慢F代价、credit预检的重叠损失及mask首版范围是否符合业务；反对时需提供同预算下可证明的替代协议。
- 核对Inbox按槽后，所有route/Active引用都在ACK前结束，控制块无动态复合指向。
- 确认同DP同层Decode多个mb可先发布，recv不会堵住后续send；M=2/3和TP/CP collective顺序一致。
- 锁定record、mask、counts、Completion、错误退出ABI；统一非去重寻址。
- 将源码版本与运行时包对应起来，先做发布原语、并发进度、窗口注册及graph门禁。
- 以延迟/零route/跨迭代相同身份/共享槽/replay反例验收；不把静态推演当作实机通过。

总体建议：先定共同协议和框架执行模型，再展开两个后端实现。容量推导可继续沿用，性能收益必须在相同端到端口径下测量。
