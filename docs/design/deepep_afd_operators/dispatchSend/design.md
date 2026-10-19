# dispatchSend 算子设计

目标：直接改造 DeepEP-Ascend，基线 `3b25377d04b24fc6154698ded78a2bcb2c59afff`。状态：详细设计，未编译/实机验证。公共ABI、消息顺序和底层helper定义见[公共契约](../common_contract.md)，也是本算子的必需代码生成输入。

## 容量与异构成员轮次

host完整输入上界 `num_token_capacity=B_cap` 与底层 `num_max_tokens_per_rank=C_max` 独立；输入/路由/权重/mask按完整行范围存活，固定tile路由准备和通信offset仅按本槽C_m及当前轮长度计算。task_info.num_tokens=B_m含masked行，不能换成有效mask计数。各rank必须传相同DP num_comm_rounds=max(1,max ceil(B_m/C_m))，kernel不得仅按本rank局部轮数提前退出。已耗尽成员继续发布零长度Inbox/Ready，最终send_status仍仅代表发布完成，最终ACK由recv收尾等待。

两个owner各256k/32k时每个send发8轮，DP每轮64k不触发错误；单lane C_m≤32768、C_m≤C_max、route编码/SGE/UB/队列上界不变。C_m=0为非法配置（测试应拒绝），B_m=0为合法事务。DP聚合不参与发送tile或本地metadata buffer大小。


## 1. 算子接口

### 1.1 完整声明

```cpp
void dispatchSend(
    const at::Tensor& x, const std::optional<at::Tensor>& sf,
    const at::Tensor& topk_idx, const at::Tensor& topk_weights,
    const std::optional<at::Tensor>& token_mask,
    const at::Tensor& task_info, const at::Tensor& context,
    at::Tensor& control_buffer, at::Tensor& round_ledger,
    at::Tensor& workspace, at::Tensor& send_status);
```

```cpp
__global__ __vector__ void dispatch_send_impl(
    __gm__ void* x, __gm__ sf_pack_t* sf,
    __gm__ int64_t* topk_idx, __gm__ float* topk_weights,
    __gm__ uint8_t* token_mask, __gm__ const AFDTaskInfo* task_info,
    __gm__ const AFDContextLayout* context,
    __gm__ void* control_buffer, __gm__ AFDRoundLedger* round_ledger,
    __gm__ void* workspace, __gm__ void* jetty_ptrs,
    __gm__ uint32_t* send_status,
    const int sf_token_stride, const int sf_hidden_stride);
```

### 1.2 参数与支持类型

| 参数 | 方向/shape/type | 约束 |
| --- | --- | --- |
| x | in `[B_capacity,H]` BF16或E4M3FN | ND连续；读取task_info.num_tokens行，不做dispatch内量化 |
| sf | in optional `[B_capacity,S/2]` 2B pack | BF16空，FP8必需；两种stride布局沿原host校验，步长单位pack |
| topk_idx/topk_weights | in `[B_capacity,K]` int64/FP32 | 有效token的K个专家互异且在[0,E)；权重不在F应用 |
| token_mask | in optional `[B_capacity]` bool | false不发payload，保留原token位置 |
| task_info/context | in byte tensor | 固定ABI；task_info保持到Etxn，不在send返回后改 |
| control_buffer | in/out byte tensor | 本地ACK及远端Ready映射；每块独立32B |
| round_ledger | in/out `[slot_count,2]` 64B条目 | 双bank，EMPTY/PUBLISHED所有权转移 |
| workspace | in/out byte tensor | 第4节给定布局，不与A recv workspace重叠 |
| send_status | out uint32[8] | PUBLISHED只表示最后发布完成，不表示输入可复用 |
| sf_token_stride/sf_hidden_stride | device int | 从sf真实stride注入，乘2后为byte offset |
| jetty_ptrs | device资源指针 | host Context注入，每核独占队列；非用户随意指定 |

函数无动态返回分配。首轮无需等旧ACK；后续轮覆写槽必须等全F Consumed。最终轮发布后返回，最终ACK由combineRecv等待。本成员B=0时仍发布DP共同计划的全部零长度轮；仅当所有成员B均为0时，共同计划为一个空通信轮。

## 2. 计算逻辑与复用

### 2.1 原生到AFD改造边界

以`dispatch_impl`和`simt_persistent_worker`为基础，保留topk分组、`simt::deduplicate`、rank/expert histogram、metadata SF打包与SF stride处理、MTE事件、peer/jetty加载和底层队列完成。把“每核整轮token×K数组”改为固定T token tile；去除同rank self-copy、全rank recv stats等待、host counts/cached handle和外部EP barrier。

A/F角色分离后A编号与F编号相等也不是self路径。原host launcher `16<=num_vec_cores`不能原封保留；本版循环与UB不依赖16核，下限1，但核共存/吞吐必须实测。

首版选择复用HcommJetty的普通逐请求put路径，每条请求CQE，按tile批量doorbell；不复用原dispatch手写多SGE批量SQE编码。因此保留路由和传输机制，但不承诺原始峰值吞吐；后续恢复多SGE必须保持相同契约。`dst_gsge_idx/sgep_to_entry/lsqe_base_idx`在此首版不使用、不分配，不能留一套看似复用却无消费者的数组。

### 2.2 两遍路由与地址

J=ceil(C/T)。第一遍每tile对所有有效route累加expert_histogram[E]，每token目的F经deduplicate后仅累加rank_histogram[N]一次。写tile_counts[J,N+E]。本地SyncAll后负责人按tile顺序生成exclusive tile_psum[J,N+E]，并生成目标F各专家的route base。

第二遍重扫本tile，重新计算local histogram游标。每个目的F代表topk位置取得`record_idx=tile_rank_psum+local_rank_ordinal`；用原`asc_ballot/asc_shfl`在同token group中把该record_idx广播给相同F的所有topk位置，不能假定代表是slot0或最低位。上游deduplicate实际选择同值组的最高lane，保持其返回值语义。

`dst_buffer_slot_idx[token*K+k]`保存该record_idx，无效=-1。每route写到目标专家连续route区，值为record_idx*K+k。每token每F只提交一次hidden和一次metadata；metadata布局含sf、原topk/weights、轮内token、valid mask。hidden直接来自完整x相应行，无整轮hidden暂存。

### 2.3 Ascend C/API阶段序列

```text
InitSocState()
for each round:
  wait_local_control(Consumed)，读取当前bank EMPTY
  计算token_begin=min(B,round*C)，num_round_tokens
  VF_CALL(route_histogram_tile):
    SIMT读取topk；检查mask/范围/重复ID
    simt::deduplicate<N>(); asc_atomic_add(UB histogram)
    asc_syncthreads(); asc_sync_notify/wait(V -> MTE3)
    asc_copy_ub2gm_align(tile_counts对应片)
  local_join()  // SyncAll，仅本kernel
  core0:按tile加载count到UB，标量prefix，MTE3写tile_psum
  local_join()
  VF_CALL(route_emit_tile):重复去重，asc_shfl广播record_idx，写dst_buffer_slot_idx
  scalar/MTE metadata阶段：沿原preload_metadata的SF/权重复制和同步
  metadata_send_buffer写GM完成后：
    对本tile各目的record：jetty.put_at_offset(hidden)，jetty.put_at_offset(metadata)
    tile route/count按目标F区间put_at_offset
    jetty.ring_doorbell(); jetty.drain<timeout>()
  所有发送核完成 local_join()
  core0: write_control的相同MTE发布序列写Inbox64B，确认完成
  core0: Ledger字段写完，再写PUBLISHED
  core0: write_control(InputReady, input_bit)，全部目的完成
  local_join(); 推进发布计划
send_status=PUBLISHED
```

API调用的重载继承原`dispatch.hpp`/`handle.hpp`实际调用；新`put_at_offset`完整展开见公共契约。控制只在data/metadata可见后提交。每个tile drain是首版有意选择，限制在途元数据复用和队列深度；不是必须新增的跨卡barrier，且不等任何F计算完成。

## 3. Tiling策略

### 3.1 参数结构与推导

```cpp
struct DispatchSendTilingData {
  int32_t num_ranks, num_experts, num_topk, num_vec_cores;
  int32_t num_hidden_bytes, num_sf_packs, num_max_tokens_per_rank;
  int32_t num_tokens_per_tile, num_tokens_per_metadata_mte;
  int32_t num_ub_bytes, metadata_stride, record_stride;
  uint64_t workspace_bytes;
  int64_t num_timeout_cycles;
};
```

host取实际UB为U，元数据每stage T_m=4个token，两个stage。选择最大的T∈{32,64,128,256}满足下述UB不等式、线程组/请求上界及C容量；C<32仍T=32，尾部mask。若最小tile不满足，拒绝该family，不把T变成依赖B的巨大数组。E≤512、N≤512、N|E，K≤32；更大E必须改原512-thread histogram覆盖逻辑。

Block级：tile_idx=vec_core_idx, vec_core_idx+N_s,...；一个tile只有一个owner。尾tile有效数min(T,num_round_tokens-tile_begin)，空核参加局部同步。不按B増加核数；A发送核配额来自Context，recv另外保留可运行资源。

### 3.2 UB逐项表与系数

所有数组独立A32对齐，裸UB arena从offset0顺序分配；下面列出全部常驻/暂存区，无额外隐式TPipe。

| buffer名称 | 字节数 | 数量/用途 |
| --- | ---: | --- |
| expert_histogram、expert_cursor | A32(4E)各一 | 计数及emit游标 |
| rank_histogram、rank_psum | A32(4N)各一 | 去重计数及本tile分配 |
| dst_rank_idx、dst_slot_idx | A32(4TK)各一 | 当前tile；复用名称不复用整轮容量 |
| peer_info | A32(48N) | 1 |
| jetty_info | 96 | 1；源码实际88B |
| ub_sqe | 512 | 1；实际最大SQE≤512编译断言 |
| control_stage | 256 | 1；Inbox64、control32、状态/计数/零区，共不超过256 |
| metadata | A32(4M) | 2；M见公共record公式 |

`U_send=2A32(4E)+2A32(4N)+2A32(4TK)+A32(48N)+96+512+256+2A32(4M)`。

bufferCoefficient对token tile为8K B/token，BF16/FP8相同；dtype差异进入metadata固定部分（BF16 M=128、FP8例M=352），不能错误地把hidden按T份计入UB。E=64,N=8,K=8,T=256：BF16 U_send=19232B，FP8=21024B；以U=253952B演示，分别7.57%/8.28%。这里UB不是越满越好：tile受队列/时延和固定路由上限约束，增大T需基准测试，不能以填满UB为目标破坏A并发资源。

事件：metadata stage0/1使用id2/3，各阶段S→MTE3与MTE3→S配对；SQE沿handle使用id0，阶段不重入；control id1，仅负责人且在数据drain后使用。不与SIMT worker存活期并发复用同一UB。

## 4. Workspace需求

A32累加目录：dst_buffer_slot_idx=4CK；metadata_send_buffer=C*M；tile_counts与tile_psum各4J(N+E)；route_reference_send=4CK；route目标counts/prefix=8E；rank totals=4N；send_state每slot32；error/status scratch=256。所有相加先A32，J按C上界计算。无dst_gsge_idx、无host_workspace、无hidden镜像。

C=32768,T=256,K=8,E=64,N=8：前六项依次1048576、C*M、36864、36864、1048576、512，rank totals32，单slot state32、scratch256。FP8总13706048B，BF16总6366016B（含所有上述项；见验算记录）。Ledger每slot128B与传输资源另列；通信F输入L*C*D不计入本地workspace重复求和。

metadata全轮暂存直到本轮最后drain，之后可复用；两个D槽的send在同一发送流串行，允许共享这份准备区。Etxn前完整x/sf/topk仍保活，因为recv使用原routing且框架可能存在异步依赖。

## 5. 性能优化与代价

复用SIMT warp去重及prefix避免历史前缀重复扫描；复杂度O(CK)+O(J(E+N))。队列首版用普通put并每tile drain，易验证但小消息/CQE开销增加；测量后才恢复多SGE batching，不能直接声称与原dispatch性能相等。P通信窗口按C复用，D多槽仅增加通信预留，不复制临时路由准备区。

## 6. Kernel实现要点

所有地址首项uint64，例如`uint64_t(token_begin+token_idx)*H*element_bytes`；SF地址=`(uint64_t(global_token)*sf_token_stride+pack_idx*sf_hidden_stride)*2`。复制不Cast，payload浮点bit不变；metadata int64专家先校验再转local int32。mask=false和无token仍发空Inbox/Ready；不调用空URMA请求代替控制。

发布Ready前把错误在local_join边界统一广播；任何核非法route都禁止整个本轮Ready。已写远端但未Ready的内容属于未提交数据，错误域终止；不可回退假成功。A recv必须能在send内部后续轮等待期间执行，核共存是部署门禁。

## 7. 实现检查清单

- [ ] 在DeepEP `impls/afd/dispatch_send.hpp`适配原dispatch及worker；`layout/afd/token.hpp`实现record；不复制GMM。
- [ ] `csrc/kernels/afd/dispatch.hpp`沿原launch_dispatch的jit->compile、launch方式，新增AFD template key；`csrc/buffers/afd.hpp`预分配资源，python_api.cpp暴露dispatchSend。
- [ ] AFD专属差异标明来源commit；原EP dispatch入口行为不变；多SGE恢复作为后续独立优化。
- [ ] host校验dtype/stride/C/E/K/UB/SQ bounds，C++ sizeof/offset static_assert与Python预算一致。
- [ ] 单元：C=0拒绝；C=1/T−1/T/T+1/32768合法边界，B=0合法零长度，mask全false，K=1/8/32，同F多专家去重，FP8两种SF stride。
- [ ] 联调：M=2/3发布可达、TP/CP空成员、ACK延迟、连续三轮bit翻转；每record/route与CPU路由oracle对齐。
- [ ] litmus：最后一个jetty延迟时不能提前Ready；metadata在drain前毒化必须被测试检测；成功后无悬挂源引用。

## 8. 参考实现与DoD

[原dispatch入口](https://github.com/deepseek-ai/DeepEP-Ascend/blob/3b25377d04b24fc6154698ded78a2bcb2c59afff/deep_ep/include/deep_ep/impls/ep/dispatch.hpp#L551)、[去重](https://github.com/deepseek-ai/DeepEP-Ascend/blob/3b25377d04b24fc6154698ded78a2bcb2c59afff/deep_ep/include/deep_ep/common/simt.hpp#L25)、[host launcher](https://github.com/deepseek-ai/DeepEP-Ascend/blob/3b25377d04b24fc6154698ded78a2bcb2c59afff/csrc/kernels/ep/dispatch.hpp#L20)、[SQ/CQ](https://github.com/deepseek-ai/DeepEP-Ascend/blob/3b25377d04b24fc6154698ded78a2bcb2c59afff/deep_ep/include/deep_ep/comm/handle.hpp#L358)。

设计交付覆盖：完整签名、dtype、API阶段映射、tiling结构/公式、逐buffer UB与例算、GM生命周期、搬运不Cast的理由、实现/测试清单。未通过的事项是A5跨通路可见性、注册与核共存实测，不以文档完成替代平台准入。
