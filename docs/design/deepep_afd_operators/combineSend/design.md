# combineSend 算子设计

实现基线：DeepEP-Ascend `3b25377d04b24fc6154698ded78a2bcb2c59afff`。必须同时阅读[公共契约](../common_contract.md)。本文为设计产物，未完成编译/实机验收。

## 多来源片数与末片完成

返回地址始终由source_lane_idx找到该A物理槽，再按token_in_round*K+topk定位；每lane检查自身C_m，不使用DP合计32k限制。Q/Q_alloc约束当前片x和metadata，更多来源只能增加片数。Continuation持有同一条目grant，只有entry.is_final且该DP在本F全部成员的route完成，才按全成员发Done/Consumed；零route成员也参与。返回payload引用结束后才能发布完成。完整事务maxBS不参与返回槽寻址。


## 1. 算子接口

### 1.1 完整声明

```cpp
void combineSend(
    const at::Tensor& x, const at::Tensor& src_metadata,
    const at::Tensor& context, const at::Tensor& active,
    at::Tensor& control_buffer, at::Tensor& execution_state,
    at::Tensor& workspace, at::Tensor& completion);
```

```cpp
__global__ __vector__ void combine_send_impl(
    __gm__ bfloat16_t* x, __gm__ int* src_metadata,
    __gm__ const AFDContextLayout* context, __gm__ const AFDActive* active,
    __gm__ void* control_buffer, __gm__ AFDExecutionState* execution_state,
    __gm__ void* workspace, __gm__ void* jetty_ptrs,
    __gm__ AFDCompletion* completion, const int rank_idx);
```

### 1.2 参数说明

| 参数 | 方向/shape/type | 约束 |
| --- | --- | --- |
| x | in `[Q_alloc,H]` BF16 | GMM2已完成；已注册可供URMA直接读取，无另一个send_buffer |
| src_metadata | in int32 `[Q_alloc,4]` | dispatchRecv输出lane/token/topk/entry；padding全-1 |
| context/active | in固定byte ABI | Active在本次所有发送完成前不可变 |
| execution_state | in/out | WORK必须COMPUTE_READY；CONTROL为INPUT_READY；保存已取grant |
| control_buffer | in/out | 本地credit及远端Done/Consumed；全F独立运行 |
| workspace | in/out byte | 第4节定义；不能alias x |
| completion | out固定byte ABI | 先完整副本，最后READY；后续scheduler应用一次 |
| rank_idx/jetty_ptrs | device静态身份/资源 | 本F、每发送核独占jetty |

删除原topk_weights、psum_num_recv_tokens_per_rank、combine_recv_buffer对称指针及整片send_buffer参数；回源信息在src_metadata/Context。不存在FP8发送输出族，BF16是唯一数值dtype；metadata整型不做算术转换。

## 2. 算法与原生复用

### 2.1 复用范围

以`combine_impl`、`load_meta_issue_wqe`和HcommJetty为基础，保留按任务/核分摊、peer与jetty加载、SQE写入、doorbell、CQ回收、drain。移除HiddenReducer及同rank多专家聚合、权重尾段和rank代表topk槽。原`num_slots==1`直接引用x的分支成为首版所有route的发送方式。

首版同dispatchSend使用每WRITE请求CQE的HcommJetty普通put_at_offset，改造面小且完成覆盖清楚；不同时保留原手工batch WQE和普通put两套head推进。预计提交开销高于原SIMT多任务batch，保留后续优化点，不声称可零成本原样复用完整combine。

### 2.2 API与授权/提交序列

```text
InitSocState(); 检查全核一致gate
core0: 遍历Active.entries
  NONE: copy_control_to_ub轮询所有member的return_credit
        expected_return_bit匹配后把ExecutionState.grant置HELD
  HELD:核对owner/位快照，Continuation不再次消费
local_join()  // 必须先取齐整个Active所有条目再发送
每核按row任务读取src_metadata (asc_copy_gm2ub_align)
  等MTE2 -> S后；padding lane=-1跳过
  peer_info按source_lane.src_rank_idx定位
  remote_offset=lane.return_offset+(uint64(token)*K+slot)*return_stride
  src_ptr=byte_ptr(x)+uint64(row)*2H
  jetty.put_at_offset<timeout,false>(peer,offset,src_ptr,2H)
每tile ring_doorbell(); drain()；metadata双buffer轮换
local_join()  // 全部发送核source引用及远端完成条件成立
core0: 从Active快照准备Completion的entry/member/delta
  对is_final的entry：所有member write_control(Done)
                       结束所有输入route引用
                       所有member write_control(Consumed)
                       Completion.entry.grant=FINISHED
  对非末片entry：grant保持HELD，只提交delta
  Completion.state=READY; execution.stage=COMMITTED
```

核心地址API展开在公共契约；不发0字节WRITE替代CONTROL。多个entry中一个末片、一个续跑分别处理。不能将member局部无route视为整个DP末片。Completion里的delta由本Active各Segment num_routes按member/expert归并得出；padding不计数。core0用一个R维计数tile生成delta，无全Q行重复扫描。

## 3. 两级Tiling与UB

### 3.1 结构与计算

```cpp
struct CombineSendTilingData {
  int32_t num_src_ranks, num_source_lanes, num_vec_cores;
  int32_t hidden, num_topk, num_experts_per_rank;
  int32_t num_max_padded_tokens, num_rows_per_tile;
  int32_t num_ub_bytes, return_stride;
  uint64_t workspace_bytes;
  int64_t num_timeout_cycles;
};
```

Block级按本片P=physical rows：每核处理floor(P*i/N_c)..floor(P*(i+1)/N_c)。无数据核仍参与授权和发送完成SyncAll。核内T条metadata一次搬运，T∈{8,16,32,64,128}选最大满足UB及预设最大批请求数128；每tile后drain，避免无界在途请求。

### 3.2 UB明细与系数

| buffer | 大小B | 数量/生命周期 |
| --- | ---: | --- |
| metadata_stage | A32(16T) | 2，当前片行路由 |
| peer_info | A32(48N_A) | 1，N_A=物理A数；不能按F数分配 |
| jetty_info | 96 | 1 |
| ub_sqe | 512 | 1，原handle内部MTE3完成后可复用 |
| entry_member_stage | 128 | 1，两种32B对象分块，不驻留所有L |
| grant_control_stage | 64 | 1，输入/输出控制独立32B |
| delta_stage | A32(4R) | 1，生成Completion delta |
| error_zero_stage | 128 | 1 |

`U_send=2A32(16T)+A32(48N_A)+96+512+128+64+A32(4R)+128`，bufferCoefficient=32B/row，BF16 hidden直发不占UB。T=128,N_A=8,R=8时U_send=5440B；U=253952时2.14%。低UB用量合理，因为算子是控制和GM直发；不为凑利用率复制hidden到UB。

事件id0由原handle SQE拷贝占用，metadata id2/3，控制id1；纯scalar授权阶段没有存活的SQE搬运。peer按原load_peer_info缓存，首版一个远端region/channel。SQE每条num_bytes=2H≤底层长度字段上界，超过时拒绝family，不能截断。

## 4. Workspace

业务scratch=`256+A32(32N_c)`。Completion外部分配`32+32L+32L+A32(4LR)`，ExecutionState外部分配`32+A32(4L)`。L=8,R=8,N_c=8：scratch512B、Completion800B、ExecutionState64B。Active/Context/src_metadata/x为已有输入不重复计账。

x占`Q_alloc*H*2`，Q_alloc32768/H7168为448MiB。必须直接注册为URMA源；若注册失败阻断该实现，不偷偷分配另一个448MiB send_buffer。每tile drain后可复用metadata stage，但只有整个算子完成后才能让下一Active覆盖x（简单且可验证的首版）。

W_transport由实际队列/注册资源逐allocation求和，与业务scratch分开。申请预算检查发生于图外；x注册对象比本次调用长寿，注销前必须完成所有队列drain及全设备退出。

## 5. 性能与简化

去掉rank局部归并和结果镜像减少本地读写，但逐route返回增加网络量，不能与上游rank-partial时延直接比较。与总设计同预算/同数值语义；借用原任务轮转改善peer分布可以后续做，不改变本版唯一writer和完成条件。

## 6. Kernel实现要点

不乘topk权重，不Cast；输入BF16按位发送。元数据必须检查lane/token/slot/entry与Active成员归属，不能因entry_idx合法就允许访问未注册lane。先uint64乘法再加offset。src_metadata padding=-1先分支，不能把-1转uint变成巨大peer索引。

COMPUTE_AHEAD等待credit时x必须保留，单workspace不能绕过该Active；PRECHECK下通常已HELD，但仍验证owner一致。无route任务取credit且真实Done→ACK，NOP不产生Completion。最终ACK后所有提交内容只读Active/本地快照，不访问被A覆写的Inbox。

## 7. 实现检查清单

- [ ] 新`impls/afd/combine_send.hpp`从原combine模板改造；`comm/handle.hpp`以独立helper扩展offset接口，原put语义不改。
- [ ] 新`csrc/kernels/afd/combine.hpp`沿原launch_combine的JIT加载方式；不创建原send_buffer。
- [ ] 单元逐行目的地址与CPU oracle一致，包括相同token不同A/slot、K=32最高位、>4GiB offset、padding行。
- [ ] CONTINUATION三片只取一次授权，最后一片才Done/ACK；两个聚合DP一末片一续跑分别提交。
- [ ] 全route、零route、慢credit、最后CQE延迟、队列wrap；Ready/Done可见性在目标A5执行litmus。
- [ ] 验证元数据、x注册保活、无第二份QH镜像；精度应由A最终归并验证，F不得提前乘权。
- [ ] 错误核一致终止，不留下别的AIV挂在SyncAll，不伪ACK。

## 8. 参考实现与DoD

[combine入口](https://github.com/deepseek-ai/DeepEP-Ascend/blob/3b25377d04b24fc6154698ded78a2bcb2c59afff/deep_ep/include/deep_ep/impls/ep/combine.hpp#L162)、[元数据/WQE](https://github.com/deepseek-ai/DeepEP-Ascend/blob/3b25377d04b24fc6154698ded78a2bcb2c59afff/deep_ep/include/deep_ep/impls/ep/combine.hpp#L59)、[put/drain](https://github.com/deepseek-ai/DeepEP-Ascend/blob/3b25377d04b24fc6154698ded78a2bcb2c59afff/deep_ep/include/deep_ep/comm/handle.hpp#L358)、[launcher](https://github.com/deepseek-ai/DeepEP-Ascend/blob/3b25377d04b24fc6154698ded78a2bcb2c59afff/csrc/kernels/ep/combine.hpp#L18)。

签名、dtype、API映射、两级tiling、完整UB例算、业务workspace、精度/生命周期及开发清单已列明。实机注册/完成/资源结果不能由设计稿代替。
