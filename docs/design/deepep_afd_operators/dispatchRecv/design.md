# dispatchRecv 算子设计

实现基线：DeepEP-Ascend `3b25377d04b24fc6154698ded78a2bcb2c59afff`。必须同时阅读[公共契约](../common_contract.md)。本文为设计产物，未完成编译/实机验收。

## 多来源容量边界

输入来自Active选中lane的各自C_m分区；不可验证或假定Σ_DP C_m≤32768。当前片仅受 `num_expanded_tokens≤Q`、`num_padded_tokens≤Q_alloc` 和每专家padding约束，host仍要求Q_alloc≥Q。每lane record/ref检查按本lane容量，Segment目的检查按本片专家prefix；源计数累加先用uint64且检查ABI可表示范围。来自同DP两个32k owner的524288条集中route（K=8）由scheduler分片，recv不按整轮申请expanded_x；Q=Q_alloc=32768、a=1时处理16次即可。来自多个Decode mb的合片仍只允许同phase同layer。


## 1. 算子接口

### 1.1 完整声明

```cpp
void dispatchRecv(
    const at::Tensor& buffer, const at::Tensor& context,
    const at::Tensor& active, at::Tensor& execution_state,
    at::Tensor& workspace, at::Tensor& recv_x,
    std::optional<at::Tensor>& recv_sf,
    at::Tensor& recv_src_metadata);
```

```cpp
__global__ __vector__ void dispatch_recv_impl(
    __gm__ void* buffer, __gm__ const AFDContextLayout* context,
    __gm__ const AFDActive* active,
    __gm__ AFDExecutionState* execution_state, __gm__ void* workspace,
    __gm__ void* recv_x, __gm__ sf_pack_t* recv_sf,
    __gm__ int* recv_src_metadata, const int rank_idx,
    const int recv_sf_token_stride, const int recv_sf_hidden_stride);
```

### 1.2 参数说明

| 参数 | 方向、shape/dtype | 约束 |
| --- | --- | --- |
| buffer | in uint8，F注册输入区 | lane offset由Context提供；必须已经Ready后被scheduler选中 |
| context | in固定byte ABI | L/N/E/K/H/record/SF布局与当前编译族一致 |
| active | in header+Entry/Member/Segment数组 | immutable；num_rows≤Q，padded≤Q_alloc |
| execution_state | in/out | SELECTED进入，成功INPUT_READY；NOP不改成功状态 |
| workspace | in/out uint8 | 本地stage错误/同步空间，不存整轮展开 |
| recv_x | out `[Q_alloc,H]` BF16或E4M3FN | 与发送dtype匹配；只写本片物理行，包括padding |
| recv_sf | out optional `[Q_alloc,num_sf_packs]` int16存储 | SF格式和两个stride匹配下游GMM；BF16为空 |
| recv_src_metadata | out int32 `[Q_alloc,4]` | 每专家行lane/token/topk/entry，padding全-1；与上游K+2布局不兼容 |
| rank_idx | device int | 本F专家编号，用于验证record里的专家是否匹配Segment |
| recv_sf_*_stride | device int | pack元素步长，不是字节；host据预分配tensor注入 |

不输出recv_topk_weights：F不应用权重，A归并时使用原topk_weights。通信轮count/prefix不暴露为该函数动态shape；当前片prefix由scheduler独立产出并交GMM。

## 2. 计算逻辑和API映射

### 2.1 复用边界

直接从`dispatch_copy_epilogue_impl`改造：保留token_layout字段访问、分stage GM→UB→GM搬运、sf stride分支、padding填零以及notify/wait复用规则。去除全rank barrier与完整接收流histogram；已由A构建稳定expert route表，scheduler输出当前片Segment，因此不使用原PSumIterator遍历所有来源的完整token流。

每Segment给`source_lane_idx,local_expert_idx,route_begin,num_routes,dst_row_begin`。以它对应的route ref定位record，再读取其中topk字段验证所属专家；不能把record_idx当作原token_idx。

### 2.2 API级处理顺序

```text
InitSocState(); 全核读一致Active gate
NOP/ABORT -> 在尚未开始任何同步/搬运前统一返回
CONTROL -> core0置INPUT_READY，其他核走相同结束边界
WORK:
  按物理row区间分配，每核找到相交Segment
  对每stage最多T条有效route:
    asc_copy_gm2ub_align(route refs，连续段可成批搬运)
    asc_sync_notify/wait(MTE2 -> S, id=4/5)
    scalar: ref/K得到record_idx，ref%K得到topk_slot_idx
    对各record调用asc_copy_gm2ub_align到当前token stage
    asc_sync_notify/wait(MTE2 -> S, id=0/1)
    校验mask/expert/token/record bounds
    asc_copy_ub2gm_align(recv_x+dst_row*X, token.hidden, 1,X,...)
    沿原sf stride分支复制SF -> recv_sf
    写回源metadata 4个int32到metadata stage
    asc_copy_ub2gm_align(recv_src_metadata对应区间)
    asc_sync_notify/wait(MTE3 -> S,id=0/1及2/3)，再复用stage
  对padding行：asc_set_copy_pad_val(0)及zero stage搬运hidden/sf
    metadata写[-1,-1,-1,-1]，不读取无效ref
  local_join(); 无错误core0置INPUT_READY
```

事件id0/1服务record双stage，2/3服务metadata双stage，4/5服务refs双stage；相同方向的事件必须配对回收后重用，至多6个stage ID。SF不连续时沿原asc_set_ub2gm_loop1_stride/asc_copy_ub2gm_align分支，或每pack显式地址拷贝；选择前者，并在每次改变loop stride前结束前次copy，结束后恢复默认loop配置，不能污染后续metadata写。

## 3. 两级Tiling与UB

### 3.1 参数结构及核间切分

```cpp
struct DispatchRecvTilingData {
  int32_t num_vec_cores, num_ranks, num_experts, num_topk;
  int32_t num_hidden_bytes, num_sf_packs, record_stride;
  int32_t num_max_expanded_tokens, num_max_padded_tokens;
  int32_t num_tokens_per_stage, num_ub_bytes, expert_alignment;
  uint64_t workspace_bytes;
};
```

每核处理`[floor(P*i/N_r),floor(P*(i+1)/N_r))`，P=active.num_padded_tokens。按物理row划分，完整覆盖无重叠；核内遍历与自身区间相交的Segment，segment可切到两核但每row仅有一个writer。未被Segment覆盖且位于本片prefix范围的row是padding。禁止把分核范围扩大到Q_alloc尾部后读旧metadata。

### 3.2 UB分配与tile求解

裸UB arena：

| buffer | 大小B | 数量 |
| --- | ---: | ---: |
| token_stage | A32(T*D) | 2 |
| metadata_stage | A32(16T) | 2 |
| route_reference_stage | A32(4T) | 2 |
| segment_stage | 32 | 2 |
| zero_stage | 512 | 1 |
| control_stage | 256 | 1 |

`U_recv=2A32(TD)+2A32(16T)+2A32(4T)+64+512+256`。有效bufferCoefficient为`2D+40` B/row，外加832B固定区及对齐。FP8示例D7680→15400B/row，BF16 D14848→29736B/row。SF包含在D里，不另乘同一scale缓冲。

host从T∈{1,2,4,8,16}选择最大满足U_recv≤实际U且6事件ID可用的值；为复用上游约64KiB token stage目标，再限制单stage TD≤65536（更大record只允许T=1且仍满足UB）。示例选择T=4：FP8 U_recv=62464B，BF16=119808B；U=253952时约24.60%/47.18%。示例用于对照，实际选值由完整公式确定；零route只需控制区但复用同family无需动态UB分配。

UB偏移严格按表顺序A32累加；stage入站完成后才能读，出站完成后才能覆盖。metadata不能与record stage别名，因为两者MTE3生命周期重叠。

## 4. Workspace与输出容量

本算子临时workspace=`A32(32N_r)+256`：每核错误/完成报告32B以及负责人广播256B；采用原local SyncAll所需平台workspace另从runtime独立返回。N_r=8时业务scratch512B。

显式输出：recv_x=`Q_alloc*X`，recv_sf=`storage_span(sf_shape,strides)*2`，recv_src_metadata=`16Q_alloc`。SF span=`(Q_alloc-1)*stride0+(num_sf_packs-1)*stride1+1`，任一维为0时0；不能按numel低估带padding stride的storage。Q_alloc32768,H7168：FP8 x224MiB，BF16 x448MiB，metadata0.5MiB；FP8 S224且紧凑布局sf7MiB。它们是计算workspace组成，不是额外通信payload复制。

buffer、route表、Active是已有输入分配，本函数不再分配。只物化当前片，禁止先原生完整epilogue展开再切slice。输出保留到GMM消费结束；每次新片复用同一分配。

## 5. 性能优化

复用原多stage搬运；按Segment连续route批读，减少重复histogram。一个record命中多个专家可能在不同片重复本地读取，这是单Q workspace代价；不为此缓存整轮展开。随着Q增大对比gather有效带宽和GMM效率；不能扩大UB token stage挤掉SF/metadata事件。

## 6. Kernel实现与精度

byte地址=`lane_base+uint64_t(record_idx)*D`；输出hidden=`recv_x+uint64_t(dst_row_idx)*X`。ref有效、record_idx<num_recv_records、topk slot<K、token<C、专家所属F及Segment一致，全部校验后才写输出。byte地址不能在int32中乘H。

BF16/FP8均位搬运，不调用Cast，不重解释SF内容。padding hidden/sf为零，metadata为-1；后续GMM可计算padding，但combineSend绝不发送padding。错误核必须参与本地一致错误汇合；INPUT_READY只在全部copy输出完成后设置，不能部分成功。

## 7. 实现检查清单

- [ ] 基于DeepEP `impls/ep/dispatch_copy_epilogue.hpp`建立`impls/afd/dispatch_recv.hpp`，保留MTE/SF/padding代码，标出来源与AFD差异。
- [ ] host `csrc/kernels/afd/dispatch.hpp`沿原launch_dispatch_copy_epilogue的JIT路径；输出预分配，删除动态recv counts/CPU sync。
- [ ] 单元：Q=1、片跨专家、单专家跨片、物理padding跨核、SF两种stride、token尾tile、NOP/CONTROL输出不读旧行。
- [ ] 元数据oracle逐row验证lane/token/topk/entry，检查不同成员相同token编号不混淆。
- [ ] 末片之前毒化未消费输入应触发失败测试，证明没有错误提前ACK；正确实现本算子从不发ACK。
- [ ] 精度：BF16/FP8输入输出bitwise相同（padding例外），SF逐pack相等，不用近似误差掩盖搬运错位。
- [ ] UB公式、实际struct偏移与编译UB报告一致；图replay不重新分配recv张量。

## 8. 参考与交付状态

[原epilogue签名](D:/Code/refs/DeepEP-Ascend/deep_ep/include/deep_ep/impls/ep/dispatch_copy_epilogue.hpp:175)、[stage与UB](D:/Code/refs/DeepEP-Ascend/deep_ep/include/deep_ep/impls/ep/dispatch_copy_epilogue.hpp:217)、[metadata与SF](D:/Code/refs/DeepEP-Ascend/deep_ep/include/deep_ep/impls/ep/dispatch_copy_epilogue.hpp:464)、[host launcher](D:/Code/refs/DeepEP-Ascend/csrc/kernels/ep/dispatch.hpp:84)。

DoD：签名、类型、API映射、tiling结构/公式、各dtype UB表/例算、workspace、精度和实施测试项均已定义。scheduler内部算法不影响本接口：开发时可先以人工Active fixture验证本算子。
