# combineRecv 算子设计

实现基线：DeepEP-Ascend `3b25377d04b24fc6154698ded78a2bcb2c59afff`。必须同时阅读[公共契约](../common_contract.md)。本文定义 A 侧按通信轮归并；不是原生 rank-partial epilogue 的原样调用。平台和 API 验收尚未完成。

## 完整事务与返回窗口分离

combined_x按单rank完整事务B_cap分配、写本次B_m行（含inactive清零）；返回通信预留按本A各槽Σ C_lane*K*return_stride，绝不能按DP其他rank的BS或窗口求和。num_comm_rounds使用DP共同计划；本rank提前耗尽仍处理后续零长度Ledger、全F Done、credit及最终ACK，不能在ceil(B_m/C_m)后提前退出。真实输出偏移来自本rank token_begin，不使用DP前缀；异构C_m按各自轮范围归并。两个owner各256k/32k时各归并8轮，DP每轮64k不影响各自地址。


## 1. 算子接口

### 1.1 完整声明

```cpp
void combineRecv(
    const at::Tensor& combined_topk_idx,
    const at::Tensor& topk_weights,
    const std::optional<at::Tensor>& token_mask,
    const at::Tensor& task_info, const at::Tensor& context,
    at::Tensor& buffer, at::Tensor& control_buffer,
    at::Tensor& round_ledger, at::Tensor& workspace,
    at::Tensor& combined_x, at::Tensor& recv_status);
```

```cpp
__global__ __vector__ void combine_recv_impl(
    __gm__ bfloat16_t* combined_x, __gm__ int64_t* combined_topk_idx,
    __gm__ float* topk_weights, __gm__ uint8_t* token_mask,
    __gm__ bfloat16_t* buffer, __gm__ const AFDTaskInfo* task_info,
    __gm__ const AFDContextLayout* context, __gm__ void* control_buffer,
    __gm__ AFDRoundLedger* round_ledger, __gm__ void* workspace,
    __gm__ uint32_t* recv_status);
```

### 1.2 参数与支持范围

| 参数 | 方向、shape/dtype | 契约 |
| --- | --- | --- |
| combined_topk_idx | in int64[B_capacity,K] | 与 dispatchSend 使用同一路由，事务结束前稳定；有效专家互异 |
| topk_weights | in FP32[B_capacity,K] | 权重只在此乘一次，不因同 rank 去重而删权重 |
| token_mask | in optional bool[B_capacity] | token 级；false 输出严格零，不能读取无效路由 |
| task_info/context | in 固定 ABI | 真实 B、物理槽、事务身份及 return_stride；层与发送一致 |
| buffer | in/out uint8 注册返回区 | 实际只读有效 route 的 BF16 hidden；区域所有权由 credit 释放 |
| control_buffer | in/out byte | 读本地全 F Done/Consumed，向全 F 写 credit |
| round_ledger | in/out 每槽两个64B bank | 读已发布的轮描述，归并结束释放 bank；SPSC |
| workspace | in/out uint8 | 与 dispatchSend workspace 不重叠，大小见第4节 |
| combined_x | out BF16[B_capacity,H] | 写 [0,B)，inactive 行零，容量尾部不写；不可 alias 返回窗口 |
| recv_status | out uint32[8] | 完成、错误和最后轮次；成功须包含最终输入 ACK |

只支持 BF16 返回、FP32 乘加，不支持 FP16 返回或训练权重输出。上游 bias0/bias1 不纳入此接口。本成员B=0仍完成DP共同计划的全部零长度轮握手；仅当所有成员B均为0时，共同计划为一个空轮。

## 2. 计算逻辑与 API 序列

### 2.1 语义及复用边界

对有效 token t，`combined_x[t,h]=BF16_RN(sum_{k=0..K-1}(float(buffer[t_round,k,h])*topk_weights[t,k]))`，按 k 升序做 FP32 乘法后加法。所有 K 条 route 有独立返回地址；不按 F 去重归并，不采用原 count=2 的 BF16 快捷加法。inactive 输出零且不读返回 payload。

复用 `combine_reduce_epilogue_impl` 的核间 token 切分，以及 `RingReducer` 的 GM/UB 搬运和 stage 生命周期。新增加权 primitive 是必要的语义改造：原 `combine_reduce_vf` 的无权重 rank-partial 求和不能直接使用。首版保留其 BF16→FP32→BF16 的精度要求，采用下述 Ascend C 高层向量 API 表达加权序列；原 SIMD 的 conversion/store 代码可作为后续等价替换参考，不能把不同实现混算 UB。

### 2.2 API 级伪代码

```text
InitSocState(); 校验 task/context；所有核采用同一错误分支
for round_idx in [0,num_comm_rounds):
  core0: 等对应 Ledger bank 的 PUBLISHED，校验轮次/身份
         copy_control_to_ub：逐个等所有 F 的 Done==ledger.return_bit
  local_join()                       // 此轮 payload 已稳定
  每核负责不重叠 token 区间：
    对 token 的各 hidden chunk：
      AscendC::Duplicate(acc_fp32, 0.0f, h_count)
      若 mask=true：按 k 升序
        asc_copy_gm2ub_align(input_stage[k%2], 返回地址, 1, 2*h_count,...)
        asc_sync_notify/wait(MTE2 -> V, input_event)
        AscendC::Cast(value_fp32, input_bf16, CAST_NONE, h_count)
        AscendC::PipeBarrier<PIPE_V>()
        AscendC::Muls(value_fp32, value_fp32, weight[k], h_count)
        AscendC::PipeBarrier<PIPE_V>()
        AscendC::Add(acc_fp32, acc_fp32, value_fp32, h_count)
        AscendC::PipeBarrier<PIPE_V>()
      AscendC::Cast(output_bf16, acc_fp32, CAST_RINT, h_count)
      asc_sync_notify/wait(V -> MTE3, output_event)
      asc_copy_ub2gm_align(combined_x[t,hidden_begin], output_bf16,...)
      asc_sync_notify/wait(MTE3 -> S, output_event) // 输出UB再复用
  local_join()                       // 全核已结束返回区读取，输出落GM
  core0: 对全 F write_control(ReturnCredit, ledger.return_bit ^ 1)
         完成控制DMA后将本 bank 置 EMPTY
  local_join()
core0: 等最后一轮所有 F 的 Consumed==最后 input_bit
       recv_status=COMPLETE
```

两个输入 stage 用 id0/1，输出 id2，控制 id3，读取 metadata 与控制复用只发生在不重叠阶段。所有相同 pipeline 对的 event 使用必须配对；输入 stage 在 V 使用结束后才允许下次 MTE2 覆写，需 V→MTE2 notify/wait。不得只设置 MTE2→V 单向依赖。高层 LocalTensor 仅映射第3节同一裸 UB arena，不另申请 TPipe 隐含空间。

`Cast/Muls/Add` 为拟采用的具体 API，不是已在 DeepEP 基线找到的同名调用。BF16 的 CAST_NONE/CAST_RINT、FP32 Muls/Add 原地重载和高低层混用事件在目标 CANN 编译器上必须单独编译验证；不支持时阻断该 primitive，不回退 BF16 累加。API依据为本地 arithmetic skill 与原生 FP32 归约数值流程；尚不宣称完成目标版本官方 API 适用性验收。

## 3. 两级 Tiling 与 UB 分配

```cpp
struct CombineRecvTilingData {
  int32_t num_ranks, num_vec_cores, hidden, num_topk;
  int32_t num_max_tokens_per_rank, num_hidden_per_tile;
  int32_t num_ub_bytes, return_stride;
  uint64_t workspace_bytes;
  int64_t num_timeout_cycles;
};
```

Block级：本轮 n 个 token，核 i 处理 `[floor(n*i/N_c),floor(n*(i+1)/N_c))`。N_c 来自 A 侧接收核配额；空核仍参加轮间同步。每个输出 token 唯一 writer，不用 atomic add，不拆一个 token 到多个核。

UB级：一次处理一个 token 的 h 个 hidden，h 是128的倍数，`h=128*floor(min(H,(U-F)/14)/128)`；F为下表固定空间，h≥128，否则拒绝启动。H是128倍数，因此末chunk也按128对齐，不读写 H 外。B与通信轮尾使用实际 token 数。

| buffer | 单份字节 | 数量 | 用途 |
| --- | ---: | ---: | --- |
| input_stage | A32(2h) | 2 | BF16 route 双缓冲 |
| acc_fp32 | A32(4h) | 1 | 当前 token 加权累加 |
| value_fp32 | A32(4h) | 1 | 当前 route 转换及乘权 |
| output_bf16 | A32(2h) | 1 | 最终输出 |
| weights | A32(4K) | 1 | 本 token 权重 |
| expert_idx | A32(8K) | 1 | 本 token 路由校验 |
| ledger_stage | 64 | 1 | 本轮稳定描述 |
| control_stage | 64 | 1 | 读/写各32B |
| status_zero_stage | 128 | 1 | 错误、mask、零初始化暂存 |

`U_recv=2A32(2h)+2A32(4h)+A32(2h)+A32(4K)+A32(8K)+256`。bufferCoefficient=14 B/hidden，仅有 BF16 返回族。K=8 时 F=352B；h=2048→29024B，h=4096→57696B，H=h=7168→100704B。U=253952B 示例可容纳整行，使用率39.65%。更大 H 按公式分 hidden chunk，无全K隐藏矩阵驻留UB。

## 4. Workspace 与内存

业务 workspace=`256+A32(32N_c)`，N_c=8 时512B，含核错误/状态快照，不含独立 RoundLedger。Ledger每槽两bank共128B；持久返回控制数组随Context单独计账。没有 B*H FP32 GM 累加区，没有额外全量返回镜像。

通信预留每槽 `C*K*return_stride`；示例 C=32768,K=8,return_stride=15360，预留3.75GiB。实际有效 payload 每完整轮为`有效token*K*2H`，H7168时满轮3.5GiB。B=262144时普通 HBM combined_x=3.5GiB；它不随通信窗口缩小。三者不是同一笔内存。输入 x/topk/weights 的全事务保活由调用者负责，不能按 C 预算它们。

归并完成后即可释放返回窗口，不必等最终输入ACK再发credit；最终 recv_status 则必须等ACK。send_status 不提供槽复用权限。

## 5. 性能与可选优化

按token分核，无全局FP32归约；两stage可提前搬下一route，但首版保持固定k顺序和明确事件，测量后才扩流水深度。全F Done 包含零route，是交替位闭合条件，不因某F本轮没路由而省略。先等整轮Done再归并会产生尾部等待，首版接受；提前按route归并需要额外可见性标志，不在本次接口内暗加通知。

## 6. Kernel 偏移、精度及生命周期

地址以byte计算：`return_base + slot_offset + (uint64(token_in_round)*K+k)*return_stride + 2*hidden_begin`。输出地址为`byte_ptr(combined_x)+uint64(token_begin+token_in_round)*2H+2*hidden_begin`。检查offset与长度时先转uint64。容量尾不参与任何归约。

masked token 不读取expert_idx和weights中的无效值，直接输出零；有效token非法expert或重复expert导致事务失败。NaN/Inf精度语义遵FP32乘加再BF16 RN，不允许以权重为零跳过读取而改变NaN传播。FP32按固定顺序相加、不承诺跨编译器bitwise相等；数值验收与BF16参考结果比较。

短时序：F0 Done→A仍等F1→F1 Done→A读完所有route→A credit翻位→F可写下一轮；A最终再等所有Consumed→recv COMPLETE。下一轮send可因Consumed先到而发布，但对应Ledger bank未EMPTY不可覆写。全F每轮参与、每槽仅一个未确认输入、同事务串行轮次共同防止旧bit冒充新轮。域超时/错误不伪发credit或ACK，所有本地核一致退出并通知域失败。

## 7. 实现检查清单

- [ ] `impls/afd/combine_recv.hpp` 基于原epilogue与RingReducer拆出A本地返回归并；`csrc/kernels/afd/combine.hpp`沿原JIT入口增加launcher。
- [ ] 验证具体Cast/Muls/Add重载、LocalTensor裸UB视图与事件合法性；BF16转换RN行为与参考一致。
- [ ] K=1/2/8/32、同F多个专家、inactive、零token、全部零route、非满尾轮、H跨chunk。
- [ ] TP/CP多成员、P跨mb共享slot、D三slot发布、延迟Done/Consumed、0→1→0多次复用、图replay不重置位。
- [ ] 验证返回预留/实际流量/输出HBM分别计账；无全B FP32临时张量。
- [ ] 验证send/recv并发资源，等待kernel不占满AIV造成对方无法运行；失败时全域终止可观测。

## 8. 源码与交付状态

[原epilogue入口](D:/Code/refs/DeepEP-Ascend/deep_ep/include/deep_ep/impls/ep/combine_reduce_epilogue.hpp:45)、[原数值归约](D:/Code/refs/DeepEP-Ascend/deep_ep/include/deep_ep/impls/ep/combine_utils.hpp:15)、[RingReducer](D:/Code/refs/DeepEP-Ascend/deep_ep/include/deep_ep/impls/ep/combine_utils.hpp:97)、[原launcher](D:/Code/refs/DeepEP-Ascend/csrc/kernels/ep/combine.hpp:65)。API补充依据：[arithmetic最佳实践](C:/Users/puxiaokun/.agents/skills/ascendc-api-best-practices/references/api-arithmetic.md:80)。

按skill提供接口、API序列、两级tiling、UB系数、workspace、精度和实施检查清单。编译、跨通路可见性、图和A5性能均为待验收，不能将本文视为已运行代码。

## 9. 本次实现落点（待 CANN/A5 验证）

实现为 `deep_ep/include/deep_ep/impls/afd/combine_recv.hpp`，模板 `<E,N,K,H,HC,NC,Timeout>`，`HC` 为128倍数的 hidden chunk，`NC` 为实际 launch 核数。注册返回区 device 参数采用 `uint8_t* buffer`，地址先按byte定位再转换BF16。反向目录从 `context.reserved_address` 读取 `AFDReturnDirectory/AFDReturnLane`；credit直接复用已有 `store_control` 对 F 的UBMEM控制映射写入，无需额外jetty参数。payload传输仍由combineSend的Hcomm路径完成。

实现沿原epilogue的token核间切分，复用RingReducer的批量DMA形状和双stage生命周期；两个输入stage预加载当前/下一route，每stage完成V读取后才允许MTE2 refill。原生rank-partial去重和BF16 count=2快捷归约不适用于逐route权重语义，数值primitive依第2节改为FP32固定K顺序乘加、最后一次BF16转换。路由预检按核分配token，K-wide metadata一次DMA入UB，汇总后才输出。无整轮FP32 GM镜像。

全部本地核完成返回区读取后，core0完成所有F的credit写入，再释放Ledger bank；最后状态需等全部Consumed。零长度轮同样执行这些控制步骤。错误不发布伪credit/ACK，运行时观察ERROR后终止该域，不能继续复用旧槽。

`CombineRecvUB<K,HC>`符合第3节14字节系数；`combine_recv_workspace_bytes(NC)`符合第4节。六项CPU语义/协议测试已通过，不能替代编译或设备结果。LocalTensor裸UB绑定、BF16 Cast重载、FP32原地运算、混合事件依赖与UBMEM跨卡可见性仍为A5上卡门禁。
