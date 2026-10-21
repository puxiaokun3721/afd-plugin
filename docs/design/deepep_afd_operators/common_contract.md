# DeepEP-AFD 算子公共接口、布局与原语契约

本文件与四个通信算子的 design.md 合起来构成代码生成输入；ffnScheduler/design.md 的范围为接口契约，当前开发实现状态见[设计入口](../README.md)。它们细化[五算子总设计](../a5_async_afd_deepep_operator_design.md)。实现目标是 DeepEP-Ascend 仓库，基线commit `3b25377d04b24fc6154698ded78a2bcb2c59afff`，不是另建一个无关Ascend算子工程。

## 1. 公共约定与类型

B_cap,m（maxBS）为单Attention rank完整事务最大行容量；B_m（简写B）为本次算子的逻辑行范围长度，包含mask=false位置；C_m（简写C）为该rank单物理槽单轮通信容量，独立于B_cap,m。H为hidden，K为top-k，E为总专家，N为FFN rank数，R=E/N，L为注册(member,slot)数，Q/Q_alloc为F单计算片有效/物理专家行容量，a为专家行对齐。DP每轮容量 C_DP=Σ_m C_m，不设固定32768合计上限；同一成员的其他D槽不重复计入一个DP条目，而F注册预留必须包含所有槽。当前单lane实现族保持0<C_m≤32768，不能把取消DP限制理解为放宽单lane实现边界。`A32(x)=32*ceil(x/32)`、`A512(x)=512*ceil(x/512)`；所有公式先按uint64计算。GiB=2^30B、MiB=2^20B。

容量与字段映射：当前host `num_token_capacity` 对应 B_cap,m，device `AFDTaskInfo.num_tokens` 对应 B_m，lane_config 的容量对应 C_m；保留上游名的 `num_max_tokens_per_rank` / 模板容量为 C_max=max_lane C_m。上游同名参数约束一次原生EP调用容量；AFD以单通信轮映射，不能用它限制完整事务maxBS。Q/Q_alloc在设计接口中分别称为 `num_max_expanded_tokens/num_max_padded_tokens`，当前实现入口为 `prepare_recv(q,q_alloc,...)`，host保存为 `recv_q/recv_q_alloc` 并传入JIT模板；当前prepare_recv要求Q_alloc≥Q，并要求a为正2次幂。选片同时满足有效行 `Σ_e n_e≤Q` 与物理行 `Σ_e align_a(n_e)≤Q_alloc`，不足时继续分片，不自动扩大张量。

DP共同轮数 `num_comm_rounds=max(1,max_m ceil(B_m/C_m))`；成员m第j轮为 `[min(B_m,j*C_m),min(B_m,(j+1)*C_m))`。C_m异构时各自推进；短成员耗尽后仍发剩余全部轮的零长度Ready并参加全F完成闭合。B_m=0合法，但注册C_m仍为正。框架rank真实BS记为B_rank：普通owner不压缩时B_m=B_rank；TP非owner可用B_m=0，或保留副本行范围并全mask=false，后者输出相应行清零；非连续owner压缩时必须保留token映射。CP不相交owner按各自B_m发送完整H维数据，不重复计算TP副本。

例：同DP两个owner均B_cap=B_m=262144、C_m=32768，统一8轮，每轮DP合计65536。K=8且单F集中时每轮524288条route；a=1、Q=Q_alloc=32768时16片。增加来源改变待处理量、注册输入和route表预留，不改变单片workspace。两owner的F输入为2*32768*record_stride，A各自返回仍32768*K*return_stride；不可声称全域或F单卡总额不变。

所有聚合和地址乘积用uint64计算；每lane route reference仍uint32，Active/prefix仍受int32及既有ABI边界约束，必须检查后再窄化。Pending记录成员游标和计数，不申请DP总token数组。scheduler/combine现已提供源码及host/JIT入口，尚未通过CANN编译或A5设备验收；不能把接口文档或CPU模型通过视为设备能力证明。

输入BF16（2B）或E4M3FN（1B）+上游sf_pack_t（int16存储，2B），不支持FP16。sf不是INT16数值量化参数的同义词。返回为BF16；topk_idx int64、weights FP32、mask bool。BF16搬运按位复制，不升精度；只有combineRecv算术使用FP32。首版K≤32、R≤512、H为128元素倍数；该H族用于直接复用上游128元素SIMD循环，其他H需新tiling族并单独验收，不能静默padding整行改变接口。

通信模板沿用kNumRanks/kNumExperts/kNumTopk/kNumVecCores/kNumUbBytes/kNumTimeoutCycles；新增来源维度kNumSrcRanks/kNumSourceLanes，容量kNumMaxExpandedTokens/kNumMaxPaddedTokens。num_ranks仅指F；src_rank_idx指A；peer_idx是通信域编号。shape、dtype、设备、stride、合法层族在host校验；实际num_tokens、round、layer由device快照读取。

所有输出在调用前分配，host入口返回void（out式接口），不照搬模板中单输出Tensor返回例。公共算子名遵用户要求lowerCamelCase；内部保留snake_case，不能把x改名hiddenStates或把sf改成含义不明的scale。NumPy当前安装公开命名检查无五个同名接口；本机无torch，PyTorch官方文档限定域检索无同名结果，未伪称执行了torch运行时检查。

## 2. 线性布局和固定大小结构

以下为新增AFD ABI v1，不能与上游EPWorkspaceLayout或cached handle互传。所有uint32枚举与count使用显式值；uint64 offset以当前所属allocation起点为基准。每种table按A32(stride*capacity)预留；get_* accessor通过已冻结directory取地址。禁止device std::vector/host pointer。

```cpp
struct alignas(32) AFDControlBlock { uint32_t value, reserved[7]; }; // 32B
struct alignas(32) AFDTaskInfo { // 64B
  uint32_t abi_version, src_rank_idx, payload_slot_idx, microbatch_idx;
  uint32_t layer_idx, num_tokens, num_comm_rounds, reserved[9];
};
struct alignas(32) AFDInbox { // 64B
  uint32_t microbatch_idx, layer_idx, round_idx, num_comm_rounds;
  uint32_t num_round_tokens, return_bit, num_recv_records, reserved[9];
};
struct alignas(32) AFDRoundLedger { // 64B
  uint32_t state, input_bit, return_bit, round_idx;
  uint32_t token_begin, num_round_tokens, final_round, microbatch_idx;
  uint32_t reserved[8];
};
struct alignas(32) AFDActive { // 64B，数组放在同一active allocation之后
  uint32_t valid, kind, phase, layer_idx;
  uint32_t num_entries, num_members, num_segments, num_expanded_tokens;
  uint32_t num_padded_tokens, reserved;
  uint64_t entries_offset, members_offset, segments_offset;
};
struct alignas(32) AFDActiveEntry { // 32B
  uint32_t pending_idx, member_begin, num_members, segment_begin;
  uint32_t num_segments, num_rows, is_final, grant_state;
};
struct alignas(32) AFDActiveMember { // 32B
  uint32_t source_lane_idx, input_bit, return_bit, delta_begin;
  uint32_t num_experts, reserved[3];
};
struct alignas(32) AFDActiveSegment { // 32B
  uint32_t entry_idx, source_lane_idx, local_expert_idx, route_begin;
  uint32_t num_routes, dst_row_begin, reserved[2];
};
struct alignas(32) AFDExecutionState { // 32B头，后接uint32 grant_state[L]
  uint32_t stage, error_code, num_entries, reserved[5];
};
struct alignas(32) AFDCompletion { // 32B头，后接entry/member快照和uint32 delta[L*R]
  uint32_t state, error_code, num_entries, num_members;
  uint32_t num_deltas, reserved[3];
};
```

Active的offset由host初始化并保持不变；scheduler只写内容/有效count。ActiveEntry容量L、Member容量L、Segment容量LR。按表切片而非另一份Q行desc；Segment改为32B对齐stride，取代总设计“6个uint32=24B”作为申请大小的简写，逻辑字段不变，新增开销8LR字节必须计账。Completion布局固定为`32 + A32(32L) + A32(32L) + A32(4LR)`，entries与members与Active字段同构；entry.grant_state在完成副本中更新，delta以member.delta_begin定位。

ActiveEntry.is_final表示本F该DP条目所有成员的剩余route都被本片覆盖，而非某成员局部末片。各成员delta是R维有效消费行数；padding不消费route。所有Entry/Member从Active复制后再ACK，ACK后Completion不读Inbox。

`kind`: NOP=0,WORK=1,CONTROL=2,ABORT=3；grant NONE=0,HELD=1,FINISHED=2；execution EMPTY=0,SELECTED=1,INPUT_READY=2,COMPUTE_READY=3,COMMITTED=4,ABORTED=5；Completion EMPTY=0,ARMED=1,READY=2。通信控制位与这些本地枚举独立。

Context只读目录必须包含：lane(C、DP/member/phase/slot/F输入offset/A返回offset)、DP成员映射、layer_table、各peer注册地址/长度/域编号、buffer/control/workspace offsets。目录自身由host对象构建，绑定类型`const at::Tensor& context`；每个region条目`uint64 offset,length; uint32 stride,capacity; uint32 reserved[2]`为32B。所有地址检查用`offset<=length && bytes<=length-offset`。动态层不能改变H、dtype、scale格式或tiling族。

## 3. wire record 和输出

X=H*input_element_bytes，S=2*num_sf_packs。record字节offset依次为hidden=0、sf=X、topk_idx=X+A32(S)、topk_weights=topk_idx+A32(8K)、src_token_idx=topk_weights+4K、valid_topk_mask=src_token_idx+4。D=A512(valid_topk_mask+4)。metadata_stride=M=A32(A32(S)+A32(8K)+4K+8)。

H=7168,K=8时：FP8 S=224 → X=7168,M=352,D=7680；BF16 S=0 → X=14336,M=128,D=14848。metadata padding写零；payload record的尾512B对齐空洞不传、不读。source token为当前轮内下标，来源lane从固定分区得出。route_reference=record_idx*K+topk_slot_idx，uint32；UINT32_MAX非法；乘积必须严格小于它。

F route表为每lane每local expert连续区；count[R]/exclusive-end prefix[R]无padding，A发布；计算片prefix[R]含padding，scheduler输出，两者禁止混用。recv_src_metadata为int32[Q_alloc,4]，字段lane/token/topk/entry；padding全-1。combineSend的src_metadata引用同一张量，不能传上游原K+2布局。

A返回布局从Context.return_stride读取，要求≥2H且512B倍数。本版沿总设计使用**15360B作为示例预留stride**；H=7168时最小A512(2H)=14336B，但不擅自缩减原预算配置。该stride不承载MTE data-as-flag。所有公式使用注册的return_stride，wire仅写2H。不能将预算常量和实际数据长度混用。

## 4. 发布、完成和公共原语的API展开

这些是要新增的DeepEP helper，不是已有API。按本地源码API族实现；远端可见性仍有明确实机门禁。

1. `copy_control_to_ub`：`asc_copy_gm2ub_align(ub32, local_gm,1,32,...)`；用上游已用缓存重载；`asc_sync_notify(PIPE_MTE2,PIPE_S,id)`/`asc_sync_wait`后读UB uint32。轮询不读远端，不循环GlobalTensor::GetValue。
2. `write_control`：负责人在独占UB32写value、reserved清零；S→MTE3 notify/wait；`asc_copy_ub2gm_align(mapped_peer_control,ub32,1,32,...)`；MTE3→S notify/wait。只作用于注册的UBMEM映射地址。前一DMA完成前不能重用ub32。
3. `local_join`：继承barrier.hpp中的`AscendC::SyncAll<true>()`局部阶段，所有启动AIV同次序调用，空工作核不早退；不同并发kernel不得共用自建同步区。采用CANN该启动模式的SyncAll资源配置，不推断全rank barrier已经取消就不需本地同步。
4. `put_at_offset(peer,remote_offset,src_ptr,num_bytes)`：从HcommJetty::build_write_sqe/put改出；校验peer.rmt_buffer_size；remote_addr=peer.rmt_base_addr+remote_offset。保留强序、fence、CQE、SQ overflow轮询、copy_sqe_to_sq、advance_sq、ring_doorbell。首版每请求CQE，不再使用无完成覆盖证明的手写批量SQE快捷路。
5. `drain_payload`：每发送AIV调用其`HcommJetty::drain<kNumTimeoutCycles>()`；确认packed_head统计覆盖本轮全部请求，随后local_join。发布Ready/Done之前需要URMA写→UBMEM控制的远端可见性保证，此跨通路保证必须在A5目标版本验证，不能用本地event替代。
6. `publish_completion`：先输出所有entry/member/delta（普通GM MTE3），MTE3→S依赖及local_join后负责人写READY。没有跨F同步。

`HcommPeerInfo=48B`，`HcommJettyInfo=88B`为本基线static_assert。为每核预留96B jetty区、512B SQE暂存；要求`HcommJetty::kNumMaxSQEBytes<=512`编译断言。每个目标channel首版只注册一个payload region，沿用原`remoteBufferNum==1`校验：F一个输入大region内划lane，A一个返回大region内划slot。控制UBMEM另行建链。这样不改load_peer_info多region行为；A/F两个region可不同大小。

## 5. 生命周期和公开状态

input_ready/Done/Consumed初值1，credit初值0，首次期待0；每轮全F都参加，包括零route。A输入覆写等全Consumed；返回覆写等credit；最后MoE事务跨mb槽复用等Etxn。两个通道独立，send独占input位和计划return位，recv独占归并return位。

A workspace拆send/recv两份，Ledger严格SPSC两bank；不允许图alias分析强制send全事务后才recv。F执行链每次scheduler→recv→GMM→send，单Active；GMM2适配尾部在全片写完后设置COMPUTE_READY。CONTROL跳过GMM但经combineSend完成；NOP不产生Completion。错误不能让单核提前退出并遗留SyncAll参与者，错误状态先本地一致广播再走域终止。

多本地槽多通信轮使用 combineRecv round-step：同一接收stream依次probe所有活动槽，未到齐不等待，每次至多归并一轮；最终ACK同样非阻塞。每槽额外96B持久进度、共享单recv scratch，固定地址的首次start与重复sweep分离。全事务模式只准入单本地注册槽，或多槽单轮。Context原reserved字段命名为`return_progress_mode`（0=全事务，1=round-step），结构长度不变；初始化冻结，send与recv共同校验，旧二进制不得处理mode=1。完整调用契约见[combineRecv](combineRecv/design.md)，活性证明见[总体设计§4.5.2](../a5_async_afd_design.md#452-单-workspace-先算后等的进度证明)。

CONTROL（包括WORK中的零行entry）只有全部成员credit可用才选入Active；非空组仍保留COMPUTE_AHEAD。Pending私有原reserved 4B保存`num_remaining_routes`，在接纳时由已检查的count求和、应用Completion时按delta扣减，并与count/cursor核验；Pending保持96B，wire、Ready/Consumed/Done/credit及Q/Q_alloc均不变。

错误码OK=0,BAD_ABI=1,BAD_TASK=2,BAD_LAYER=3,BAD_ROUTE=4,CAPACITY=5,STATE_ORDER=6,TIMEOUT=7,TRANSPORT=8。host异步提交返回不代表device成功；32B status内state/error/round记录进度，成功条件以具体算子文档为准。

## 6. 配置验证与代码落点

新增源码放DeepEP仓库的`deep_ep/include/deep_ep/impls/afd/`和`layout/afd/`；host接入`csrc/kernels/afd/`、`csrc/buffers/afd.hpp`及现有`csrc/python_api.cpp`，保留runtime JIT、通信资源管理和原EP实现。所有这些是计划位置，本次不创建源码。不得把新功能塞进原EP默认分支后破坏训练接口。

文档中的UB预算上限示例U=248KiB仅用于算术验算；host必须用原runtime->get_num_ubuf_bytes_per_vec_core()取得实际U。每个kernel显式从同一裸UB arena分配，不另建TPipe/TQue偷偷占用未记账UB。编译器额外需求若存在，必须体现在可用U或编译诊断，不能静默溢出。

W_transport定义为实际创建的SQ allocation bytes + CQ allocation bytes + channel/peer注册资源bytes +平台runtime workspace bytes，逐allocation ID去重求和。它是启动时资源返回值，不是sizeof tiling；创建后必须与用户配置memory_budget_bytes比较，超预算拒绝启动。四文档的确定性GM区间另行给出，不能用W_transport掩盖漏记的业务数组。


## 7. 位、Ledger与错误的补充约束

位按物理lane跨microbatch和graph replay持久保存，不能每次调用从0重新开始。初始化Ready/Consumed/Done=1，credit=0；send计划input_bit=0、return_bit=0，F期待输入0，A期待Done0。第r轮使用当前计划位，发布后send只推进自己的下一次计划；F在Completion应用后推进输入期待位，A在本轮归并结束后推进返回期待并写credit。状态字段均uint32，单独32B控制块唯一writer。

Ledger.state明确为EMPTY=0、PUBLISHED=1；每槽两bank按本事务round_idx%2定位。send必须等目标bank EMPTY，再写完整字段和PUBLISHED；recv只有读到PUBLISHED才取快照，所有本轮返回读结束且credit DMA提交完成后置EMPTY。bank没有独立generation，因为同槽单一生产者/消费者顺序交接、前一bank释放后才能重用。事务切换还须Etxn，不允许把新microbatch绕过旧事务的最终ACK。仅当DP所有成员B均为0时num_comm_rounds=1；一般情况使用max(1,max_m ceil(B_m/C_m))，包括B_m=0及提前耗尽的成员都继续发布零token轮直到共同末轮。

交替位闭合短证明：F_i 的第r+1轮返回必须看见A释放r的credit；A只有收齐所有F第r轮Done并读完数据才释放它；A想使credit再翻回r的值，必须再收齐包含F_i在内的r+1轮Done。因此一个尚未观察新credit的慢F不能被其他F带着跨过两轮而误认旧值。零route F也执行同一Done/credit握手；注册F集合在事务间也不可临时省略，变更需全域静止后重新初始化。证明依赖每channel单writer、写完成后才复用UB控制源、无重传/重连/旧DMA残留；不覆盖网络乱序重传。

同理输入最多一波未确认：A不能在旧Consumed前发布同槽下一输入，F未应用末片Completion前不接收该槽下一Ready，因而每次匹配仅入队一次。ACK发布前，所有后续所需的Inbox字段/成员/delta必须已复制到Active或Completion；ACK后只使用本地快照。

SQ批提交边界：`put_at_offset<timeout,false>`不得在尚未doorbell的请求填满SQ后进入等待CQ的路径。必须在余量不足下一WQE前先ring_doorbell，再poll回收；tile末仍ring+drain。直接复用原poll_cq_when_sq_overflow之前增加这一步，或者使用每请求commit=true的保守实现。两者保持同一API；不得假定tile大小小于任意运行环境SQ容量。

所有算子发现错误先停止新增通信，收敛本地核错误状态，完成已启动本地DMA的安全退出；不能跳过其他核已进入的local_join。外部运行时必须能终止对端poll/通信域并向host报告错误。具体A5异步域终止API若未验证则列为部署阻断；超时不等于成功，也不允许通过伪ACK继续运行。
