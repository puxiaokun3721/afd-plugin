# CAM Async MoE 通信算子优化笔记

> 配套文档：`CAM_ASYNC_OPERATORS_CODE_WALKTHROUGH.md`（源码精读）。
> 本文从 10 个角度分析四个算子（`cam_moe_distribute_dispatch_send / dispatch_recv / combine_send / combine_recv`）的可优化点。
> 结论均来自源码分析，未做实机 profile；涉及「收益/成本」的判断为经验估计，落地前需用 `timeCycleTensor_` 埋点（combine_send 已有）或 profiling 工具验证。

---

## 0. 先说结论（优化优先级总览）

| # | 方向 | 收益 | 改造成本 | 优先级 |
| --- | --- | --- | --- | --- |
| 1 | 分核策略 | 中 | 中 | ★★★ |
| 2 | MTE2→V→MTE3 流水线 bubble | 高 | 高 | ★★★★★ |
| 3 | UB 内存复用 | 中 | 低 | ★★★★ |
| 4 | GM 访问粒度 / burst | 高 | 中 | ★★★★ |
| 5 | 等待开销（自旋 + SyncAll） | 高 | 中 | ★★★★★ |
| 6 | PreProcess 前缀和 GM 读写 | 中 | 中 | ★★★ |
| 7 | Expert Token Offset 批量 flush（16） | 低-中 | 低 | ★★ |
| 8 | 路由表 64KB 页缓存 | 中 | 低 | ★★★ |
| 9 | 动态量化开销 | 中 | 中 | ★★★ |
| 10 | winSizePerAttn_ 窗口大小 | 高（内存侧） | 低 | ★★★★ |

> **2026-08-21 更新**：dispatch_send 走读完成后，与作者对齐了三大重构方向及落地顺序，见文末 **§11 走读确认的重构路线**。其中「共享专家上移 Attention 侧」为架构级决策，排在所有代码级优化之前。

---

## 1. 分核策略：当前按 batch 均分，是否有更好的方式？

### 现状

- dispatch_send：按 **batch** 均分（dispatch_send.h 第 251–263 行），每个 AIV 负责 `[start*topk, end*topk)` 的 topkId 区间。
- dispatch_recv / combine_send：按 **token** 均分（dispatch_recv.h 第 465–475 / 630–639 行；combine_send.h 第 263–274 / 399–410 行）。
- combine_recv：按 **batch** 均分（combine_recv.h 第 178–187 行）。

### 问题

1. **负载不均衡**：dispatch_send 按 batch 均分，但不同 token 的路由分布不均匀（MoE 的 load-imbalance 是常态）。极端情况下某个 AIV 负责的 batch 段里所有 token 都路由到少数专家，导致该 AIV 的 `DataCopy` 次数远超其他核，出现「木桶短板」。PreProcess 里用前缀和算了每个 AIV 的发送量（`aivCntTensor`），**这份统计其实已经能算出每个 AIV 的负载**，但分核发生在 Init（第 252 行），统计发生在 PreProcess（第 671 行），二者没打通——分核没利用负载信息。

2. **分核粒度是 batch（token），与共享内存窗口按 attnRankId 切分的结构耦合**：dispatch_send 每个 AIV 写多个 MoE 卡窗口时，跨卡写地址不连续，导致后面的 DMA 无法合并。

### 建议

- **负载感知分核**：把 PreProcess 的统计结果前置（或做一次轻量预估），按「每个 AIV 负责的 token 数大致相等」而不是「batch 数相等」来划分。这需要一次额外的路由表预扫描，但 dispatch_send 本来就要 `GetIds` 遍历一遍路由表（PreProcess 第 671 行），可以顺便把「每个 token 的专家目标」缓存下来，二次遍历时直接按目标分核。
- **按专家维度分核**（dispatch_recv/combine_send 场景）：当前按 token 均分，但 token 的目标专家分布不均。改为「按专家分核 + 专家内 token 均分」能天然对齐 `expertTokenCount` 结构，减少 `tpExpertTokenOffset` 的前缀和计算。
- **动态核数**：`aivNum_` 取自 `GetCoreNumAiv()`（各 host 第 189 行），上限 48。当 batch 很小时（如 < aivNum），大量 AIV 空转。可在 tiling 阶段按 batchSize 收缩 blockDim。

---

## 2. 流水线：MTE2→V→MTE3 的 pipeline 是否充分？

### 现状（以 dispatch_send 非量化路径为例，第 479–483 行）

```cpp
DataCopy(xTensor_, xGMTensor_[hiddenSize_*tokenId], hiddenSize_);  // MTE2 读
SyncFunc<HardEvent::MTE2_S>();                                     // 等读完
SyncFunc<HardEvent::S_MTE2>();
SyncFunc<HardEvent::S_MTE3>();                                     // 才允许写
```

### 问题：几乎是「读一个 → 等一个 → 写一个」的串行，没有软件流水

1. **token 级 bubble 严重**：每搬一个 token 都要 `MTE2_S → S_MTE2 → S_MTE3` 全等，MTE2/MTE3 之间没有任何 overlap。一个 token（hiddenSize=7168 的 BF16 约 14KB）DMA 是微秒级，但同步开销 + 标量计算（`GetIds`、`PushExpertTokenOffset`）会放大。
2. **`xTensor_` 单缓冲**：`xTensor_` 只有 1 个 token 大小，无法双缓冲（double-buffering）。dispatch_send 有 `ubReuseSize_`（第 369 行）却没用上（见 §3），完全可以开双缓冲。
3. **combine_send 反而有批量 staging**（`xTensor_` 吃满剩余 UB，第 195、201 行），是四算子里唯一有 pipeline 意识的地方；但它仍是「读入 cnt 个 → 全部读完 → 再写出 cnt 个」（combine_send.h 第 340–363 行），读与写之间没有 ping-pong。

### 建议

- **双缓冲 / 多缓冲**：把 `xTensor_` 开成 2~4 个 token 的环，用 `DataCopy` 的异步语义 + `SetFlag/WaitFlag` 让「第 i 个 token 的 MTE3 写」与「第 i+1 个 token 的 MTE2 读」重叠。这是收益最大的一项，但也最需要重写主循环。
- **降低同步粒度**：`MTE2_S → S_MTE2` 两个事件可以用一个 `MTE2_V` 或更粗的 barrier 替代；对纯搬运（非量化、无 V 依赖）的路径，读和写之间其实只有「数据源不变」这一个约束，可用 `DataCopy` 的连续地址一次搬多个 token（见 §4）。
- **combine_send 已有埋点**（`timeCycleTensor_`，第 221–245 行），可直接用它量化 bubble 占比再动手。

---

## 3. UB 内存：分配是否紧凑？有没有可复用空间？

### 现状

四个算子都用 `GetWithOffset` 顺序分配，分配后**几乎不复用**。dispatch_send 的 UB 布局见 walkthrough §4.3。

### 问题

1. **`aivCntTensor_` 用完即弃但占大头**：dispatch_send 里 `aivCntTensor_` = `uint32 × (moeNum+expertNum) × aivNum`。以 moeNum=16、expertPerRank=8（expertNum=144）、aivNum=48 计：`(16+144)*48*4B ≈ 30KB`。它只在 PreProcess 用一次，之后永远不再碰，但分配在 UB 中间（第 632 行），把后续张量往后挤。**应该把 PreProcess 临时区放到 UB 末尾或复用**。

2. **`expertTokenOffsetTensor_`（16×expertNum uint16）与 `expertTokenCountOnSendTensor_` 语义重叠**：前者是「待 flush 的批量缓存」，后者是「已发送计数」，都按 expert 索引。批量 flush 完成后前者可释放复用。

3. **dispatch_recv 的 `tokenAddrOffsetTensor_` 吃满剩余 UB（第 257–259 行）**，这其实是好的设计——越大越能一次读入更多偏移。但反过来说明前面几个固定张量没做空间回收。

4. **combine_recv 三个 float 缓冲**（`rowTmpFloatLocal_/sumFloatBufLocal_/mulBufLocal_`，各 28KB）只用于累加，`rowTmpFloatLocal_` 和 `mulBufLocal_` 可用 `sumFloatBufLocal_` 原地替代（`Muls` 支持 in-place），省 56KB。

### 建议

- PreProcess 的 `aivCntTensor_` 挪到 `ubReuseSize_` 区（dispatch_send 第 369 行已算出剩余空间），用完即释放给 token staging。
- 量化与非量化路径的 `xTensor_/xOutTensor_` 与 `quant*` 三件套互斥，可以显式 union 化。
- 对 BF16（非量化），`xTensor_` 与 `xOutTensor_` 是同一块（第 343–344 行同 offset），但语义上可以合并变量名，减少误用。

---

## 4. GM 访问：DataCopy 粒度是否最优？有没有 burst 优化？

### 现状

- dispatch_send 每次 `DataCopy` 只搬 **1 个 token**（`hiddenSize_` 个元素，第 459/506/565 行）。
- dispatch_recv 每次 `DataCopy` 只搬 **1 个 token**（第 520/531/684/695 行）。
- combine_send 有批量（`cnt` 个 token，第 340、362 行），但 `cnt = min(..., maxUbTokenNum_)` 受 UB 限制。
- token 偏移索引用 `uint16_t` 数组，读回时 `DataCopyPad`（第 485/649 行）。

### 问题

1. **逐 token 搬运 = 大量小 DMA**。GM 访问最优是**连续大 burst**。dispatch_send 里 token 在共享内存是连续排布的（`tokenOffset + tokenOffsetOnSend*hiddenSize`），同一 MoE 卡的多个 token 相邻，理论上可一次搬多个，但被「1 token 1 次」拆碎。

2. **跨卡写地址不连续**：`dstRankId = attnNum_ + moeRankId`，`GetWindAddrByRankId` 每次取不同卡的 peer 地址（第 503、555 行），即便 token 连续，目标也因卡而异，天然不利于 burst。这是**通信算子的固有约束**，只能靠「按卡聚合 token」缓解。

3. **`PushExpertTokenOffset` 用 `uint16_t` 存偏移**：当 `maxBatchSize > 65535` 时会溢出（`tokenOffsetOnSend` 是 uint16）。目前 `maxBs` 上限 `maxSeqLen/tpSize`，若 maxSeqLen 大则溢出风险存在（walkthrough §3.1 已提 `tokenAddrWinSize` 用 uint16）。这是正确性隐患，不只是性能。

### 建议

- **按目标卡聚合 token 后批量搬运**：dispatch_send 主循环可先按 `moeRankId` 把本 AIV 的 token 排序/分组（PreProcess 的 `aivCntTensor` 已按卡统计），然后对同一张卡连续 `DataCopy` 一整段。代价是重排 token 顺序 + 需要额外 staging。
- **把 `uint16_t` 偏移升级到 `uint32_t`**（或按 batchSize 自适应），消除溢出隐患；`tokenAddrOffset` 区当前用 `uint16` 省空间，但正确性优先。
- **combine_send 的 `cnt` 可再放大**：`maxUbTokenNum_` 受 UB 限制，可配合 §3 的复用把 staging 再扩大，减少循环次数。

---

## 5. 等待开销：SyncAll 频率与 flag 轮询

### 现状

- `SyncAll<true>()` 出现 8 处（见 walkthrough §10.2），其中 dispatch_send 3 次、dispatch_recv 2 次、combine_send 2 次、combine_recv 1 次。
- dispatch 方向：aiv0 紧自旋轮询所有 Attention 卡的 flag（dispatch_recv.h 第 776–805 行），其他 AIV 轮询 aiv0 的通知（第 806–824 行）。
- combine 方向：所有 AIV 各自 `while(true)` 紧自旋轮询 MoE 卡 flag（combine_recv.h 第 523–539 行）。

### 问题

1. **紧自旋无 backoff**：`while` 循环里 `DataCopyPad` 读远端 GM 是**高频跨卡读**，会占用 interconnect 带宽、和正常数据搬运抢资源。`comm_args.h` 里有 `MAX_WAIT_ROUND_UNIT = 10*1000*1000`（第 37 行）这种「正常等待轮数」概念，但这两个等待循环**没有用任何轮数上限/退避**。

2. **combine_recv 所有 AIV 各自轮询**：与 dispatch_recv「只有 aiv0 轮询 + 广播」的设计不一致。48 个 AIV 同时跨卡读同一个 flag 区，放大无效流量。**应该复刻 dispatch_recv 的 aiv0-单点轮询-广播模式**。

3. **SyncAll 不能省，但可减少次数**：dispatch_send 的 3 次 SyncAll 中，PreProcess 的 2 次（第 717、741 行）是前缀和必需的；SetFlags 的 1 次（第 778 行）是为了让所有 AIV 写完 token 数据再统一写 flag。理论上 SetFlags 的 SyncAll 可与 PreProcess 的第二次 SyncAll 合并（若数据依赖允许）。

### 建议

- **flag 轮询加退避**：轮询循环里插入 `NOP`/短等待，或用 `MAX_WAIT_ROUND_UNIT` 做超时保护（防死锁）。有硬件 idle 指令时用 idle 替代空转。
- **combine_recv 改 aiv0 单点轮询**，与 dispatch_recv 对齐。
- **评估用 CAM 通信层原生 notify 替代 flag 自旋**：`comm_args.h` 提到 `NOTIFY_DISPATCH_BUFF_OFFSET`（第 20 行），说明 CAM 层可能有 notify 原语，比轮询更省带宽。

---

## 6. PreProcess 开销：前缀和的 GM 读写能否减少？

### 现状（dispatch_send.h 第 629–773 行）

前缀和流程涉及 **4 次跨 GM 读写 + 2 次 SyncAll**：

1. 各 AIV 写自己的统计到 GM（第 713 行）
2. 读回所有统计（第 720 行）
3. 按列并行前缀和，**每步写回 GM 一个元素**（第 736 行）
4. 读回完整前缀和（第 744 行）

### 问题

1. **第 3 步「每步写回一个元素」是最大的浪费**：`for row in [1, aivNum_): DataCopyPad(aivCntGMTensor[...], aivCntTensor, {1, sizeof(uint32)})` —— 每次只写 4 字节，还带 `MTE3_S` 同步。aivNum=48 时，每个负责列要写 47 次、每次 4B。**前缀和完全可以在 UB 里算完，最后一次性写回**，中途不需要把中间结果落 GM。

2. **`aivCntTensor(0) = sum`（第 734 行）**：细读勘误——这**不是调试残留**，而是功能性代码：`DataCopyPad` 的源必须是 UB 地址，标量 `sum` 无法直接 DMA，所以借 UB 位置 0 当 staging 区把单个 uint32 写回 GM。它确实会破坏 `aivCntTensor[0][0]` 的本地值，之所以没出 bug，是因为第 744 行会整表重新读回覆盖。**写法脆弱但正确**；若按下一条建议改成「UB 内算完一次性写回」，这个 staging 技巧连同其脆弱性一起消失。

3. **两次 SyncAll + 两次全量读回**：可以改为「写统计 → 1 次 SyncAll → 各 AIV 在 UB 里做完整前缀和 → 1 次 SyncAll → 各 AIV 读自己需要的列」，省掉第二次全量读回（第 744 行）。

### 建议

- 把按列前缀和整体搬到 UB 内计算（每个 AIV 已有全量数据在 UB 里），**只在需要跨 AIV 交换时写 GM**，最终结果写回一次（同时消除第 734 行的 staging 写法）。
- 若坚持按列并行，可改为「每列负责 AIV 算完本列前缀和后一次性 `DataCopy` 写整列」，而不是逐元素写。

### 6.1 额外方向：PreProcess 中大量标量循环可用向量 API 替换

PreProcess 内有至少 **4 处** `for` 循环 + 逐元素赋值，全部跑在标量单元（Scalar Unit）上，效率很低：

| 位置 | 循环体 | 当前做标量什么 | 可替换为向量 API |
| --- | --- | --- | --- |
| `DispatchSend` 第 396–406 行 | `for expertId: expertPreTokenNum_ = expertPreTokenNum_ + count` | 逐元素标量累加前缀和 | `AscendC::CumSum` 或逐段 `Adds` |
| `PreProcess` 第 650–658 行 | `for moeRankId: expertTokenCount_ = N or N+1` | 逐元素标量写值 | `AscendC::Duplicate`（常量填充）+ 尾部修补 |
| `PreProcess` 第 671–706 行 | `for topkId: GetIds + isTokenSend + 计数` | 标量遍历路由表 + 分支判断 | 无法整体向量化（GetIds 有随机访问），但可部分用向量计数加速 |
| `PreProcess` 第 751–773 行 | `for col: expertTokenCount_ / tokenOffsetOnSend_ / expertTokenCountOnSend_` | 逐元素标量赋值 | `AscendC::Adds`（常量加）或 `Adds` + `Select` 组合 |

**典型优化思路**（以第 650–658 行为例）：

```cpp
// 当前：标量循环
for (uint32_t moeRankId = 0; moeRankId < moeNum_; ++moeRankId) {
    if (moeRankId < remainder)
        expertTokenCountTensor_(offset) = base + 1;
    else
        expertTokenCountTensor_(offset) = base;
}

// 向量化：一次 Duplicate 填 base，前 remainder 个元素用 SetValue 覆盖
Duplicate(expertTokenCountTensor_, base, moeNum_);
for (uint32_t i = 0; i < remainder; ++i)
    expertTokenCountTensor_(offset + i) = base + 1;
```

**注意**：向量化不是银弹。像第 671–706 行的主循环统计涉及 `GetIds`（随机访问路由表）和条件分支，很难整体向量化——这部分应重点优化 GM 读写（见 §6 上半部分），而不是强制替换循环。

**总体评估**：PreProcess 的标量循环优化潜力约 **2-3 倍**（省掉标量流水线的指令发射和分支开销），但收益远不如 §6 上半部分的前缀和 GM 读写优化（那是 **15 倍**级别）。建议先做 GM 读写优化，标量循环优化作为次要方向跟进。

---

## 7. Expert Token Offset 的批量 flush：16 个一批是否最优？

### 现状

`expertTokenOffsetPerExpert_ = UB_ALIGN / sizeof(uint16_t) = 16`（dispatch_send.h 第 351 行）。`PushExpertTokenOffset` 攒满 16 个 `uint16`（32 字节）后 `Flush` 一次 `DataCopyPad`。

### 分析

- 「16 个一批」的由来是 **32 字节 UB 对齐**（16×2B=32B），对齐是对的。
- 但**批量大小是每专家维度的**：`expertTokenOffsetTensor_(16*expertId + num)`。每个专家单独攒 16 个。若某专家只收到 1~2 个 token，则到 `DispatchSend` 结束才由第 620–623 行的收尾 `Flush` 统一刷，**flush 粒度不均**，小专家走「攒不满 → 收尾强制刷」的路径。
- 写目标 `dstTokenAddrOffsetGMTensor[expertTokenOffset]` 是**每专家独立地址**，16 个 uint16 是连续写，已经不错；进一步可考虑**跨专家合并**：不同专家在同一 Attention 窗口内的 `tokenAddrOffset` 区是连续排列的，可以把多个专家的待刷数据拼成一段更大连续写。

### 建议

- 低优先级。16 是「32B 对齐」的自然选择，改动收益有限。
- 若要优化，方向是**把 flush 从「每专家 16 个」升级为「跨专家连续段」**：由于 `tokenAddrOffset` 区按专家顺序连续排布，可累积「一整段专家区间」的偏移一次性 `DataCopyPad`，DMA 更聚合。代价是 staging 结构要重排。

---

## 8. 路由表缓存：64KB 页的命中率

### 现状

`GetIds`（dispatch_send.h 第 74–86 行、combine_recv.h 第 81–93 行）：

```cpp
uint32_t idsPageIndex = idsIndex / IDS_PAGE_ELEMENT_NUM;   // IDS_PAGE_ELEMENT_NUM = 16384
if (forceLoad || idsPageIndex != idsPageIndex_) {
    DataCopy(idsPageTensor_, idsGMTensor_[IDS_PAGE_ELEMENT_NUM * idsPageIndex], IDS_PAGE_ELEMENT_NUM);
    SyncFunc<HardEvent::MTE2_S>();
    idsPageIndex_ = idsPageIndex;
}
return idsPageTensor_(idsIndex % IDS_PAGE_ELEMENT_NUM);
```

- 每页 64KB = 16384 个 int32 路由 id。
- `expertIds` 的形状是 `[batchSize, topk]`，**按行优先**，即 `idsIndex = tokenId * topk + k`。所以遍历顺序是「token 0 的 topk 个 → token 1 的 topk 个 → ...」。

### 命中率分析

- **dispatch_send 的遍历是顺序的**（`for topkId in [start*topk, end*topk)`，第 454 行），跨页只会发生在 `tokenId` 越过 16384/topk 的边界。由于顺序遍历，**每页只 miss 一次**，命中率极高（接近 100% 减去 page 切换）。
- 但 **PreProcess 会先 `GetIds` 全量扫一遍路由表**（第 671–706 行），然后 DispatchSend 又扫一遍（第 454 行）。两遍之间 `idsPageIndex_` 已经指向最后访问的页，**DispatchSend 从头开始遍历时会先 miss 回第一页**（第 456 行首次 `GetIds`）。这是一次性小 miss，可忽略。
- **combine_recv 的遍历也是顺序的**，同样高命中。

### 结论与建议

- 64KB 页对这个「顺序遍历」模式**已经足够好**，无需 LRU 或多级缓存。
- 唯一浪费：PreProcess 与 DispatchSend **各扫一遍路由表**（两次 `GetIds` 全量遍历）。可以把 PreProcess 扫路由时的结果（每个 token 的 `moeRankId/expertId`）缓存下来，DispatchSend 直接复用，省掉第二遍路由表读。代价是缓存 `batchSize×topk` 个 int32（若 batch=8192、topk=8 约 256KB），需放共享内存或分块。
- 若未来 batch 更大，`IDS_PAGE_ELEMENT_NUM`（16384）相对 `batchSize×topk` 仍是一页内多 token，顺序遍历下命中率不受影响。

---

## 9. 动态量化的额外开销

### 现状（dispatch_send.h 第 843–885 行）

per-token 量化链：

```
Cast(float) → Abs → ReduceMax → GetValue(0)读标量 → Muls(×scale) → Cast(rint) → Cast(round via half) → Cast(trunc → int8)
```

### 问题

1. **`ReduceMax` + `GetValue(0)` 是硬串行点**：`ReduceMax` 归约 `hiddenSize` 个元素到 1 个标量，然后 `GetValue(0)` 把标量从 UB 读回标量寄存器，紧接着 `Muls` 依赖这个 scale。这里 `V_S → S_V`（第 856、859 行）两次同步把向量流水完全打断。量化路径因此比非量化多一段无法隐藏的延迟。

2. **量化与其他计算无重叠**：量化发生在 `topkId % topk_ == 0` 分支（第 458–477 行），即每个 token 的第一个 topk 处，与 token 搬运是**串行**的（`MTE2_V → QuantProcess → 之后才发共享专家`）。没有把「token i 的量化」与「token i-1 的搬运」重叠。

3. **经 half 中转**：`Cast(int32, CAST_RINT)` → `Cast(half, CAST_ROUND)` → `Cast(int8, CAST_TRUNC)` 三步 Cast 是为了「rint 到 int8」的舍入语义。中间多一次 half 转换，可评估是否能一步 `Cast(int8, CAST_RINT)`（需确认 AscendC 支持的舍入模式）。

4. **`ReduceMax` 用 `GetValue(0)` 逐 token 读标量**：若 scale 计算能批量（多个 token 一起算 max），可减少标量回读次数。

### 建议

- **多 token 批量量化**：把 `QuantProcess` 改成对「一批 token」做 `Abs/ReduceMax/Muls/Cast`（向量化，减少 per-token 标量同步次数），scale 数组批量 `GetValue`。
- **量化与搬运软件流水**（配合 §2 的双缓冲）：token i 读入后立刻量化，量化的同时 token i+1 的 MTE2 读并行。
- **评估去掉 half 中转**，或确认 `Cast(int8, CAST_RINT)` 直接可用。
- 若量化后 FFN 侧还要反量化，评估「发送端量化 + 接收端反量化」的端到端精度是否值得这份开销（可能量化省带宽，但换来算子内更多同步）。

---

## 10. 共享内存窗口大小：winSizePerAttn_ 是否过大/过小？

### 现状

`winSizePerAttn_` 的 token 区按 **`maxBatchSize_`**（= `maxSeqLen/tpSize`）预留，而不是实际 `batchSize_`（dispatch_send.h 第 233–239 行）。窗口布局见 walkthrough §3.1。

### 问题

1. **按最大 batch 预留 → 大多数时间浪费**：token 区 = `maxBatchSize × hiddenSize × sizeof(type)`。若 maxSeqLen=8192、hiddenSize=7168、BF16，单 token 区 ≈ 112MB。但实际 batch 往往远小于 maxSeqLen，**静态预留了用不满的空间**。这是「窗口过大」的直接后果。

2. **`tokenAddrWinSize = maxBatchSize × (topk+1) × uint16`**：`(topk+1)` 里 +1 是给共享专家的偏移留位（walkthrough §4.4.2）。这块也按 maxBatchSize 预留。

3. **窗口上限受 `GetMaxWindowSize()`（默认 408MB）约束**：host 端 `OP_TILING_CHECK(sharedMemSize < sharedMemAttnNeedSize)`（dispatch_send host 第 168 行）。`combineTokenSize` 注释写「1008MB 最大」（第 165 行），但 `GetMaxWindowSize` 默认 408MB，**当 maxSeqLen 大时 combineTokenSize 会超 408MB 导致 tiling 失败**——这意味着 maxSeqLen 设大时，要么靠 `LCCL_BUFFER_SIZE` 调大窗口，要么算子直接不支持。这是「窗口大小」最需要处理的地方。

4. **combine 方向**：`combineTokenSize` 按 `hiddenSize × batchSize × (topk+1)` 预留（dispatch_send host 第 165 行、combine_recv host 第 166 行），同样按 batch 而非 maxBatch，但 batch 是实际值，相对更紧。

### 建议

- **评估改成按实际 batchSize 预留窗口**（若 tiling 能在运行时拿到真实 batch，则 `maxBatchSize_` 不必参与布局）。收益是省大量共享内存、支持更大 batch 上限；代价是窗口布局在 batch 变化时要重算（需确认 tiling 缓存/复用策略）。
- **量化模式下 token 区缩小一半**：INT8 的 `hiddenSizeQuant_` ≈ hiddenSize/2 + 32，token 区可对应减半，但当前仍按 `maxBatchSize × hiddenSize` 预留（dispatch_send.h 第 233 行），量化路径没吃到省空间红利。应让 token 区按 `DynamicQuant` 分支用 `hiddenSizeQuant_`。
- **确认 `GetMaxWindowSize` 与 maxSeqLen 的匹配**：要么在 host 端更早地对 `combineTokenSize` 做「超 408MB 即降级」处理，要么文档化 `LCCL_BUFFER_SIZE` 的必须性。

---

## 附：跨切面的几个问题（不属于上面 10 点，但值得记）

1. **常量重复定义**：`UB_ALIGN / CAM_MAX_RANK_SIZE / MAX_AIV_NUM / BATCH_INFO_VAL_NUM / IDS_PAGE_SIZE` 在四个算子 `.h` 里各复制一份（见 walkthrough §2.3），应统一进 `comm_args.h`，避免改一处漏三处。

2. **`uint16_t` 偏移溢出风险**：`tokenOffsetOnSend`/`tokenAddrOffset` 用 `uint16_t`，maxBatchSize > 65535 时溢出（见 §4）。

3. **combine_recv 的 scale 不对称**：路由专家累加乘 scale（combine_recv.h 第 391 行），共享专家累加不乘 scale（第 446 行）。需核对共享专家的量化语义是否一致（walkthrough §7.4 已标记）。

4. **性能埋点只埋在 combine_send**（`timeCycleTensor_`，combine_send.h 第 221–245 行），且只在特定 rank 打印。建议四个算子统一埋点，且把「打印条件」做成可配置（env 开关），否则调优时只能在 8/15 号 rank 上看到数据。

5. **`error_log.h` 里的 `GetBatchSizeFactor` 只被 dispatch_recv 用**（dispatch_recv host 第 166 行），`EP_BALANCE_FACTOR = 1.2`（第 75 行）定义了但未使用——疑似预留的负载均衡系数，可在 §1 分核优化时启用。

---

## 11. 走读确认的重构路线（2026-08-21 与作者讨论结论）

dispatch_send 走读完成后确认的三大方向 + 补充项。**落地顺序：方向 C（架构）→ 方向 A（可读性）→ 方向 B（性能）**——先定架构再改名，避免在将被删除的代码上浪费重命名和向量化的功夫。

### 方向 C（最优先）：共享专家上移 Attention 侧

**动机**：共享专家没有路由——每个 token 都要过它，根本不需要"分发"。当前实现为它付出的代价：

1. **通信量**：每个 token 都要发给共享专家所在 rank（即使该 rank 没有路由命中），combine 再传回。窗口里 `(topk+1)` 的 `+1` 全是共享专家开销，topk=8 时约 **11% 的传输量和窗口空间**。
2. **代码复杂度**：dispatch_send 里最绕的几段全是共享专家引入的——650-658 均分、661-666/447-453 两处重复的定位 while、679-685 边界推进、`expertNumPerMoe_ = 1 + moeExpertNumPerMoe_` 导致的满地 `+1` 偏移。删掉后认知负担估计降一半。
3. **顺带消灭已知隐患**：combine_recv 对路由/共享专家 scale 处理不对称的问题（附录 3）直接消失。

**额外收益**：Attention 卡在等 FFN 返回时本来就有算力空闲，共享专家计算正好填入这个空窗——不只是省通信，还是 overlap。

**影响面（跨层改动，需单独设计评审）**：

- 四个算子全动：dispatch_recv 删 `expandXShared/dynamicScalesShared/epRecvCountShared` 三个输出，combine 两侧同理；
- Python 层：`AFDF2ATransferPayload` 已有 `routed_output/shared_output` 分离字段，async_cam 连接器和 NPU 模型 forward 要跟着改；
- 模型包装层：Attention 侧要构建并加载共享专家权重。有先例可循——AFD `deepseek_v2.py` 的 `compute_gate_on_attention` 已把 gate 和 dense 层放在 Attention 侧构建，共享专家走同一条 wiring 路径，`load_weights` 的按角色过滤加分支即可；
- 代价：Attention 节点多存每层一个小 MLP 的权重（显存可控），多算一次共享 FFN（填入等待空窗，实际代价低）。

### 方向 A：变量命名 + 死代码清理（dispatch_send 为样板，四算子推广）

**命名规律统一成三类前缀**：`total*`（全卡总量）、`*BaseIdx/*Cursor`（写入进度）、`*RegionOffset`（布局偏移）。现状是三类语义全叫 count/offset/num，这是混乱根源。

| 现名 | 语义 | 建议名 |
| --- | --- | --- |
| `expertTokenCountTensor_` | 本卡给每个专家的总命中数 | `totalTokensPerExpert_` |
| `expertTokenCountOnSendTensor_` | 前面 AIV 已给该专家发的数量 | `expertSendBaseIdx_` |
| `tokenOffsetOnSendTensor_` | 该 MoE rank token 区的写入游标（去重后） | `rankTokenWriteCursor_` |
| `expertPreTokenNumTensor_` | 专家区域在卡内 token 区的起始偏移 | `expertRegionStartOffset_` |
| `moeTokenNumTensor_` | 本卡给每个 MoE rank 的 token 总数 | `totalTokensPerRank_` |
| `isTokenSendTensor_` | 当前 token 是否已发给该 rank | `tokenSentToRank_` |
| `aivCntTensor`（PreProcess 局部） | 跨 AIV 协调计数矩阵 | `aivSendCountMatrix` |
| `expertTokenOffsetTensor_` | 位置索引批量 flush 暂存区 | `offsetStagingBuf_` |
| `expertTokenOffsetInfoTensor_` | 暂存区元信息（条数+首偏移） | `offsetStagingMeta_` |

**可直接删除**：

- `xOutFloatTensor_`：仅 `#ifdef DIS_VERIFY` 日志用（第 521 行），量产路径无消费；
- `ubSizeSum_` 之后的 `ubReuseSize_`（dispatch_send 内）：算了、打了日志，但 dispatch_send 内无消费者（dispatch_recv 才用）；
- `sharedExpertNumPerMoe_`（硬编码 1）：方向 C 落地后整个消失；
- `magic` 属性：tiling 存了但 kernel 内未找到消费点，确认后删；
- `currSharedTokenCount / nextSharedTokenCount / currSharedMoeRankId`：方向 C 落地后整组消失。

**同批次修复**：int32/uint32 混用（`expertTokenCountTensor_` 是 int32、其余计数是 uint32，无理由）；误导性注释（第 660 行"统计路由专家"实为定位共享专家起点）；`// 32 B or 64 B` 类只对 expertNum≤2 成立的注释。

### 方向 B：前缀和 scalar → vector（见 §6/§6.1，此处补充主次结论）

- **主**：先做"UB 内算完 + 整列一次写回"（§6 建议，估算 15 倍收益，纯 DMA 结构改动不碰算法）；
- **次**：幸存 scalar 热点向量化（§6.1 表格，2-3 倍收益）——650-658 均分填充、751-773 批量提取是好目标；671-706 主统计循环因 `GetIds` 随机访问 + 分支无法整体向量化；
- 具体展开待后续讨论。

### 补充项（讨论中新增，按值得程度排序）

1. **uint16 偏移溢出**（附录 2 已标）：方向 A 改名时顺手升 uint32 + tiling 校验；
2. **每 token 的 O(moeNum) 标量循环**：清空 isTokenSend（486-488）与刷新游标（613-618）每 token 跑 moeNum 次，moeNum 大时改位图（uint64 装 64 个 rank 标记）；
3. **同步冗余**：成对 `MTE2_S + S_MTE2` 保守同步、PreProcess 两次 SyncAll 能否合一，并入方向 B；
4. **重复代码收敛**：两处相同的共享专家定位 while（方向 C 不落地才需要）；四算子各自复制的常量收敛进 comm_args.h（附录 1）。
5. **SetFlags 每卡 4 次 DMA 可降为 2 次**（2026-08-21 SetFlags 走读结论）：
   - **约束前提（作者确认的硬件机理）**：跨卡链路报文以 **512B** 为单位拆分，报文间不保证到达顺序——所以携带 flag 的写必须与数据写分离，且 flag 写本身要小于单报文（当前 8B，安全）。这是"flag 单独最后写"的真正原因，不是管道顺序。
   - **当前**：batchInfo(占位0) → count → chunk → batchInfo 首元素(flag)，4 次 DMA + 8 次同步；
   - **优化**：count + chunk 物理连续，合一次 DMA；然后**带最终 flag 的 batchInfo 整体**一次发出（5×8B=40B < 512B 单报文，原子性有保证）——batchInfo 不必发两次，共 2 次 DMA。
6. **SetFlags 的 O(moeNum²) 标量前缀和**（第 799–803 行）：每个 rank 的 `moePreTokenNum` 都从 0 重新累加，一个 AIV 处理多个 rank 时重复计算；本质是一次前缀和，进循环前算好一张表即可。并入方向 B 的标量优化。

### 落地顺序

```
方向 C（共享专家上移）→ 架构定型，删掉大块代码，需先出小设计（算子接口/Python 层/精度验证方案）
    ↓
方向 A（改名 + 删死代码 + 类型统一 + uint16 修复）→ 在幸存代码上做
    ↓
方向 B（前缀和 GM 写回 → scalar/vector → 同步精简）→ 纯性能，随时可做
```
