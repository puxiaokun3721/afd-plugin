# CAM Async MoE 通信算子源码精读

> 范围：`cam-aynsc/src/comm_operator/ascend_kernels/` 下四个异步 MoE 通信算子
> `cam_moe_distribute_dispatch_send / dispatch_recv / combine_send / combine_recv`
> 配套基础设施：`utils/op_kernel/op_def.h`、`comm_args.h`、`utils/op_host/error_log.h`
>
> 本文目的：让读者「对着代码看明白」，关键逻辑均标注文件与行号。

---

## 0. 阅读地图

四个算子分两对、方向相反，共享同一套基础设施与共享内存协议：

| 算子 | 运行在哪一侧 | 方向 | 核心文件（kernel 主体都在 `.h` 里） |
| --- | --- | --- | --- |
| `dispatch_send` | Attention rank | Attention → Expert（路由 token） | `.../dispatch_send/op_kernel/cam_moe_distribute_dispatch_send.h` |
| `dispatch_recv` | Expert rank | 收 Attention 发来的 token | `.../dispatch_recv/op_kernel/cam_moe_distribute_dispatch_recv.h` |
| `combine_send` | Expert rank | Expert → Attention（FFN 结果回写） | `.../combine_send/op_kernel/cam_moe_distribute_combine_send.h` |
| `combine_recv` | Attention rank | 收 Expert 回写的 FFN 结果 | `.../combine_recv/op_kernel/cam_moe_distribute_combine_recv.h` |

每个算子的代码组织一致：

```
op_host/cam_moe_distribute_*.cpp     # OpDef 注册 + TilingFunc + InferShape（host 端，图编译期）
op_kernel/cam_moe_distribute_*.cpp   # kernel 入口 extern "C" __global__（GET_COMM_ARGS + TILING_KEY 分派）
op_kernel/cam_moe_distribute_*.h     # 真正的类实现（Init / Process / 私有方法）
op_kernel/cam_moe_distribute_*_tiling.h  # Tiling 数据结构体
```

> 注意：与常见的「kernel 写在 .cpp」不同，这四个算子把 `Init`/`Process` 及所有私有方法写在 `.h` 里，`.cpp` 只留入口函数。这是 AscendC 里常见的「header-only kernel + 入口 .cpp」拆分方式。

---

## 1. 算子在 AFD 架构中的位置

**一句话定位**：这四个算子是 AFD `CAMAsyncAFDConnector`（异步 NPU 通信连接器）在 NPU 上实现 MoE 专家并行（EP）路由的底层原语，用**共享内存（peer GM）+ 自旋等待 flag** 替代 NCCL 的 AllToAll/AllGather，负责把每个 token 按路由结果搬到它该去的专家卡、再把专家算完的结果搬回来。

数据流（一次 MoE 前向，跨 Attention/Expert 两组 rank）：

```
Attention rank（每 token 产生 topk 个专家路由）
  │
  │ ① dispatch_send：把 hidden state 按路由写到 Expert rank 的共享内存窗口
  │    （同一 token 若命中同一张 MoE 卡的多个专家，只搬一次完整 hidden state）
  ▼
Expert rank 的共享内存（peer GM）
  │ ② dispatch_recv：Expert rank 轮询 flag，就绪后把 token 搬进本地 expandX/expandXShared
  │
  ▼
Expert 本地 FFN 计算（不在本算子内，是 GEMM 算子）
  │
  │ ③ combine_send：把 FFN 结果按 token 位置写回 Attention rank 的共享内存窗口
  ▼
Attention rank 的共享内存（peer GM）
  │ ④ combine_recv：Attention rank 轮询 flag，读出各专家结果 × scale 后累加成 [batch, hidden]
  ▼
Attention 侧 FFN 输出
```

对应 Python 层调用链（见 `CAM_ASYNC_OPERATOR_OPENSOURCE_PLAN.md`）：

```
torch.ops.umdk_cam_op_lib.async_dispatch_send / async_dispatch_recv
        / async_combine_send / async_combine_recv
  ↓ (pybind → TORCH_LIBRARY 注册，本仓库为源码编译目标)
AscendC kernel（本文四个算子）
  ↓ 通过 Xccl::CommArgs.peerMems 做 IPC 共享内存通信
CAM 通信层（.run 安装器提供的底层通信原语，如 GetWindAddrByRankId 读到的 peer 地址）
```

---

## 2. 共用基础设施

### 2.1 `op_def.h` —— `GET_COMM_ARGS` 宏

`utils/op_kernel/op_def.h`（全文 9 行）：

```cpp
#define GET_COMM_ARGS \
    GlobalTensor<int> commArgsGm; \
    commArgsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(commArgs), 6); \
    int rank = commArgsGm.GetValue(0); \
    int localRank = commArgsGm.GetValue(1); \
    int rankSize = commArgsGm.GetValue(2); \
    int localRankSize = commArgsGm.GetValue(3); \
    uint32_t extraFlag = commArgsGm.GetValue(4); \
    int testFlag = commArgsGm.GetValue(5);
```

作用：把 `commArgs` 这块 GM 当 6 个 `int` 读出来：`rank / localRank / rankSize / localRankSize / extraFlag / testFlag`。四个算子的入口 `.cpp` 都在 `GET_COMM_ARGS` 之后才 `GET_TILING_DATA_WITH_STRUCT`（见各 `.cpp` 第 27 行左右）。

### 2.2 `comm_args.h` —— `Xccl::CommArgs` 与关键常量

`utils/op_kernel/comm_args.h`：

- 常量（第 18–48 行）：
  - `CAM_MAX_RANK_SIZE = 384`（通信库最大支持 NPU 卡数，peerMems 数组长度）
  - `IPC_BUFF_MAX_SIZE = 100 * 1024 * 1024`（100MB，每个 peer 缓冲区上限）
  - `IPC_DATA_OFFSET = 2 * 1024 * 1024`（前 2MB 用作 flag，之后 100MB 存数据）
  - `UB_SINGLE_DMA_SIZE_MAX = 190 * 1024`（单次 DMA 上限 190KB）
  - `UB_ALIGN_SIZE = 32`（UB 对齐；注意算子内部另定义了同义的 `UB_ALIGN`，见 §2.3）
  - `MAX_CORE_NUM = 48`（AIV 最大核数，与各算子 `MAX_AIV_NUM` 一致）
- `struct CommArgs`（第 58–82 行）关键字段：
  - `rank / localRank / rankSize / localRankSize / extraFlag / testFlag`（拓扑信息）
  - `GM_ADDR peerMems[CAM_MAX_RANK_SIZE]`：**核心**，每张卡的共享内存首地址，IPC 通信全靠它
  - `sendCountMatrix / sendCounts / sdispls / recvCounts / rdispls`（all2allvc/all2allv 用，这四个算子未直接使用）
  - `batchSize / hiddenSize / topk / sharedExpertRankNum / expertNumPerRank`（MoE 形状参数）

kernel 里获取 peer 地址的统一写法（四算子相同）：

```cpp
camMemsAddrGMTensor_.SetGlobalBuffer(&((__gm__ Xccl::CommArgs *)commArgs)->peerMems[0], CAM_MAX_RANK_SIZE);
GM_ADDR GetWindAddrByRankId(const int32_t rankId) {
    return (GM_ADDR)camMemsAddrGMTensor_.GetValue(rankId);
}
```

（例：dispatch_send.h 第 248 行 + 第 69–72 行）

### 2.3 各算子头文件里重复定义的常量

四个 `.h` 各自 `namespace MoeDistribute...Impl` 里重复定义（而非集中到 comm_args.h），包括：
`UB_ALIGN = 32`、`CAM_MAX_RANK_SIZE = 384`、`MAX_AIV_NUM = 48`、`BATCH_INFO_VAL_NUM = 5`、`IDS_PAGE_SIZE = 64 * 1024`、`IDS_PAGE_ELEMENT_NUM = 16384`。

dispatch_send.h 第 33–39 行是典型例子。**这份「各算子各自复制一份常量」的写法是潜在维护隐患**（优化笔记里会再提）。

### 2.4 `error_log.h` —— Tiling 侧检查 + 窗口大小

`utils/op_host/error_log.h`：

- `OP_TILING_CHECK(cond, log_func, expr)`（第 43–49 行）：host 端参数校验宏，`if (cond) { log_func; expr; }`。
- `GetMaxWindowSize()`（第 67–87 行）：返回共享内存窗口上限，默认 `DEFAULT_BUFFER_SIZE = 2*(200+4) = 408MB`，可用环境变量 `LCCL_BUFFER_SIZE` 覆盖（单位 MB，上限 32GB）。
- `GetBatchSizeFactor()`（第 89–102 行）：`BATCH_SIZE_FACTOR` 环境变量，默认 1.0，用于 dispatch_recv 侧 `maxTokenNum` 的放缩（见 §5.1）。

四个 host 端 `.cpp` 都用 `GetMaxWindowSize()` 拿到 `sharedMemSize` 后做 `OP_TILING_CHECK(sharedMemSize < needSize, ...)` 校验（如 dispatch_send host 第 168–171 行）。

### 2.5 Tiling 结构体（四个算子各一份，字段有差异）

任务描述里给的「四个算子共有同一个 Info 结构体」与源码有出入，实际是**四个算子各自的 Info 字段略有不同**：

| 字段 | dispatch_send | dispatch_recv | combine_send | combine_recv |
| --- | --- | --- | --- | --- |
| `magic` | ✓ | ✓ | ✓ | ✓ |
| `maxBs` | ✓（= maxSeqLen/tpSize） | ✗ | ✗ | ✗ |
| `bs / h / k` | ✓ | ✓ | ✓ | ✓ |
| `expertRankSize / attentionRankSize / expertPerRank` | ✓ | ✓ | ✓ | ✓ |
| `rank / worldSize` | ✓ | ✓ | ✓ | ✓ |
| `tpSize` | ✓ | ✓ | ✓ | ✗（combine_recv 无） |
| `aivNum / totalUbSize / totalWinSize` | ✓ | ✓ | ✓ | ✓ |
| `layerIndex` | ✓ | ✗ | ✗ | ✗ |
| `dynamicQuant` | ✓ | ✓ | ✗ | ✗ |
| `maxTokenNum` | ✗ | ✓ | ✗ | ✗ |

对应文件：
- `dispatch_send_tiling.h` 第 17–34 行（含 `maxBs`、`layerIndex`、`dynamicQuant`）
- `dispatch_recv_tiling.h` 第 17–33 行（含 `maxTokenNum`、`dynamicQuant`，无 `maxBs/layerIndex`）
- `combine_send_tiling.h` 第 17–31 行（含 `tpSize`）
- `combine_recv_tiling.h` 第 17–30 行（无 `tpSize`、无 `dynamicQuant`）

外层 `*TilingData` 都是 `Mc2InitTiling + Mc2CcTiling + Mc2CcTiling + *Info` 四段（`Mc2*` 来自 `kernel_tiling.h`，供 CAM 通信层初始化用，四个算子自身只消费 `*Info`）。

### 2.6 UB 内存分配：`TBuf::GetWithOffset<T>` 用法详解

四个算子的 `Init` 函数里都有一大段"声明 bufSize → 分配 tensor → 推进 bufOffset"的代码，这是 AscendC 中 **UB（Unified Buffer）手动内存管理**的标准模式。理解这段代码是读懂 Init 的前提。

#### 2.6.1 核心约束：每个 AIV 核只有 192KB UB

`comm_args.h` 第 47 行定义了：

```cpp
constexpr static int64_t UB_SINGLE_TOTAL_SIZE_MAX = 192 * 1024; // 192KB
```

910C 每个 AIV 核的 Unified Buffer 就这么大。Init 里的全部 buffer 分配加起来**不能超过这个值**（dispatch_send.h 第 362 行做了检查：`if (bufOffset > ubSizeSum_)`）。

#### 2.6.2 `GetWithOffset<T>(elementCount, offset)` 的参数语义

```cpp
LocalTensor<T> ubBuffer_.GetWithOffset(size_t elementCount, uint64_t& offset);
//                                 ↑ 元素个数（不是字节数）    ↑ 字节偏移（in-out 参数，自动推进）
```

- **模板参数 `<T>`**：决定返回的 `LocalTensor<T>` 类型，也决定了后续代码按什么类型访问这块内存。UB 本身是裸字节，`<T>` 只是"类型视角"——同一段字节用 `<int32_t>` 访问按 4B 索引，用 `<uint16_t>` 访问按 2B 索引。
- **第一个参数 `elementCount`**：要分配的**元素个数**，不是字节数。
- **第二个参数 `offset`**：in-out 参数，传入当前字节偏移，返回后自动推进 `elementCount × sizeof(T)` 字节。

#### 2.6.3 标准分配三步曲

每个 buffer 的分配都遵循同一个模式（以 dispatch_send.h 第 272–275 行为例）：

```cpp
// ① 算出字节数（可能 MathCeil 对齐）
bufSize = MathCeil(IDS_PAGE_SIZE, UB_ALIGN);       // 64KB，字节

// ② 字节→元素个数，传给 GetWithOffset
idsPageTensor_ = ubBuffer_.GetWithOffset<int32_t>(bufSize / sizeof(int32_t), bufOffset);
//                                              ↑ 64KB / 4B = 16384 个元素

// ③ offset 按字节推进
bufOffset += bufSize;                               // +64KB
```

**为什么需要 "字节数 → 除以 sizeof(T) → 元素个数" 这个转换？** 因为 `GetWithOffset` 要元素个数，但对齐必须在字节维度做（`MathCeil` 按 32B 对齐），两个单位不同，所以每行都要做一次除法。

**为什么不能直接传 `N`（元素数）给 `MathCeil`？** 看 dispatch_send.h 第 279 行：

```cpp
bufSize = MathCeil(sizeof(int32_t) * expertNum_, UB_ALIGN);   // 先乘 sizeof → 字节 → 字节对齐
expertTokenCountTensor_ = ubBuffer_.GetWithOffset<int32_t>(bufSize / sizeof(int32_t), bufOffset);
//                                              ↑ 再除回来 → 元素个数（含 padding）
```

如果直接 `MathCeil(expertNum_, 32)`，变成按**元素**对齐（9 个专家 → 32 个元素 = 128B），比按字节对齐（9×4B=36B → 64B）多浪费一倍。所以正确做法是**字节维度对齐再换算回元素个数**。

#### 2.6.4 UB 内存布局图（以 dispatch_send 为例）

Init 函数从 `bufOffset=0` 开始，顺序分配所有 buffer，最终布局：

```
UB 总空间 (192KB)
│
├─ [0,         64KB)     idsPageTensor_          int32[16384]     路由表分页缓存
├─ [64KB,      64KB+N)   expertTokenCountTensor_ int32[N]         每个专家收多少 token
├─ [+N,        +small)   isTokenSendTensor_      uint8[expertNum] 每个 MoE rank 是否已发
├─ [+small,    +small)   tokenOffsetOnSend...    uint16[moeNum]   每个 MoE rank 的发送偏移
├─ [+small,    +small)   expertTokenCountOnSend  uint16[expertNum]
│
├─ ┌── PreProcess 临时区 ──────────────────────────────┐
│  │ [+off,     +big)     aivCntTensor  uint32[(moeNum+expertNum)*aivNum]  跨 AIV 前缀和  │
│  └────────────────────────────────────────────────────┘
│  (PreProcess 返回后，这段 UB 不再访问，等效于"释放")
│
├─ [+off,      +small)   moeTokenNumTensor_      uint32[moeNum]
├─ [+off,      +small)   expertPreTokenNumTensor uint32[expertNum]
├─ [+off,      40B)      tokenNumAttn...Tensor_  int64[5]         batch info
│
├─ ┌── 数据搬运区 ──────────────────────────────────────┐
│  │ 非量化: [+off, +off+2*hs) xTensor_ / xOutTensor_ 同一偏移，复用同一块 UB  │
│  │ 量化:   [+off, +hs)       xTensor_ / xOutFloat   (FP16→float)           │
│  │         [+off, +hs)       quantFloatTensor_ / quantFloatAbsTensor_      │
│  └────────────────────────────────────────────────────┘
│
├─ [+off,      +small)   expertTokenOffset...    uint16[16*expertNum]  批量 flush 缓冲
├─ [+off,      +small)   expertTokenOffsetInfo   uint32[2*expertNum]
│
└─ [bufOffset, 192KB)    ubReuseSize_            剩余可复用空间
```

#### 2.6.5 UB 空间复用手法

三个层面的复用：

**(A) 显式同 offset 复用**（dispatch_send.h 第 342–345 行）：

```cpp
// 非量化路径：xTensor_ 和 xOutTensor_ 分配在同一个 bufOffset
xTensor_   = ubBuffer_.GetWithOffset<XType>(bufSize / sizeof(XType), bufOffset);
xOutTensor_ = ubBuffer_.GetWithOffset<ExpandXOutType>(bufSize / sizeof(ExpandXOutType), bufOffset);
//          ↑ 两个 tensor 指向同一物理地址，类型不同
```

这完全合法：`DataCopy GM→UB` 写完 xTensor_ 后，量化/转写结果写到 xOutTensor_ 时，xTensor_ 的原始数据已不再需要。**非量化路径 XType==ExpandXOutType，是同一类型同一地址；量化路径 XType=bfloat16_t、ExpandXOutType=int8_t，是不同类型同一地址（内存重用）**。

**(B) PreProcess 临时区复用**：PreProcess 内部调用 `ubBuffer_.GetWithOffset` 分配了巨大的 `aivCntTensor`（前缀和用），PreProcess 返回后这块 UB 不再被访问。后续 buffer 的分配从 PreProcess 之前的 bufOffset 之后开始（因为 offset 是累计推进的），但 PreProcess 区域在物理上被后续的"必要 buffer"之后的空间"覆盖"了——等效于栈式分配的"函数返回后自动释放"。

**(C) ubReuseSize_ 兜底**（dispatch_recv.h 第 254–258 行）：

```cpp
ubReuseSize_ = ubSizeSum_ - bufOffset;   // 所有"必要" buffer 分完后剩余
tokenAddrOffsetTensor_ = ubBuffer_.GetWithOffset<uint16_t>(bufSize / sizeof(uint16_t), bufOffset);
//            ↑ 把剩余空间全给 tokenAddrOffset，按需使用，不浪费
```

这是"最后一大块不固定大小"的 buffer 的标准分配方式：先算完所有固定大小的，再把剩余全给最大的动态 buffer。

#### 2.6.6 为什么要这么精细地手动管理 UB

- **UB 只有 192KB**，不像 GPU 有几 GB 显存可以随便 malloc
- 910C 没有 UB 内存分配器/释放器，AscendC 只提供 `TBuf`（buffer 管理器）和 `GetWithOffset`（基于偏移的分配），需要程序员自己规划
- **不复用 = 装不下**：dispatch_send 的"必要" buffer 加起来经常超过 192KB，尤其 DynamicQuant 路径多了 quantFloat (28KB) + quantFloatAbs (28KB)，必须和 token 数据区复用
- **优化 UB 布局 = 支撑更大 batch/更多核**：xTensor_ 和 token 区大小正比于 batchSize × hiddenSize，UB 紧凑程度直接决定算子能支持的最大形状

---

## 3. 共享内存协议（四个算子的「协议层」）

这是理解整套代码的地基，先讲清楚再读算子。

### 3.1 每个 rank 的共享内存是一块大 GM，被切成 `winSizePerAttn_` 大小的小窗口

关键概念：`winSizePerAttn_` 是**每个 Attention rank 占据的子窗口大小**，任何一张卡的共享内存里都按 `winSizePerAttn_ * attnRankId` 切出子窗口，**子窗口下标 = Attention rank id**（不是本卡自己的 rank id）。

`winSizePerAttn_` 的计算在 dispatch_send.h 第 228–239 行、dispatch_recv.h 第 162–174 行，**两者算的是同一套 MoE 侧窗口布局**：

```
batchInfoWinSize          = ceil(sizeof(int64)*5, 32)          // 5 个 int64 → 40B → 对齐 64B
expertTokenCountWinSize   = ceil(sizeof(int32)*expertNumPerMoe, 32)
expertTokenCountChunkWinSize = ceil(sizeof(int32)*expertNumPerMoe, 32)
tokenWinSize              = maxBatchSize * hiddenSize * sizeof(type)   // 或量化后
tokenAddrWinSize          = maxBatchSize * (topk+1) * sizeof(uint16)
totalWinSize              = 上述之和
winSizePerAttn_           = align32(totalWinSize)
```

> 注意 `tokenWinSize` 用的是 `maxBatchSize_`（= `maxSeqLen/tpSize`，来自 tiling 的 `maxBs`），而不是实际 `batchSize_`。所以窗口大小按**最大可能 batch** 预留，这是后文优化笔记「窗口大小」一节的根源。

### 3.2 两块窗口的用途不同

由于 dispatch 方向写的是 **Expert rank 的窗口**、combine 方向写的是 **Attention rank 的窗口**，实际存在**两种窗口布局**：

**(A) Expert rank 的共享内存**（dispatch 方向，dispatch_send 写、dispatch_recv 读）——每个 Attention rank 一个子窗口，偏移 `winSizePerAttn_ * attnRankId`：

```
+0               batchInfo[5]             int64 × 5   tokenNum | attnRankId | layerIndex | moePreTokenNum | flag
+64B             expertTokenCount[]       int32 × expertNumPerMoe   每个专家收多少 token
+...             expertTokenCountChunk[]  int32 × expertNumPerMoe   分 chunk 时记录该专家是否已收（进度位）
+...             token data               maxBatchSize × hiddenSize × type  实际 token
+...             tokenAddrOffsets[]       maxBatchSize × (topk+1) × uint16  每个专家该从 token 区第几个开始取
```

dispatch_send.h 第 421–434 行算出了这些偏移（`tokenNumAttnRankIdLayerIndexOffset / expertTokenCountOffset / expertTokenCountChunkOffset / tokenOffset / tokenAddrOffset`），dispatch_recv.h 第 431–440 行用同一套公式在 Expert 侧读回。

**(B) Attention rank 的共享内存**（combine 方向，combine_send 写、combine_recv 读）——Expert 把 FFN 结果写回 Attention 窗口：

```
+0                       aivCnt 前缀和区    (moeNum + expertNum) × MAX_AIV_NUM × uint32   ← dispatch_send 的 PreProcess 用它
+dispatchOffset_         combine flag       moeNum × uint32                                ← combine_send 写 moePreTokenNum+1
+dispatchOffset_+flag    combine token      hiddenSize × moePreTokenNum × type             ← FFN 结果
```

其中 `dispatchOffset_ = ceil(sizeof(uint32)*(moeNum+expertNum)*MAX_AIV_NUM, 32)`，combine_send.h 第 146 行、combine_recv.h 第 165 行。它正好等于 dispatch_send host 端 tiling 里算的 `dispatchNeedSize`（dispatch_send host 第 161 行），即 dispatch 阶段跨 AIV 前缀和占用的大小。combine 结果的写入偏移公式（combine_send.h 第 292–294 行、第 431–433 行）：

```cpp
offset = ceil(sizeof(uint32)*moeNum_, 32) + sizeof(ExpandXType)*hiddenSize*moePreTokenNum
```

即「跳过 flag 区」+「按前面几张 MoE 卡已回写的 token 数偏移」。

### 3.3 这套协议一句话总结

- **dispatch 方向**：Attention 卡把「batch info + 各专家 token 计数 + token 本体 + token 偏移索引」写进 **Expert 卡**的、以 `attnRankId` 为下标、`winSizePerAttn_` 为步长的子窗口。
- **combine 方向**：Expert 卡把「FFN 结果 + 完成 flag」写进 **Attention 卡**的、以 `attnRankId` 为下标的子窗口（跳过前缀和区）。
- 两次数据搬运**都不走 NCCL**，全部是 `DataCopy` 到对方卡的 peer GM，再靠 flag 自旋等待。

---

## 4. 算子精读 1：`cam_moe_distribute_dispatch_send`（Attention 侧发送）

### 4.1 角色 / I/O / Tiling Key

- **角色**：Attention rank 上，把 hidden states 按 MoE 路由结果发给对应 Expert rank。
- **输入**：`x [batchSize, hiddenSize]`、`expertIds [batchSize, topk]`、`commArgs`。
- **输出**：`expandX [1]`（flag，实际数据都进共享内存，见 host 第 215 行 `SetDim(0,1)`）。
- **Tiling Key**：`100 = BF16`、`101 = FP16`（dispatch_send host 第 53–54 行；入口 `.cpp` 第 32–52 行按 key + `dynamicQuant` 四选一实例化模板）。
- **模板**：`CamMoeDistributeDispatchSend<XType, ExpandXOutType, DynamicQuant>`（dispatch_send.h 第 48–54 行）。
  - 非量化：`<bfloat16_t, bfloat16_t, false>` 或 `<float16_t, float16_t, false>`
  - 量化：`<bfloat16_t, int8_t, true>` 或 `<float16_t, int8_t, true>`

### 4.2 分核策略（Init 第 251–263 行）

按 batch 维均分到 `aivNum_` 个 AIV，余数摊到前几个核：

```cpp
uint32_t batchSizePerAiv = batchSize_ / aivNum_;
uint32_t batchSizePerAivRemain = batchSize_ % aivNum_;
batchSizePerAivStart_ = batchSizePerAiv * aivId_;
if (aivId_ < batchSizePerAivRemain) { batchSizePerAiv += 1; batchSizePerAivStart_ += aivId_; }
else { batchSizePerAivStart_ += batchSizePerAivRemain; }
batchSizePerAivEnd_ = batchSizePerAivStart_ + batchSizePerAiv;
```

每个 AIV 负责 `[batchSizePerAivStart_, batchSizePerAivEnd_)` 范围的 token（对应 topkId 范围 `[start*topk, end*topk)`）。

### 4.3 UB 布局（Init 第 269–370 行）

按 offset 递增依次分配（`ubBuffer_.GetWithOffset`）：

| 序 | 张量 | 大小 | 用途 |
| --- | --- | --- | --- |
| 1 | `idsPageTensor_` | 64KB | 路由表分页缓存（int32） |
| 2 | `expertTokenCountTensor_` | `int32 × expertNum_` | 各专家收到的 token 数 |
| 3 | `isTokenSendTensor_` | `uint8 × moeNum_` | 本 token 是否已给某 MoE 卡发过 |
| 4 | `tokenOffsetOnSendTensor_` | `uint16 × moeNum_` | 本 AIV 给每张 MoE 卡的发送偏移 |
| 5 | `expertTokenCountOnSendTensor_` | `uint16 × expertNum_` | 本 AIV 给每个专家的发送偏移 |
| 6 | `aivCntTensor_` | `uint32 × (moeNum+expertNum) × aivNum` | **PreProcess 临时区**（第 632–633 行分配，用完即弃） |
| 7 | `moeTokenNumTensor_` | `uint32 × moeNum_` | 本 attn 给每张 MoE 卡发多少 token |
| 8 | `expertPreTokenNumTensor_` | `uint32 × expertNum_` | 每个专家范围内前缀和 |
| 9 | `tokenNumAttnRankIdLayerIndexTensor_` | `int64 × 5` | batch info 组装区 |
| 10 | `xTensor_ / xOutTensor_`（+`xOutFloatTensor_`） | `hiddenSize` | token 搬运缓冲 / 量化缓冲 |
| 11 | （量化时）`quantFloatTensor_ / quantFloatAbsTensor_ / quantRowMaxTensor_` | 各 `hiddenSize`/`8` float | 量化中间结果 |
| 12 | `expertTokenOffsetTensor_` | `uint16 × 16 × expertNum_` | 专家 token 位置批量缓存（16 个一批） |
| 13 | `expertTokenOffsetInfoTensor_` | `uint32 × 2 × expertNum_` | 每专家已缓存数 + flush 起点 |

第 362–366 行做 UB 超限检查（`isError_`），第 369 行把剩余 UB 记为 `ubReuseSize_`（dispatch_send 实际没用上复用区）。

### 4.4 Process 主流程

`Process()`（第 373–383 行）只做一件事：`DispatchSend()`。

#### 4.4.1 `DispatchSend()`（第 386–626 行）—— 数据搬运主循环

**① 初始化计数缓冲（第 388–393 行）**：`moeTokenNumTensor_ / expertPreTokenNumTensor_ / expertTokenOffsetTensor_ / expertTokenOffsetInfoTensor_` 全部清零。

**② 按专家粒度算发送偏移（第 396–406 行）**：

```cpp
for expertId in [0, expertNum_):
    moeTokenNumTensor_(expertId / expertNumPerMoe_) += expertTokenCountTensor_(expertId);   // 每张 MoE 卡发多少
    if ((expertId % expertNumPerMoe_) == 0) continue;   // 跳过共享专家（其偏移单独处理）
    expertPreTokenNumTensor_(expertId) = expertPreTokenNumTensor_(expertId-1) + expertTokenCountTensor_(expertId-1);
```

`expertTokenCountTensor_` 在 PreProcess 阶段已经算好（每个专家最终收多少 token），这里只是做卡内前缀和。

**③ 计算共享内存窗口内偏移（第 421–434 行）**：见 §3.2(A)，算出 `tokenOffset`（token 数据起点）和 `tokenAddrOffset`（token 偏移索引起点）。

**④ 主循环（第 454–619 行）**：`for topkId in [topk*start, topk*end)`，每 `topk` 个对应一个 token：

- `tokenId = topkId / topk_`。
- **`topkId % topk_ == 0` 分支（第 456–548 行）= 每个 token 的第一个 topk**：
  - 读 token：非量化 `DataCopy(xTensor_, xGMTensor_[hiddenSize_*tokenId], hiddenSize_)` + `MTE2_S → S_MTE2 → S_MTE3`；量化则 `MTE2_V → V_MTE2` 后 `QuantProcess`（第 477 行，见 §6）。
  - 清空发送表 `isTokenSendTensor_`（第 486–488 行）。
  - 找到该 token 属于哪个共享专家（第 491–499 行）：`currSharedTokenCount` 用共享专家计数逐卡推进，`expertTokenOffset = tokenId - currSharedTokenCount`。
  - **发共享专家**（第 502–511 行）：`dstRankId = attnNum_ + currSharedMoeRankId`，写进 Expert 卡窗口的 token 区，`tokenOffsetOnSend` 为当前偏移。`isTokenSendTensor_(currSharedMoeRankId) = 1`。
  - `PushExpertTokenOffset` 记录位置（第 545 行）。
- **路由专家发送（第 550–618 行）**：
  - `moeExpertId = GetIds(topkId)`（路由表，见 §6.2）。
  - `moeRankId = moeExpertId / moeExpertNumPerMoe_`；`expertId = expertNumPerMoe_*moeRankId + 1 + moeExpertId % moeExpertNumPerMoe_`（全局专家 id）。
  - `preTokenNum = expertPreTokenNumTensor_(expertId)`；`idx = expertTokenCountOnSendTensor_(expertId)`；`expertTokenOffset = preTokenNum + idx`。
  - **去重发送（第 562–572 行）**：`if (isTokenSendTensor_(moeRankId) == 0)` 才真正 `DataCopy` 一次完整 hidden state（同一 token 命中同卡多个专家时只搬一次），随后置 1。
  - `PushExpertTokenOffset(expertId, tokenOffsetOnSend, expertTokenOffset, tokenAddrOffset)`（第 603 行），更新 `expertTokenCountOnSendTensor_(expertId) = idx + 1`（第 609 行）。
- **每个 token 结束（第 611–618 行）**：`(topkId+1) % topk_ == 0` 时，把本轮发过的 MoE 卡的 `tokenOffsetOnSendTensor_` 各 +1。

**⑤ 收尾（第 620–625 行）**：`FlushExpertTokenOffset` 刷完所有剩余，然后 `SetFlags()`。

#### 4.4.2 `PushExpertTokenOffset` / `FlushExpertTokenOffset`（第 154–200 行）—— 映射表的批量写机制

**要解决的问题**：token 本体在 Expert 卡 token 区是**按发送顺序去重存放**的（一个 slot 一个 token，不按专家分区），但同一 token 可能属于多个专家（topk 命中 + 共享）。接收端按专家取数据（"expert 7 的第 0 个 token 在哪个 slot？"），token 区本身回答不了——所以需要一张**映射表**（窗口布局里的 `tokenAddrOffsets` 区）：

```
映射表（uint16 数组，写在对端 Expert 卡窗口的 tokenAddrOffset 区）：
  下标 = 专家累计序号（expertTokenOffset = expertPreTokenNum + idx，按专家分段）
  值   = token 区 slot 号（tokenOffsetOnSend）

例：下标 [2,3] 是 expert 7 的段（preTokenNum=2），值 [0,1] → e7 的两个 token 在 slot 0、1
```

Push/Flush 就是填这张表的过程。逐条写的话每条 2B 一次 DMA，效率极差，所以**在 UB 攒批、满 16 条一次写**。

**涉及的变量**：

```cpp
expertTokenOffsetPerExpert_ = 16;              // 每专家暂存容量 = UB_ALIGN/sizeof(uint16) = 32/2
expertTokenOffsetTensor_    [16 × expertNum]   // 暂存的"值"（slot 号），每专家一段 16 格
expertTokenOffsetInfoTensor_[2 × expertNum]    // 每专家元信息：(2i)=count 已攒条数，(2i+1)=firstIdx 本批起点下标
```

**Push（第 154–174 行）**：
1. 若 `count == 0`（本批第一条），把 `expertTokenOffset` 记为 `firstIdx`（第 158–160 行）；
2. 把值存进 `expertTokenOffsetTensor_(16*expertId + count)`，`count+1`（第 162–164 行）；
3. `count == 16` 时触发 Flush（第 171–173 行）。

**核心不变式（Push 只记 firstIdx 就够的原因）**：同一专家的连续 Push，映射表下标**必然连续 +1**——路由专家 `expertTokenOffset = preTokenNum + idx`，idx 每次命中 +1（第 609 行）；共享专家 `= tokenId - currSharedTokenCount`，本 AIV 的 tokenId 连续递增。所以一批 16 条的下标一定是 `[firstIdx .. firstIdx+15]`，flush 时一次连续写入即可。

**Flush（第 176–200 行）——直接写对端 GM，无中转**：

```cpp
uint32_t dstRankId = attnNum_ + expertId / expertNumPerMoe_;       // 该专家所在 Expert 卡
GM_ADDR dstGM = GetWindAddrByRankId(dstRankId);                     // 对端卡共享内存地址（IPC）
dstTokenAddrOffsetGMTensor.SetGlobalBuffer((__gm__ uint16_t*)(dstGM + tokenAddrOffset));  // 映射表区基址
DataCopyPad(dstTokenAddrOffsetGMTensor[expertTokenOffset],          // GM 目标 = 基址 + firstIdx×2B
    expertTokenOffsetTensor_[16 * expertId],                         // UB 源 = 该专家 staging 段头
    {1U, sizeof(uint16_t) * count, 0U, 0U, 0U});                     // 一次写 count×2B
expertTokenOffsetInfoTensor_(2 * expertId) = 0;                      // 清 count；firstIdx 不用清，下批 Push 时重记
```

注意：**这一次 `DataCopyPad` 就是跨卡写**——`GetWindAddrByRankId` 拿到的是对端 Expert 卡的 peer GM 地址，映射表不经过本卡 GM 中转。空批（`count==0`）直接 return（第 185–187 行），这是为了主循环结尾的兜底 flush（第 621–623 行：逐专家强制刷掉不满 16 条的尾巴）。

**为什么用 `DataCopyPad` 不用 `DataCopy`**：GM 目标地址 `基址 + firstIdx×2B` 只有 2B 对齐、写入量 `count×2B` 不一定是 32B 整数倍，`DataCopyPad` 支持这种非对齐小粒度写。

**完整时序示例**（expert 7，preTokenNum=4，本 AIV 命中 3 次，token 在 slot 0/1/3）：

```
Push(e7, slot=0, offset=4):  count 0→1，firstIdx=4，staging = [0]
Push(e7, slot=1, offset=5):  count 1→2，          staging = [0,1]
Push(e7, slot=3, offset=6):  count 2→3，          staging = [0,1,3]
主循环结束（count=3<16 未自动 flush）→ 兜底 Flush(e7)：
  DataCopyPad 把 [0,1,3]（6B）写到对端映射表下标 4 处
  → 对端 GM：entry[4]=0, entry[5]=1, entry[6]=3
dispatch_recv：由同样的 preTokenNum 推出 e7 段从下标 4 开始，
  读出 [0,1,3] → 去 token 区 slot 0、1、3 取回 3 个 token。
```

#### 4.4.3 `SetFlags()`（第 778–842 行）—— 元信息 + 就绪信号的发布

数据面（token 本体 + 映射表）在主循环里已全部写进对端 GM，SetFlags 负责把**元信息包和就绪信号**发布过去。入口先 `SyncAll<true>()`（第 780 行）：写 flag 的只是个别 AIV，但 flag 语义是"本 attn 卡的**全部**数据就绪"，必须等所有 AIV 的数据面写完才允许发布。

**分工**（第 792 行）：`moeRankId % aivNum_ == aivId_`，每张 Expert 卡的发布恰好由一个 AIV 负责。

对每张负责的 MoE 卡，依次（第 793–839 行）：

1. **算 `moePreTokenNum`**（第 799–803 行）：对 `moeTokenNumTensor_` 做截止到当前 rank 的标量前缀和 =「本 attn 卡发给 rank 0..r-1 的 token 总数」。
2. **写 batchInfo 5 个 int64**：`[0, rankId, layerIndex, moePreTokenNum, 0]`（第 805–813 行）。注意 flag 位（首元素）**先占 0**，[4] 也清 0（供接收端使用，见下）。
3. **写 `expertTokenCount`**（第 819–825 行）：把本 MoE 卡对应的 `expertNumPerMoe_` 个计数**前移压缩到 tensor 开头**再 DMA——接收端的"任务清单"。前移会覆盖 rank0 段数据，但 UB 是 AIV 私有的且 aiv0 首个处理的就是 rank0（自拷贝无害），**正确但脆弱**（改分工顺序会踩坑）。
4. **清零 `expertTokenCountChunk`**（第 828–832 行）：接收端的消费进度位归零。注意这步用 `Duplicate`（V 管道）填零，同步事件是全函数唯一的 `V_MTE3`。
5. **最后写 flag**（第 835–839 行）：`batchInfo[0] = moeTokenNumTensor_(moeRankId) + 1`，单独一次 DMA 只写 1 个 int64。**`+1` 编码**是因为 tokenNum 可能合法为 0（batchSize < moeNum 时），flag 判据是"非零即就绪"，接收端解码 `真实 tokenNum = flag - 1`。

**为什么 flag 必须单独一次 DMA（硬件层原因）**：batchInfo + count + chunk + token 数据加起来远超 **512B**，跨卡链路上会被拆成多个 512B 报文，**报文之间不保证到达顺序**——如果 flag 和数据在同一次大 DMA 里，携带 flag 的报文可能先到，接收端读到非零 flag 时其余数据还在路上。所以就绪信号必须等所有数据 DMA 确认完成（`MTE3_S`）后，作为独立的、小于单报文的写操作最后发出。

> 优化点（已记入优化笔记）：五步共 4 次 DMA。batchInfo 其实不必发两次——count + chunk 物理连续可合一次 DMA，然后带最终 flag 的 batchInfo 整体一次发出（5×8B=40B < 512B 单报文，原子性有保证），4 次 DMA 可降为 2 次。

**batchInfo[0] 与 [4] 的两级 flag 机制**（消费端证据：dispatch_recv.h `WaitFromAttn` 第 762–834 行）：

| 槽位 | 谁写 | 谁读 | 语义 |
| --- | --- | --- | --- |
| `[0]` tokenNum+1 | dispatch_send（attn 卡） | **仅 Expert 卡的 aiv0**（第 790 行） | 卡间 flag："本 attn rank 数据就绪" |
| `[4]` | Expert 卡的 aiv0（第 799–800 行） | **其余 47 个 AIV**（第 818 行） | 卡内广播："我已裁决处理这一组 dp，就是这组" |

为什么要两级：判断"就绪"不是看一个 flag，而是要确认**一整个 dp 组（tpSize 个 attn rank）全部**非零；若 48 个 AIV 各自做全量检查，不仅轮询流量 ×48，更严重的是**多组同时就绪时各 AIV 可能选中不同的组**（决策竞争）。所以 aiv0 做唯一裁决：轮询 [0] 确认整组就绪 → 往被选中组的 0 号卡窗口 [4] 写非零 → 其他 AIV 扫 [4]，扫到哪个窗口非零就跟进哪组（[4] 同时携带"就绪"+"是哪组"两个信息）。SetFlags 第 2 步把 [4] 清零，就是发送端替接收端复位这个广播位。

**`moePreTokenNum` 是 combine 方向的"回信地址"**（证据链：dispatch_recv.h 第 609 行存入 `tpBatchInfoTensor_` → 随 `expertTokenNums` 输出 → combine_send.h 第 291–293 行消费）：

```
Attention 卡的 combine 接收区按 MoE rank 顺序连续排列：
[flag区][ moe0 的结果 ][ moe1 的结果 ]...
combine_send 写回偏移 = ceil(uint32×moeNum,32) + hiddenSize × moePreTokenNum × sizeof(type)
```

Expert rank r 写回 FFN 结果时必须知道自己的区段起点 = attn 卡发给 rank 0..r-1 的 token 总数。**只有 Attention 卡知道这个全局分布**（每张 Expert 卡只知道自己收了多少，Expert 卡之间无通信），所以由发送端在 dispatch 时捎带过去，省掉一轮 Expert 间的 allgather。

### 4.5 PreProcess —— 跨 AIV 协调（第 629–773 行）

这是本算子最复杂的一段，单独放 §7 讲。先记结论：它在 `Init` 里被调用一次（第 299 行），产出三个偏移量 `expertTokenCountTensor_ / expertTokenCountOnSendTensor_ / tokenOffsetOnSendTensor_`，供 DispatchSend 主循环使用。

---

## 5. 算子精读 2：`cam_moe_distribute_dispatch_recv`（Expert 侧接收）

### 5.1 角色 / I/O / 分核

- **角色**：Expert rank 上，从共享内存接收 Attention 发来的 token。
- **输入**：`x [1]`（flag）、`commArgs`。
- **输出**（host 第 49–55 行 / InferShape 第 206–264 行）：
  - `expandX [maxTokenNum, hiddenSize]`（路由专家 token）
  - `expandXShared [maxSharedTokenNum, hiddenSize]`（共享专家 token）
  - `dynamicScales [maxTokenNum]` / `dynamicScalesShared [maxSharedTokenNum]`（量化 scale，非量化时 shape=1）
  - `expertTokenNums [batchInfoNum]`（`INFO_NUM + tpSize + (expertPerRank+1)*tpSize`，回传给 combine_send 用）
  - `epRecvCount [expertPerRank]` / `epRecvCountShared [1]`
- **分核**：`DispatchRecv` 里按 **token 维度**均分（`tokenNumPerAiv`，见 DispatchRecvShared 第 465–475 行、DispatchRecvRouted 第 630–639 行），不是按 batch。因为 Expert 侧收到的 token 数由路由决定，不固定。
- **Tiling 特殊点**：`maxTokenNum = MAX_BATCH_SIZE * batchSizeFactor`（host 第 166–167 行，`MAX_BATCH_SIZE = 1024*256`，`batchSizeFactor` 来自环境变量 `BATCH_SIZE_FACTOR`，默认 1.0）。`maxTokenNum_` 用于 `expandX` 输出尺寸和分 chunk 上限。

### 5.2 UB 布局（Init 第 196–260 行）

| 序 | 张量 | 用途 |
| --- | --- | --- |
| 1 | `tokenNumAttnRankIdLayerIndexTensor_` | int64 × 5，batch info 读入区 |
| 2 | `tpBatchInfoTensor_` | `int64 × (INFO_NUM + tpSize + expertNumPerMoe*tpSize)`，组装回传 combine_send 的信息 |
| 3 | `expertTokenCountOutTensor_` | 输出 recvCounts 组装区 |
| 4 | `expertTokenCountTensor_` | int32 × expertNumPerMoe |
| 5 | `expertTokenCountChunkTensor_` | int32 × expertNumPerMoe，分 chunk 进度 |
| 6 | `xOutTensor_`（+`xOutFloatTensor_`） | token 搬运缓冲 |
| 7 | `tpExpertTokenCountTensor_` | uint32 × 2×expertNumPerMoe，跨 TP 累计 |
| 8 | `tokenAddrOffsetTensor_` | uint16 × `ubReuseSize_`，**吃满剩余 UB** 当 token 偏移索引的 staging |

第 257–259 行：`tokenAddrOffsetTensor_` 直接用 `ubReuseSize_`（剩余 UB），越大越好，用来一次 `DataCopyPad` 读入一批 token 偏移。

### 5.3 Process 主流程

`Process()`（第 262–272 行）→ `DispatchRecv()`（第 275–422 行）：

**① `WaitFromAttn()`（第 277 行，详见 §8）**：轮询等 Attention 就绪，返回就绪的 `tpAttnRankId`。

**② 汇总 TP 内各专家 token 数（第 283–307 行）**：对 `tpSize_` 个 Attention rank，逐个读 `expertTokenCount` 并累加进 `tpExpertTokenCountTensor_`（前 `expertNumPerMoe_` 个元素存「本轮待收」，后 `expertNumPerMoe_` 个存「已收进度」，第 299–301 行）。

**③ 读 chunk 进度（第 309–317 行）**：读 `expertTokenCountChunk`（上一轮已收过的专家标记）。

**④ 算本轮处理的专家区间 `[startExpert, endExpert]`（第 319–334 行）**：

```cpp
for i in [1, expertNumPerMoe_):
    if (expertTokenCountChunkTensor_(i) == 0) {          // 未收过的专家
        if (startExpert == 0) startExpert = i;
        if (tokenCnt + tpExpertTokenCountTensor_(i) <= maxTokenNum_) { tokenCnt += ...; endExpert = i; }
        else break;                                       // 超出 maxTokenNum_ 就断，留给下一轮
    }
```

这就是**分 chunk（multi-round）**机制：`expandX` 只有 `maxTokenNum_` 行，一次收不下所有专家时就按 `maxTokenNum_` 截断，剩下专家下一轮再收。`SyncAll<true>()`（第 335 行）让所有 AIV 对齐这个区间。

**⑤ 逐个 Attention rank 接收（第 349–356 行）**：

```cpp
for tpIndex in [0, tpSize_):
    if (startExpert == 1) DispatchRecvShared(tpAttnRankId, tpIndex);   // 共享专家只在第一轮处理
    DispatchRecvRouted(tpAttnRankId, tpIndex, startExpert, endExpert);
```

**⑥ aiv0 写输出与 nextAttnRankId（第 358–421 行）**：
- 汇总 `tokenNum`，写 `tpBatchInfoTensor_` 的 `[tokenNum, attnRankId, layerIndex, startExpert, endExpert]`，`DataCopyPad` 到 `expertTokenNumsOut`（第 385–388 行）。
- 写 `epRecvCount / epRecvCountShared`（第 390–407 行）。
- 写 `nextAttnRankId`（第 410–420 行）：`endExpert == expertNumPerMoe_-1`（本轮全收完）时 `next = (tpAttnRankId+tpSize) % attnNum_`，否则保持 `tpAttnRankId`（下一轮继续收同一组）。写到共享内存尾部 `winSizeSum_ - UB_ALIGN`。

### 5.4 DispatchRecvShared（第 425–577 行）

接收共享专家 token：
1. 重读该 attn 的 batchInfo + expertTokenCount（第 448–461 行）。
2. 共享专家只有一个，`tokenNumSum = expertTokenCountTensor_(0)`，按 token 均分到 AIV（第 465–475 行）。
3. `DataCopyPad` 读入本 AIV 的 tokenAddrOffset 段（第 485–488 行）。
4. 逐 token：定位 `tpExpertTokenOffset`（本专家在 expandXShared 里的目标位置，第 512–513 行），`tokenOffsetOnSend` 从偏移索引表取源位置（第 515 行），`DataCopy` 从共享内存读 token → `DataCopy` 写到 `xOutSharedGMTensor_`（第 519–538 行）。
5. 量化时额外 `DataCopyPad` 写 scale 到 `quantScaleSharedGMTensor_`（第 527–528 行，见 §6.3）。

### 5.5 DispatchRecvRouted（第 580–759 行）

类似 Shared，但处理 `[startExpert, endExpert]` 范围：
- token 写到 `xOutGMTensor_`（非 Shared buffer）。
- 目标偏移 `tpExpertTokenOffset = preExpertTokenNum + tpExpertTokenCountTensor_(expertNumPerMoe_+currExpertId) + (expertTokenOffset - currExpertTokenNum)`（第 676–677 行），即「之前专家已发 + 当前专家已发 + 当前偏移」。
- **收完后 `aiv0` 更新 chunk 进度位（第 735–745 行）**：`[0, endExpert]` 全置 1 写回 `expertTokenCountChunk`。
- **`endExpert == expertNumPerMoe_-1` 时清 flag（第 747–749 行）** → `ClearFlags(attnRankId)`（见 §8）。
- 更新 `tpExpertTokenCountTensor_` 后半段的已收进度（第 751–753 行）。

---

## 6. 算子精读 3：`cam_moe_distribute_combine_send`（Expert 侧发送）

### 6.1 角色 / I/O / 分核

- **角色**：Expert rank 上，FFN 算完后把结果按 token 位置发回对应 Attention rank。
- **输入**：`expandX`（路由专家 FFN 结果）、`expandXShared`（共享专家 FFN 结果）、`expertTokenNums`（dispatch_recv 回传的 batch info，`batchInfoNum` 个 int64）、`commArgs`。
- **输出**：`expandX [1]`（flag）。
- **分核**：按 token 均分（`totalTokenNumPerAiv`，见 CombineSendShared 第 263–274 行、CombineSend 第 399–410 行）。
- **模板**：单类型 `CamMoeDistributeCombineSend<ExpandXType>`（第 60–64 行），BF16/FP16 二选一。

### 6.2 UB 布局（Init 第 162–203 行）

| 序 | 张量 | 用途 |
| --- | --- | --- |
| 1 | `tpBatchInfoTensor_` | `int64 × (INFO_NUM+tpSize+expertNumPerMoe*tpSize)`，读入 expertTokenNums |
| 2 | `tpExpertTokenNumTensor_` | uint32 × expertNumPerMoe×tpSize |
| 3 | `tpExpertTokenCountTensor_` | uint32 × expertNumPerMoe×2 |
| 4 | `timeCycleTensor_` | uint64 × 100，**性能埋点**（GetSystemCycle） |
| 5 | `moePreTokenNumTensor_` | uint32 × moeNum_，写 flag 用 |
| 6 | `xTensor_` | `maxUbTokenNum_ × hiddenSize`，staging，`maxUbTokenNum_ = ubReuseSize_/(sizeof*hiddenSize)`（第 195 行） |

`xTensor_` 吃掉剩余 UB 当「批量 staging 缓冲」，`maxUbTokenNum_` 决定每次循环最多搬多少个 token。

### 6.3 Process（第 207–246 行）

1. `DataCopyPad` 读入 `expertTokenNums` 到 `tpBatchInfoTensor_`（第 215–219 行）。
2. 清零计时器（第 221–223 行）。
3. `CombineSendShared()`（第 226 行）→ `CombineSend()`（第 228 行）。
4. 特定 rank 打印分段耗时（第 231–245 行，`rankId_==8||15` 且 `aivId_` 特定值才打印，用于调优）。

### 6.4 CombineSendShared（第 249–382 行）

共享专家结果回写：
- 仅当 `startExpert == 1`（即第一轮，第 251–254 行）才处理共享专家。
- 统计共享专家总 token 数（第 257–260 行），按 token 均分到 AIV。
- 主循环（第 300–379 行）：逐个 token 段推进 `currSendId`（按 `tpIndex = currSendId % tpSize_` 轮转各 Attention rank），每个段 `DataCopy` 从 `xSharedGMTensor_` 读 `cnt` 个 token 到 UB（第 340 行）→ `DataCopy` 写回 Attention 卡窗口（第 362 行）。
- 结尾 `SyncAll<true>()`（第 381 行）。

### 6.5 CombineSend（第 385–526 行）

路由专家结果回写，结构与 Shared 类似但多两点：
1. **只处理 `[startExpert, endExpert]` 区间**（第 390–397 行统计），因为 dispatch_recv 分 chunk，combine_send 也要跟着 chunk。
2. **每轮结束写 flag**（第 505–525 行）：`endExpert == expertNumPerMoe_-1`（最后一轮）时，`i % aivNum_ == aivId_` 的 AIV 给对应 Attention 卡写 `moePreTokenNumTensor_(0) = moePreTokenNum+1`（第 519–522 行），作为 combine_recv 的就绪标志。

`xTensor_` staging + `maxUbTokenNum_` 分批（第 467–468 行）：每次搬 `min(expertTokenNumRemain, sendTokenNumRemain, maxUbTokenNum_)` 个 token，避免 UB 溢出，也是唯一有「批量搬运」的地方。

---

## 7. 算子精读 4：`cam_moe_distribute_combine_recv`（Attention 侧接收）

### 7.1 角色 / I/O / 分核

- **角色**：Attention rank 上，从共享内存接收 Expert FFN 输出并累加成 `[batch, hidden]`。
- **输入**：`expandX`（flag）、`expertIds [batch, topk]`（路由表）、`expertScales [batch, topk]`（每 token 的量化 scale）、`commArgs`。
- **输出**：`xOut [batchSize, hiddenSize]`。
- **分核**：按 batch 均分（`batchSizePerAivStart_/End_`，Init 第 178–187 行），遍历 `topkId` 时 `if (tokenId >= batchSizePerAivEnd_) break`（第 349–351 行）。
- **模板**：`CamMoeDistributeCombineRecv<XOutType>`。

### 7.2 UB 布局（Init 第 195–255 行）

| 序 | 张量 | 用途 |
| --- | --- | --- |
| 1 | `idsPageTensor_` | 64KB 路由表分页缓存 |
| 2 | `expertTokenCountTensor_` | uint16 × expertNum_ |
| 3 | `aivCntTensor_`（LoadPreProcess 内） | 读 dispatch_send 的前缀和结果 |
| 4 | `expertPreTokenNumTensor_` | uint32 × expertNum_，专家内前缀和 |
| 5 | `expertTokenNumOnRecvTensor_` | uint16 × expertNum_，每专家已收进度 |
| 6 | `rowTmpFloatLocal_` / `sumFloatBufLocal_` / `mulBufLocal_` | float × hiddenSize，累加中间量 |
| 7 | `xOutTensor_` | XOutType × hiddenSize，token 搬运 |
| 8 | `moePreTokenNumTensor_` | uint32 × moeNum_，等待 flag |

### 7.3 LoadPreProcess（第 259–318 行）

与 dispatch_send 的 PreProcess 结果对接：读回 `aivCntTensor` 前缀和，还原 `expertTokenCountTensor_`（各专家最终收多少 token）。注意第 307 行、第 314 行两处注释掉的代码——dispatch_recv 侧其实**不需要** `tokenOffsetOnSend`/`expertTokenCountOnSend` 这两个偏移，所以只还原了 `expertTokenCountTensor_`。

### 7.4 CombineRecv（第 333–485 行）

1. **`PreProcess()`（第 335 行，详见 §8）**：等待所有 MoE 卡 combine_send 就绪。
2. 主循环（第 342–470 行）：`for topkId in [0, topk*batchSizePerAivEnd_)`：
   - `moeExpertId = GetIds(topkId)`；`moeRankId / expertId`（第 343–345 行）。
   - `topkIndex = topkId % topk_`；`topkIndex == 0` 时清零 `sumFloatBufLocal_`（第 357–360 行）——每个 token 的累加器。
   - **路由专家累加（第 362–394 行）**：读 token（`DataCopy` from 共享内存，第 372 行），`Cast` 到 float，`Muls` 乘 scale（`scaleVal = scalesGMTensor_(topkId)`），`Add` 累加到 `sumFloatBufLocal_`。
   - **共享专家累加（第 409–447 行）**：`topkIndex == topk_-1`（最后一个 topk）时，定位该 token 属于哪个共享专家（第 413–421 行），读共享专家结果并 `Add` 累加（**注意：共享专家不乘 scale**，第 444–446 行只有 `Cast + Add`）。
   - **输出（第 462–466 行）**：`Cast` 回 `XOutType`，`DataCopy` 到 `xOutGMTensor_[tokenId*hiddenSize_]`。
   - `expertTokenNumOnRecvTensor_(expertId) = idx + 1`（第 469 行）。
3. **清 flag（第 472–484 行）**：`SyncAll` 后 `aiv0` 把 Attention 卡窗口里的 combine flag 区清零，供下一轮复用。

> **值得注意的不对称**：combine_recv 对路由专家**乘 scale**（第 391 行 `Muls`），对共享专家**不乘 scale**（第 446 行只有 `Add`）。而 dispatch_recv 在量化模式下对 shared 也输出了 `quantScaleShared`。两处是否自洽（共享专家是否量化、scale 是否该乘）值得在优化笔记里标记为「需核对」。

---

## 8. 量化路径（DynamicQuant，BF16/FP16 → INT8）

### 8.1 dispatch_send 的 `QuantProcess`（第 843–885 行）

per-token 动态量化，流程：

```cpp
Cast(quantFloatTensor_, xTensor, CAST_NONE, hiddenSize_);   // → float
Abs(quantFloatAbsTensor_, quantFloatTensor_);               // |x|
ReduceMax(quantRowMaxTensor_, quantFloatAbsTensor_, ...);   // max|x|
dynamicScale = 127.0 / quantRowMaxTensor_.GetValue(0);      // 对称量化 scale
Muls(quantFloatTensor_, quantFloatTensor_, dynamicScale);   // x * scale
Cast(int32LocalTemp, quantFloatTensor_, CAST_RINT);         // round to nearest
Cast(halfLocalTemp, int32LocalTemp, CAST_ROUND);            // 经 half 中转（截断 int8 前）
Cast(xOutTensor, halfLocalTemp, CAST_TRUNC);                // → int8
// 把 1/dynamicScale 存到 xOut 缓冲尾部（第 880–884 行）
floatLocalTemp.SetValue(hiddenSize_/sizeof(float), 1.0/dynamicScale);
```

要点：
- 用 `ReduceMax` + `GetValue(0)` 读标量，中间强制 `V_S / S_V` 同步（第 856、859 行），是量化路径最大的串行点。
- **scale 以「反量化 scale」`1/dynamicScale` 内嵌存到 int8 token 缓冲尾部**（第 880–883 行），位置是 `hiddenSize_/sizeof(float)`。这就是 dispatch_recv 第 527 行 `xOutFloatTensor_[hiddenSize_/sizeof(float)]` 读出来的东西。
- 量化后 token 占用 `hiddenSizeQuant_ = (sizeof(int8)*hiddenSize + 32) / sizeof(int8)`（第 317 行），即每个 token 的 int8 数据 + 32 字节 scale 尾注。

### 8.2 dispatch_recv 侧的 scale 处理

量化模式下（DispatchRecvShared 第 519–529 行、DispatchRecvRouted 第 683–693 行）：
1. `DataCopy` 读 `hiddenSizeQuant_` 个 int8（含尾部 scale）。
2. `DataCopy` 把前 `hiddenSize_` 个 int8 写到 `expandX/expandXShared`（**注意这里没反量化，int8 原样落地**，反量化在 FFN 或后续算子做）。
3. `DataCopyPad` 把尾部 scale（`xOutFloatTensor_[hiddenSize_/sizeof(float)]`）写到 `quantScaleGMTensor_[tpExpertTokenOffset]`。

### 8.3 combine_recv 侧的 scale 消费

combine_recv 输入 `expertScales [batch, topk]`，路由专家累加时 `Muls(rowTmpFloatLocal_, ..., scaleVal)`（第 391 行）。即 FFN 输出回到 Attention 后，按 token 对应的 scale 反量化再累加。

### 8.4 量化路径小结

```
dispatch_send:  BF16 → float → |·| → max → scale=127/max → x*scale → rint → int8 (scale 内嵌尾部)
dispatch_recv:  int8 原样搬进 expandX + scale 单独进 dynamicScales
combine_send:   (FFN 输出原样回写，不涉及 scale)
combine_recv:   路由专家结果 × scale 累加；共享专家不乘 scale
```

---

## 9. 跨 AIV 协调机制（PreProcess 前缀和）

`dispatch_send::PreProcess`（第 629–773 行）是全套代码里唯一的「跨 AIV 归约」，作用：每个 AIV 只知道自己负责的那段 batch 会给每张 MoE 卡/每个专家发多少 token，但最终每个 AIV 写共享内存时的**偏移**取决于「前面所有 AIV 发了多少」。因此需要一次全局前缀和。

流程（对照代码行号）：

1. **分配临时区**（第 631–633 行）：`aivCntTensor`，尺寸 `(moeNum_ + expertNum_) * aivNum_`，行 = AIV，列 = 「moeNum_ 张卡 + expertNum_ 个专家」。
2. **清零（第 643–647 行）** + 先算共享专家计数（第 650–658 行）：

   ```cpp
   sharedTokenNumPerMoe = batchSize_ / moeNum_;   // 共享专家收所有 token，均匀分到各 MoE 卡
   sharedTokenNumPerMoeRemain = batchSize_ % moeNum_;
   ```

   > 注意：共享专家计数是**静态均分**（`batchSize/moeNum`），不看 expertIds——这是「共享专家 = 所有 token 都过一遍」的语义。
3. **各 AIV 统计自己负责的发送量（第 671–706 行）**：遍历 `[topk*start, topk*end)`，用 `GetIds` 查路由，`aivCntTensor(moeNum_ + expertId)++` 记专家计数，`aivCntTensor(moeRankId)++` 记卡计数（去重后，第 698–705 行）。
4. **写自己的统计到共享内存（第 708–715 行）**：`srcGM = GetWindAddrByRankId(rankId_)`（写本卡自己的窗口），`aivCntGMTensor[aivCntPerAiv * aivId_]` 处 `DataCopyPad` 一行。**这解释了下标「attnRankId」——Attention 卡的共享内存窗口第 0 区就是这块前缀和区**。
5. **`SyncAll<true>()`（第 717 行）**：等所有 AIV 写完。
6. **读回全部 AIV 的统计（第 720–723 行）**。
7. **按列并行前缀和（第 726–739 行）**：`col % aivNum_ == aivId_` 的 AIV 负责第 `col` 列，逐行累加（第 731–738 行），每步 `DataCopyPad` 写回共享内存单元素。
8. **`SyncAll<true>()`（第 741 行）**。
9. **读回完整前缀和（第 744–747 行）**。
10. **各 AIV 更新自己的偏移（第 750–772 行）**：
    - `aivId_ == 0`：只还原 `expertTokenCountTensor_`（各专家总数 = 最后一行前缀和）。
    - 其他 AIV：`tokenOffsetOnSendTensor_(col) += aivCntTensor(aivCntPerAiv*(aivId_-1)+col)`（本 AIV 之前所有 AIV 发给该卡的数量，即本 AIV 的起始偏移），`expertTokenCountOnSendTensor_(expertId)` 同理，`expertTokenCountTensor_` 取最后一行。

### 9.1 第 734 行 `aivCntTensor(0) = sum;` 的真实作用（勘误）

初读疑似调试残留，细读后确认这是**功能性代码**：紧随其后的第 736 行要把单个 uint32 的 `sum` 写回 GM（`DataCopyPad(aivCntGMTensor[...], aivCntTensor, {1U, sizeof(uint32_t), ...})`），而 `DataCopyPad` 的源操作数必须是 UB 上的 `LocalTensor`，标量寄存器里的 `sum` 无法直接 DMA——所以借用 `aivCntTensor` 的位置 0 当 **DMA staging 区**（注意 `DataCopyPad` 的源就是 `aivCntTensor` 本身，从位置 0 取 4 字节）。

代价与脆弱性：它破坏了 `aivCntTensor[0][0]` 的本地副本值。之所以没出 bug，是因为第 744–747 行会整表重新从 GM 读回覆盖。这是"能跑但依赖后续整表重读"的脆弱写法——若优化时去掉第二次整表读回，必须同时改掉这个 staging 技巧（比如单独分配一个 4B 的 staging tensor）。

---

## 10. 等待与同步机制

四算子里出现了三类同步，容易混，这里集中梳理。

### 10.1 硬件管道同步 `SyncFunc<event>`（各算子顶部）

```cpp
template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}
```

（dispatch_send.h 第 41–46 行）——这是 AscendC 的**事件同步**，保证 MTE2(读)/V(向量)/MTE3(写) 三条流水之间 data dependency。典型序列：
- `MTE2_S`：等读完成再进标量/后续
- `S_MTE2`：标量/寄存器就绪后放行读
- `S_MTE3`：数据就绪后放行写
- `MTE3_S`：等写完成
- `V_S / S_V / V_MTE2 / V_MTE3`：向量流水与其他流水之间的同步

### 10.2 块内同步 `SyncAll<true>()`

`SyncAll<true>` 是 **AI Core 块内所有 AIV 的 barrier**（`<true>` 表示带内存可见性/屏障语义）。出现位置：
- dispatch_send PreProcess：第 717、741 行（前缀和两步之间）；SetFlags 开头第 778 行。
- dispatch_recv：第 335 行（算完 start/end expert 后对齐）、ClearFlags 第 839 行。
- combine_send：CombineSendShared 第 381 行、CombineSend 第 502 行。
- combine_recv：第 472 行（清 flag 前）。

### 10.3 跨卡等待：flag 自旋（IPC 同步）

这是「异步算子」的核心，用共享内存里的 flag 字段做**生产者-消费者**，不依赖 NCCL。

**dispatch 方向**（`dispatch_recv::WaitFromAttn`，第 762–834 行）：

```cpp
// 1. 从共享内存尾部读上次处理的 nextAttnRankId（第 767 行）
DataCopy(tokenNumAttnRankIdLayerIndexTensor_, dstMoeTokenNumGMTensor, UB_ALIGN/sizeof(int64_t));
uint32_t attnRankId = tokenNumAttnRankIdLayerIndexTensor_(0);

// 2. aiv0 轮询所有 Attention rank 的 flag（第 776–805 行）
if (aivId_ == 0) {
    while (!isAllRecv) {
        for i in [0, attnNum_/tpSize_):         // 遍历 dp 组
            for j in [tpSize_, 1):              // 组内每个 rank
                srcRankId = (attnRankId + tpSize_*i + j-1) % attnNum_;
                DataCopyPad(batchInfo);           // 读 flag
                if (tokenNum == 0) { isAllRecv=false; break; }   // 第 790 行
            if (isAllRecv) {
                attnRankId = (attnRankId + tpSize_*i) % attnNum_;
                DataCopyPad(写 BATCH_INFO_VAL_NUM-1 位置);  // 通知其他 AIV（第 799 行）
                break;
            }
    }
} else {
    // 3. 其他 AIV 轮询 aiv0 的「就绪标记」（第 806–824 行），不再自己去读每张卡的 flag
    while (!isAllRecv) { ... if (batchInfo[BATCH_INFO_VAL_NUM-1] != 0) { attnRankId = srcRankId; isAllRecv=true; } }
}
```

设计意图：只有 `aiv0` 做真正的跨卡轮询（读所有 Attention 卡的 flag），其他 AIV 只轮询「aiv0 写在共享内存里的通知」，减少跨卡读的压力。但**没有 sleep/backoff**，是紧自旋（后文优化笔记会展开）。

**combine 方向**（`combine_recv::PreProcess`，第 488–541 行）：

```cpp
while (true) {
    DataCopyPad(moePreTokenNumTensor_, dataInfoGMTensor, sizeof(uint32)*moeNum_);  // 第 524 行
    SyncFunc<MTE2_S>(); SyncFunc<S_MTE2>();
    bool isAllRecv = true;
    for i in [0, moeNum_): if (moePreTokenNumTensor_(i) == 0) { isAllRecv=false; break; }
    if (isAllRecv) break;
}
```

与 dispatch 方向不同，combine_recv 是**所有 AIV 各自独立轮询**（没有 aiv0 广播），且 `while(true)` 无界自旋。

### 10.4 两处 flag 清零

- `dispatch_recv::ClearFlags`（第 837–856 行）：Expert 收完某 attn 全部专家后，`SyncAll` + `aiv0` 把该 attn 的 batchInfo 清零，让 Attention 侧可以复用窗口。
- `combine_recv` 末尾（第 474–484 行）：Attention 收完 combine 结果后，`aiv0` 把 combine flag 区清零。

---

## 11. 代码精读标注（关键行号速查表）

以下路径均相对 `cam-aynsc/src/comm_operator/ascend_kernels/`。

| 关注点 | 文件 | 行号 |
| --- | --- | --- |
| GET_COMM_ARGS 读 6 个 int | `utils/op_kernel/op_def.h` | 1–9 |
| CommArgs 结构 + 常量 | `utils/op_kernel/comm_args.h` | 18–82 |
| GetMaxWindowSize（默认 408MB） | `utils/op_host/error_log.h` | 67–87 |
| SyncFunc 事件同步宏 | dispatch_send.h | 41–46 |
| 路由表分页缓存 GetIds | dispatch_send.h | 74–86 |
| winSizePerAttn_ 计算（MoE 窗口） | dispatch_send.h | 228–239 |
| 按 batch 均分 AIV | dispatch_send.h | 251–263 |
| UB 布局分配 | dispatch_send.h | 269–370 |
| **DispatchSend 主循环** | dispatch_send.h | 386–626 |
| 专家内前缀和 | dispatch_send.h | 396–406 |
| 共享专家静态均分 | dispatch_send.h | 650–658 |
| **PreProcess 跨 AIV 前缀和** | dispatch_send.h | 629–773 |
| 前缀和写共享内存 | dispatch_send.h | 708–715 |
| 按列并行前缀和（含可疑第 734 行） | dispatch_send.h | 726–739 |
| **SetFlags 写 flag** | dispatch_send.h | 776–840 |
| **QuantProcess 量化** | dispatch_send.h | 843–885 |
| Push/Flush 16 个一批 | dispatch_send.h | 154–200 |
| **WaitFromAttn 轮询** | dispatch_recv.h | 762–834 |
| 分 chunk 算 start/end expert | dispatch_recv.h | 319–334 |
| DispatchRecvShared | dispatch_recv.h | 425–577 |
| DispatchRecvRouted | dispatch_recv.h | 580–759 |
| ClearFlags | dispatch_recv.h | 837–856 |
| dispatchOffset_（combine 窗口偏移） | combine_send.h | 146 |
| CombineSendShared | combine_send.h | 249–382 |
| CombineSend + 写 flag | combine_send.h | 385–526 |
| combine 结果写回偏移公式 | combine_send.h | 292–294, 431–433 |
| LoadPreProcess（读前缀和） | combine_recv.h | 259–318 |
| CombineRecv 主循环（×scale 累加） | combine_recv.h | 333–485 |
| combine_recv 等待 MoE 就绪 | combine_recv.h | 488–541 |
| Tiling 校验 + sharedMem 上限 | dispatch_send host .cpp | 161–171 |
| maxTokenNum / batchSizeFactor | dispatch_recv host .cpp | 166–167 |

---

## 附：与任务描述有出入的几点勘误

1. **「四个算子共有同一 Info 结构体」不准确**：实际是四个 Info 字段各有差异（见 §2.5 表），dispatch_send 独有 `maxBs/layerIndex/dynamicQuant`，dispatch_recv 独有 `maxTokenNum`，combine 系列无 `dynamicQuant`。
2. **「每个 Attention rank 在共享内存中有一个 winSizePerAttn_ 窗口」方向搞反了**：`winSizePerAttn_` 是 Expert 侧窗口布局（dispatch 方向），窗口下标是 Attention rank id；Attention 侧另有 combine 方向的窗口布局（见 §3.2）。
3. **共享内存布局图里 `flag` 是独立字段不准确**：`flag` 就是 batchInfo 首元素 `tokenNum`，值为 `tokenNum+1`（非零即就绪），不是单独字段。
4. **combine_send「类似 dispatch_send 但方向相反」过于笼统**：combine_send **没有** PreProcess 前缀和阶段（它直接从 `expertTokenNums` 读 dispatch_recv 算好的分布），也**没有**路由表缓存，其复杂度远低于 dispatch_send。
