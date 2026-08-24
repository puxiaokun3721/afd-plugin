# 异步 AFD 相关论文/博客结构化摘要

> 汇总日期：2026-08-23
> 范围：三个来源，每篇独立一节。其中第 1、2 篇为异步 AFD 论文，第 3 篇为 AFD Plugin 的论文索引 Issue（用于交叉对照论文间关系）。

---

## 来源 1 — ASAP：MoE Prefill 的异步 AFD 推理系统（作者早期版本）

- **论文标题**：ASAP: A Disaggregated and Asynchronous Inference System for MoE Prefill
- **作者**：Weiwei Chen, Shuang Chen, Lele Li, Qiang Hu, Han Li, Xin Ye, Ming Yan, Zhibin Yu（单位：Huawei）
- **发表时间**：arXiv:2606.22541，v1，2026-06-21
- **arXiv 分类**：cs.DC

### 解决的核心问题（一句话）
MoE 推理在 prefill 阶段，attention 采用数据并行（DP）、MoE 采用专家并行（EP）的混合并行，二者之间每层都要做全局同步 barrier；在线服务下请求到达率和序列长度高度异构（DP 不均衡），导致大量同步等待（straggler effect），拖垮 TTFT 和吞吐。

### 核心方案概述
ASAP 把 attention 与 MoE 阶段**物理解耦（disaggregate）到不同设备**，用一套**非阻塞、非对称的异步通信原语**替代同步 barrier，实现完全异步的执行流水线，并配套四项优化来抵消"异步带来的碎片化/乱序执行"的副作用。

### 关键技术点

**1. 异步通信原语（核心底座）**
- 每个设备分配一块**静态、全局可见的共享内存 buffer**，充当发送方与接收方之间的"superhub"。
- 发送方写入目标 buffer 并置位 flag 后立即返回，无需等接收方确认；接收方在自己任务空闲时轮询 bitmap、取数、清位。
- 定义了四个原语：`async-dispatch-send` / `async-dispatch-recv`（attention→MoE 的下发）、`async-combine-send` / `async-combine-recv`（MoE→attention 的回传），替代同步 all-to-all。
- 反压机制：若目标 flag 未清则发送方阻塞写，保证数据完整性。
- 在 Ascend NPU 上用 AscendC 实现，约 4000 行；远程内存读写用 DataCopy/DataCopyPad，微秒级延迟。

**2. 请求调度层两项优化**
- **Length-aware batching（长度感知组 batch）**：由于 MoE 阶段存在"内存受限平台期→计算受限线性区"的拐点（本例约 2k tokens），调度器刻意把请求聚合成超过拐点的 batch，保证 MoE 计算强度；解耦后无需再做跨 DP 组的负载均衡，组 batch 算法被大幅简化。
- **Dual-batch interleaving（双 batch 交错）**：给每个 attention DP 组同时调度两个 batch 交错执行——一个 batch 下发到 MoE 时，attention 设备立即处理另一个 batch，消除 attention 空闲。对超长序列（attention 远大于 MoE，如 >16k）关闭交错，避免排队。

**3. 模型执行层两项优化**
- **Communication-computation overlapping（三重流）**：每个设备上一条 compute 流 + 两条 communication 流；cube core/vector core 静态分区（24 个 cube + 24 个 vector 给 compute，每条 comm 流 12 个 vector core）。实测发现 attention 设备上通信流量会干扰计算内核，故三重流**只在 MoE 设备上启用**。
- **Bubble-free kernel dispatching（层无关 MoE Super Kernel）**：异步导致 MoE 层乱序执行，层号运行时才确定，CPU 无法预先派发内核（单层派发开销约 220μs，DeepSeek-V3.2 有 61 层）。把逐层 MoE GMM 内核改造成**单一"层无关"Super Kernel**：内核持有全部 L 层权重指针、预计算地址索引表、把 layer ID 作为设备端动态参数，从而可提前派发、消除内核间气泡。

**4. 拓扑 / 硬件**
- 部署在华为 **CloudMatrix384 超节点**：48 节点 × 8 NPU，共 384 颗 Ascend 910 NPU；每个 NPU die 24 个 cube core、48 个 vector core、64GB HBM；通过两级分层 Unified-Bus (UB) mesh 互联，任意 NPU 对之间双向 400GB/s、微秒级 RDMA。
- 评估配置：DeepSeek-V3.2（671B 总参 / 256 专家 + 1 共享专家 / 每 token 激活 37B），ASAP 用 D=4、T=4、E=16 共 32 NPU，与同步基线 DP8-TP4-EP32 设备数一致。

### 关键实验数据
- 低负载（RPS=1）：mean TTFT 350ms，比 Default（533ms）低 34.3%，比 ChunkedPrefill（388ms）低 9.8%。
- RPS=4：TTFT 比 Default 降 54.9%、比 ChunkedPrefill 降 41.8%；RPS=8：Default 已无法支撑，ASAP 比 ChunkedPrefill 降 68.3%。
- **SLO 吞吐（TTFT ≤ 5s）**：ASAP 支撑 20 RPS，Default 6.8 RPS、ChunkedPrefill 10.5 RPS → 相对 Default **+194%**、相对 ChunkedPrefill **+90%**（论文摘要所强调的"90%"即相对最强同步基线 ChunkedPrefill）。
- 时延分解：短请求（<512 token）下同步系统非内核开销占 TTFT 约 85%（同步等待 55% + 排队 30%）；ASAP 把非内核延迟最多降 80%。
- 通信：async-dispatch 延迟约为同步 P2P 的 1/4（1k token）~1/5.8（8k token）；ASAP 发 512 token <0.1ms，优于先前 GPU 异步 P2P 方案的 >0.5ms。
- 消融：dual-batch interleaving +14.3% SLO 吞吐、三重流 +12.4%、Super Kernel 省约 13.4ms TTFT 且 +6% SLO 吞吐。

### 与 AFD Plugin 的关系
- 属于"**attention-FFN 解耦 + 异步执行**"路线，与 AFD Plugin 的核心理念（attention/FFN 分离部署）同源。
- 但 ASAP 面向 **Ascend NPU（CloudMatrix384 超节点 + AscendC + CANN）**，与 AFD Plugin 的 vLLM-Ascend 生态高度相关，其异步通信原语、AscendC 实现细节对 AFD Plugin 的 NPU 异步 connector 有直接借鉴价值。
- 差异点：ASAP 是完整的生产推理框架级改造，而 AFD Plugin 以 vLLM 插件形式实现；ASAP 主攻 **prefill**，AFD 生态同时覆盖 prefill 与 decode。

---

## 来源 2 — AMoE：面向 decode 的异步 AFD（含组 batch 算法讨论）

- **论文标题**：Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony
- **作者**：Shaoyu Wang*, Guangrong He*, Geon-Woo Kim, Yanqi Zhou, Seo Jin Park（*共同一作；单位：University of Southern California、University of Texas at Austin、Google）
- **发表时间**：arXiv:2505.08944，v1 2025-05-13，v2 2025-05-29
- **arXiv 分类**：cs.DC

### 解决的核心问题（一句话）
MoE 推理服务中专家负载倾斜（expert load skew）导致 EP 部署下 GPU 等待最热专家（straggler）与冷专家小 batch 浪费带宽，同步 barrier 通信进一步放大低效。

### 核心方案概述
提出 **Asynchronous Expert Parallelism (AEP)** 范式：把层执行从 barrier 式同步中解耦，每层按 token 粒度**μ-queuing 排队 + 自适应 re-batch**，GPU 不等慢专家、哪个层就绪就执行哪个层；实现为服务系统 **AMoE**，将 attention 与 expert 层解耦，并用 defragging 调度器减少 batch 碎片化。

### 关键技术点

**1. μ-queuing + 自适应组 batch（层粒度排队）**
- 以"单个层"为粒度排队 token（μ-queue），在层自身队列上攒够 token 后即时 re-batch 执行，替代"整批固定贯穿所有层"的传统方式。
- 同一 GPU 上 colocate 多个专家/层，调度器可复用多个层：优先执行有足够 token 的热专家，让冷专家多攒 token 再执行。

**2. Defragging 调度器（组 batch 算法，核心贡献）**
- 动机：对比两个朴素策略——MTFS（最多 token 的层优先，会留下"孤儿"碎片）和 FLFS（最早层优先，新请求会抢占、可能活锁）。
- 算法（Algorithm 1）：为每个 (block, expert) 队列计算一个分数 = 本队列 token 数 + 带衰减系数 δ 的 **lookahead 加权分**（向下游各层 token 密度加权，越远的层贡献衰减越多）；选分数最高的层执行。既像 FLFS 一样促进批次合并，又像 MTFS 一样考虑队列占用，平衡碎片化与前进性。

**3. 通信方式**
- 运行时间点对点（P2P）通信，主传输用 **NCCL**；为应对 NCCL 需收发双方同步、且变长 tensor 的问题，采用**两阶段协议**：Phase 1 通过 CPU 侧 **ZeroMQ** 消息队列先传元数据（告知对方 tensor 尺寸），Phase 2 再启动 NCCLSend/Recv；CPU 启动 NCCL 内核后不等待、继续处理下一任务，实现单线程并发收发多批。
- token 数据常驻 GPU 显存，CPU 侧只维护元数据（RequestID / LayerID / Prefill_length / Topk_weights 等）。

**4. 拓扑 / 结构**
- 分两层拓扑：attention 层用 DP、expert 层用 EP（对齐行业标准），但**不用阻塞式 group communication**。
- 同一 DP rank 的请求在整个自回归解码期保持绑定，复用同一 GPU 的 KV cache。
- 刻意**不做 pipeline parallelism（PP）分块**，而是把各层类型跨 block colocate，避免 PP 的负载不均与高 inter-token 延迟（GPU 充裕时才允许形成多个 pipeline stage）。
- 解耦 attention/expert 还允许二者用不同数量/类型的 GPU（如 KV cache 受限时给 attention 更多 GPU，或引入异构 attention 硬件）。

**5. 运行时四阶段流水**：receptor（按 LayerID 入队，含 token pool 做 top-K 合并前的 token 聚合）→ scheduler（选层）→ executor（自定义 CUDA kernel 把零散 token batch 拼成连续 batch；管 KV cache/page table）→ dispatcher（按 expert ID 或 DP rank 重排 token 再分发）。

### 关键实验数据
- 硬件：8×A100-40GB/80GB（NVSwitch 600GB/s，4×100Gbps EFA），规模扩展用 2 节点 16×A100。
- 模型：Mixtral 8x7B（8 专家，GQA 改 MQA 以缓解 KV cache 限制），专家负载按 Dolly 数据集剖面拟合的指数分布模拟真实 skew。
- 吞吐：Top-1 路由下 AMoE 相对 SGLang 提升 **2.7×（短）/ 2.3×（中）/ 2.0×（reasonable）**；Top-2 路由下提升略小（激活专家增多稀释 skew + top-K merge 引入部分同步点）。
- 多节点扩展（16 专家 / 16 GPU）：AMoE 达 SGLang-EP 的 **3×** 吞吐、自身相对 8GPU 场景 **1.92×** 提升；而 SGLang 在 GPU 翻倍后吞吐**几乎不增长**（印证 AEP 近线性扩展）。
- 调度器：defragging 在 80% 负载下同时优于 MTFS 与 FLFS（更低 ITL、更高吞吐）；FLFS 在持续新请求下会出现活锁、完成率落后到达率。
- 开销分解：单步 attention 2.7ms、expert 0.8ms；attention 侧 page table 与 post-processing 开销显著，expert 侧主要时间在内核计算；调度阶段本身占比很小。

### 与 AFD Plugin 的关系
- 直接贡献了**异步 expert 并行 + attention/expert 解耦**这一范式，是 AFD 异步方向的代表性学术工作，且在 AFD Plugin 的论文索引 Issue（来源 3）中被列为"直接关于 AFD"的第 8 篇。
- 代码与 AFD Plugin 同源：AMoE **基于 vLLM 构建**（约 6K 行 Python + 4.8K 行 C++，通过 pybind11 暴露），调度器/communicator/receptor/dispatcher 用 C++ 以压低逐层调度开销——与 AFD Plugin 的 vLLM 插件化路线高度同构。
- 差异点：AMoE 面向 **GPU（NVIDIA + NCCL + CUDA）**且聚焦 **decode** 阶段，而 AFD Plugin 落地 Ascend NPU；二者在"异步调度 / 组 batch / 通信去 barrier"思想上可互相印证。

---

## 来源 3 — AFD Plugin 论文索引 Issue（交叉对照）

- **标题**：[Doc]: important papers of attention-ffn disaggregation
- **作者**：hsliuustc0106
- **时间**：创建 2026-06-23，最后更新 2026-08-19；标签 `documentation`；状态 open；0 条评论
- **地址**：https://github.com/vllm-project/afd-plugin/issues/32

### 核心内容
作者用多种检索词（`Attention-FFN Disaggregation`、`attention FFN disaggregation`、`AFD LLM serving`、`disaggregated expert parallelism` 等）在 arXiv 上系统性搜集 AFD 相关论文，整理成一份索引，并建议**把"直接 AFD 论文"与"仅在某个组件/仿真/异构算子放置/注意力卸载中用到 AFD 的相邻工作"分开跟踪**。

### 与本文档两篇论文的对应关系
- **来源 1（ASAP，arXiv:2606.22541）** 在该 Issue 中被列为"直接关于 AFD"的 **第 5 篇**，标注要点为"MoE prefill 服务系统，解耦 attention 与 MoE 阶段，用异步执行流水线替代全局同步 barrier，提升 SLO 合规的 prefill 吞吐"。
- **来源 2（AMoE，arXiv:2505.08944）** 被列为"直接关于 AFD"的 **第 8 篇**，标注要点为"AMoE 将 attention 从 expert 层解耦，用异步 expert 并行、逐层 μ-queuing、自适应 re-batching 与 defragmentation 缓解 expert skew 与同步开销"。

### 与 AFD Plugin 的关系
- 这是 AFD Plugin 项目的官方论文/相关工作索引，说明社区已把"异步 AFD"作为一条主线来追踪，且 ASAP、AMoE 与 MegaScale-Infer、Janus、FinDEP、ExpertPlex 等并列为该方向的关键参考文献。
- 对 AFD Plugin 的价值：这份清单可直接作为 AFD Plugin 相关工作的比对基线；本项目（vllm-afd-plugin，vLLM-Ascend 上的 AFD 插件）在引用或对齐社区做法时，可据此区分"直接 AFD"与"相邻工作"。

---

## 三篇横向小结

| 维度 | ASAP（来源 1） | AMoE（来源 2） | Issue 索引（来源 3） |
|---|---|---|---|
| 阶段 | **Prefill** | **Decode** | 全量索引 |
| 硬件 | Ascend NPU（CloudMatrix384） | NVIDIA GPU（A100 + NCCL） | — |
| 去 barrier 手段 | 共享内存 buffer + 4 个异步原语 | μ-queuing + NCCL/ZeroMQ 两阶段 P2P | — |
| 组 batch 算法 | Length-aware batching + dual-batch interleaving | Defragging 调度器（lookahead 加权） | — |
| 主要指标 | SLO 吞吐 +90% vs ChunkedPrefill | 吞吐最高 2.7×/3× vs SGLang | — |

共同主线：**Attention-FFN 解耦（AFD）+ 消除全局同步 barrier + 异步执行**，是 vllm-afd-plugin 项目最直接对齐的学术方向之一。
