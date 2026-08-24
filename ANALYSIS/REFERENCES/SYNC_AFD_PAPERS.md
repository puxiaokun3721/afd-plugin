# AFD（Attention-FFN Disaggregation）相关论文/博客结构化摘要

> 说明：本文件汇总三份关于 **AFD（Attention-FFN 解耦，Attention-FFN Disaggregation）** 的来源。
> 需要特别指出的是，用户给出的两个 arXiv 链接 **并不是 "FastAFD 论文"**：
> - `arXiv:2504.02263` 是 **MegaScale-Infer**（AFD 的第一篇工作）；
> - `arXiv:2507.19427` 是 **Step-3**（字节跳动 StepFun 的 VLM，首次提出并命名为 AFD）。
>
> 真正介绍 **FastAFD** 的是第三个来源（Hao AI Lab @ UCSD 的博客）。截至 2026-08-23，FastAFD 尚未有独立的 arXiv 论文（经 arXiv API 检索 `all:FastAFD` 无结果）。
> 因此下文按"来源 1 = MegaScale-Infer、来源 2 = Step-3、来源 3 = FastAFD 博客"组织，三者共同构成 AFD 技术的演进脉络。

---

## 来源 1：MegaScale-Infer（arXiv:2504.02263）

### 基本信息
- **标题**：MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism
- **作者**：Ruidong Zhu, Ziheng Jiang, Chao Jin, Peng Wu, Cesar A. Stuardo, Dongyang Wang, Xinlei Zhang, Huaping Zhou, Haoran Wei, Yang Cheng, Jianzhe Xiao, Xinyi Zhang, Lingjun Liu, Haibin Lin, Li-Wen Chang, Jianxi Ye, Xiao Yu, Xuanzhe Liu, Xin Jin, Xin Liu（字节跳动 ByteDance Seed 团队）
- **发表时间**：2025-04-03 首次提交（v1），2025-07-26 最后修订（v4）
- **链接**：https://arxiv.org/abs/2504.02263

### 核心问题（一句话）
大规模 MoE 大模型推理中，稀疏激活使 FFN 从"计算密集"变成"显存/访存密集"，导致 GPU 利用率低下、运营成本高。

### 核心方案概述
提出 MegaScale-Infer，将每个 Transformer 层内的 **Attention 模块与 FFN 模块解耦（disaggregation）**，让两类模块独立扩缩容、采用不同的并行策略、并异构部署，从而在 MoE 稀疏特性下最大化 GPU 利用率。

### 关键技术点
- **通信方式**：自研高性能 **M2N 通信库**，消除不必要的 GPU→CPU 拷贝、group 初始化开销与 GPU 同步；通信控制路径放在 CPU 侧（基于 RDMA write-with-immediate + GDRCopy flush），以不占用 GPU SM（这一设计在后来的 FastAFD 中被明确指出是为 RDMA 互联机群做的取舍）。
- **拓扑**：Attention 与 FFN 解耦后各自独立部署，形成 M 个 Attention 节点 + N 个 FFN 节点的 M:N 拓扑；token 经 "M2N dispatch"（Attention→FFN）与 "N2M combine"（FFN→Attention）跨节点流动。
- **调度策略**：**乒乓流水线（ping-pong pipeline parallelism）**——把请求 batch 切成 micro-batch，在 Attention 与 FFN 之间往返穿梭；配合两类模块各自的模型并行（model parallelism）隐藏通信开销。给出理想化三条件模型：① Attention 与 MoE 计算均衡（T_a ≈ T_e）；② 单向通信时间短于流水线周期；③ micro-batch 数量足够覆盖往返通信。
- **模型改动**：只改 placement（放置方式），不改模型计算本身。

### 关键实验数据
- 相对 SOTA 方案，**单 GPU 吞吐最高提升 1.90x**。

### 与 AFD Plugin 的关系
**AFD 概念的最早来源。** afd-plugin 实现的正是这篇工作提出的 "Attention-FFN 解耦 + M2N dispatch / N2M combine" 架构：plugin 中的 `attention_worker` / `ffn_worker`、`ffn_model_runner` / `attention_model_runner` 角色拆分、`num_attention_ranks` / `num_ffn_ranks` 拓扑、以及 connector 驱动的 token 跨角色搬运，均可视为 MegaScale-Infer 思路在 vLLM 插件层的落地。注意：MegaScale-Infer 的 CPU 侧通信控制路径与 afd-plugin 的 CUDA/NPU connector（P2P、CAMP2P）不同，后者更接近 FastAFD 的 GPU 侧方案。

---

## 来源 2：Step-3（arXiv:2507.19427）

### 基本信息
- **标题**：Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding
- **作者**：StepFun（上海阶跃星辰，Bin Wang, Bojun Wang, Changyi Wan 等约 200 人联合署名，含 Daxin Jiang、Heung-Yeung Shum、Xiangyu Zhang 等）
- **发表时间**：2025-07-25 提交
- **链接**：https://arxiv.org/abs/2507.19427

### 核心问题（一句话）
大模型（尤其长上下文推理）解码阶段硬件效率低下、解码成本过高。

### 核心方案概述
提出 **Step-3**：一个 321B 参数的 VLM，采用"模型-系统协同设计（model-system co-design）"来最小化解码成本，其中系统侧的核心创新即 **AFD（Attention-FFN Disaggregation）** 分布式推理系统——把 Attention 层与 FFN 层解耦成专用子系统。

### 关键技术点
- **通信方式**：自研 **StepMesh** 通信库，构建在 BytePS 之上；专用网络线程在 CPU 上执行 RDMA 操作，GPU Direct RDMA 保持 tensor 负载在 GPU 显存中。**同样把通信控制路径放在 CPU 侧**，跨节点走 RDMA fabric。
- **拓扑**：Attention 与 FFN 解耦为两类 worker（与 MegaScale-Infer 同类的 M:N 结构），worker/server push-pull API。
- **调度策略**：与 MegaScale-Infer 同为 Attention-FFN 解耦 + token 跨角色流动；结合 MoE 稀疏激活（每 token 激活 38B 参数）。
- **模型侧创新（与 AFD 互补）**：**Multi-Matrix Factorization Attention（MFA）**，显著降低 KV cache 大小与计算量，同时保持注意力表达能力。

### 关键实验数据
- 每 token 激活 38B 参数（多于 DeepSeek-V3 与 Qwen3 MoE 235B），理论解码成本显著低于二者，且上下文越长收益越大。
- 在 Hopper GPU 上实现 **4039 tokens/s/GPU** 的解码吞吐（50ms TPOT SLA、4K context、FP8、无 MTP），高于同配置下 DeepSeek-V3 的 2324，刷新 LLM 解码的 Pareto 前沿。

### 与 AFD Plugin 的关系
**AFD 这个名称的命名来源，也是 AFD 落地的第二个重要系统实现。** 本仓库的 vLLM 插件（afd-plugin）直接沿用 "AFD = Attention-FFN Disaggregation" 这一术语与角色拆分。Step-3 与 MegaScale-Infer 一样把通信控制路径放 CPU/RDMA 侧，与 afd-plugin 的 GPU 侧/NPU 侧 connector 方案形成对照，是理解 afd-plugin 为何选 P2P/CAMP2P 连接器的关键背景。

---

## 来源 3：FastAFD（Hao AI Lab @ UCSD 博客）

### 基本信息
- **标题**：FastAFD: Open-Source Large-Scale Attention-FFN Disaggregation on Blackwell NVL72
- **作者**：Yichao Fu*, Yuxuan Zhang*, Ruitian Wang, Junda Chen, Hao Zhang（Hao AI Lab, UCSD；* 为共同第一作者）
- **发表时间**：2026-07-06
- **链接**：https://haoailab.com/blogs/fastafd/ （阅读时长约 38 分钟）

### 核心问题（一句话）
在 GB200 NVL72 这类 scale-up 强互联系统上，之前的 AFD 收益是否仍然成立？以及展望 Vera Rubin + LPX/LPU 异构硬件上 AFD 是否仍有益？

### 核心方案概述
FastAFD 是**开源**的 AFD serving 原型，首次系统性地在 **GB200 NVL72**（单 NVLink/NVSwitch 域，72 GPU）上实测 AFD 性能，并用实测的延迟分解去**外推（project）** NVIDIA Vera Rubin + LPX/LPU 上的表现。核心结论：在 NVL72 上 AFD 必须靠"FFN 侧聚合 + 隐藏 dispatch/combine + 隐藏控制面开销"才能赢过已经很强的 colocated vLLM 基线。

### 关键技术点
- **通信方式**：**DeepEP 风格**的 dispatch/combine 内核，通过 NVLink/NVSwitch 做 GPU 侧 token 搬运（**通信控制路径放 GPU，而非 MegaScale-Infer/Step-3 的 CPU 侧**）。用"薄适配器"把 AFD 映射到 DeepEP 的对称 EP 契约上：给每个 Attention rank 分配一段 router 永不选中的 dummy expert slot，把真实 expert id 重映射到 FFN rank 的 slot 区间。发送走 FP8、返回走 BF16（约 3 字节/隐藏元素）。
- **拓扑**：M 个 Attention 节点 + N=1 个 FFN 节点；每 GB200 NVL72 节点 4 GPU、每 GPU 1 个 worker。评测覆盖 Qwen3-235B（128 experts / 8 激活）与 MiniMax-M2.5（约 230B / 256 experts / 8 激活）。
- **调度策略**：
  - **megakernel（MegaMoE 风格融合）**：把 token 分组、打包、dispatch/combine 处理、expert GEMM、SwiGLU 激活融进一个 FFN 侧持久内核（persistent kernel），消除逐层/逐 micro-batch 的 launch 开销；FFN 侧内核独占整块 GPU，从 descriptor table 读取每层权重。
  - **microbatch 流水线（mb=2）**：经验上 2 个 microbatch 即可同时隐藏两个方向的通信；mb 更多只徒增小内核开销。
  - **CUDA graph replay**：把整个 decode 路径 capture 进 CUDA graph。
  - **zero-overhead scheduling**：借鉴 SGLang 的零开销调度，coordinator 提前一步准备下一 step 的 plan（经 ZMQ 只传控制命令与采样 token，不传数据路径），让 CPU 控制面不落在 decode 关键路径上。
- **运行位置**：**T_e ≤ T_a**（FFN 侧"隐藏"在 Attention 工作之后），选使 FFN 仍被隐藏的最大 M，而非 MegaScale-Infer 的 T_a ≈ T_e 均衡点——作者论证均衡点正好落在"便宜失败"与"昂贵失败"的分界线上。

### 关键实验数据
- **GB200 NVL72 实测加速（对比调优过的 colocated vLLM + DP/EP + DeepEP + DeepGEMM 基线）**：
  - Qwen3-235B：**1.41x（8K）**、**1.44x（16K）**；
  - MiniMax-M2.5：**1.45x（8K）**、**1.35x（16K）**。
- 加速来源的恒等式拆解：speedup = **batch expansion** × **latency ratio** × **FFN-node tax (M/(M+N))**；实测 batch expansion 为 **1.5x**（Qwen 8K: 96 vs 64 请求/GPU；16K: 48 vs 32；MiniMax 8K: 72 vs 48；16K: 36 vs 24）。
- 消融：MegaMoE 融合使 8 节点 decode-step 延迟降低 **44%（Qwen）/ 42%（MiniMax）**；关闭 cross-step overlap 使 Qwen 4 节点 8K step 从 32.826ms 增至 42.612ms（**23% 吞吐损失**）。
- **Vera Rubin + LPX/LPU 投影（非实测）**：MoE-only 边界下 1.57x–1.75x；若把 dense 也卸载则 2.07x–2.42x（作者注明这是上界设计点，非当前 FastAFD 边界）。

### 与 AFD Plugin 的关系
**FastAFD 是与 afd-plugin 最直接对应的开源实现。** 二者都实现 "Attention worker + FFN worker + M2N dispatch/N2M combine + microbatch 流水线" 的 AFD runtime：
- 概念/术语高度重合：afd-plugin 的 `attention_worker`/`ffn_worker`、`ffn_model_runner`/`attention_model_runner`、`num_attention_ranks`/`num_ffn_ranks`、connector 驱动的 token 跨角色搬运，与 FastAFD 的角色专业化 runtime 一一对应。
- 差异在于**目标硬件与通信层**：FastAFD 面向 GPU 侧 NVLink/NVSwitch（DeepEP 风格、CUDA graph）；afd-plugin 面向 CUDA（`P2pNcclAFDConnector`）与 **Ascend NPU**（`CAMP2pAFDConnector` 同步 / `CAMAsyncAFDConnector` 异步，HCCL/CAMP2P + ACL graph），并支持 DeepSeek V2/V3/V3.2/V4 与 Qwen3 MoE 等模型。
- FastAFD 的 "FFN 侧聚合提升 MoE 利用率"、`T_e ≤ T_a` 运行点、mb=2 等结论，可作为 afd-plugin 调优（尤其 NPU 上 CAMP2P/CAMAsync connector 的 batch 配比与 ubatch 数）的直接参考。
