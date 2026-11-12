# F 侧计算事务与整专家组分片

适用路径是 UMDK CAM 合入 #3860、#3861 后的事务模式。通信算子以 UMDK `cam_feature@9179865e` 为基点，owner 和 F 侧 JIT 调度以 DeepEP-Ascend `19fe6b7a` 为基点。原五算子文档描述的逐片 scheduler 仍对应默认模式。启用本模式时，以本文的选批、容量和 F 本地进度约定为准。

## 三个边界

- 一个通信轮对应一个 Attention DP 的一个 microbatch、一个层和一个轮号。DP/phase 下配置的全部逻辑成员必须到齐，包含本 F 零 route 成员。同一 DP 轮的成员总轮数一致。
- 一个 F 计算事务是这张 F 一次有效选批确定的**完整通信轮集合**。沿用 scheduler 的优先级，优先选不同于上次的就绪 phase，其余按首次就绪顺序。每批只选一种 phase，允许多个 DP、MB、轮号以及注册的同构层。每张 F 独立选批。
- 一个计算片含当前事务中的若干完整 `(layer, local_expert)` 组。组内来自所有入选通信轮的 route 在同一片完成。每片依次执行 DR、FFN、CS。

## 本地事务与发布

`ffnPoll()` 在设备读 Ready，输出每 lane 的发现状态、观察位、期待位和 phase。它只读事务所有权。Scheduler 首次接管 Ready 时写 Pending 快照，等齐后一次性冻结 Entry/Member、各 Member 的本地 expert count 和跨层 group count。在同一个本地事务进行中，scheduler 不做第二次有效选批。

`Active` 的旧布局保留为当前片描述；事务模式在该布局尾部追加 32B 事务头、`L` 个 32B Entry、`L` 个 32B Member、对齐的 `uint32[L,R]` route count 和 `uint32[NL,R]` group count。每个固定 Entry 记录层槽、剩余行数、grant 和退役状态。Header mode 和混层 sentinel 由新旧 F 侧实现区分。F 侧 owner、调度器、DR 和 CS 必须使用匹配版本，A/F wire Inbox、Ledger、返回地址和位协议沿用原布局。

DR 从已提交的 group 游标开始按层槽、expert 升序装片。组整体放得下才加入，目的行按计算组对齐。源 route interval 仍按真实 lane 和本地 expert 校验。DR 构造片内 Entry、Member、Segment 并写原四列来源 metadata。计算端按层槽选择权重，W2 维持 DR 的行序，整个片完成后才调用 `computeReady()`。同流提交 DR/计算/CS；跨流接入需要由框架给这三个阶段建立明确的完成依赖。

CS 先检查片与固定事务快照一致，排空该片回包写入，再提交剩余计数和退役状态。某轮在本 F 剩余行归零时，CS 给该轮的全部成员依次发 Done、Consumed；后续片跳过已退役 Entry，即使 A 已在该 lane 发布新轮也只会被下一次选批接纳。CS 统一推进 group 游标，全部入选轮完成时发布事务结束。失败不推进到下片，框架按失败通信域处理。

## 容量和后端

首版 `Q` 是每片有效行硬上限，`Q_alloc` 是补齐后物理行硬上限。单 token 同一 expert 的 TopK 路由唯一，因此 `max_phase Σ C_lane` 是本实现保守的单组行数上界。所有注册 lane 可能进入同一 phase 的事务时一并计入。启动检查 `Q >= bound`、`Q_alloc >= align_up(bound, alignment)`、`Q_alloc <= backend_max_rows`，同时校验预分配预算。配置不足时明确返回容量错误。单组实际计数也在选批和装片时复核。

4A4F、每张 A 一个 C=32K 的槽且同 phase 时，保守单组上界为 128K。该数字是正确性所需的行容量，并未证明目标 GMM 支持 128K。框架须提交已核实的后端行数限制，并保证 backend 能处理当前注册层和 expert 组合的分组数、dtype、对齐和 workspace。组数或物理容量不足时，应选择更小 C/更少同 phase lane，或者引入有容量约束的整轮选批策略。任一组仍须保持完整。

## 验证

CPU 模拟使用独立的场景生成器和断言。诊断 runner 使用真实设备 poll、scheduler、CAM DR 和 CS 入口，按本地事务号记录 `poll`、`scheduler`、`dispatch_recv`、`gmm`、`combine_send`、逐轮确认和结束。BF16 小形状的 grouped GEMM 诊断使用层相关权重以检测串层。CANN/Bisheng 编译、A5 四/八卡 transport、目标 FFN 后端可用形状和图模式仍须按实际环境验证。CPU 场景及 host trace 不作为这些能力的设备证明。
