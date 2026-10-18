# 剩余算子源码交付记录

2026-10-16。实现仓D:/Code/refs/DeepEP-Ascend，基线3b25377d04b24fc6154698ded78a2bcb2c59afff。

沿用此前用户授权的CANNBot角色适配：设计补全→代码/测试→独立交叉检视→集成修正。现有工程和JIT保持DeepEP组织，不另建ascend-kernel工程。已读取operator-dev/code-gen/compile-debug技能及本地ops-direct-invoke角色资料；因当前无已确认CANN/A5环境，不将通用技能七阶段门禁标记为全通过。

## 本次修改

- 新增ffn_scheduler.hpp及私有scheduler.hpp：DP到齐、Pending、两种credit策略、有界切片、Completion一次消费。
- 新增combine_send.hpp：原生HcommJetty/SQ/CQ、直发GMM2输出、双stage metadata、32请求批提交及末片提交。
- 新增combine_recv.hpp：按原生token分核和stage搬运方式归并，必要新增逐route FP32加权、零轮/credit/Ledger协议。
- 新增return.hpp：Context预留offset指向32B返回目录，旧224B Context/Active/dispatch ABI不改。
- 新增remaining.hpp；扩展AFDBuffer图外准备、预算/UB校验、直接输出注册及五算子绑定。
- 新增三个CPU测试文件、真实五算子A5入口、实现说明；细化scheduler设计和已实施状态。
- 不改原生EP算子，不覆盖dispatch历史审查记录，不修改PPT目录。

## 交叉检查后的修正

1. combineSend必须接受ARMED，不能误要求EMPTY。
2. Ready/credit按独立32B DMA读取，Inbox按64B快照。
3. combineRecv动态task显式bypass快照；mask32token分块读取，尾部精确长度。
4. prepare_recv要求Q_alloc≥max(Q,a)，保证非空片能前进。
5. prepare_combine_recv限制2NC≤可用AIV；仅为必要条件，仍需验证实际并发驻留和SyncAll隔离。
6. Completion在ACK前完整复制；scheduler对照旧Active验证全部delta后再消费。

## 验证

- 本地unittest全量51/51通过（原33+新增18）。只证明CPU规格、数值模型和部分真实ABI/源码契约。
- tests/afd共19个Python文件AST通过。
- git diff --check通过（新建未跟踪文件不等于已做C++编译）。
- host/JIT C++编译：未执行。
- A5 kernel、实际网络、图replay、性能：未执行。

run_pipeline_a5.py覆盖真实send→scheduler→recv→identity expert→combineSend→combineRecv，可选2/4rank、多轮、Q切片、短成员零尾、全mask与PRECHECK。没有模拟ACK/credit。identity不是模型GMM；有host观测，不作为图模式或性能测试。

## 仍待平台验收

裸UB LocalTensor、BF16 Cast与FP32原地操作重载、1–31B mask尾DMA、源注册与tokenId0行为、CQ drain覆盖、URMA→UBMEM可见性、A双stream活性及SyncAll隔离、D多槽和跨事务replay、域错误退出、性能。scheduler单AIV匹配最坏O(L³) metadata，是需测量的性能边界。上层GMM须按device Active gate并写注册combine_x；本次没有实现模型GMM适配或故障恢复服务。

源码已交付待设备验证，不是生产验收通过。
