# 容量口径澄清：工作范围与独立核查

2026-10-16。以本轮用户说明为准，32k/256k 指单 Attention rank 的完整事务 maxBS，不是 DP 合计硬上限。

## 已确认的原始问题

- 有效总体 v3 和 common_contract 将 DP 成员通信容量和限制为32768。
- 实际 DeepEP AFD host 的生产构造与本地接收测试构造各有一处 DP 合计32768检查。
- 当前 dispatchSend kernel 已按自己的 lane.capacity 切轮，task_info.num_tokens 是本地逻辑输入行范围（包括 mask=false 行），不是 mask 的非零计数。
- 当前 dispatchRecv 的当前片受 Q/Q_alloc 约束；其单 lane 的32768限制与 DP 合计限制是不同概念。
- AFD ffnScheduler/combineSend/combineRecv 尚未实现；只能修订其设计，不能声称修改或验证了不存在的设备代码。

## 修订边界

修改有效 v3、五算子总设计、五算子详细设计、必要 host/布局注释、实现使用文档和测试。保留原设计、v2、历史审查/实施/QA/哈希材料；不修改 PPT 预览目录。

保持单槽 C_m 当前实现上界、Q/Q_alloc、UB tile 约束、索引/DMA宽度和真实内存预算；删除未经授权的 DP 合计固定32k限制。协议位数量及生命周期不因聚合 token 增加而改变，较大的工作量通过更多计算片消费，所有成员（含零长度）仍参加每个共同通信轮。

采用本地 ascendc-tiling-design 的容量分层方法及 Ascend C API/代码审查资料；具体 API 能力以目标版本为准。复用已确认的本地环境信息，无已验证的 A5/CANN 编译或运行环境。
