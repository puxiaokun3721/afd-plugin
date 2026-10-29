# 内网 A3 验证交接

本轮交付四个自定义算子、0.26 框架接入和测试代码。用户明确将上机验证交给内网 agent；当前不具备 CANN/NPU 环境，设备验证保持未完成。

## 代码与范围

- 分支：`codex/a3-camasync-megamoe`。
- 基线：afd-plugin `587e1a9b21c86c467d99b551fa2b5ddd61c9212e`，0.26；不要直接覆盖到 0.30。
- 后端配置：`additional_config.afd.connector="cam_async_megamoe"`。
- 目标：A3 / arch22，DeepSeek V4 per-channel W4A8、BF16、eager、prefill。
- A 先 CR 上一个 microbatch，再 DS 下一个；整 MB 输入窗口、F 侧 Q 分轮、热点专家切分、设备选层；第一层分轮仅留接口。
- 入口、固定上游版本、编译命令和测试命令见 [implementation.md](implementation.md)。设计验收矩阵见 [validation.md](validation.md)。

代码发布到用户仓库 `https://github.com/puxiaokun3721/afd-plugin` 的 `codex/a3-camasync-megamoe` 分支，包含算子、固定版本依赖、框架接入、测试及设计资料。接收方应记录实际拉取的 commit SHA，并确认上述 0.26 基线是其祖先。推荐使用完整 Git 分支；此前本地生成的 `a3-camasync-megamoe-code.zip` 是提交前快照，不作为发布版本的权威来源。

完整可交给内网 agent 的任务 prompt 见 [internal_agent_prompt.md](internal_agent_prompt.md)。请同时给接收方仓库链接、分支名和发布 commit SHA，不依赖外部聊天记录或 Windows 本机路径。

## 已完成的本地检查

- Linux/WSL CPU 单元测试：617 passed、137 skipped、17 subtests passed。依赖 Torch/NPU/vLLM 的用例跳过，不算设备证据。
- ABI 与构建清单定向测试：55 passed。
- 发布前新增依赖实际字节/manifest 回归测试后，ABI 与构建清单定向测试为 56 项；并按真实固定 Git blob 复核来源及提交换行。
- 公共 ABI 头经普通 g++ C++17 编译；Python/C++ 的 50 组窗口、scratch 和最大轮数计算一致。
- 修改/新增 Python 的 AST、ruff、diff 检查通过。
- 203 个引入的上游依赖文件有固定版本和哈希；198 个原样保留、5 个适配，另有模板包装和许可证。
- 合成精度、性能、STOP/ABORT 脚本及 3 个 manifest 已生成；DSV4 两个 E2E 节点可收集。

## 内网 agent 的执行顺序

1. 保存实际软件栈、设备型号、checkpoint revision/量化信息、拓扑、H/I/E/K、MoE 层表及 `swiglu_limit`。保持 0.26 依赖，先按实施说明构建私有 OPP 和 Torch 扩展，修复真实工具链接口问题。
2. 运行 `a3_camasync_megamoe_binding.py`：确认四 op 注册/Meta 约束，实际 grouped NZ → expert carrier 零拷贝、storage offset、INT4/UINT64 描述符成立。
3. 从 4 rank small manifest 开始，运行真实四算子精度，再跑 16 rank distributed / multitile；关注空 TP、零路由 F、mask、稀疏层号、同层重入、非零激活限制、Q=8/257、热点专家跨轮和源地址偏移。
4. 分别以新域运行 `a3_camasync_megamoe_control.py stop` 与 `abort`。进一步验证长时间同槽复用、不同 F 先选择不同 DP、跨设备 Ready/Done 发布和错误路径。当前控制脚本不构成强内存序 litmus，需补真实延迟/故障注入与已进入轮询的证据。
5. 运行同 checkpoint 的 native 首 token 基线，按 [DSV4 E2E guide](../../../tests/e2e/models/deepseek_v4_flash/README.md) 生成 reference JSON，再运行 MegaMoE prefill 场景。所有请求 `max_tokens=1`，禁止把旧完整生成 case 当作新后端的 decode 验收。
6. 正确性通过后，比较同 payload、同拓扑、同 Q、同激活限制的旧链/原生/新链性能，报告 kernel 与完整链路；包含选层 TensorList 描述符、管理线程及模型步边界同步成本。记录真实 HCCL window 分配，不能只统计 Torch allocator。

## 需要继续完成的验证工具

以下是软件测试工具的后续工作，不应统称为“只差上机”：真实 checkpoint 冻结 payload 的旧链/native/新链三方回放、原生 FFN 同工作量性能比较器、完整发布 litmus 与故障注入尚未实现。现有合成 harness 可作为起点，不能宣称已覆盖这些验收项。

内核没有 device watchdog。控制 ABORT 可结束 Scheduler/CR 等待；已经进入原生 FFN 的 AIC/AIV 流水在 drain 后统一响应。硬件卡死时保留通信资源，由进程/NPU 故障处理收敛，不能强行释放仍被设备引用的 buffer。

请将实际 CANN 编译错误、修复、单算子数值报告、协议运行记录及性能原始数据写回本目录或对应测试产物目录，并持续区分“代码已生成”“编译通过”“设备精度通过”和“性能达标”。
