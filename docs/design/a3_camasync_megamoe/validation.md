# A3 CAMAsync MegaMoE 测试与验收设计

状态：完整验收计划，**设备验收尚未执行**。首版已生成 CPU、binding、合成精度/性能、控制与 E2E 入口；当前可运行命令和已执行结果见 [实施说明](implementation.md)。本文件与[总体设计](README.md)、[算子与协议接口](operators.md)、[框架接入](integration.md)共同定义验收范围，尚未实现的完整 payload 回放及 litmus 要求仍保留，不能视作已有结果。

## 1. 基线、场景和证据边界

所有现有 afd-plugin 测试引用固定在 `587e1a9b21c86c467d99b551fa2b5ddd61c9212e`，即 `upstream/release/v0.26.0rc2`。本次文档所在 main 的 0.30 实现和运行记录不能代替该基线。原生 MegaMoE 对照固定 ops-transformer `8d46ac50deabb544f59f56f5b82a995159907c03` 的 arch22 W4A8 路径。

| 项目 | 首版验收配置 |
| --- | --- |
| 设备与软件 | A3/910C；记录实际 SKU、CANN、HCCL、驱动、固件、PyTorch、torch-npu、vLLM、vLLM-Ascend 和编译器版本 |
| 模型 | 用户当前 DeepSeek V4 Flash **per-channel W4A8** checkpoint，BF16 模型输入；保留实际 compensation 和激活限制 |
| 默认部署 | A DP2/TP4，F EP8/TP1，共 16 个进程；每 DP MBT=8192；PCP/DCP=1 |
| 工作项 | 一个完整 Attention DP、一个 layer、一个 microbatch；F 独立选择 DP，无逐任务 F-F collective |
| 执行方式 | prefill、eager；图模式、decode、其他量化格式不计入首版通过范围 |
| 内存与分轮 | 同 DP `CR(MB0) -> DS(MB1)` 复用同一物理槽；完整 MB 输入/返回窗常驻；每个 1024-token tile 内按 Q 路由行做 F workspace 分轮 |
| 第一层通信分轮 | 仅允许默认单通信轮；非默认接口值在发布 Ready 前拒绝，不静默回退 |

H、I、E、K、MoE 层列表、权重逻辑 shape/物理 NZ shape、scale dtype/编码、compensation、`swiglu_limit`、输入容量均从实际 checkpoint 及加载后的参数读取并记录，不能套用名称相似模型的维度。MBT=8192 是默认调度容量，不保证每个 microbatch 有 4096 个 token，也不保证每个 TP rank 实际有 1024 个 token。

以下既有材料可以复用，但各自只证明其原有范围。行号均属于上述 0.26 commit，可用 `git show 587e1a9b:<path>` 核对。

| 既有文件与行号 | 可复用内容 | 不能据此宣称的结论 |
| --- | --- | --- |
| `tests/npu/async_cam_precision.py:40-89`、`async_cam_common.py:539` | 多 rank 启动、独立元数据检查、逐阶段报告 | 未覆盖拟新增四算子或真实 W4A8 FFN |
| `tests/npu/async_cam_common.py:1272-1344` | shape/dtype、非有限值拒绝、逐元素误差和首个差异报告 | CPU golden 通过不等于设备同步正确 |
| `tests/npu/async_cam_performance.py:183-230,268-279` | 先精度预检，再 warmup；逐 iteration 取 rank 最大值 | 现有 synthetic expert transform 不等于 MegaMoE FFN 性能 |
| `tests/npu/test_async_cam_layered_w4a8.py:18-115` | W4 打包/NZ、per-channel scale、非零 compensation、乱序层选择、尾部扰动的构造思路 | 小维度单卡 GMM 不覆盖分布式 Ready/Done、完整模型、非零激活限制 |
| `tests/e2e/models/deepseek_v4_flash/config.py:16-44,349-365` | DP2TP4/EP8、MBT8192、两 ubatch 和模型部署配置 | 部署形状相同不等于权重类型和精度已通过 |
| `tests/e2e/models/deepseek_v4_flash/test_async_cam_npu.py:22-94` | 16 设备启动、环境、退出及结果归档 | 不能直接改 backend 名称就省略 W4A8 预检 |
| `tests/e2e/models/deepseek_v4_flash/completions.py:23-63,117-135` | 并发 HTTP 请求、回答和终止原因检查 | 十个加法请求不建立通用模型精度，也不是 prefill 性能基准 |
| `tests/e2e/README.md:152-189` | 当前 async 场景的配置与限制 | 文档明确使用 **W8A8**，不是本次 W4A8 的既有验收证据 |

## 2. 测试分层与文件归属

先覆盖算子 seam，再覆盖设备协议和完整模型。测试通过记录必须明确标注 `CPU contract`、`compiled meta`、`A3 numerical`、`multi-rank protocol` 或 `E2E`，不合并成含义不明的“测试通过”。

| 层级 | 计划位置 | 内容与 owner |
| --- | --- | --- |
| CPU connector seam | `tests/unit/connectors/npu/test_async_cam_megamoe.py` | 镜像拟新增 `afd_plugin/connectors/npu/async_cam_megamoe.py`；配置、公开 payload/descriptor、生命周期错误；归 connector contracts 的现有 `tests/unit/connectors/**` |
| CPU op adapter seam | `tests/unit/compat/npu/test_cam_async_megamoe.py` | 镜像 `afd_plugin/compat/npu/cam_async_megamoe.py`；形状、dtype、容量和格式拒绝；归 execution platforms |
| CPU layered executor seam | `tests/unit/v1/worker/npu/test_async_cam_megamoe.py` | 镜像 `afd_plugin/v1/worker/npu/async_cam_megamoe.py`；参数表合法性、异构层拒绝、启动失败；归 FFN runtime |
| 构建/Meta | 复用 `tests/unit/package/test_ascend_build_files.py`；设备扩展 Meta 检查放 `tests/npu/` | build manifest 可按文本检查；算子 schema、输出和容量须调用实际接口 |
| A3 单算子/协议精度 | 拟新增 `tests/npu/a3_camasync_megamoe_precision.py`、共享 `a3_camasync_megamoe_common.py` | 一张参数表覆盖四算子与完整轮转；真实 NPU、真实 HCCL window；归 execution platforms，与 connector/FFN owner 联合审阅 |
| A3 性能 | 拟新增 `tests/npu/a3_camasync_megamoe_performance.py` | 共享相同 payload/权重/路由及精度预检，不复制一套 golden |
| 模型 E2E | 扩展 `tests/e2e/models/deepseek_v4_flash/{config.py,test_async_cam_npu.py,completions.py}` | 新场景 `afd-dsv4-flash-async-cam-megamoe-dp2tp4-ep8`，复用模型部署和请求 fixture；归 E2E testing/model integration |

0.26 的 `execution_platforms.md`、`ffn_runtime.md` 没有覆盖上表全部新路径。实施 PR 必须先同步登记相应模块的 `validation_paths` 和实际 owner，再增加测试；本设计不把“计划归属”伪装成已有登记。E2E 性能 scenario 及 prefill 数值采样是计划新增能力，0.26 没有独立的 DSV4 性能 fixture 可直接引用。

遵守仓库测试约束：CPU-only 主机不安装 torch/vLLM 也能收集 `tests/unit`；需要运行时的测试在导入被测模块前 `pytest.importorskip`。共享 fake 按上游概念放 `tests/unit/fakes/`。断言公开返回值、协议消息、配置结果和最终状态，不断言私有属性、内部调用参数或次数；不读取生产 Python 源码作字符串断言。不能用 fake 模拟 MTE、核间同步或 NPU 计算后称为设备证据。E2E harness 自身通过执行 E2E 验证，不在 `tests/unit` 再写 harness mock 测试。

## 3. Golden 与统一判定

### 3.1 三种独立参考

1. **路由参考**：根据原始 `(source_rank, token_id, topk_slot, global_expert)` 构造整数表，不调用待测 kernel 的 prefix/round-table helper。每个路由拥有唯一标识，检查 count、完整专家前缀、tile 内 source 行、F 合并行和 Q 轮内行之间的转换。
2. **数学参考**：独立解包 signed INT4、还原 per-channel scale，按真实量化规则重建激活和两次 GEMM、SwiGLU、截断、再量化，最后按原 top-k 次序加权归并。以原始 signed 权重构造数学参考时，不额外重复加 native 编码的补偿项。kernel 输入仍须带正确、通常非零的 W1/W2 compensation。
3. **设备对照**：同一个冻结的模型层 payload 对比现有 CAMAsync W4A8 小算子链、候选 MegaMoE 链；在原生 MegaMoE 已独立验证的范围内增加 native 对照。原生多 source 多轮 offset 的疑点未澄清前，不能将其输出作为唯一真值。

三者不互相替代。路由数据搬运必须精确；量化算术误差必须按阶段解释；真实模型对照用于验证加载布局和模型集成。

### 3.2 有限值、精度和失败报告

- 任何有效输出或参考出现 NaN/Inf，立即失败并保存对应层、专家、source、token、top-k、tile、round。禁止 `nan_to_num`、`equal_nan=True`、忽略非有限行或以零填补失败参考。
- integer count、offset、sequence、ticket、row mapping、输入搬运 INT8 数据要求精确相等；仅搬运的浮点 scale 按位比较。尾部容量行不参与输出精度统计，但通过独立 poison/guard 检查证明没有读取或写越界。
- 浮点计算检查有效行的逐元素误差、最大绝对误差、相对 L2、超差比例和首个差异坐标。零范数 reference 独立处理，不能以除零生成 NaN 再漏判。
- 现有小维度 layered W4A8 测试的 `rtol=0.04, atol=0.05`（`test_async_cam_layered_w4a8.py:106-115`）仅作为该合成构造的初始对照，不自动升级为所有 checkpoint/输出尺度的验收阈值。
- **待评审建议**：对固定合成构造先要求上述逐元素阈值、零非有限值、零路由错误；对真实 checkpoint 先测旧链相对独立 golden 的误差，再在运行候选前冻结每阶段容差。候选相对同一 golden 的相对 L2 不高于旧链误差的 1.05 倍加明确记录的浮点数值底限；逐元素阈值也须同时满足。若旧链本身不满足 golden，应先定位，不能用旧链错误为新链放宽容差。
- 额外构造 NaN/Inf 输入和损坏 scale，验证检查器确实失败；这些负例不属于合法性能输入。不要求通过数值比较掩盖底层非法数据。

## 4. 四算子单独验证

| 算子 | 可观察契约与必测用例 |
| --- | --- |
| `AfdMegaMoeDispatchSend` | BF16 输入按 native 规则动态量化、top-k 复制、全局专家排序；完整 MB 的各 1024 tile 写不同偏移；source 的 MB token 数为 1/1023/1024/1025/2049，末 tile 为非整块；counts 总和等于有效 token×K；source rowIdx 与原 top-k slot 可逆；只在全部输入、scale、counts/descriptor 完成后发布一次任务 Ready |
| `AfdMegaMoeScheduler` | 仅完整 DP 的全部 TP 成员到齐且 `layer/seq/mb` 一致才能返回 WORK；定长 `work_desc[32]` int64 与 source descriptors 无 host `.item()`；轮转公平；跳过已完成 sequence；缺成员保持等待；STOP/ERROR 按接口终止，不能返回虚假 WORK |
| `AfdMegaMoeFfn` | 设备选择层的全部 W1/W2/scale/compensation 指针；native W4 MSD/GMM/激活/GMM2/回写；同一任务内热点 expert 跨 Q、多个 source/多个 tile；正确 source offset；零行路径以及非法层路径不进入参与核不一致的混核同步；最终输出落位后才 Done |
| `AfdMegaMoeCombineRecv` | 等固定全部 F 的匹配 ticket/seq Done，包括零路由 F；用 dispatch 保存的 rowIdx/top-k weights 恢复原 A token 顺序；只读有效路由，不读上一 MB 或尾 tile 残留；每 task 只完成一次，完成后允许下一 MB 共用通信区 |

单测 setup 可以准备真实通信窗口和合法 descriptor，但不能绕过正在测试的发布/消费阶段，先人工写 Ready 再称 DS 发布时序已验证。对 Scheduler/Ffn 的独立测试可通过定义明确的生产 ABI 初始化状态，并与四算子串联测试互补。

### 4.1 路由与分轮参数表

设 F 某 tile 的有效路由行数为 R。Q 用当前 tiling 允许的对齐值，至少覆盖一个较小 Q 和正常部署 Q；边界变化放在 R/count 中，不强迫 kernel 接受非法 Q。

| 维度 | 必测值/关系 | 主要捕获问题 |
| --- | --- | --- |
| R 相对 Q | 0、1、Q-1、Q、Q+1、2Q-1、2Q、2Q+1 | 轮数、末轮、零行、重复/遗漏 |
| 热点专家 | 一个 expert 为 Q+1、2Q+1；非首 expert 热点；其他 expert 非零 | 旧 whole-expert 限制；误把 merged `lo` 当 source offset |
| source 分布 | 单 source、多个 source；TP 内不等长；某成员零 token | 把 source_idx 当物理 rank、假定 TP 等长 |
| token mask | B>0 全false、非连续true、跨1024边界、上一MB更长且mask不同 | `sum(counts)=sum(mask)*K`；masked输出零、无效行不参与量化/路由、不读旧D；有效全零输入仍须正常处理 |
| source 前缀 | 当前 F 非第一个专家分片；同 source 前序全局 expert 有 token | prefix 必须覆盖全局专家，不能只数本 F 专家 |
| tile | 1024 边界前后、连续三个 tile、末 tile 部分有效；同专家跨 tile | 原生 tile 暂存地址误用于完整 MB 常驻布局 |
| 路由 | 平衡、稀疏、单热点、交错专家、极不均衡 | 零路由 F、count/offset、性能退化 |
| top-k | checkpoint 实际 K；合成支持范围内多 K | top-k 展开行数与最终权重应用恰好一次 |
| 多 MB | 同 DP 两个 MB；不同 layer 同 seq 负例；相同 layer 连续多 seq | 防串批、复用后旧标记、只按层号识别任务 |

每个 `(tile, expert, source)` 段独立校验：`rows=max(0,min(b+c,(r+1)Q)-max(b,rQ))`；`source_row=source_expert_base+(lo-b)`，只在 rows>0 时使用。所有轮的有效路由 ID 的有序并集应等于整批参考，每项恰好一次；输入 pull 和输出 return 必须使用同一 source 坐标。性能代码可用 prefix/紧凑表，golden 用独立路由枚举验证。

### 4.2 W4A8 与选层

采用至少三层互不相同的合法权重和 scale，层号包含非连续值；按 `L2,L0,L2,L1` 顺序执行，先在同 Q、同 shape 下保证只能通过设备 descriptor 选择层，再覆盖不同任务长度。不能用“只改变 W1”来证明所有参数都被选中了：W1、W2、两组 scale 和两组 compensation 都用不同签名；分别构造能放大遗漏补偿的非对称权重与非零行和。

测试 INT4 打包次序、NZ 物理 shape/stride、INT64 scale 编码与 FP32 原始值的一致性，含零权重、正负边界值、输入接近量化边界、较大/较小 scale。设备选择必须保持参数 tensor 生存期，不接受临时指针表指向已释放对象。

`swiglu_limit=0` 与 checkpoint 的非零限制各跑完整数值检查，构造 gate/up 在截断阈值两侧的输入，参考语义取自固定算子及实际模型实现。层间激活参数若为首版同质约束，异质表必须启动拒绝；不能忽略不同值而沿用第一层。漏 compensation、换错 scale、把 signed INT4 当 unsigned、交换 gate/up 的故障注入都必须使检测失败。

## 5. 真实多 rank 协议与生命周期

协议测试先用较小合法维度隔离内存和顺序问题，再用真实 checkpoint H/I/E/K 验证容量、对齐与 tiling。默认验收拓扑为 8A+8F；可用更小合法拓扑定位问题，但不能代替该拓扑通过。

### 5.1 Ready/Done 发布可见性

每个发送者独占的 512B 记录由 body 与最终 32bit 非零 ticket 组成，body 内含 64bit task sequence。测试不假定远端 64bit 原子提交：消费者只在合法 commit ticket 后读取、校验 body 身份。必须验证数据、counts 和 descriptor 的远端可见性先于 commit；内存模型正确性不能由一次数值 roundtrip 推断。

专门运行发布 litmus：每个 epoch 给 body、payload、count、scale 设置可追溯的不同模式；提高重复次数、改变合法参与 rank/数据量、制造受控的发布阶段交错；消费者一旦接受 ticket，立即检查整份对应内容。Ready 与 Done 两个方向分别执行。保存逐 rank 首个错误，不只保存总 PASS。重复读取旧 ticket、body 更新过程中读取、其他发送者相邻记录写入也要覆盖；这些是压力证据，仍需代码级 fence/cache 操作审查。

ticket 在域重建前禁止绕回。测试将 sequence/ticket 初始化到合法边界附近，确认到达上限后公开报错并要求重建通信域，而不是回到 0/1；错误发生时不得发布半个新任务。域重建必须初始化窗口及本地 `completed_seq[DP]`，旧域的 Ready/Done 不能在新域被接受。

### 5.2 调度、空任务、复用

| 用例 | 通过条件 |
| --- | --- |
| 完整 DP 与缺成员 | 其他 DP 已完整时可以推进；缺一个 TP Ready 的 DP 不能开始计算 |
| TP 内身份不一致 | 已提交当前expected ticket的layer/seq/mb/source membership矛盾必须ERROR，不能等descriptor自愈；缺成员或旧ticket则继续扫描其他DP，绝不拼成一个WORK |
| F 异序选择 DP | 预先发布两个完整 DP，使用合法初始轮转状态让 F0 先 DP0、F1 先 DP1；各自完成、无 F-F 同步等待，结果按 DP/seq 正确归还 |
| 持续繁忙 DP | 持续有工作的一组不能饿死已 ready 的另一组；用受控任务序列和选择记录检查轮转，不用 CPU 单测 sleep/墙钟断言 |
| 零路由 F | A 已 Ready、本 F R=0；F 必须完成零行分支并发送匹配 Done；A 不因少一个 F 永久等待 |
| 部分 TP 零 token | 该成员仍发相同身份的零 count Ready；F 不取不存在的 tile；零长度返回按接口正常完成 |
| 整 DP 零 token | 验证明确的空任务路径：全核一致跳过计算和计算同步，仍正确完成协议；不能让部分 AIC/AIV 进入等待 |
| 非法 layer/损坏参数表 | host 能判断的错误在 launch 前拒绝；设备 descriptor 错误走统一 ERROR 路径，所有参与核一致退出，不发成功 Done |
| 同 DP 两 MB 共享槽 | MB0 CR 完成前无 MB1 DS 覆写；完成后用完全不同 payload 重写原地址，确认无旧结果、旧 rowIdx、旧 count、旧 ticket 污染 |
| 不同 F 的轮数不同 | 同任务各 F 的 R 和 Q 轮数可不同；只有各自最后 tile/round 完成后 Done，A 收齐后合并 |
| 中间轮未完成 | 受控停在中间轮，A 不提前完成 CR；F 本地 scratch 可按依赖复用，完整 A X/D 槽不可释放 |
| 重复调度 | F 最终 Done 后才更新 `completed_seq[DP]`；相同 Ready 在 A 尚未发下一项时不会再次返回 WORK |

延迟/交错通过独立测试控制通道和设备事件安排，不能靠不确定 sleep 决定正确性。为避免错误 kernel 无限轮询拖住整套测试，每个危险负例放独立多进程作业，外层有有界超时和日志；超时只用于判失败，不用于证明正确的事件顺序。

### 5.3 通知与退出

保留两类同步的区别：每 MB 每 A→F 一次 Ready、每 F→A 一次最终 Done；F 内每 expert/计算块/轮次的 AIV-AIC 握手、MTE 完成事件和 workspace 复用同步仍需保留。将热点 expert 切成多轮不引入每轮 A↔F ACK。通过设备 trace 和协议记录检查外部通知的身份/完成边界，不以内部 helper 调用次数作 CPU 单测。

Scheduler 在 WORK/STOP/ERROR 之一出现前可以阻塞。退出测试必须覆盖：空闲 Scheduler、已选中任务、正在中间轮、部分 A 已失败。**排在同一条被阻塞 stream 后面的 STOP tensor 写入不能唤醒它**；必须验证实现采用独立可见控制路径或通信域 abort，且不会提前释放仍在使用的窗口、权重、graph/stream 资源。确认最终设备完成后才能报告正常退出。进程强制退出和正常 drain 在报告中分别记录。

## 6. DeepSeek V4 模型精度与 prefill E2E

### 6.1 重建 0.26 W4A8 基线

先在同一个 0.26 软件栈、同一 W4A8 checkpoint、同一 16 卡拓扑运行原 CAMAsync 小算子链。记录实际量化配置与加载后的参数表，确认并非 W8 路径；没有这份基线，不把既有 README 的 W8 成绩改名为 W4。随后唯一切换 `additional_config.afd.connector="cam_async_megamoe"` 及其必要容量配置，其余模型、tokenizer、路由、批次和并发设置保持一致。

沿用现有 DSV4 模型 fixture 的服务启动、设备分配、并发请求、日志归档和清理。新增 prefill fixture 固定 prompt token IDs、请求顺序和 batch manifest，至少包括短 prompt、1024 边界、接近 MBT 的长 prompt，以及 DP 不均衡和两 ubatch。不要以随机请求每次恰好分到相同 DP 为前提；记录实际每 DP/TP/MB token 布局，并在算子回放中冻结该布局。

### 6.2 两类精度证据

- **完整 prefill 数值对照**：从实际 W4A8 checkpoint 采集多个 MoE 层的输入、top-k IDs/weights、输出，以及最终 prefill logits；冻结 payload，逐层对比旧链/新链和独立参考。至少覆盖首个/中间/末个 routed MoE 层、同层重入、非零 `swiglu_limit`。采样关闭后再次运行，确认采样不改变模型路径。
- **HTTP prefill/首 token 验收**：新增 `test_deepseek_v4_flash_async_cam_prefill`，所有请求设置 `max_tokens=1`，使用固定prompt/tokenizer，检查首token/logits与参考及合法终止原因，不要求生成完整加法答案。单token prefill合法，阶段不能仅按B>1判断。decode/mixed batch作为准入负例，在发布Ready前拒绝。原十并发完整加法和256 output-token用例继续作为旧backend回归，不计入新backend验收；首版不新增协议回退。

真实模型精度报告同时列原链、新链、差值与有限值统计，明确是 logits、逐层激活还是任务准确率。若增加 GSM8K/固定评测集，要求相同数据版本、prompt、采样种子和 token budget，另列置信区间；其通过也不能掩盖路由映射或协议错误。

## 7. 性能测量与公平比较

所有性能运行先通过同 payload 的非计时精度预检。关闭逐 token 日志、D2H 数值采样和精度检查；warmup、编译、权重加载、通信域建立不计入稳态核时间，但单独报告启动成本。记录 Q、每 tile/round 的有效行数、routing 直方图、峰值 HBM、窗口大小与返回缓存占用。

### 7.1 分别回答三类问题

| 对照 | 固定条件与测量边界 | 允许的结论 |
| --- | --- | --- |
| native 热路径 vs 新 Ffn 热路径 | 相同 A3 F 资源、相同已量化 source payload、权重、counts/offset、Q、tile、实际 DMA 字节和 ready 输入；匹配原生 W4阶段范围；剔除调度等待而保留实际计算/搬运 | 原生计算/搬运逻辑在异步适配后是否退化 |
| 原 CAMAsync 整链 vs MegaMoE CAMAsync 整链 | 相同 8A+8F、同一个 W4层输入及路由、同一 Q/内存上限策略；包括 DS、Scheduler/DR、全部计算轮和 CR 的完成 | 新四算子整链的延迟/吞吐与内存收益 |
| DSV4 prefill E2E | 相同 checkpoint、16卡预算、请求 token manifest、负载、并发、batch策略、长度分布与预热；分别记录原链/新链 | prefill 服务级吞吐、TTFT 和尾延迟变化 |

不能将 **16 rank A+F 的整链耗时**直接除以**8 rank 原生 MegaMoE 的耗时**称为公平加速比。若 native 全算子包含不同 count allgather、通信拓扑或不同 source 数，先列差异，仅作上下文数据；热路径对比应使用同范围的 profiling 分段或专门的同 payload 对照 harness。原生源偏移风险未被验证的多轮 case，不用于热路径性能结论。

### 7.2 采样与报告

建议起点为每配置 warmup 20 次、测量 100 次、独立重启 3 组，顺序交错执行 baseline/candidate；若运行时间过长，可在执行前确定更小样本数并记录原因，不能看到结果后选择性删样本。使用设备事件/设备 profiler 测核时间；CPU enqueue 时间、同步等待、端到端完成时间分开报告。

跨 rank 指标先按同一次 iteration 取关键路径最大值，再计算 median/p95/p99；不能平均各 rank 的 p99 或混合不同轮数的样本。Scheduler 等待就绪的时间单列，不与“输入已 ready 的选择开销”混用。偏斜负载同时报告各 F work/idle 时间和实际路由行，避免仅报均衡 case。

E2E 用完成的真实输入 token 计 prefill tokens/s，列 TTFT p50/p95/p99、成功率、失败/超时数量、实际 DP 分配及总请求数。短输出或首 token 场景用于隔离 prefill；完整 decode 的整体 tokens/s 不能代替 prefill 指标。没有明确支持的回退路径时，不运行并宣传 decode 数据。

### 7.3 待评审的性能门槛

以下是建议门槛，**不是实测收益或已承诺指标**；在目标 A3 的稳定性噪声测定后、候选性能运行前冻结。

- 相同有效计算/搬运范围的 Ffn 热路径：关键配置 median 不回退超过 5%，p95 不回退超过 10%；超阈值应定位到 load、MSD、GMM、epilogue 或同步，而不是用整体均值抵消。
- 同拓扑 W4整链：主 prefill shape 的 median/p95 不退化；以 median 至少降低 10% 作为优化目标，达不到时保留真实测量与原因，不宣布性能目标完成。
- E2E：预先固定的默认/高并发两组负载中，prefill tokens/s 不低于旧链 97%，TTFT p95 不高于旧链 105%，零新增数值/协议错误。至少一组主业务负载达到可重复的净收益，才建议默认启用。
- 峰值 HBM 不超过声明的容量预算；Q 下降应降低 F 私有计算 workspace，而非谎报 A 整 MB X/D 窗也随 Q 下降。窗口不足须启动拒绝，不能靠运行期越界换性能。

## 8. 计划执行顺序与命令

以下命令在实施完成后的 **Linux A3 0.26 环境**执行，本次文档编写没有运行它们。先确认实际编译选项与四个 op 名称，不能因为命令退出成功但所有 NPU 用例 skip 就判通过。

现有基线检查命令可以直接用于对应旧算子范围：

```bash
AFD_RUN_ASCEND_OP_RUNTIME=1 SOC_VERSION=910c \
  python -m pytest -q tests/npu/test_async_cam_layered_w4a8.py

torchrun --standalone --nproc-per-node=4 \
  tests/npu/async_cam_precision.py \
  --dtype bf16 --dynamic-quant 1 --capacity 8 --iterations 3

torchrun --standalone --nproc-per-node=4 \
  tests/npu/async_cam_performance.py --measure-mode cycle \
  --warmup 5 --iterations 50 --dtype bf16 --dynamic-quant 1 \
  --output .cache/async-cam-baseline-cycle.json
```

上述 precision/performance 使用现有 synthetic transform，**不是 W4A8 MegaMoE 或模型验收**。真实旧链 W4A8 baseline 另按第 6 节构造。

下面保留评审时的**拟定 CLI 契约**，其中 `--checkpoint`、`cases/` manifest 和完整协议回放尚未实现。当前合成测试使用 `manifests/`，实际 E2E 参数化节点与这些草案不同；运行首版请使用 [实施说明第 4 节](implementation.md#4-构建与已生成的测试入口)。

```bash
# CPU-only 环境：收集必须成功；运行时测试可有明确依赖 skip。
python -m pytest --collect-only -q tests/unit
python -m pytest -q \
  tests/unit/connectors/npu/test_async_cam_megamoe.py \
  tests/unit/compat/npu/test_cam_async_megamoe.py \
  tests/unit/v1/worker/npu/test_async_cam_megamoe.py

# 两个 manifest 固定拓扑、payload种子、Q、checkpoint层与用例集合。
torchrun --standalone --nproc-per-node=16 \
  tests/npu/a3_camasync_megamoe_precision.py \
  --manifest tests/npu/cases/a3_camasync_megamoe_protocol.json \
  --output output/a3-camasync-megamoe/protocol.json

torchrun --standalone --nproc-per-node=16 \
  tests/npu/a3_camasync_megamoe_precision.py \
  --manifest tests/npu/cases/a3_camasync_megamoe_w4a8.json \
  --checkpoint "$AFD_NPU_E2E_MODEL" \
  --output output/a3-camasync-megamoe/precision.json

torchrun --standalone --nproc-per-node=16 \
  tests/npu/a3_camasync_megamoe_performance.py \
  --manifest tests/npu/cases/a3_camasync_megamoe_w4a8.json \
  --checkpoint "$AFD_NPU_E2E_MODEL" --warmup 20 --iterations 100 \
  --output output/a3-camasync-megamoe/performance.json

# 计划在现有DSV4文件中增加首token测试函数和该scenario；max_tokens=1。
export AFD_E2E_BACKEND=npu
export AFD_E2E_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export AFD_NPU_E2E_MODEL=/path/to/verified-per-channel-w4a8-checkpoint
export HCCL_IF_IP=<local-communication-ip>
export HCCL_SOCKET_IFNAME=<communication-interface>
python -m pytest -q -s \
  'tests/e2e/models/deepseek_v4_flash/test_async_cam_npu.py::test_deepseek_v4_flash_async_cam_prefill[afd-dsv4-flash-async-cam-megamoe-dp2tp4-ep8]'
```

新增 manifests 与共享 harness 同属 NPU validation paths，实施时一起登记；配置中不能把绝对 checkpoint 路径、凭据或采样的用户请求提交进仓库。E2E 性能使用扩展后的模型 fixture 启动服务，再以固定 token manifest 运行负载；其具体入口在 E2E 实施 PR 中定义，不能假装 0.26 已有此命令。

推荐门禁顺序为：CPU/Meta → A3 单算子数值 → 多 rank 发布 litmus/协议 → W4A8 层回放 → 0.26 模型 prefill → 同 payload 性能 → E2E 性能。任一正确性门禁失败，后续性能结果只可用于诊断，不计为验收。

## 9. 交付记录

每次设备运行保存：两个源码 SHA、patch/编译参数、设备及软件版本、checkpoint 标识和参数摘要、完整命令/环境白名单、case manifest/seed、每 rank 原始结果、数值误差、trace、原始计时样本、峰值内存、失败与清理状态。敏感模型输入按用户已有数据管理约束保存，不在终端输出完整 token/权重。

最终验收表逐项填 `未执行 / 通过 / 失败 / 不在首版范围`，至少包含编译、四算子精度、sourceOffset、多 Q、多 tile、选层、补偿/截断、Ready/Done litmus、零行、异序 DP、复用、序号边界、shutdown、模型 prefill 和两类性能对照。失败、skip、仅 CPU 通过和未验证项不能汇总成“全部通过”。
