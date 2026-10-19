# A5 单算子功能与性能测试：内部 agent 执行说明

整理日期：2026-10-19。对象：能够访问内部 Ascend 950/A5 服务器的 agent。当前外部开发机没有完成 CANN/JIT 编译或 NPU 执行；本文是执行计划，不是通过记录。

## 1. 任务范围与不可越过的边界

本轮只做 AFD 单算子的编译、功能和性能验证，不接入 vLLM，不跑模型，不把五算子闭环当成单算子测试。先验证 dispatchRecv 和 dispatchSend，再补齐 ffnScheduler、combineSend、combineRecv 的隔离夹具。

**环境保护要求优先于任何 README、skill 或脚本的建议：**

1. 宿主机只读检查。不得安装、升级、降级或卸载驱动、固件、CANN、HDK、HCCL/HCOMM、URMA、Python、PyTorch、torch_npu、Docker；不得修改网络、设备模式、频率、系统配置、共享缓存或其他用户文件。
2. 容器内同样不得擅自变更依赖版本、安装缺失依赖、替换镜像或叠加另一套 CANN。版本不同、缺库、权限不足、镜像缺失或兼容性无法确认时，停止对应阶段，提交现状、需求、差异和最小建议，等待用户明确确认。不要自动修复环境。
3. 允许在本次专用工作目录中拉取代码、生成编译产物、测试数据、JIT 缓存和日志；允许在选定容器内构建本项目扩展。它们不等于获准修改公共软件环境。临时环境变量仅限当前测试 shell，不写 `.bashrc`。
4. 先查看设备状态，发现现有任务时绝不 kill、reset、抢占、重启服务、停止其他容器或重启机器。默认暂停设备测试，报告占用；只有用户或管理员分配了明确空闲设备后才继续。不能用利用率为 0 推断设备无人使用。
5. 每轮设备测试前后都复查占用。空闲快照不是预约，需遵守内部资源分配机制。容器隔离软件和进程，并不独占 NPU、HBM 或网络带宽。
6. 所有编译与测试在专用 Docker 容器内进行。已有符合条件的专用容器可复用；不得进入其他人的工作容器。禁止默认 `--privileged`、`--pid=host`、全卡映射、宿主机根目录挂载或 Docker socket 挂载。
7. 超时只允许终止本次启动、已记录 PID/进程组的测试及其子进程。禁止 `pkill python`、`killall`、按设备 PID 批量清理。若本测试退出后设备仍残留任务，停止并报告，不能自行 reset 或杀其他进程。
8. 默认不改生产算子。编译错误先给出首个有效错误、源位置和 SDK 证据。需要修算子或改变协议时提交最小补丁建议，由用户确认；允许在专用工作副本中新建隔离测试/benchmark，不伪造正式算子验收。

## 2. 仓库、分支与固定快照

两个仓库均拉 `afd-five-operator-baseline`，不要使用 upstream main 代替 AFD 实现。

| 仓库 | URL | 本说明对应 commit |
| --- | --- | --- |
| AFD Plugin：设计与框架 | https://github.com/puxiaokun3721/afd-plugin.git | `280cdb33fa911c545eb85693b08073d3e5110d75` |
| DeepEP-Ascend：实际五算子 | https://github.com/puxiaokun3721/DeepEP-Ascend.git | `dcad2ca0f8383dcd99402d95ac9952e54ece2cb5` |

在新建的专用目录中执行，不能 reset/clean 已有工作区：

```bash
git clone --branch afd-five-operator-baseline --single-branch https://github.com/puxiaokun3721/afd-plugin.git
git clone --branch afd-five-operator-baseline --single-branch https://github.com/puxiaokun3721/DeepEP-Ascend.git
git -C afd-plugin rev-parse HEAD
git -C DeepEP-Ascend rev-parse HEAD
git -C DeepEP-Ascend submodule update --init --recursive third-party/deep_jit
git -C DeepEP-Ascend submodule status --recursive
git -C DeepEP-Ascend status --short
```

若分支已前移，记录实际 SHA 并核对差异；需要精确复现时，在这个新克隆中 checkout 上表 commit，然后重新更新其 DeepJIT 子模块。不要自行更新子模块到最新 upstream。不能访问 GitHub 时请用户提供含 Git 信息、上述 commit 和匹配子模块的离线副本，不换到不明来源版本。

只测试算子无需安装 AFD Plugin、vLLM 或 vLLM-Ascend。

## 3. 先读哪些文档

按顺序阅读，均相对各仓库根目录：

1. `afd-plugin/docs/design/README.md`：有效设计入口和实现边界。
2. `afd-plugin/docs/design/a5_async_afd_design.md`：场景、两级分轮、控制协议；重点理解输入 ACK 和返回 credit。
3. `afd-plugin/docs/design/a5_async_afd_deepep_operator_design.md`。
4. `afd-plugin/docs/design/deepep_afd_operators/common_contract.md`，再看目标算子的 `<算子名>/design.md`。
5. `DeepEP-Ascend/docs/afd/README.md`、`dispatch_code_map.md`，以及对应 `dispatch_send.md` / `dispatch_recv.md` / `remaining_operators.md`。
6. `DeepEP-Ascend/README.md` 的 Requirements / Installation；`tests/afd/README.md` 及要执行的脚本源码。

**文档时效提示：** `tests/afd/README.md` 混有早期 33/51 项 CPU 测试和“下游未实现”的阶段记录；代码入口已发展到五算子。`remaining_operators.md` 中早期 O(L³) 描述与后文 O(L²) 更新并存。应以固定快照源码及 `docs/afd/README.md` 当前说明为准，冲突需记录。已有 56 项 CPU 检查记录不是 C++ 编译或设备通过证据。

## 4. 第一阶段：只读环境与占用报告

用户说的工具名是 `mup-smi`。先执行 `command -v mup-smi` 和 `command -v npu-smi`：若存在内部 `mup-smi`，先阅读其帮助，使用明确的只读查询；若只有标准 `npu-smi`，报告名称差异并使用它。二者均不可用或无法可靠查看进程时，停止设备测试。

标准工具的只读起点如下，具体子命令以本机帮助支持为准，不支持就记录，不能转而执行 set/reset：

```bash
command -v mup-smi
command -v npu-smi
npu-smi info
npu-smi info -l
# 对实际枚举的卡号/芯片号查询，不照抄 0 为已获准设备：
# npu-smi info proc -i <card_id> -c <chip_id>
# npu-smi info -t board -i <card_id>
# npu-smi info -t health -i <card_id>
uname -a
uname -m
docker version
docker ps --no-trunc
docker image ls --digests
```

记录每张卡的物理标识、型号、健康、显存、利用率、已有进程 PID/归属，以及分配给本测试的设备集合。权限不足不能视为无占用。不要输出无关进程的敏感命令参数。

读取现有 CANN 安装目录的版本文件与实际路径，记录 `ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME`、`latest` 链接指向、Bisheng 和 C++ 编译器版本。容器内另外核对 Python、NumPy、PyTorch、torch_npu 的版本及加载路径。可先用 `importlib.metadata` 读取包版本，批准设备分配后再初始化 torch_npu/设备。核对 HCCL/HCOMM、UBMEM、URMA 头文件/库和互联可用性，不自动改配置。

上游固定源码 README 声明的验证栈是 950DT、CANN 9.2.0、Python 3.12、PyTorch 2.13.0+cpu、torch_npu 2.13.0rc1，以及其特定 HDK 配置。**这是上游声称的参考栈，不是本环境事实，也不意味着其他版本必然不支持。** 实际版本不同或无法判定兼容时先报告并等待用户决定是否在现状上试编译；不得安装所谓正确版本。不要按 README 中的发布计划推断商业 HDK 已安装。

先提交环境报告，再进入编译/设备阶段；只有核对一致、资源已明确分配、现有容器配置满足要求时才可直接继续。

## 5. Docker 隔离方案

Docker 是运行容器的工具。这里的容器使用已存在、已认可的镜像来提供测试软件环境，宿主机驱动和物理 NPU 仍是共享资源，不会被容器复制。

优先使用内部平台已经提供的 A5 开发容器模板及指定镜像 digest。镜像应自带匹配的 CANN、Bisheng、PyTorch/torch_npu 和编译依赖。没有合适镜像或 Docker 不可用：停止，向用户申请镜像/管理员准备，不自行 docker pull/build、安装 Docker 或修改运行时。

创建前展示并记录：镜像 digest、容器名、设备映射、挂载目录、网络/IPC 模式和资源限制。下列仅是结构模板，**不可直接复制执行**；所有占位符必须由本机检查和内部 A5 运行规范确定：

```bash
docker run -d --name <本次专用且不存在的容器名> \
  --device=<已分配的NPU设备节点> \
  <内部平台规定的必要管理和URMA设备映射> \
  --mount type=bind,src=<已确认驱动目录>,dst=<镜像要求的驱动路径>,readonly \
  --mount type=bind,src=<本次专用工作目录>,dst=/work \
  --workdir /work \
  <已审核的网络和共享内存参数> \
  <已有镜像的固定digest> sleep infinity
```

不得套用只针对其他芯片的 `/dev/davinci*` 列表当作 A5 的完整映射。使用现场受支持的设备插件/运行时和 UBMEM/URMA 映射；权限不足先停。默认容器私有 PID/IPC，只挂所需设备与专用工作目录，驱动只读；不挂整个 `/home`，不把镜像 CANN 与另一版宿主 CANN 混用。若通信必须 host network 或额外权限，先说明原因和影响并确认，不自行扩大权限。

先验证容器逻辑 device 与已分配物理卡的对应关系。测试脚本中的 `LOCAL_RANK=0/1` 是进程本地设备索引；不能假定它必然对应宿主物理卡 0/1，也不能只设置可见性变量就当作隔离成立。映射不清楚时禁止 launch。

## 6. 构建与首次 JIT

以下在已通过环境检查的专用容器、新克隆的 DeepEP-Ascend 根目录执行。建立 `/work/afd-results`，日志和缓存只写入该专用目录。若需要加载 `set_env.sh`，只能 source 已核实的现有版本到当前 shell，不改持久配置。

```bash
mkdir -p /work/afd-results/jit-cache
export EP_JIT_CACHE_DIR=/work/afd-results/jit-cache
export EP_JIT_PRINT_COMPILER_COMMAND=1
set -o pipefail
python --version
c++ --version
python -m pip --version
python -B -m unittest discover -s tests/afd -p 'test_*cpu.py' -v \
  2>&1 | tee /work/afd-results/cpu-tests.log
bash develop.sh 2>&1 | tee /work/afd-results/build.log
```

`develop.sh` 调用 `python setup.py build`，并在源码包中建立 `.so` 链接；它会删除当前仓库的 `build/`、`dist/` 和 `*.egg-info`，因此只能用于本次新克隆。不得用于已有共享开发目录。编译本项目不需要 `pip install -U` 或更换依赖；缺任何依赖就停止确认。

构建之后核实：

```bash
python -c "import deep_ep; print(deep_ep.__file__); print(hasattr(deep_ep, 'AFDBuffer'))"
git status --short
```

必须确认加载当前副本而非另一份已安装 deep_ep，同时检查实际 `.so` 的位置及依赖。Host 扩展编译通过不代表 device kernel 编译通过；device kernel 在首次调用时通过 DeepJIT 编译，CANN/Bisheng 必须在运行时可用。保存首次 JIT 命令和完整错误，不掩盖失败、不回退 CPU。

## 7. 现有单算子功能入口

每条命令执行前复查占用、设备分配及剩余显存。以下超时值为示例，不是性能要求；首次 JIT 超时应分类为未完成，不反复盲跑。`timeout` 只包裹本次测试进程组；容器中不运行其他人的任务。

### 7.1 dispatchRecv：一张已分配卡

```bash
timeout --signal=TERM --kill-after=30s 600s \
  python tests/afd/run_recv_a5.py --device 0 --cores 1

timeout --signal=TERM --kill-after=30s 1800s \
  python tests/afd/run_reuse_a5.py --device 0 --cores 8 --hidden 7168
```

第二条仅在确认至少有 8 个可用 AIV 后执行。该矩阵内部会测试 cores=1 和指定值、BF16、FP8 row/column SF、多种 route 数、跨核/乱序目的 Segment、尾部与非法 route。检查 hidden/SF/metadata 按位一致、padding、输出尾部、输入/控制不变，以及非法输入不留下部分输出。F 本地 fixture 预填 payload/Active，调用真实 dispatchRecv；不依赖在线 scheduler，不证明跨卡发布可见性。

可先缩小矩阵：`--counts 1 4 5 33 --cores 1 --hidden 7168`。这仍会覆盖 FP8；某 dtype 不兼容时记录失败并停下来确认，不能把该分支跳过后称全通过。

### 7.2 dispatchSend：两张已分配且互联可用的卡

```bash
timeout --signal=TERM --kill-after=30s 600s \
  torchrun --standalone --nproc_per_node=2 tests/afd/run_a5_single_round.py \
  --batch 33 --output /work/afd-results/send-b33
```

通过后分别运行 `--batch 0`、`--batch 33 --all-masked`、`--batch 33 --fp8 --transposed-sf`。该脚本固定 H=128、K=8、8 专家、A=0/F=1，batch 必须在 0..32768；不是 H=7168 性能测试。

只有 A 调用 dispatchSend。F 在发送完成后的 barrier 后读取真实 Inbox/payload/route/prefix/Ready，使用独立 golden 检查。F 不调用 ffnScheduler，也不伪造 ACK。每次只执行一次，不复用槽；它验证最终发布可以先于最终 ACK 返回，不验证多轮/异步可见性/图 replay。初始化会申请通信资源，不能仅按 batch×H 估显存；先审阅 AFDBuffer 分配与脚本的资源预算。

### 7.3 当前没有独立设备测试的算子

| 算子 | 当前入口情况 | 内部 agent 需要补充的隔离测试 |
| --- | --- | --- |
| ffnScheduler | CPU 模型和五算子联调入口 | 在本地 F 预置合法 Inbox/Ready/Pending/Completion，调用真实 scheduler；覆盖到齐/缺员、重复、零长、Q/Q_alloc 切片、两种 credit 策略、NOP/ABORT |
| combineSend | CPU 模型和五算子联调入口 | 两端预注册 fixture，F 预置合法 Active、metadata、COMPUTE_READY、初始 credit；单次真实发送，A 完成后校验 payload/Done/Consumed，检查 Completion |
| combineRecv | CPU 模型和五算子联调入口 | A 预置合法单轮返回窗口、Ledger、Done/Consumed 和对端控制映射；调用真实归并，按固定 topk 顺序 FP32 乘加再 BF16 比对；验证 inactive 清零、尾部及 credit |

补充测试应放在专用工作副本独立文件，沿用现有 fixture/ABI/owner。控制状态的人工预置必须在被测 kernel 运行前完成并明确标注“隔离 fixture”，不能把它当成真实生产者的协议证明。涉及远端写回必须用真实注册地址与完成等待，不猜指针。不得在等待中的同 stream 用 `.cpu()` 轮询来解锁 kernel。

`run_capacity_a5.py` 只验证真实 owner 的容量分配/校验，不执行通信 kernel。`run_send_recv_a5.py` 是两算子串接；`run_pipeline_a5.py` 是四卡五算子联调，本轮不作为单算子验收或默认执行项。

## 8. 单算子性能测试：必须单独建设基准

当前 `tests/afd/run_*a5.py` 主要是功能入口，没有完整的五算子独立 benchmark。禁止把整脚本墙钟、CPU golden、进程启动、资源注册、数据拷贝或首次 JIT 时间当成 kernel 耗时。

功能通过后，内部 agent 可在专用副本补充 benchmark，要求：

1. 固定 SHA、输入、路由种子、dtype、H/K/专家数、lane/成员数、C_m、Q/Q_alloc、核数和设备拓扑。先小规模，再按显存预算扩大，不一次分配完整大事务缓存来方便计时。
2. 单独记录编译/JIT、初始化与数据准备时间。计时前预热相同模板，建议 10 次预热、50 次有效样本；只有证明状态可合法复用后才循环。不能反复 launch 已消费的 Active 或翻转过的控制位。
3. 本地 dispatchRecv 可在每次计时外重建 fixture/ExecutionState，确认每次实际执行 WORK。直接计时真实 op 调用；不要把包含 `torch.npu.synchronize()` 和 host 比较的 `recv_npu_support.invoke()` 整体当作纯 kernel 区间。
4. 用当前 torch_npu 支持的 NPU event 或现有 profiler 测 kernel 区间，先核实 API。预分配事件，显式同步读结果；保存逐样本值，报告 median/p90/p99、min/max。profiler 不在场或权限不够时停止申请，不能自行安装或修改系统配置。
5. dispatchSend/combineSend 是有状态通信算子：每次样本必须有合法空闲槽、正确位与 credit，且上一轮 DMA/引用结束。首版可每个样本在计时外重新构建独立 owner/fixture并销毁，但应明确这是冷资源状态，不能称稳态吞吐。更高效重复采样需要真实测试消费者闭合 ACK/credit，不能在运行中 host 强行改位。若无法证明复用安全，保持性能项未完成。
6. 给出明确边界：send 发布/传输完成耗时、等待 ACK/credit 耗时、接收归并耗时分别说明。纯服务时间先保证输入和 credit 就绪；带等待的结果另表列出。跨卡记录每个角色，不把 F 的 host snapshot 计入 A kernel；只有同步口径一致才报告跨 rank 最大值。
7. 带宽分子必须是实际 payload/metadata/SF 字节，不能使用预留窗口大小。分别报告通信预留、普通 HBM 输入输出、计算 workspace；dispatchRecv 是本地展开，不称跨卡通信带宽。scheduler 报调度时间及元数据规模，不报 hidden 吞吐。
8. H=7168、BF16/FP8、mask、热点 F、不同 route/Segment/核数逐项扩展。原生 EP 与 AFD 在去重、展开、等待边界上不同，比较时先对齐实际工作量；不能直接拿上游 README GB/s 得出退化比例。
9. 设备或共享互联有其他工作负载时不做性能验收；记录干扰并暂停。功能超时、错误输出或疑似遗留设备任务时停止扩大规模。

## 9. 结果交付与停止格式

在 `/work/afd-results/` 保存：环境报告、镜像 digest/容器配置、分配设备映射、源码及子模块 SHA、构建/JIT 日志、每个用例命令和退出码、校验输出、benchmark 源码与配置、逐样本数据、测试前后设备状态。不得只交一张 PASS 截图。

按算子分别填写：Host 编译 / JIT 编译 / 真实功能 / 性能 / 未覆盖项。CPU 模型通过、fixture 通过、跨卡协议通过分栏，不能互相替代。单算子成功不等于图模式、五算子闭环或框架集成成功。

阻塞时报告：在哪一步；观察到的版本/占用/首个错误；源码依据；是否触碰环境（应为否）；最小解决建议；需要用户确认的具体事项。保留本次容器和日志，不自动清理共享资源；内部 agent 不执行 GitHub push，通过本地离线交付包交接。

## 10. 问题反馈、补丁审批与复测协作

内部 agent 负责 A5 现场验证，外部 agent 结合源码和设计分析问题。双方通过用户手动拷贝本地 Markdown 报告和补丁交接，不要求内部环境访问外部服务或向 GitHub push。不得擅自联系其他 agent 或向外部服务上传现场日志。可共享的材料先交用户，去除凭据、无关业务数据及内部敏感信息。

### 10.1 停止与继续的边界

- 环境版本不匹配、缺依赖、需要扩大权限：停止相关阶段，向用户确认，禁止自行修改环境。
- 设备被占用：暂停设备测试，禁止杀任务或抢占。
- 编译失败：保存首个有效错误及完整日志，不连续尝试安装或替换软件包。
- 输出错误、越界、超时或通信疑似死锁：停止该用例及后续压力/性能测试，只处理本次明确归属的测试进程，不重置设备。
- 一个算子失败不必阻止其他已确认安全、无依赖的用例；继续前说明隔离依据。若设备健康、残留任务或通信资源状态不明，不得继续设备测试。

### 10.2 每个问题一份可转交报告

在 `/work/afd-results/issues/ISSUE-001.md` 保存报告，后续编号递增。报告必须包含：

1. 问题摘要与分类：环境、Host 编译、JIT 编译、功能、同步/活性、性能或测试夹具。
2. 两个仓库的实际 commit、DeepJIT commit、工作区差异；记录测试说明所在 commit 与算子实现 commit，不能混为一个版本。
3. 设备型号、相关软件版本、镜像 digest、设备映射及测试前占用情况。
4. 最小复现命令、输入规模、dtype、核数、随机种子、必要环境变量（不得包含凭据）。
5. 预期结果、实际结果、退出码、首次出现阶段，以及是否稳定复现；未重跑则明确写未重跑，不为确认复现而冒险重复故障。
6. 首个有效错误及前后文；功能错误提供首个差异位置、期望值和实际值；超时指出最后完成阶段。
7. 涉及源码的仓库相对路径、函数名和行号。
8. 初步判断与依据，明确区分已证实事实和推测。
9. 最小修改建议、影响范围，以及需要用户或外部 agent 回答的具体问题。
10. 完整日志、输入/输出样本及其他附件的位置。原始材料留现场，共享副本按内部要求脱敏。

不要只贴“编译失败”或日志最后一行。每次给用户的消息先写简短摘要：**问题编号 / 当前阻塞 / 已确认事实 / 需要确认什么 / 附件位置**。

### 10.3 代码和补丁规则

- 生产算子存在问题时，先报告并提出最小补丁建议，等待用户批准后再修改。获准补充隔离测试不等于获准修改生产实现或环境。
- 修复必须尽量保留原生 DeepEP-Ascend 的逻辑、函数边界、变量命名及流水同步；不能为了通过测试改变设计契约。
- 新增测试夹具与生产修复分开记录。不得修改 golden、删除断言、放宽误差或跳过失败分支来制造通过。发现 oracle 本身有错也需报告证据，不静默调整标准。
- 获准修改后，按 §10.5 导出完整离线补丁；包含已跟踪文件的最终差异及新增文件，不以 Markdown 中的代码片段代替补丁。
- 内部 agent 不执行 push，不要求配置 GitHub 凭据；也无需创建本地提交。不得将本地修复混入原始基线结果，不覆盖他人的修改。

### 10.4 接收外部反馈与复测闭环

用户会将问题报告交给外部 agent，再把分析或候选补丁转交给内部 agent。收到反馈后：

1. 核对补丁适用的 commit、文件和当前工作区，不能盲目覆盖。
2. 若反馈与现场 SDK 或源码事实不符，附证据说明分歧；通用示例不能代替目标 SDK 事实。
3. 用户批准后应用最小改动，先重跑最小复现，再做相关回归；环境变更需单独获得明确批准。
4. 回复对应问题编号的复测结果：已解决、部分解决或仍存在，附命令、日志和修改后版本/补丁标识。
5. 功能通过后才恢复对应性能测试，明确性能数据对应修改前还是修改后。

CPU 模型通过、Host 编译通过、JIT 通过、设备功能通过及性能结果必须分别报告。测试夹具通过、真实跨卡协议通过也须分开，不能相互替代。

### 10.5 本地 Markdown 与补丁离线交付

每次交接生成独立目录，例如 `/work/afd-results/handoff-001/`。目录位于专用宿主挂载工作区内，保证退出容器后仍可由用户拷贝。后续交接使用新编号，不覆盖旧报告。

```text
handoff-001/
  README.md                 # 交接摘要、需要回答的问题、阅读及补丁应用顺序
  environment.md            # 环境与设备检查（共享副本脱敏）
  results.md                # 按算子/阶段列结果及未完成项
  issues/ISSUE-001.md        # 具体问题与复测记录
  patches/
    DeepEP-Ascend/           # 该仓库的补丁及 series.md
    afd-plugin/              # 仅有该仓库修改时提供
  attachments/              # 必要的脱敏日志、样本和逐样本性能数据
  SHA256SUMS                 # 所有交付文件的相对路径和 SHA-256（不含自身）
```

`README.md` 必须记录两个仓库的完整基线 SHA、DeepJIT SHA、测试说明版本、测试前是否干净、每份补丁所属仓库与顺序、对应问题编号和用户批准范围。说明本包是“相对固定基线的累计补丁”还是“在上一包之后的增量”，推荐每包给固定基线的累计补丁，避免依赖多次手动合并。累计包不可叠加应用。

没有代码修改时明确写“无代码补丁”，只交报告；测试/benchmark 新文件也属于代码修改，必须导出。不能只给内部机器的日志绝对路径，需将允许共享的必要附件实际放入交付目录。无权外带的材料留现场并说明缺口，不规避内部数据规则。

**补丁导出方法（Linux 容器内，逐仓库执行）：**

1. 在新克隆开始工作前记录完整基线 SHA 及工作区状态。若开始时已有改动，先记录并隔离，不把它们混入本轮补丁。不得 reset、clean 或覆盖原目录。
2. 对已跟踪文件，从记录的基线导出最终工作树差异。使用显式审核过的路径列表，不导出凭据、构建产物、缓存或不相关改动：

```bash
# BASE_SHA 是已记录且验证存在的完整基线 SHA；OUT 是本次交付目录。
# 下面的路径占位符必须替换为真实、审核过的仓库相对路径。
git diff --binary --full-index --no-ext-diff --no-textconv "$BASE_SHA" -- \
  <已修改或删除的已跟踪文件路径列表> > "$OUT/patches/DeepEP-Ascend/001-tracked.patch"
```

此命令比较基线与最终工作树，包含已暂存及未暂存的最终差异，不需要把两份 diff 再拼接。它不包含仍未跟踪的新文件。

3. 对审核后的每个未跟踪新增文件，在仓库根目录单独导出创建补丁：

```bash
git diff --no-index --binary --full-index --no-ext-diff --no-textconv -- \
  /dev/null <新增文件的仓库相对路径> > "$OUT/patches/DeepEP-Ascend/002-new-file.patch"
```

该命令存在差异时退出码 **1 是正常结果**；其他错误须检查，不能一概忽略。各新增文件使用独立补丁编号，并在 `series.md` 列明顺序。无需为了生成补丁修改真实工作区的暂存区。若重命名表现为旧文件删除和新文件创建，须同时交付两部分。

4. 在另一个一次性的干净验证副本中检出同一基线，按 `series.md` 顺序执行 `git apply --check` 再 `git apply`。这只验证补丁可应用，不代表算子可编译或设备通过。不要拿实际测试工作区试图重复应用补丁。若验证副本不可建立，明确记为“补丁应用未验证”。
5. 比较应用后受影响文件与实际测试工作区的内容哈希，核对新增、删除及文件模式，确认导出的就是受测版本。记录补丁 SHA-256；报告中的测试结果绑定该补丁集合，不能继续改文件却沿用旧结果。
6. 生成完整文件清单及 SHA-256。用户只需手动拷贝 `handoff-001/` 或其归档文件给外部 agent。不要只提供终端聊天内容，也不要要求用户代为 push 内部工作区。

外部 agent 收到离线包后先核对基线、哈希、文件清单和补丁范围，再审查或在隔离副本验证；不会直接覆盖自己的当前工作区。分析意见或候选修复同样可以通过 Markdown 与补丁由用户带回内部环境。是否提交和推送外部仓库另按用户授权执行，不是内部测试的前置条件。

## 11. 给内部 agent 的起始指令

严格执行本说明。第一步只做宿主机与候选容器的只读盘点，优先检查 mup-smi/npu-smi 占用，给出 CANN/驱动/工具链/包版本、现有镜像、可分配设备与差异报告。不得修改现有环境、不得杀已有任务、不得在宿主机编译或测试。条件满足后在专用容器按固定分支验证 dispatchRecv 与 dispatchSend；其余算子及性能基准按本文明确的缺口补充隔离测试。遇到环境版本不匹配、需要额外权限或缺依赖时，先停止并向我确认。


发现问题时按第 10 节提交编号报告，由用户转交外部 agent。生产修复须先批准；复测继续使用同一问题编号并保留原始基线结果。

内部 agent 无需向 GitHub 推送；按 §10.5 在本地生成 Markdown 报告和补丁交付包，由用户手动拷贝给外部 agent。
