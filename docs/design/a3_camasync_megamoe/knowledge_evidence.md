# A3 同步语义知识证据与验证边界

> 发布说明：本目录的 JSON 均为明确标注的可移植脱敏导出，不是原始 immutable receipt，也不能替代原始回执的审计链校验。原始回执与原始说明已在本地保留，不随本次提交发布。每个 JSON 的 `export_metadata.original_file_sha256` 记录对应原文件的 SHA-256；`exported_document` 保留查询与知识卡 provenance，本机知识目录和回执目录分别替换为 `<KNOWLEDGE_ROOT>` 与 `<AUDIT_ROOT>`。迁移到目标环境时必须通过知识工具重新生成原始 receipts 和 retrieval state。

原始说明文件 SHA-256：`82164aabf5bd907bb75a7b156b208ca9e2eef7a792f972c3ddc5df3ae347ec95`。下文命令中的占位目录必须替换为执行环境的实际路径。

本附录只记录设计使用的知识证据，不表示算子已编译、已运行或跨设备协议已验证。目标为用户明确指定的 A3、DeepSeek V4 W4A8 prefill；部署使用的 CANN / Ascend C 组件版本、编译器生成指令和设备行为尚未在本轮核验。

## 检索范围与流程

- Skills：`.agents/skills/knowledge-query/SKILL.md`、`.agents/skills/ascendc-api-knowledge-query/SKILL.md`；已读取 `query-workflows.md`、`cli.md`、`platform-identification.md`、API selection mode 和 `evidence-check.md`。
- 知识根：`<KNOWLEDGE_ROOT>`。
- 显式平台过滤：`--platform a3`；没有把 arch22 / 2201 自行当作平台 ID。
- discover：8066 张卡；index version 3；索引记录提交 `b82acf8cb04dfe4850eeacb2016f189d36cfe6a2`，记录 dirty=false。
- content fingerprint：`704118fcd54d71d02ff3efbbac85b6502288c03a1ee6121b65b11d2686e5edf6`。
- governance fingerprint：`f5b2aca18eca27003166e8306c83b99cc74cf665656c7ade6ea1eb54433331be`。
- 流程为 discover → 窄查询 → 按路径和摘要选卡 → 同一 file SHA 下连续全文阅读 → 定向字段 compare。未扫描全库兜底、未重建索引、未下载资料、未打开未经定位的 SDK。

采用命令如下；工作目录为 afd-plugin 仓根，路径参数使用调用方已授权的全新审计目录。

```powershell
$knowledgeRoot = '<KNOWLEDGE_ROOT>'
$queryScript = '.agents/skills/knowledge-query/scripts/knowledge_query.py'
$readScript = '.agents/skills/ascendc-api-knowledge-query/scripts/resolve_resources.py'
$auditRoot = '<AUDIT_ROOT>'
python -X utf8 $queryScript --knowledge-root $knowledgeRoot discover --json
python -X utf8 $queryScript --knowledge-root $knowledgeRoot search --query 'CrossCoreSetFlag' --domain ops --technology ascendc --type apis --platform a3 --status stable --match literal -k 1 --text --decision local-crosscore --receipt "$auditRoot/crosscore.json"
python -X utf8 $queryScript --knowledge-root $knowledgeRoot search --query 'SyncAll' --domain ops --technology ascendc --type apis --platform a3 --status stable --match literal -k 1 --text --decision local-crosscore --receipt "$auditRoot/syncall.json"
python -X utf8 $queryScript --knowledge-root $knowledgeRoot search --query 'SetFlag/WaitFlag' --domain ops --technology ascendc --type apis --platform a3 --status stable --match literal -k 1 --text --decision local-mte3-drain --receipt "$auditRoot/pipeline.json"
python -X utf8 $queryScript --knowledge-root $knowledgeRoot search --query 'DataCacheCleanAndInvalid' --domain ops --technology ascendc --type apis --platform a3 --status stable --match literal -k 1 --text --decision scalar-cache --receipt "$auditRoot/dcache.json"
python -X utf8 $queryScript --knowledge-root $knowledgeRoot search --query '64位原子' --domain ops --technology ascendc --type guides --platform a3 --status stable --match literal -k 3 --text --decision remote-ready-atomicity --receipt "$auditRoot/remote_atomicity.json"
```

初始只读召回还包括 `SetFlag`、`跨卡 可见性`、`64位 原子`。后两者返回了 arch22→arch35 迁移/SIMT 指南候选，未选择或采用这些不匹配的候选，不能据其摘要宣称 A3 remote publication 已有保证。

可移植脱敏导出已保存：[retrieval state 导出](knowledge_audit/retrieval_state.json)、[CrossCoreSetFlag receipt 导出](knowledge_audit/crosscore.json)、[SyncAll receipt 导出](knowledge_audit/syncall.json)、[SetFlag/WaitFlag receipt 导出](knowledge_audit/pipeline.json)、[DCache receipt 导出](knowledge_audit/dcache.json)、[跨设备原子性窄查询 receipt 导出](knowledge_audit/remote_atomicity.json)。四个 API 查询各命中一张已读卡，最后一项零命中。原始 receipt 的平台集合均为 `["a3"]`，由查询工具完成完整索引 provenance 校验后生成；原件保持不变。仓库内的导出增加了 export 标识、原文件 SHA-256 和脱敏说明，并替换本机绝对路径，因此不能当作原始 receipt 直接校验或继续审计链；移植时需通过工具重新生成。此轮没有生成 ready receipt 或 outcome。

## 已全文读取的卡片

下列路径相对于知识根的 `knowledge/`。表中 SHA 为阅读工具返回的 `file_sha256`；审计 state 使用查询 receipt 的 `card_sha256`，两种指纹不自行混用。

| 卡片 | 路径 | 已连续阅读 | file_sha256 |
|---|---|---|---|
| CrossCoreSetFlag | `ops/ascendc/apis/simd_api/basic_api/sync_control/inter_core_sync/cross_core_set_flag_isasi.md` | 1–194、195–223 | `c22663b9231bd9a6ea398a0775bf08c46472d26380f5453a4103ec3278839f7a` |
| SyncAll | `ops/ascendc/apis/simd_api/basic_api/sync_control/inter_core_sync/sync_all.md` | 1–176、177–181 | `8b1fe2623b98c2b569cace6f13c4d8944b3c844dab157e13b97dc6f70411693d` |
| SetFlag/WaitFlag | `ops/ascendc/apis/simd_api/basic_api/sync_control/intra_core_sync/set_flag_wait_flag_isasi.md` | 1–129 | `35978ff4d6cef25935981b0ea743f461599eac0e7f778babc9c251ad74a25790` |
| DataCacheCleanAndInvalid | `ops/ascendc/apis/simd_api/basic_api/cache_control/data_cache_clean_and_invalid.md` | 1–175 | `99c7070145438af1da6ce5654b99928f092d0c7db0d6a43037d414520df9a983` |

读取使用 `python -X utf8 $readScript read --card <查询返回的 local_path>`；分页严格使用返回的 `--start-line` 和 `--sha256`。SyncAll 是先沿 CrossCoreSetFlag 原始链接 `sync_all.md` 读入，再以准确标题检索进入审计链。两张卡的 `flagId` 字段已通过 `compare --card <CrossCoreSetFlag> --card <SyncAll> --field flagId` 对照。卡片声明的 `source_headers` 未读取：本轮没有已确认的部署 Ascend C 组件根，不以其他 SDK 或知识卡摘录代替部署头文件。

四张卡的状态为 `stable`，但 receipt 中的 `trust_tier` 均为 `unverified`；这里的 covered 仅表示知识卡足以回答对应的本卡 API 语义问题，不代表部署版本、编译产物或硬件行为已验证。

## 可采用的结论

### 本卡同步范围

CrossCoreSetFlag 第 29–38、81–94、120–153 行说明：A3 支持 mode 0/1/2；mode 2 是同一 AI Core 的 AIC 与两个 AIV 配对；Set 在指定流水的前序任务完成后上报，Wait 等待对应参与核。mode 2 需要支持混核调度的 kernel 类型，不能替换成纯 AIC_ONLY/AIV_ONLY 后仍假定事件可用。相同核上的同一 flag 切换同步模式前必须消耗完前一模式的 Set/Wait；连续不同 flag 的 Set 不保证执行顺序；计数范围为 0–15。

SyncAll 第 75、103–107、131–138 行说明：`SyncAll<true>` 只同步 AIV；在 Mix kernel 中它也不会自动变成 AIC+AIV 全同步。2201 表中的 AIV-only SyncAll 占用 AIV flag 14，Mix 全同步占用 11/12/13；保留 native 事件时必须逐一检查使用区间，不能把两种调用语义混在一起。

这些约束支持保留 native 的 AIV 全核收束与 AIC/AIV 分阶段握手，但不证明某一跨卡协议正确。CrossCoreSetFlag 第 157–164 行、SyncAll 第 108–115 行还明确指出：多流、多个核间同步算子并发且核数需求超过物理资源时，可能形成调度死锁。是否开启 batchmode 及并发资源分配必须纳入目标构建和设备验证，不能只靠减少调用数推断安全。

### MTE3 drain 与缓存访问

SetFlag/WaitFlag 第 33–39、59、68–74、94–104 行说明：这是核内流水同步；Set 等源流水前序读写完成后置事件，Wait 在目的流水消费并清除事件。保留 `MTE3_V` / `MTE3_S` 等正确配对，能够描述本核前序 MTE3 操作完成后继续执行的依赖。它没有把本核某个 Set/Wait 定义为所有 AIV 的 drain；跨核完成仍需要收集所有参与核。

DataCacheCleanAndInvalid 第 32–51、90、105–107、139–148 行区分了 DMA 访问与 scalar DCache 访问：scalar GM polling 可能读缓存，必要时需刷新；多核更新同一 cache line 即使地址不同也可能因清回而互相覆盖。独立生产者状态应避免共享被 scalar 写回的 cache line，具体地址/对齐应与最终实现一致。此卡讨论 DCache↔GM，一样不能推出跨设备 publication 的原子性。

## 与 native 源码的对应关系

源码基线为 ops-transformer `8d46ac50deabb544f59f56f5b82a995159907c03`；以下路径相对于该仓。

| native 位置 | 本轮已核验的源码事实 | 设计要求 |
|---|---|---|
| `mc2/mega_moe/op_kernel/arch22/mega_moe.cpp:42` | mixed AIC 1:2 kernel | 保留与 mode 2 对应的混核类型 |
| `mega_moe_kernel_a3.hpp:1335–1404` | 轮表/专家 pull 后经本卡收束，向 AIC 发布 phase flag | 不将 AIV 全核同步当作 AIC 已消费完成 |
| 同文件 `:1473–1493` | C2V 等待、SwiGLU、AIV 收束和 V2C 发布 | 保留配对及空专家的事件次数 |
| 同文件 `:1919–1926` | drain 未消费的专家完成 flag，并调用 combine epilogue Finalize | 轮间复用不得留下 flag 或写出流水 |
| `utils/block_epilogue_w4a8post_pertoken_v2.hpp:155–161,295–320` | 每个 UB stage 等 MTE3_V；结果经 MTE3 写 peer window | 所有参与 AIV 完成后才允许形成事务 Done |
| `mega_moe_kernel_a3.hpp:1561,1704–1752` | 单轮 AIV 收束；全部轮后 reset counts、CrossRankSyncV2、unpermute | 替换全 EP 屏障时保留事务完成依赖 |

上述后五行文件均位于 `mc2/mega_moe/op_kernel/arch22/` 下。源代码只证明现有调用顺序，不能替代硬件契约。

## 未闭合决策与 A3 litmus 门禁

**保留 native AIV SyncAll、CrossCoreWait 和 epilogue Finalize，不足以证明跨设备 Ready/Done 的发布完整性及 release/acquire 顺序。** 最终 v1 ABI 已采用独占记录中的32位 commit ticket，64位 task sequence 放在提交后保持稳定的 body，不以远端64位原子提交为前提。原始检索中对64位原子的缺口按原样保留，不把该零命中推断成32位提交已获保证。相关跨设备决策保持 `gap`，不生成 ready receipt 或验证 outcome。零命中不表示硬件不支持。

实现前应固定目标 CANN/编译器、确切发布/轮询 primitive、字段宽度与对齐，然后完成以下设备测试：

1. **Ready 与 payload/metadata 的先后关系。** 多 AIV 写入带事务代数的非恒定数据、counts 和 offsets，经候选 drain+fan-in 再发布 Ready；消费端使用候选 polling/cache 序列观察 Ready 后检查全部区段，覆盖本机跨卡及实际同超节点跨机路径。检查不能额外加入会掩盖协议缺陷的全域同步。
2. **32位提交与稳定 body 完整性。** 检查32位commit的有效值观测；body中64位字段的高低32位均变化且混搭可识别。只有接受当前ticket后才消费body，检测提交后任何混合代、旧代或跨字段不一致。测试通过只是在给定构建/拓扑下的证据；不将压力测试当作通用原子保证，也不假设提交前的body读取有效。
3. **Done 与最终回写的关系。** 每个 F 的所有 AIV 排空回写后汇合，再发布一次 Done；A 观察全部应答后立即 unpermute 并检查最终输出。包含空专家、零路由 F、热点专家跨多个 receiver round、每轮不同数据。
4. **物理槽复用与代数。** 在不违反单槽 ownership 的前提下反复复用槽，插入 A/F 进度偏斜，验证旧 Ready/Done 不被新事务接受；字段跨界/回绕按实际 ABI 单独覆盖。slot 选择与 task_seq 分开验证。
5. **并发活性。** 在计划允许的多流和核资源配置下，确认 scheduler、MegamoeFfn 及其他同步算子不存在资源互等。第一层通信轮尚未启用，不能用其未来 credit 设计解释当前活性。

若选定的32位提交及数据先行发布方式缺少契约或测试失败，应在接入前调整 publication primitive/ABI，再重跑相同门禁；不能只在设计文字中写“release store”便认为实现已经具备该语义。本地原始查询 receipt 保持不可变；此处仅发布脱敏导出。后续针对确切提交 primitive 的补充检索必须在目标环境另建原始 receipt。
