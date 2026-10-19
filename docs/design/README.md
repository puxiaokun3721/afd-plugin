# A5 异步 AFD 设计入口

这里保留当前有效设计。文件名不再带修订版本号，修订历史通过 Git 提交追踪。

## 阅读顺序

| 层级 | 文档 | 用途 |
| --- | --- | --- |
| 总体方案 | [A5 异步 AFD 设计](a5_async_afd_design.md) | 场景、后端选择、整体协议、两级分轮、容量与验收条件 |
| 五算子总设计 | [DeepEP-AFD 算子设计](a5_async_afd_deepep_operator_design.md) | 五算子分工、公共接口与执行链 |
| 详细契约 | [公共契约](deepep_afd_operators/common_contract.md) | 容量术语、ABI、布局、状态、错误与生命周期 |
| 逐算子设计 | [详细设计索引](deepep_afd_operators/README.md) | 四个通信算子详细方案及 ffnScheduler 接口设计 |

初版实现采用 DeepEP-Ascend 小算子路线；总体方案中的 MegaMoE 和 MoeDistribute MTE 是其他后端设计，不表示已经实现。

## 代码与验收状态

五算子的开发实现位于独立的 [DeepEP-Ascend AFD 代码说明](https://github.com/puxiaokun3721/DeepEP-Ascend/blob/afd-five-operator-baseline/docs/afd/README.md)，分支为 `afd-five-operator-baseline`。本仓库承载框架和设计，不能据此认为五算子已经接入框架并完成端到端验收。

已有 CPU 模型/ABI 检查记录；没有 CANN/Bisheng/JIT 编译或 A5 设备通过证据。精度、通信可见性、并发活性、图 replay 和性能仍需上卡验证。ffnScheduler 的独立设计文档目前以接口契约为主；当前内部实现说明见代码仓，不能把接口文档当作完整内部设计。

设计定义目标契约，代码说明描述当前实现。若二者存在差异，应明确核查，不能将未验收目标自动视为实现能力。

## 历史资料

旧版本、独立审查、作者回复和开发过程记录不再放在当前分支的文档目录。本机归档位于仓库根目录 `.local-archive/afd-docs-2026-10-19/`，已排除 Git 跟踪；其中包含整理前快照和校验清单。此前提交中的材料仍可通过 Git 历史追溯，不改写历史。

[module](module/) 是原有模块设计目录，本次保持不变。PPT 预览及其他本地工作材料不属于本次整理范围。
