# ffnScheduler 实施记录

- 新增 `D:/Code/refs/DeepEP-Ascend/deep_ep/include/deep_ep/layout/afd/scheduler.hpp`：私有持久状态、Pending lane和精确workspace/UB size helpers。
- 新增 `D:/Code/refs/DeepEP-Ascend/deep_ep/include/deep_ep/impls/afd/ffn_scheduler.hpp`：单AIV有限元数据调度、Completion一次消费、快照/到齐、phase公平和同层聚合、双预算选片、device computeReady gate。
- 更新分算子 `docs/design/deepep_afd_operators/ffnScheduler/design.md`：接口-only改为内部可执行规格；第7节给出接口、算法、布局、tiling、状态和复用边界。
- 新增 `D:/Code/refs/DeepEP-Ascend/tests/afd/test_scheduler_cpu.py`：7项可执行规格测试，均通过（命令：python tests/afd/test_scheduler_cpu.py）。CPU模型不执行C++，不作为编译/设备精度/性能证据。

## 授权与skill适配

按主agent授权读取本地ascendc-operator-design及design-template/general-tiling参考、ascendc-api-best-practices/pipeline参考、ascendc-operator-code-gen及GUIDE/data-copy参考。用户要求直接基于DeepEP代码组织，故保留DeepEP JIT头文件和低层C API，不复制普通elementwise/row工程模板，也不使用模型端到端优化workflow。没有CANN/NPU环境，未声称完成skill的编译安装、设备精度或性能门禁。

## 关键契约

- default COMPUTE_AHEAD；PRECHECK可以绕过无credit候选。Continuation保持HELD。
- InputReady/credit都32B DMA；Inbox64B。ACK后只用Completion和旧Active快照，不再读取旧Inbox。
- Ready初始化1，期待输入0；Finished使期待位翻转；末片实际ACK由combineSend负责。
- 静态DP物理member去重，不把slot当member；同组键包含num_comm_rounds；支持多个microbatch和跨DP同layer聚合。
- route_begin采用专家prefix+cursor。有效Q与含padding QA独立限制，QA>=max(Q,a)确保非空任务不会永久NOP。
- 控制region layout/host/JIT由主agent集成。共享Completion由return.hpp定义，未更改既有Active/Context结构。

## 复用和待验证

DeepEP没有独立AF scheduler。只声称复用prefix/连续route语义以及现有dispatchRecv的recv_load/store/read低层原语；Pending/到齐/credit/Completion为必要新增，未伪装为上游已有能力。

单AIV不扫描token，但静态成员搜索最坏O(L^3)metadata比较是明确首版性能风险。可后续预建DP/member索引，不能承诺当前大L吞吐。workspace=32+96L+2*A32(4LR)，UB=256+5*A32(4R)，与DP总token/完整maxBS无关。

设备待验证：目标C API签名编译、控制可见性、图replay、实际GMM→computeReady顺序、异步对端错误/超时、最大L元数据调度延迟。computeReady不是GMM，不生成FFN结果。
