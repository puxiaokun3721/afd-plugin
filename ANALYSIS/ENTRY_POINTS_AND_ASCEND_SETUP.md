# vLLM 插件发现机制与昇腾 AFD 开发环境

## 1. 三层代码的职责

在昇腾设备上运行 AFD 时，运行栈可以理解为：

```text
AFD 功能层：vllm-afd-plugin
        ↓ 调用和扩展
推理框架层：vLLM
        ↓ 通过平台插件选择昇腾后端
硬件适配层：vllm-ascend
        ↓
PyTorch / torch-npu / CANN / 昇腾驱动
```

- `vLLM` 提供通用推理引擎、调度器、模型执行框架和插件加载器。
- `vllm-ascend` 是 vLLM 的平台插件，负责把通用执行框架落到昇腾 NPU。
- `vllm-afd-plugin` 是通用功能插件，向 vLLM 注册 AFD 模型、worker、runner、patch 和连接器。

它们需要安装在同一个 Python 运行环境中，但不要求三个源码仓一起编译。

## 2. `entry_points` 是什么

`entry_points` 是 Python 包安装元数据中的“服务登记表”。插件包用它声明：

```text
我属于哪个插件组（group）
我的登记名是什么（name）
加载我时应导入哪个 Python 对象（module:object）
```

本仓 `pyproject.toml` 中的配置是：

```toml
[project.entry-points."vllm.general_plugins"]
afd = "afd_plugin:register_afd"
```

含义分别是：

- group：`vllm.general_plugins`
- name：`afd`
- object reference：`afd_plugin:register_afd`

安装本仓后，pip 会把这条记录写入已安装发行包的
`*.dist-info/entry_points.txt`。vLLM 启动时通过
`importlib.metadata.entry_points(group="vllm.general_plugins")` 查找记录，
调用 `EntryPoint.load()` 导入 `afd_plugin.register_afd`，随后调用该函数。

完整链路如下：

```text
pyproject.toml 中声明 entry point
        ↓ pip install
安装环境生成 *.dist-info/entry_points.txt
        ↓ vLLM 进程启动
按 vllm.general_plugins 分组查询 entry points
        ↓ EntryPoint.load()
导入 afd_plugin:register_afd
        ↓ 调用 register_afd()
注册 AFD 模型并应用兼容性/昇腾 patch
```

因此，entry point 解决的是“运行时发现 Python 插件”，不是把三个工程
编译或静态链接到一起。每个 vLLM 子进程都可能执行插件注册，所以注册逻辑
应当可重复执行；本仓使用 `_registered` 标志保证单进程内幂等。

## 3. AFD 插件和昇腾插件不是同一种插件

vLLM 将插件分为两类：

- 平台插件：例如 `vllm-ascend`，告诉 vLLM 当前硬件平台和对应实现。
- 通用插件：例如 `vllm-afd-plugin`，增加与硬件平台相对独立的功能。

运行时可以简化为：

```text
vllm-ascend 平台插件 → 选择 NPU Platform / Worker / ModelRunner 基础能力
AFD 通用插件         → 注册 AFD 模型，并在 NPU 环境采用昇腾专用实现和算子
```

AFD 并没有自己重新实现一整套昇腾后端；它在 NPU 路径上继承和调用
`vllm-ascend` 提供的 worker、runner 等能力。

## 4. 是否需要下载并联合编译三个仓

通常不需要。需要区分“运行环境中安装了包”和“本地持有其源码仓”：

| 场景 | vLLM | vllm-ascend | afd-plugin |
|---|---|---|---|
| 只开发 AFD（推荐） | 镜像内预装 | 镜像内预装 | 克隆源码并 editable 安装 |
| 不使用镜像 | 安装匹配 wheel | 安装匹配 wheel | 克隆源码并构建安装 |
| 同时修改三个项目 | 克隆/源码安装 | 克隆/源码安装 | 克隆/源码安装 |

即使最后一种情况，也是在同一 Python 环境中分别构建、分别安装，通常按
`vLLM → vllm-ascend → afd-plugin` 的顺序进行，而不是一个构建系统做
“联合编译”。

## 5. 本仓推荐的昇腾开发方式

本仓当前配套版本是：

- vLLM `0.19.1`
- vLLM Ascend `0.19.1rc1`
- Python `3.10` 或 `3.11`
- CANN/NNAL `8.5.1`
- torch/torch-npu `2.9.0/2.9.0`

推荐在实际昇腾 Linux 机器上启动项目 README 指定的配套镜像。镜像负责提供
已经验证过的 vLLM、vllm-ascend、PyTorch、torch-npu 和 CANN 用户态环境；
宿主机仍要安装匹配的昇腾驱动并把 NPU 设备挂载进容器。

容器内只需挂载或克隆本仓，然后执行：

```bash
AFD_BUILD_ASCEND_OPS=1 \
SOC_VERSION=ascend910_9391 \
python -m pip install -v --no-build-isolation --no-deps -e .
```

- `-e .`：Python 源码直接指向工作目录，修改 `.py` 文件后重启进程即可生效。
- `--no-deps`：不让 pip 擅自替换镜像中已经配套的 vLLM 等依赖。
- `--no-build-isolation`：使用当前环境里的 torch-npu/CANN 工具链构建。
- `AFD_BUILD_ASCEND_OPS=1`：构建 AFD 自己需要的昇腾扩展和自定义算子。

修改 Python 代码一般不必重装；修改 `csrc` 下的 C++/CANN 算子后需要重新构建。

## 6. 检查插件是否被发现

```bash
python - <<'PY'
from importlib.metadata import entry_points, version

for package in ("vllm", "vllm-ascend", "vllm-afd-plugin"):
    print(package, version(package))

for group in ("vllm.platform_plugins", "vllm.general_plugins"):
    print(f"\n[{group}]")
    for ep in entry_points(group=group):
        print(ep.name, "->", ep.value)
PY
```

还应检查 Python/NPU 和 AFD 算子：

```bash
python - <<'PY'
import torch
import torch_npu
import vllm_ascend
from afd_plugin.compat.npu import ensure_afd_ascend_ops_loaded

assert torch.npu.is_available()
ensure_afd_ascend_ops_loaded()
print("Ascend and AFD ops are ready")
PY
```

## 7. 官方资料

- [Python Entry Points 规范](https://packaging.python.org/en/latest/specifications/entry-points/)
- [Python 插件发现指南](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/)
- [vLLM 插件系统](https://docs.vllm.ai/en/v0.19.1/design/plugin_system/)
- [vLLM Ascend 版本配套策略](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html)
- [vLLM Ascend 安装说明](https://docs.vllm.ai/projects/ascend/en/latest/installation.html)

## 8. 安装期与启动期生命周期

“注册”在这里有两个不同含义：

```text
安装期：把入口描述写入 Python 环境的包元数据
启动期：vLLM 发现并调用入口，把对象写入当前进程的注册表或修改运行状态
```

```text
源码仓                         安装后的 Python 环境
┌────────────────────┐        ┌────────────────────────────────────┐
│ vllm-ascend         │ pip    │ vllm_ascend-*.dist-info/           │
│ setup.py            ├───────>│   entry_points.txt                 │
│ platform/general EP │        │   platform: ascend -> register     │
└────────────────────┘        │   general: connector/loader/...    │
                              │                                    │
┌────────────────────┐        │ vllm_afd_plugin-*.dist-info/        │
│ afd-plugin          │ pip    │   entry_points.txt                 │
│ pyproject.toml      ├───────>│   general: afd -> register_afd     │
└────────────────────┘        └─────────────────┬──────────────────┘
                                               │ importlib.metadata
                                               v
                              ┌────────────────────────────────────┐
                              │ vLLM 各进程的插件加载器             │
                              └──────────────┬─────────────────────┘
                                             │
                 ┌───────────────────────────┴──────────────────────────┐
                 v                                                      v
     vllm.platform_plugins                                  vllm.general_plugins
     调用 ascend register()                                 调用通用注册函数
                 │                                                      │
                 v                                                      v
     返回 NPUPlatform 类路径                              注册 AFD 模型/应用 patch，
     vLLM 创建 current_platform                           注册 Ascend connector/loader
```

vLLM 0.19.1 对平台插件采用惰性检测：第一次访问
`vllm.platforms.current_platform` 时，加载 `vllm.platform_plugins`，调用每个
候选检测函数；返回非空平台类路径的插件被视为激活。Ascend 的 `register()`
返回 `vllm_ascend.platform.NPUPlatform`，于是 vLLM 实例化它作为当前平台。

通用插件则由 `load_general_plugins()` 加载：它查找
`vllm.general_plugins`，对每条记录先执行 `EntryPoint.load()` 得到 Python
函数，再调用函数。通用插件会在 vLLM 的多个进程中加载，每个进程有自己的
内存注册表和 `_registered` 状态，并不存在一个跨进程共享的“中央插件表”。

`vllm-ascend 0.19.1rc1` 实际同时声明：

```text
vllm.platform_plugins
  ascend                  -> vllm_ascend:register

vllm.general_plugins
  ascend_kv_connector     -> vllm_ascend:register_connector
  ascend_model_loader     -> vllm_ascend:register_model_loader
  ascend_service_profiling -> vllm_ascend:register_service_profiling
```

AFD 声明：

```text
vllm.general_plugins
  afd -> afd_plugin:register_afd
```

最后，模型请求不会再次扫描 `entry_points`。启动阶段完成平台选择和模型注册
后，模型加载器根据模型架构名查询进程内 `ModelRegistry`，选出 AFD 注册的
模型实现；worker/runner 再通过已经选定的 `NPUPlatform` 和 AFD 配置执行。

## 9. 插件加载、功能启用与 patch

插件被加载不等于 AFD 被启用。AFD 的功能开关是有效的
`additional_config["afd"]`：存在并通过校验时启用，省略时关闭。插件加载后
仍会登记 AFD 模型并安装兼容 wrapper，但非 AFD 配置会委托回原始 vLLM
实现，不会自动选择 Attention/FFN 分离 worker。

```text
afd entry point 被加载
        ↓
注册能力、安装有条件 wrapper
        ↓
启动参数是否包含 additional_config["afd"]？
        ├── 否：原始 vLLM / vllm-ascend 路径
        └── 是：选择 AFD role、worker、模型和 connector 路径
```

vllm-ascend 不同：它是平台插件。在安装且未被 `VLLM_PLUGINS` 过滤时，
其平台入口返回 `vllm_ascend.platform.NPUPlatform`，vLLM 将它保存为当前
平台。此后普通推理和 AFD 推理都会建立在该 NPU 平台实现上。若完全不想使用
Ascend，应使用没有安装/没有启用 vllm-ascend 的独立 CUDA/CPU 环境，而不应
仅靠省略某个 AFD 配置。

### Patch 的含义

普通扩展点由主框架主动调用，例如：

```python
hook = registry.get(name)
hook(config)
```

Monkey patch 则由插件在运行时直接替换已经存在的类方法或模块函数：

```python
original = VllmConfig.__post_init__

def wrapped(self):
    if has_afd_config(self):
        # AFD 特殊处理
        ...
    return original(self)

VllmConfig.__post_init__ = wrapped
```

本仓确实使用这种方式替换 vLLM 的配置校验、EngineCore、forward context，
以及 vllm-ascend 的部分配置标准化逻辑。这些 wrapper 尽量按
`additional_config["afd"]` 收窄行为，并对非 AFD 配置调用原函数。

Monkey patch 的主要代价是依赖内部实现、存在全局副作用、受导入顺序影响、
容易与其他插件冲突，并且上游重构函数签名或控制流后可能静默失效。因此更
理想的长期方案是先向 vLLM 贡献通用、硬件中立的 hook/registry/虚方法，再由
外部插件通过公开接口实现特性。vllm-ascend 自身也将 patch 定义为临时兼容
手段，原则是能不用就不用，并要求说明未来删除方案。

## 10. AFD 的 monkey patch 清单与上游化方向

AFD 整体不是 monkey patch；它是一个混合使用 entry point、继承、Registry、
自定义算子和 monkey patch 的插件。当前直接替换 15 个函数/方法绑定，并同步
修正最多 4 个模块内已经导入的 `set_forward_context` 别名：

| Patch 模块 | 被替换的 vLLM/vllm-ascend 绑定 | 目的 |
|---|---|---|
| `config_validation` | `EngineArgs.create_engine_config`、`VllmConfig.__post_init__` | 放宽 AFD ubatching 的原生 DeepEP 校验，自动选择 AFD worker |
| `async_dp_engine` | `EngineCoreProc.run_engine_core`、两个模块中的 `launch_core_engines`、`DPAsyncMPClient.add_request_async` | 让异步 AFD Attention rank 独立推进，绕开 DP wave/FIRST_REQ 控制 |
| `async_dp_forward_context` | `set_forward_context` 及已导入别名 | AFD async-DP 使用 connector 控制面时跳过原生 DPMetadata 协调 |
| `engine_core` | `EngineCore.__init__`、`_initialize_kv_caches`、`shutdown`、两个 CoreProc 的 `run_busy_loop` | 让 FFN rank 成为无 scheduler/KV 的 connector daemon |
| `npu/ascend_platform` | `NPUPlatform._fix_incompatible_config` | 防止 vllm-ascend 标准化过程清除 AFD 自己管理的 DBO/ubatch 配置 |
| `npu/force_load_balance` | `AscendFusedMoE.__init__`、`AscendW8A8DynamicFusedMoEMethod.apply` | 为通信 profiling 注入确定性的伪 expert 路由 |

AFD 在能继承的边界已经继承，例如 NPU Attention/FFN worker 继承
`NPUWorker`，NPU model runner 继承 `NPUModelRunner`，AFD DeepSeek 模型继承
原生模型。上述位置无法只靠继承的共同原因是 vLLM 在内部直接构造或调用固定
类/模块函数，没有 factory、strategy 或 hook 让外部插件提供子类；定义一个
子类但主框架从不实例化它，不会改变运行行为。

建议按责任仓分批上游化，而不是提交一个 AFD 专用总 hook：

1. 向 vLLM 提议通用配置扩展/能力校验接口，消除配置校验 patch。
2. 提议可选择的 `EngineCoreProc`/生命周期或 engine-mode factory，让 AFD 用
   子类/strategy 实现 FFN daemon 与 async-DP，而不替换 EngineCore 方法。
3. 提议 DP metadata provider/policy，让 connector 驱动的实现可以正式跳过
   原生 DP wave/metadata 控制面。
4. 向 vllm-ascend 提交外部 ubatching 能力协商或配置标准化 hook，消除
   `NPUPlatform` patch。
5. 将强制负载均衡建模为 vllm-ascend 的正式 profiling/router policy，消除
   FusedMoE patch。

大型架构改动应先提交 vLLM RFC，说明跨硬件用例、接口契约、默认零开销、
生命周期、多进程语义、向后兼容、测试计划以及 AFD 中哪些 patch 会被删除。
通常更现实的终态是“通用 hook 合入 vLLM/vllm-ascend，AFD 继续作为外部功能
插件”，而不是把所有 AFD 和昇腾算子代码都塞入 vLLM 主仓。
