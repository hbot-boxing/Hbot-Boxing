# RDK S600 端侧调研（取证版）

> 取证日期 2026-11-04，全部数字来自地瓜官方文档（链接见 §1），未做任何算力换算或推算。
> 配套工具：[tools/check_bpu_ops.py](file:///home/yjh/H-Bot/tools/check_bpu_ops.py)（本机可跑，不需要设备）。

## 结论先说

1. **我方 `policy_v1` 的 16 种算子全部在 S600 BPU 支持表内**，算子层面没有硬门槛；但按约束表，`Gemm` / `LayerNormalization` / `ReduceL2` 的 BPU 列只收 int8/int16/float16，**float32 模型无法整体落 BPU，量化是端侧的必要条件**。
2. **A 的 atk2 模型有 2 个 `OneHot`**（来自 `ArgMin → OneHot → Cast`），官方 BPU 与 CPU 两列都是 `N`，即工具链完全不支持 —— `hb_compile` 会直接拒绝该模型。这是编译门槛，不是性能问题。
3. **现阶段不给任何端侧耗时承诺**。官方 benchmark 里 ResNet18(INT8, 1×3×224×224) 单线程 0.384 ms，只能说明 BPU 算力不是瓶颈；我们的模型耗时取决于动态形状、CPU 回退算子与前后处理，必须实机 `hrt_model_exec` 才成立。无实机 → G5 系列门禁维持 `BLOCKED`。

## 1. 取证清单（硬件与工具链）

| 项 | 值 | 出处 |
| --- | --- | --- |
| BPU | 4× Nash core，最高 560 TOPS | RDK S600 开发者套件页 |
| CPU | 18× Arm Cortex-A78AE 2.0GHz | 同上 |
| MCU | 6× Arm Cortex-R52+（1× DCLS，2× Split-Lock） | 同上 |
| 内存 | 32GB 或 64GB LPDDR5，256-bit，最高 6400MT/s | 同上 |
| 存储 | 64GB / 256GB UFS 3.1 + M.2 NVMe | 同上 |
| 供电 | 12–28V，最大 16A | 同上 |
| 算力单位说明 | 「TOPS」= 每秒万亿次整数运算，衡量 NPU 吞吐，不等于我们的帧率 | — |
| 工具链版本 | OE 3.7.0，包名 `oe-package-3.7.0-s100-s600.tgz` | 算法工具链 v3.7.0 页 |
| 开发镜像 | `ai_toolchain_ubuntu_22_s100_s600_cpu/gpu:v3.7.0`（registry.d-robotics.cc） | 同上 |
| 编译参数 `--march` | S600 用 `nash-p`（S100 用 `nash-e`，S100P 用 `nash-m`） | PTQ 快速上手页 |
| 接受的 ONNX opset | 10–19（我们是 17，在范围内） | 同上 |
| 板上运行件 | `hb_compile` 产出 `.hbm`，板端用 `hrt_model_exec` / UCP API / `hbm_infer` | 模型部署原理页、UCP 章节 |
| 参考耗时 | ResNet18 INT8：Latency 0.384ms、14316 FPS（thread_num:12）；ResNet50 INT8：0.587ms | S600 模型性能 Benchmark 页 |

链接：

- 硬件规格：https://developer.d-robotics.cc/rdk_s_doc/01_Quick_start/01_hardware_introduction/02_rdk_s600/01_rdk_s600_kit
- 工具链版本与下载：https://developer.d-robotics.cc/rdk_s_doc/Advanced_development/toolchain_development/algorithm_toolchain/v3_2_0
- PTQ 快速上手：https://toolchain.d-robotics.cc/guide/faststart/ptq_quickstart.html
- S600 算子支持表：https://toolchain.d-robotics.cc/guide/appendix/supported_op_list/operator_support/onnx_operator_support_j6p.html
- S600 算子 BPU 约束表：https://toolchain.d-robotics.cc/guide/appendix/supported_op_list/operator_constraints_bpu/onnx_operator_constraints_list_j6p.html
- S600 性能 Benchmark：https://toolchain.d-robotics.cc/guide/benchmark/j6ph_benchmark.html

## 2. 算子清单体检（本机结果）

方法：把官方 S600 支持表（193 项）写进 `check_bpu_ops.py`，逐个统计 ONNX 节点的算子类型并判定 BPU/CPU 归属。执行：

```
~/hb-env/bin/python tools/check_bpu_ops.py models/policy_v1_20261103.onnx models/opponent_atk2_20261103.onnx
```

| 模型 | 节点数 | 算子种类 | 结论 | 退出码 |
| --- | --- | --- | --- | --- |
| `policy_v1_20261103.onnx`（我方） | 88 | 16 | 全部在 BPU 支持范围 | 0 |
| `opponent_atk2_20261103.onnx`（A） | 525 | 29 | `OneHot`×2 工具链不支持 | 2 |

我方 16 种算子（全部 BPU+CPU 双 Y）：`Constant` 32、`Mul` 13、`Add` 9、`Div` 6、`Gemm` 5、`Erf` 4、`LayerNormalization` 3、`Slice` 3、`Clip` 2、`Expand` 2、`ReduceL2` 2、`Reshape` 2、`Shape` 2、`Concat` 1、`ReduceSum` 1、`Sub` 1。

需要盯住的约束（官方原文摘录）：

| 算子 | 约束 | 对我们的含义 |
| --- | --- | --- |
| `Gemm` | BPU 只收 int8/int16 输入（输出可为 float32）；K ≤ 8192，N ≤ 1048576 | 量化后成立；我们的 512→512 全连接在其范围内 |
| `LayerNormalization` | BPU 只收 int8/int16/float16 | float32 直出不行，必须量化 |
| `ReduceL2` | BPU 只收 int8/int16/float16 | 同上（25 槽归一化用到） |
| `Slice` | 仅 starts/ends/axes/steps 为**常量**才上 BPU | 我们 3 处 Slice 的参数来源需在量化编译后看落点 |
| `Expand` | 类型不限，形状需可静态推导 | 我们输入固定 `[1,1097]`，风险低 |
| `Cast` | 不支持 float16/float32 → bool8 | 我们目前无此用法 |
| `ScatterND` | float32 输入要求 scatterReduceMode=add；updates 只支持 float16 | A 的模型有 2 处，需按此约束改图 |
| `OneHot` | BPU=N、CPU=N | A 的模型有 2 处，会挡编译 |
| `ArgMin` | int32/float32 时只支持单轴 | A 的模型有 2 处（OneHot 的上游） |

顺带量化出的平台差异：**S600 与 S100 的 ONNX 支持表 193 项里只差 2 行** —— `ScatterND`、`ScatterElements` 在 S600 可上 BPU（`YY`），在 S100 只能 CPU（`NY`）。决赛机型是 S600，这条差异对我们有利。

## 3. 仿真 → 端侧的差异风险

| 风险 | 现象 | 影响 | 验证动作 |
| --- | --- | --- | --- |
| 动态形状 | 图里有 `Shape`/`Reshape`/`Slice` 依赖运行时维度 | 编译期需可静态推导，否则插 CPU 节点或拒编 | 官方 `hb_compile --model x.onnx --march nash-p`（只做 check） |
| 量化精度 | int8 后 `LayerNormalization`/`ReduceL2` 的误差会累积到 25 槽姿态 | 输出漂移 → 裁判判定接触冲量变化 | `hb_verifier` 与 float32 ONNX 逐层对比 |
| 不支持算子 | `OneHot` 双 N | 整模型被拒，无运行可能 | 编译日志 `hb_compile.log` 会点名 |
| CPU 回退 | 部分算子只能 CPU，调度开销不确定 | 单帧耗时不可用仿真里的 1.10ms 推 | `hrt_model_exec` 单线程测 latency |
| 输入语义 | yaml 的 `input_type_rt`/`input_type_train` 针对图像；我们是 1097 维 float32 向量 | 预处理节点可能不适用 | 需读「动态输入介绍」与 UCP 张量接口后再定 yaml |
| 校准集规模 | 官方示例建议约 100 份；我们阶段五文档写「≥500 帧」 | 两处口径不一致 | 见 §5 待决项 |

## 4. 阶段五预习清单（不需要设备的部分）

| 编号 | 动作 | 判据 | 状态 |
| --- | --- | --- | --- |
| P-01 | 拉 OE 3.7.0 CPU 镜像，对 `policy_v1` 跑 `hb_compile --march nash-p`（仅 check，不转换） | 输出 `[Warning]`/`[Error]` 为空，无拒编算子 | 待做（需下载数 GB 镜像） |
| P-02 | 用 `hb_model_info` 看每层 BPU/CPU 归属 | CPU 节点占比与算子清单一一对应 | 待做（依赖 P-01） |
| P-03 | 写 yaml（`onnx_model` + 校准集路径 + 输入类型） | 能生成 `.hbm` | 待做（依赖 §3 输入语义结论） |
| P-04 | `hb_verifier` 跑量化前后一致性 | 相对误差在可接受范围（阈值待与契约对齐后定） | 待做 |
| P-05 | 实机 `hrt_model_exec` 测 latency | 无实机 → `BLOCKED` | BLOCKED |
| P-06 | 减参/蒸馏（阶段五文档已写路线） | 参数量与耗时目标待定 | 未启动 |
| P-07 | A 侧同步：把 `OneHot` 换成 `ReduceMin`/`Gather` 等受支持写法 | 体检退出码从 2 变 0 | 待 A 处理 |

## 5. 未取证与不确定项

- **无实机**：S600 开发套件不在手边，所有板端耗时与温度/功耗数据均为 `BLOCKED`。
- **系统版本绑定**：算法工具链 3.7.0 页写「对应系统软件 4.0.5」，另有一段写「5.1.0」；RDK 文档站当前版本是 5.1.1 Community，三者对应关系未查清，装板前必须核对。
- **向量输入的支持路径**：UCP 章节有「动态输入介绍」，但对我们这种单张量 `[1,1097]` float32 输入的 yaml 写法与预处理节点行为，尚未逐页读完。
- **校准集口径**：官方示例约 100 份 vs 我们阶段五文档 ≥500 帧，二者不一致，需你决定以哪个为准（未擅自改文档）。
- **不做推算**：不把 560 TOPS 折算成「我们能跑多少帧」，也不把 ResNet18 的 0.384ms 当成我们模型的耗时。
