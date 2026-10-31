# 阶段三：ONNX 导出与本地校验

> 对应 [架构文档](../架构文档.md) 第六、七章；对应 [任务文档](../任务文档.md) 阶段 3。
> 本文档描述从 PyTorch checkpoint 到「通过官方校验 + 完成本地对战」的完整导出与验证流程。

## 目录

- [1. 目标与验收](#1-目标与验收)
- [2. 导出流程](#2-导出流程)
- [3. 导出脚本](#3-导出脚本)
- [4. 本地校验](#4-本地校验)
- [5. 推理耗时基准](#5-推理耗时基准)
- [6. 本地对战与回放分析](#6-本地对战与回放分析)
- [7. 常见失败原因](#7-常见失败原因)
- [8. 独立检验入口](#8-独立检验入口)
- [9. 产出物](#9-产出物)

---

## 1. 目标与验收

### 1.1 目标

导出一个满足全部硬约束的 ONNX 文件，并通过官方 `--validate` 与本地 `--match`。

### 1.2 验收标准

| 项 | 标准 | 检查方式 |
| --- | --- | --- |
| 输入名/形状 | `obs`，float32，`[1,1097]` | ONNX 解析 + 运行时检查 |
| 输出名/形状 | `target_pose`，float32，`[1,225]` | 同上 |
| 文件大小 | 严格 < 50,000,000 字节 | `os.path.getsize` |
| 单文件 | 所有权重内嵌，无外部 data 文件 | 检查目录内仅有 `.onnx` |
| 官方校验 | `--validate` 通过 | 镜像输出 |
| 本地对战 | 跑完 60s，≥ **10 局**，无 NaN、无异常倒地，产出 replay | 镜像输出 + 对战日志 |
| 推理耗时 | 越接近 10ms 越好，建议 < 5ms | `--benchmark` |

---

## 2. 导出流程

```
checkpoint (.pt)
      │
      ▼  加载模型 + 切 eval()
PyTorch 模型
      │
      ▼  torch.onnx.export(opset=13+, dynamic_axes=None)
ONNX（单文件、权重内嵌）
      │
      ├─► 结构检查：I/O 名称、形状、dtype
      ├─► 数值检查：PyTorch vs ONNX Runtime 输出误差
      ├─► 大小检查：< 50,000,000 字节
      │
      ▼
./holoboxing/run.sh --validate models/xxx.onnx
      │
      ▼
./holoboxing/run.sh --match models/xxx.onnx models/baseline.onnx
```

**关键顺序**：先结构检查，再数值对齐，最后才是大小与耗时。数值不一致时不要急着打包，否则本地对战结果不可信。

---

## 3. 导出脚本

### 3.1 导出核心代码

```python
import torch, os

def export_onnx(model, path, opset=13):
    model.eval()
    dummy = torch.zeros(1, 1097, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            path,
            input_names=["obs"],
            output_names=["target_pose"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,          # 固定 batch=1，禁用动态 shape
        )

    size = os.path.getsize(path)
    assert size < 50_000_000, f"文件过大: {size}"
    print(f"导出完成: {path} ({size} bytes)")
```

### 3.2 数值一致性校验

```python
import onnxruntime as ort, numpy as np

def check_parity(model, onnx_path, n=5):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    for _ in range(n):
        x = np.random.randn(1, 1097).astype(np.float32)
        with torch.no_grad():
            y_torch = model(torch.from_numpy(x)).numpy()
        y_onnx = sess.run(None, {"obs": x})[0]
        diff = np.abs(y_torch - y_onnx).max()
        assert diff < 1e-4, f"数值不一致: max diff={diff}"
```

### 3.3 结构检查

```python
import onnx

m = onnx.load(onnx_path)
for inp in m.graph.input:
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(inp.name, dims, inp.type.tensor_type.elem_type)   # obs [1,1097] float32
for out in m.graph.output:
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(out.name, dims, out.type.tensor_type.elem_type)   # target_pose [1,225] float32
```

### 3.4 导出注意事项（对应架构 6.2）

| 事项 | 要求 |
| --- | --- |
| opset | ≥ 13，且与平台 ONNX Runtime 兼容 |
| 动态维度 | 禁止；固定 `[1,1097]` / `[1,225]` |
| 算子 | 只用标准算子；避免自定义 op、控制流过复杂 |
| 精度 | float32；初赛无需量化 |
| 外部权重 | 禁止 `external_data`，必须单文件内嵌 |
| 文件大小 | 目标 ≤ 40MB，留 10MB buffer |

---

## 4. 本地校验

```bash
./holoboxing/run.sh --validate models/my_policy_v1.onnx
```

**校验关注点**：

| 输出项 | 含义 | 不通过时的排查 |
| --- | --- | --- |
| 接口检查 | I/O 名称/形状/dtype | 看 ONNX 结构，确认导出参数 |
| 加载检查 | ONNX Runtime 能否初始化 | 换 opset、检查算子支持 |
| 推理检查 | 能否产出合法动作 | 检查输出值域、NaN |
| 耗时检查 | 完整链时延（需 `--benchmark`） | 见下一节 |

> 普通 `--validate` **不测完整推理链时延**，测时需显式加 `--benchmark`。

---

## 5. 推理耗时基准

```bash
./holoboxing/run.sh --benchmark models/my_policy_v1.onnx
```

### 5.1 结果解读

| 指标 | 含义 | 处理 |
| --- | --- | --- |
| 上层单模型耗时 | 仅策略前向 | 应 ≤ 3ms（目标） |
| 完整链耗时 | 上层 + HoloMotion | 初赛硬指标 ≤ 10ms |
| 抖动（P99） | 最慢 1% 请求 | 关注是否偶尔超 10ms |

### 5.2 本地结果的使用边界

- 本地 RTX 4090 的测时**不能作为初赛验收依据**；
- 本地只用于**相对比较**（v1 vs v2 是否更快）与**发现明显超标**；
- 是否达标以平台 RTX 5090 实测为准，需尽早安排一次平台验证确认。

### 5.3 耗时优化手段（按优先级）

1. 减少层数与宽度（最有效）；
2. 去掉不必要的归一化层（LayerNorm 在端侧可能较慢，需实测）；
3. 合并线性层（导出前做 weight fusion）；
4. 用 ONNX Runtime 的图优化（`ORT_ENABLE_ALL`）；
5. 最后才考虑量化（初赛不要求，且可能掉点）。

---

## 6. 本地对战与回放分析

### 6.1 对战

```bash
./holoboxing/run.sh --match models/my_policy_v1.onnx models/baseline.onnx
./holoboxing/run.sh --view results/match/replay.hdf5
```

**多次对战建议**：

```bash
for i in $(seq 1 20); do
  ./holoboxing/run.sh --match models/my_policy_v1.onnx models/baseline.onnx \
    --output "results/match/run_$i"
done
```

> 已有结果不会被覆盖；重复测试必须用 `--output` 指定新目录。

### 6.2 回放分析清单

| 观察项 | 判断 | 对应改进 |
| --- | --- | --- |
| 是否频繁倒地 | 平衡能力不足 | 提高倒地惩罚、加入站姿奖励 |
| 出拳是否落空 | 距离/时机不准 | 增加对手距离特征、调整输出平滑 |
| 是否只攻不守 | 被命中率高 | 提高 `hit_taken` 权重 |
| 动作是否抖动 | 输出不平滑 | 增大平滑系数 α |
| 是否贴边/绕圈 | 奖励塑形导致 | 加入位置约束或能耗惩罚 |
| 是否出现「站桩」 | 探索不足 | 提高 entropy 系数 |

### 6.3 对比实验规范

每次只改一个变量，记录：

| 版本 | 改动 | 胜率 | 平均得分 | 倒地率 | 完整链耗时 |
| --- | --- | --- | --- | --- | --- |
| v1 | baseline 架构 | — | — | — | — |
| v2 | 加深一层 | — | — | — | — |
| v3 | 调整奖励权重 | — | — | — | — |

---

## 7. 常见失败原因

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| 校验报「输入形状不符」 | 导出了动态 batch 或维度顺序错 | 关闭 dynamic_axes，确认 `[1,1097]` |
| 校验报「输出名称不符」 | 未指定 `output_names` | 显式命名 `target_pose` |
| 校验报「算子不支持」 | opset 过低或用了自定义 op | 升 opset / 改标准算子 |
| 加载成功但动作异常（NaN） | 训练时有 NaN 或除零 | 检查归一化分母、加 eps |
| 文件超 50MB | 参数量过大或含外部权重 | 减参、`do_constant_folding`、检查 external_data |
| 本地对战瞬间倒地 | 输出值域错误（绝对/残差混淆） | 回查阶段一的动作对齐结论 |
| `--benchmark` 超 10ms | 模型过大或端侧算子慢 | 减层、融合、换算子 |

---

## 8. 独立检验入口

```bash
python3 tools/verify_phase3.py --onnx models/policy_v{N}_{YYYYMMDD}.onnx \
  --ckpt outputs/train/ckpt_{step}.pt \
  --json outputs/verify/phase3.json
```

| 检查 | 对应门禁 | 依赖 | 说明 |
| --- | --- | --- | --- |
| 输入名 `obs`、`[1,1097]`、float32 | G3-01 | `onnx`（纯 CPU） | 解析 graph input |
| 输出名 `target_pose`、`[1,225]`、float32 | G3-02 | `onnx`（纯 CPU） | 解析 graph output |
| 文件大小 < 50,000,000 字节 | G3-03 | 文件系统 | `os.path.getsize` |
| PyTorch ↔ ONNX 最大误差 < 1e-4 | G3-04 | `torch` + `onnxruntime`（CPU） | 5 组随机输入 |
| 无动态轴、无 external data | G3-08 | `onnx`（纯 CPU） | 检查 `dim_param` 与 `external_data` |
| 官方 `--validate` | G3-05 | Docker + 镜像 | 缺失记 `BLOCKED` |
| `--benchmark` 完整链耗时 | G3-06 | Docker + GPU | 缺失记 `BLOCKED` |
| 本地对战 ≥ 10 局 | G3-07 | Docker + GPU | 缺失记 `BLOCKED` |

**独立自证边界**：前五项（G3-01/02/03/04/08）不需要 Docker、不需要 GPU，只要 `pip install onnx onnxruntime` 就能跑完，因此即使镜像环境未就绪，阶段三也能先判定"模型格式与数值正确"。后三项依赖镜像，无 Docker 时退出码为 `2`，对应条目记 `BLOCKED`。

---

## 9. 产出物

| 产出 | 位置 |
| --- | --- |
| 导出脚本 | `src/export/export_onnx.py` |
| 一致性检查脚本 | `src/export/check_parity.py` |
| ONNX 模型 | `models/policy_v{N}_{YYYYMMDD}.onnx` |
| 校验日志 | `outputs/eval/validate_v{N}.log` |
| 基准日志 | `outputs/eval/benchmark_v{N}.log` |
| 对战回放 | `results/match/run_*/replay.hdf5` |
| 对比实验表 | `outputs/eval/compare.csv` |
| 检验入口 | `tools/verify_phase3.py` |
| 检验结果 | `outputs/verify/phase3.json` |

## 相关文档

- [阶段二：策略模型设计与训练](./阶段二-策略模型设计与训练.md)
- [阶段四：平台提交与对战迭代](./阶段四-平台提交与对战迭代.md)
