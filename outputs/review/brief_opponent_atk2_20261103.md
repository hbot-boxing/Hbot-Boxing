# 验收简报 opponent_atk2_20261103

**结论：PASS（受限）** ｜ 退出码 2 ｜ 2026-11-03T22:33:05

**验收模式：ONNX-only（直接验收交付件：跳过 ckpt 体检 / 重新导出 / 数值一致性）**

- checkpoint：`—（ONNX-only 模式未提供 .pt）`（参数量 —）
- ONNX：`/home/yjh/H-Bot/models/opponent_atk2_20261103.onnx`

## 关键指标

| 指标 | 值 |
| --- | --- |
| 文件大小 | 2359996 字节（上限 50000000） |
| 接口 | obs[1, 1097] → target_pose[1, 225] |
| opset | 17 |
| sha256 | `caad2aea3dc452b8…` |
| 数值一致性 | 未测（ONNX-only，无 .pt 对照） |

## 八步流水线

| 步 | 动作 | 状态 | 说明 |
| --- | --- | --- | --- |
| 1 | checkpoint 体检 | **SKIP** | ONNX-only：未提供 .pt，跳过权重体检 |
| 2 | 导出 ONNX | **SKIP** | ONNX-only：直接验收交付件 opponent_atk2_20261103.onnx（未重新导出） |
| 3 | 结构 / 大小 / 静态图 | **PASS** | 2359996 字节，opset 17，obs[1, 1097] → target_pose[1, 225] |
| 4 | 数值一致性 | **SKIP** | ONNX-only：跳过（未提供 checkpoint，无法做数值一致性比对） |
| 5 | 官方镜像 --validate | **PASS** | status=PASS size=2359996 用时 1s → validate_opponent_atk2_20261103_20261103-223303.log |
| 6 | 完整链耗时 ≤ 10ms | **BLOCKED** | 可用内存 6.7GB < 8.0GB，完整链加载会 OOM |
| 7 | 本地对战 ≥ 10 局 | **SKIP** | 未启用 --matches；G3-07 记为 BLOCKED |
| 8 | 生成简报 | **PASS** | brief_opponent_atk2_20261103.md / .json（结论 PASS（受限）） |

## 门禁对照

| 门禁 | 状态 | 检查项 |
| --- | --- | --- |
| G3-01 | **PASS** | ONNX 输入 obs[1,1097] float32 |
| G3-02 | **PASS** | ONNX 输出 target_pose[1,225] float32 |
| G3-03 | **PASS** | 文件大小 < 50,000,000 字节 |
| G3-08 | **PASS** | 无动态轴 / 无 external data / opset ≥ 13 |
| G3-04 | **BLOCKED** | PyTorch ↔ ONNX 最大误差 < 1e-4 |
| G3-05 | **PASS** | 官方 --validate 通过 |
| G3-06 | **BLOCKED** | 完整链耗时 ≤ 10ms（本地参考） |
| G3-07 | **BLOCKED** | 本地对战 ≥ 10 局（镜像输出） |

## 下一步

- 受限：4 项 BLOCKED，可带受限项提交，但平台验证前请补齐证据。
  - 第 6 步 完整链耗时 ≤ 10ms：可用内存 6.7GB < 8.0GB，完整链加载会 OOM
  - G3-04 PyTorch ↔ ONNX 最大误差 < 1e-4：未取得本地证据
  - G3-06 完整链耗时 ≤ 10ms（本地参考）：未取得本地证据
  - G3-07 本地对战 ≥ 10 局（镜像输出）：未取得本地证据
