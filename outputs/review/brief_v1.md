# 验收简报 v1

**结论：PASS（受限）** ｜ 退出码 2 ｜ 2026-11-03T19:46:57

- checkpoint：`outputs/train/ckpt_0016.pt`（参数量 1280883）
- ONNX：`/home/yjh/H-Bot/models/policy_v1_20261103.onnx`

## 关键指标

| 指标 | 值 |
| --- | --- |
| 文件大小 | 5135076 字节（上限 50000000） |
| 接口 | obs[1, 1097] → target_pose[1, 225] |
| opset | 17 |
| sha256 | `2069ca94bc76598c…` |
| 数值一致性 | max\|Δ\|=1.79e-07（阈值 1e-04） |
| 完整链耗时 | mean 4.13ms / p99 5.14ms / max 5.24ms（上限 10.0ms） |
| 帧率 | 242 fps（NVIDIA GeForce RTX 4060 Laptop GPU） |

## 八步流水线

| 步 | 动作 | 状态 | 说明 |
| --- | --- | --- | --- |
| 1 | checkpoint 体检 | **PASS** | 参数量 1280883，obs 1097 → target 225，第 16 次更新 |
| 2 | 导出 ONNX | **PASS** | 同权重已有 policy_v1_20261103.onnx，复用 v1（sha 2069ca94bc76…） |
| 3 | 结构 / 大小 / 静态图 | **PASS** | 5135076 字节，opset 17，obs[1, 1097] → target_pose[1, 225] |
| 4 | 数值一致性 | **PASS** | 64 组随机输入，max \|Δ\|=1.79e-07，X/Z 契约=True |
| 5 | 官方镜像 --validate | **PASS** | status=PASS size=5135076 用时 1s → validate_v1_20261103-194656.log |
| 6 | 完整链耗时 ≤ 10ms | **PASS** | mean=4.13ms p99=5.14ms max=5.24ms（上限 10.0ms）→ benchmark_v1.log（复用，同 upper_sha256），未重复加载 |
| 7 | 本地对战 ≥ 10 局 | **SKIP** | 未启用 --matches；G3-07 记为 BLOCKED |
| 8 | 生成简报 | **PASS** | brief_v1.md / .json（结论 PASS（受限）） |

## 门禁对照

| 门禁 | 状态 | 检查项 |
| --- | --- | --- |
| G3-01 | **PASS** | ONNX 输入 obs[1,1097] float32 |
| G3-02 | **PASS** | ONNX 输出 target_pose[1,225] float32 |
| G3-03 | **PASS** | 文件大小 < 50,000,000 字节 |
| G3-08 | **PASS** | 无动态轴 / 无 external data / opset ≥ 13 |
| G3-04 | **PASS** | PyTorch ↔ ONNX 最大误差 < 1e-4 |
| G3-05 | **PASS** | 官方 --validate 通过 |
| G3-06 | **PASS** | 完整链耗时 ≤ 10ms（本地参考） |
| G3-07 | **BLOCKED** | 本地对战 ≥ 10 局（镜像输出） |

## 下一步

- 受限：1 项 BLOCKED，可带受限项提交，但平台验证前请补齐证据。
  - G3-07 本地对战 ≥ 10 局（镜像输出）：未取得本地证据
