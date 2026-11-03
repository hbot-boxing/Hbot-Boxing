# H-Bot 拳击竞赛（赛题二）

H-Bot 地平线机甲巅峰赛·赛题二：Unitree G1 戴拳套的 1v1 拳击对抗。
本仓库是参赛侧的工程实现：官方环境接入、观测/动作适配、上层策略训练、本地对战评测与阶段性门禁自证。

- 官方代码与资源包只读，仓库里以 `holoboxing`（软链或克隆）引用，不修改官方实现。
- 全部验收结论按三态记录：`PASS` / `FAIL` / `BLOCKED`，无法检查的项不得记为通过。

## 1. 接口与硬约束

| 项 | 要求 |
| --- | --- |
| 模型输入 | `obs` float32 `[1,1097]` |
| 模型输出 | `target_pose` float32 `[1,225]`（25 槽 × [位置3 + X轴3 + Z轴3]，X/Z 必须正交单位向量） |
| 模型文件 | < 50,000,000 字节（float32 参数量上限约 12.5M） |
| 完整链推理 | ≤ 10ms（初赛 RTX 5090，决赛 RDK S600） |
| 场地/时长 | 5m × 5m，每局 60s；有效命中 ≥ 5N·s（20ms 窗口），同手间隔 ≥ 200ms；倒地 < 0.30m，连续倒地 3s 判 KO |

## 2. 目录结构

```
H-Bot/
├── docs/                     # 比赛规则、架构、阶段任务、阶段审查（门禁权威来源）
├── scripts/
│   ├── setup_env.sh          # 阶段一环境准备（镜像、官方代码、基础自检）
│   ├── train_env.sh          # 阶段二环境变量（内存限制、PYTHONPATH），需 source
│   ├── run_validate.sh       # 官方镜像校验单个 ONNX
│   └── run_match.sh          # 官方镜像跑双模型对战
├── src/
│   ├── adapters/             # obs[1097] 切分 / target_pose[225] 还原（训练与推理共用同一份）
│   ├── models/policy.py      # 上层策略网络（1,280,883 参数）
│   ├── train/                # PPO 训练入口、奖励、配置
│   └── eval/torch_rollout.py # torch 原生对战评测（不依赖 ONNX 导出）
├── tools/
│   ├── verify_phase1.py      # 阶段一门禁（G1-01~G1-10）
│   └── verify_phase2.py      # 阶段二门禁（G2-01~G2-08）
└── outputs/                  # 训练/评测/门禁产物（不入库）
```

## 3. 环境要求

- Ubuntu 22.04+，NVIDIA GPU（≥ 8GB 显存），驱动 ≥ 525
- Docker + NVIDIA Container Toolkit（只有官方镜像复现、`--validate`、真实对战需要）
- Python 3.10（系统 Python 即可），`git`、`curl`
- 内存建议 ≥ 24GB；16GB 机器必须 source `scripts/train_env.sh`（见 §5.5）

## 4. 部署步骤

### 4.1 克隆仓库

```bash
git clone git@github.com:hbot-boxing/Hbot-Boxing.git H-Bot
cd H-Bot
```

### 4.2 取得官方代码

镜像里带官方代码，但训练与评测脚本需要在宿主机 import 官方包，因此宿主机也要一份只读副本：

```bash
git clone --depth 1 https://github.com/HorizonRobotics/HoloMotion.git /mnt/hbotdata/tmp/HoloMotion
ln -sfn /mnt/hbotdata/tmp/HoloMotion/holoboxing ./holoboxing
```

### 4.3 官方评测镜像

```bash
docker pull horizonrobotics/holomotion:holoboxing-eval-v12-amd64
```

镜像内含 warp 1.11.0 / mujoco 3.5.0 / mujoco-warp 3.5.0 / onnxruntime 1.22.0，**但没有 torch**，所以训练与 torch 评测走宿主机环境。

### 4.4 资源包与底层控制器

资源包（G1 机器人模型、拳台场景）：HuggingFace `HorizonRobotics/HoloMotion_assets` 的 `holoboxing/` 子目录，放到

```
/mnt/hbotdata/tmp/HoloMotion/thirdparties/HoloMotion_assets/holoboxing
```

底层控制器（HoloMotion 2.0，约 1.65GB）：按官方清单 `holoboxing/evaluation/frozen-model.json` 下载，国内建议走 hf-mirror：

```bash
BASE="https://hf-mirror.com/HorizonRobotics/HoloMotion_models/resolve/e3c90dd1f7483d02d3b4d2e5fb3a359c6ff9c67a/HoloMotion_motion_tracking_model_v2.0.0"
DEST=/mnt/hbotdata/tmp/HoloMotion/models/holomotion-v2
mkdir -p "$DEST/exported"
for f in config.yaml link_command_modes.json exported/model_10500.onnx; do
  curl -L -C - --retry 20 --retry-all-errors -o "$DEST/$f" "$BASE/$f"
done
```

下载后必须按官方清单校验 size + sha256：

```bash
PYTHONPATH=/mnt/hbotdata/tmp/HoloMotion python3 -c \
  "from holoboxing.evaluation.models import verify_bundle; \
   print(verify_bundle('/mnt/hbotdata/tmp/HoloMotion/models/holomotion-v2'))"
# 期望输出：/mnt/hbotdata/tmp/HoloMotion/models/holomotion-v2
```

> 官方 `python -m holoboxing download` 的 URL 硬编码 `huggingface.co`，国内会超时；用上面的 hf-mirror 等价下载后再用官方 `verify_bundle` 校验即可。
> 注意 hf-mirror 对 `HoloMotion_models` **不能**加 `datasets/` 前缀，加了会 404。

### 4.5 Python 训练环境

系统缺 `python3-venv` 时用 `virtualenv`；复用已有的 torch，避免重新下载 900MB 轮子：

```bash
pip install --user virtualenv
python3 -m virtualenv --system-site-packages "$HOME/hb-env"
"$HOME/hb-env/bin/pip" install \
  warp-lang==1.11.0 mujoco==3.5.0 mujoco-warp==3.5.0 \
  onnx==1.18.0 onnxruntime-gpu==1.22.0 PyYAML==6.0.2 \
  omegaconf loguru tabulate h5py scipy tqdm gdown huggingface_hub
```

自检（`warp-lang` 自带 CUDA Toolkit，宿主机不需要装 `/usr/local/cuda`）：

```bash
"$HOME/hb-env/bin/python" -c "import warp as wp; wp.init()"
# Warp 1.11.0 initialized: CUDA Toolkit 12.9, Driver xx
# "cuda:0" : "NVIDIA GeForce ..." (x GiB, sm_8x)
```

### 4.6 环境变量

```bash
source scripts/train_env.sh
```

该脚本导出 `MALLOC_ARENA_MAX=2`、`MALLOC_TRIM_THRESHOLD_`、`MALLOC_MMAP_THRESHOLD_`、`OMP_NUM_THREADS=1` 与 `PYTHONPATH`（官方仓库根 + 本仓库根）。
`MALLOC_*` 只在进程启动前生效，所以必须 `source`，在 Python 里设无效。

## 5. 跑起来

所有命令都在仓库根执行，先 `source scripts/train_env.sh`。

### 5.1 训练（官方 MuJoCo 训练环境 + 本仓库 PPO）

```bash
"$HB_PY" -m src.train.train --config src/train/configs/ppo.yaml --updates 16
```

- 配置里的 `assets` / `lower` / `output` 是绝对路径，换机器要改。
- 产物：`outputs/train/ckpt_XXXX.pt`（含策略权重、中性参考位姿、log_std）、`outputs/train/train_log.csv`。
- 策略输出是**相对官方中性参考位姿的残差**：位置按米缩放叠加，轴向旋转后归一化并正交化，因此 `target_pose` 的正交单位向量契约恒成立，开局即能站住。

### 5.2 torch 原生对战评测（不导出 ONNX）

```bash
"$HB_PY" -m src.eval.torch_rollout \
  --checkpoint outputs/train/ckpt_0016.pt \
  --assets /mnt/hbotdata/tmp/HoloMotion/thirdparties/HoloMotion_assets/holoboxing \
  --lower  /mnt/hbotdata/tmp/HoloMotion/models/holomotion-v2 \
  --episodes 10 --start-index 1
```

- 通过官方 `run_match(reference_factory=...)` 注入口把上层 ONNX 会话换成 torch 策略，判定、观测编码、物理全走官方实现。
- 每局约 48s（3000 帧 / 60s），结果追加写入 `outputs/eval/torch_rollout_<ckpt>.csv`；用 `--start-index` 分批续跑凑满 20 局。
- 记录字段：`score_*`、`hit_count_*`、`fall_count_*`、`knockdown_count_*`、`ko_count_*`、`end_reason`、`winner`、`sanitized_components`、`nonfinite_values`、`abnormal_fall`（后两项非零即视为仿真/策略异常）。

### 5.3 门禁自证

```bash
python3 tools/verify_phase1.py --json outputs/verify/phase1.json
"$HB_PY" tools/verify_phase2.py --json outputs/verify/phase2.json
```

退出码：`0` 全部通过 / `1` 存在 FAIL / `2` 无 FAIL 但存在 BLOCKED。

### 5.4 官方镜像路径（阶段三及提交用）

```bash
scripts/run_validate.sh models/red.onnx --benchmark
scripts/run_match.sh models/red.onnx models/blue.onnx outputs/eval/match
```

## 6. 当前进度

| 阶段 | 门禁 | 状态 |
| --- | --- | --- |
| 一：环境搭建与接口对齐 | G1-01~G1-10 | 全 PASS（`outputs/verify/phase1.json`） |
| 二：策略设计与训练 | G2-01~G2-08 | 全 PASS（`outputs/verify/phase2.json`） |
| 三：ONNX 导出与本地校验 | G3 | 未开始 |
| 四：提交与初赛 | G4 | 未开始 |
| 五：端侧部署 | G5 | 未开始 |

阶段二实测数据（单世界、4096 步 × 16 次更新）：

- 参数量 1,280,883（上限 12.5M），`obs[1,1097] → target_pose(1,225)`
- 平均奖励 0.00312 → 0.0121，训练期倒地率 0（中性参考初始化后不再一开始就趴地）
- 20 局官方对局：0 异常倒地、0 非有限值，但**0 命中、0 分**——策略目前只学会站桩，还没学会出拳

## 7. 常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| 构建环境时进程被 `Killed`（rc=137） | 宿主机内存不足。`source scripts/train_env.sh`（`MALLOC_ARENA_MAX=2` 把峰值从 ~9.7GB 压到 ~4.8GB） |
| `verify_bundle` 失败 / sha256 不符 | 下载被截断。`curl -C -` 续传后重跑校验；不要手动改文件名 |
| hf-mirror 404 | URL 里多了 `datasets/`；`HoloMotion_models` 是 model 仓库，用 `/resolve/` 直连 |
| `docker` 镜像里 `import torch` 失败 | 镜像本身不含 torch；训练/评测在宿主机 venv 跑，镜像只做格式校验与真实对战 |
| `wp.init()` 报找不到 CUDA | `warp-lang` 1.11.0 轮子自带 CUDA Toolkit 12.9，只需驱动；别装 `/usr/local/cuda` 覆盖 |
| 训练报 `Trying to backward through the graph a second time` | rollout 采样必须包在 `torch.no_grad()` 里（仓库代码已处理） |

## 8. 已知限制

1. 训练预算远低于文档建议（20M 步 + 课程学习），目前只有「站住不倒地」能力，尚无进攻行为。
2. `G2-05` 严格口径要求每个 checkpoint ≥ 20 局；当前只对 `ckpt_0016` 跑满 20 局（每局约 48s）。
3. 单机单世界训练，未启用并行环境；16 路并行需更大内存/显存。
4. 阶段二验收不导出 ONNX（审查文档 C-10/C-11），格式合规与推理耗时留到阶段三。

## 9. 文档索引

- [比赛详情](file:///home/yjh/H-Bot/docs/比赛详情.md) — 规则、接口、提交指南
- [任务文档](file:///home/yjh/H-Bot/docs/任务文档.md) — 五阶段分解与里程碑
- [架构文档](file:///home/yjh/H-Bot/docs/架构文档.md) — 两层控制架构与数据流
- [阶段审查文档](file:///home/yjh/H-Bot/docs/阶段审查文档.md) — 契约与门禁权威来源
- 阶段文档 `docs/阶段文档/` — 每阶段工程指南与 `tools/verify_phaseN.py` 自证入口
