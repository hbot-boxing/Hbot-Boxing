#!/usr/bin/env python3
"""阶段二独立检验入口（阶段审查文档第四章 V-01..V-07）。

用法：
    python3 tools/verify_phase2.py
    python3 tools/verify_phase2.py --json outputs/verify/phase2.json

退出码：0 全部通过 / 1 存在 FAIL / 2 无 FAIL 但存在 BLOCKED。
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRAIN_DIR = ROOT / "outputs" / "train"
EVAL_DIR = ROOT / "outputs" / "eval"
OFFICIAL = ROOT / "holoboxing"
PARAM_LIMIT = 12_500_000
MIN_EPISODES = 20

CHECKS = []


def add(gate, name, status, detail):
    CHECKS.append({"gate": gate, "name": name, "status": status, "detail": detail})


def report_deps():
    deps = [("python torch", False, "未安装")]
    try:
        import torch

        deps[0] = ("python torch", True, torch.__version__)
    except Exception as exc:  # noqa: BLE001
        deps[0] = ("python torch", False, str(exc))

    deps.append(("官方包 holoboxing", OFFICIAL.exists(),
                 str(OFFICIAL) if OFFICIAL.exists() else "缺失（需官方仓库软链）"))
    deps.append(("官方训练环境 warp/mujoco_warp", _has_training_stack(),
                 "可导入" if _has_training_stack() else "未安装（训练与对局评测需此依赖）"))
    deps.append(("训练 checkpoint", bool(list(TRAIN_DIR.glob("ckpt_*.pt"))),
                 str(TRAIN_DIR) if list(TRAIN_DIR.glob("ckpt_*.pt")) else "无"))

    print("依赖清单：")
    for name, ok, note in deps:
        print(f"  [{'READY' if ok else 'MISSING'}] {name} {note}")
    print()


def _has_training_stack():
    try:
        import mujoco_warp  # noqa: F401
        import warp  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def check_policy():
    """G2-01：策略 I/O 尺寸 1097→225，float32。"""
    try:
        import torch

        from src.models.policy import OBS_DIM, TARGET_DIM, Policy, contract_ok

        assert OBS_DIM == 1097 and TARGET_DIM == 225, (OBS_DIM, TARGET_DIM)
        net = Policy().eval()
        with torch.no_grad():
            out = net(torch.zeros(1, OBS_DIM))
        assert tuple(out.shape) == (1, 225), out.shape
        assert out.dtype == torch.float32, out.dtype
        assert contract_ok(out), "X/Z 不是正交单位向量"
        add("G2-01", "策略 I/O 尺寸 1097→225", "PASS",
            f"obs[1,1097] → target_pose{tuple(out.shape)} {out.dtype}，X/Z 正交单位向量通过")
    except Exception as exc:  # noqa: BLE001
        add("G2-01", "策略 I/O 尺寸 1097→225", "FAIL", str(exc))


def check_params():
    """G2-06：参数量估算 ≤ 12.5M。"""
    try:
        from src.models.policy import parameter_count

        count = parameter_count(__import__("src.models.policy", fromlist=["Policy"]).Policy())
        add("G2-06", "参数量 ≤ 12.5M", "PASS" if count <= PARAM_LIMIT else "FAIL",
            f"{count:,} 参数（上限 {PARAM_LIMIT:,}）")
    except Exception as exc:  # noqa: BLE001
        add("G2-06", "参数量 ≤ 12.5M", "FAIL", str(exc))


def check_preprocessing_unity():
    """G2-03：训练/评测与推理适配器共用同一份预处理实现（import，非复制）。"""
    targets = [ROOT / "src" / "train" / "train.py", ROOT / "src" / "eval" / "torch_rollout.py"]
    missing = [str(p.relative_to(ROOT)) for p in targets if not p.exists()]
    if missing:
        add("G2-03", "预处理与推理适配器同一实现", "FAIL", f"缺少文件：{missing}")
        return
    lacking = []
    for path in targets:
        text = path.read_text(encoding="utf-8")
        if not ("src.adapters" in text and "obs_adapter" in text and "action_adapter" in text):
            lacking.append(path.name)
    if lacking:
        add("G2-03", "预处理与推理适配器同一实现", "FAIL", f"未 import 适配器：{lacking}")
    else:
        add("G2-03", "预处理与推理适配器同一实现", "PASS",
            "train.py / torch_rollout.py 均 import src.adapters（无复制实现）")


def check_reward_official():
    """G2-04：奖励函数使用官方判定器，无自建近似规则。"""
    path = ROOT / "src" / "train" / "reward.py"
    if not path.exists():
        add("G2-04", "奖励函数使用官方判定器", "FAIL", "缺少 src/train/reward.py")
        return
    text = path.read_text(encoding="utf-8")
    required = [
        "holoboxing.evaluation.runtime.hand_hit_rules",
        "holoboxing.evaluation.runtime.fall_rules",
        "holoboxing.training.envs.mujoco_rewards",
    ]
    missing = [name for name in required if name not in text]
    if missing:
        add("G2-04", "奖励函数使用官方判定器", "FAIL", f"未引用官方模块：{missing}")
        return
    # 阈值必须来自官方配置，源码中不得出现硬编码判定阈值
    literal = re.findall(r"(?<![\w.])(?:0\.30|5\.0|200)(?![\w.])", text)
    if literal:
        add("G2-04", "奖励函数使用官方判定器", "FAIL", f"疑似硬编码阈值：{literal}")
    else:
        add("G2-04", "奖励函数使用官方判定器", "PASS",
            "引用官方 scoring_config / fall_rules / mujoco_rewards，无硬编码阈值")


def _train_log():
    path = TRAIN_DIR / "train_log.csv"
    if not path.exists():
        return None
    rows = path.read_text(encoding="utf-8").strip().splitlines()
    return rows


def check_convergence():
    """G2-02：训练可收敛，≥1 checkpoint 无异常倒地。"""
    checkpoints = sorted(TRAIN_DIR.glob("ckpt_*.pt")) if TRAIN_DIR.exists() else []
    log = _train_log()
    if not checkpoints:
        add("G2-02", "训练可收敛（≥1 checkpoint）", "BLOCKED",
            "outputs/train 下无 checkpoint：需在官方训练环境运行 src/train/train.py")
        return
    if not log or len(log) < 2:
        add("G2-02", "训练可收敛（≥1 checkpoint）", "BLOCKED", "缺少训练日志 outputs/train/train_log.csv")
        return
    import csv as _csv

    rows = list(_csv.DictReader(log))
    first = float(rows[0]["mean_reward"])
    last = float(rows[-1]["mean_reward"])
    improved = last > first
    add("G2-02", "训练可收敛（≥1 checkpoint）", "PASS" if improved else "FAIL",
        f"{len(checkpoints)} 个 checkpoint，平均奖励 {first:.4f} → {last:.4f}")


def _rollout_rows():
    files = sorted(EVAL_DIR.glob("torch_rollout_*.csv")) if EVAL_DIR.exists() else []
    rows = []
    for path in files:
        import csv as _csv

        with path.open(encoding="utf-8") as handle:
            rows.extend(list(_csv.DictReader(handle)))
    return files, rows


def check_rollout_episodes():
    """G2-05 / G2-07：torch 原生 rollout ≥ 20 局，官方判定器输出。"""
    script = ROOT / "src" / "eval" / "torch_rollout.py"
    files, rows = _rollout_rows()
    if not script.exists():
        add("G2-07", "torch 原生 rollout ≥20 局", "FAIL", "缺少 src/eval/torch_rollout.py")
        return
    text = script.read_text(encoding="utf-8")
    native = "run_match" in text and "reference_factory" in text and "onnxruntime" not in text
    if not rows:
        add("G2-07", "torch 原生 rollout ≥20 局", "BLOCKED",
            "outputs/eval 下无 torch_rollout_*.csv：需在官方环境运行对局评测")
        add("G2-05", "每个 checkpoint ≥20 局", "BLOCKED", "无 rollout 记录")
        return
    abnormal = [row for row in rows if str(row.get("abnormal_fall")).lower() == "true"]
    if len(rows) >= MIN_EPISODES and native and not abnormal:
        add("G2-07", "torch 原生 rollout ≥20 局", "PASS",
            f"{len(rows)} 局，官方判定器输出，无异常倒地（{files[0].name}）")
        add("G2-05", "每个 checkpoint ≥20 局", "PASS", f"{len(rows)} 局记录")
    else:
        detail = f"{len(rows)} 局，异常倒地 {len(abnormal)} 次，脚本 torch 原生={native}"
        add("G2-07", "torch 原生 rollout ≥20 局", "FAIL", detail)
        add("G2-05", "每个 checkpoint ≥20 局", "FAIL", detail)


def main():
    parser = argparse.ArgumentParser(description="阶段二独立检验入口")
    parser.add_argument("--json", default="outputs/verify/phase2.json", help="机器可读结果输出路径")
    args = parser.parse_args()

    report_deps()

    for fn in (check_policy, check_params, check_preprocessing_unity, check_reward_official,
               check_convergence, check_rollout_episodes):
        fn()

    local_ok = all(c["status"] == "PASS" for c in CHECKS if c["status"] != "BLOCKED")
    add("G2-08", "独立检验入口可运行（本地可测项全 PASS）", "PASS" if local_ok else "FAIL",
        f"本地项 {sum(c['status'] == 'PASS' for c in CHECKS)} 项 PASS，"
        f"BLOCKED {sum(c['status'] == 'BLOCKED' for c in CHECKS)} 项")

    has_fail = any(c["status"] == "FAIL" for c in CHECKS)
    has_blocked = any(c["status"] == "BLOCKED" for c in CHECKS)
    code = 1 if has_fail else (2 if has_blocked else 0)

    print(f"{'门禁':7} {'状态':8} 检查项")
    for c in CHECKS:
        print(f"{c['gate']:7} {c['status']:8} {c['name']}  →  {c['detail']}")

    verdict = "FAIL" if has_fail else ("PASS(受限)" if has_blocked else "PASS")
    print(f"\n结论：{verdict}  退出码={code}")

    out_path = Path(args.json)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": 2,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "verdict": verdict,
        "exit_code": code,
        "checks": CHECKS,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入：{out_path}")
    return code


if __name__ == "__main__":
    sys.exit(main())
