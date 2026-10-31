#!/usr/bin/env python3
"""阶段一独立检验入口（阶段审查文档第四章 V-01..V-07）。

用法：
    python3 tools/verify_phase1.py
    python3 tools/verify_phase1.py --json outputs/verify/phase1.json

退出码：0 全部通过 / 1 存在 FAIL / 2 无 FAIL 但存在 BLOCKED。
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIRROR_IMAGE = "horizonrobotics/holomotion:holoboxing-eval-v12-amd64"
EVAL_DIR = ROOT / "outputs" / "eval"

CHECKS = []


def add(gate, name, status, detail):
    CHECKS.append({"gate": gate, "name": name, "status": status, "detail": detail})


def report_deps():
    """V-07：启动时打印依赖清单与就绪状态。"""
    deps = []

    try:
        import numpy

        deps.append(("numpy", True, numpy.__version__))
    except Exception as exc:  # noqa: BLE001
        deps.append(("numpy", False, str(exc)))

    for mod in ("src.adapters.obs_adapter", "src.adapters.action_adapter"):
        try:
            __import__(mod)
            deps.append((mod, True, "可导入"))
        except Exception as exc:  # noqa: BLE001
            deps.append((mod, False, str(exc)))

    docker = shutil.which("docker")
    deps.append(("docker", docker is not None, docker or "未安装"))

    if docker is None:
        deps.append((f"评测镜像 {MIRROR_IMAGE}", False, "docker 未安装"))
    else:
        deps.append((f"评测镜像 {MIRROR_IMAGE}", docker_image_present(), "本地镜像存在" if docker_image_present() else "需 docker pull"))

    print("依赖清单：")
    for name, ok, note in deps:
        print(f"  [{'READY' if ok else 'MISSING'}] {name} {note}")
    print()


def docker_image_present():
    if shutil.which("docker") is None:
        return False
    try:
        out = subprocess.run(
            ["docker", "images", "-q", MIRROR_IMAGE],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except Exception:  # noqa: BLE001
        return False
    return bool(out.stdout.strip())


def check_obs_layout():
    """G1-03：字段区间求和 = 1097，且对 1097 维无缝覆盖。"""
    try:
        from src.adapters.obs_adapter import OBS_DIM, OBS_FIELDS, FRAME_DIM, FRAME_FIELDS, check_layout

        check_layout()
        top_sum = sum(hi - lo for lo, hi in OBS_FIELDS.values())
        frame_sum = sum(hi - lo for lo, hi in FRAME_FIELDS.values())
        add("G1-03", "OBS_FIELDS 区间求和 = 1097", "PASS",
            f"顶层 {top_sum} / 1097，单帧 {frame_sum} / {FRAME_DIM}，无重叠无遗漏")
    except Exception as exc:  # noqa: BLE001
        add("G1-03", "OBS_FIELDS 区间求和 = 1097", "FAIL", str(exc))


def check_obs_split():
    """G1-03：假输入切分后各字段无重叠、无遗漏。"""
    try:
        import numpy as np

        from src.adapters.obs_adapter import OBS_DIM, split

        obs = np.arange(OBS_DIM, dtype=np.float32).reshape(1, OBS_DIM)
        parts = split(obs)
        assert parts["self_history"].shape == (10, 108)
        assert parts["keypart_pose"].shape == (10, 5, 9)
        assert parts["keypart_lin_vel"].shape == (10, 5, 3)
        assert parts["opponent_pose"].shape == (5, 3)
        assert float(parts["score_diff"]) == float(OBS_DIM - 1)
        assert float(parts["self_history"][-1][-1]) == 1079.0
        add("G1-03", "假输入切分字段形状与边界", "PASS", "切片边界与官方切片一致")
    except Exception as exc:  # noqa: BLE001
        add("G1-03", "假输入切分字段形状与边界", "FAIL", str(exc))


def check_action_shape():
    """G1-05：输出形状严格 [1,225]、dtype float32。"""
    try:
        import numpy as np

        from src.adapters.action_adapter import ActionAdapter

        out = ActionAdapter().to_target_pose(np.zeros((1, 225), dtype=np.float32))
        assert out.shape == (1, 225), out.shape
        assert out.dtype == np.float32, out.dtype
        add("G1-05", "动作输出形状 [1,225] 且 float32", "PASS", f"shape={out.shape} dtype={out.dtype}")
    except Exception as exc:  # noqa: BLE001
        add("G1-05", "动作输出形状 [1,225] 且 float32", "FAIL", str(exc))


def check_clip():
    """G1-06：越界输入被夹到 pose_min / pose_max。"""
    try:
        import numpy as np

        from src.adapters.action_adapter import ActionAdapter

        adapter = ActionAdapter(pose_min=-1.0, pose_max=1.0, smooth_alpha=0.3)
        out = adapter.to_target_pose(np.full((1, 225), 5.0, dtype=np.float32))
        assert np.all(out <= 1.0 + 1e-6), out.max()
        assert np.all(out >= -1.0 - 1e-6), out.min()
        add("G1-06", "值域裁剪生效", "PASS", f"5.0 输入被夹到 max={float(out.max()):.3f}")
    except Exception as exc:  # noqa: BLE001
        add("G1-06", "值域裁剪生效", "FAIL", str(exc))


def check_smoothing():
    """G1-06：一阶低通让相邻帧差分下降，α 落在 [0.2, 0.4]。"""
    try:
        import numpy as np

        from src.adapters.action_adapter import ActionAdapter

        alpha = 0.3
        adapter = ActionAdapter(smooth_alpha=alpha)
        adapter.to_target_pose(np.zeros((1, 225), dtype=np.float32))
        out = adapter.to_target_pose(np.ones((1, 225), dtype=np.float32))
        step = float(np.abs(out - 0.0).max())
        assert abs(step - alpha) < 1e-5, step
        assert step < 1.0, "平滑后单步差分未下降"
        add("G1-06", "平滑生效（α=0.3，单步差分 1.0→0.3）", "PASS", f"max step={step:.3f}")
    except Exception as exc:  # noqa: BLE001
        add("G1-06", "平滑生效", "FAIL", str(exc))


def check_commit():
    """G1-08：适配器代码已纳入版本控制（只读检查，不执行任何 git 写操作）。"""
    try:
        out = subprocess.run(
            ["git", "ls-files", "src/adapters"],
            capture_output=True, text=True, cwd=ROOT, timeout=20, check=False,
        )
        files = [ln for ln in out.stdout.splitlines() if ln.strip()]
        if files:
            add("G1-08", "适配器代码已纳入版本控制", "PASS", f"{len(files)} 个文件已跟踪")
        else:
            add("G1-08", "适配器代码已纳入版本控制", "FAIL",
                "src/adapters 未被 git 跟踪（需你确认后执行 git add/commit）")
    except Exception as exc:  # noqa: BLE001
        add("G1-08", "适配器代码已纳入版本控制", "FAIL", str(exc))


def check_docker():
    """G1-01 / G1-02：镜像就绪与 --validate 可运行。"""
    docker = shutil.which("docker")
    if docker is None:
        add("G1-01", "评测镜像可拉取", "BLOCKED", "本机无 docker 命令")
        add("G1-02", "run.sh --validate 可运行", "BLOCKED", "依赖 docker 与官方镜像")
        return

    if not docker_image_present():
        add("G1-01", "评测镜像可拉取", "BLOCKED", f"镜像 {MIRROR_IMAGE} 本地不存在")
        add("G1-02", "run.sh --validate 可运行", "BLOCKED", "镜像未就绪")
        return

    add("G1-01", "评测镜像可拉取", "PASS", MIRROR_IMAGE)
    log = EVAL_DIR / "validate_phase1.log"
    if log.exists():
        add("G1-02", "run.sh --validate 可运行", "PASS", str(log))
    else:
        add("G1-02", "run.sh --validate 可运行", "BLOCKED", "镜像就绪但无校验日志，需运行 scripts/run_validate.sh")


def check_replay():
    """G1-07：dummy 策略跑完 60s 并产出 replay。"""
    if shutil.which("docker") is None:
        add("G1-07", "dummy 跑完 60s 产出 replay", "BLOCKED", "依赖 docker + GPU")
        return
    replays = list(EVAL_DIR.rglob("*.hdf5")) if EVAL_DIR.exists() else []
    if replays:
        add("G1-07", "dummy 跑完 60s 产出 replay", "PASS", str(replays[0]))
    else:
        add("G1-07", "dummy 跑完 60s 产出 replay", "BLOCKED", "outputs/eval 下无 replay.hdf5")


def check_assets():
    """G1-09：资源包与场景可加载。"""
    assets = ROOT / "holoboxing" / "assets"
    if assets.exists():
        add("G1-09", "资源包与场景可加载", "BLOCKED", f"{assets} 存在，但未做仿真启动验证")
    else:
        add("G1-09", "资源包与场景可加载", "BLOCKED", "未获取官方资源包（holoboxing/assets）")


def check_perturbation():
    """G1-04：单变量扰动法验证字段语义（需仿真，无法脚本判定）。"""
    add("G1-04", "单变量扰动法验证 ≥3 个字段语义", "BLOCKED", "需仿真运行与人工记录")


def main():
    parser = argparse.ArgumentParser(description="阶段一独立检验入口")
    parser.add_argument("--json", default="outputs/verify/phase1.json", help="机器可读结果输出路径")
    args = parser.parse_args()

    report_deps()

    for fn in (check_obs_layout, check_obs_split, check_action_shape, check_clip,
               check_smoothing, check_commit, check_docker, check_replay,
               check_assets, check_perturbation):
        fn()

    local_ok = all(c["status"] == "PASS" for c in CHECKS if c["status"] != "BLOCKED")
    add("G1-10", "独立检验入口可运行（本地可测项全 PASS）", "PASS" if local_ok else "FAIL",
        f"本地项 {sum(c['status'] == 'PASS' for c in CHECKS)} 项 PASS，"
        f"BLOCKED {sum(c['status'] == 'BLOCKED' for c in CHECKS)} 项")

    has_fail = any(c["status"] == "FAIL" for c in CHECKS)
    has_blocked = any(c["status"] == "BLOCKED" for c in CHECKS)
    code = 1 if has_fail else (2 if has_blocked else 0)

    print(f"{'门禁':7} {'状态':8} 检查项")
    for c in CHECKS:
        print(f"{c['gate']:7} {c['status']:8} {c['name']}  →  {c['detail']}")

    verdict = "FAIL" if has_fail else ("PASS(受限)" if has_blocked else "PASS")
    print(f"\n结论：{verdict}  退出码={code}"
          f"（0=全通过 / 1=存在 FAIL / 2=存在 BLOCKED）")

    out_path = Path(args.json)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": 1,
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
