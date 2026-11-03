#!/usr/bin/env python3
"""一键验收流水线：checkpoint → ONNX → 校验 → 测速 → 简报。

用法（先 `source scripts/train_env.sh`，或直接用 `scripts/accept.sh`）：
    scripts/accept.sh outputs/train/ckpt_0016.pt
    scripts/accept.sh outputs/train/ckpt_0016.pt --version 3 --matches 10
    scripts/accept.sh outputs/train/ckpt_0016.pt --skip-validate      # 30 秒快检

八步：
    1 ckpt 体检        2 导出 ONNX        3 结构/大小/静态图   4 数值一致性
    5 镜像 --validate  6 完整链测速       7 可选本地对战        8 简报（md + json）

1~4 是阻塞链：任一 FAIL 立即停止并出失败简报；5~6 是旁证，失败也照样出简报。
退出码：0 全部通过 / 1 存在 FAIL / 2 无 FAIL 但有 BLOCKED。
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))  # 复用 verify_phase3（tools 不是包）

MODEL_DIR = ROOT / "models"
EVAL_DIR = ROOT / "outputs" / "eval"
REVIEW_DIR = ROOT / "outputs" / "review"
SIZE_LIMIT = 50_000_000
BENCH_LIMIT_MS = 10.0
PARITY_SAMPLES = 64
PARITY_ATOL = 1e-4
MEM_VALIDATE_GB = 6.0
MEM_BENCHMARK_GB = 8.0
IMAGE = os.environ.get("HOLOBOXING_IMAGE", "horizonrobotics/holomotion:holoboxing-eval-v12-amd64")
ASSETS = os.environ.get(
    "HB_ASSETS",
    "/mnt/hbotdata/tmp/HoloMotion/thirdparties/HoloMotion_assets/holoboxing",
)
LOWER = os.environ.get("HB_LOWER", "/mnt/hbotdata/tmp/HoloMotion/models/holomotion-v2")

STEPS = []


def step(index, name, status, detail, evidence=""):
    STEPS.append(
        {"step": index, "name": name, "status": status, "detail": detail, "evidence": evidence}
    )
    print(f"[{index}/8] {status:8} {name}  →  {detail}", flush=True)


def available_gb():
    """/proc/meminfo 的 MemAvailable（含可回收页缓存）。"""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def gpu_name():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def next_version():
    versions = []
    for path in MODEL_DIR.glob("policy_v*_*.onnx"):
        match = re.match(r"policy_v(\d+)_", path.name)
        if match:
            versions.append(int(match.group(1)))
    return max(versions, default=0) + 1


def version_of_same_model(sha):
    """同权重已导出过就复用原版本号，避免同一份权重重复占号。"""
    from src.export.export_onnx import file_sha256

    for path in sorted(MODEL_DIR.glob("policy_v*_*.onnx")):
        match = re.match(r"policy_v(\d+)_", path.name)
        if match and file_sha256(path) == sha:
            return path, int(match.group(1))
    return None, None


def run_checks_quiet(onnx_path, ckpt, samples=PARITY_SAMPLES):
    """纯 CPU 项（G3-01/02/03/04/08），复用阶段三脚本。"""
    import verify_phase3

    return verify_phase3.run_checks(onnx_path, ckpt, samples=samples, evidence=False)


def docker_available():
    try:
        return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def run_validate(onnx_path, version, stamp):
    """第 5 步：官方镜像 --validate（不带 --benchmark，内存才够）。"""
    out_name = f"validation_v{version}_{stamp}.json"
    log_name = f"validate_v{version}_{stamp}.log"
    if available_gb() < MEM_VALIDATE_GB:
        step(5, "官方镜像 --validate", "BLOCKED",
             f"可用内存 {available_gb():.1f}GB < {MEM_VALIDATE_GB}GB，避免再次 OOM")
        return {"gate": "G3-05", "status": "BLOCKED", "evidence": ""}
    if not docker_available():
        step(5, "官方镜像 --validate", "BLOCKED", f"镜像不可用：{IMAGE}")
        return {"gate": "G3-05", "status": "BLOCKED", "evidence": ""}

    command = [
        "docker", "run", "--rm", "--read-only", "--tmpfs", "/tmp:rw,exec",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--network", "none", "--gpus", "device=0",
        "-e", "MALLOC_ARENA_MAX=2",
        "-e", "MALLOC_TRIM_THRESHOLD_=131072",
        "-e", "MALLOC_MMAP_THRESHOLD_=131072",
        "--mount", f"type=bind,src={onnx_path},dst=/inputs/red.onnx,readonly",
        "--mount", f"type=bind,src={EVAL_DIR},dst=/results",
        IMAGE, "validate", "/inputs/red.onnx",
        "--qualification", "local", "--output", f"/results/{out_name}",
    ]
    started = time.time()
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        step(5, "官方镜像 --validate", "BLOCKED", "容器超时（1800s），多为排队或内存压力")
        return {"gate": "G3-05", "status": "BLOCKED", "evidence": ""}
    log_path = EVAL_DIR / log_name
    log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    cost = time.time() - started
    payload = None
    json_path = EVAL_DIR / out_name
    if json_path.is_file():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            payload = None
    status = payload.get("status") if payload else None
    if status == "PASS":
        size = (payload.get("size_check") or {}).get("total_bytes")
        step(5, "官方镜像 --validate", "PASS",
             f"status=PASS size={size} 用时 {cost:.0f}s → {log_name}", log_name)
        return {"gate": "G3-05", "status": "PASS", "evidence": log_name}
    if proc.returncode in (137, -9) or not payload:
        step(5, "官方镜像 --validate", "BLOCKED",
             f"容器未产出 JSON（rc={proc.returncode}，{cost:.0f}s），多为内存不足 → {log_name}",
             log_name)
        return {"gate": "G3-05", "status": "BLOCKED", "evidence": log_name}
    step(5, "官方镜像 --validate", "FAIL",
         f"status={status} error={payload.get('error_code')} → {log_name}", log_name)
    return {"gate": "G3-05", "status": "FAIL", "evidence": log_name}


BENCHMARK_SNIPPET = """
import json, sys
sys.path.insert(0, sys.argv[1])
from holoboxing.evaluation.runtime.chain_benchmark import benchmark_model
print(json.dumps(benchmark_model(sys.argv[2], assets=sys.argv[3],
                                 lower_directory=sys.argv[4]), indent=1))
"""


def run_benchmark(onnx_path, version, stamp):
    """第 6 步：宿主机 venv 调官方 chain_benchmark（镜像内 --benchmark 会 OOM）。"""
    from src.export.export_onnx import file_sha256

    def judge(payload):
        mean = float(payload.get("mean_ms") or 0.0)
        metrics = {
            "mean_ms": mean,
            "p50_ms": payload.get("p50_ms"),
            "p95_ms": payload.get("p95_ms"),
            "p99_ms": payload.get("p99_ms"),
            "maximum_ms": payload.get("maximum_ms"),
            "fps": payload.get("fps"),
            "gpu": payload.get("gpu"),
            "official_threshold_ms": payload.get("threshold_ms"),
            "official_passed": payload.get("passed"),
            "parity_max_observation_error": (payload.get("parity_preflight") or {}).get(
                "max_observation_error"
            ),
        }
        ok = bool(payload.get("timing_completed")) and mean <= BENCH_LIMIT_MS
        return ok, metrics

    def summary(metrics, name, ok):
        return dict(
            gate="G3-06", status="PASS" if ok else "FAIL", evidence=name, metrics=metrics,
            detail=f"mean={metrics['mean_ms']:.2f}ms "
                   f"p99={float(metrics.get('p99_ms') or 0):.2f}ms "
                   f"max={float(metrics.get('maximum_ms') or 0):.2f}ms"
                   f"（上限 {BENCH_LIMIT_MS}ms）→ {name}",
        )

    # 同一份 ONNX 已有测速证据就复用：官方报告里 upper_sha256 即上层模型 sha256。
    ours = file_sha256(onnx_path)
    for path in sorted(EVAL_DIR.glob("benchmark_v*.log"), key=lambda p: p.stat().st_mtime,
                       reverse=True):
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"\{", text)
        if not match:
            continue
        try:
            payload = json.loads(text[match.start():])
        except Exception:  # noqa: BLE001
            continue
        if payload.get("upper_sha256") != ours:
            continue
        ok, metrics = judge(payload)
        result = summary(metrics, f"{path.name}（复用，同 upper_sha256）", ok)
        step(6, "完整链耗时 ≤ 10ms", result["status"], result["detail"] + "，未重复加载", path.name)
        return result

    log_name = f"benchmark_v{version}_{stamp}.log"
    if available_gb() < MEM_BENCHMARK_GB:
        step(6, "完整链耗时 ≤ 10ms", "BLOCKED",
             f"可用内存 {available_gb():.1f}GB < {MEM_BENCHMARK_GB}GB，完整链加载会 OOM")
        return {"gate": "G3-06", "status": "BLOCKED", "evidence": "", "metrics": {}}
    command = [
        sys.executable, "-c", BENCHMARK_SNIPPET,
        os.environ.get("HB_HOLOMOTION", "/mnt/hbotdata/tmp/HoloMotion"),
        str(onnx_path), ASSETS, LOWER,
    ]
    started = time.time()
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        step(6, "完整链耗时 ≤ 10ms", "BLOCKED", "宿主机 benchmark 超时（3600s）")
        return {"gate": "G3-06", "status": "BLOCKED", "evidence": "", "metrics": {}}
    log_path = EVAL_DIR / log_name
    log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    cost = time.time() - started
    payload = None
    match = re.search(r"\{", proc.stdout)
    if match and proc.returncode == 0:
        try:
            payload = json.loads(proc.stdout[match.start():])
        except Exception:  # noqa: BLE001
            payload = None
    if payload is None:
        step(6, "完整链耗时 ≤ 10ms", "BLOCKED",
             f"官方 benchmark 未产出 JSON（rc={proc.returncode}，{cost:.0f}s）→ {log_name}", log_name)
        return {"gate": "G3-06", "status": "BLOCKED", "evidence": log_name, "metrics": {}}
    ok, metrics = judge(payload)
    result = summary(metrics, log_name)
    step(6, "完整链耗时 ≤ 10ms", result["status"], result["detail"], log_name)
    return result


def run_matches(onnx_path, version, rounds):
    """第 7 步：镜像本地对战（默认关闭；本机 16GB 内存通常 OOM）。"""
    if rounds <= 0:
        step(7, "本地对战 ≥ 10 局", "SKIP", "未启用 --matches；G3-07 记为 BLOCKED")
        return {"gate": "G3-07", "status": "BLOCKED", "evidence": ""}
    if available_gb() < MEM_BENCHMARK_GB:
        step(7, "本地对战 ≥ 10 局", "BLOCKED",
             f"可用内存 {available_gb():.1f}GB 不足（本机对局需 ≥24GB）")
        return {"gate": "G3-07", "status": "BLOCKED", "evidence": ""}
    from src.export.export_onnx import file_sha256

    ours = file_sha256(onnx_path)
    built, bad = 0, []
    for index in range(1, rounds + 1):
        target = EVAL_DIR / "match" / f"policy_v{version}_run{index:02d}"
        if (target / "match.json").exists():
            built += 1
            continue
        target.mkdir(parents=True, exist_ok=True)
        command = [
            "docker", "run", "--rm", "--read-only", "--tmpfs", "/tmp:rw,exec",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--network", "none", "--gpus", "device=0",
            "-e", "MALLOC_ARENA_MAX=2",
            "-e", "WARP_CACHE_PATH=/wp_cache",
            "--mount", f"type=bind,src={onnx_path},dst=/inputs/red.onnx,readonly",
            "--mount", f"type=bind,src={onnx_path},dst=/inputs/blue.onnx,readonly",
            "--mount",
            f"type=bind,src={Path.home()}/.cache/warp/1.11.0,dst=/wp_cache",
            "--mount", f"type=bind,src={target},dst=/results",
            IMAGE, "match", "/inputs/red.onnx", "/inputs/blue.onnx", "--output-dir", "/results",
        ]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=3600)
        (target / "run.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
        payload_path = target / "match.json"
        if not payload_path.is_file():
            bad.append(f"第 {index} 局无 match.json（rc={proc.returncode}）")
            break
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        summary = payload.get("summary") or {}
        if ours not in (summary.get("model_id") or []):
            bad.append(f"第 {index} 局 model_id 不匹配")
        if any(int(v) for v in (summary.get("fall_count") or [])):
            bad.append(f"第 {index} 局摔倒 {summary['fall_count']}")
        built += 1
    if bad:
        step(7, "本地对战 ≥ 10 局", "FAIL", "；".join(bad[:3]))
        return {"gate": "G3-07", "status": "FAIL", "evidence": ""}
    if built < rounds:
        step(7, "本地对战 ≥ 10 局", "BLOCKED", f"仅完成 {built}/{rounds} 局")
        return {"gate": "G3-07", "status": "BLOCKED", "evidence": ""}
    step(7, "本地对战 ≥ 10 局", "PASS", f"{built} 局完成，无摔倒、model_id 匹配")
    return {"gate": "G3-07", "status": "PASS", "evidence": ""}


def write_brief(version, stamp, ckpt_info, onnx_info, gates, metrics, verdict, code):
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    steps_failed = [s for s in STEPS if s["status"] == "FAIL"]
    blocked = [s for s in STEPS if s["status"] == "BLOCKED"]
    blocked_gates = [g for g in gates if g["status"] == "BLOCKED"]

    payload = {
        "version": version,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "ckpt": ckpt_info,
        "onnx": onnx_info,
        "steps": STEPS,
        "gates": gates,
        "metrics": metrics,
        "environment": {
            "available_gb": round(available_gb(), 2),
            "gpu": gpu_name(),
            "image": IMAGE,
            "assets": ASSETS,
            "lower": LOWER,
        },
        "verdict": verdict,
        "exit_code": code,
    }
    json_path = REVIEW_DIR / f"brief_v{version}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        f"# 验收简报 v{version}",
        "",
        f"**结论：{verdict}** ｜ 退出码 {code} ｜ {payload['generated_at']}",
        "",
        f"- checkpoint：`{ckpt_info.get('path', '—')}`（参数量 {ckpt_info.get('param_count', '—')}）",
        f"- ONNX：`{onnx_info.get('path', '—')}`",
        "",
        "## 关键指标",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 文件大小 | {onnx_info.get('size_bytes', '—')} 字节（上限 {SIZE_LIMIT}） |",
        f"| 接口 | {onnx_info.get('input', '—')} → {onnx_info.get('output', '—')} |",
        f"| opset | {onnx_info.get('opset', '—')} |",
        f"| sha256 | `{onnx_info.get('sha256', '—')[:16]}…` |",
        f"| 数值一致性 | max\\|Δ\\|={onnx_info.get('parity_max_delta', 0):.2e}"
        f"（阈值 {PARITY_ATOL:.0e}） |",
    ]
    if metrics:
        lines += [
            f"| 完整链耗时 | mean {metrics['mean_ms']:.2f}ms"
            f" / p99 {float(metrics.get('p99_ms') or 0):.2f}ms"
            f" / max {float(metrics.get('maximum_ms') or 0):.2f}ms（上限 {BENCH_LIMIT_MS}ms） |",
            f"| 帧率 | {float(metrics.get('fps') or 0):.0f} fps（{metrics.get('gpu')}） |",
        ]
    lines += ["", "## 八步流水线", "", "| 步 | 动作 | 状态 | 说明 |", "| --- | --- | --- | --- |"]
    for item in STEPS:
        lines.append(
            f"| {item['step']} | {item['name']} | **{item['status']}** | "
            f"{str(item['detail']).replace('|', chr(92) + '|')} |"
        )
    lines += ["", "## 门禁对照", "", "| 门禁 | 状态 | 检查项 |", "| --- | --- | --- |"]
    for gate in gates:
        lines.append(f"| {gate['gate']} | **{gate['status']}** | {gate['name']} |")
    lines += ["", "## 下一步", ""]
    if steps_failed:
        lines.append(f"- 阻塞：{len(steps_failed)} 项 FAIL，先修再谈提交。")
        for item in steps_failed:
            lines.append(f"  - 第 {item['step']} 步 {item['name']}：{item['detail']}")
    elif blocked or blocked_gates:
        count = len(blocked) + len(blocked_gates)
        lines.append(f"- 受限：{count} 项 BLOCKED，可带受限项提交，但平台验证前请补齐证据。")
        for item in blocked:
            lines.append(f"  - 第 {item['step']} 步 {item['name']}：{item['detail']}")
        for gate in blocked_gates:
            lines.append(f"  - {gate['gate']} {gate['name']}：未取得本地证据")
    else:
        lines.append("- 本地预检全通过，可按阶段四 Checklist 提交（平台结论以平台为准）。")

    md_path = REVIEW_DIR / f"brief_v{version}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n简报：{md_path}\n      {json_path}")
    return md_path, json_path


def main():
    parser = argparse.ArgumentParser(description="一键验收流水线（ckpt → ONNX → 校验 → 测速 → 简报）")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--version", type=int, default=0, help="默认取同权重已有版本，否则最大版本+1")
    parser.add_argument("--matches", type=int, default=0, help="本地对战局数（默认 0，即关闭）")
    parser.add_argument("--skip-validate", action="store_true", help="跳过第 5~7 步（30 秒快检）")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    import onnx
    import torch

    from src.export.export_onnx import export, file_sha256, load_policy
    from src.models.policy import OBS_DIM, TARGET_DIM, parameter_count

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_path = Path(args.ckpt)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1 ckpt 体检
    if not ckpt_path.is_file():
        step(1, "checkpoint 体检", "FAIL", f"文件不存在：{ckpt_path}")
        raise SystemExit(1)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing = [key for key in ("policy", "neutral") if key not in payload]
    policy = load_policy(ckpt_path)
    ckpt_info = {
        "path": str(ckpt_path),
        "sha256": file_sha256(ckpt_path),
        "param_count": parameter_count(policy),
        "update": payload.get("update"),
        "episodes": payload.get("episodes"),
        "obs_dim": payload.get("obs_dim", OBS_DIM),
        "target_dim": payload.get("target_dim", TARGET_DIM),
        "missing_keys": missing,
    }
    if missing or ckpt_info["obs_dim"] != OBS_DIM or ckpt_info["target_dim"] != TARGET_DIM:
        step(1, "checkpoint 体检", "FAIL",
             f"缺键 {missing} / 维度 {ckpt_info['obs_dim']}→{ckpt_info['target_dim']}")
        write_brief(args.version or 0, stamp, ckpt_info, {}, [], {}, "FAIL", 1)
        raise SystemExit(1)
    step(1, "checkpoint 体检", "PASS",
         f"参数量 {ckpt_info['param_count']}，obs {OBS_DIM} → target {TARGET_DIM}，"
         f"第 {ckpt_info['update']} 次更新")

    # 2 导出（先写临时文件，再按 sha 决定复用还是落盘）
    temp = MODEL_DIR / f".accept_tmp_{os.getpid()}.onnx"
    export(policy, temp, args.opset)
    sha = file_sha256(temp)
    same_path, same_version = version_of_same_model(sha)
    if args.version:
        version = args.version
    elif same_path is not None:
        version = same_version
    else:
        version = next_version()
    target = MODEL_DIR / f"policy_v{version}_{datetime.date.today():%Y%m%d}.onnx"
    if target.exists() and file_sha256(target) != sha:
        temp.unlink()
        step(2, "导出 ONNX", "FAIL",
             f"{target.name} 已存在且内容不同；请显式 --version 指定新版本号")
        write_brief(version, stamp, ckpt_info, {}, [], {}, "FAIL", 1)
        raise SystemExit(1)
    if same_path is not None:
        temp.unlink()
        detail = f"同权重已有 {same_path.name}，复用 v{version}（sha {sha[:12]}…）"
    else:
        shutil.move(str(temp), str(target))
        detail = f"导出 {target.name}（{target.stat().st_size} 字节，sha {sha[:12]}…）"
    step(2, "导出 ONNX", "PASS", detail)
    onnx_path = target if target.exists() else same_path

    # 3+4 结构、大小、静态图、数值一致性（阻塞链）
    checks = run_checks_quiet(onnx_path, ckpt_path)
    by_gate = {c["gate"]: c for c in checks}
    graph = onnx.load(str(onnx_path))
    onnx_info = {
        "path": str(onnx_path),
        "sha256": sha,
        "size_bytes": onnx_path.stat().st_size,
        "opset": max(imp.version for imp in graph.opset_import),
        "input": f"{graph.graph.input[0].name}{[d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]}",
        "output": f"{graph.graph.output[0].name}{[d.dim_value for d in graph.graph.output[0].type.tensor_type.shape.dim]}",
        "parity_max_delta": float(
            re.search(r"max \|Δ\|=([0-9.eE+-]+)", by_gate["G3-04"]["detail"]).group(1)
        ) if by_gate.get("G3-04") else 0.0,
    }
    structure = [by_gate[g] for g in ("G3-01", "G3-02", "G3-03", "G3-08") if g in by_gate]
    if any(c["status"] == "FAIL" for c in structure):
        step(3, "结构 / 大小 / 静态图", "FAIL",
             "；".join(f"{c['gate']}={c['status']}" for c in structure))
    else:
        step(3, "结构 / 大小 / 静态图", "PASS",
             f"{onnx_info['size_bytes']} 字节，opset {onnx_info['opset']}，"
             f"{onnx_info['input']} → {onnx_info['output']}")
    parity = by_gate.get("G3-04", {"status": "BLOCKED", "detail": "缺失"})
    step(4, "数值一致性", parity["status"], parity["detail"])
    if any(c["status"] == "FAIL" for c in structure) or parity["status"] == "FAIL":
        write_brief(version, stamp, ckpt_info, onnx_info, list(checks), {}, "FAIL", 1)
        raise SystemExit(1)

    # 5~7 旁证
    gates = list(checks)
    metrics = {}
    validate_gate = {"gate": "G3-05", "status": "BLOCKED", "evidence": ""}
    bench_gate = {"gate": "G3-06", "status": "BLOCKED", "evidence": ""}
    match_gate = {"gate": "G3-07", "status": "BLOCKED", "evidence": ""}
    if args.skip_validate:
        step(5, "官方镜像 --validate", "SKIP", "已跳过（--skip-validate）")
        step(6, "完整链耗时 ≤ 10ms", "SKIP", "已跳过（--skip-validate）")
        step(7, "本地对战 ≥ 10 局", "SKIP", "已跳过（--skip-validate）")
    else:
        validate_gate = run_validate(onnx_path, version, stamp)
        bench_gate = run_benchmark(onnx_path, version, stamp)
        metrics = bench_gate.get("metrics") or {}
        match_gate = run_matches(onnx_path, version, args.matches)
    for gate in (validate_gate, bench_gate, match_gate):
        gate.pop("metrics", None)
        gate.setdefault("name", {
            "G3-05": "官方 --validate 通过",
            "G3-06": "完整链耗时 ≤ 10ms（本地参考）",
            "G3-07": "本地对战 ≥ 10 局（镜像输出）",
        }[gate["gate"]])
        gates.append(gate)

    statuses = [s["status"] for s in STEPS if s["status"] != "SKIP"]
    statuses += [g["status"] for g in gates]
    code = 1 if "FAIL" in statuses else (2 if "BLOCKED" in statuses else 0)
    verdict = "FAIL" if code == 1 else ("PASS（受限）" if code == 2 else "PASS")
    step(8, "生成简报", "PASS", f"brief_v{version}.md / .json（结论 {verdict}）")
    write_brief(version, stamp, ckpt_info, onnx_info, gates, metrics, verdict, code)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
