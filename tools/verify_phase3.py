#!/usr/bin/env python3
"""阶段三独立检验入口（阶段审查文档 G3-01~G3-09）。

纯 CPU 项（结构/大小/一致性/无动态轴）不依赖 Docker 与 GPU；
镜像项（官方 --validate、完整链基准、对战回放）缺失时记 BLOCKED，不得记 PASS。

用法：
    python3 tools/verify_phase3.py
    python3 tools/verify_phase3.py --onnx models/policy_v1_20261103.onnx \
        --ckpt outputs/train/ckpt_0016.pt --json outputs/verify/phase3.json

退出码：0 全部通过 / 1 存在 FAIL / 2 无 FAIL 但存在 BLOCKED。
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVAL_DIR = ROOT / "outputs" / "eval"
MODEL_DIR = ROOT / "models"
SIZE_LIMIT = 50_000_000
PARITY_ATOL = 1e-4
BENCH_LIMIT_MS = 10.0
OPSET_MIN = 13

CHECKS = []


def add(gate, name, status, detail):
    CHECKS.append({"gate": gate, "name": name, "status": status, "detail": detail})


def report_deps(onnx_path, ckpt):
    deps = []

    try:
        import onnx

        deps.append(("onnx", True, onnx.__version__))
    except Exception as exc:  # noqa: BLE001
        deps.append(("onnx", False, str(exc)))

    try:
        import onnxruntime

        deps.append(("onnxruntime", True, onnxruntime.__version__))
    except Exception as exc:  # noqa: BLE001
        deps.append(("onnxruntime", False, str(exc)))

    try:
        import torch

        deps.append(("torch", True, torch.__version__))
    except Exception as exc:  # noqa: BLE001
        deps.append(("torch", False, str(exc)))

    deps.append(("ONNX 模型", bool(onnx_path and Path(onnx_path).is_file()), str(onnx_path)))
    deps.append(
        ("checkpoint（G3-04 用）", bool(ckpt and Path(ckpt).is_file()), str(ckpt) or "未指定")
    )

    print("依赖清单：")
    for name, ok, note in deps:
        print(f"  [{'READY' if ok else 'MISSING'}] {name} {note}")
    print()


def pick_onnx(explicit):
    if explicit:
        return Path(explicit)
    candidates = sorted(MODEL_DIR.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_graph(onnx_path):
    import onnx

    return onnx.load(str(onnx_path))


def dims_of(value_info):
    return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]


def dynamic_dims_of(value_info):
    return [dim.dim_param for dim in value_info.type.tensor_type.shape.dim if dim.dim_param]


def check_io(graph):
    """G3-01 / G3-02：输入 obs[1,1097]、输出 target_pose[1,225]，float32。"""
    try:
        inputs = list(graph.graph.input)
        outputs = list(graph.graph.output)
        if len(inputs) != 1 or len(outputs) != 1:
            add("G3-01", "ONNX 输入 obs[1,1097] float32", "FAIL",
                f"输入 {len(inputs)} 个 / 输出 {len(outputs)} 个，应为单入单出")
            return
        inp, out = inputs[0], outputs[0]
        ok_in = inp.name == "obs" and dims_of(inp) == [1, 1097] and inp.type.tensor_type.elem_type == 1
        ok_out = (
            out.name == "target_pose"
            and dims_of(out) == [1, 225]
            and out.type.tensor_type.elem_type == 1
        )
        add("G3-01", "ONNX 输入 obs[1,1097] float32",
            "PASS" if ok_in else "FAIL",
            f"name={inp.name} shape={dims_of(inp)} elem_type={inp.type.tensor_type.elem_type}")
        add("G3-02", "ONNX 输出 target_pose[1,225] float32",
            "PASS" if ok_out else "FAIL",
            f"name={out.name} shape={dims_of(out)} elem_type={out.type.tensor_type.elem_type}")
    except Exception as exc:  # noqa: BLE001
        add("G3-01", "ONNX 输入 obs[1,1097] float32", "FAIL", str(exc))
        add("G3-02", "ONNX 输出 target_pose[1,225] float32", "FAIL", str(exc))


def check_size(onnx_path):
    """G3-03：严格 < 50,000,000 字节。"""
    try:
        size = Path(onnx_path).stat().st_size
        status = "PASS" if size < SIZE_LIMIT else "FAIL"
        add("G3-03", "文件大小 < 50,000,000 字节", status,
            f"{size} 字节（{size / 1e6:.2f} MB，上限 {SIZE_LIMIT}）")
    except Exception as exc:  # noqa: BLE001
        add("G3-03", "文件大小 < 50,000,000 字节", "FAIL", str(exc))


def check_static_graph(graph):
    """G3-08：无动态 shape、无 external data、opset ≥ 13。"""
    try:
        versions = {imp.domain: imp.version for imp in graph.opset_import}
        dynamic = []
        for value in list(graph.graph.input) + list(graph.graph.output):
            if dynamic_dims_of(value):
                dynamic.append(value.name)
        external = [
            init.name
            for init in graph.graph.initializer
            if init.data_location == 1  # EXTERNAL_DATA
        ]
        opset = max(versions.values()) if versions else 0
        ok = not dynamic and not external and opset >= OPSET_MIN
        add("G3-08", "无动态轴 / 无 external data / opset ≥ 13",
            "PASS" if ok else "FAIL",
            f"opset={opset} 动态维={dynamic or '无'} 外部权重={external or '无'}")
    except Exception as exc:  # noqa: BLE001
        add("G3-08", "无动态轴 / 无 external data / opset ≥ 13", "FAIL", str(exc))


def check_parity(onnx_path, ckpt, samples=5):
    """G3-04：PyTorch 与 ONNX Runtime(CPU) 最大绝对误差 < 1e-4。"""
    if not ckpt or not Path(ckpt).is_file():
        add("G3-04", "PyTorch ↔ ONNX 最大误差 < 1e-4", "BLOCKED",
            "未提供 checkpoint，无法做数值一致性比对")
        return
    try:
        import numpy as np
        import onnxruntime as ort
        import torch

        from src.export.export_onnx import load_policy
        from src.models.policy import OBS_DIM, contract_ok

        policy = load_policy(ckpt)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        generator = torch.Generator().manual_seed(0)
        worst, contract = 0.0, True
        for _ in range(samples):
            obs = torch.randn(1, OBS_DIM, generator=generator) * 3.0
            with torch.no_grad():
                expected = policy(obs).numpy()
            actual = session.run(["target_pose"], {"obs": obs.numpy()})[0]
            worst = max(worst, float(np.abs(expected - actual).max()))
            contract = contract and contract_ok(torch.from_numpy(actual))
        status = "PASS" if worst < PARITY_ATOL and contract else "FAIL"
        add("G3-04", "PyTorch ↔ ONNX 最大误差 < 1e-4", status,
            f"{samples} 组随机输入，max |Δ|={worst:.2e}，X/Z 契约={contract}")
    except Exception as exc:  # noqa: BLE001
        add("G3-04", "PyTorch ↔ ONNX 最大误差 < 1e-4", "FAIL", str(exc))


def find_latest(pattern, parse_json):
    """取 outputs/eval 下最新匹配文件；解析 JSON 时容忍日志前缀文本。"""
    candidates = sorted(EVAL_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None, None
    path = candidates[0]
    if not parse_json:
        return path, None
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\{", text)
    if not match:
        return path, None
    try:
        return path, json.loads(text[match.start():])
    except Exception:  # noqa: BLE001
        return path, None


def check_validate():
    """G3-05：官方镜像 --validate 通过（日志 + JSON 双证据）。"""
    log, _ = find_latest("validate_v*.log", parse_json=False)
    path, payload = find_latest("validation_v*.json", parse_json=True)
    if log is None and path is None:
        add("G3-05", "官方 --validate 通过", "BLOCKED",
            "outputs/eval 下无 validate_v*.log / validation_v*.json")
        return
    if payload is None:
        add("G3-05", "官方 --validate 通过", "BLOCKED",
            f"{path.name if path else log.name} 存在但无法解析 JSON")
        return
    status = payload.get("status")
    size = (payload.get("size_check") or {}).get("total_bytes")
    add("G3-05", "官方 --validate 通过", "PASS" if status == "PASS" else "FAIL",
        f"{path.name} status={status} size={size} → 日志 {log.name if log else '缺失'}")


def check_benchmark():
    """G3-06：完整链耗时 ≤ 10ms（本地参考，非平台验收）。"""
    path, payload = find_latest("benchmark_v*.log", parse_json=True)
    if path is None or payload is None:
        add("G3-06", "完整链耗时 ≤ 10ms（本地参考）", "BLOCKED",
            "outputs/eval 下无可解析的 benchmark_v*.log（需镜像 --benchmark 或官方 chain_benchmark）")
        return
    mean = float(payload.get("mean_ms") or 0.0)
    p99 = payload.get("p99_ms")
    maximum = payload.get("maximum_ms")
    ok = bool(payload.get("timing_completed")) and mean <= BENCH_LIMIT_MS
    add("G3-06", "完整链耗时 ≤ 10ms（本地参考）", "PASS" if ok else "FAIL",
        f"mean={mean:.2f}ms p99={p99:.2f}ms max={maximum:.2f}ms "
        f"gpu={payload.get('gpu')} → {path.name}")


def check_matches(onnx_path):
    """G3-07：镜像本地对战 ≥ 10 局，无 NaN、无异常倒地。

    `summary.model_id[0]` 记录红方 ONNX 的 sha256，据此只统计本次提交模型的
    对局，避免把其他模型的旧对局算进本阶段验收。
    """
    from src.export.export_onnx import file_sha256

    directory = EVAL_DIR / "match"
    runs = sorted(
        p for p in directory.glob("**/match.json") if p.is_file()
    ) if directory.exists() else []
    if not runs:
        add("G3-07", "本地对战 ≥ 10 局（镜像输出）", "BLOCKED",
            "outputs/eval/match 下无 match.json；需 scripts/run_match.sh 跑 ≥10 局")
        return
    ours = file_sha256(onnx_path) if onnx_path and Path(onnx_path).is_file() else ""
    matched, foreign, bad = 0, 0, []
    for path in runs:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "NaN" in text or "Infinity" in text:
            bad.append(f"{path.name}:非有限值")
            continue
        try:
            payload = json.loads(text)
        except Exception:  # noqa: BLE001
            bad.append(f"{path.name}:无法解析")
            continue
        ids = (payload.get("summary") or {}).get("model_id") or []
        if ours and (not ids or ids[0] != ours):
            foreign += 1
            continue
        matched += 1
        summary = payload.get("summary") or {}
        if summary.get("fall_count") and any(int(v) for v in summary["fall_count"]):
            bad.append(f"{path.name}:摔倒 {summary['fall_count']}")
    scope = f"本模型 {matched} 局（其他模型旧对局 {foreign} 局不计）"
    if bad:
        add("G3-07", "本地对战 ≥ 10 局（镜像输出）", "FAIL", f"{scope}；" + "；".join(bad[:5]))
    elif matched < 10:
        add("G3-07", "本地对战 ≥ 10 局（镜像输出）", "BLOCKED",
            f"{scope}，需累计 ≥10 局")
    else:
        add("G3-07", "本地对战 ≥ 10 局（镜像输出）", "PASS",
            f"{scope}，无 NaN、无异常倒地")


def run_checks(onnx_path, ckpt="", samples=5, evidence=True):
    """跑全部门禁并返回结果列表（供 `tools/accept_ckpt.py` 编排复用）。

    `evidence=False` 只做纯 CPU 项（G3-01/02/03/04/08），不读取镜像证据。
    """
    CHECKS.clear()
    if onnx_path is None or not Path(onnx_path).is_file():
        for gate, name in (
            ("G3-01", "ONNX 输入 obs[1,1097] float32"),
            ("G3-02", "ONNX 输出 target_pose[1,225] float32"),
            ("G3-03", "文件大小 < 50,000,000 字节"),
            ("G3-08", "无动态轴 / 无 external data / opset ≥ 13"),
        ):
            add(gate, name, "BLOCKED", "models/ 下无 ONNX 文件")
        add("G3-04", "PyTorch ↔ ONNX 最大误差 < 1e-4", "BLOCKED", "缺少 ONNX 文件")
        return list(CHECKS)

    graph = load_graph(onnx_path)
    check_io(graph)
    check_size(onnx_path)
    check_static_graph(graph)
    check_parity(onnx_path, ckpt, samples)
    if evidence:
        check_validate()
        check_benchmark()
        check_matches(onnx_path)
    return list(CHECKS)


def main():
    parser = argparse.ArgumentParser(description="阶段三独立检验入口")
    parser.add_argument("--onnx", default="", help="默认取 models/ 下最新的 .onnx")
    parser.add_argument("--ckpt", default="", help="G3-04 一致性比对用 checkpoint")
    parser.add_argument("--json", default="outputs/verify/phase3.json")
    args = parser.parse_args()

    onnx_path = pick_onnx(args.onnx)
    report_deps(onnx_path, args.ckpt)

    run_checks(onnx_path, args.ckpt)

    cpu_items = [c for c in CHECKS if c["gate"] in ("G3-01", "G3-02", "G3-03", "G3-04", "G3-08")]
    cpu_ok = all(c["status"] == "PASS" for c in cpu_items)
    add("G3-09", "独立检验入口可运行（纯 CPU 项全 PASS）",
        "PASS" if cpu_ok else "FAIL",
        f"纯 CPU 项 {sum(c['status'] == 'PASS' for c in cpu_items)}/{len(cpu_items)} PASS，"
        f"BLOCKED {sum(c['status'] == 'BLOCKED' for c in CHECKS)} 项")

    has_fail = any(c["status"] == "FAIL" for c in CHECKS)
    has_blocked = any(c["status"] == "BLOCKED" for c in CHECKS)
    code = 1 if has_fail else (2 if has_blocked else 0)

    print(f"{'门禁':7} {'状态':8} 检查项")
    for check in CHECKS:
        print(f"{check['gate']:7} {check['status']:8} {check['name']}  →  {check['detail']}")

    verdict = "FAIL" if has_fail else ("PASS（受限）" if has_blocked else "PASS")
    print(f"\n结论：{verdict}  退出码={code}")

    payload = {
        "phase": 3,
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "onnx": str(onnx_path) if onnx_path else "",
        "ckpt": args.ckpt,
        "checks": CHECKS,
        "exit_code": code,
        "verdict": verdict,
    }
    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"结果已写入 {out}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
