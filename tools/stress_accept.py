"""抗压测试：构造 12 个坏 ONNX，检查 `scripts/accept.sh --onnx` 是否优雅拦截。

每个用例的判据四件：不出现 Traceback、退出码为 1、失败简报结论为 FAIL、models/ 无临时残留。
用例文件放 `outputs/stress/cases/`，结果表放 `outputs/stress/`。

用法：
    python3 tools/stress_accept.py [--base models/policy_v1_20261103.onnx]
退出码：0 全部用例符合预期 / 1 存在不符合预期的用例。
"""

import argparse
import datetime
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRESS_DIR = ROOT / "outputs" / "stress"
CASE_DIR = STRESS_DIR / "cases"
REVIEW_DIR = ROOT / "outputs" / "review"
MODEL_DIR = ROOT / "models"
TIMEOUT_S = 240


def _dump(graph, path):
    import onnx

    onnx.save(graph, str(path))


def build_cases(base_path):
    """返回 [(name, 说明, 构造函数)]，构造函数返回用例文件路径。"""

    def empty():
        path = CASE_DIR / "empty.onnx"
        path.write_bytes(b"")
        return path

    raw = base_path.read_bytes()

    def truncate_1mb():
        path = CASE_DIR / "truncated_1mb.onnx"
        path.write_bytes(raw[: 1024 * 1024])
        return path

    def truncate_half():
        path = CASE_DIR / "truncated_half.onnx"
        path.write_bytes(raw[: len(raw) // 2])
        return path

    def zip_header_only():
        path = CASE_DIR / "zip_header_only.onnx"
        path.write_bytes(raw[:22])
        return path

    def tar_gz_renamed():
        path = CASE_DIR / "targz_renamed.onnx"
        path.write_bytes(gzip.compress(b"not a model, just a gzip stream" * 400))
        return path

    def garbage_21():
        path = CASE_DIR / "garbage_21b.onnx"
        path.write_bytes(bytes(range(21)))
        return path

    def load_base():
        import onnx

        return onnx.load(str(base_path))

    def wrong_input_name():
        graph = load_base()
        graph.graph.input[0].name = "observation"
        path = CASE_DIR / "wrong_input_name.onnx"
        _dump(graph, path)
        return path

    def wrong_output_name():
        graph = load_base()
        graph.graph.output[0].name = "action"
        path = CASE_DIR / "wrong_output_name.onnx"
        _dump(graph, path)
        return path

    def input_dim_1096():
        graph = load_base()
        graph.graph.input[0].type.tensor_type.shape.dim[-1].dim_value = 1096
        path = CASE_DIR / "input_dim_1096.onnx"
        _dump(graph, path)
        return path

    def float16_input():
        import onnx

        graph = load_base()
        graph.graph.input[0].type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
        path = CASE_DIR / "float16_input.onnx"
        _dump(graph, path)
        return path

    def dynamic_axis():
        graph = load_base()
        graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
        path = CASE_DIR / "dynamic_axis.onnx"
        _dump(graph, path)
        return path

    def oversized_60mb():
        import numpy as np
        import onnx
        from onnx import numpy_helper

        graph = load_base()
        blob = numpy_helper.from_array(np.zeros(15_000_000, dtype=np.float32), name="junk_padding")
        graph.graph.initializer.append(blob)
        path = CASE_DIR / "oversized_60mb.onnx"
        _dump(graph, path)
        return path

    return [
        ("empty", "空文件（0 字节）", empty),
        ("truncated_1mb", "截断到 1MB", truncate_1mb),
        ("truncated_half", "截断到 50%", truncate_half),
        ("zip_header_only", "只剩 22 字节 zip 头", zip_header_only),
        ("targz_renamed", "gzip 流改名为 .onnx", tar_gz_renamed),
        ("garbage_21b", "21 字节垃圾数据", garbage_21),
        ("wrong_input_name", "输入名改成 observation", wrong_input_name),
        ("wrong_output_name", "输出名改成 action", wrong_output_name),
        ("input_dim_1096", "输入维度改成 1096", input_dim_1096),
        ("float16_input", "输入 dtype 改成 float16", float16_input),
        ("dynamic_axis", "batch 轴改动态（dim_param）", dynamic_axis),
        ("oversized_60mb", "文件超过 50MB（塞 60MB 假权重）", oversized_60mb),
    ]


def run_case(name, path):
    started = time.time()
    proc = subprocess.run(
        [str(ROOT / "scripts" / "accept.sh"), "--onnx", str(path), "--skip-validate"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    elapsed = time.time() - started
    output = (proc.stdout or "") + (proc.stderr or "")
    step3 = ""
    for line in output.splitlines():
        if line.startswith("[3/8]"):
            step3 = line.strip()
            break
    brief = REVIEW_DIR / f"brief_{name}.md"
    brief_text = brief.read_text(encoding="utf-8") if brief.exists() else ""
    leftovers = [p.name for p in MODEL_DIR.glob(".accept_tmp_*.onnx")]
    return {
        "name": name,
        "size_bytes": path.stat().st_size,
        "exit_code": proc.returncode,
        "traceback": "Traceback (most recent call last)" in output,
        "brief_fail": "**结论：FAIL**" in brief_text,
        "step3": step3,
        "leftovers": leftovers,
        "seconds": round(elapsed, 1),
    }


def verdict_of(case, row):
    problems = []
    if row["traceback"]:
        problems.append("出现 Traceback")
    if row["exit_code"] != 1:
        problems.append(f"退出码 {row['exit_code']}（应为 1）")
    if not row["brief_fail"]:
        problems.append("失败简报缺失或结论不是 FAIL")
    if row["leftovers"]:
        problems.append("models/ 有临时残留：" + ", ".join(row["leftovers"]))
    row["problems"] = problems
    row["status"] = "PASS" if not problems else "FAIL"
    return row


def main():
    parser = argparse.ArgumentParser(description="accept.sh 坏文件抗压测试")
    parser.add_argument("--base", default=str(MODEL_DIR / "policy_v1_20261103.onnx"))
    args = parser.parse_args()
    base_path = Path(args.base)
    if not base_path.is_file():
        print(f"[FAIL] 基准模型不存在：{base_path}")
        raise SystemExit(1)

    CASE_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, desc, builder in build_cases(base_path):
        try:
            path = builder()
        except Exception as exc:  # noqa: BLE001  构造失败也要记录，不中断整轮
            rows.append({"name": name, "desc": desc, "status": "FAIL", "problems": [f"构造失败：{exc}"]})
            continue
        row = run_case(name, path)
        row["desc"] = desc
        rows.append(verdict_of({"name": name}, row))
        print(f"[{row['status']}] {name:18s} exit={row['exit_code']} "
              f"traceback={row['traceback']} {row['step3'][:80]}")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    failed = [r for r in rows if r["status"] != "PASS"]
    md = [f"# 抗压测试报告 {stamp}", "", f"基准模型：`{base_path}`", "",
          "| 用例 | 说明 | 退出码 | Traceback | 简报 FAIL | 第 3 步输出 | 耗时 | 结论 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        if row.get("status") == "FAIL" and "构造失败" in "".join(row.get("problems", [""])):
            md.append(f"| `{row['name']}` | {row['desc']} | — | — | — | 构造失败 | — | **FAIL** |")
            continue
        md.append(
            f"| `{row['name']}` | {row['desc']} | {row['exit_code']} | "
            f"{'是' if row['traceback'] else '否'} | {'是' if row['brief_fail'] else '否'} | "
            f"{row['step3'].replace('|', '/')[:90]} | {row['seconds']}s | **{row['status']}** |"
        )
    md += ["", f"合计 {len(rows)} 例，PASS {len(rows) - len(failed)}，FAIL {len(failed)}。", ""]
    (STRESS_DIR / f"stress_{stamp}.md").write_text("\n".join(md), encoding="utf-8")
    (STRESS_DIR / f"stress_{stamp}.json").write_text(
        json.dumps({"generated_at": stamp, "base": str(base_path), "cases": rows},
                   ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n报告：{STRESS_DIR / f'stress_{stamp}.md'}")
    if failed:
        print("未通过：" + ", ".join(r["name"] for r in failed))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    sys.exit(main())
