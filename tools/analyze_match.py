"""对局体检报告：把一个 match.json 转成一页 Markdown（附同名 JSON）。

只输出数字，不给策略建议——建议由人写进反馈文档，报表只负责让数字可复现。

用法：
    python3 tools/analyze_match.py --input match.json [--label atk2] [--output outputs/report/]
    python3 tools/analyze_match.py --input r1/match.json r2/match.json ... [--label atk4]
                                       # 多局 → 汇总报告（matches_*.md）

报表七节：结论 / 接触总览 / 冲量分布 / 左右手×部位 / 时间线 / 两侧不对称 / 异常与性能。
多局汇总三节：汇总 / 逐局明细 / 一致性检查（model_id、帧数、门槛）。
退出码：0 正常 / 1 缺关键字段或无法解析 / 2 数据中存在非有限值（NaN、inf）。
"""

import argparse
import datetime
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "report"
BUCKET_S = 10.0


def percentile(values, q):
    """线性插值分位数；空列表返回 0。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (pos - low))


def is_finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def collect(payload):
    """把逐帧 contacts 摊平，同时统计帧级信息。"""
    frames = payload.get("frames") or []
    flat = []
    contact_frames = 0
    for index, frame in enumerate(frames):
        contacts = frame.get("contacts") or []
        if contacts:
            contact_frames += 1
        for contact in contacts:
            if isinstance(contact, dict):
                flat.append(dict(contact, frame_index=index))
    return frames, flat, contact_frames


def count_nonfinite(payload, frames, flat):
    """统计帧级与接触级数值中的非有限值个数（NaN/inf 即仿真或策略异常）。"""
    bad = 0
    for frame in frames:
        for key in ("root_pos", "root_quat", "score"):
            value = frame.get(key)
            if isinstance(value, list):
                bad += sum(1 for item in value if is_finite(item) is False and isinstance(item, (int, float)))
        for key in ("upper_policy_start_ns", "upper_policy_end_ns", "policy_start_ns", "policy_end_ns"):
            value = frame.get(key)
            if isinstance(value, list):
                bad += sum(1 for item in value if not is_finite(item))
            elif isinstance(value, (int, float)) and not is_finite(value):
                bad += 1
    for contact in flat:
        for key in ("peak_force", "closing_speed", "normal_impulse_ns", "normal_impulse_step_ns"):
            if not is_finite(contact.get(key)):
                bad += 1
    return bad


def timing_stats(frames):
    """上层策略单次推理耗时（两侧各记一次，单位 ms）。"""
    values = []
    for frame in frames:
        start, end = frame.get("upper_policy_start_ns"), frame.get("upper_policy_end_ns")
        if isinstance(start, list) and isinstance(end, list):
            values.extend((b - a) / 1e6 for a, b in zip(start, end))
        elif is_finite(start) and is_finite(end):
            values.append((end - start) / 1e6)
    return values


def inp_matrix(flat):
    """左手/右手 × 目标部位：计数与最大冲量。"""
    cells = {}
    for contact in flat:
        hand = contact.get("hand") or "unknown"
        part = contact.get("body_part") or contact.get("target_body") or "unknown"
        impulse = contact.get("normal_impulse_ns") or 0.0
        cell = cells.setdefault((hand, part), {"count": 0, "max": 0.0, "valid": 0})
        cell["count"] += 1
        cell["max"] = max(cell["max"], float(impulse))
        cell["valid"] += 1 if contact.get("valid") else 0
    return cells


def actor_matrix(flat):
    cells = {}
    for contact in flat:
        actor = contact.get("actor")
        cell = cells.setdefault(actor, {"count": 0, "max": 0.0, "valid": 0})
        cell["count"] += 1
        cell["max"] = max(cell["max"], float(contact.get("normal_impulse_ns") or 0.0))
        cell["valid"] += 1 if contact.get("valid") else 0
    return cells


def timeline(flat, total_frames, control_hz):
    buckets = {}
    for contact in flat:
        time_s = contact.get("physics_time_s")
        if not is_finite(time_s):
            bucket = -1
        else:
            bucket = int(float(time_s) // BUCKET_S)
        cell = buckets.setdefault(bucket, {"count": 0, "max": 0.0, "valid": 0})
        cell["count"] += 1
        cell["max"] = max(cell["max"], float(contact.get("normal_impulse_ns") or 0.0))
        cell["valid"] += 1 if contact.get("valid") else 0
    total_s = (total_frames / control_hz) if control_hz else 0.0
    rows = []
    span = int(math.ceil(total_s / BUCKET_S)) if total_s else (max(buckets) + 1 if buckets else 0)
    for index in range(span):
        cell = buckets.get(index) or {"count": 0, "max": 0.0, "valid": 0}
        rows.append({
            "window": f"{index * BUCKET_S:.0f}-{(index + 1) * BUCKET_S:.0f}s",
            **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in cell.items()},
        })
    return rows


def build_report(payload, label, source, missing):
    frames, flat, contact_frames = collect(payload)
    summary = payload.get("summary") or {}
    metadata = payload.get("metadata") or {}
    control_hz = metadata.get("control_hz") or 0
    total_ms = (len(frames) / control_hz * 1000) if control_hz else 0.0

    impulses = [float(c.get("normal_impulse_ns") or 0.0) for c in flat]
    thresholds = [c.get("impulse_threshold_ns") for c in flat if is_finite(c.get("impulse_threshold_ns"))]
    threshold = float(thresholds[0]) if thresholds else 5.0
    valid_count = sum(1 for c in flat if c.get("valid"))
    max_impulse = max(impulses) if impulses else 0.0
    ratio = (max_impulse / threshold) if threshold else 0.0
    peak_force = max((float(c.get("peak_force") or 0.0) for c in flat), default=0.0)
    closing = max((float(c.get("closing_speed") or 0.0) for c in flat), default=0.0)
    first_time = min((float(c["physics_time_s"]) for c in flat if is_finite(c.get("physics_time_s"))), default=0.0)
    reasons = {}
    for contact in flat:
        reason = contact.get("reason") or "unknown"
        reasons[reason] = reasons.get(reason, 0) + 1
    nonfinite = count_nonfinite(payload, frames, flat)
    timings = timing_stats(frames)

    hold = (
        f"得分 {summary.get('final_score')}，胜者 {summary.get('winner')}，结束原因 "
        f"{summary.get('end_reason')}，有效命中 {valid_count} 次。"
        f"最大单次冲量 {max_impulse:.2f} N·s（门槛 {threshold:.1f}，达成率 {ratio * 100:.0f}%）。"
    )

    lines = [
        f"# 对局体检报告：{label}",
        "",
        f"**{hold}**",
        "",
        f"- 数据源：`{source}`",
        f"- 对局标识：`{(summary.get('model_id') or [''])[0][:16]}…` ｜ "
        f"帧数 {len(frames)} ｜ 控制频率 {control_hz or '—'}Hz ｜ 时长 {total_ms / 1000:.1f}s",
        "",
        "## 1. 结论",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        f"| 比分（红/蓝） | {summary.get('final_score')} |",
        f"| 胜者 | {summary.get('winner')} |",
        f"| 结束原因 | {summary.get('end_reason')} |",
        f"| 官方命中 / 摔倒 / 击倒 / KO | {summary.get('hit_count')} / "
        f"{summary.get('fall_count')} / {summary.get('knockdown_count')} / {summary.get('ko_count')} |",
        f"| 有效命中（逐帧统计） | {valid_count} |",
        f"| 最大冲量 / 门槛 | {max_impulse:.2f} / {threshold:.1f} N·s |",
        "",
        "## 2. 接触总览",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 有接触的帧 | {contact_frames} / {len(frames)}"
        f"（{(contact_frames / len(frames) * 100) if frames else 0:.0f}%） |",
        f"| 接触事件数 | {len(flat)} |",
        f"| 有效接触（valid=true） | {valid_count} |",
        f"| 最大峰值法向力 | {peak_force:.1f} N |",
        f"| 最大接近速度 | {closing:.2f} m/s |",
        f"| 首次接触时刻 | {first_time:.3f}s |",
        f"| 未通过原因 | {', '.join(f'{k}={v}' for k, v in sorted(reasons.items())) or '—'} |",
        "",
        "## 3. 冲量分布",
        "",
        "| 分位 | 冲量 N·s |",
        "| --- | --- |",
        f"| 均值 | {sum(impulses) / len(impulses) if impulses else 0:.3f} |",
        f"| P50 | {percentile(impulses, 0.50):.3f} |",
        f"| P90 | {percentile(impulses, 0.90):.3f} |",
        f"| 最大 | {max_impulse:.3f} |",
        f"| 达到门槛的条数 | {sum(1 for v in impulses if v >= threshold)} |",
        f"| 与门槛的差距 | {('还差 %.2f 倍' % (threshold / max_impulse)) if max_impulse else '无接触'} |",
        "",
        "## 4. 左右手 × 目标部位",
        "",
    ]
    cells = inp_matrix(flat)
    if cells:
        parts = sorted({part for _, part in cells})
        hands = sorted({hand for hand, _ in cells})
        lines.append("| 手 \\ 部位 | " + " | ".join(parts) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in parts) + " |")
        for hand in hands:
            row = []
            for part in parts:
                cell = cells.get((hand, part))
                row.append(f"{cell['count']} 次 / max {cell['max']:.2f}" if cell else "—")
            lines.append(f"| {hand} | " + " | ".join(row) + " |")
    else:
        lines.append("（无接触事件）")
    lines += ["", "## 5. 时间线（每 10 秒）", "", "| 窗口 | 接触数 | 最大冲量 N·s | 有效 |", "| --- | --- | --- | --- |"]
    for row in timeline(flat, len(frames), control_hz):
        lines.append(f"| {row['window']} | {row['count']} | {row['max']:.2f} | {row['valid']} |")

    lines += ["", "## 6. 两侧不对称", "", "| actor | 接触数 | 最大冲量 N·s | 有效命中 |", "| --- | --- | --- | --- |"]
    actors = actor_matrix(flat)
    for actor in sorted(actors, key=lambda item: (item is None, item)):
        cell = actors[actor]
        lines.append(f"| {actor} | {cell['count']} | {cell['max']:.2f} | {cell['valid']} |")
    if len(actors) == 2 and all(actors[a]["count"] for a in actors):
        smaller = min(actors[a]["count"] for a in actors)
        larger = max(actors[a]["count"] for a in actors)
        lines.append(f"\n两侧接触次数比 **{larger / smaller:.1f} : 1**。")

    lines += [
        "",
        "## 7. 异常与性能",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        f"| 非有限值（NaN/inf） | {nonfinite} |",
        f"| 上层策略单次推理 | mean {sum(timings) / len(timings) if timings else 0:.2f}ms ｜ "
        f"P95 {percentile(timings, 0.95):.2f}ms ｜ max {max(timings) if timings else 0:.2f}ms |",
        f"| 官方摘要耗时 | {summary.get('avg_latency_ms')} ms ｜ 官方摘要帧率 {summary.get('avg_fps')} |",
        f"| 输出清洗累计 | {metadata.get('competitor_sanitization_total')} |",
        f"| 缺失字段 | {', '.join(missing) if missing else '无'} |",
        "",
    ]

    payload_json = {
        "label": label,
        "source": str(source),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "totals": {
            "frames": len(frames),
            "contact_frames": contact_frames,
            "contacts": len(flat),
            "valid_contacts": valid_count,
            "threshold_ns": threshold,
            "max_impulse_ns": max_impulse,
            "max_peak_force_n": peak_force,
            "max_closing_speed": closing,
            "first_contact_s": first_time,
            "nonfinite": nonfinite,
        },
        "reasons": reasons,
        "impulse": {
            "mean": round(sum(impulses) / len(impulses), 3) if impulses else 0.0,
            "p50": round(percentile(impulses, 0.50), 3),
            "p90": round(percentile(impulses, 0.90), 3),
            "max": round(max_impulse, 3),
        },
        "hands": {f"{hand}/{part}": cell for (hand, part), cell in cells.items()},
        "actors": {str(actor): cell for actor, cell in actors.items()},
        "timeline": timeline(flat, len(frames), control_hz),
        "timing_ms": {
            "mean": round(sum(timings) / len(timings), 3) if timings else 0.0,
            "p95": round(percentile(timings, 0.95), 3),
            "max": round(max(timings), 3) if timings else 0.0,
        },
        "missing_fields": missing,
    }
    return "\n".join(lines), payload_json


def build_batch_report(rows, label, sources, errors):
    """多局汇总：逐局明细 + 合计/分散度 + 一致性检查（model_id、帧数、门槛）。"""
    n = len(rows)

    def stat(values):
        return {
            "sum": sum(values),
            "mean": round(sum(values) / n, 2) if n else 0.0,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
        }

    valid = [r["totals"]["valid_contacts"] for r in rows]
    contacts = [r["totals"]["contacts"] for r in rows]
    frames = [r["totals"]["frames"] for r in rows]
    maximp = [r["totals"]["max_impulse_ns"] for r in rows]
    p90 = [r["impulse"]["p90"] for r in rows]
    nonfinite = sum(r["totals"]["nonfinite"] for r in rows)
    falls = sum(sum(int(v) for v in (r["summary"].get("fall_count") or [0, 0])) for r in rows)
    knockdown = sum(sum(int(v) for v in (r["summary"].get("knockdown_count") or [0, 0])) for r in rows)
    ko = sum(sum(int(v) for v in (r["summary"].get("ko_count") or [0, 0])) for r in rows)
    scored = sum(1 for r in rows if sum(int(v) for v in (r["summary"].get("final_score") or [0, 0])) > 0)
    left = sum(c["count"] for r in rows for k, c in r["hands"].items() if k.startswith("left/"))
    right = sum(c["count"] for r in rows for k, c in r["hands"].items() if k.startswith("right/"))
    model_ids = sorted({(r["summary"].get("model_id") or [""])[0] for r in rows})

    ratio = f"{right / left:.1f} : 1" if left else "—"
    hold = (
        f"共 {n} 局：有效命中合计 {sum(valid)}（场均 {sum(valid) / n:.1f}，最少 {min(valid)}，最多 {max(valid)}）；"
        f"有得分 {scored}/{n} 局；全局最大冲量 {max(maximp):.2f} N·s；摔倒合计 {falls}。"
    )
    lines = [f"# 多局汇总报告：{label}", "", f"**{hold}**", "", "- 数据源：" + "、".join(f"`{s}`" for s in sources)]
    if errors:
        lines.append(f"- 跳过 {len(errors)} 个（{'; '.join(errors)}）")
    lines += [
        "",
        "## 1. 汇总",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 局数 | {n} |",
        f"| 有效命中（合计 / 场均 / 最少 / 最多） | {stat(valid)['sum']} / {stat(valid)['mean']} / "
        f"{stat(valid)['min']} / {stat(valid)['max']} |",
        f"| 有得分的局 | {scored} / {n} |",
        f"| 接触事件（场均） | {stat(contacts)['mean']} |",
        f"| 单局最大冲量（最少 / 场均 / 最多） | {min(maximp):.2f} / {stat(maximp)['mean']:.2f} / "
        f"{max(maximp):.2f} N·s |",
        f"| 单局 P90 冲量（场均） | {stat(p90)['mean']:.3f} N·s |",
        f"| 摔倒 / 击倒 / KO | {falls} / {knockdown} / {ko} |",
        f"| 左右手接触（合计） | left {left} / right {right}（{ratio}） |",
        f"| 非有限值（合计） | {nonfinite} |",
        "",
        "## 2. 逐局明细",
        "",
        "| # | 文件 | 比分 | 有效命中 | 接触事件 | 最大冲量 N·s | 摔倒 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, row in enumerate(rows, 1):
        summary = row["summary"]
        totals = row["totals"]
        score = summary.get("final_score") or []
        lines.append(
            f"| {index} | {row['source'].split('/')[-1]} | {' : '.join(str(v) for v in score) or '—'} | "
            f"{totals['valid_contacts']} | {totals['contacts']} | {totals['max_impulse_ns']:.2f} | "
            f"{sum(int(v) for v in (summary.get('fall_count') or [0, 0]))} |"
        )
    lines += ["", "## 3. 一致性检查", "", "| 项 | 结果 |", "| --- | --- |"]
    if len(model_ids) == 1:
        lines.append(f"| model_id | 全部 {n} 局一致：`{model_ids[0][:16]}…` |")
    else:
        lines.append(f"| model_id | **不一致（{len(model_ids)} 个）**："
                     + "、".join(f"`{m[:12]}…`" for m in model_ids) + " |")
    lines.append(f"| 帧数 | {'全部 ' + str(frames[0]) if len(set(frames)) == 1 else '不一致：' + str(sorted(set(frames)))} |")
    thresholds = sorted({r["totals"]["threshold_ns"] for r in rows})
    lines.append(f"| 冲量门槛 | {'、'.join(f'{t:.1f}' for t in thresholds)} N·s |")
    lines += ["", f"**非有限值合计 {nonfinite}**，{'需排查' if nonfinite else '无异常'}。"]

    payload_json = {
        "label": label,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "matches": n,
        "sources": [str(s) for s in sources],
        "skipped": errors,
        "aggregate": {
            "valid_contacts": stat(valid),
            "contacts": stat(contacts),
            "frames": stat(frames),
            "max_impulse_ns": stat(maximp),
            "p90_impulse_ns": stat(p90),
            "scored_matches": scored,
            "falls": falls,
            "knockdowns": knockdown,
            "ko": ko,
            "hands_left": left,
            "hands_right": right,
            "nonfinite": nonfinite,
        },
        "consistency": {"model_ids": model_ids, "frame_counts": sorted(set(frames))},
        "rows": rows,
    }
    return "\n".join(lines), payload_json


def main():
    parser = argparse.ArgumentParser(description="match.json → Markdown 体检报告（可一次多局）")
    parser.add_argument("--input", required=True, nargs="+", help="一个或多个 match.json")
    parser.add_argument("--label", default="", help="报告标签；单局默认取文件名 stem，多局默认 batch")
    parser.add_argument("--output", default=str(DEFAULT_OUT), help="输出目录（默认 outputs/report/）")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    loaded, errors = [], []
    for raw in args.input:
        source = Path(raw)
        if not source.is_file():
            errors.append(f"输入不存在：{source}")
            continue
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"无法解析 JSON（{source.name}）：{exc}")
            continue
        missing = [key for key in ("frames", "summary") if key not in payload]
        if missing:
            errors.append(f"缺少关键字段（{source.name}）：{', '.join(missing)}")
            continue
        loaded.append((source, payload))

    if not loaded:
        for line in errors:
            print(f"[FAIL] {line}")
        raise SystemExit(1)
    for line in errors:
        print(f"[WARN] 跳过 {line}")

    if len(loaded) == 1:
        source, payload = loaded[0]
        label = args.label or source.stem
        text, data = build_report(payload, label, source, [])
        md_path = out_dir / f"match_{label}_{stamp}.md"
        json_path = out_dir / f"match_{label}_{stamp}.json"
        md_path.write_text(text, encoding="utf-8")
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        totals = data["totals"]
        print(f"接触帧 {totals['contact_frames']}/{totals['frames']} ｜ 事件 {totals['contacts']} ｜ "
              f"有效 {totals['valid_contacts']} ｜ 最大冲量 {totals['max_impulse_ns']:.2f} N·s")
        print(f"报告：{md_path}\n      {json_path}")
        raise SystemExit(2 if totals["nonfinite"] else 0)

    rows = []
    for source, payload in loaded:
        _, row = build_report(payload, source.stem, source, [])
        rows.append(row)
    label = args.label or "batch"
    text, data = build_batch_report(rows, label, [source for source, _ in loaded], errors)
    md_path = out_dir / f"matches_{label}_{stamp}.md"
    json_path = out_dir / f"matches_{label}_{stamp}.json"
    md_path.write_text(text, encoding="utf-8")
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    agg = data["aggregate"]
    print(f"局数 {data['matches']} ｜ 有效命中 合计 {agg['valid_contacts']['sum']}"
          f"（场均 {agg['valid_contacts']['mean']}，最少 {agg['valid_contacts']['min']}） ｜ "
          f"有得分 {agg['scored_matches']}/{data['matches']} 局 ｜ "
          f"非有限值 {agg['nonfinite']}")
    print(f"报告：{md_path}\n      {json_path}")
    raise SystemExit(2 if agg["nonfinite"] else 0)


if __name__ == "__main__":
    main()
