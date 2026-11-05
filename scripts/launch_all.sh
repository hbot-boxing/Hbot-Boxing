#!/usr/bin/env bash
# 总控 Launch：验收 → 端侧算子体检 → 对战体检 → 总览。
#
# 用法：
#   scripts/launch_all.sh models/policy_v1_20261103.onnx
#   scripts/launch_all.sh models/policy_v1_20261103.onnx --skip-validate
#   scripts/launch_all.sh models/xxx.onnx --match-json /tmp/atk2/atk2-full/match.json
#
# 参数：<model.onnx> 必填；--match-json <json>（可重复，多局走汇总报告）只在本脚本消费；
#       --skip-validate / --matches N / --version N / --opset N /
#       --evidence-validation <json> --evidence-benchmark <json> 透传给 accept.sh（白名单，
#       避免 --match-json 被误传成 --matches）。
#
# 退出码：0 全绿 / 1 验收或对战体检 FAIL / 2 无 FAIL 但有 BLOCKED 或算子警告。
#
# 不用 set -e（子脚本 rc=1/2 是正常信号）；不 source train_env.sh（accept.sh 内部自己 source，
# 而它是 exec，环境不会回传父进程，这里只需自己解析 HB_PY）。

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${HB_PY:-$HOME/hb-env/bin/python}"
STAMP="$(date +%Y%m%d-%H%M%S)"

usage() {
  cat <<'EOF'
用法：scripts/launch_all.sh <model.onnx> [--match-json <match.json> ...] [--skip-validate]
                            [--matches N] [--version N] [--opset N]
                            [--evidence-validation <json>] [--evidence-benchmark <json>]
EOF
}

# 打印总览并生成 outputs/review/launch_<model>_<stamp>.md，同时设置全局 CODE
summary() {
  local concl code
  if [[ $fail -eq 1 ]]; then
    code=1; concl="FAIL（存在失败项）"
  elif [[ $blocked -eq 1 && $status2 == "WARN" ]]; then
    code=2; concl="受限（有 BLOCKED 项）+ 算子警告"
  elif [[ $blocked -eq 1 ]]; then
    code=2; concl="受限（有 BLOCKED 项）"
  elif [[ $status2 == "WARN" ]]; then
    code=2; concl="通过，但有算子警告"
  else
    code=0; concl="全绿"
  fi

  local lines=(
    "1 验收接口/耗时  [$status1]  exit=${rc1:-—}"
    "  结论     $verdict1"
    "  简报     ${BRIEF#$ROOT/}"
    "2 端侧算子体检   [$status2]  exit=${rc2:-—}"
    "  结论     ${note2:-—}"
    "3 对战体检       [$status3]  exit=${rc3:-—}"
    "  数据     ${note3:-—}"
  )

  echo
  echo "================ Launch 总览 ================"
  echo "模型：$(basename "$MODEL")    时间：$STAMP"
  printf '%s\n' "${lines[@]}"
  echo "---------------------------------------------"
  echo "结论：$concl ｜ 退出码 $code"

  LAUNCH_MD="$ROOT/outputs/review/launch_${STEM}_${STAMP}.md"
  mkdir -p "$ROOT/outputs/review"
  {
    echo "# Launch 总览：$(basename "$MODEL")"
    echo
    echo '```'
    printf '%s\n' "${lines[@]}"
    echo "结论：$concl ｜ 退出码 $code"
    echo '```'
  } > "$LAUNCH_MD"
  echo "总览已落盘：${LAUNCH_MD#$ROOT/}"
  CODE=$code
}

MODEL="${1:-}"
if [[ -z "$MODEL" || "$MODEL" == --* ]]; then
  usage
  exit 1
fi
shift

MATCH_JSONS=()
VERSION=""
PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --match-json)
      [[ -n "${2:-}" ]] || { echo "[FAIL] --match-json 缺少路径"; exit 1; }
      MATCH_JSONS+=("$2"); shift 2 ;;
    --skip-validate)
      PASSTHRU+=("$1"); shift ;;
    --version)
      VERSION="${2:-}"
      [[ -n "$VERSION" ]] || { echo "[FAIL] --version 缺少值"; exit 1; }
      PASSTHRU+=("--version" "$VERSION"); shift 2 ;;
    --matches|--opset|--evidence-validation|--evidence-benchmark)
      [[ -n "${2:-}" ]] || { echo "[FAIL] $1 缺少值"; exit 1; }
      PASSTHRU+=("$1" "$2"); shift 2 ;;
    *)
      echo "[FAIL] 未知参数：$1"
      usage
      exit 1 ;;
  esac
done

# 0 前置校验
[[ -f "$MODEL" ]] || { echo "[FAIL] 模型不存在：$MODEL"; exit 1; }
[[ "$MODEL" == *.onnx ]] || { echo "[FAIL] 只接受 .onnx：$MODEL"; exit 1; }
[[ -r "$MODEL" ]] || { echo "[FAIL] 模型不可读：$MODEL"; exit 1; }
MODEL="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"   # docker bind 挂载要绝对路径
STEM="$(basename "$MODEL" .onnx)"
LABEL="$STEM"
[[ -n "$VERSION" ]] && LABEL="v$VERSION"          # 复用 accept_ckpt.py 的简报命名规则
BRIEF="$ROOT/outputs/review/brief_${LABEL}.md"

# 1 验收（FAIL 立即停，不跑 2/3）
echo
echo "[1/3] 验收接口/耗时：scripts/accept.sh --onnx $(basename "$MODEL")"
"$ROOT/scripts/accept.sh" --onnx "$MODEL" "${PASSTHRU[@]}"
rc1=$?
verdict1="$(grep -m1 -o '结论：[^｜]*' "$BRIEF" 2>/dev/null | sed 's/结论：//;s/\*//g;s/ *$//')"
[[ -n "$verdict1" ]] || verdict1="—"

if [[ $rc1 -eq 1 ]]; then
  status1="FAIL"; fail=1; blocked=0
  status2="SKIP"; note2="第 1 步 FAIL，已停止"
  status3="SKIP"; note3="第 1 步 FAIL，已停止"
  summary
  exit "$CODE"
fi
[[ $rc1 -eq 2 ]] && { status1="受限"; blocked=1; } || { status1="PASS"; blocked=0; }
fail=0

# 2 端侧算子体检（只警告，不挡流程）
echo
echo "[2/3] 端侧算子体检：tools/check_bpu_ops.py"
ops_out="$("$PY" "$ROOT/tools/check_bpu_ops.py" "$MODEL")"
rc2=$?
printf '%s\n' "$ops_out"
note2="$(printf '%s\n' "$ops_out" | grep -m1 '^结论：' | sed 's/结论：//')"
if [[ $rc2 -eq 2 ]]; then
  status2="WARN"
  bad_ops="$(printf '%s\n' "$ops_out" | grep '[UNSUPPORTED]' | awk '{print $1}' | paste -sd, -)"
  echo "[WARN] 工具有不支持的算子（仅提醒，继续执行）：$bad_ops"
elif [[ $rc2 -eq 1 ]]; then
  status2="WARN"
  echo "[WARN] 有算子只能退回 CPU（仅提醒，继续执行）"
else
  status2="PASS"
fi

# 3 对战体检（仅在给了 --match-json 时；多个文件走多局汇总）
status3="SKIP"; note3="未提供 --match-json"
if [[ ${#MATCH_JSONS[@]} -gt 0 ]]; then
  echo
  echo "[3/3] 对战体检：tools/analyze_match.py（${#MATCH_JSONS[@]} 局）"
  EXISTING=()
  for path in "${MATCH_JSONS[@]}"; do
    if [[ -f "$path" ]]; then
      EXISTING+=("$path")
    else
      echo "[WARN] --match-json 文件不存在，已跳过：$path"
      note3="有文件不存在，已跳过"
    fi
  done
  if [[ ${#EXISTING[@]} -eq 0 ]]; then
    echo "[WARN] 无可用 match.json，第 3 步记 SKIP"
    note3="无可用 match.json"
  else
    match_out="$("$PY" "$ROOT/tools/analyze_match.py" --input "${EXISTING[@]}" --label "${LABEL}-match")"
    rc3=$?
    printf '%s\n' "$match_out"
    note3="$(printf '%s\n' "$match_out" | head -1)"
    report="$(printf '%s\n' "$match_out" | grep -m1 '报告：' | sed 's/报告：//')"
    [[ -n "$report" ]] && note3="$note3 → ${report#$ROOT/}"
    if [[ $rc3 -eq 1 ]]; then
      status3="FAIL"; fail=1
    elif [[ $rc3 -eq 2 ]]; then
      status3="受限"; blocked=1
    else
      status3="PASS"
    fi
  fi
fi

summary
exit "$CODE"
