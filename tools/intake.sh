#!/usr/bin/env bash
# 一键接包：tar.gz → 解包 → sha256 校验 → 交付件对齐 → 留档 models/ → launch_all。
#
# 用法：tools/intake.sh <包.tar.gz> [launch_all 额外参数...]
# 说明：包内若有 validation.json / benchmark.json，自动作为外部证据传给第 5/6 步；
#       有 match.json 自动作为 --match-json；models/ 里已有同内容 ONNX 则复用，不重复留档。
#
# 退出码：解包/sha 校验/缺 model.onnx → 1；其后与 launch_all 一致（0/1/2）。

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${HB_PY:-$HOME/hb-env/bin/python}"

PKG="${1:-}"
if [[ -z "$PKG" || ! -f "$PKG" ]]; then
  echo "[FAIL] 用法：tools/intake.sh <包.tar.gz> [launch_all 参数...]"
  exit 1
fi
shift

STEM="$(basename "$PKG" .tar.gz)"
SAFE="$(printf '%s' "$STEM" | tr -c 'A-Za-z0-9._-' '_')"
WORK="/tmp/hb_intake/$SAFE"
LOG="/tmp/hb_intake/$SAFE.sha.log"

echo "============================================="
echo "接包：$STEM"
echo "============================================="

rm -rf "$WORK"
mkdir -p "$WORK"
tar xzf "$PKG" -C "$WORK" || { echo "[FAIL] 解包失败：$PKG"; exit 1; }

SUMS="$(find "$WORK" -maxdepth 3 -name SHA256SUMS.txt -print -quit)"
if [[ -z "$SUMS" ]]; then
  echo "[FAIL] 包内无 SHA256SUMS.txt"
  exit 1
fi
DIR="$(dirname "$SUMS")"

echo
echo "[1/4] sha256 校验"
if ! (cd "$DIR" && sha256sum -c SHA256SUMS.txt > "$LOG" 2>&1); then
  echo "[FAIL] sha256 校验未通过，明细：$LOG"
  grep -v ': 成功$' "$LOG" | head -5
  exit 1
fi
echo "  $(grep -c ': 成功$' "$LOG") 个文件全部成功"

ONNX="$DIR/model.onnx"
if [[ ! -f "$ONNX" ]]; then
  echo "[FAIL] 包内无 model.onnx"
  exit 1
fi

echo
echo "[2/4] 交付件对齐（match.json 的 model_id vs ONNX sha256）"
"$PY" - "$ONNX" "$DIR/match.json" <<'PY'
import hashlib, json, sys
from pathlib import Path

onnx, match = Path(sys.argv[1]), Path(sys.argv[2])
sha = hashlib.sha256(onnx.read_bytes()).hexdigest()
print(f"  ONNX   {onnx.name}  {onnx.stat().st_size:,} 字节  sha256={sha[:16]}…")
if not match.is_file():
    print("  [SKIP] 包内无 match.json")
    raise SystemExit(0)
payload = json.loads(match.read_text(encoding="utf-8"))
model_id = (payload.get("summary") or {}).get("model_id") or ""
if isinstance(model_id, list):
    model_id = model_id[0] if model_id else ""
if model_id == sha:
    print(f"  match.json model_id = {model_id[:16]}…  对齐 PASS")
else:
    print(f"  [WARN] match.json model_id = {str(model_id)[:16]}… ≠ ONNX sha {sha[:16]}…"
          "（对局可能不是这份交付件跑的）")
PY

echo
echo "[3/4] 留档到 models/"
TARGET=""
for candidate in "$ROOT"/models/*.onnx; do
  [[ -f "$candidate" ]] || continue
  if cmp -s "$ONNX" "$candidate"; then
    TARGET="$candidate"
    break
  fi
done
if [[ -n "$TARGET" ]]; then
  echo "  已存在同内容文件：${TARGET#$ROOT/}，直接复用"
else
  TARGET="$ROOT/models/$SAFE.onnx"
  mkdir -p "$ROOT/models"
  cp "$ONNX" "$TARGET"
  echo "  已写入 models/$SAFE.onnx"
fi

echo
echo "[4/4] 交验：launch_all.sh"
ARGS=()
[[ -f "$DIR/match.json" ]] && ARGS+=(--match-json "$DIR/match.json")
[[ -f "$DIR/validation.json" ]] && ARGS+=(--evidence-validation "$DIR/validation.json")
[[ -f "$DIR/benchmark.json" ]] && ARGS+=(--evidence-benchmark "$DIR/benchmark.json")

exec "$ROOT/scripts/launch_all.sh" "$TARGET" "${ARGS[@]}" "$@"
