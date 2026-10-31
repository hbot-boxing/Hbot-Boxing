#!/usr/bin/env bash
# 用官方镜像运行双模型对战并产出 replay。
# 用法：scripts/run_match.sh models/red.onnx models/blue.onnx [输出目录]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RED="${1:?用法: scripts/run_match.sh <red.onnx> <blue.onnx> [out_dir]}"
BLUE="${2:?缺少 blue.onnx}"
OUT="${3:-outputs/eval/match}"

export HOLOBOXING_IMAGE="${HOLOBOXING_IMAGE:-horizonrobotics/holomotion:holoboxing-eval-v12-amd64}"

mkdir -p "$OUT"
./holoboxing/run.sh --match "$RED" "$BLUE" --output "$OUT" 2>&1 | tee "$OUT/match.log"
