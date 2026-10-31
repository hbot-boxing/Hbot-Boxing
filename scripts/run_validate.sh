#!/usr/bin/env bash
# 用官方镜像校验单个 ONNX 模型（接口 + 运行要求）。
# 用法：scripts/run_validate.sh models/red.onnx [--benchmark]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${1:?用法: scripts/run_validate.sh <model.onnx> [--benchmark]}"
shift || true

export HOLOBOXING_IMAGE="${HOLOBOXING_IMAGE:-horizonrobotics/holomotion:holoboxing-eval-v12-amd64}"

mkdir -p outputs/eval
./holoboxing/run.sh --validate "$MODEL" "$@" 2>&1 | tee "outputs/eval/validate_phase1.log"
