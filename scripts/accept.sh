#!/usr/bin/env bash
# 一键验收流水线：ckpt → ONNX → 校验 → 测速 → 简报。
# 用法：scripts/accept.sh outputs/train/ckpt_0016.pt [--version N] [--matches N] [--skip-validate]
#
# 必须 source train_env.sh：MALLOC_* 要在 Python 进程启动前生效，否则完整链加载会 OOM。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/train_env.sh"
cd "$ROOT"

CKPT="${1:?用法: scripts/accept.sh <ckpt.pt> [--version N] [--matches N] [--skip-validate]}"
shift || true

exec "$HB_PY" tools/accept_ckpt.py --ckpt "$CKPT" "$@"
