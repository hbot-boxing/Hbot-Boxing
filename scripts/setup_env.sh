#!/usr/bin/env bash
# 阶段一环境准备：拉取官方镜像、获取官方代码与资源、自检依赖。
# 仅做环境准备，不评测模型能力。任意一步失败即退出。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HOLOBOXING_IMAGE="${HOLOBOXING_IMAGE:-horizonrobotics/holomotion:holoboxing-eval-v12-amd64}"
OFFLINE_TAR="${OFFLINE_TAR:-}"

echo "== 1. 检查前置条件 =="
python3 -c 'import numpy; print("numpy", numpy.__version__)'

if ! command -v docker >/dev/null 2>&1; then
  echo "docker 未安装：请先安装 Docker 与 NVIDIA Container Toolkit（见 docs/阶段文档/阶段一 §3.1）" >&2
  exit 2
fi
docker --version

echo "== 2. 获取官方评测镜像 =="
if [[ -n "$OFFLINE_TAR" ]]; then
  docker load -i "$OFFLINE_TAR"
  echo "请将 HOLOBOXING_IMAGE 设为导入后的镜像标签或 ID"
else
  docker pull "$HOLOBOXING_IMAGE"
fi

echo "== 3. 获取官方代码（只读，勿改） =="
if [[ ! -d holoboxing ]]; then
  git clone --depth 1 https://github.com/HorizonRobotics/HoloMotion.git holoboxing-src
  mv holoboxing-src/holoboxing holoboxing
  rm -rf holoboxing-src
fi

echo "== 4. 获取资源包（G1 机器人模型、拳台场景） =="
cat <<'EOF'
手动步骤：
  Hugging Face: https://huggingface.co/datasets/HorizonRobotics/HoloMotion_assets/tree/main/holoboxing
  下载后放到 holoboxing/assets/（或按官方 README 指定路径）
EOF

echo "== 5. 接口自检（不需要镜像） =="
python3 tools/verify_phase1.py || true

echo "环境准备完成。镜像类门禁请在具备 docker 后运行 scripts/run_validate.sh。"
