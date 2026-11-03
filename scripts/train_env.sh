#!/usr/bin/env bash
# 阶段二训练/评测的环境变量（source 使用，不要直接执行）。
#
#   source scripts/train_env.sh
#   $HB_PY -m src.train.train --config src/train/configs/ppo.yaml
#
# 说明：官方 mujoco_warp + 1.65GB 底层控制器一起加载，峰值常驻约 9.7GB；
# 在 16GB 内存的机器上会被 OOM 杀掉。下面三个 MALLOC_* 参数把峰值压到约 4.8GB。
# MALLOC_* 必须在进程启动前生效，因此必须 source 而不是在 Python 里设置。

HB_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HB_REPO
# 官方 HoloMotion 仓库根（含 holoboxing 与 holomotion 命名空间包）
export HB_HOLOMOTION="${HB_HOLOMOTION:-/mnt/hbotdata/tmp/HoloMotion}"
# Python 解释器：需装齐 torch / warp-lang / mujoco-warp / onnxruntime-gpu
export HB_PY="${HB_PY:-$HOME/hb-env/bin/python}"

export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_MMAP_THRESHOLD_=131072
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="$HB_HOLOMOTION:$HB_REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "HB_REPO=$HB_REPO"
echo "HB_HOLOMOTION=$HB_HOLOMOTION"
echo "HB_PY=$HB_PY"
echo "MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX"
