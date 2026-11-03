"""torch 与 ONNX Runtime(CPU) 的数值一致性校验（阶段三 G3-04）。

用法：
    python -m src.export.check_parity --ckpt outputs/train/ckpt_0016.pt --onnx models/policy_v16.onnx

判据：逐元素最大绝对误差 ≤ 1e-4，且 ONNX 输出的 X/Z 仍为正交单位向量。
"""

import argparse

import numpy as np
import onnxruntime as ort
import torch

from src.export.export_onnx import load_policy
from src.models.policy import OBS_DIM, contract_ok

ATOL = 1e-4


def parity(policy, onnx_path, samples=64, atol=ATOL):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    generator = torch.Generator().manual_seed(0)
    worst = 0.0
    for _ in range(samples):
        obs = torch.randn(1, OBS_DIM, generator=generator) * 3.0
        with torch.no_grad():
            expected = policy(obs).numpy()
        actual = session.run(["target_pose"], {"obs": obs.numpy()})[0]
        worst = max(worst, float(np.abs(expected - actual).max()))
        if not contract_ok(torch.from_numpy(actual)):
            return worst, False
    return worst, worst <= atol


def main():
    parser = argparse.ArgumentParser(description="ONNX / torch 数值一致性")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()

    policy = load_policy(args.ckpt)
    worst, ok = parity(policy, args.onnx, args.samples)
    print(f"样本数        : {args.samples}")
    print(f"最大绝对误差  : {worst:.3e}（阈值 {ATOL:.0e}）")
    print(f"契约保持      : {ok}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
