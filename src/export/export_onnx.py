"""把训练 checkpoint 导出为提交用 ONNX（`obs[1,1097] → target_pose[1,225]`）。

用法：
    python -m src.export.export_onnx --ckpt outputs/train/ckpt_0016.pt
    python -m src.export.export_onnx --ckpt ... --out models/policy_v2_20261103.onnx

固定 batch=1、单输入单输出、无动态轴、权重内嵌，避免官方校验报动态形状或外部数据。
"""

import argparse
import hashlib
from pathlib import Path

import torch

from src.models.policy import OBS_DIM, TARGET_DIM, Policy, contract_ok, parameter_count

ROOT = Path(__file__).resolve().parents[2]


def load_policy(checkpoint, device="cpu"):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy = Policy().to(device).eval()
    policy.load_state_dict(payload["policy"])
    if "neutral" in payload:
        policy.load_reference(payload["neutral"])
    return policy


def file_sha256(path, block=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(policy, out_path, opset=17):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, OBS_DIM)
    torch.onnx.export(
        policy,
        (dummy,),
        str(out_path),
        input_names=["obs"],
        output_names=["target_pose"],
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,  # 走 TorchScript 导出，避免额外依赖 onnxscript
        training=torch.onnx.TrainingMode.EVAL,
        dynamic_axes=None,
    )
    with torch.no_grad():
        reference = policy(dummy)
    return out_path, reference


def main():
    parser = argparse.ArgumentParser(description="阶段三 ONNX 导出")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    policy = load_policy(args.ckpt)
    out = args.out or f"models/policy_{Path(args.ckpt).stem.replace('ckpt_', 'v')}.onnx"
    out_path, reference = export(policy, ROOT / out if not Path(out).is_absolute() else out,
                                 args.opset)
    size = out_path.stat().st_size
    print(f"checkpoint : {args.ckpt}")
    print(f"onnx       : {out_path}")
    print(f"opset      : {args.opset}")
    print(f"参数量     : {parameter_count(policy)}")
    print(f"文件大小   : {size} 字节（上限 50,000,000）")
    print(f"sha256     : {file_sha256(out_path)}")
    print(f"I/O        : obs[1,{OBS_DIM}] → target_pose{tuple(reference.shape)}")
    print(f"契约       : X/Z 正交单位向量 {contract_ok(reference)}")


if __name__ == "__main__":
    main()
