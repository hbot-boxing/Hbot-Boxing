"""G2-07：torch 原生对局评测（不导出 ONNX）。

复用官方对局循环 `run_match` 的 `reference_factory` 注入口，把上层 ONNX 会话
换成 torch 策略；观测编码、五点位姿重定向、target_pose 清理、官方判定器
（`MatchStateMachine` / `CudaReferee`）与回放记录全部沿用官方实现。

用法：
    python3 -m src.eval.torch_rollout --checkpoint outputs/train/ckpt_0016.pt \
        --assets <holoboxing assets> --lower <holomotion-v2> --episodes 20
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.adapters import action_adapter, obs_adapter  # noqa: F401  G2-03：与推理同一份输入/输出适配
from src.models.policy import OBS_DIM, TARGET_DIM, Policy

ROOT = Path(__file__).resolve().parents[2]


class TorchSession:
    """把 torch 策略包装成 `run_competitor` 需要的会话接口。

    该策略在 CUDA 上执行，因此 `boxing_cuda_verified` 置真以跳过 ONNX
    专属的 profiler 校验（此处不存在 ONNX 会话）。
    """

    boxing_cuda_verified = True

    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    def run(self, _names, feeds):
        obs = np.asarray(feeds["obs"], dtype=np.float32)
        if obs.shape != (1, OBS_DIM):
            raise ValueError(f"观测形状异常：{obs.shape}")
        obs_adapter.split(obs)  # 与推理链路共用同一份切分实现
        tensor = torch.as_tensor(obs, device=self.device)
        with torch.no_grad():
            out = self.policy(tensor)
        value = out.detach().cpu().numpy().astype(np.float32)
        action_adapter.to_traj(value)  # 与推理链路共用同一份 [1,225] → [1,5,5,9] 校验
        if value.shape != (1, TARGET_DIM):
            raise ValueError(f"策略输出形状异常：{value.shape}")
        return [value]

    def get_providers(self):
        return ["CUDAExecutionProvider"]


def load_policy(checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy = Policy().to(device).eval()
    policy.load_state_dict(payload["policy"])
    if "neutral" in payload:
        policy.load_reference(payload["neutral"])
    return policy


def build_reference_pair_class():
    from holoboxing.evaluation.runtime.compiled_artifacts import reference_schema
    from holoboxing.evaluation.runtime.model_contract import INPUT_LINK_NAMES
    from holoboxing.evaluation.runtime.upper import FivePointReferencePair

    class TorchReferencePair(FivePointReferencePair):
        """官方 provider 的 torch 版本：跳过 ONNX 建模，其余逻辑沿用官方。"""

        def __init__(self, policies, scene, lower_policies, *, device="cuda:0"):
            self.scene = scene
            self.device = torch.device(device)
            self.schemas = [reference_schema(), reference_schema()]
            self.sessions = [TorchSession(policy, self.device) for policy in policies]
            self.match = None
            self.body_ids = np.asarray(
                [
                    [scene.model.body(prefix + name).id for name in INPUT_LINK_NAMES]
                    for prefix in ("p1_", "p2_")
                ]
            )
            for policy in lower_policies:
                policy.attach_scene(scene)
            self.body_gpu = torch.as_tensor(self.body_ids, device=self.device)
            # fast_runtime 的下层 `LinkPolicy.observation` 依赖这三项（官方
            # `pair_init` 的职责），此处按同一约定补齐。
            for policy, schema in zip(lower_policies, self.schemas):
                policy.schema = schema
                policy.upper_body = self.body_gpu
                policy.upper_body_host = self.body_ids
            scene._upper_pair = self
            self.metadata = {
                "upper_policy_source": "torch_checkpoint",
                "upper_policy_contract": "phase2-torch-native",
                "upper_policy_provider": "torch-cuda",
                "upper_policy_device": str(self.device),
                "upper_observation_device": "cpu",
                "upper_policy_timing_scope": "torch_forward",
                "reference_source": "online_red_blue_torch_policy",
                "five_point_link_names": list(INPUT_LINK_NAMES),
                "five_point_schema_source_hashes": self.schemas[0].source_hashes,
            }
            self.reset()

        @torch.no_grad()
        def window(self, kinematics, *, fighters=(0, 1)):
            """与官方 GPU 观测路径同源：由 `_observation_fk` 取当前位姿/速度。

            官方 fast_runtime 会把 `window` 换成读 GPU 常驻缓冲的版本；此处
            保持同一数据来源，只把会话换成 torch 策略。
            """
            import warp as wp

            from holoboxing.evaluation.runtime.policy_output import (
                run_competitor,
                sanitize_target_pose,
            )
            from holoboxing.evaluation.runtime.upper import check_target_pose

            fk = self.scene._observation_fk
            positions = wp.to_torch(fk.positions)[self.body_gpu].float()
            quaternions = wp.to_torch(fk.rotations)[self.body_gpu].float()
            quaternions = quaternions / torch.linalg.vector_norm(
                quaternions, dim=-1, keepdim=True
            )
            cvel = wp.to_torch(fk.cvel)[self.body_gpu].float()
            roots = torch.as_tensor(
                self.scene.model.body_rootid[self.body_ids], device=self.device
            )
            com = wp.to_torch(fk.subtree_com)[roots].float()
            velocity = cvel[..., 3:] + torch.cross(
                cvel[..., :3], positions - com, dim=-1
            )
            qvel = wp.to_torch(fk.qvel)
            omega = torch.stack(
                [qvel[v + 3 : v + 6].float() for v in self.scene._root_dof]
            )
            observations = self.observations(
                positions, quaternions, velocity, omega
            )
            if not bool(torch.isfinite(observations).all()):
                raise ValueError("Nonfinite measured upper-policy observation")
            results, starts, ends, sanitized, sanitation_ns = [], [], [], [], []
            for fighter in fighters:
                start = time.perf_counter_ns()
                output = run_competitor(
                    self.sessions[fighter],
                    observations[fighter : fighter + 1].cpu().numpy(),
                )
                end = time.perf_counter_ns()
                sanitation_start = time.perf_counter_ns()
                output, statistics = sanitize_target_pose(output)
                sanitation_ns.append(time.perf_counter_ns() - sanitation_start)
                sanitized.append(statistics)
                self.sanitization_total[fighter] += statistics[
                    "nonfinite_components"
                ]
                check_target_pose(output)
                results.append(
                    self.reference(
                        torch.from_numpy(output[0]).to(self.device),
                        fighter,
                        float(positions[fighter, 0, 2]),
                    )
                )
                starts.append(start)
                ends.append(end)
            self.history_steps += 1
            self.frame_data = dict(
                upper_policy_start_ns=starts,
                upper_policy_end_ns=ends,
                upper_history_step=self.history_steps,
                competitor_sanitization=sanitized,
                competitor_sanitization_total=self.sanitization_total.copy(),
                competitor_sanitization_ns=sanitation_ns,
            )
            return results

    return TorchReferencePair


def run_episode(runtime, resources, checkpoint, policies, reference_class, output_dir, index):
    from holoboxing.evaluation.runtime import lower as lower_runtime

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "match.json"
    if target.exists():
        target.unlink()
    config = runtime.MatchConfig(
        retargeted=resources.initial_state,
        policy_0=(lower_runtime.MODEL_ROOT / "exported/model_10500.onnx"),
        policy_1=(lower_runtime.MODEL_ROOT / "exported/model_10500.onnx"),
        mujoco_model=resources.scene,
        mesh_root=resources.root / "meshes",
        output=target,
        match_id=f"phase2-torch-{index:03d}",
    )
    runtime.run_match(
        config,
        reference_factory=lambda scene, lowers: reference_class(
            policies, scene, lowers, device=config.device
        ),
    )
    return json.loads(target.read_text())


def _nonfinite_count(value):
    """统计回放中非有限数值个数（NaN/inf 即仿真或策略异常）。"""
    if isinstance(value, dict):
        return sum(_nonfinite_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nonfinite_count(item) for item in value)
    if isinstance(value, float):
        return 0 if math.isfinite(value) else 1
    return 0


def summarize(payload, index):
    """只取官方对局输出：`summary.final_score` / 命中 / 倒地 / KO / 结束原因。"""
    summary = payload.get("summary") or {}
    metadata = payload.get("metadata") or {}
    frames = payload.get("frames") or []
    score = summary.get("final_score") or [0, 0]
    hits = summary.get("hit_count") or [0, 0]
    falls = summary.get("fall_count") or [0, 0]
    knockdowns = summary.get("knockdown_count") or [0, 0]
    kos = summary.get("ko_count") or [0, 0]
    sanitization = metadata.get("competitor_sanitization_total") or [0, 0]
    nonfinite = _nonfinite_count(frames)
    return {
        "episode": index,
        "score_red": int(score[0]),
        "score_blue": int(score[1]),
        "hit_count_red": int(hits[0]),
        "hit_count_blue": int(hits[1]),
        "fall_count_red": int(falls[0]),
        "fall_count_blue": int(falls[1]),
        "knockdown_count_red": int(knockdowns[0]),
        "knockdown_count_blue": int(knockdowns[1]),
        "ko_count_red": int(kos[0]),
        "ko_count_blue": int(kos[1]),
        "end_reason": summary.get("end_reason"),
        "winner": summary.get("winner"),
        "frames": len(frames),
        "match_over": bool(metadata.get("match_completed", False)),
        "sanitized_components": int(sum(sanitization)),
        "nonfinite_values": nonfinite,
        "abnormal_fall": bool(nonfinite or sum(sanitization)),
    }


def main():
    parser = argparse.ArgumentParser(description="torch 原生对局评测（G2-07）")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--lower", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="outputs/eval")
    parser.add_argument("--workdir", default="/mnt/hbotdata/tmp/eval/torch_rollout")
    args = parser.parse_args()

    from holoboxing.evaluation.resources import ResourceTree
    from holoboxing.evaluation.runtime.holomotion_v2_runtime import install_runtime

    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device)
    reference_class = build_reference_pair_class()
    runtime = install_runtime(args.lower)
    resources = ResourceTree(args.assets)
    workdir = Path(args.workdir)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"torch_rollout_{Path(args.checkpoint).stem}.csv"
    rows = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    for index in range(int(args.start_index), int(args.start_index) + int(args.episodes)):
        payload = run_episode(
            runtime, resources, args.checkpoint, [policy, policy],
            reference_class, workdir / f"episode_{index:03d}", index,
        )
        row = summarize(payload, index)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)

    print(f"{len(rows)} 局结果已写入 {csv_path}")


if __name__ == "__main__":
    main()
