"""阶段二训练入口：官方 MuJoCo 拳击环境 + 官方奖励项 + 本阶段策略网络。

用法（需已安装官方训练依赖并下载底层控制器）：
    python3 -m src.train.train --config src/train/configs/ppo.yaml --updates 32
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from src.adapters import action_adapter, obs_adapter  # noqa: F401  G2-03：同一份预处理实现
from src.models.policy import OBS_DIM, TARGET_DIM, Policy, parameter_count
from src.train.reward import BoxingReward

ROOT = Path(__file__).resolve().parents[2]
AGENT = "robot_A"
OPPONENT = "robot_B"

DEFAULTS = {
    "num_envs": 1,
    "rollout_steps": 256,
    "updates": 32,
    "epochs": 4,
    "minibatch": 64,
    "gamma": 0.99,
    "lam": 0.95,
    "clip": 0.2,
    "lr": 3.0e-4,
    "lr_value": 1.0e-3,
    "init_std": 0.1,
    "entropy": 1.0e-3,
    "seed": 42,
    "device": "cuda:0",
    "save_every": 4,
    "assets": "",
    "lower": "",
    "eval_every": 4,
    "eval_episodes": 20,
}


def load_config(path):
    text = Path(path).read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    config = dict(DEFAULTS)
    config.update(data or {})
    return config


def build_env(config):
    """官方环境构造；缺依赖或资源时给出明确缺失项。"""
    from holoboxing.data.contract import CompetitionAssets
    from holoboxing.training.envs.mujoco_env import MujocoSandbagEnv
    from holoboxing.training.resources import resolve_resources

    assets = CompetitionAssets.load(config["assets"])
    _, paths, _ = resolve_resources(config["assets"], config["lower"])
    env = MujocoSandbagEnv(
        assets,
        paths,
        int(config["num_envs"]),
        config["device"],
        config["output"],
        seed=int(config["seed"]),
    )
    return env, paths


class ValueNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class PPO:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.policy = Policy().to(self.device)
        self.value = ValueNet().to(self.device)
        self.log_std = nn.Parameter(
            torch.full((TARGET_DIM,), math.log(config["init_std"]), device=self.device)
        )
        self.policy_opt = torch.optim.Adam(
            list(self.policy.parameters()) + [self.log_std], lr=config["lr"]
        )
        self.value_opt = torch.optim.Adam(
            self.value.parameters(), lr=config["lr_value"]
        )
        self.reward = BoxingReward(agent=AGENT)

    def act(self, obs):
        mean = self.policy(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, dist.entropy().sum(-1)

    def update(self, rollout, last_value):
        obs = torch.stack(rollout["obs"])  # [T, B, 1097]
        actions = torch.stack(rollout["action"])
        logp_old = torch.stack(rollout["logp"])
        rewards = torch.stack(rollout["reward"])
        dones = torch.stack(rollout["done"]).float()
        values = torch.stack(rollout["value"])

        steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(rewards.shape[1], device=self.device)
        for t in reversed(range(steps)):
            next_value = last_value if t == steps - 1 else values[t + 1]
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.config["gamma"] * next_value * mask - values[t]
            gae = delta + self.config["gamma"] * self.config["lam"] * mask * gae
            advantages[t] = gae
        returns = advantages + values
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        flat_obs = obs.reshape(-1, obs.shape[-1])
        flat_actions = actions.reshape(-1, actions.shape[-1])
        flat_logp = logp_old.reshape(-1)
        flat_adv = adv.reshape(-1)
        flat_returns = returns.reshape(-1)
        total = flat_obs.shape[0]
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        for _ in range(self.config["epochs"]):
            order = torch.randperm(total, device=self.device)
            for start in range(0, total, self.config["minibatch"]):
                idx = order[start : start + self.config["minibatch"]]
                mean = self.policy(flat_obs[idx])
                std = self.log_std.exp().expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(flat_actions[idx]).sum(-1)
                ratio = (logp - flat_logp[idx]).exp()
                clipped = ratio.clamp(
                    1 - self.config["clip"], 1 + self.config["clip"]
                )
                policy_loss = -(
                    torch.min(ratio * flat_adv[idx], clipped * flat_adv[idx])
                ).mean()
                entropy = dist.entropy().sum(-1).mean()
                policy_loss = policy_loss - self.config["entropy"] * entropy
                self.policy_opt.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + [self.log_std], 1.0
                )
                self.policy_opt.step()

                value_pred = self.value(flat_obs[idx])
                value_loss = ((value_pred - flat_returns[idx]) ** 2).mean()
                self.value_opt.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
                self.value_opt.step()

                stats["policy_loss"] += float(policy_loss)
                stats["value_loss"] += float(value_loss)
                stats["entropy"] += float(entropy)
        batches = self.config["epochs"] * math.ceil(total / self.config["minibatch"])
        return {k: v / batches for k, v in stats.items()}

    def save(self, path, update, episodes):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "neutral": self.policy.neutral.detach().cpu(),
                "log_std": self.log_std.detach().cpu(),
                "config": self.config,
                "update": update,
                "episodes": episodes,
                "obs_dim": OBS_DIM,
                "target_dim": TARGET_DIM,
                "param_count": parameter_count(self.policy),
            },
            path,
        )


def train(config):
    torch.manual_seed(int(config["seed"]))
    env, paths = build_env(config)
    learner = PPO(config)
    from holoboxing.training.resources import load_neutral_reference

    reference = load_neutral_reference(paths["lower_onnx"], paths["arena_scene"])
    learner.policy.load_reference(reference)
    print(
        f"中性参考位姿已载入：位置范围 "
        f"[{float(reference[0::9].min()):.2f}, {float(reference[0::9].max()):.2f}] m",
        flush=True,
    )
    log_path = ROOT / "outputs" / "train" / "train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "update", "wall_s", "mean_reward", "mean_episode_length",
        "fallen_fraction", "policy_loss", "value_loss", "entropy", "checkpoint",
    ]
    rows = []
    observations = env.reset()
    obs = observations[AGENT]["task_policy"]
    episodes = 0
    started = time.time()
    for update in range(1, int(config["updates"]) + 1):
        rollout = {k: [] for k in ("obs", "action", "logp", "reward", "done", "value")}
        ep_lens, ep_returns, ep_steps = [], [], 0
        ep_return = torch.zeros(obs.shape[0], device=learner.device)
        fallen_count = 0
        for _ in range(int(config["rollout_steps"])):
            with torch.no_grad():
                action, logp, _entropy = learner.act(obs)
                value = learner.value(obs)
            actions = {
                AGENT: action,
                OPPONENT: torch.zeros_like(action),
            }
            next_observations, task_rewards, done, _ = env.step(actions)
            reward = learner.reward.combine(task_rewards[AGENT])
            fallen = env.metrics.get(AGENT, {}).get("fallen")
            fallen_count += int(fallen.sum()) if fallen is not None else 0
            for key, value_ in (
                ("obs", obs), ("action", action), ("logp", logp),
                ("reward", reward), ("done", done), ("value", value),
            ):
                rollout[key].append(value_)
            ep_steps += 1
            ep_return += reward
            finished = int(done.sum())
            if finished:
                ep_lens.extend([ep_steps] * finished)
                ep_returns.extend(float(ep_return.mean()) for _ in range(finished))
                ep_steps = 0
                ep_return = torch.zeros_like(ep_return)
                episodes += finished
            obs = next_observations[AGENT]["task_policy"]
        with torch.no_grad():
            last_value = learner.value(obs)
        stats = learner.update(rollout, last_value)
        checkpoint = ""
        if update % int(config["save_every"]) == 0 or update == int(config["updates"]):
            path = ROOT / "outputs" / "train" / f"ckpt_{update:04d}.pt"
            learner.save(path, update, episodes)
            checkpoint = path.name
        row = {
            "update": update,
            "wall_s": round(time.time() - started, 1),
            "mean_reward": round(float(torch.stack(rollout["reward"]).mean()), 5),
            "mean_episode_length": round(sum(ep_lens) / len(ep_lens), 2) if ep_lens else 0.0,
            "fallen_fraction": round(fallen_count / (int(config["rollout_steps"]) * obs.shape[0]), 6),
            **{k: round(v, 5) for k, v in stats.items()},
            "checkpoint": checkpoint,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    env.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description="阶段二 PPO 训练")
    parser.add_argument("--config", default="src/train/configs/ppo.yaml")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--assets", default=None)
    parser.add_argument("--lower", default=None)
    parser.add_argument("--output", default=None, help="环境日志目录")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    for key, value in (
        ("updates", args.updates),
        ("num_envs", args.num_envs),
        ("assets", args.assets),
        ("lower", args.lower),
        ("output", args.output),
    ):
        if value is not None:
            config[key] = value
    train(config)


if __name__ == "__main__":
    main()
