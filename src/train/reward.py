"""阶段二奖励函数：官方任务项加权 + 本侧训练期 shaping。

- 任务奖励项：直接调用官方 `holoboxing.training.envs.mujoco_rewards.terms`
  （落地阈值、KO 确认时长、回合时长均来自官方 `scoring_config()`）；
- 比赛奖励：读取官方判定器（`MatchStateMachine` / `CudaReferee`）输出的比分
  增量与倒地标记，不重建任何「命中冲量 / 同手间隔 / 倒地高度」判定。
- 本侧 shaping（距离带、贴脸计时、命中-回撤双条件）只在训练侧生效，
  不参与官方计分；冲量信号仍取自官方 `offensive_hit_reward_punch`。

B 侧新增项：

1. 距离带：两躯干水平间距落在 `band_near~band_far` 内，按秒给正奖励；
2. 贴脸计时：间距 < `band_near` 时按已停留秒数线性加重惩罚，离开即清零；
3. 双条件：官方冲量命中先在「已回撤」的手上计满分，未回撤的手按折扣计，
   腕心到自身躯干的距离要缩短 `retract_m` 才算回撤、重新上膛。
"""

import torch

from src.models.policy import OBS_DIM  # noqa: F401  （与策略共享维度常量）

SHAPING = {
    "band_near": 0.9,  # 距离带内边界（米）
    "band_far": 1.4,  # 距离带外边界（米）
    "band_bonus": 0.5,  # 带内每秒奖励
    "clinch_rate": 1.0,  # <band_near 时，每秒 × 已停留秒数 的惩罚
    "clinch_cap_s": 2.5,  # 计时封顶，避免长贴脸把惩罚放大到不可恢复
    "retract_gate": 0.2,  # 未回撤就再次命中时，命中奖励的折扣系数
    "retract_m": 0.10,  # 腕心到躯干缩短多少米算完成回撤
}

DEFAULT_WEIGHTS = {
    "task": 1.0,  # 官方任务奖励项总权重
    "hit": 1.0,  # 己方得分 +1 的奖励
    "hit_taken": 0.5,  # 对方得分 +1 的惩罚系数
    "fallen": -1.0,  # 官方倒地判定为真时的惩罚
    "ko": 1.0,  # 官方 KO（对手连续倒地满确认时长）的奖励系数
}


def official_constants():
    """读取官方判定器常量（阈值、版本号），供训练与记录引用。"""
    from holoboxing.evaluation.runtime.fall_rules import FALL_BODY_NAMES, FALL_HEIGHT_M
    from holoboxing.evaluation.runtime.hand_hit_rules import (
        SCORING_VERSION,
        scoring_config,
    )

    return {
        "scoring_version": SCORING_VERSION,
        "scoring_config": scoring_config(),
        "fall_height_m": FALL_HEIGHT_M,
        "fall_body_names": list(FALL_BODY_NAMES),
    }


def official_task_terms(agent):
    """官方任务奖励项清单：(名称, 权重, 函数, 参数)。"""
    from holoboxing.training.envs.mujoco_rewards import terms

    return terms(agent)


class BoxingReward:
    """官方奖励项的加权组合 + 本侧训练 shaping。"""

    def __init__(self, weights=None, agent="robot_A", shaping=None):
        self.agent = agent
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            unknown = set(weights) - set(self.weights)
            if unknown:
                raise ValueError(f"未知奖励键：{sorted(unknown)}")
            self.weights.update(weights)
        self.terms = official_task_terms(agent)
        self.last_terms = {}
        self.shaping = dict(SHAPING)
        if shaping:
            unknown = set(shaping) - set(self.shaping)
            if unknown:
                raise ValueError(f"未知 shaping 键：{sorted(unknown)}")
            self.shaping.update(shaping)
        self._rows = 0
        self._clinch_s = None
        self._armed = None
        self._ext = None

    def task_reward(self, env):
        """按官方公式累加任务奖励项：weight × term × env.step_dt。"""
        total = None
        self.last_terms = {}
        for name, weight, fn, params in self.terms:
            value = weight * fn(env, **params) * env.step_dt
            self.last_terms[name] = value
            total = value.clone() if total is None else total + value
        return total

    def score_reward(self, previous, current, fighter=0):
        """官方判定器比分增量 → 比赛奖励（previous/current 为 (envs,2) 比分）。"""
        delta = current[:, fighter] - previous[:, fighter]
        conceded = current[:, 1 - fighter] - previous[:, 1 - fighter]
        return self.weights["hit"] * delta - self.weights["hit_taken"] * conceded

    def fall_penalty(self, fallen):
        """官方倒地标记（fallen_from_scene，高度阈值取自 fall_rules）→ 惩罚。"""
        return self.weights["fallen"] * fallen.float()

    def ko_bonus(self, opponent_ko):
        """官方 KO 判定（连续倒地满确认时长）→ 奖励，按官方 KO 分值缩放。"""
        ko_points = float(official_constants()["scoring_config"]["ko_bonus"])
        return self.weights["ko"] * ko_points / 1000.0 * opponent_ko.float()

    def _ensure_state(self, env):
        """按环境行数惰性建立 shaping 状态：贴脸计时、双手是否上膛、命中时伸展量。"""
        rows = int(env.num_envs)
        if self._rows != rows:
            self._clinch_s = torch.zeros(rows, device=env.device)
            self._armed = torch.ones(rows, 2, dtype=torch.bool, device=env.device)
            self._ext = torch.zeros(rows, 2, device=env.device)
            self._rows = rows

    def reset(self, done):
        """回合重置的行清零贴脸计时与回撤状态。"""
        self._clinch_s[done] = 0.0
        self._armed[done] = True
        self._ext[done] = 0.0

    def _distance(self, env):
        """双方躯干原点的水平间距（米）。"""
        ego = env.scene[self.agent]
        opponent = env.get_opponent(self.agent)
        delta = (
            ego.data.body_pos_w[:, ego.body_names.index("torso_link"), :2]
            - opponent.data.body_pos_w[:, opponent.body_names.index("torso_link"), :2]
        )
        return delta.norm(dim=-1)

    def distance_shaping(self, env):
        """距离带正奖励 + 贴脸（< band_near）按停留时长递增的惩罚。"""
        distance = self._distance(env)
        dt = float(env.step_dt)
        inside = (distance >= self.shaping["band_near"]) & (
            distance <= self.shaping["band_far"]
        )
        bonus = inside.float() * self.shaping["band_bonus"] * dt
        clinch = distance < self.shaping["band_near"]
        self._clinch_s = torch.where(
            clinch, self._clinch_s + dt, torch.zeros_like(self._clinch_s)
        )
        dwell = self._clinch_s.clamp(max=self.shaping["clinch_cap_s"])
        penalty = -self.shaping["clinch_rate"] * dwell * dt
        return bonus + torch.where(clinch, penalty, torch.zeros_like(penalty))

    def _hit_term(self):
        for name, weight, fn, params in self.terms:
            if name == "offensive_hit_reward_punch":
                return weight, fn, params
        raise ValueError("官方奖励项缺少 offensive_hit_reward_punch")

    def retract_gate(self, env):
        """命中-回撤双条件：未回撤的手再次命中只按 retract_gate 计入。

        官方聚合项已含全部命中奖励，本函数只返回需要扣减的差额（≤ 0）。
        手命中后进入「待回撤」，腕心到自身躯干的距离缩短 `retract_m` 才重新上膛。
        """
        from holoboxing.training.envs.mujoco_rewards import entity

        weight, fn, params = self._hit_term()
        ego = env.scene[self.agent]
        torso = ego.data.body_pos_w[:, ego.body_names.index("torso_link")]
        dt = float(env.step_dt)
        total = torch.zeros(self._rows, device=ego.data.body_pos_w.device)
        for hand, part in enumerate(("left_wrist_yaw_link", "right_wrist_yaw_link")):
            hand_params = dict(params)
            hand_params["sensor_cfg1"] = entity(params["sensor_cfg1"].name, [part])
            raw = weight * fn(env, **hand_params) * dt
            reach = (ego.data.body_pos_w[:, ego.body_names.index(part)] - torso).norm(
                dim=-1
            )
            retracted = reach <= self._ext[:, hand] - self.shaping["retract_m"]
            self._armed[:, hand] |= retracted
            gain = torch.where(
                self._armed[:, hand], 1.0, self.shaping["retract_gate"]
            )
            hit = raw > 0
            self._ext[:, hand] = torch.where(hit, reach, self._ext[:, hand])
            self._armed[:, hand] &= ~hit
            total = total + raw * (gain - 1.0)
        return total

    def combine(
        self, task_reward, env=None, done=None,
        score_delta=None, fallen=None, opponent_ko=None,
    ):
        """把官方任务奖励项与官方判定器输出合成为总奖励。

        传入 env 时追加本侧训练 shaping（距离带 / 贴脸计时 / 命中-回撤双条件），
        这些项不参与官方计分；比分/倒地/KO 只在官方判定器可用时传入。
        """
        total = self.weights["task"] * task_reward
        if env is not None:
            self._ensure_state(env)
            with torch.no_grad():
                total = total + self.distance_shaping(env) + self.retract_gate(env)
            if done is not None:
                self.reset(done)
        if score_delta is not None:
            total = total + self.weights["hit"] * score_delta
        if fallen is not None:
            total = total + self.fall_penalty(fallen)
        if opponent_ko is not None:
            total = total + self.ko_bonus(opponent_ko)
        return total
