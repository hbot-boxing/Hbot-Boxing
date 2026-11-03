"""阶段二奖励函数：只做权重映射，判定与阈值全部来自官方实现。

- 任务奖励项：直接调用官方 `holoboxing.training.envs.mujoco_rewards.terms`
  （落地阈值、KO 确认时长、回合时长均来自官方 `scoring_config()`）；
- 比赛奖励：读取官方判定器（`MatchStateMachine` / `CudaReferee`）输出的比分
  增量与倒地标记，不重建任何「命中冲量 / 同手间隔 / 倒地高度」判定。
"""

from src.models.policy import OBS_DIM  # noqa: F401  （与策略共享维度常量）

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
    """官方奖励项的加权组合。"""

    def __init__(self, weights=None, agent="robot_A"):
        self.agent = agent
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            unknown = set(weights) - set(self.weights)
            if unknown:
                raise ValueError(f"未知奖励键：{sorted(unknown)}")
            self.weights.update(weights)
        self.terms = official_task_terms(agent)
        self.last_terms = {}

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

    def combine(self, task_reward, score_delta=None, fallen=None, opponent_ko=None):
        """把官方任务奖励项与官方判定器输出合成为总奖励。

        训练环境在 step 内部已按官方公式算好 task_reward；比分/倒地/KO
        只在官方判定器可用（对局评测）时传入，训练阶段其余项为零。
        """
        total = self.weights["task"] * task_reward
        if score_delta is not None:
            total = total + self.weights["hit"] * score_delta
        if fallen is not None:
            total = total + self.fall_penalty(fallen)
        if opponent_ko is not None:
            total = total + self.ko_bonus(opponent_ko)
        return total
