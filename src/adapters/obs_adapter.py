"""obs[1,1097] 字段表与解析。

字段定义来源：HoloMotion 官方 holoboxing 接口文档（README.zh-CN.md，master）。
索引含起点、不含终点；多维展平时最后一维变化最快。
"""

import numpy as np

OBS_DIM = 1097
HISTORY_FRAMES = 10
FRAME_DIM = 108
FRAME_INTERVAL_MS = 20
N_KEYPARTS = 5

# 关键部位顺序：目标位姿按同一顺序排列
KEYPARTS = (
    "pelvis",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)

# 顶层切片
OBS_FIELDS = {
    "self_history": (0, 1080),
    "opponent_pose": (1080, 1095),
    "time_left": (1095, 1096),
    "score_diff": (1096, 1097),
}

# 单帧内切片（相对帧起点）
FRAME_FIELDS = {
    "pelvis_ang_vel": (0, 3),
    "keypart_pose": (3, 48),
    "keypart_lin_vel": (48, 63),
    "keypart_pose_prev": (63, 108),
}


def check_layout():
    """校验字段表：顶层与单帧均无缝覆盖，且帧数×帧长与历史段一致。"""
    covered = []
    for name, (lo, hi) in OBS_FIELDS.items():
        assert 0 <= lo < hi <= OBS_DIM, (name, lo, hi)
        covered.extend(range(lo, hi))
    assert len(covered) == OBS_DIM and len(set(covered)) == OBS_DIM, "OBS_FIELDS 未无缝覆盖 1097"

    frame_covered = []
    for name, (lo, hi) in FRAME_FIELDS.items():
        assert 0 <= lo < hi <= FRAME_DIM, (name, lo, hi)
        frame_covered.extend(range(lo, hi))
    assert len(frame_covered) == FRAME_DIM and len(set(frame_covered)) == FRAME_DIM, "FRAME_FIELDS 未无缝覆盖 108"

    self_lo, self_hi = OBS_FIELDS["self_history"]
    assert self_hi - self_lo == HISTORY_FRAMES * FRAME_DIM, "历史段长度与 10×108 不一致"


def split(obs):
    """把 obs[1,1097] 拆成语义化字段。当前帧即历史序列的最后一帧。"""
    obs = np.asarray(obs)
    assert obs.shape == (1, OBS_DIM), obs.shape
    assert obs.dtype == np.float32, obs.dtype
    x = obs[0]

    self_hist = x[0:1080].reshape(HISTORY_FRAMES, FRAME_DIM)
    return {
        "self_history": self_hist,
        "pelvis_ang_vel": self_hist[:, 0:3],
        "keypart_pose": self_hist[:, 3:48].reshape(HISTORY_FRAMES, N_KEYPARTS, 9),
        "keypart_lin_vel": self_hist[:, 48:63].reshape(HISTORY_FRAMES, N_KEYPARTS, 3),
        "keypart_pose_prev": self_hist[:, 63:108].reshape(HISTORY_FRAMES, N_KEYPARTS, 9),
        "opponent_pose": x[1080:1095].reshape(N_KEYPARTS, 3),
        "time_left": x[1095],
        "score_diff": x[1096],
    }


def current_frame(obs):
    """当前帧状态，等价于 split()['self_history'][-1]（官方文档 obs[:,972:1080]）。"""
    return split(obs)["self_history"][-1]


class ObsAdapter:
    """观测适配：按字段归一化后保持 [1,1097] 形状传给模型。"""

    def __init__(self, stats=None):
        self.stats = stats or {}  # {字段名: (mean, std)}，缺省不归一化
        check_layout()

    def to_model_input(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        assert obs.shape == (1, OBS_DIM), obs.shape
        x = obs.copy()
        for name, (lo, hi) in OBS_FIELDS.items():
            if name in self.stats:
                mean, std = self.stats[name]
                x[:, lo:hi] = (x[:, lo:hi] - mean) / (std + 1e-6)
        return x.astype(np.float32)
