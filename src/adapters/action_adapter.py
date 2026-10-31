"""target_pose[1,225] 适配：重排为 [1,5,5,9]。

四个维度为批次 × 未来时刻 × 关键部位 × 位姿分量；
位姿分量 9 维 = 位置 3 维 + 旋转矩阵第一列 X 与第三列 Z 共 6 维。
来源：HoloMotion 官方 holoboxing 接口文档（README.zh-CN.md，master）。
"""

import numpy as np

TARGET_DIM = 225
N_FUTURE = 5
FUTURE_MS = (20, 40, 60, 80, 100)
N_KEYPARTS = 5
POSE_DIM = 9
N_POS = 3
N_ROT = 6

KEYPARTS = (
    "pelvis",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)

# 有限值修复口径：位置非有限值补 0，旋转非有限值补单位旋转的 X/Z 轴
IDENTITY_ROT = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def to_traj(y):
    """把 [1,225] 重排为 [1,5,5,9]，便于按关键部位/位姿分量访问。"""
    y = np.asarray(y, dtype=np.float32)
    assert y.shape == (1, TARGET_DIM), y.shape
    return y.reshape(1, N_FUTURE, N_KEYPARTS, POSE_DIM)


def rot_to_matrix(rot6):
    """6 维旋转（X、Z 两列）恢复为旋转矩阵：Y = Z × X，R = [X Y Z]。"""
    rot6 = np.asarray(rot6, dtype=np.float32)
    assert rot6.shape[-1] == N_ROT, rot6.shape
    x_axis = rot6[..., 0:3]
    z_axis = rot6[..., 3:6]
    y_axis = np.cross(z_axis, x_axis)
    return np.stack((x_axis, y_axis, z_axis), axis=-1)


class ActionAdapter:
    """动作适配：裁剪 → 一阶低通平滑 → 非法值修复，输出 [1,225] float32。"""

    def __init__(self, pose_min=None, pose_max=None, smooth_alpha=0.3):
        assert 0.2 <= smooth_alpha <= 0.4, f"smooth_alpha 超出 [0.2, 0.4]: {smooth_alpha}"
        self.pose_min = pose_min
        self.pose_max = pose_max
        self.alpha = smooth_alpha
        self._prev = None

    def reset(self):
        self._prev = None

    def sanitize(self, y):
        """非有限值修复：位置补 0，旋转补单位旋转的 X/Z 轴。"""
        y = np.asarray(y, dtype=np.float32).copy()
        traj = y.reshape(1, N_FUTURE, N_KEYPARTS, POSE_DIM)
        pos = traj[..., :N_POS]
        rot = traj[..., N_POS:]
        pos_mask = ~np.isfinite(pos)
        rot_mask = ~np.isfinite(rot)
        pos[pos_mask] = 0.0
        rot[rot_mask] = np.broadcast_to(IDENTITY_ROT, rot.shape)[rot_mask]
        return traj.reshape(1, TARGET_DIM)

    def to_target_pose(self, y):
        y = np.asarray(y, dtype=np.float32)
        assert y.shape == (1, TARGET_DIM), y.shape

        if self.pose_min is not None and self.pose_max is not None:
            y = np.clip(y, self.pose_min, self.pose_max)

        if self._prev is not None:
            y = self.alpha * y + (1.0 - self.alpha) * self._prev
        self._prev = y.copy()

        return self.sanitize(y).astype(np.float32)
