"""阶段二策略网络：obs[1,1097] → target_pose[1,225]。

结构按《阶段二-策略模型设计与训练》§2.2：
LayerNorm → 1097→512 → 残差×2 → 512→256 → 256→225。
输出是相对官方中性参考位姿（`load_neutral_reference`）的残差：
位置按米缩放叠加，轴向按参考轴旋转后再归一化/正交化，
保证官方 sanitize_target_pose 的「正交单位向量」契约恒成立。
"""

import torch
from torch import nn

OBS_DIM = 1097
TARGET_DIM = 225
SLOTS = 25  # 5 未来时刻 × 5 关键部位
SLOT_DIM = 9  # 位置 3 + X 轴 3 + Z 轴 3
PARAM_LIMIT = 12_500_000


def default_reference(target_dim=TARGET_DIM):
    """占位中性参考：位置归零、每槽 X=(1,0,0)、Z=(0,0,1)。

    训练与评测启动时用官方 `load_neutral_reference` 覆写为真实站姿。
    """
    points = torch.zeros(1, SLOTS, SLOT_DIM)
    points[..., 3] = 1.0
    points[..., 8] = 1.0
    return points.reshape(1, target_dim)


class ResidualBlock(nn.Module):
    """Pre-LN 残差块：x + GELU(Linear(LayerNorm(x)))。"""

    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc = nn.Linear(width, width)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.fc(self.norm(x)))


class Policy(nn.Module):
    """共享 MLP 骨干，输出 225 维关键点轨迹。"""

    def __init__(self, obs_dim=OBS_DIM, target_dim=TARGET_DIM, hidden=512, mid=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.target_dim = target_dim
        self.position_scale = 0.1  # 每单位隐层输出对应的米数
        self.axis_scale = 0.25  # 相对参考轴的旋转幅度
        self.backbone = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, hidden),
            nn.GELU(),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
            nn.Linear(hidden, mid),
            nn.GELU(),
            nn.Linear(mid, target_dim),
        )
        self.register_buffer("neutral", default_reference(target_dim))

    def load_reference(self, reference):
        """写入官方中性参考位姿（225 维，冻结为 buffer 随 checkpoint 保存）。"""
        ref = torch.as_tensor(reference, dtype=self.neutral.dtype)
        self.neutral.copy_(ref.reshape(1, self.target_dim))

    def forward(self, obs):
        raw = self.backbone(obs)
        points = raw.reshape(-1, SLOTS, SLOT_DIM)
        ref = self.neutral.reshape(-1, SLOTS, SLOT_DIM)
        pos = ref[..., :3] + self.position_scale * points[..., :3]
        x_axis = nn.functional.normalize(
            ref[..., 3:6] + self.axis_scale * points[..., 3:6], dim=-1, eps=1e-6
        )
        z_axis = ref[..., 6:9] + self.axis_scale * points[..., 6:9]
        z_axis = z_axis - (x_axis * z_axis).sum(-1, keepdim=True) * x_axis
        z_axis = nn.functional.normalize(z_axis, dim=-1, eps=1e-6)
        return torch.cat((pos, x_axis, z_axis), dim=-1).reshape(raw.shape)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def contract_ok(target_pose, atol=1e-3):
    """按官方 sanitize_target_pose 的口径检查 X/Z 正交单位向量。"""
    points = target_pose.reshape(-1, SLOTS, SLOT_DIM)
    x_axis, z_axis = points[..., 3:6], points[..., 6:9]
    norms_x = x_axis.norm(dim=-1)
    norms_z = z_axis.norm(dim=-1)
    dots = (x_axis * z_axis).sum(-1)
    return bool(
        (norms_x - 1).abs().max() <= atol
        and (norms_z - 1).abs().max() <= atol
        and dots.abs().max() <= atol
    )


if __name__ == "__main__":
    net = Policy().eval()
    with torch.no_grad():
        out = net(torch.zeros(1, OBS_DIM))
    print("输出形状", tuple(out.shape), "dtype", out.dtype)
    print("参数量", parameter_count(net), "上限", PARAM_LIMIT)
    print("X/Z 正交单位向量契约", contract_ok(out))
