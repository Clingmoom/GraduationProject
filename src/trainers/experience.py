import torch

from dataclasses import dataclass


@dataclass
class Experience:
    completion: torch.Tensor # 模型生成的文本序列
    actor_log_probs: torch.Tensor # 策略网络（actor）在每个时间步选择的动作的对数概率 配合advantage计算策略梯度
    w_log_probs: torch.Tensor
    step_log_probs: torch.Tensor
    attention_mask: torch.Tensor # 有效token的掩码标识
    kl_penalized_reward: torch.Tensor # 经过KL散度惩罚后的奖励值
    advantage: torch.Tensor # 优势函数计算值（用于策略梯度更新）

    w_advantage: torch.Tensor
    step_advantage: torch.Tensor
    num_actions: int
    estimated_kl: torch.Tensor # 估计的KL散度，用于控制策略更新的幅度
    w_estimated_kl: torch.Tensor
    step_estimated_kl: torch.Tensor
    values: torch.Tensor # Critic模型预测值，结合 advantage 优化价值函数
    action_mask: torch.Tensor