import torch
from torch import nn


class PolicyLoss(nn.Module):
    """
    Proximal Policy Optimization Algorithms
    https://arxiv.org/pdf/1707.06347.pdf
    """
    def __init__(self, eps=0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps

    def forward(self, new_actor_log_probs: torch.Tensor,
                old_actor_log_probs: torch.Tensor, advantage: torch.Tensor, action_mask: torch.Tensor):
        # reverse the log to get π_new(a_t|s_t) / π_old(a_t|s_t)
        ratio = (new_actor_log_probs - old_actor_log_probs).exp()  # (B, num_actions)
        surrogate_objectives = torch.min(
            ratio * advantage,
            ratio.clamp(1 - self.eps, 1 + self.eps) * advantage)  # (B, num_actions)
        # minimize the negative loss -> maximize the objective
        loss = -surrogate_objectives  # (B, num_actions)
        return loss.mean()