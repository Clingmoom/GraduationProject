import torch
from torch import nn


class ValueLoss(nn.Module):

    def __init__(self, eps=0.4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps

    def forward(self, values: torch.Tensor, reward: torch.Tensor, old_values: torch.Tensor, action_mask: torch.Tensor):
        # https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py#L69-L75
        # https://github.com/openai/baselines/issues/91
        values_clipped = old_values + (values - old_values).clamp(-self.eps, self.eps)
        surrogate_values = torch.max(torch.square(values - reward), torch.square(values_clipped - reward))
        return surrogate_values.mean()  # (B, 1)