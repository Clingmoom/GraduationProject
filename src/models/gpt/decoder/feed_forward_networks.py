import math
import torch
import loralib as lora

from torch import nn
from src.configs import TrainingConfig


class FeedForwardNetworks(nn.Module):

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        if cfg.lora_rank > 0:
            self.fc1 = lora.Linear(cfg.embedding_dim,
                                   4 * cfg.embedding_dim,
                                   bias=cfg.use_bias,
                                   r=cfg.lora_rank)
            self.fc2 = lora.Linear(4 * cfg.embedding_dim,
                                   cfg.embedding_dim,
                                   bias=cfg.use_bias,
                                   r=cfg.lora_rank)
            # self.fc1 = nn.Linear(cfg.embedding_dim,
            #                      4 * cfg.embedding_dim,
            #                      bias=cfg.use_bias)
            # self.fc2 = nn.Linear(4 * cfg.embedding_dim,
            #                      cfg.embedding_dim,
            #                      bias=cfg.use_bias)
        else:
            self.fc1 = nn.Linear(cfg.embedding_dim,
                                 4 * cfg.embedding_dim,
                                 bias=cfg.use_bias)
            self.fc2 = nn.Linear(4 * cfg.embedding_dim,
                                 cfg.embedding_dim,
                                 bias=cfg.use_bias)
        self.dropout = nn.Dropout(cfg.dropout_rate)

    def gelu(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        y = self.dropout(x)
        return y