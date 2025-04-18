import math
import torch
import loralib as lora

from torch import nn
from torch import Tensor
from torch.nn import functional as F
from src.configs import TrainingConfig


class MaskedMultiheadSelfAttention(nn.Module):

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        # Figure 2 in [1]
        self.cfg: TrainingConfig = cfg
        if self.cfg.lora_rank > 0:
            self.qkv_projection = lora.Linear(cfg.embedding_dim,
                                              3 * cfg.embedding_dim,
                                              bias=cfg.use_bias,
                                              r=cfg.lora_rank)
            self.output_projection = lora.Linear(cfg.embedding_dim,
                                                 cfg.embedding_dim,
                                                 bias=cfg.use_bias,
                                                 r=cfg.lora_rank)
            # self.qkv_projection = nn.Linear(cfg.embedding_dim,
            #                                 3 * cfg.embedding_dim,
            #                                 bias=cfg.use_bias)
            # self.output_projection = nn.Linear(cfg.embedding_dim,
            #                                    cfg.embedding_dim,
            #                                    bias=cfg.use_bias)
        else:
            self.qkv_projection = nn.Linear(cfg.embedding_dim,
                                            3 * cfg.embedding_dim,
                                            bias=cfg.use_bias)
            self.output_projection = nn.Linear(cfg.embedding_dim,
                                               cfg.embedding_dim,
                                               bias=cfg.use_bias)
        # 定义dropout层 防止过拟合
        self.attention_dropout = nn.Dropout(cfg.dropout_rate)
        self.output_dropout = nn.Dropout(cfg.dropout_rate)

        # construct a mask like this 构造一个下三角掩码矩阵，确保在自注意力计算中，每个单词只能关注到它之前的单词。
        # [[1, 0, 0]
        #  [1, 1, 0]]
        #  [1, 1, 1]] when block_size is 3
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        # insert (B, T) dimension for broadcasting later
        mask = mask.view(1, 1, cfg.block_size, cfg.block_size)
        # 使用register_buffer方法将掩码矩阵注册为模型的缓冲区，不会被视为模型参数
        # mask is a constant and shouldn't be considered as parameters
        # (1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor, attention_mask: Tensor = None):
        """
        x: shape of (B, T, C)
        """
        B, T, C = x.size() # T7,C1024,B1
        # Project x three times and split into Q,K,V
        x3 = self.qkv_projection(x)  # (B, T, 3C)   TODO: cross-att?
        Q, K, V = x3.split(self.cfg.embedding_dim,
                           dim=2)  # (B, T, C) for each

        # Prepare Q,K,V into desired shape for multi-head attention
        # Multi-head attention is equivalent to single-head attention on sequence-tensor form
        # see 3.1 in [3]
        Q = Q.view(B, T, self.cfg.n_heads,
                   C // self.cfg.n_heads)  # (B, T, h, h_dim)
        Q = Q.transpose(1, 2)  # (B, h, T, h_dim)
        K = K.view(B, T, self.cfg.n_heads,
                   C // self.cfg.n_heads)  # (B, T, h, h_dim)
        K = K.transpose(1, 2)  # (B, h, T, h_dim)
        V = V.view(B, T, self.cfg.n_heads,
                   C // self.cfg.n_heads)  # (B, T, h, h_dim)
        V = V.transpose(1, 2)  # (B, h, T, h_dim)

        # (B, h, T, h_dim) @ (B, h, h_dim, T) -> (B, h, T, T)
        attention = Q @ K.transpose(2, 3)
        attention *= 1.0 / math.sqrt(K.size(-1))
        # In transformer decoder, one word can only attend to words before itself
        attention = attention.masked_fill(self.mask[:, :, :T, :T] == 0,
                                          float('-inf'))  # (B, h, T, T)
        if attention_mask is not None:
            # https://github.com/huggingface/transformers/blob/c7f3abc257af9dfb6006a76f2b09b48355322d4d/src/transformers/models/gpt2/modeling_gpt2.py#L805
            # also, we don't need attend to padding tokens
            attention_mask = attention_mask[:, None,
                             None, :]  # (B, T) -> (B, 1, 1, T)
            attention_mask = (1.0 - attention_mask) * torch.finfo(
                attention.dtype).min
            # This will broadcast to each row of the last dimension of attention map
            # [[[[1, -inf, -inf],
            #    [1, 1,    -inf],
            #    [1, 1,    1   ]]]]]  + [[[[0, 0, -float.min]]]]]
            # =
            # [[[[1, -inf, -inf       ],
            #    [1, 1,    -inf       ],
            #    [1, 1,    1-float.min]]]]]
            attention = attention + attention_mask

        attention = F.softmax(attention, dim=-1)  # (B, h, T, T)
        attention = self.attention_dropout(attention)
        # (B, h, T, T) @ (B, h, T, h_dim) -> (B, h, T, h_dim)
        # V weighted by attention
        weighted_value = attention @ V
        # restore the original shape (B, T, C)
        weighted_value = weighted_value.transpose(1, 2).contiguous().view(
            B, T, C)

        # Finally, linearly project the weighted value to get the output
        y = self.output_projection(weighted_value)
        y = self.output_dropout(y)
        return y