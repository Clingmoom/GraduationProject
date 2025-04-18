from torch import nn
from torch import Tensor
from src.configs import TrainingConfig
from masked_multihead_self_attention import MaskedMultiheadSelfAttention
from feed_forward_networks import FeedForwardNetworks


class TransformerDecoderBlock(nn.Module):

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        self.cfg: TrainingConfig = cfg
        # 层归一化 防止梯度爆炸（自注意力前）Pre-LN结构
        self.ln1 = nn.LayerNorm(cfg.embedding_dim,
                                elementwise_affine=cfg.use_bias)
        # 掩码多头自注意力
        self.mmsa = MaskedMultiheadSelfAttention(cfg)
        # 层归一化 （前馈网络前）Pre-LN结构
        self.ln2 = nn.LayerNorm(cfg.embedding_dim,
                                elementwise_affine=cfg.use_bias)
        # 前馈神经网络
        self.ffn = FeedForwardNetworks(cfg)

    def forward(self, x: Tensor, attention_mask: Tensor = None):
        # 残差连接1（自注意力分支）
        identity1 = x
        x = self.ln1(x)
        x = self.mmsa(x, attention_mask)
        x = identity1 + x
        # 残差连接2（前馈网络分支）
        identity2 = x
        x = self.ln2(x)
        x = self.ffn(x)
        y = identity2 + x
        return y
