import torch

from torch import nn
from torch import Tensor
from src.configs import TrainingConfig
from torch.utils.checkpoint import checkpoint
from transformer_decoder_block import TransformerDecoderBlock


class TransformerDecoder(nn.Module):

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding_layer = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)  # 词嵌入 token映射为向量(Vocab, d)
        self.postion_embedding_layer = nn.Embedding(cfg.block_size,cfg.embedding_dim) # 位置编码 捕获序列顺序信息
        self.input_dropout = nn.Dropout(cfg.dropout_rate)
        self.decoder_blocks = nn.ModuleList(
            [TransformerDecoderBlock(cfg) for _ in range(cfg.n_layers)])# 核心：堆叠 N个 DecoderBlock
        # nnMlist = []
        # for _ in range(cfg.n_layers):
        #     self.nnMlist.append(TransformerDecoderBlock(cfg))
        # self.decoder_blocks = nn.ModuleList(nnMlist)
        # 最终层归一化
        self.ln = nn.LayerNorm(cfg.embedding_dim,
                               elementwise_affine=cfg.use_bias)

    def forward(self, x: Tensor, attention_mask: Tensor = None):
        B, T = x.size()
        token_embeddings = self.token_embedding_layer(x)  # (B, T, d)

        pos = torch.arange(0 , T , dtype=torch.long,
                               device=x.device).unsqueeze(0)

        pos_embeddings = self.postion_embedding_layer(pos)  # (B, T, d)
        # 正则化
        x = self.input_dropout(token_embeddings + pos_embeddings)

        # N decoder blocks
        # 每个block依次进行：
        # x = layer_norm(x)
        # x = masked_attention(x) + x  # 残差连接1
        # x = layer_norm(x)
        # x = feed_forward(x) + x     # 残差连接2
        for block in self.decoder_blocks:
            if self.cfg.activation_checkpointing:
                x = checkpoint(block, x, attention_mask)
            else:
                x = block(x, attention_mask)

        y = self.ln(x)
        return y
