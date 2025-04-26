import torch
import torch.nn.functional as F

from torch import nn


class CrossEntropyLoss(nn.Module):

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):
        """
        y_hat: (B, T, vocab_size) -模型输出（未归一化的logits）
        y: (B, T)                 -目标token索引
        """
        # Convert y_hat to (B*T, vocab_size), y to (B*T)
        # 使用.view(-1)会将其展平成一维，形状变成(B*T,)
        # size(-1)是获取张量最后一个维度的大小 即 vocab_size
        return F.cross_entropy(y_hat.view(-1, y_hat.size(-1)),
                               y.view(-1),
                               ignore_index=-1)
# if __name__ == "__main__":
#     B = 1
#     T = 2
#     vocab_size = 3
#     y_hats = torch.randn(B*T, vocab_size)
#     y = [None] * B * T
#     for y_hat,y in zip(y_hats,y):
#         模型预测下一个token正确的概率=y_hat[y]