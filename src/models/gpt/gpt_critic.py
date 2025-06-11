import torch
import pdb

from torch import Tensor
from src.configs import TrainingConfig
from .gpt_reward import GPTRewardModel


# 价值模型（长期收益）
class GPTCritic(GPTRewardModel):

    def forward_critic(self,
                       x: Tensor,
                       attention_mask: Tensor = None,
                       num_actions = 0) -> torch.Tensor:
        '''
        计算输入序列的长期收益。
        Args:
            x: shape (B, T)，其中 B 是批次大小，T 是序列长度。
            attention_mask: shape (B, T)，用于指示序列中哪些位置是有效的。默认 None。
            num_actions: 新生成动作数量。默认 0。
        Returns:
            Tensor: 长期收益，形状为 (B, 1)
        '''
        hidden = self.backbone(x, attention_mask)  # (B, T, vocab_size)

        values = self.value_head(hidden).squeeze(-1)  # (B, T, 1)->(B, T)
        # Vt only depends on st
        values = values * attention_mask
        values = values[:, :-num_actions].mean(dim=1) # 只要原始 prompt 部分
        if torch.isnan(values).any().item(): # 如果 values中至少存在一个 NaN值
            print("values nan:" ,values.shape ,values)
            pdb.set_trace()
        return values  # (B, 1)

    @classmethod
    def from_checkpoint(cls,
                        cfg: TrainingConfig,
                        ckpt_path: str,
                        strict=False,
                        compile=False):
        '''
        从检查点加载模型。
        Args:
            cfg: 模型配置对象，包含模型的各种参数。
            ckpt_path: 检查点路径。
            strict (bool, optional): 是否严格加载模型。默认为 False。
            compile (bool, optional): 是否使用 PyTorch 的编译功能。默认为 False。
        Returns:
            GPTCritic: 加载了检查点的 GPTCritic 实例。
        '''
        model = GPTCritic(cfg)
        if compile:
            model = torch.compile(model)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        return model