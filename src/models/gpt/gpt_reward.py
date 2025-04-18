import torch
import loralib as lora

from torch import nn
from torch import Tensor
from src.configs import TrainingConfig
from gpt import GPT


# 奖励模型（即时反馈）
class GPTRewardModel(nn.Module):

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = GPT(cfg)
        # 将 GPT 模型的语言模型头（lm_head）替换为 nn.Identity()，即一个恒等映射，不进行任何操作
        self.backbone.lm_head = nn.Identity()
        # no need for LoRA here as we won't have weights anyway
        # 定义一个线性层 self.value_head，用于从 GPT 模型的隐藏状态生成奖励分数
        self.value_head = nn.Linear(cfg.embedding_dim, 1, bias=False)
        # self.value_head = nn.Sequential(
        # nn.Linear(cfg.embedding_dim, cfg.embedding_dim * 2),
        # nn.ReLU(),
        # nn.Linear(cfg.embedding_dim * 2, 1)
        # )

    def forward(self, x: Tensor, attention_mask: Tensor = None):
        '''
        前向传播函数，计算输入序列的奖励分数。
        Args:
            x (Tensor): 输入序列，形状为 (B, T)，其中 B 是批次大小，T 是序列长度。
            attention_mask (Tensor, optional): 注意力掩码，形状为 (B, T)，用于指示序列中哪些位置是有效的。默认为 None
        Returns:
            Tensor: 奖励分数，形状为 (B, 1)
        Notes:
            定义模型的前向传播过程;
            使用 GPT 模型计算隐藏状态 hidden
            通过 value_head 计算每个输入的奖励分数，并取平均值。
            返回奖励分数。
        '''
        hidden = self.backbone(x, attention_mask)
        score = self.value_head(hidden).mean(dim=1)
        return score

    def freeze_weights(self, finetune_method):
        '''
        冻结模型权重
        Args:
            finetune_method: 微调方法，字符串类型，可选值为 "lora" 或 "last_block"。
        Notes:
            根据指定的微调方法，冻结模型的部分权重。
            如果微调方法是 "lora" 且 lora_rank 大于 0，则调用 lora.mark_only_lora_as_trainable 方法，
            将模型的部分权重标记为可训练。
            如果微调方法是 "last_block"，则遍历模型的所有参数，
            将除了最后一个 Transformer 块的权重之外的所有参数的 requires_grad 属性设置为 False。
            最后，打印出不支持的微调方法的信息。
        '''
        if finetune_method == "lora" and self.cfg.lora_rank > 0:
            lora.mark_only_lora_as_trainable(self)
        elif finetune_method == "last_block":
            trainable_params = [
                "backbone.transformer.decoder_blocks.35.mmsa.mask",
                "backbone.transformer.decoder_blocks.35.mmsa.qkv_projection.weight",
                "backbone.transformer.decoder_blocks.35.mmsa.qkv_projection.bias",
                "backbone.transformer.decoder_blocks.35.mmsa.output_projection.weight",
                "backbone.transformer.decoder_blocks.35.mmsa.output_projection.bias",
                "backbone.transformer.decoder_blocks.35.ln2.weight",
                "backbone.transformer.decoder_blocks.35.ln2.bias",
                "backbone.transformer.decoder_blocks.35.ffn.fc1.weight",
                "backbone.transformer.decoder_blocks.35.ffn.fc1.bias",
                "backbone.transformer.decoder_blocks.35.ffn.fc2.weight",
                "backbone.transformer.decoder_blocks.35.ffn.fc2.bias",
                "backbone.transformer.ln.weight",
                "backbone.transformer.ln.bias", "value_head.weight"
            ]
            for name, param in self.named_parameters():
                if name not in trainable_params:
                    param.requires_grad = False
                else:
                    print(f"{name} is trainable.")
        else:
            print(
                f"Unsupported method {finetune_method} (lora rank = {self.cfg.lora_rank})"
            )

    @classmethod
    def from_backbone_checkpoint(cls, cfg: TrainingConfig, ckpt_path: str):
        '''
        从预训练模型的检查点加载模型。
        Args:
            cfg: 模型配置对象，包含模型的各种参数。
            ckpt_path: 预训练模型的检查点路径。
        Returns:
            GPTRewardModel: 加载了预训练模型权重的 GPTRewardModel 实例。
        Notes:
            从指定的检查点路径加载预训练模型的权重。
            创建一个新的 GPTRewardModel 实例，并将其 backbone 设置为从检查点加载的 GPT 模型。
            将 backbone 的语言模型头（lm_head）替换为 nn.Identity()，即一个恒等映射，不进行任何操作。
            返回加载了预训练模型权重的 GPTRewardModel 实例。
        '''
        cfg.pretrain = ckpt_path
        model = GPTRewardModel(cfg)
        model.backbone = GPT.from_checkpoint(cfg, ckpt_path)
        model.backbone.lm_head = nn.Identity()
        return model

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
            GPTRewardModel: 加载了检查点的 GPTRewardModel 实例。
        Notes:
            从指定的检查点路径加载模型。
            创建一个新的 GPTRewardModel 实例。
            如果 compile 为 True，则使用 PyTorch 的编译功能对模型进行编译。
            加载检查点中的模型状态字典，并将其加载到模型中。
            如果 strict 为 True，则在加载模型状态字典时使用严格模式。
            返回加载了检查点的 GPTRewardModel 实例。
        '''
        model = GPTRewardModel(cfg)
        if compile:
            model = torch.compile(model)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        return model

    @classmethod
    def from_pretrained(cls, cfg: TrainingConfig):
        '''
        从预训练模型加载模型。
        Args:
            cfg: 模型配置对象，包含模型的各种参数。
        Returns:
            GPTRewardModel: 加载了预训练模型的 GPTRewardModel 实例。
        Notes:
            从预训练模型加载模型。
            创建一个新的 GPTRewardModel 实例。
            将模型的 backbone 设置为从预训练模型加载的 GPT 模型。
            将 backbone 的语言模型头（lm_head）替换为 nn.Identity()，即一个恒等映射，不进行任何操作。
        '''
        model = GPTRewardModel(cfg)
        model.backbone = GPT.from_pretrained(cfg)
        model.backbone.lm_head = nn.Identity()
        return model
