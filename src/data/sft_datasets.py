import random
import numpy as np
import torch

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


class SFT_Datasets(Dataset):
    def __init__(self,device="cuda",block_size=77)->None:
        super().__init__()
        print("Load SFT Datasets.")
        self.device = device
        self.block_size = block_size

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2",device=device)
        tokenizer.pad_token = tokenizer.eos_token # 填充符

        def replace_period_with_comma(text):
            # 文本预处理函数 随机 逗号替换为句号
            return ''.join(
                ['.' if c == ',' and random.random() < 0.5
                 else c for c in text]
            )

        all_tokens = []

        prompt_list = np.load("train_data.npy")
        for prompt in prompt_list:
            response_text = prompt.lower() if prompt.isupper() else (prompt.capitalize() if random.random() < 0.5 else prompt)
            processed_text = replace_period_with_comma(response_text)
            suffix = "" if "<|endoftext|>" in processed_text else "<|endoftext|>"
            response = tokenizer(processed_text + suffix)


            all_tokens.extend(response["input_ids"])

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)

    def __len__(self):
        import sys
        return sys.maxsize

    def __getitem__(self, index):
        # 随机选择起始位置 (确保最后一个token有对应的后续token)
        max_start = len(self.tokens) - self.block_size - 1
        start = random.randint(0, max(max_start, 0))  # 处理短于 block_size 的情况，避免负数索引，长度不够自动归零

        x = self.tokens[start:start + self.block_size]  # 输入序列 x 包含 block_size 个 token
        y = self.tokens[start + 1:start + self.block_size + 1]  # 目标序列 y 是 x 的右移版本（自回归预测下一个token）
        return x, y  # 形状均为（block_size）
