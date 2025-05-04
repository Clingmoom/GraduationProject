import torch
import random
import numpy as np

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from src.configs import ROOT_DIR


class PPO_Dataset(Dataset):
    def __init__(self, device="cuda", block_size: int=77) -> None:
        if not (isinstance(block_size, int) or 0< block_size <=77):
            raise ValueError("block_size must be a positive integer and less than or equal to 77.")
        super().__init__()
        print("Load PPO Dataset.")
        self.device = device
        self.block_size = block_size
        self.tokens = []

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", device=device)
        tokenizer.pad_token = tokenizer.eos_token

        def replace_period_with_comma(text):
            # 文本预处理函数 随机 逗号替换为句号
            return ''.join(
                ['.' if c == ',' and random.random() < 0.5
                 else c for c in text]
            )

        prompt_list = np.load(ROOT_DIR / "data" / "training_data" / "train_data.npy")

        # TODO:考虑优化 先预处理再一起给tokenizer
        for prompt in prompt_list:
            # 随机选择首字母 进行大小写替换 isupper()
            if random.random() < 0.5:
                first_term = prompt
                if first_term.isupper(): # 判断字符串所有字母是否全部大写
                    first_term = first_term.lower()
                else:
                    first_term = first_term.capitalize() # 首字母大写 其余小写
                response_text = first_term
            else:
                response_text = prompt

            if "<|endoftext|>" in response_text:
                tokens = tokenizer(replace_period_with_comma(response_text),
                                   max_length=77,
                                   padding="max_length",
                                   truncation=True,
                                   return_tensors="pt")
            else:
                tokens = tokenizer(replace_period_with_comma(response_text) + "<|endoftext|>",
                                   max_length=77,
                                   padding="max_length",
                                   truncation=True,
                                   return_tensors="pt")

            self.tokens.append([tokens['input_ids'],
                 tokens['attention_mask'],
                 torch.sum(tokens['attention_mask']) # 统计非填充token数量（真实token）
            ])

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx][0], self.tokens[idx][1], self.tokens[idx][2]
        # (prompt, mask, input_length)