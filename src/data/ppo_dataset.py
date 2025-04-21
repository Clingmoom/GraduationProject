import torch
import random
import numpy as np

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


class PPO_Dataset(Dataset):
    def __init__(self, device="cuda", block_size=77) -> None:
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

        prompt_list = np.load("train_data.npy")
        for prompt in prompt_list:
            response_text = prompt.lower() if prompt.isupper() else (prompt.capitalize() if random.random() < 0.5 else prompt)

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
                 torch.sum(tokens['attention_mask'])
            ])

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx][0], self.tokens[idx][1], self.tokens[idx][2]
