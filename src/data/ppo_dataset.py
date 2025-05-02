import torch
import random
import numpy as np

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from src.configs import ROOT_DIR


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

        prompt_list = np.load(ROOT_DIR / "data" / "training_data" / "train_data.npy")

        processed_prompts = []
        for prompt in prompt_list:
            text = prompt.lower() if prompt.isupper() else (prompt.capitalize() if random.random() < 0.5 else prompt)
            text = replace_period_with_comma(text)
            if "<|endoftext|>" not in text:
                text += "<|endoftext|>"
            processed_prompts.append(text)

        encoded = tokenizer.batch_encode_plus(
            processed_prompts,
            max_length=self.block_size,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        for i in range(len(processed_prompts)):
            self.tokens.append([
                input_ids[i],
                attention_mask[i],
                torch.sum(attention_mask[i])
            ])

        # DO:考虑优化 先预处理再一起给tokenizer
        # for prompt in prompt_list:
        #     response_text = prompt.lower() if prompt.isupper() else (prompt.capitalize() if random.random() < 0.5 else prompt)
        #
        #     if "<|endoftext|>" in response_text:
        #         tokens = tokenizer(replace_period_with_comma(response_text),
        #                            max_length=77,
        #                            padding="max_length",
        #                            truncation=True,
        #                            return_tensors="pt")
        #     else:
        #         tokens = tokenizer(replace_period_with_comma(response_text) + "<|endoftext|>",
        #                            max_length=77,
        #                            padding="max_length",
        #                            truncation=True,
        #                            return_tensors="pt")
        #
        #     self.tokens.append([tokens['input_ids'],
        #          tokens['attention_mask'],
        #          torch.sum(tokens['attention_mask']) # 统计非填充token数量（真实token）
        #     ])

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx][0], self.tokens[idx][1], self.tokens[idx][2]
        # (prompt, mask, input_length)