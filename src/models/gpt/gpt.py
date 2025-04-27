import torch
import loralib as lora

from .decoder import TransformerDecoder
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from src.configs import TrainingConfig
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer


class GPT(nn.Module):

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        self.cfg: TrainingConfig = cfg
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.transformer = TransformerDecoder(cfg)
        # Final linear layer as language model head w/o softmax
        if cfg.lora_rank > 0:
            self.lm_head = lora.Linear(cfg.embedding_dim,
                                       cfg.vocab_size,
                                       bias=False,
                                       r=cfg.lora_rank)
            # self.lm_head = nn.Linear(cfg.embedding_dim,
            #                 cfg.vocab_size,
            #                 bias=False)
        else:
            self.lm_head = nn.Linear(cfg.embedding_dim,
                                     cfg.vocab_size,
                                     bias=False)

    # 前向传播
    def forward(self, x: Tensor, attention_mask: Tensor = None):
        """
        x: Shape of (B, T)
        """
        x = self.transformer(x, attention_mask)  # x = (B, T, embedding_dim)
        logits = self.lm_head(x)  # logits = (B, T, voca_size)
        return logits

    # 从检查点加载
    @classmethod
    def from_checkpoint(cls,
                        cfg: TrainingConfig,
                        ckpt_path: str,
                        compile=False):
        model = GPT(cfg)
        if compile:
            model = torch.compile(model)
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        return model

    # 从预训练模型加载权重
    @classmethod
    def from_pretrained(cls, cfg: TrainingConfig):
        """
        https://github.com/karpathy/nanoGPT/blob/master/model.py#L213
        实现了将HuggingFace GPT-2的权重映射到当前模型结构
        包含权重转置等适配处理
        """
        def convert_state_key(k):
            huggingface_names = {
                "token_embedding_layer": "wte",
                "postion_embedding_layer": "wpe",
                "decoder_blocks": "h",
                "mmsa": "attn",
                "ln1": "ln_1",
                "ln2": "ln_2",
                "ffn": "mlp",
                "fc1": "c_fc",
                "fc2": "c_proj",
                "qkv_projection": "c_attn",
                "output_projection": "c_proj",
                "ln": "ln_f",
            }
            hf_key = []
            for name in k.split('.'):
                hf_key.append(huggingface_names.get(name, name))
            return '.'.join(hf_key)

        def should_transpose(k):
            transposed = [
                'attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight',
                'mlp.c_proj.weight'
            ]
            for t in transposed:
                if k.endswith(t):
                    return True
            return False

        model = GPT(cfg)

        model_states = model.state_dict()

        model_states_keys = [
            k for k in model_states.keys() if not k.endswith('.mmsa.mask')
        ]

        model_states_keys = [k for k in model_states_keys if not 'lora' in k]

        model_pretrained = GPT2LMHeadModel.from_pretrained("gpt2")

        pretrained_states = model_pretrained.state_dict()

        pretrained_states_keys = [
            k for k in pretrained_states.keys()
            if not k.endswith('.attn.masked_bias')
        ]
        pretrained_states_keys = [
            k for k in pretrained_states_keys if not k.endswith('.attn.bias')
        ]

        for dst_key in model_states_keys:
            src_key = convert_state_key(dst_key)
            if should_transpose(src_key):
                assert pretrained_states[src_key].shape[::-1] == model_states[
                    dst_key].shape
                with torch.no_grad():
                    model_states[dst_key].copy_(pretrained_states[src_key].t())
            else:
                assert pretrained_states[src_key].shape == model_states[
                    dst_key].shape
                with torch.no_grad():
                    model_states[dst_key].copy_(pretrained_states[src_key])

        return model

    # 自回归文本生成
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        https://github.com/karpathy/nanoGPT/blob/master/model.py#L343
        自回归生成文本
        使用温度调节和top-k采样控制生成多样性
        Take a conditioning sequence of idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(
                1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            next_id = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, next_id), dim=1)

        return idx

    # 批量生成
    @torch.no_grad()
    def batch_generate(self,
                       idx: torch.Tensor,
                       input_masks: torch.Tensor,
                       input_lengths: torch.Tensor,
                       max_new_tokens: int,
                       temperature=1.0,
                       top_k=None):
        """
        idx: (B, T)
        input_masks: (B, T)
        """
        B, T = idx.size()
        min_input_length = torch.min(input_lengths)  # (B)
        max_input_length = torch.max(input_lengths)  # (B)
        total_length = min(self.cfg.block_size,
                           max_input_length + max_new_tokens)

        if T < total_length:
            idx = F.pad(idx, (0, total_length - T), value=int(50256))
            input_masks = F.pad(input_masks, (0, total_length - T), value=0.0)
        input_masks = input_masks.bool()

        for curr_pos in range(min_input_length, total_length):
            # forward the model to get the logits for the index in the sequence

            logits = self(idx[:, :curr_pos])
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:

                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            next_id = torch.multinomial(probs, num_samples=1).view(-1)
            next_id = torch.where(input_masks[:, curr_pos], idx[:, curr_pos],
                                  next_id)
            # append sampled index to the running sequence and continue
            idx[:, curr_pos] = next_id

        return idx

    def forward_actor(self,
                      x: Tensor,
                      attention_mask: Tensor = None,
                      num_actions=1):
        """
        x (B, T)
        """
        logits = self.forward(
            x, attention_mask)  # logits = (B, T, voca_size)
        log_prob_all_vocab = F.log_softmax(logits[:, :-1, :],
                                           dim=2)  # (B, T-1, vocab_size)
        # no need to know the logits of last token because we don't have the index of that token in x
        index = x[:, 1:].unsqueeze(-1)  # (B, T-1, 1)
        log_prob_output = log_prob_all_vocab.gather(
            dim=2,
            index=index)  # teacher-forcing, get the prob of each gt token

        return log_prob_output[:, -num_actions:, 0] # (B, T)
