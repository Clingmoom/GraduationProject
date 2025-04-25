import torch
import loralib as lora
import random

from decoder import TransformerDecoder
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from src.configs import TrainingConfig
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer


class GPTActor(nn.Module):

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        self.cfg: TrainingConfig = cfg
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        self.tokenizer.pad_token = '<|endoftext|>'  # '<|endoftext|>'  self.tokenizer.eos_token
        # 构建Transformer解码器主干网络
        self.transformer = TransformerDecoder(cfg)
        # Final linear layer as language model head w/o softmax
        if cfg.lora_rank > 0:
            # LoRA（低秩适应）的线性层 lm_head:language model head
            self.lm_head = lora.Linear(cfg.embedding_dim,
                                       cfg.vocab_size,
                                       bias=False,
                                       r=cfg.lora_rank)

        else:
            self.lm_head = nn.Linear(cfg.embedding_dim,
                                     cfg.vocab_size,
                                     bias=False)

        # 预测权重的线性层5分类（特定任务） 输入维度（Transformer隐藏层维度），输出维度
        self.predict_weight_token = nn.Linear(cfg.embedding_dim, 5, bias=False)
        # 预测差分步数的线性层3分类（特定任务）
        self.predict_diffstep_token = nn.Linear(cfg.embedding_dim, 3, bias=False)

    def forward(self, x: Tensor, attention_mask: Tensor = None):
        """
        x: Shape of (B, T) Batch_size,
        """
        x = self.transformer(x, attention_mask)  # x = (B, T, embedding_dim) （B,T,d）
        logits = self.lm_head(x)  # logits = (B, T, voca_size)
        diffw = self.predict_weight_token(x)
        diffstep = self.predict_diffstep_token(x)

        return logits, diffw, diffstep

    # 专门为强化学习设计的输出
    # 返回最后 num_actions 个 token 的对数概率，以及对应的权重和步数预测的对数概率
    def forward_actor(self,
                      x: Tensor,
                      attention_mask: Tensor = None,
                      num_actions=1):
        """
        x (B, T)
        """
        logits, diffw, diffstep = self.forward(x, attention_mask)  # logits = (B, T, voca_size)
        # token预测
        log_prob_all_vocab = F.log_softmax(logits[:, :-1, :], dim=2)  # (B, T-1, vocab_size)
        index = x[:, 1:].unsqueeze(-1)  # (B, T-1, 1)获取真实token索引
        # input.gather(dim，index):
        # 根据给定的索引（index）张量，
        # 从源张量（input）沿指定的维度（dim）抓取对应位置的值，放入新张量
        # index各个维度大小要与input除dim外保持一致
        # dim表示在哪个维度上进行softmax操作
        # gather输出形状是index的形状
        log_prob_output = log_prob_all_vocab.gather(dim=2, index=index) # 教师强制：取出真实下一个 token 的对数概率

        # 权重预测
        log_prob_all_w = F.log_softmax(diffw[:, :-1, :], dim=2)  # (B, T-1, 5)
        w_index = torch.ones(log_prob_all_w.shape[:2]).long().unsqueeze(-1).to(diffw.device) * 2
        log_prob_w_output = log_prob_all_w.gather(dim=2, index=w_index)  # 取类别 2 的对数概率（固定 teacher-forcing）
        # 步数预测
        log_prob_all_step = F.log_softmax(diffstep[:, :-1, :], dim=2)  # (B, T-1, 3)
        step_index = torch.ones(log_prob_all_w.shape[:2]).long().unsqueeze(-1).to(diffw.device)
        log_prob_step_output = log_prob_all_step.gather(dim=2, index=step_index) # 取类别 1 的对数概率

        return (log_prob_output[:, -num_actions:, 0],
                log_prob_w_output[:, -num_actions:, 0],
                log_prob_step_output[:, -num_actions:, 0])  # (B, 1)

    # 批量生成序列及对应的权重和步数预测
    @torch.no_grad()
    def batch_generate_first(self,
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

        total_length = min(max_input_length + random.randint(15, 77), 154)

        if T < total_length:
            idx = F.pad(idx, (0, total_length - T), value=int(50256))
            input_masks = F.pad(input_masks, (0, total_length - T), value=0.0)
        input_masks = input_masks.bool()

        diffw_list = torch.ones_like(idx) * 2
        diffstep_list = torch.ones_like(idx)
        for curr_pos in range(min_input_length, total_length):
            # forward the model to get the logits for the index in the sequence

            logits, diffw, diffstep = self(idx[:, :curr_pos])  # B, T, D

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            diffw = diffw[:, -1, :] / temperature
            diffstep = diffstep[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            next_id = torch.multinomial(probs, num_samples=1).view(-1)
            next_id = torch.where(input_masks[:, curr_pos], idx[:, curr_pos],
                                  next_id)  #
            # append sampled index to the running sequence and continue
            idx[:, curr_pos] = next_id

            diffw_probs = F.softmax(diffw, dim=-1)
            diffw_next_id = torch.multinomial(diffw_probs, num_samples=1).view(-1)
            diffw_list[:, curr_pos] = diffw_next_id

            diffstep_probs = F.softmax(diffstep, dim=-1)
            diffstep_next_id = torch.multinomial(diffstep_probs, num_samples=1).view(-1)
            diffstep_list[:, curr_pos] = diffstep_next_id

        return idx, diffw_list, diffstep_list

    # 包装方法，处理生成结果的掩码等
    def batch_generate(self,
                       idx,
                       input_masks: torch.Tensor,
                       input_lengths: torch.Tensor,
                       max_new_tokens,
                       temperature=1,
                       top_k=None):
        """
        idx: Shape of (B, T)
        """

        B, T = idx.size()
        completions, diffw_list, diffstep_list = self.batch_generate_first(idx, input_masks, input_lengths,
                                                                           max_new_tokens,
                                                                           temperature,
                                                                           top_k)  # completions = (B, T)

        attention_mask = torch.where(completions != int(50256),
                                     torch.ones_like(completions),
                                     torch.zeros_like(completions))
        action_mask = torch.ones_like(completions, dtype=torch.bool)
        action_mask[:, :T] = 0.0
        action_mask = action_mask[:, 1:]
        # we can only take the minimum among all instances in this batch as common num_actions
        num_actions = completions.size(1) - T
        return completions, attention_mask, num_actions, action_mask[:, -num_actions:], diffw_list, diffstep_list

    # 动态生成序列及对应的权重和步数预测，直到遇到结束符
    @torch.no_grad()
    def generate_dy(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        https://github.com/karpathy/nanoGPT/blob/master/model.py#L343

        Take a conditioning sequence of idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # 初始化三个空张量
        diffw_list = torch.tensor([], device=idx.device)
        diffstep_list = torch.tensor([], device=idx.device)
        new_idx = torch.tensor([], device=idx.device)
        # 循环 max_new_tokens 次，每次生成一个新的token
        for _ in range(max_new_tokens):
            # 如果当前序列长度小于等于模型配置的block_size，则使用当前序列；否则，截取最后block_size长度的序列if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]

            # 将处理后的序列输入模型，得到三个输出forward the model to get the logits for the index in the sequence
            logits, diffw, diffstep = self(idx_cond)

            # 将logits、diffw和diffstep的最后一个时间步的输出除以温度参数temperature，以控制生成过程的随机性pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            diffw = diffw[:, -1, :] / temperature
            diffstep = diffstep[:, -1, :] / temperature

            # 如果设置了top_k，则只保留概率最高的top_k个选项，其他选项的logits设为负无穷 optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # 使用softmax将logits转换为概率分布，然后从概率分布中采样得到下一个令牌的索引next_id apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            next_id = torch.multinomial(probs, num_samples=1)

            # 将新采样的令牌next_id添加到当前序列idx和new_idx中 append sampled index to the running sequence and continue
            idx = torch.cat((idx, next_id), dim=1)
            new_idx = torch.cat((new_idx, next_id), dim=1)

            # 对diffw和diffstep也进行相同的处理，计算概率分布并采样，然后将结果添加到diffw_list和diffstep_list中
            diffw_probs = F.softmax(diffw, dim=-1)
            diffw_next_id = torch.multinomial(diffw_probs, num_samples=1).view(-1)
            diffw_list = torch.cat((diffw_list, diffw_next_id.unsqueeze(0)), dim=1)
            diffstep_probs = F.softmax(diffstep, dim=-1)
            diffstep_next_id = torch.multinomial(diffstep_probs, num_samples=1).view(-1)
            diffstep_list = torch.cat((diffstep_list, diffstep_next_id.unsqueeze(0)), dim=1)
            # 如果生成的令牌索引为50256（句子结束符），则终止循环
            if next_id.item() == 50256:
                break

        return new_idx, diffw_list, diffstep_list

    @classmethod
    def from_checkpoint(cls,
                        cfg: TrainingConfig,
                        ckpt_path: str,
                        compile=False):
        model = GPTActor(cfg)

        if compile:
            model = torch.compile(model)

        checkpoint = torch.load(ckpt_path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model

    # 从预训练模型加载权重
    @classmethod
    def from_pretrained(cls, cfg: TrainingConfig):
        """
        https://github.com/karpathy/nanoGPT/blob/master/model.py#L213
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

        model = GPTActor(cfg)

        model_states = model.state_dict()

        model_states_keys = [
            k for k in model_states.keys() if not k.endswith('.mmsa.mask')
        ]
        model_states_keys = [k for k in model_states_keys if not 'lora' in k]

        model_pretrained = GPT2LMHeadModel.from_pretrained("gpt2-medium")

        pretrained_states = model_pretrained.state_dict()

        pretrained_states_keys = [
            k for k in pretrained_states.keys() if not k.endswith('.attn.masked_bias')
        ]
        pretrained_states_keys = [
            k for k in pretrained_states_keys if not k.endswith('.attn.bias')
        ]

        for dst_key in model_states_keys:
            if "predict_weight_token" in dst_key or "predict_diffstep_token" in dst_key:
                continue
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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        https://github.com/karpathy/nanoGPT/blob/master/model.py#L343

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

