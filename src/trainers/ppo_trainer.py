import re
import torch
import torch.optim as optim
from torch import Tensor

from tqdm import tqdm
from datetime import datetime
from typing import cast, Tuple
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from .trainer import Trainer
from .experience import Experience
from src.configs import TrainingConfig
from src.models import GPTActor, GPTCritic
from src.loss import ValueLoss, PolicyLoss
from .prompt_scorer import PromptScorer


class PPOTrainer(Trainer):
    def __init__(
        self,
        cfg: TrainingConfig,
        actor: GPTActor,
        critic: GPTCritic, # GPT
        sft_model: GPTActor,
        train_dataset,
        device,
        num_images_per_prompt = None
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_name = f"ppo_{cfg.exp_name}_{datetime.now().strftime('%m%d%H')}"
        print(f"self.run_name:{self.run_name}")
        self.device = device
        self.max_new_tokens = 77
        self.pattern = r'\[([^]]*):0-1:1\.0\]'#r'\[(\s*\w+):0-1:1\.0\]'

        self.orig_actor = actor
        self.orig_critic = critic
        #self.orig_sft_model = sft_model
        self.sft_model = sft_model

        self.actor = cast(GPTActor, torch.compile(self.orig_actor)) # 策略网络（生成文本）
        self.critic = cast(GPTCritic, torch.compile(self.orig_critic)) # 评价网络
        #self.sft_model = cast(GPTActor, torch.compile(self.orig_sft_model)) # 参考网络

        # 初始化评分器 （基于StableDiffusion生成图片后的PickScore+CLIP+Aesthetic）
        self.scorer =PromptScorer(device=device,num_images_per_prompt=num_images_per_prompt)

        # Separate actor loss from critic loss to save optimizer memory
        self.actor_criterion = PolicyLoss()
        self.critic_criterion = ValueLoss()

        self.step_dict={
            0:self.actor.tokenizer.encode("0-0.5"),
            1:self.actor.tokenizer.encode("0-1"),
            2:self.actor.tokenizer.encode("0.5-1"),
        }

        self.w_dict={
            0:self.actor.tokenizer.encode("0.5"),
            1:self.actor.tokenizer.encode("0.75"),
            2:self.actor.tokenizer.encode("1"),
            3:self.actor.tokenizer.encode("1.25"),
            4:self.actor.tokenizer.encode("1.5"),
        }

        self.token_dict={
            ",":self.actor.tokenizer.encode(","),
            ".":self.actor.tokenizer.encode("."),
            ":":self.actor.tokenizer.encode(":"),
            " [":self.actor.tokenizer.encode(" ["),
            "[":self.actor.tokenizer.encode("["),
            "]":self.actor.tokenizer.encode("]"),
            " ":self.actor.tokenizer.encode(" "),
        }

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            num_workers=12,
            prefetch_factor=4,
            pin_memory=True,
        )

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=cfg.actor_lr,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=cfg.critic_lr,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
        )

        self.step=0

        self.writer = SummaryWriter(f"./runs/{self.run_name}/logs", max_queue=50)
        self.total_epochs = cfg.total_epochs
        self.debug = False
        self.save_freq = 1000
        self.dtype = torch.float16

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.finetune_method = cfg.finetune_method

        hp = {
            "max_new_tokens": self.max_new_tokens,
            "train_dataset": type(train_dataset).__name__,
            "train_dataset_len": len(train_dataset),
            "dtype": str(self.dtype),
            **cfg.dict(),
        }
        self.save_hyperparams(hp)
        print("Initialized PPO Trainer")

    def trans_token(self, bef_list, diffw_list, diffstep_list):
        '''
        将原始 token 序列转换为带动态参数格式
        例：[house] -> [house:0-1:1.0]
        插入逻辑：根据 diffw/diffstep 统计结果选择最优参数
        '''
        if len(bef_list) == 0:
            return bef_list
        aft_list = torch.tensor([], device=bef_list.device)
        ind = 0
        token = bef_list[ind]

        def get_modes(special_token_ind_list):
            """根据 special token 索引计算 diffw 和 diffstep 的 mode"""
            try:
                w_counts = torch.bincount(diffw_list[special_token_ind_list])
                w_mode = int(torch.argmax(w_counts).item())
            except (RuntimeError, ValueError, IndexError):
                w_mode = 2
            try:
                counts = torch.bincount(diffstep_list[special_token_ind_list])
                mode = int(torch.argmax(counts).item())
            except (RuntimeError, ValueError, IndexError):
                mode = 1
            return w_mode, mode

        def insert_special_tokens(special_token_ind_list, w_mode, mode):
            """根据 mode 和 w_mode 插入动态 token 格式"""
            for idx in special_token_ind_list:
                s_token = bef_list[idx]
                aft_list_local = torch.cat([
                    self.token_dict["["].unsqueeze(0),
                    s_token.unsqueeze(0),
                    self.token_dict[":"].unsqueeze(0),
                    self.step_dict[mode],
                    self.token_dict[":"].unsqueeze(0),
                    self.w_dict[w_mode],
                    self.token_dict["]"].unsqueeze(0)
                ])
                nonlocal aft_list
                aft_list = torch.cat([aft_list, aft_list_local])

        if not (token == self.token_dict[","] or token == self.token_dict["."]):
            # --- 处理开头非逗号句号的情况 ---
            while not (token == self.token_dict[","] or token == self.token_dict["."] or token == self.token_dict[" "]):
                token = bef_list[ind]
                aft_list = torch.cat([aft_list, token.unsqueeze(0)])
                ind += 1
                if ind >= len(bef_list):
                    break
            if ind < len(bef_list):
                token = bef_list[ind]

            special_token_ind_list = []
            while ind < len(bef_list) and not (
                    token == self.token_dict[","] or token == self.token_dict["."] or self.tokenizer.decode(
                    [token.long()]).startswith(" ")):
                if token == self.token_dict[" "]:
                    aft_list = torch.cat([aft_list, token.unsqueeze(0)])
                    ind += 1
                    if ind >= len(bef_list):
                        break
                    token = bef_list[ind]
                else:
                    special_token_ind_list.append(ind)
                    ind += 1
                    if ind >= len(bef_list):
                        break
                    token = bef_list[ind]
                    if token == self.token_dict[","] or token == self.token_dict["."]:
                        break

            w_mode, mode = get_modes(special_token_ind_list)
            insert_special_tokens(special_token_ind_list, w_mode, mode)
            ind += 1

        # --- 开始统一处理后续所有 token ---
        while ind < len(bef_list):
            token = bef_list[ind]

            if not (token == self.token_dict[","] or token == self.token_dict["."]):
                aft_list = torch.cat([aft_list, token.unsqueeze(0)])
                ind += 1
            else:
                ind += 1
                if ind >= len(bef_list):
                    break
                token = bef_list[ind]

                special_token_ind_list = []
                while not (token == self.token_dict[","] or token == self.token_dict["."]):
                    special_token_ind_list.append(ind)
                    ind += 1
                    if ind >= len(bef_list):
                        break
                    token = bef_list[ind]
                    if token == self.token_dict[","] or token == self.token_dict["."]:
                        break

                aft_list = torch.cat([aft_list, self.token_dict[","].unsqueeze(0)])

                w_mode, mode = get_modes(special_token_ind_list)
                insert_special_tokens(special_token_ind_list, w_mode, mode)
                ind += 1

        return aft_list

    def kl_penalized_reward(
            self,
            reward: torch.Tensor,
            log_prob_rl: torch.Tensor,
            log_prob_sft: torch.Tensor,
            action_mask: torch.Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        '''
        计算 KL 惩罚后的奖励。
        Args:
            reward: (B, 1) 初始环境奖励
            log_prob_rl: (B, T) 当前策略生成动作的log概率
            log_prob_sft: (B, T) 参考策略生成动作的log概率
            action_mask: (B, T) 是否是有效动作的位置掩码
        Returns:
            penalized_reward: (B, 1) 加了KL惩罚后的奖励
            estimated_kl: (B, 1) 平均KL散度
        '''
        # log(π_RL(y|x) / π_SFT(y|x)) = log(π_RL(y|x)) - log(π_SFT(y|x))
        ratio = log_prob_rl - log_prob_sft
        estimated_kl = (torch.exp(ratio) - 1) - ratio  # 二阶近似
        if action_mask is not None:
            estimated_kl = (estimated_kl * action_mask).sum(dim=1) / action_mask.sum(dim=1)
        else:
            estimated_kl = estimated_kl.mean(dim=1)
        estimated_kl = estimated_kl.unsqueeze(-1)  # 保持输出shape (B, 1)

        penalized_reward = reward - self.cfg.kl_beta * estimated_kl
        return penalized_reward, estimated_kl

    @torch.no_grad()
    def make_experience(self, idx, input_masks, input_lengths):
        # self.reward_model.eval()
        self.sft_model.eval()
        self.actor.eval()
        self.critic.eval()
        (   completion,     # 完整扩展token序列 （B，T）
            attention_mask, # 标记哪些位置是有效token
            num_actions,    # 新生成动作的数量
            action_mask,    # 标记哪些是actor新生成的动作
            diffw_list,
            diffstep_list
        ) = self.actor.batch_generate(
            idx,
            input_masks,
            input_lengths,
            self.max_new_tokens,
            # 采样方式
            temperature=1.0,
            top_k=50,
        )

        if self.debug:
            print(" --- Make Experience --- ")
            print("completion", completion.shape)
            print("input_masks", input_masks.shape)
            print("num_actions", num_actions)
            print("action_mask", action_mask.shape)
            print("idx", idx.shape)
            print("input_masks", input_masks.shape)

        #计算生成文本的动作log概率
        actor_log_probs, w_log_probs, step_log_probs = self.actor.forward_actor(
            completion, attention_mask, num_actions  # (B, num_actions)
        )
        # 参考模型（来自train_stage_1）预测动作log概率（用于KL对比）
        sft_log_probs, sft_w_log_probs, sft_step_log_probs = self.sft_model.forward_actor(
            completion, attention_mask, num_actions
        )  # (B, num_actions)
        # 估算每个 completion 的价值
        values = self.critic.forward_critic(completion, attention_mask, num_actions).view(-1, 1)  # (B, 1)

        # 构造原始提示的文本
        input_prompt = [self.tokenizer.decode(completion[i,:input_lengths[i]]) for i in range(completion.size(0))]
        output_prompt=[]
        target = [torch.tensor(220,device = completion.device),
                  torch.tensor(50256, device = completion.device)]
        target_value = torch.tensor(50256, device = completion.device)

        for i in range(completion.size(0)):
            res = completion[i, input_lengths[i]:]
            input_w = diffw_list[i, input_lengths[i]:]
            input_step = diffstep_list[i, input_lengths[i]:]
            # 裁剪新生成的动作 直到end前
            indices = [i for i, sublist in enumerate(zip(res, res[1:])) if list(sublist) == target]
            if len(indices) > 0:
                end = int(indices[0])
                res = res[:end]
                input_w = input_w[:end]
                input_step = input_step[:end]
            if target_value in res:
                end = res.cpu().numpy().tolist().index(target_value)
                res = res[:end]
                input_w = input_w[:end]
                input_step = input_step[:end]

            # 把新生成的 token（res）+ 对应的 diffw 和 diffstep
            output_tokens = self.trans_token(res, input_w, input_step)
            res = self.tokenizer.decode( torch.cat([completion[i, :input_lengths[i]], output_tokens]) )

            end = res.find("[<|endoftext|>")
            if end > 0:
                res = res[:end]
            end = res.find("<|endoftext|>")
            if end > 0:
                res = res[:end]
            res = re.sub(self.pattern, r'\1', res)
            output_prompt.append(res)

        reward = self.scorer.get_score_batched(prompts = output_prompt, plain_texts = input_prompt).unsqueeze(1) # (B, 1)

        if self.debug:
            print("actor_log_probs", actor_log_probs.shape)
            print("sft_log_probs", sft_log_probs.shape)
            print("values", values.shape)
            print("reward", reward.shape)

        kl_penalized_reward, estimated_kl = self.kl_penalized_reward(reward, actor_log_probs, sft_log_probs)
        w_kl_penalized_reward, w_estimated_kl = self.kl_penalized_reward(reward, w_log_probs, sft_w_log_probs)
        step_kl_penalized_reward, step_estimated_kl = self.kl_penalized_reward(reward, step_log_probs, sft_step_log_probs)

        advantage = kl_penalized_reward - values
        w_advantage = w_kl_penalized_reward - values
        step_advantage = step_kl_penalized_reward - values

        # advantage normalization ~ 标准化
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        w_advantage = (w_advantage - w_advantage.mean()) / (w_advantage.std() + 1e-8)
        step_advantage = (step_advantage - step_advantage.mean()) / (step_advantage.std() + 1e-8)

        if self.debug:
            print("kl_penalized_reward", kl_penalized_reward)
            print("advantage", advantage.shape) #[B, 1]

        return Experience(
            completion,
            actor_log_probs,
            w_log_probs,
            step_log_probs,
            attention_mask,
            kl_penalized_reward,
            advantage,
            w_advantage,
            step_advantage,
            num_actions,
            estimated_kl,
            w_estimated_kl,
            step_estimated_kl,
            values,
            action_mask,
        )

    def save_states(self,
        step,
        is_last=False
    ):
        file_name = (
            "actor_final.pt"
            if is_last
            else f"actor_step{step}.pt"
        )
        torch.save(
            {
                "step": step,
                "model_state_dict": self.orig_actor.state_dict(),  # Save the unoptimized model
                "optimizer_state_dict": self.actor_optimizer.state_dict(),
            },
            f"./runs/{self.run_name}/{file_name}",
        )
        file_name = (
            f"critic_final.pt"
            if is_last
            else f"critic_step{step}.pt"
        )
        torch.save(
            {
                "step": step,
                "model_state_dict": self.orig_critic.state_dict(),
                "optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            f"./runs/{self.run_name}/{file_name}",
        )

    def fit(self):
        scaler = GradScaler(enabled = self.dtype != torch.float32)

        print(f"self.total_epochs: {self.total_epochs} self.train_dataloader:{len(self.train_dataloader)}")

        for epoch in range(self.total_epochs):
            # 遍历训练数据加载器
            for step, (prompt, input_masks, input_lengths) in enumerate(
                pbar := tqdm(self.train_dataloader) # pbar 进度条 是tqdm的实例
            ):
                step = step + self.step
                if len(prompt.shape) == 3:
                    prompt = prompt.squeeze(1)
                    input_masks = input_masks.squeeze(1)

                prompt, input_masks, input_lengths = (
                    prompt.to(self.device),
                    input_masks.to(self.device),
                    input_lengths.to(self.device),
                )

                if self.debug:
                    print("prompt", prompt.shape)

                real_input_length = torch.max(input_lengths)
                prompt = prompt[:, :real_input_length] # 截断到实际长度

                if self.debug:
                    print("input_lengths", input_lengths)
                    print("prompt after", prompt.shape)

                total_steps = step + epoch * len(self.train_dataloader)

                # 混合精度训练
                with torch.autocast(device_type = "cuda", dtype = self.dtype, enabled = self.dtype != torch.float32):
                    experience = self.make_experience(prompt, input_masks, input_lengths)
                    # 策略网络更新
                    self.actor.train()
                    curr_actor_log_probs, diffw_log_probs, diffstep_log_probs = self.actor.forward_actor(
                        experience.completion,
                        experience.attention_mask,
                        experience.num_actions,
                    )

                    if self.debug:
                        print("curr_actor_log_probs", curr_actor_log_probs.shape)
                        print("actor_log_probs", experience.actor_log_probs.shape)

                    actor_loss_token = self.actor_criterion(
                        curr_actor_log_probs,
                        experience.actor_log_probs,
                        experience.advantage,
                        experience.action_mask,
                    )
                    actor_loss_w = self.actor_criterion(
                        diffw_log_probs,
                        experience.w_log_probs,
                        experience.w_advantage,
                        experience.action_mask,
                    )
                    actor_loss_step = self.actor_criterion(
                        diffstep_log_probs,
                        experience.step_log_probs,
                        experience.step_advantage,
                        experience.action_mask,
                    )
                    actor_loss = actor_loss_token + actor_loss_w + actor_loss_step

                    scaler.scale(actor_loss).backward()
                    scaler.step(self.actor_optimizer)
                    self.actor_optimizer.zero_grad(set_to_none = True)
                    actor_lossf = actor_loss.item()

                    # 价值网络更新
                    self.critic.train()
                    new_values = self.critic.forward_critic(
                        experience.completion,
                        experience.attention_mask,
                        experience.num_actions,
                    ).view(-1, 1)

                    if self.debug:
                        print("new_value", new_values.shape)
                        print("reward", experience.kl_penalized_reward.shape)

                    critic_loss = self.critic_criterion(
                        new_values,
                        experience.kl_penalized_reward,
                        experience.values,
                        experience.action_mask,
                    )

                    scaler.scale(critic_loss).backward()
                    scaler.step(self.critic_optimizer)
                    self.critic_optimizer.zero_grad(set_to_none = True)
                    critic_lossf = critic_loss.item()
                    scaler.update()

                # 日志与检查点
                self.writer.add_scalar("KL", experience.estimated_kl.mean(), total_steps)
                self.writer.add_scalar("mean_advantage", experience.advantage.mean(), total_steps)
                self.writer.add_scalar("mean_reward", experience.kl_penalized_reward.mean(), total_steps)
                self.writer.add_scalar("mean_value", new_values.mean(), total_steps)
                self.writer.add_scalar("Loss/actor/step", actor_lossf, total_steps)
                self.writer.add_scalar("Loss/token/step", actor_loss_token.item(), total_steps)
                self.writer.add_scalar("Loss/w/step", actor_loss_w.item(), total_steps)
                self.writer.add_scalar("Loss/step/step", actor_loss_step.item(), total_steps)
                self.writer.add_scalar("Loss/critic/step", critic_lossf, total_steps)

                pbar.set_description(
                    f"actor loss {round(actor_lossf, 3)}, critic loss {round(critic_lossf, 3)}"
                )

                if ((total_steps == 500 or (total_steps != 0 and total_steps % self.save_freq == 0))):
                    self.save_states(total_steps)

        self.save_states(None, True)

