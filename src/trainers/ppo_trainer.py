import re
import torch
import torch.optim as optim

from tqdm import tqdm
from datetime import datetime
from typing import Union
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from trainer import Trainer
from experience import Experience
from src.configs import TrainingConfig
from src.models import GPTActor, GPTCritic
from src.loss import ValueLoss, PolicyLoss
from prompt_scorer import PromptScorer


class PPOTrainer(Trainer):
    def __init__(
        self,
        cfg: TrainingConfig,
        actor: GPTActor,
        critic: GPTCritic, # GPT
        sft_model: GPTActor,
        train_dataset,
        device,
        num_images_per_prompt=None
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
        self.orig_sft_model = sft_model

        self.actor = torch.compile(self.orig_actor) # 策略网络（生成文本）
        self.critic = torch.compile(self.orig_critic) # 评价网络
        self.sft_model = torch.compile(self.orig_sft_model) # 参考网络

        # 初始化评分器
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
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta1),
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=cfg.critic_lr,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta1),
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

    def find_special_token_indices(self, bef_list, start_ind):
        ind = start_ind
        token = bef_list[ind]
        special_token_ind_list = []
        while ind < len(bef_list) and not (
                token == self.token_dict[","] or token == self.token_dict["."] or self.tokenizer.decode(
                [token.long()]).startswith(" ")):
            if token == self.token_dict[" "]:
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
        return special_token_ind_list

    def calculate_optimal_params(self, diffw_list, diffstep_list, special_token_ind_list):
        try:
            w_counts = torch.bincount(diffw_list[special_token_ind_list])
            w_mode = int(torch.argmax(w_counts).item())
        except:
            w_mode = 2

        try:
            counts = torch.bincount(diffstep_list[special_token_ind_list])
            mode = int(torch.argmax(counts).item())
        except:
            mode = 1
        return w_mode, mode

    def insert_dynamic_params(self, aft_list, bef_list, special_token_ind_list, w_mode, mode):
        for ind in special_token_ind_list:
            aft_list = torch.cat([aft_list, self.token_dict["["].unsqueeze(0)])
            s_token = bef_list[ind]
            aft_list = torch.cat([aft_list, s_token.unsqueeze(0)])
            aft_list = torch.cat([aft_list, self.token_dict[":"].unsqueeze(0)])
            aft_list = torch.cat([aft_list, self.step_dict[mode]])
            aft_list = torch.cat([aft_list, self.token_dict[":"].unsqueeze(0)])
            aft_list = torch.cat([aft_list, self.w_dict[w_mode]])
            aft_list = torch.cat([aft_list, self.token_dict["]"].unsqueeze(0)])
        return aft_list

    def trans_token(self,bef_list,diffw_list,diffstep_list):
        '''
        将原始token序列转换为包含动态参数的格式
        示例转换：[house] -> [house:0-1:1.0]
        实现动态提示参数的插入逻辑
        根据diffw/diffstep统计结果选择最优参数组合
        输入：
            bef_list: 原始token序列
            diffw_list: 权重差异统计
            diffstep_list: 步数差异统计
        输出：
            aft_list: 转换后的token序列
        '''
        # 如果输入的原始token序列为空，直接返回空列表
        if len(bef_list) == 0:
            return bef_list
        # 初始化转换后的token序列，使用和bef_list相同的设备
        aft_list = torch.tensor([], device=bef_list.device)
        ind = 0

        while ind < len(bef_list):
            token = bef_list[ind]
            # 如果当前token不是逗号或句号
            if not (token == self.token_dict[","] or token == self.token_dict["."]):
                # 处理非特殊token
                while not (token == self.token_dict[","] or token == self.token_dict["."] or token == self.token_dict[
                    " "]):
                    aft_list = torch.cat([aft_list, token.unsqueeze(0)])
                    ind += 1
                    if ind >= len(bef_list):
                        break
                    token = bef_list[ind]

                # 定位特殊token范围
                special_token_ind_list = self.find_special_token_indices(bef_list, ind)

                # 统计最优参数组合
                w_mode, mode = self.calculate_optimal_params(diffw_list, diffstep_list, special_token_ind_list)

                # 插入动态参数
                aft_list = self.insert_dynamic_params(aft_list, bef_list, special_token_ind_list, w_mode, mode)
                ind = max(ind, max(special_token_ind_list) + 1 if special_token_ind_list else ind)
            else:
                ind += 1
                if ind >= len(bef_list):
                    break
                token = bef_list[ind]
                # 定位特殊token范围
                special_token_ind_list = self.find_special_token_indices(bef_list, ind)

                aft_list = torch.cat([aft_list, self.token_dict[","].unsqueeze(0)])

                # 统计最优参数组合
                w_mode, mode = self.calculate_optimal_params(diffw_list, diffstep_list, special_token_ind_list)

                # 插入动态参数
                aft_list = self.insert_dynamic_params(aft_list, bef_list, special_token_ind_list, w_mode, mode)
                ind = max(ind, max(special_token_ind_list) + 1 if special_token_ind_list else ind)

        return aft_list

    def _append_dynamic_params(self, aft_parts, s_token, step_mode, weight_mode):
        """动态参数插入统一方法"""
        aft_parts.append(self.token_dict["["].unsqueeze(0))
        aft_parts.append(s_token.unsqueeze(0))
        aft_parts.append(self.token_dict[":"].unsqueeze(0))
        aft_parts.append(self.step_dict[step_mode])
        aft_parts.append(self.token_dict[":"].unsqueeze(0))
        aft_parts.append(self.w_dict[weight_mode])
        aft_parts.append(self.token_dict["]"].unsqueeze(0))

    def save_states(self, step, is_last=False):
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

    def kl_penalized_reward(
        self,
        reward: torch.Tensor,
        log_prob_rl: torch.Tensor,
        log_prob_sft: torch.Tensor,
        action_mask: torch.Tensor = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        # log(π_RL(y|x) / π_SFL(y|x)) = log(π_RL(y|x)) - log(π_SFL(y|x))
        ratio = log_prob_rl - log_prob_sft
        # k3 in http://joschu.net/blog/kl-approx.html
        estimated_kl = (torch.exp(ratio) - 1) - ratio # 二阶近似计算
        if action_mask:
            estimated_kl = estimated_kl * action_mask
            estimated_kl.sum(dim=1) / action_mask.sum(dim=1)
        estimated_kl = estimated_kl.mean(
            dim=1, keepdim=True)  # estimated_kl -> (B, 1)
        return reward - self.cfg.kl_beta * estimated_kl, estimated_kl

    @torch.no_grad()
    def make_experience(self, idx, input_masks, input_lengths):
        # self.reward_model.eval()
        self.sft_model.eval()
        self.actor.eval()
        self.critic.eval()
        (
            completion,
            attention_mask,
            num_actions,
            action_mask,diffw_list,diffstep_list
        ) = self.actor.batch_generate(
            idx,
            input_masks,
            input_lengths,
            self.max_new_tokens,
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

        actor_log_probs,w_log_probs,step_log_probs = self.actor.forward_actor(
            completion, attention_mask, num_actions  # (B, num_actions)
        )
        sft_log_probs,sft_w_log_probs,sft_step_log_probs = self.sft_model.forward_actor(
            completion, attention_mask, num_actions
        )  # (B, num_actions)

        values = self.critic.forward_critic(completion, attention_mask, num_actions).view(
            -1, 1
        )  # (B, 1)

        #
        input_prompt=[ self.tokenizer.decode(completion[i,:input_lengths[i]]) for i in range(completion.size(0))]

        output_prompt=[]
        target =  [torch.tensor(220,device=completion.device), torch.tensor(50256,device=completion.device)]
        target_value = torch.tensor(50256,device=completion.device)
        for i in range(completion.size(0)):

            res=completion[i,input_lengths[i]:]

            input_w=diffw_list[i,input_lengths[i]:]
            input_step=diffstep_list[i,input_lengths[i]:]
            indices = [i for i, sublist in enumerate(zip(res, res[1:])) if list(sublist) == target]
            if len(indices) > 0:

                end=int(indices[0])
                res=res[:end]
                input_w=input_w[:end]
                input_step=input_step[:end]

            if target_value in res:
                end = res.cpu().numpy().tolist().index(target_value)
                res=res[:end]
                input_w=input_w[:end]
                input_step=input_step[:end]

            output_tokens=self.trans_token(res,input_w,input_step)
            res=self.tokenizer.decode(torch.cat([completion[i,:input_lengths[i]], output_tokens]))

            end = res.find("[<|endoftext|>")
            if end > 0:
                res=res[:end]
            end = res.find("<|endoftext|>")
            if end > 0:
                res=res[:end]

            res=re.sub(self.pattern, r'\1', res)
            output_prompt.append(res)

        reward=self.scorer.get_score_batched(prompts=output_prompt,plain_texts=input_prompt).unsqueeze(1) #(B,1)
        #
        if self.debug:
            print("actor_log_probs", actor_log_probs.shape)
            print("sft_log_probs", sft_log_probs.shape)
            print("values", values.shape)
            print("reward", reward.shape)

        kl_penalized_reward, estimated_kl = self.kl_penalized_reward(
            reward, actor_log_probs, sft_log_probs
        )

        w_kl_penalized_reward, w_estimated_kl = self.kl_penalized_reward(
            reward, w_log_probs, sft_w_log_probs
        )

        step_kl_penalized_reward, step_estimated_kl = self.kl_penalized_reward(
            reward, step_log_probs, sft_step_log_probs
        )

        advantage = kl_penalized_reward - values
        w_advantage = w_kl_penalized_reward - values
        step_advantage = step_kl_penalized_reward - values

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

    def fit(self):
        scaler = GradScaler(enabled=self.dtype != torch.float32)

        print(f"self.total_epochs: {self.total_epochs} self.train_dataloader:{len(self.train_dataloader)}")

        for epoch in range(self.total_epochs):
            # 遍历训练数据加载器
            for step, (prompt, input_masks, input_lengths) in enumerate(
                pbar := tqdm(self.train_dataloader)
            ):
                step=step+self.step
                if len(prompt.shape)==3:
                    prompt=prompt.squeeze(1)
                    input_masks=input_masks.squeeze(1)
                prompt, input_masks, input_lengths = (
                    prompt.to(self.device),
                    input_masks.to(self.device),
                    input_lengths.to(self.device),
                )
                if self.debug:
                    print("prompt", prompt.shape)

                max_input_length = torch.max(input_lengths)
                prompt = prompt[:, :max_input_length]

                if self.debug:
                    print("input_lengths", input_lengths)
                    print("prompt after", prompt.shape)

                total_steps = step + epoch * len(self.train_dataloader)

                # 混合精度训练
                with torch.autocast(
                    device_type="cuda",
                    dtype=self.dtype,
                    enabled=self.dtype != torch.float32,
                ):
                    experience = self.make_experience(
                        prompt, input_masks, input_lengths
                    )
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
                    actor_loss_w=self.actor_criterion(
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

                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_lossf = actor_loss.item()

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
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    critic_lossf = critic_loss.item()

                    scaler.update()

                self.writer.add_scalar(
                    "KL", experience.estimated_kl.mean(), total_steps
                )
                self.writer.add_scalar(
                    "mean_advantage", experience.advantage.mean(), total_steps
                )
                self.writer.add_scalar(
                    "mean_reward", experience.kl_penalized_reward.mean(), total_steps
                )
                self.writer.add_scalar("mean_value", new_values.mean(), total_steps)
                self.writer.add_scalar("Loss/actor/step", actor_lossf, total_steps)
                self.writer.add_scalar("Loss/token/step", actor_loss_token.item(), total_steps)
                self.writer.add_scalar("Loss/w/step", actor_loss_w.item(), total_steps)
                self.writer.add_scalar("Loss/step/step", actor_loss_step.item(), total_steps)
                self.writer.add_scalar("Loss/critic/step", critic_lossf, total_steps)

                pbar.set_description(
                    f"actor loss {round(actor_lossf, 3)}, critic loss {round(critic_lossf, 3)}"
                )

                if (
                    (total_steps==500 or (total_steps != 0 and total_steps % self.save_freq == 0))
                ):
                    self.save_states(total_steps)

        self.save_states(None, True)