import time
import torch
import torch.optim as optim
import numpy as np

from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from src.configs import TrainingConfig
from .trainer import Trainer
from src.loss import CrossEntropyLoss


class SFTTrainer_head(Trainer):
    def __init__(
        self, cfg: TrainingConfig, device, model: nn.Module, train_dataset
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_name = f"{cfg.exp_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(f"self.run_name:{self.run_name}")
        self.device = device
        self.max_steps = cfg.max_steps
        self.save_freq = 2e4  # 模型保存频率

        self.train_dataloader = iter(DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            num_workers=6,  # 6个子进程加载数据
            pin_memory=True  # 将数据加载到固定内存中，加快GPU传输
        ))

        self.model = model
        self.criterion = CrossEntropyLoss()  # 交叉熵损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)  # Adam优化器,lr学习率

        self.grad_clip = cfg.grad_clip  # 梯度裁剪阈值
        self.dtype = torch.float16  # 混合精度训练
        self.finetune_method = cfg.finetune_method  # 微调方法

        hp = {
            "dtype": str(self.dtype),
            "train_dataset": type(train_dataset).__name__,
            "train_dataset_len": len(train_dataset),
            **cfg.dict(),
        }
        self.save_hyperparams(hp)

    def generate_tensor(self, mean, values, shape):
        # 使用正态分布生成随机值，然后将其裁剪到指定范围，并取整
        random_values = np.random.normal(mean, scale=0.5, size=shape)
        random_values = np.clip(random_values, min(values), max(values))
        random_values = np.round(random_values)
        tensor = torch.tensor(random_values, device=self.device)
        return tensor

    def fit(self):
        if self.finetune_method:
            self.model.freeze_weights(self.finetune_method)

        opt_model = self.model
        opt_model.to(self.device)
        # 记录训练日志  max_queue：攒够这么多条才写入文件
        # 查看本地日志：tensorboard --logdir=./runs
        # TODO：自动 flush() ；根据 batch size 自动调整 max_queue
        writer = SummaryWriter(f"./logs/{self.run_name}", max_queue=40)
        scaler = GradScaler(enabled = self.dtype != torch.float32)  # 混合精度训练

        opt_model.train()  # 训练模式
        step = 0
        t0 = time.time()
        # 训练循环
        while step < self.max_steps:
            x, y = next(self.train_dataloader)  # 获取下一个训练批次
            x = x.to(self.device)
            y = y.to(self.device)  # y是x的下一个token

            # 自动混合精度
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                y_hat, diffw, diffstep = opt_model(x) # (B,T,V)
                loss = self.criterion(y_hat, y)
                loss_w = self.criterion(diffw, self.generate_tensor(2, [0, 1, 2, 3, 4], y.shape).long())
                loss_step = self.criterion(diffstep, self.generate_tensor(1, [0, 1, 2], y.shape).long())
                loss = loss + loss_w + loss_step

            scaler.scale(loss).backward()  # 反向传播，得到scaled梯度 scale缩放解决梯度下溢问题

            scaler.unscale_(self.optimizer)  # 反缩放，把梯度还原成真实大小（非常重要！）

            if self.grad_clip != 0.0:
                # 裁剪真实梯度 防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(opt_model.parameters(), self.grad_clip)

            scaler.step(self.optimizer)  # 用（裁剪过的）真实梯度更新参数  调用 optimizer.step()

            scaler.update()  # 更新grad scaler内部状态

            self.optimizer.zero_grad(set_to_none=True)  # 清空梯度

            # 训练监控
            lossf = loss.item()
            iter_time = time.time() - t0
            t0 = time.time()
            print(
                f"step {step}, batch loss {round(lossf, 3)}, {round(1.0 / iter_time, 2)} iters/s"
            )
            writer.add_scalar("Loss/train/step", lossf, step)
            writer.add_scalar("loss_w/train/step", loss_w.item(), step)
            writer.add_scalar("loss_step/train/step", loss_step.item(), step)

            if step != 0 and step % self.save_freq == 0 or step == 50000:
                self.save_states(step)

            step += 1
        # 最终保存模型
        self.save_states(step, True)