import torch
import time
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from src.configs import TrainingConfig
from .trainer import Trainer
from src.loss import CrossEntropyLoss


class SFTTrainer(Trainer):
    def __init__(
            self, cfg: TrainingConfig, device, model: nn.Module, train_dataset
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_name = (
            f"sft_{cfg.exp_name}_{datetime.now().strftime('%m%d%H%M')}"  # %Y%m%d%H%M
        )
        self.device = device
        # assert self.device == "cuda"
        self.max_steps = cfg.max_steps
        self.eval_freq = 1
        self.save_freq = 1e3  #
        self.train_dataloader = iter(
            DataLoader(
                train_dataset,
                batch_size=cfg.batch_size,
                num_workers=6,
                pin_memory=True
            )
        )

        self.model = model

        self.criterion = CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.grad_clip = cfg.grad_clip
        self.dtype = torch.float16

        self.finetune_method = cfg.finetune_method

        hp = {
            "dtype": str(self.dtype),
            "train_dataset": type(train_dataset).__name__,
            "train_dataset_len": len(train_dataset),
            **cfg.dict(),
        }
        self.save_hyperparams(hp)

    def fit(self):
        if self.finetune_method:
            self.model.freeze_weights(self.finetune_method)

        opt_model = torch.compile(self.model)
        opt_model.to(self.device)
        writer = SummaryWriter(f"./runs/{self.run_name}/logs", max_queue=40)
        scaler = GradScaler(enabled=self.dtype != torch.float32)

        opt_model.train()
        step = 0

        t0 = time.time()
        while step < self.max_steps:
            x, y = next(self.train_dataloader)
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.autocast(device_type="cuda", dtype=self.dtype):

                y_hat = opt_model(x)  # (B, 1)
                loss = self.criterion(y_hat, y)  # (B, 1)

            if self.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(opt_model.parameters(), self.grad_clip)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            lossf = loss.item()

            iter_time = time.time() - t0
            t0 = time.time()
            print(f"step {step}, batch loss {round(lossf, 3)}, {round(1.0 / iter_time, 2)} iters/s")
            writer.add_scalar("Loss/train/step", lossf, step)

            if step != 0 and step % self.save_freq == 0:
                self.save_states(step)
            step += 1

        self.save_states(step, True)