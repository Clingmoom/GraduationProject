import os
import json
import torch
import random

from src.configs import ROOT_DIR


class Trainer:
    def __init__(self) -> None:
        self.model = None
        self.optimizer = None
        random.seed(1) # 设置随机数种子为 1，确保实验的可重复性

    def save_hyperparams(self, hp):
        # 检查保存超参数的目录是否存在，如果不存在则创建
        if not os.path.exists(f"./runs/{self.run_name}"):
            os.makedirs(f"./runs/{self.run_name}")
        # 将超参数以 JSON 格式保存到指定文件中
        with open(f"./runs/{self.run_name}/hyperparams.json", "w") as fp:
            json.dump(hp, fp, indent=4)

    def save_metrics(self, metrics):
        if not os.path.exists(f"./runs/{self.run_name}"):
            os.makedirs(f"./runs/{self.run_name}")
        with open(f"./runs/{self.run_name}/metrics.json", "w") as fp:
            json.dump(metrics, fp, indent=4)

    def save_states(self, step = None, is_last=False):
        save_dir = ROOT_DIR / "ckpt" / "train" / f"{self.run_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 根据是否为最后一步确定保存的文件名
        file_name = (
            "final.pt" if is_last else f"step{step}.pt"
        )
        save_path = save_dir / file_name
        # 保存模型的当前步数、模型状态字典和优化器状态字典
        torch.save(
            {
                'step': step,
                "model_state_dict": self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            save_path,
        )