import click
import time

from src.trainers import SFTTrainer_head
from src.models import GPTActor
from src.data import SFT_Datasets
from src.configs import get_configs

# 内存碎片优化
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

tic=time.time()
def train(batch_size, exp_name, step, card):
    device = f'cuda:{card}'
    cfg = get_configs("gpt2-medium")
    cfg.max_steps = step
    cfg.exp_name = exp_name
    cfg.batch_size = batch_size

    model = GPTActor.from_pretrained(cfg)
    train_ds = SFT_Datasets( block_size=256, device=device ) # block_size 模型一次处理的最大token数
    trainer = SFTTrainer_head(cfg, device, model, train_ds)
    trainer.fit()


@click.command()
@click.option('--batch-size', '-b', default=1)
@click.option('--exp-name', '-n', default="sft")
@click.option('--step', '-t', default=5e5)
@click.option('--card', '-card', default=0)


def main( batch_size, exp_name,step,card):
    train(batch_size, exp_name,step,card)


if __name__ == "__main__":
    main()
    toc=time.time()
    print(f"time:{toc-tic}")