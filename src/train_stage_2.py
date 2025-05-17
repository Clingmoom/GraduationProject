import click
from src.configs import get_configs
from src.models import GPTActor, GPTCritic
from src.data import PPO_Dataset
from src.trainers import PPOTrainer


def train(batch_size, exp_name, actor_weights, critic_weights, epoch, card, num_images_per_prompt):
    cfg = get_configs("gpt2-medium")
    device = f"cuda:{card}"

    cfg.batch_size = batch_size
    cfg.exp_name = exp_name
    cfg.actor_weights = actor_weights
    cfg.critic_weights = critic_weights
    cfg.total_epochs = epoch # 训练总轮数

    cfg.sft_model_weights = cfg.actor_weights
    cfg.reward_model_weights = cfg.critic_weights

    actor = GPTActor.from_checkpoint(cfg, cfg.actor_weights)
    actor.to(device)

    sft_model = GPTActor.from_checkpoint(cfg, cfg.sft_model_weights)
    sft_model.to(device)

    cfg2 = get_configs("gpt2-medium/lora")
    critic = GPTCritic.from_checkpoint(cfg2, cfg.critic_weights)
    critic.to(device)

    critic.freeze_weights("lora")

    dataset = PPO_Dataset(device = device)
    trainer = PPOTrainer(cfg, actor, critic, sft_model, dataset, num_images_per_prompt = num_images_per_prompt, device = device)
    trainer.fit()


@click.command()
@click.option('--batch-size', '-b', default = 2)
@click.option('--exp-name', '-n', default = "ppo")
@click.option('--actor', '-a', default = "./ckpt/train/sft_20250517-155945/final.pt")
@click.option('--critic', '-c', default = "./ckpt/train/sft_20250517-155945/final.pt")
@click.option('--epoch', '-e', default = 1)
@click.option('--card', '-card', default = "0")
@click.option('--num_images_per_prompt', '-num_images_per_prompt', default = 2)


def main( batch_size, exp_name, actor, critic, epoch, card, num_images_per_prompt):
    train(batch_size, exp_name, actor, critic, epoch, card, num_images_per_prompt)


if __name__ == "__main__":
    main()