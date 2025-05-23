import torch
import time
import numpy as np
import argparse, os

from torchvision import transforms
from transformers import GPT2TokenizerFast as GPT2Tokenizer
from src.configs import get_configs, ROOT_DIR
from src.models import GPTActor
from src.trainers import PromptScorer
import torchvision.utils as vutils

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
WANDB_KEY = "31f49565acf4d198ed0a419fb67527f0668b9d03"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt",
    type=str,
    nargs="?",
    default=ROOT_DIR / "ckpt" / "train" / "ppo_20250521-100258" /"actor_step3000.pt",
)
parser.add_argument(
    "--save",
    type=str,
    nargs="?",
    default="./result",
)
parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="A photo of a cat",
)
parser.add_argument(
    "--seed",
    type=int,
    nargs="?",
    default=42,
)
parser.add_argument(
    "--card",
    type=int,
    nargs="?",
    default=0,
)
parser.add_argument(
    "--data",
    type=str,
    nargs="?",
    default="coco",
)

def main():
    import wandb
    wandb.login(key=WANDB_KEY)

    opt_a = parser.parse_args()
    torch.manual_seed(opt_a.seed)
    use_model = opt_a.ckpt
    device = f"cuda:{opt_a.card}"
    cfg = get_configs("gpt2-medium")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def prepare_gpt2_input(prompt, device):
        print("正在准备gpt2的输入...")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        decode = lambda ids: tokenizer.decode(ids)
        return input_ids, decode

    step_dict = {
        0: torch.tensor(tokenizer.encode("0-0.5"), device=device),  # 0-0.5
        1: torch.tensor(tokenizer.encode("0-1"), device=device),  # 0-1
        2: torch.tensor(tokenizer.encode("0.5-1"), device=device),  # 0.5-1
    }
    w_dict = {
        0: torch.tensor(tokenizer.encode("0.5"), device=device),
        1: torch.tensor(tokenizer.encode("0.75"), device=device),
        2: torch.tensor(tokenizer.encode("1.0"), device=device),
        3: torch.tensor(tokenizer.encode("1.25"), device=device),
        4: torch.tensor(tokenizer.encode("1.5"), device=device),
    }
    token_dict = {
        ",": torch.tensor(tokenizer.encode(",")[0], device=device),
        ".": torch.tensor(tokenizer.encode(".")[0], device=device),
        ":": torch.tensor(tokenizer.encode(":")[0], device=device),
        " [": torch.tensor(tokenizer.encode(" [")[0], device=device),
        "[": torch.tensor(tokenizer.encode("[")[0], device=device),
        "]": torch.tensor(tokenizer.encode("]")[0], device=device),
        " ": torch.tensor(tokenizer.encode(" ")[0], device=device)
    }

    # def trans_token(bef_list, diffw_list, diffstep_list):
    #     if len(bef_list) == 0:
    #         return bef_list
    #
    #     aft_list = torch.tensor([], device=device)
    #     ind = 0
    #
    #     def get_modes(special_token_ind_list):
    #         try:
    #             w_counts = torch.bincount(diffw_list[special_token_ind_list])  # 统计每个权重出现的次数
    #             w_mode = int(torch.argmax(w_counts).item())  # 找到出现次数最多的权重
    #         except:
    #             w_mode = 2
    #         try:
    #             counts = torch.bincount(diffstep_list[special_token_ind_list])
    #             mode = int(torch.argmax(counts).item())
    #         except:
    #             mode = 1
    #         return w_mode, mode
    #
    #     def insert_tokens(special_token_ind_list, w_mode, mode):
    #         nonlocal aft_list
    #         for idx in special_token_ind_list:
    #             s_token = bef_list[idx]
    #             parts = [
    #                 token_dict["["].unsqueeze(0),
    #                 s_token.unsqueeze(0),
    #                 token_dict[":"].unsqueeze(0),
    #                 step_dict[mode],
    #                 token_dict[":"].unsqueeze(0),
    #                 w_dict[w_mode],
    #                 token_dict["]"].unsqueeze(0),
    #             ]
    #             aft_list = torch.cat([aft_list] + parts)
    #
    #     while ind < len(bef_list):
    #         token = bef_list[ind]
    #         # 当前 token 是普通 token，继续收集
    #         if token not in [token_dict[","], token_dict["."]]:
    #             special_token_ind_list = []
    #             # 是否是“词的开始”（避免子词或空格开头）
    #             while ind < len(bef_list):
    #                 token = bef_list[ind]
    #                 if token in [token_dict[","], token_dict["."], token_dict[" "]] or tokenizer.decode(
    #                     [token.long()]).startswith(" "):
    #                     break
    #                 aft_list = torch.cat([aft_list, token.unsqueeze(0)])
    #                 ind += 1
    #             if ind >= len(bef_list):
    #                 break
    #         else:
    #             # 遇到标点，标点后开始收集下一段 special tokens
    #             punct_token = token
    #             ind += 1
    #             special_token_ind_list = []
    #
    #             while ind < len(bef_list):
    #                 token = bef_list[ind]
    #                 if token in [token_dict[","], token_dict["."]]:
    #                     break
    #                 if not tokenizer.decode([token.long()]).startswith(" "):
    #                     special_token_ind_list.append(ind)
    #                 ind += 1
    #
    #             aft_list = torch.cat([aft_list, punct_token.unsqueeze(0)])
    #
    #             if special_token_ind_list:
    #                 w_mode, mode = get_modes(special_token_ind_list)
    #                 insert_tokens(special_token_ind_list, w_mode, mode)
    #
    #     return aft_list
    def trans_token(bef_list, diffw_list, diffstep_list):
        if len(bef_list) == 0:
            return bef_list
        aft_list = torch.tensor([], dtype=torch.long, device=device)

        ind = 0
        token = bef_list[ind]
        if not (token == token_dict[","] or token == token_dict["."]):
            special_token_ind_list = []

            while not (
                token == token_dict[","] or token == token_dict["."] or token == token_dict[" "] or tokenizer.decode(
                [token.long()]).startswith(" ")):
                token = bef_list[ind]
                aft_list = torch.cat([aft_list, token.unsqueeze(0)])
                ind += 1

                if ind >= (len(bef_list)):
                    break
            if ind < (len(bef_list)):
                token = bef_list[ind]
            while ind < (len(bef_list)) and not (token == token_dict[","] or token == token_dict["."]):
                if token == token_dict[" "] or token == token_dict[","] or token == token_dict["."]:
                    aft_list = torch.cat([aft_list, token.unsqueeze(0)])
                    ind += 1
                    if ind >= (len(bef_list)):
                        break
                    token = bef_list[ind]
                else:
                    special_token_ind_list.append(ind)

                    ind += 1
                    if ind >= (len(bef_list)):
                        break
                    token = bef_list[ind]

                    if token == token_dict[","] or token == token_dict["."]:
                        break

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

            for ind in special_token_ind_list:
                aft_list = torch.cat([aft_list, token_dict["["].unsqueeze(0)])
                s_token = bef_list[ind]

                aft_list = torch.cat([aft_list, s_token.unsqueeze(0)])
                aft_list = torch.cat([aft_list, token_dict[":"].unsqueeze(0)])
                aft_list = torch.cat([aft_list, step_dict[mode]])
                aft_list = torch.cat([aft_list, token_dict[":"].unsqueeze(0)])
                aft_list = torch.cat([aft_list, w_dict[w_mode]])
                aft_list = torch.cat([aft_list, token_dict["]"].unsqueeze(0)])
            ind += 1

            while ind < len(bef_list):

                token = bef_list[ind]

                if not (token == token_dict[","] or token == token_dict["."]):
                    aft_list = torch.cat([aft_list, token.unsqueeze(0)])
                    ind += 1
                else:
                    ind += 1
                    if ind >= len(bef_list):
                        break
                    token = bef_list[ind]
                    special_token_ind_list = []
                    while not (token == token_dict[","] or token == token_dict["."]):

                        special_token_ind_list.append(ind)

                        ind += 1
                        if ind >= (len(bef_list)):
                            break
                        token = bef_list[ind]

                        if token == token_dict[","] or token == token_dict["."]:
                            break

                    aft_list = torch.cat([aft_list, token_dict[","].unsqueeze(0)])
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

                    for ind in special_token_ind_list:
                        aft_list = torch.cat([aft_list, token_dict["["].unsqueeze(0)])
                        s_token = bef_list[ind]
                        aft_list = torch.cat([aft_list, s_token.unsqueeze(0)])
                        aft_list = torch.cat([aft_list, token_dict[":"].unsqueeze(0)])
                        aft_list = torch.cat([aft_list, step_dict[mode]])
                        aft_list = torch.cat([aft_list, token_dict[":"].unsqueeze(0)])
                        aft_list = torch.cat([aft_list, w_dict[w_mode]])
                        aft_list = torch.cat([aft_list, token_dict["]"].unsqueeze(0)])
                    ind += 1


        else:
            while ind < len(bef_list):

                token = bef_list[ind]

                if not (token == token_dict[","] or token == token_dict["."]):
                    aft_list = torch.cat([aft_list, token.unsqueeze(0)])
                    ind += 1
                else:
                    ind += 1
                    if ind >= len(bef_list):
                        break
                    token = bef_list[ind]
                    special_token_ind_list = []
                    while not (token == token_dict[","] or token == token_dict["."]):

                        special_token_ind_list.append(ind)

                        ind += 1
                        if ind >= (len(bef_list)):
                            break
                        token = bef_list[ind]

                        if token == token_dict[","] or token == token_dict["."]:
                            break

                    aft_list = torch.cat([aft_list, token_dict[","].unsqueeze(0)])
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

                    for ind in special_token_ind_list:
                        aft_list = torch.cat([aft_list, token_dict["["].unsqueeze(0)])
                        s_token = bef_list[ind]
                        aft_list = torch.cat([aft_list, s_token.unsqueeze(0)])
                        aft_list = torch.cat([aft_list, token_dict[":"].unsqueeze(0)])
                        aft_list = torch.cat([aft_list, step_dict[mode]])
                        aft_list = torch.cat([aft_list, token_dict[":"].unsqueeze(0)])
                        aft_list = torch.cat([aft_list, w_dict[w_mode]])
                        aft_list = torch.cat([aft_list, token_dict["]"].unsqueeze(0)])
                    ind += 1

        return aft_list.long()

    def generate_gpt2(model, prompt, device):
        temperature = 0.9
        top_k = 200

        x, decode = prepare_gpt2_input(prompt, device)
        max_new_tokens = 75 - x.shape[-1]
        y, diffw_list, diffstep_list = model.generate_dy(x, max_new_tokens, temperature=temperature, top_k=top_k)
        print("序列生成完成……")
        if y.shape == torch.Size([0]):
            return prompt
        y_0 = y[0].long()  # 生成的token序列
        input_w = diffw_list[0].long()
        input_step = diffstep_list[0].long()

        target_value = torch.tensor(50256, device=device)

        end = (y_0 == target_value).nonzero(as_tuple=True)[0]
        if end.numel() > 0:
            y_0 = y_0[:end[0]]
            input_w = input_w[:end[0]]
            input_step = input_step[:end[0]]
        print("正在准备编码……")
        res = decode(torch.cat([x[0], trans_token(y_0, input_w, input_step)]))
        print("编码后的结果为：", res)
        end = res.find("[<|endoftext|>")
        if end > 0:
            res = res[:end]
        end = res.find("<|endoftext|>")
        if end > 0:
            res = res[:end]

        return res

    tic = time.time()
    scorer = PromptScorer(device=device, num_images_per_prompt=1, seed=opt_a.seed)

    if opt_a.data == "coco":
        filename = ROOT_DIR / "data" / "evaluate_data" / "COCO_test_1k.npy"
        data = np.load(filename)
    elif opt_a.data == "diffusiondb_test":
        data = np.load(ROOT_DIR / "data" / "evaluate_data" / "diffusiondb_test_1k.npy")
        data = data.reshape(1000, 1)
    elif opt_a.data == "lexica":
        data = np.load(ROOT_DIR / "data" / "evaluate_data" / "lexica_test_1k.npy")
        data = data.reshape(1000, 1)

    wandb.init(
        project="基于提示优化的文本到图像生成方法研究",
        name="evaluate_" + opt_a.data,
        config={
            "ckpt": str(opt_a.ckpt),
            "data": opt_a.data,
            "seed": opt_a.seed,
        }
    )

    prompt_all = []
    pick_scores_sum = torch.tensor(0.0)

    with torch.inference_mode():
        model = GPTActor.from_checkpoint(cfg, use_model).to(device)
        model.eval()
        last_tic = time.time()
        for i in range(0, len(data), 25):
            if i + 25 < len(data):
                p = i + 25
            else:
                p = len(data)

            plain_texts = [s[0] for s in data[i:p]]  # ["A photo of a cat", "A photo of a dog"...]
            prompt = [generate_gpt2(model, s, device) for s in plain_texts]  # s:"A photo of a cat"

            prompt_all += prompt
            if p > 998:
                print(prompt, i)
            try:
                print("正在生成图像……")
                images = scorer.gen_image_batched(plain_texts)
                dy_images = scorer.gen_image_batched(prompt)
                short_prompt_index = 0
                eval_model_index = 1

                pick_scores = torch.Tensor(
                    [scorer.eval_pick_score(p, [img, dy_img])[eval_model_index] for p, img, dy_img in zip(plain_texts,images, dy_images)]
                )

                print(f"pick_scores:{pick_scores}")
                pick_scores_sum += torch.Tensor(pick_scores).sum()

                print("✏️记录日志~")
                image_inx = 0
                to_tensor = transforms.ToTensor()
                example_image = vutils.make_grid(
                    [to_tensor(images[image_inx]), to_tensor(dy_images[image_inx])],
                    nrow=2,
                    padding=20,         # 每张图之间 8 像素间隔
                    pad_value=255      # 间隔填白色，看起来更清晰
                )
                wandb.log({
                    "pick_mean/batch": pick_scores.float().mean().item(),
                    # 示例图像（取本批第 1 张）
                    "example_image": wandb.Image(example_image, caption=f"left is short_prompt|right is dy_prompt\ndy_prompt:{prompt[image_inx]}"),
                    # 批次耗时
                    "time/batch": time.time() - last_tic
                })
                last_tic = time.time()

            except Exception as e:
                raise e
    pick_score = round(pick_scores_sum.item() / len(data), 3)
    print(pick_score)

    data_dict = {
        "PickScore": pick_score,
    }

    toc = time.time()
    total_time = toc - tic
    print(f"time:{total_time}")

    wandb.log({
        "eval/pickscore": data_dict["pickscore"],
        "eval/total_time": total_time
    })
    wandb.finish()

if __name__ == "__main__":
    main()