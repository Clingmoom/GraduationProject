import torch
import argparse
import os

from src.models import GPTActor
from src.configs import get_configs
from transformers import GPT2Tokenizer


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt",
    type=str,
    nargs="?",
    default="ckpt/PAE/actor_step3000.pt",
)
parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="a photo of a happy cat",
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

opt_a = parser.parse_args()
torch.manual_seed(opt_a.seed)
sft = opt_a.ckpt
device = f"cuda:{opt_a.card}"
cfg = get_configs("gpt2-medium")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2", device=device)

def prepare_gpt2_input(prompt, device):
    enc = tokenizer
    # 定义编码和解码函数，允许使用 文本结束 的特殊字符
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    indices = encode(prompt)
    # 使用 None 在第0维添加一个新维度，使得张量的形状从 [n] 变为 [1, n]
    # 其中 ... 表示保持 indices 的原始形状不变
    x = (torch.tensor(indices, dtype=torch.long, device=device)[None, ...])
    return x, decode

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

pattern = r'\[([^]]*):0-1:1\.0\]'  # r'\[(\s*\w+):0-1:1\.0\]'

def trans_token(bef_list, diffw_list, diffstep_list):
    if len(bef_list) == 0:
        return bef_list

    aft_list = torch.tensor([], device=device)
    ind = 0

    def get_modes(special_token_ind_list):
        try:
            w_counts = torch.bincount(diffw_list[special_token_ind_list]) # 统计每个权重出现的次数
            w_mode = int(torch.argmax(w_counts).item()) # 找到出现次数最多的权重
        except:
            w_mode = 2
        try:
            counts = torch.bincount(diffstep_list[special_token_ind_list])
            mode = int(torch.argmax(counts).item())
        except:
            mode = 1
        return w_mode, mode

    def insert_tokens(special_token_ind_list, w_mode, mode):
        nonlocal aft_list
        for idx in special_token_ind_list:
            s_token = bef_list[idx]
            parts = [
                token_dict["["].unsqueeze(0),
                s_token.unsqueeze(0),
                token_dict[":"].unsqueeze(0),
                step_dict[mode],
                token_dict[":"].unsqueeze(0),
                w_dict[w_mode],
                token_dict["]"].unsqueeze(0),
            ]
            aft_list = torch.cat([aft_list] + parts)

    while ind < len(bef_list):
        token = bef_list[ind]
        # 当前 token 是普通 token，继续收集
        if token not in [token_dict[","], token_dict["."]]:
            special_token_ind_list = []
            # 是否是“词的开始”（避免子词或空格开头）
            while ind < len(bef_list):
                token = bef_list[ind]
                if token in [token_dict[","], token_dict["."], token_dict[" "]] or tokenizer.decode([token.long()]).startswith(" "):
                    break
                aft_list = torch.cat([aft_list, token.unsqueeze(0)])
                ind += 1
            if ind >= len(bef_list):
                break
        else:
            # 遇到标点，标点后开始收集下一段 special tokens
            punct_token = token
            ind += 1
            special_token_ind_list = []

            while ind < len(bef_list):
                token = bef_list[ind]
                if token in [token_dict[","], token_dict["."]]:
                    break
                if not tokenizer.decode([token.long()]).startswith(" "):
                    special_token_ind_list.append(ind)
                ind += 1

            aft_list = torch.cat([aft_list, punct_token.unsqueeze(0)])

            if special_token_ind_list:
                w_mode, mode = get_modes(special_token_ind_list)
                insert_tokens(special_token_ind_list, w_mode, mode)

    return aft_list


def generate_gpt2(model, prompt, device):
    temperature = 0.9 # 控制文本生成的随机性
    top_k = 200 # 模型生成新词的候选词数量top_k

    x, decode = prepare_gpt2_input(prompt, device)
    max_new_tokens = 75 - x.shape[-1]
    y, diffw_list, diffstep_list = model.generate_dy(x, max_new_tokens, temperature=temperature, top_k=top_k)
    if y.shape == torch.Size([0]):
        return prompt
    y_0 = y[0].long()
    input_w = diffw_list[0].long()
    input_step = diffstep_list[0].long()

    target_value = torch.tensor(50256, device=device)
    # nonzero 返回y_0中非零元素的索引
    end = (y_0 == target_value).nonzero(as_tuple=True)[0]
    if end.numel() > 0:
        y_0 = y_0[:end[0]]
        input_w = input_w[:end[0]]
        input_step = input_step[:end[0]]

    res = decode(torch.cat([x[0], trans_token(y_0, input_w, input_step)]))
    end = res.find("[<|endoftext|>")
    if end > 0:
        res = res[:end]
    end = res.find("<|endoftext|>")
    if end > 0:
        res = res[:end]

    return res


with torch.inference_mode():
    gpt_sft = torch.compile(GPTActor.from_checkpoint(cfg, sft)).to(device)
    gpt_sft.eval()
    result = generate_gpt2(gpt_sft, opt_a.prompt, device)
    print(result)
